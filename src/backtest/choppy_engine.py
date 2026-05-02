"""
Choppy Market Engine

Mean-reversion trading engine for choppy/range-bound markets.
Uses RSI extremes + liquidity sweeps for entry signals.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from enum import Enum

from .regime_detector import RegimeDetector, MarketRegime
from .liquidity_sweep import (
    LiquiditySweepDetector,
    RangeStructureDetector,
    FairValueGapDetector,
    SweepResult
)


class ChoppySignal(Enum):
    """Choppy engine signal types."""
    BUY = 'BUY'
    SELL = 'SELL'
    HOLD = 'HOLD'
    NO_TRADE = 'NO_TRADE'


@dataclass
class ChoppyResult:
    """Result from choppy engine evaluation."""
    signal: ChoppySignal
    confidence: float
    sl_atr: float
    tp_atr: float
    size_multiplier: float
    reason: str
    details: Dict


class ChoppyMarketEngine:
    """
    Mean-reversion engine for choppy/range markets.
    
    Entry logic:
    1. Market must be in CHOPPY regime
    2. Price in extreme zone (upper/lower third of range)
    3. RSI confirms extreme (< 30 or > 70)
    4. Optional: Liquidity sweep confirmation
    
    Risk logic:
    - Wider SL (2x ATR) - allow oscillation
    - Smaller TP (1x ATR) - target mean
    - Reduced position size (0.5x)
    """
    
    # RSI thresholds
    RSI_OVERSOLD = 30
    RSI_OVERBOUGHT = 70
    RSI_EXTREME_OVERSOLD = 25
    RSI_EXTREME_OVERBOUGHT = 75
    RSI_NEUTRAL_LOW = 40
    RSI_NEUTRAL_HIGH = 60
    
    # RSI Z-score thresholds
    RSI_ZSCORE_THRESHOLD = 1.5
    
    # SL/TP parameters (ATR multiples)
    SL_ATR = 2.0
    TP_ATR = 1.0
    
    # Position sizing
    BASE_SIZE_MULT = 0.5  # Half of trend size
    
    def __init__(self):
        self.regime_detector = RegimeDetector()
        self.sweep_detector = LiquiditySweepDetector(lookback=20)
        self.range_detector = RangeStructureDetector(lookback=50)
        self.fvg_detector = FairValueGapDetector()
    
    def evaluate(
        self,
        features: Dict[str, float],
        df: Optional[pd.DataFrame] = None
    ) -> ChoppyResult:
        """
        Evaluate choppy market conditions and generate signal.
        
        Args:
            features: Normalized features dict
            df: Optional DataFrame for sweep/range detection
            
        Returns:
            ChoppyResult with signal and parameters
        """
        # Check regime first
        regime_result = self.regime_detector.detect(features)
        
        if regime_result.regime != MarketRegime.CHOPPY:
            return ChoppyResult(
                signal=ChoppySignal.NO_TRADE,
                confidence=0.0,
                sl_atr=0,
                tp_atr=0,
                size_multiplier=0,
                reason=f"Regime is {regime_result.regime.value}, not CHOPPY",
                details={'regime': regime_result.regime.value}
            )
        
        # Check no-trade zones
        no_trade, no_trade_reason = self._check_no_trade_zone(features)
        if no_trade:
            return ChoppyResult(
                signal=ChoppySignal.NO_TRADE,
                confidence=0.0,
                sl_atr=0,
                tp_atr=0,
                size_multiplier=0,
                reason=no_trade_reason,
                details={'no_trade_zone': True}
            )
        
        # Get RSI signal
        rsi_signal = self._get_rsi_signal(features)
        
        if rsi_signal == 'NEUTRAL':
            return ChoppyResult(
                signal=ChoppySignal.HOLD,
                confidence=0.0,
                sl_atr=0,
                tp_atr=0,
                size_multiplier=0,
                reason="RSI in neutral zone",
                details={'rsi': features.get('rsi_14', 50)}
            )
        
        # Get range structure
        range_struct = self._get_range_structure(features, df)
        
        # Get sweep confirmation (if df available)
        sweep = self._get_sweep(df) if df is not None else None
        
        # Evaluate BUY
        if rsi_signal == 'BUY':
            return self._evaluate_buy(features, range_struct, sweep)
        
        # Evaluate SELL
        if rsi_signal == 'SELL':
            return self._evaluate_sell(features, range_struct, sweep)
        
        return ChoppyResult(
            signal=ChoppySignal.HOLD,
            confidence=0.0,
            sl_atr=0,
            tp_atr=0,
            size_multiplier=0,
            reason="No valid signal",
            details={}
        )
    
    def _check_no_trade_zone(self, features: Dict) -> Tuple[bool, str]:
        """Check if in a no-trade zone."""
        
        # 1. Dead market
        atr_ratio = features.get('atr_ratio_14_50', 1.0)
        if atr_ratio < 0.5:
            return True, "ATR_TOO_LOW"
        
        # 2. RSI neutral
        rsi = features.get('rsi_14', 50)
        if self.RSI_NEUTRAL_LOW <= rsi <= self.RSI_NEUTRAL_HIGH:
            return True, "RSI_NEUTRAL"
        
        # 3. BB squeeze (breakout imminent)
        bb_width_zscore = features.get('bb_width_zscore', 0)
        if bb_width_zscore < -1.5:
            return True, "BB_SQUEEZE"
        
        return False, ""
    
    def _get_rsi_signal(self, features: Dict) -> str:
        """Get RSI-based signal direction."""
        rsi = features.get('rsi_14', 50)
        rsi_zscore = features.get('rsi_zscore', 0)
        
        # BUY: RSI oversold + Z-score confirms
        if rsi <= self.RSI_OVERSOLD and rsi_zscore <= -self.RSI_ZSCORE_THRESHOLD:
            return 'BUY'
        
        # SELL: RSI overbought + Z-score confirms
        if rsi >= self.RSI_OVERBOUGHT and rsi_zscore >= self.RSI_ZSCORE_THRESHOLD:
            return 'SELL'
        
        return 'NEUTRAL'
    
    def _get_range_structure(
        self, 
        features: Dict,
        df: Optional[pd.DataFrame]
    ) -> Dict:
        """Get range structure from features or DataFrame."""
        
        if df is not None:
            return self.range_detector.detect_from_df(df)
        
        # Fallback: use price_position from features
        position = features.get('price_position_50', 0.5)
        
        if position < 0.33:
            zone = 'LOWER_THIRD'
        elif position > 0.67:
            zone = 'UPPER_THIRD'
        else:
            zone = 'MIDDLE_THIRD'
        
        return {
            'zone': zone,
            'position_in_range': position,
            'range_high': 0,
            'range_low': 0,
            'range_mid': 0,
        }
    
    def _get_sweep(self, df: pd.DataFrame) -> Optional[SweepResult]:
        """Get liquidity sweep result."""
        try:
            return self.sweep_detector.detect_from_df(df)
        except Exception:
            return None
    
    def _evaluate_buy(
        self,
        features: Dict,
        range_struct: Dict,
        sweep: Optional[SweepResult]
    ) -> ChoppyResult:
        """Evaluate BUY signal."""
        
        zone = range_struct.get('zone', 'MIDDLE_THIRD')
        rsi = features.get('rsi_14', 50)
        
        # Calculate confidence
        confidence = 0.5  # Base
        
        # Bonus: In lower zone
        if zone == 'LOWER_THIRD':
            confidence += 0.2
        elif zone == 'MIDDLE_THIRD':
            confidence -= 0.2  # Penalty for middle
        
        # Bonus: Extreme RSI
        if rsi < self.RSI_EXTREME_OVERSOLD:
            confidence += 0.15
        
        # Bonus: Liquidity sweep
        if sweep and sweep.swept_low:
            confidence += 0.15
        
        # Minimum confidence for entry
        if confidence < 0.5:
            return ChoppyResult(
                signal=ChoppySignal.HOLD,
                confidence=confidence,
                sl_atr=0,
                tp_atr=0,
                size_multiplier=0,
                reason=f"BUY confidence {confidence:.2f} < 0.5",
                details={'zone': zone, 'rsi': rsi}
            )
        
        # Calculate size multiplier based on confidence
        size_mult = self._calc_size_mult(confidence)
        
        return ChoppyResult(
            signal=ChoppySignal.BUY,
            confidence=min(confidence, 1.0),
            sl_atr=self.SL_ATR,
            tp_atr=self.TP_ATR,
            size_multiplier=size_mult,
            reason=f"Choppy BUY: RSI={rsi:.0f}, zone={zone}",
            details={
                'zone': zone,
                'rsi': rsi,
                'sweep_low': sweep.swept_low if sweep else False,
                'position': range_struct.get('position_in_range', 0.5)
            }
        )
    
    def _evaluate_sell(
        self,
        features: Dict,
        range_struct: Dict,
        sweep: Optional[SweepResult]
    ) -> ChoppyResult:
        """Evaluate SELL signal."""
        
        zone = range_struct.get('zone', 'MIDDLE_THIRD')
        rsi = features.get('rsi_14', 50)
        
        # Calculate confidence
        confidence = 0.5
        
        if zone == 'UPPER_THIRD':
            confidence += 0.2
        elif zone == 'MIDDLE_THIRD':
            confidence -= 0.2
        
        if rsi > self.RSI_EXTREME_OVERBOUGHT:
            confidence += 0.15
        
        if sweep and sweep.swept_high:
            confidence += 0.15
        
        if confidence < 0.5:
            return ChoppyResult(
                signal=ChoppySignal.HOLD,
                confidence=confidence,
                sl_atr=0,
                tp_atr=0,
                size_multiplier=0,
                reason=f"SELL confidence {confidence:.2f} < 0.5",
                details={'zone': zone, 'rsi': rsi}
            )
        
        size_mult = self._calc_size_mult(confidence)
        
        return ChoppyResult(
            signal=ChoppySignal.SELL,
            confidence=min(confidence, 1.0),
            sl_atr=self.SL_ATR,
            tp_atr=self.TP_ATR,
            size_multiplier=size_mult,
            reason=f"Choppy SELL: RSI={rsi:.0f}, zone={zone}",
            details={
                'zone': zone,
                'rsi': rsi,
                'sweep_high': sweep.swept_high if sweep else False,
                'position': range_struct.get('position_in_range', 0.5)
            }
        )
    
    def _calc_size_mult(self, confidence: float) -> float:
        """Calculate position size multiplier based on confidence."""
        if confidence < 0.5:
            return 0.0
        elif confidence < 0.7:
            return self.BASE_SIZE_MULT * 0.5  # 0.25x
        elif confidence < 0.9:
            return self.BASE_SIZE_MULT  # 0.5x
        else:
            return self.BASE_SIZE_MULT * 1.5  # 0.75x
    
    def should_early_exit(
        self,
        features: Dict,
        side: str
    ) -> Tuple[bool, str]:
        """
        Check if should exit early (before SL/TP).
        
        Returns:
            (should_exit, reason)
        """
        rsi = features.get('rsi_14', 50)
        position = features.get('price_position_50', 0.5)
        trend_zscore = abs(features.get('trend_50_zscore', 0))
        
        # 1. RSI returned to neutral
        rsi_neutral = self.RSI_NEUTRAL_LOW <= rsi <= self.RSI_NEUTRAL_HIGH
        
        # 2. Price near range midpoint
        near_mid = 0.4 <= position <= 0.6
        
        # 3. Regime changed to trending
        regime_changed = trend_zscore > 1.0
        
        if rsi_neutral and near_mid:
            return True, "Mean reversion complete (RSI neutral + mid range)"
        
        if regime_changed:
            return True, f"Regime changed to TREND (trend_z={trend_zscore:.2f})"
        
        return False, ""


# ============================================================================
# Convenience function
# ============================================================================

def evaluate_choppy(
    features: Dict[str, float],
    df: Optional[pd.DataFrame] = None
) -> ChoppyResult:
    """Quick choppy evaluation."""
    engine = ChoppyMarketEngine()
    return engine.evaluate(features, df)
