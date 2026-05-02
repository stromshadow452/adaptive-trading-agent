"""
Hybrid Decision Brain

Routes signals to appropriate engine based on market regime:
- TREND regime → ML Trend Brain
- CHOPPY regime → Mean Reversion Engine
- TRANSITION/DEAD → No trade
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from enum import Enum

from .regime_detector import RegimeDetector, MarketRegime
from .choppy_engine import ChoppyMarketEngine, ChoppyResult, ChoppySignal
from .universal_ml_brain import UniversalMLBrain
from .asset_tiers import get_asset_config, is_tradeable

import pandas as pd


class SignalSource(Enum):
    """Source of the trading signal."""
    TREND_BRAIN = 'TREND_BRAIN'
    CHOPPY_ENGINE = 'CHOPPY_ENGINE'
    NO_TRADE = 'NO_TRADE'


@dataclass
class HybridResult:
    """Result from hybrid brain."""
    signal: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float
    source: SignalSource
    regime: MarketRegime
    sl_atr: float
    tp_atr: float
    size_multiplier: float
    reason: str
    details: Dict


class HybridDecisionBrain:
    """
    Routes to appropriate brain based on market regime.
    
    Regime routing:
    - TREND → UniversalMLBrain (trend-following)
    - CHOPPY → ChoppyMarketEngine (mean-reversion)
    - TRANSITION → No trade (unclear regime)
    - DEAD → No trade (no volatility)
    """
    
    # Default SL/TP for trend brain (ATR multiples)
    TREND_SL_ATR = 1.5
    TREND_TP_ATR = 2.5
    
    def __init__(self):
        self.regime_detector = RegimeDetector()
        self.trend_brain = UniversalMLBrain()
        self.choppy_engine = ChoppyMarketEngine()
    
    def decide(
        self,
        features: Dict[str, float],
        symbol: str,
        df: Optional[pd.DataFrame] = None
    ) -> HybridResult:
        """
        Make trading decision based on regime.
        
        Args:
            features: Normalized features dict
            symbol: Trading symbol (for asset tier config)
            df: Optional DataFrame for sweep detection
            
        Returns:
            HybridResult with signal and parameters
        """
        # Get asset configuration
        asset_config = get_asset_config(symbol)
        
        # Detect regime
        regime_result = self.regime_detector.detect(features)
        regime = regime_result.regime
        
        # Route based on regime
        if regime == MarketRegime.DEAD:
            return self._no_trade("Market is DEAD (low volatility)", regime)
        
        if regime == MarketRegime.TRANSITION:
            return self._no_trade("Market in TRANSITION (regime unclear)", regime)
        
        if regime == MarketRegime.TREND:
            return self._route_to_trend(features, symbol, asset_config, regime)
        
        if regime == MarketRegime.CHOPPY:
            return self._route_to_choppy(features, symbol, asset_config, df, regime)
        
        return self._no_trade(f"Unknown regime: {regime}", regime)
    
    def _no_trade(self, reason: str, regime: MarketRegime) -> HybridResult:
        """Return no-trade result."""
        return HybridResult(
            signal='HOLD',
            confidence=0.0,
            source=SignalSource.NO_TRADE,
            regime=regime,
            sl_atr=0,
            tp_atr=0,
            size_multiplier=0,
            reason=reason,
            details={}
        )
    
    def _route_to_trend(
        self,
        features: Dict,
        symbol: str,
        asset_config,
        regime: MarketRegime
    ) -> HybridResult:
        """Route to trend-following ML brain."""
        
        # Check if asset allows trend trading
        if asset_config.size_multiplier == 0:
            return self._no_trade(
                f"{symbol} is SHADOW for trend trading",
                regime
            )
        
        # Get ML prediction
        signal, confidence = self.trend_brain.predict(features)
        
        if signal == 'HOLD':
            return HybridResult(
                signal='HOLD',
                confidence=0.0,
                source=SignalSource.TREND_BRAIN,
                regime=regime,
                sl_atr=0,
                tp_atr=0,
                size_multiplier=0,
                reason="Trend brain: HOLD",
                details={'ml_signal': signal}
            )
        
        # Check confidence threshold
        min_conf = asset_config.min_confidence
        if confidence < min_conf:
            return HybridResult(
                signal='HOLD',
                confidence=confidence,
                source=SignalSource.TREND_BRAIN,
                regime=regime,
                sl_atr=0,
                tp_atr=0,
                size_multiplier=0,
                reason=f"Trend confidence {confidence:.2f} < {min_conf}",
                details={'ml_signal': signal, 'ml_confidence': confidence}
            )
        
        return HybridResult(
            signal=signal,
            confidence=confidence,
            source=SignalSource.TREND_BRAIN,
            regime=regime,
            sl_atr=self.TREND_SL_ATR,
            tp_atr=self.TREND_TP_ATR,
            size_multiplier=asset_config.size_multiplier,
            reason=f"Trend {signal} @ {confidence:.2f}",
            details={'ml_signal': signal, 'ml_confidence': confidence}
        )
    
    def _route_to_choppy(
        self,
        features: Dict,
        symbol: str,
        asset_config,
        df: Optional[pd.DataFrame],
        regime: MarketRegime
    ) -> HybridResult:
        """Route to choppy market engine."""
        
        # Check if asset allows choppy trading
        choppy_enabled = getattr(asset_config, 'choppy_enabled', False)
        if not choppy_enabled:
            return self._no_trade(
                f"{symbol} choppy trading disabled",
                regime
            )
        
        # Get choppy engine result
        result = self.choppy_engine.evaluate(features, df)
        
        if result.signal in [ChoppySignal.HOLD, ChoppySignal.NO_TRADE]:
            return HybridResult(
                signal='HOLD',
                confidence=0.0,
                source=SignalSource.CHOPPY_ENGINE,
                regime=regime,
                sl_atr=0,
                tp_atr=0,
                size_multiplier=0,
                reason=result.reason,
                details=result.details
            )
        
        # Get choppy size multiplier from asset config
        choppy_size = getattr(asset_config, 'choppy_size_mult', 0.5)
        final_size = result.size_multiplier * choppy_size
        
        return HybridResult(
            signal=result.signal.value,
            confidence=result.confidence,
            source=SignalSource.CHOPPY_ENGINE,
            regime=regime,
            sl_atr=result.sl_atr,
            tp_atr=result.tp_atr,
            size_multiplier=final_size,
            reason=result.reason,
            details=result.details
        )
    
    def get_exit_params(
        self,
        entry_price: float,
        side: str,
        atr: float,
        source: SignalSource,
        sl_atr: float,
        tp_atr: float
    ) -> Dict[str, float]:
        """
        Calculate SL and TP prices.
        
        Returns:
            Dict with 'sl' and 'tp' prices
        """
        if side == 'BUY':
            sl = entry_price - (sl_atr * atr)
            tp = entry_price + (tp_atr * atr)
        else:  # SELL
            sl = entry_price + (sl_atr * atr)
            tp = entry_price - (tp_atr * atr)
        
        return {
            'sl': sl,
            'tp': tp,
            'sl_distance': abs(entry_price - sl),
            'tp_distance': abs(entry_price - tp),
            'rr_ratio': tp_atr / sl_atr if sl_atr > 0 else 0
        }
    
    def should_early_exit(
        self,
        features: Dict,
        side: str,
        source: SignalSource
    ) -> Tuple[bool, str]:
        """
        Check if should exit early based on current source.
        
        Only choppy engine has early exit logic.
        """
        if source == SignalSource.CHOPPY_ENGINE:
            return self.choppy_engine.should_early_exit(features, side)
        
        return False, ""


# ============================================================================
# Convenience functions
# ============================================================================

def hybrid_decide(
    features: Dict[str, float],
    symbol: str,
    df: Optional[pd.DataFrame] = None
) -> HybridResult:
    """Quick hybrid decision."""
    brain = HybridDecisionBrain()
    return brain.decide(features, symbol, df)
