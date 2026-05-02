"""
SCOPUS Adaptive Regime Engine V2

Production-safe regime detection with:
- Adaptive thresholds based on AdaptiveState
- Volatility-aware regime classification
- DANGER regime for capital preservation
- Kalman slope integration
- Hysteresis to prevent flipping

Follows Jarvis-approved architecture.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from src.backtest.adaptive_state import AdaptiveState

# Import exact RANGE detector
try:
    from src.backtest.range_detector import ExactRangeDetector
    HAS_RANGE_DETECTOR = True
except ImportError:
    HAS_RANGE_DETECTOR = False

logger = logging.getLogger(__name__)


class RegimeEngineV1:
    """
    Adaptive Regime Engine - Dynamic thresholds with AdaptiveState integration.
    
    Detects market regimes: TREND, RANGE, DANGER
    
    Uses:
    - Volatility level from AdaptiveState
    - Kalman slope from features
    - Volatility change detection
    """
    
    def __init__(self, config: dict = None):
        config = config or {}
        
        # Base thresholds (adjusted by volatility level)
        self.base_trend_threshold = config.get('trend_threshold', 0.015)
        self.base_range_threshold = config.get('range_threshold', 0.02)
        self.danger_threshold = config.get('danger_threshold', 3.2)
        self.vol_change_danger = config.get('vol_change_danger', 0.9)
        self.extreme_adx_threshold = config.get('extreme_adx_threshold', 40.0)
        self.extreme_atr_pctile = config.get('extreme_atr_pctile', 0.95)
        self.danger_instability_bb_mult = config.get('danger_instability_bb_mult', 1.5)
        self.max_danger_bars = config.get('max_danger_bars', 10)
        
        # Hysteresis to prevent flipping
        self.hysteresis_buffer = config.get('hysteresis_buffer', 0.2)
        
        # State
        self.current_regime = 'UNKNOWN'
        self.current_confidence = 0.0
        self.current_reason = 'UNKNOWN'
        self.prev_atr = None
        self.atr_history = []
        self.danger_bar_count = 0
        self.last_danger_reason = None
        
        # Exact RANGE detector (optional enhancement)
        self.use_exact_range = config.get('use_exact_range', False)
        self.range_detector = None
        if self.use_exact_range and HAS_RANGE_DETECTOR:
            self.range_detector = ExactRangeDetector(config.get('range_detection', {}))
            logger.info("[RegimeEngine] ExactRangeDetector enabled")
        
        logger.info(f"RegimeEngineV1 initialized with adaptive thresholds")
    
    def detect(self, features: pd.Series, 
               adaptive_state: Optional['AdaptiveState'] = None,
               operating_mode: str = 'CONFIRMATION') -> Tuple[str, float]:
        """
        Detect market regime using adaptive logic.
        
        Args:
            features: Feature series with indicators
            adaptive_state: Optional AdaptiveState for adaptive adjustments
            operating_mode: 'LEARNING' or 'CONFIRMATION' - affects DANGER handling
        
        Returns:
            (regime, confidence) tuple
            In LEARNING mode, DANGER returns 0.3 confidence for size reduction
            In CONFIRMATION mode, DANGER returns full confidence for blocking
        """
        # Get volatility level from AdaptiveState or compute locally
        if adaptive_state:
            vol_level = adaptive_state.volatility_level
            vol_percentile = adaptive_state.volatility_percentile
        else:
            vol_level = 'MID'
            vol_percentile = 0.5
        
        # Get Kalman slope (prefer from features, fallback to SMA-based)
        kalman_slope = features.get('kalman_slope', None)
        if kalman_slope is None:
            kalman_slope = self._compute_slope(features)
        
        # Get ATR and compute volatility change
        atr = features.get('atr_14', 0)
        vol_change = self._compute_vol_change(atr)
        atr_zscore = self._compute_atr_zscore(features)
        bb_width = features.get('bb_width', 0)
        
        adx_val = features.get('adx_14', features.get('M5_adx_14', features.get('adx14', 0.0)))
        atr_pctile = features.get('atr_pctile', features.get('M5_atr_pctile', 0.5))

        adx_val = float(adx_val)
        atr_pctile = float(atr_pctile)

        # === HARD ADX HIERARCHY (non-bypassable) ===
        if adx_val >= 30.0:
            regime = 'TREND'
            confidence = min(1.0, max(0.6, adx_val / 40.0))
            reason = 'ADX_HARD_TREND'
            self.current_regime = regime
            self.current_confidence = confidence
            self.current_reason = reason
            if adaptive_state:
                adaptive_state.update_regime(regime)
            return regime, confidence

        if adx_val <= 20.0:
            regime = 'RANGE'
            confidence = min(1.0, max(0.6, (20.0 - adx_val) / 20.0 + 0.6))
            reason = 'ADX_HARD_RANGE'
            self.current_regime = regime
            self.current_confidence = confidence
            self.current_reason = reason
            if adaptive_state:
                adaptive_state.update_regime(regime)
            return regime, confidence

        # === ADAPTIVE CLASSIFICATION ===
        regime, confidence, reason = self._classify_adaptive(
            kalman_slope=kalman_slope,
            bb_width=bb_width,
            atr_zscore=atr_zscore,
            vol_level=vol_level,
            vol_change=vol_change,
            vol_percentile=vol_percentile,
            adx_val=adx_val,
            atr_pctile=atr_pctile,
        )
        
        # === LEARNING MODE: Soft DANGER handling ===
        # In LEARNING mode, DANGER regime allows trades with reduced size
        # instead of blocking completely
        if regime == 'DANGER' and operating_mode == 'LEARNING':
            # Return DANGER but with lower confidence (0.3) to allow
            # trades with 50% size reduction instead of blocking
            logger.info(f"LEARNING mode: DANGER soft-pass (size reduction, not block)")
            confidence = 0.3  # Signal for 50% size reduction
        
        # Apply hysteresis
        regime = self._apply_hysteresis(regime, confidence)

        if regime == 'DANGER':
            if self.current_regime == 'DANGER':
                self.danger_bar_count += 1
            else:
                self.danger_bar_count = 1
            self.last_danger_reason = reason

            if self.danger_bar_count > self.max_danger_bars:
                logger.info("DANGER timeout reached (%s bars) -> downgrade to NEUTRAL", self.max_danger_bars)
                regime = 'NEUTRAL'
                confidence = 0.35
                reason = 'danger_timeout_neutral'
                self.danger_bar_count = 0
                self.last_danger_reason = reason
        else:
            self.danger_bar_count = 0
            if regime != 'UNKNOWN':
                self.last_danger_reason = None

        # Update state
        self.current_regime = regime
        self.current_confidence = confidence
        self.current_reason = reason
        
        # Update AdaptiveState if provided
        if adaptive_state:
            adaptive_state.update_regime(regime)
        
        return regime, confidence
    
    def _classify_adaptive(
        self,
        kalman_slope: float,
        bb_width: float,
        atr_zscore: float,
        vol_level: str,
        vol_change: float,
        vol_percentile: float,
        adx_val: float,
        atr_pctile: float,
    ) -> Tuple[str, float, str]:
        """
        Classify regime with adaptive thresholds.
        
        Priority: DANGER > TREND > RANGE
        """
        # === DANGER Detection (Priority 1) ===
        # DANGER requires stronger confirmation to avoid over-defensive paralysis.
        vol_spike = atr_zscore > self.danger_threshold
        rapid_vol_expansion = vol_change > self.vol_change_danger
        regime_instability = (
            vol_percentile > 0.80
            and bb_width > (self.base_range_threshold * self.danger_instability_bb_mult)
            and abs(kalman_slope) < (self.base_trend_threshold * 1.2)
        )
        extreme_trend_vol = adx_val > self.extreme_adx_threshold and atr_pctile > self.extreme_atr_pctile

        if (vol_spike and regime_instability) or (rapid_vol_expansion and regime_instability):
            trigger_strength = max(
                atr_zscore / max(self.danger_threshold, 1e-6),
                vol_change / max(self.vol_change_danger, 1e-6),
            )
            confidence = min(1.0, trigger_strength)
            reason = (
                f"danger_confirmed: vol_spike={vol_spike} "
                f"vol_change={vol_change:.2f} instability={regime_instability}"
            )
            logger.debug("DANGER: %s", reason)
            return 'DANGER', confidence, reason

        if extreme_trend_vol:
            confidence = min(1.0, max(adx_val / self.extreme_adx_threshold, atr_pctile / self.extreme_atr_pctile) - 0.5)
            reason = f"danger_extreme_combo: adx={adx_val:.2f} atr_pctile={atr_pctile:.2f}"
            logger.debug("DANGER: %s", reason)
            return 'DANGER', confidence, reason
        
        # === Adaptive Trend Threshold ===
        # Lower threshold in low vol (clearer signals), higher in high vol
        trend_thresh = self.base_trend_threshold
        if vol_level == 'LOW':
            trend_thresh *= 0.8  # 20% lower threshold
        elif vol_level == 'HIGH':
            trend_thresh *= 1.3  # 30% higher threshold
        
        # === TREND Detection ===
        if abs(kalman_slope) > trend_thresh:
            confidence = min(1.0, abs(kalman_slope) / trend_thresh)
            return 'TREND', confidence, 'trend_strength'
        
        # === RANGE Detection ===
        # Clearer in low volatility
        if vol_level == 'LOW' and abs(kalman_slope) < 0.008:
            return 'RANGE', 0.75, 'low_vol_range'

        if bb_width < self.base_range_threshold and bb_width > 0:
            confidence = 1.0 - (bb_width / self.base_range_threshold)
            return 'RANGE', max(0.5, confidence), 'compressed_range'

        # === Default: NEUTRAL with medium confidence ===
        return 'NEUTRAL', 0.5, 'neutral_default'
    
    def _compute_slope(self, features: pd.Series) -> float:
        """Compute trend slope from SMAs (fallback if no Kalman)"""
        sma_20 = features.get('sma_20', features.get('M5_sma_20', 0))
        sma_50 = features.get('sma_50', features.get('M5_sma_50', 0))
        
        if sma_50 == 0:
            return 0
        
        return (sma_20 - sma_50) / sma_50
    
    def _compute_atr_zscore(self, features: pd.Series) -> float:
        """Compute ATR z-score"""
        atr = features.get('atr_14', features.get('M5_atr_14', 0))
        close = features.get('close', features.get('Close', 1))
        
        # ATR as % of price
        atr_pct = atr / close if close > 0 else 0
        
        # Z-score approximation (assuming mean=0.01, std=0.003)
        z_score = (atr_pct - 0.01) / 0.003
        
        return z_score
    
    def _compute_vol_change(self, current_atr: float) -> float:
        """
        Compute volatility change rate.
        
        Detects sudden volatility spikes for DANGER detection.
        """
        self.atr_history.append(current_atr)
        if len(self.atr_history) > 10:
            self.atr_history = self.atr_history[-10:]
        
        if len(self.atr_history) < 5:
            return 0.0
        
        # Compare recent 3 vs older 3
        recent = sum(self.atr_history[-3:]) / 3
        older = sum(self.atr_history[-6:-3]) / 3
        
        if older == 0:
            return 0.0
        
        return (recent - older) / older
    
    def _apply_hysteresis(self, new_regime: str, confidence: float) -> str:
        """
        Apply hysteresis to prevent regime flipping.
        
        Requires new regime to have confidence > current + buffer
        to switch.
        """
        if self.current_regime == 'UNKNOWN':
            return new_regime
        
        # Always switch to DANGER immediately (no hysteresis for safety)
        if new_regime == 'DANGER':
            if new_regime != self.current_regime:
                logger.info(f"Regime DANGER activated (no hysteresis)")
            return new_regime
        
        # If same regime, keep it
        if new_regime == self.current_regime:
            return new_regime
        
        # If different, require confidence + buffer
        if confidence > (self.current_confidence + self.hysteresis_buffer):
            logger.info(f"Regime change: {self.current_regime} → {new_regime} (conf={confidence:.2f})")
            return new_regime
        
        # Otherwise, keep current regime
        return self.current_regime
    
    def get_regime_context(self) -> dict:
        """Get current regime context for logging"""
        return {
            'regime': self.current_regime,
            'confidence': self.current_confidence,
            'regime_reason': self.current_reason,
            'danger_reason': self.last_danger_reason,
            'danger_duration_bars': self.danger_bar_count,
        }
