"""
MARK-2 Regime Strength Module

Provides continuous regime measurement instead of binary labels:
- Regime Strength: How strong is the current regime (0.0 - 1.0)
- Transition Probability: How likely the regime will change
- Hysteresis: Prevents flip-flopping

Uses ONLY OHLCV-derived statistics.

"Weather forecasting, not labeling."
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class RegimeStrengthOutput:
    """Output from regime strength calculation."""
    current_regime: str
    regime_strength: float  # 0.0 - 1.0
    transition_prob: float  # 0.0 - 1.0
    bars_in_regime: int
    
    def to_dict(self) -> dict:
        return {
            'current_regime': self.current_regime,
            'regime_strength': self.regime_strength,
            'transition_prob': self.transition_prob,
            'bars_in_regime': self.bars_in_regime,
        }


class RegimeStrength:
    """
    MARK-2 Regime Strength System
    
    Features:
    - Continuous regime strength (0.0 - 1.0)
    - Transition probability estimation
    - Hysteresis to prevent flip-flopping
    - Exponential smoothing for stability
    
    Uses only OHLCV-derived features.
    """
    
    # Hysteresis thresholds
    ENTRY_THRESHOLD = 0.6   # Strength needed to ENTER a regime
    EXIT_THRESHOLD = 0.3    # Strength needed to EXIT a regime
    
    # Smoothing
    SMOOTHING_ALPHA = 0.3   # How fast strength responds (0 = slow, 1 = instant)
    
    # Minimum duration
    MIN_REGIME_BARS = 5     # Regime must persist for 5 bars
    
    def __init__(self):
        self.current_regime: str = "RANGE"
        self.bars_in_regime: int = 0
        
        # Smoothed strengths
        self.smoothed_trend_strength: float = 0.5
        self.smoothed_range_strength: float = 0.5
        self.smoothed_danger_strength: float = 0.5
        
        # Current output
        self.regime_strength: float = 0.5
        self.transition_prob: float = 0.0
        
        logger.info("RegimeStrength initialized - Continuous regime measurement active")
    
    def update(self, features: dict) -> RegimeStrengthOutput:
        """
        Update regime strength from latest features.
        
        Required features (from existing feature pipeline):
        - adx_14 or ADX_14
        - atr_14 or ATR_14
        - atr_ratio (optional, will calculate if missing)
        - close, sma_20, high_20, low_20 (optional)
        """
        # Calculate raw strengths
        raw_trend = self._calculate_trend_strength(features)
        raw_range = self._calculate_range_strength(features)
        raw_danger = self._calculate_danger_strength(features)
        
        # Apply exponential smoothing
        self.smoothed_trend_strength = self._smooth(
            self.smoothed_trend_strength, raw_trend
        )
        self.smoothed_range_strength = self._smooth(
            self.smoothed_range_strength, raw_range
        )
        self.smoothed_danger_strength = self._smooth(
            self.smoothed_danger_strength, raw_danger
        )
        
        # Check for regime transition
        self._maybe_transition()
        
        # Update current regime strength
        if self.current_regime == "TREND":
            self.regime_strength = self.smoothed_trend_strength
        elif self.current_regime == "RANGE":
            self.regime_strength = self.smoothed_range_strength
        elif self.current_regime == "DANGER":
            self.regime_strength = self.smoothed_danger_strength
        else:
            self.regime_strength = 0.5
        
        # Calculate transition probability
        self.transition_prob = self._calculate_transition_probability(features)
        
        # Increment bars in regime
        self.bars_in_regime += 1
        
        return RegimeStrengthOutput(
            current_regime=self.current_regime,
            regime_strength=self.regime_strength,
            transition_prob=self.transition_prob,
            bars_in_regime=self.bars_in_regime
        )
    
    def _calculate_trend_strength(self, features: dict) -> float:
        """
        Calculate trend strength from features.
        Higher = stronger trend.
        """
        # Get ADX (primary trend indicator)
        adx = self._get_feature(features, ['adx_14', 'ADX_14', 'M5_adx_14'], default=25.0)
        adx_score = self._normalize(adx, 15.0, 40.0)
        
        # Get price vs SMA
        close = self._get_feature(features, ['close', 'Close'], default=0.0)
        sma = self._get_feature(features, ['sma_20', 'SMA_20', 'M5_sma_20'], default=close)
        atr = self._get_feature(features, ['atr_14', 'ATR_14', 'M5_atr_14'], default=1.0)
        
        if atr > 0 and close > 0:
            price_vs_sma = abs(close - sma) / atr
            distance_score = self._normalize(price_vs_sma, 0.0, 3.0)
        else:
            distance_score = 0.5
        
        # Get direction consistency (use volatility as proxy)
        direction_score = self._normalize(adx, 20.0, 35.0)
        
        # Weighted combination
        trend_strength = (
            0.4 * adx_score +
            0.3 * distance_score +
            0.3 * direction_score
        )
        
        return max(0.0, min(1.0, trend_strength))
    
    def _calculate_range_strength(self, features: dict) -> float:
        """
        Calculate range strength from features.
        Higher = stronger range/mean-reversion environment.
        """
        # Inverse ADX (low ADX = ranging)
        adx = self._get_feature(features, ['adx_14', 'ADX_14', 'M5_adx_14'], default=25.0)
        inv_adx_score = 1.0 - self._normalize(adx, 15.0, 40.0)
        
        # Tight channel (low range width relative to ATR)
        high_20 = self._get_feature(features, ['high_20', 'High_20'], default=0.0)
        low_20 = self._get_feature(features, ['low_20', 'Low_20'], default=0.0)
        atr = self._get_feature(features, ['atr_14', 'ATR_14', 'M5_atr_14'], default=1.0)
        
        if atr > 0 and high_20 > low_20:
            range_width = (high_20 - low_20) / atr
            channel_score = 1.0 - self._normalize(range_width, 2.0, 6.0)
        else:
            channel_score = 0.5
        
        # Oscillation score (approximated by inverse ADX)
        oscillation_score = 1.0 - self._normalize(adx, 18.0, 35.0)
        
        # Weighted combination
        range_strength = (
            0.4 * inv_adx_score +
            0.4 * channel_score +
            0.2 * oscillation_score
        )
        
        return max(0.0, min(1.0, range_strength))
    
    def _calculate_danger_strength(self, features: dict) -> float:
        """
        Calculate danger/volatility strength from features.
        Higher = more dangerous/volatile environment.
        """
        # ATR ratio (current ATR vs historical)
        atr_ratio = self._get_feature(features, ['atr_ratio', 'ATR_ratio'], default=1.0)
        vol_spike_score = self._normalize(atr_ratio, 1.0, 2.5)
        
        # High volatility (absolute ATR level)
        atr = self._get_feature(features, ['atr_14', 'ATR_14', 'M5_atr_14'], default=0.0)
        close = self._get_feature(features, ['close', 'Close'], default=1.0)
        
        if close > 0:
            atr_pct = atr / close
            atr_level_score = self._normalize(atr_pct, 0.001, 0.005)
        else:
            atr_level_score = 0.5
        
        # Chaos score (approximated by very low or high ADX extremes)
        adx = self._get_feature(features, ['adx_14', 'ADX_14', 'M5_adx_14'], default=25.0)
        if adx < 15 or adx > 50:
            chaos_score = 0.7
        else:
            chaos_score = 0.3
        
        # Weighted combination
        danger_strength = (
            0.5 * vol_spike_score +
            0.3 * atr_level_score +
            0.2 * chaos_score
        )
        
        return max(0.0, min(1.0, danger_strength))
    
    def _calculate_transition_probability(self, features: dict) -> float:
        """
        Estimate probability of regime transition.
        Higher = more likely to change.
        """
        signals = 0.0
        
        # ADX crossing 25 warning
        adx = self._get_feature(features, ['adx_14', 'ADX_14', 'M5_adx_14'], default=25.0)
        if 23 < adx < 27:  # Near the 25 threshold
            signals += 0.4
        
        # Volatility shift
        atr_ratio = self._get_feature(features, ['atr_ratio', 'ATR_ratio'], default=1.0)
        vol_change = abs(atr_ratio - 1.0)
        signals += min(0.3, vol_change * 0.3)
        
        # Regime strength decay
        if self.regime_strength < 0.4:
            signals += (0.4 - self.regime_strength) * 0.75
        
        # Breakout pressure (for RANGE)
        if self.current_regime == "RANGE":
            # Use smoothed trend strength as breakout indicator
            if self.smoothed_trend_strength > 0.5:
                signals += 0.3
        
        # Reduce if regime is new (still stabilizing)
        if self.bars_in_regime < self.MIN_REGIME_BARS:
            signals *= 0.5
        
        return min(1.0, signals)
    
    def _maybe_transition(self):
        """Check if regime should change (with hysteresis)."""
        
        # DANGER takes priority
        if self.smoothed_danger_strength > self.ENTRY_THRESHOLD:
            if self.current_regime != "DANGER":
                self._transition_to("DANGER")
            return
        
        # Exit DANGER check
        if self.current_regime == "DANGER":
            if self.smoothed_danger_strength < self.EXIT_THRESHOLD:
                # Allow fall-through to TREND/RANGE check
                pass
            else:
                return  # Stay in DANGER
        
        # TREND vs RANGE
        if self.smoothed_trend_strength > self.ENTRY_THRESHOLD:
            if self.current_regime != "TREND":
                self._transition_to("TREND")
        elif self.smoothed_trend_strength < self.EXIT_THRESHOLD:
            if self.smoothed_range_strength > self.EXIT_THRESHOLD:
                if self.current_regime != "RANGE":
                    self._transition_to("RANGE")
    
    def _transition_to(self, new_regime: str):
        """Switch to new regime."""
        old_regime = self.current_regime
        self.current_regime = new_regime
        self.bars_in_regime = 0
        logger.info(f"REGIME: Transition {old_regime} → {new_regime}")
    
    def _smooth(self, old_value: float, new_value: float) -> float:
        """Exponential smoothing."""
        return (
            self.SMOOTHING_ALPHA * new_value +
            (1 - self.SMOOTHING_ALPHA) * old_value
        )
    
    def _normalize(self, value: float, min_val: float, max_val: float) -> float:
        """Normalize value to 0.0 - 1.0 range."""
        if max_val <= min_val:
            return 0.5
        return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))
    
    def _get_feature(self, features: dict, keys: list, default: float = 0.0) -> float:
        """Get feature by trying multiple key names."""
        for key in keys:
            if key in features:
                val = features[key]
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    return float(val)
        return default
    
    def get_modifiers(self) -> Tuple[float, float, float]:
        """
        Returns (size_modifier, rr_adjustment, exploration_boost).
        
        size_modifier: 0.5 - 1.0
        rr_adjustment: -0.1 to +0.2
        exploration_boost: 1.0 - 1.5
        """
        # Size modifier (weak regime = reduce size)
        if self.regime_strength < 0.5:
            size_mod = 0.5 + self.regime_strength  # 0.5 to 1.0
        else:
            size_mod = 1.0
        
        # RR adjustment
        if self.regime_strength >= 0.7:
            rr_adj = -0.1  # Strong regime = accept lower RR
        elif self.regime_strength < 0.4:
            rr_adj = +0.2  # Weak regime = require higher RR
        else:
            rr_adj = 0.0
        
        # Exploration boost (high transition prob = more exploration)
        if self.transition_prob > 0.5:
            explore_boost = 1.5
        else:
            explore_boost = 1.0
        
        return (size_mod, rr_adj, explore_boost)
    
    def get_state(self) -> dict:
        """Get current regime strength state for logging."""
        size_mod, rr_adj, explore_boost = self.get_modifiers()
        
        return {
            'current_regime': self.current_regime,
            'regime_strength': self.regime_strength,
            'transition_prob': self.transition_prob,
            'bars_in_regime': self.bars_in_regime,
            'size_modifier': size_mod,
            'rr_adjustment': rr_adj,
            'exploration_boost': explore_boost,
            'smoothed_trend': self.smoothed_trend_strength,
            'smoothed_range': self.smoothed_range_strength,
            'smoothed_danger': self.smoothed_danger_strength,
        }
    
    def reset(self):
        """Reset regime strength state."""
        self.current_regime = "RANGE"
        self.bars_in_regime = 0
        self.smoothed_trend_strength = 0.5
        self.smoothed_range_strength = 0.5
        self.smoothed_danger_strength = 0.5
        self.regime_strength = 0.5
        self.transition_prob = 0.0
        logger.info("REGIME STRENGTH: Reset complete")
