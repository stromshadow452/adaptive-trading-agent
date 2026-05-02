"""
Meta-Gating Brain - Stage 9

Regime detection and model routing based on market conditions.
Uses CSV-derived indicators (ADX, volatility, volume) to classify market regime.

Regimes:
- TREND: Strong directional movement (ADX > 25, low volatility)
- RANGE: Sideways consolidation (ADX < 20, moderate volatility)
- CRASH: High volatility spike (volatility > 2x normal)
- UNCERTAIN: Mixed signals (default fallback)
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class MetaGatingBrain:
    """
    Regime-aware meta-gating for model selection.
    
    Determines market regime and provides routing decisions for:
    - Primary ML Brain
    - RL Brain
    - No-trade scenarios
    """
    
    def __init__(self,
                 adx_trend_threshold: float = 25.0,
                 adx_range_threshold: float = 20.0,
                 vol_crash_multiplier: float = 2.0,
                 lookback: int = 20):
        """
        Initialize Meta-Gating Brain.
        
        Args:
            adx_trend_threshold: ADX above this = TREND regime
            adx_range_threshold: ADX below this = RANGE regime
            vol_crash_multiplier: Volatility spike threshold for CRASH
            lookback: Rolling window for volatility calculation
        """
        self.adx_trend_threshold = adx_trend_threshold
        self.adx_range_threshold = adx_range_threshold
        self.vol_crash_multiplier = vol_crash_multiplier
        self.lookback = lookback
        
        # Historical volatility for baseline
        self.vol_history = []
        self.max_history = 100
    
    def classify_regime(self,
                       features: Dict[str, float],
                       volatility: Optional[float] = None,
                       symbol: str = "UNKNOWN") -> Dict[str, any]:
        """
        Classify current market regime.
        
        Args:
            features: Feature dict with indicators (close, atr14, rsi14, etc.)
            volatility: Current volatility (if available)
            symbol: Trading symbol
            
        Returns:
            {
                "regime": str,  # TREND, RANGE, CRASH, UNCERTAIN
                "action": str,  # ALLOW, BLOCK, REDUCE
                "size_multiplier": float,  # Size adjustment factor
                "confidence": float,  # Regime confidence [0-1]
                "reason": str  # Human-readable explanation
            }
        """
        # Extract indicators from features
        atr   = features.get("atr14", 0.0)
        close = features.get("close", 1.0)

        # True Wilder's ADX — computed upstream by calculate_wilder_adx()
        # adx14 is now part of FEATURE_LIST in common_features.py
        # Range: 0-100.  >25 = trend, <20 = range.
        adx_proxy = float(features.get("adx14", 0.0))

        # atr_pct still used for crash / volatility detection
        atr_pct = (atr / close) * 100 if close > 0 else 0
        
        # Calculate volatility if not provided
        if volatility is None:
            volatility = features.get("volatility", atr_pct)
        
        # Update volatility history
        self.vol_history.append(volatility)
        if len(self.vol_history) > self.max_history:
            self.vol_history.pop(0)
        
        # Calculate baseline volatility
        baseline_vol = np.mean(self.vol_history) if self.vol_history else volatility
        vol_ratio = volatility / baseline_vol if baseline_vol > 0 else 1.0
        
        # Regime classification logic
        regime = "UNCERTAIN"
        action = "ALLOW"
        size_multiplier = 1.0
        confidence = 0.5
        reason = ""
        
        # CRASH: Volatility spike
        if vol_ratio > self.vol_crash_multiplier:
            regime = "CRASH"
            action = "BLOCK"
            size_multiplier = 0.0
            confidence = min(vol_ratio / self.vol_crash_multiplier, 1.0)
            reason = f"Volatility spike detected ({vol_ratio:.2f}x normal)"
        
        # TREND: High ADX (true Wilder's), moderate volatility
        elif adx_proxy > self.adx_trend_threshold and vol_ratio < 1.5:
            regime = "TREND"
            action = "ALLOW"
            size_multiplier = 1.0
            confidence = min(adx_proxy / 50, 1.0)
            reason = f"Strong trend (ADX: {adx_proxy:.1f})"

        # RANGE: Low ADX, low volatility
        elif adx_proxy < self.adx_range_threshold and vol_ratio < 1.2:
            regime = "RANGE"
            action = "REDUCE"
            size_multiplier = 0.7  # Reduce size in ranging markets
            confidence = 1.0 - (adx_proxy / self.adx_range_threshold)
            reason = f"Range-bound market (ADX: {adx_proxy:.1f})"

        # UNCERTAIN: Mixed signals
        else:
            regime = "UNCERTAIN"
            action = "REDUCE"
            size_multiplier = 0.8
            confidence = 0.5
            reason = f"Mixed signals (ADX: {adx_proxy:.1f}, Vol: {vol_ratio:.2f}x)"
        
        logger.info(f"[META-GATING] {symbol}: regime={regime}, action={action}, "
                   f"size_mult={size_multiplier:.2f}, conf={confidence:.2f}")
        
        return {
            "regime": regime,
            "action": action,
            "size_multiplier": size_multiplier,
            "confidence": confidence,
            "reason": reason,
            "adx": adx_proxy,          # renamed from adx_proxy — now real ADX
            "adx_proxy": adx_proxy,    # kept for backward compat with callers
            "vol_ratio": vol_ratio,
        }
    
    def get_model_weights(self, regime: str) -> Dict[str, float]:
        """
        Get model routing weights based on regime.
        
        Args:
            regime: Current market regime
            
        Returns:
            {
                "primary": float,  # Weight for Primary ML Brain
                "rl": float,       # Weight for RL Brain
                "no_trade": float  # Weight for no-trade
            }
        """
        # RL agent is DISABLED (prototype with mock features — see src/rl/env.py).
        # All weight routes to primary ML Brain.
        weights = {
            "TREND":     {"primary": 1.0, "rl": 0.0, "no_trade": 0.0},
            "RANGE":     {"primary": 0.8, "rl": 0.0, "no_trade": 0.2},
            "CRASH":     {"primary": 0.0, "rl": 0.0, "no_trade": 1.0},
            "UNCERTAIN": {"primary": 0.7, "rl": 0.0, "no_trade": 0.3},
        }
        
        return weights.get(regime, {"primary": 0.5, "rl": 0.3, "no_trade": 0.2})
    
    def should_trade(self, regime_info: Dict[str, any]) -> bool:
        """
        Simple helper to determine if trading is allowed.
        
        Args:
            regime_info: Output from classify_regime()
            
        Returns:
            True if trading allowed, False otherwise
        """
        return regime_info["action"] != "BLOCK"
    
    def reset(self):
        """Reset volatility history (for new backtest session)."""
        self.vol_history = []


# Factory function for easy initialization
def create_meta_gating_brain(**kwargs) -> MetaGatingBrain:
    """Create and return MetaGatingBrain instance."""
    return MetaGatingBrain(**kwargs)
