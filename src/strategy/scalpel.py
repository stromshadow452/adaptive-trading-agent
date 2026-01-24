"""
WEAPON SYSTEM: Scalpel Strategies (Micro Trading)
==================================================

Scalpel = small, precise, low-risk trades
- Size: 10-15% of base size
- Used when ML confidence is low-medium (0.45-0.65)
- Only in calm market conditions
"""

from typing import Optional
import logging

from src.strategy.base import (
    Strategy, WeaponClass, Signal,
    MarketSnapshot, AgentContext, StrategySignal
)

logger = logging.getLogger(__name__)


# ============================================================================
# SCALPEL BASE CLASS
# ============================================================================

class ScalpelStrategy(Strategy):
    """
    Base class for micro/exploration strategies.
    
    Size: 10-15% of base size
    Conditions: Low-medium confidence, B/C tier, calm markets
    """
    
    SIZE_MULTIPLIER = 0.12  # 12% of base size
    
    @property
    def weapon_class(self) -> WeaponClass:
        return WeaponClass.SCALPEL
    
    def get_size_multiplier(self) -> float:
        return self.SIZE_MULTIPLIER
    
    def is_allowed(self, context: AgentContext) -> bool:
        """Check if scalpel trading is allowed."""
        # Base checks
        if not super().is_allowed(context):
            return False
        
        # Scalpel-specific conditions
        if context.ml_confidence < 0.45 or context.ml_confidence > 0.70:
            return False
        if context.edge_tier not in ('B', 'C'):
            return False
        if context.regime not in ('RANGE', 'TREND'):
            return False
        if context.regime_strength > 0.75:  # Too volatile
            return False
        if context.memory_pain_level > 0.5:  # In pain zone
            return False
        return True


# ============================================================================
# MICRO STRATEGY 1: Range Mean-Reversion
# ============================================================================

class RangeMicroMR(ScalpelStrategy):
    """
    Mean-reversion in tight ranges.
    
    Logic:
    - Detects price at range extremes (using Bollinger bands)
    - Fades moves with tight SL
    - Small size (12%)
    
    Entry Conditions:
    - RANGE regime
    - Price at/beyond Bollinger bands
    - Low-medium ML confidence
    """
    
    @property
    def name(self) -> str:
        return "range_micro_mr"
    
    def evaluate(
        self,
        snapshot: MarketSnapshot,
        context: AgentContext,
    ) -> Optional[StrategySignal]:
        """Evaluate for mean-reversion opportunity."""
        
        # Only in RANGE regime
        if context.regime != 'RANGE':
            return None
        
        # Check if allowed
        if not self.is_allowed(context):
            return None
        
        price = snapshot.price
        atr = snapshot.atr
        
        # Use Bollinger bands if available, else ATR-based levels
        upper = snapshot.bb_upper if snapshot.bb_upper > 0 else snapshot.sma_20 + 2 * atr
        lower = snapshot.bb_lower if snapshot.bb_lower > 0 else snapshot.sma_20 - 2 * atr
        mid = snapshot.sma_20 if snapshot.sma_20 > 0 else (upper + lower) / 2
        
        # Skip if bands not valid
        if upper <= lower or mid <= 0:
            return None
        
        # Check for overbought (price at upper band)
        if price >= upper - atr * 0.1:
            sl = price + atr * 0.5
            tp = mid  # Target mean
            
            return StrategySignal(
                signal=Signal.SELL,
                confidence=0.52,
                suggested_sl=sl,
                suggested_tp=tp,
                reason=f"Range micro MR: overbought fade @ {price:.5f}",
                strategy_name=self.name,
                weapon_class=self.weapon_class.value,
                size_multiplier=self.get_size_multiplier()
            )
        
        # Check for oversold (price at lower band)
        if price <= lower + atr * 0.1:
            sl = price - atr * 0.5
            tp = mid  # Target mean
            
            return StrategySignal(
                signal=Signal.BUY,
                confidence=0.52,
                suggested_sl=sl,
                suggested_tp=tp,
                reason=f"Range micro MR: oversold fade @ {price:.5f}",
                strategy_name=self.name,
                weapon_class=self.weapon_class.value,
                size_multiplier=self.get_size_multiplier()
            )
        
        return None


# ============================================================================
# MICRO STRATEGY 2: Liquidity Sweep Fade
# ============================================================================

class LiquiditySweepFade(ScalpelStrategy):
    """
    Fade liquidity sweeps (stop hunts).
    
    Logic:
    - Detects sweep of recent swing high/low
    - Waits for rejection (price returns inside range)
    - Fades the sweep
    
    Entry Conditions:
    - Calm market (regime_strength < 0.65)
    - Price swept then rejected from 20-bar high/low
    - Small size (12%)
    """
    
    SWEEP_THRESHOLD = 0.3  # ATR multiplier for sweep detection
    
    @property
    def name(self) -> str:
        return "liquidity_sweep_fade"
    
    def evaluate(
        self,
        snapshot: MarketSnapshot,
        context: AgentContext,
    ) -> Optional[StrategySignal]:
        """Evaluate for liquidity sweep fade opportunity."""
        
        # Only when calm
        if context.regime_strength > 0.65:
            return None
        
        # Only in RANGE or calm TREND
        if context.regime not in ('RANGE', 'TREND'):
            return None
        
        # Check if allowed
        if not self.is_allowed(context):
            return None
        
        price = snapshot.price
        atr = snapshot.atr
        high_20 = snapshot.high_20
        low_20 = snapshot.low_20
        
        # Skip if swing levels not valid
        if high_20 <= 0 or low_20 <= 0:
            return None
        
        sweep_buffer = atr * self.SWEEP_THRESHOLD
        
        # Detect high sweep and rejection
        # Price was above high_20 but now back below
        if price < high_20 and price > high_20 - sweep_buffer:
            # Possible high sweep rejection
            sl = high_20 + atr * 0.7
            tp = price - atr * 1.2
            
            return StrategySignal(
                signal=Signal.SELL,
                confidence=0.50,
                suggested_sl=sl,
                suggested_tp=tp,
                reason=f"Liquidity sweep fade: high swept @ {high_20:.5f}",
                strategy_name=self.name,
                weapon_class=self.weapon_class.value,
                size_multiplier=self.get_size_multiplier()
            )
        
        # Detect low sweep and rejection
        # Price was below low_20 but now back above
        if price > low_20 and price < low_20 + sweep_buffer:
            # Possible low sweep rejection
            sl = low_20 - atr * 0.7
            tp = price + atr * 1.2
            
            return StrategySignal(
                signal=Signal.BUY,
                confidence=0.50,
                suggested_sl=sl,
                suggested_tp=tp,
                reason=f"Liquidity sweep fade: low swept @ {low_20:.5f}",
                strategy_name=self.name,
                weapon_class=self.weapon_class.value,
                size_multiplier=self.get_size_multiplier()
            )
        
        return None
