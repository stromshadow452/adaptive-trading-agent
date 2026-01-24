"""
WEAPON SYSTEM: Rifle Strategies (Standard Trading)
===================================================

Rifle = standard full-size trades
- Size: 100% of base size (subject to EDGE + MARK-2)
- Used when ML confidence is high (>0.60)
- Primary strategy route
"""

from typing import Optional
import logging

from src.strategy.base import (
    Strategy, WeaponClass, Signal,
    MarketSnapshot, AgentContext, StrategySignal
)

logger = logging.getLogger(__name__)


# ============================================================================
# RIFLE BASE CLASS
# ============================================================================

class RifleStrategy(Strategy):
    """
    Base class for standard trading strategies.
    
    Size: 100% of base size
    Conditions: High confidence, A/B tier, not DANGER
    """
    
    SIZE_MULTIPLIER = 1.0  # Full base size
    
    @property
    def weapon_class(self) -> WeaponClass:
        return WeaponClass.RIFLE
    
    def get_size_multiplier(self) -> float:
        return self.SIZE_MULTIPLIER
    
    def is_allowed(self, context: AgentContext) -> bool:
        """Check if rifle trading is allowed."""
        # Base checks
        if not super().is_allowed(context):
            return False
        
        # Rifle-specific conditions
        if context.ml_confidence < 0.60:
            return False
        if context.edge_tier not in ('A', 'B'):
            return False
        return True


# ============================================================================
# PRIMARY STRATEGY: ML-Driven Direction
# ============================================================================

class MLPrimaryStrategy(RifleStrategy):
    """
    Primary ML-driven strategy.
    
    Logic:
    - Uses ML Brain prediction as primary signal
    - Full size with EDGE + MARK-2 modifiers
    - Main trading path
    """
    
    @property
    def name(self) -> str:
        return "ml_primary"
    
    def evaluate(
        self,
        snapshot: MarketSnapshot,
        context: AgentContext,
    ) -> Optional[StrategySignal]:
        """Evaluate ML prediction for trade signal."""
        
        # Check if allowed
        if not self.is_allowed(context):
            return None
        
        # Use ML prediction directly
        if context.ml_prediction == 'HOLD':
            return None
        
        price = snapshot.price
        atr = snapshot.atr
        rr = context.rr_ratio
        
        if context.ml_prediction == 'BUY':
            sl = price - atr * 1.0
            tp = price + atr * rr
            
            return StrategySignal(
                signal=Signal.BUY,
                confidence=context.ml_confidence,
                suggested_sl=sl,
                suggested_tp=tp,
                reason=f"ML Primary: BUY conf={context.ml_confidence:.2f}",
                strategy_name=self.name,
                weapon_class=self.weapon_class.value,
                size_multiplier=self.get_size_multiplier()
            )
        
        if context.ml_prediction == 'SELL':
            sl = price + atr * 1.0
            tp = price - atr * rr
            
            return StrategySignal(
                signal=Signal.SELL,
                confidence=context.ml_confidence,
                suggested_sl=sl,
                suggested_tp=tp,
                reason=f"ML Primary: SELL conf={context.ml_confidence:.2f}",
                strategy_name=self.name,
                weapon_class=self.weapon_class.value,
                size_multiplier=self.get_size_multiplier()
            )
        
        return None
