"""
MARK-2 Intelligence Integration

Combines all MARK-2 modules into a single intelligence layer:
- Memory Module: Pain zones, regime pain, loss clusters
- Ego Control: Overconfidence prevention
- Regime Strength: Continuous regime measurement

"MARK-2: The suit that throttles itself before the pilot does something stupid."
"""

from typing import Tuple, Optional
from dataclasses import dataclass
import logging

from src.backtest.memory_module import MemoryModule
from src.backtest.ego_control import EgoControl
from src.backtest.regime_strength import RegimeStrength

logger = logging.getLogger(__name__)


@dataclass
class MARK2Output:
    """Combined output from all MARK-2 modules."""
    
    # Combined modifiers
    final_size_modifier: float  # 0.1 - 1.0
    final_rr_adjustment: float  # -0.1 to +0.2
    final_conf_boost: float     # 0.0 - 0.15
    exploration_boost: float    # 1.0 - 1.5
    
    # Can trade check
    can_trade: bool
    cooldown_remaining_min: float
    
    # Individual components (for logging)
    memory_mod: float
    ego_mod: float
    regime_mod: float
    
    # Regime info
    regime_strength: float
    transition_prob: float
    
    # Ego info
    ego_score: float
    
    def to_dict(self) -> dict:
        return {
            'final_size_modifier': self.final_size_modifier,
            'final_rr_adjustment': self.final_rr_adjustment,
            'final_conf_boost': self.final_conf_boost,
            'exploration_boost': self.exploration_boost,
            'can_trade': self.can_trade,
            'memory_mod': self.memory_mod,
            'ego_mod': self.ego_mod,
            'regime_mod': self.regime_mod,
            'regime_strength': self.regime_strength,
            'transition_prob': self.transition_prob,
            'ego_score': self.ego_score,
        }


class MARK2Intelligence:
    """
    MARK-2 Unified Intelligence Layer
    
    Integrates:
    - Memory (pain avoidance)
    - Ego (overconfidence prevention)
    - Regime Strength (continuous regime awareness)
    
    All effects are MULTIPLICATIVE.
    Final size modifier NEVER goes to zero (floor at 10%).
    
    Survival > Perfection
    """
    
    # Minimum size floor
    SIZE_FLOOR = 0.1
    
    def __init__(self):
        self.memory = MemoryModule()
        self.ego = EgoControl()
        self.regime = RegimeStrength()
        
        logger.info("MARK-2 Intelligence initialized - All modules active")
    
    def update_regime(self, features: dict):
        """
        Update regime strength from features.
        Call this on each candle.
        """
        self.regime.update(features)
    
    def get_modifiers(
        self,
        price: float,
        regime: str,
        side: str,
        base_min_conf: float = 0.35
    ) -> MARK2Output:
        """
        Get combined modifiers from all MARK-2 modules.
        
        Args:
            price: Current price for pain zone check
            regime: Current regime label
            side: Trade side (BUY/SELL)
            base_min_conf: Base minimum confidence threshold
        
        Returns:
            MARK2Output with all modifiers and status
        """
        # === MEMORY MODULE ===
        memory_mod = self.memory.get_memory_modifier(price, regime, side)
        
        # === EGO CONTROL ===
        ego_size_mod, cooldown_mult, conf_boost = self.ego.get_ego_modifiers()
        can_trade = self.ego.can_trade_now()
        
        # Calculate remaining cooldown
        if not can_trade and self.ego.last_trade_time:
            from datetime import datetime
            minutes_since = (datetime.utcnow() - self.ego.last_trade_time).total_seconds() / 60
            effective_cooldown = self.ego.BASE_COOLDOWN_MINUTES * cooldown_mult
            cooldown_remaining = max(0, effective_cooldown - minutes_since)
        else:
            cooldown_remaining = 0.0
        
        # === REGIME STRENGTH ===
        regime_size_mod, rr_adj, explore_boost = self.regime.get_modifiers()
        
        # === COMBINE MODIFIERS ===
        # Multiplicative combination for size
        combined_size = memory_mod * ego_size_mod * regime_size_mod
        combined_size = max(self.SIZE_FLOOR, combined_size)
        
        # RR adjustment is additive (from regime only)
        final_rr_adj = rr_adj
        
        # Confidence boost is from ego only
        final_conf_boost = conf_boost
        
        # Log if significant modification
        if combined_size < 0.9:
            logger.info(
                f"MARK-2: Size modifier={combined_size:.2f} "
                f"(memory={memory_mod:.2f}, ego={ego_size_mod:.2f}, regime={regime_size_mod:.2f})"
            )
        
        return MARK2Output(
            final_size_modifier=combined_size,
            final_rr_adjustment=final_rr_adj,
            final_conf_boost=final_conf_boost,
            exploration_boost=explore_boost,
            can_trade=can_trade,
            cooldown_remaining_min=cooldown_remaining,
            memory_mod=memory_mod,
            ego_mod=ego_size_mod,
            regime_mod=regime_size_mod,
            regime_strength=self.regime.regime_strength,
            transition_prob=self.regime.transition_prob,
            ego_score=self.ego.ego_score,
        )
    
    def record_trade_result(
        self,
        entry_price: float,
        atr: float,
        regime: str,
        side: str,
        is_win: bool,
        confidence: float,
        r_multiple: float = 0.0,
        loss_streak: int = 0
    ):
        """
        Record trade result to all memory systems.
        Call this after each trade closes.
        """
        # Update memory
        self.memory.record_trade_result(
            entry_price=entry_price,
            atr=atr,
            regime=regime,
            side=side,
            is_loss=not is_win,
            r_multiple=r_multiple
        )
        
        # Update ego
        self.ego.update_on_trade(
            is_win=is_win,
            confidence=confidence,
            regime=regime,
            side=side,
            loss_streak=loss_streak
        )
    
    def get_effective_min_confidence(self, base_conf: float) -> float:
        """Get minimum confidence with ego boost applied."""
        return self.ego.get_effective_min_confidence(base_conf)
    
    def get_effective_min_rr(self, base_min_rr: float) -> float:
        """Get minimum RR with regime adjustment applied."""
        _, rr_adj, _ = self.regime.get_modifiers()
        return max(1.0, base_min_rr + rr_adj)
    
    def get_state(self) -> dict:
        """Get complete MARK-2 state for logging/debugging."""
        return {
            'memory': self.memory.get_state(),
            'ego': self.ego.get_state(),
            'regime': self.regime.get_state(),
        }
    
    def reset(self):
        """Reset all MARK-2 modules."""
        self.memory.reset()
        self.ego.reset()
        self.regime.reset()
        logger.info("MARK-2: Full reset complete")


# ============================================================================
# INTEGRATION WITH RISKBRAIN
# ============================================================================

def apply_mark2_to_size(
    base_size: float,
    mark2: MARK2Intelligence,
    price: float,
    regime: str,
    side: str
) -> Tuple[float, dict]:
    """
    Apply MARK-2 modifiers to base position size.
    
    Returns:
        (final_size, mark2_info_dict)
    """
    output = mark2.get_modifiers(price, regime, side)
    final_size = base_size * output.final_size_modifier
    
    return final_size, output.to_dict()


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_mark2() -> MARK2Intelligence:
    """Factory function to create MARK-2 instance."""
    return MARK2Intelligence()
