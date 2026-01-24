"""
Risk Evolution System - Size Ladder & Earned Risk
===================================================

Implements the Month-2 to Year-3+ risk evolution plan:
- EARNED_RISK formula per pattern
- 4-tier size ladder (BASE → TIER-3)
- Auto-revert on losses
- Downside controls

RULES (LOCKED):
- MARK-2 veto remains absolute
- No blind size increase
- Risk is earned, never forced
- Per-pattern, not global
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, List
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS & CONSTANTS
# ============================================================================

class SizeTier(Enum):
    """Size tier levels."""
    BASE = "BASE"       # 1.00×
    TIER_1 = "TIER_1"   # 1.10×
    TIER_2 = "TIER_2"   # 1.15×
    TIER_3 = "TIER_3"   # 1.25×


TIER_MULTIPLIERS = {
    SizeTier.BASE: 1.00,
    SizeTier.TIER_1: 1.10,
    SizeTier.TIER_2: 1.15,
    SizeTier.TIER_3: 1.25,
}

# ============================================================================
# PHASE THRESHOLDS
# ============================================================================

@dataclass
class PhaseThresholds:
    """Thresholds for each evolution phase."""
    min_reps: int
    min_success_rate: float
    min_avg_r: float
    min_erm_confidence: float
    max_tier: SizeTier


# Phase definitions
PHASES = {
    "MONTH_2_3": PhaseThresholds(
        min_reps=8,
        min_success_rate=0.55,
        min_avg_r=0.10,
        min_erm_confidence=0.50,
        max_tier=SizeTier.TIER_1
    ),
    "MONTH_4_6": PhaseThresholds(
        min_reps=6,
        min_success_rate=0.55,
        min_avg_r=0.15,
        min_erm_confidence=0.55,
        max_tier=SizeTier.TIER_2
    ),
    "MONTH_7_12": PhaseThresholds(
        min_reps=5,
        min_success_rate=0.52,
        min_avg_r=0.12,
        min_erm_confidence=0.50,
        max_tier=SizeTier.TIER_3
    ),
    "YEAR_2_PLUS": PhaseThresholds(
        min_reps=4,
        min_success_rate=0.50,
        min_avg_r=0.10,
        min_erm_confidence=0.45,
        max_tier=SizeTier.TIER_3
    ),
}


# ============================================================================
# PATTERN RECORD
# ============================================================================

@dataclass
class PatternRecord:
    """Tracks a pattern's performance for tier calculation."""
    pattern_key: str
    
    # Performance metrics
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    total_r: float = 0.0
    
    # Recent performance
    consecutive_losses: int = 0
    last_5_results: List[bool] = field(default_factory=list)
    
    # Edge stability
    edge_scores: List[float] = field(default_factory=list)
    
    # Current tier
    tier: SizeTier = SizeTier.BASE
    tier_reps: int = 0  # Reps since tier upgrade
    
    # Timestamps
    first_seen: float = 0.0
    last_trade: float = 0.0
    tier_upgraded_at: float = 0.0
    
    @property
    def success_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.wins / self.total_trades
    
    @property
    def avg_r(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.total_r / self.total_trades
    
    @property
    def edge_stability(self) -> float:
        """Calculate edge score stability (lower std = more stable)."""
        if len(self.edge_scores) < 5:
            return 0.0
        
        import numpy as np
        recent = self.edge_scores[-10:]
        mean = np.mean(recent)
        std = np.std(recent)
        
        if mean == 0:
            return 0.0
        
        # Stability = 1 - (std/mean), capped at 0-1
        stability = 1.0 - (std / mean)
        return max(0.0, min(1.0, stability))
    
    def add_result(self, is_win: bool, r_multiple: float, edge_score: float):
        """Record a trade result."""
        self.total_trades += 1
        self.tier_reps += 1
        self.last_trade = time.time()
        
        if is_win:
            self.wins += 1
            self.consecutive_losses = 0
        else:
            self.losses += 1
            self.consecutive_losses += 1
        
        self.total_r += r_multiple
        
        # Track last 5 results
        self.last_5_results.append(is_win)
        if len(self.last_5_results) > 5:
            self.last_5_results = self.last_5_results[-5:]
        
        # Track edge scores
        self.edge_scores.append(edge_score)
        if len(self.edge_scores) > 20:
            self.edge_scores = self.edge_scores[-20:]


# ============================================================================
# EARNED RISK CALCULATOR
# ============================================================================

class EarnedRiskCalculator:
    """
    Calculates whether a pattern has EARNED additional risk.
    
    EARNED_RISK = (reps >= MIN) AND (SR >= MIN) AND (avgR >= MIN) AND (stability >= 0.6)
    """
    
    def __init__(self, phase: str = "MONTH_2_3"):
        self.phase = phase
        self.thresholds = PHASES.get(phase, PHASES["MONTH_2_3"])
    
    def set_phase(self, phase: str):
        """Update current phase."""
        if phase in PHASES:
            self.phase = phase
            self.thresholds = PHASES[phase]
            logger.info(f"[RISK-EVO] Phase set to {phase}")
    
    def is_earned(self, pattern: PatternRecord, erm_confidence: float = 0.5) -> bool:
        """
        Check if pattern has earned additional risk.
        
        Returns True if ALL conditions met:
        1. Minimum repetitions
        2. Minimum success rate
        3. Minimum avg R
        4. Minimum edge stability
        5. Minimum ERM confidence
        """
        # Check repetitions
        if pattern.total_trades < self.thresholds.min_reps:
            return False
        
        # Check success rate
        if pattern.success_rate < self.thresholds.min_success_rate:
            return False
        
        # Check avg R
        if pattern.avg_r < self.thresholds.min_avg_r:
            return False
        
        # Check edge stability
        if pattern.edge_stability < 0.60:
            return False
        
        # Check ERM confidence
        if erm_confidence < self.thresholds.min_erm_confidence:
            return False
        
        return True
    
    def get_max_tier(self) -> SizeTier:
        """Get maximum tier allowed in current phase."""
        return self.thresholds.max_tier


# ============================================================================
# SIZE LADDER MANAGER
# ============================================================================

class SizeLadderManager:
    """
    Manages the 4-tier size ladder.
    
    Tiers:
    - BASE: 1.00× (default)
    - TIER-1: 1.10× (8+ reps, earned)
    - TIER-2: 1.15× (20+ reps, earned)
    - TIER-3: 1.25× (40+ reps, A-grade, fragile)
    
    Rules:
    - Per-pattern, not global
    - Auto-revert on losses
    - TIER-3 is fragile (1 loss = back to BASE)
    """
    
    # Rep requirements for each tier
    TIER_1_REPS = 8
    TIER_2_REPS = 20
    TIER_3_REPS = 40
    
    def __init__(self, phase: str = "MONTH_2_3"):
        self.patterns: Dict[str, PatternRecord] = {}
        self.risk_calc = EarnedRiskCalculator(phase)
        
        # Monthly stats for downside control
        self.mtd_pnl: float = 0.0
        self.mtd_trades: int = 0
        self.frozen_until: float = 0.0
    
    def set_phase(self, phase: str):
        """Update evolution phase."""
        self.risk_calc.set_phase(phase)
    
    def get_or_create_pattern(self, pattern_key: str) -> PatternRecord:
        """Get or create pattern record."""
        if pattern_key not in self.patterns:
            self.patterns[pattern_key] = PatternRecord(
                pattern_key=pattern_key,
                first_seen=time.time()
            )
        return self.patterns[pattern_key]
    
    def get_size_multiplier(
        self, 
        pattern_key: str, 
        erm_confidence: float = 0.5,
        edge_tier: str = "B"
    ) -> float:
        """
        Get size multiplier for a pattern.
        
        Args:
            pattern_key: Unique pattern identifier
            erm_confidence: ERM confidence level
            edge_tier: EDGE tier (A/B/C/D)
        
        Returns:
            Size multiplier (1.0 to 1.25)
        """
        # Check if frozen (monthly drawdown control)
        if time.time() < self.frozen_until:
            logger.debug(f"[RISK-EVO] Frozen - returning BASE")
            return TIER_MULTIPLIERS[SizeTier.BASE]
        
        pattern = self.get_or_create_pattern(pattern_key)
        
        # Check if earned risk
        if not self.risk_calc.is_earned(pattern, erm_confidence):
            return TIER_MULTIPLIERS[SizeTier.BASE]
        
        # Calculate eligible tier based on reps
        max_tier = self.risk_calc.get_max_tier()
        
        if pattern.total_trades >= self.TIER_3_REPS and edge_tier == "A":
            eligible_tier = SizeTier.TIER_3
        elif pattern.total_trades >= self.TIER_2_REPS:
            eligible_tier = SizeTier.TIER_2
        elif pattern.total_trades >= self.TIER_1_REPS:
            eligible_tier = SizeTier.TIER_1
        else:
            eligible_tier = SizeTier.BASE
        
        # Cap at phase max
        tier_order = [SizeTier.BASE, SizeTier.TIER_1, SizeTier.TIER_2, SizeTier.TIER_3]
        eligible_idx = tier_order.index(eligible_tier)
        max_idx = tier_order.index(max_tier)
        
        final_tier = tier_order[min(eligible_idx, max_idx)]
        
        # Update pattern tier if upgrading
        if tier_order.index(final_tier) > tier_order.index(pattern.tier):
            pattern.tier = final_tier
            pattern.tier_reps = 0
            pattern.tier_upgraded_at = time.time()
            logger.info(f"[RISK-EVO] Pattern {pattern_key} upgraded to {final_tier.value}")
        
        return TIER_MULTIPLIERS[pattern.tier]
    
    def record_trade_result(
        self, 
        pattern_key: str, 
        is_win: bool, 
        r_multiple: float,
        edge_score: float,
        pnl: float
    ):
        """
        Record trade result and apply tier reversion rules.
        
        Reversion Rules:
        - TIER-3: Any loss → back to BASE
        - TIER-2: 1 loss → TIER-1
        - TIER-1: 2 consecutive losses → BASE
        """
        pattern = self.get_or_create_pattern(pattern_key)
        pattern.add_result(is_win, r_multiple, edge_score)
        
        # Update MTD stats
        self.mtd_pnl += pnl
        self.mtd_trades += 1
        
        # Apply reversion on loss
        if not is_win:
            old_tier = pattern.tier
            
            if pattern.tier == SizeTier.TIER_3:
                # TIER-3 is fragile - instant reset
                pattern.tier = SizeTier.BASE
                pattern.tier_reps = 0
                logger.warning(f"[RISK-EVO] {pattern_key}: TIER-3 → BASE (loss)")
            
            elif pattern.tier == SizeTier.TIER_2:
                # TIER-2: 1 loss → TIER-1
                pattern.tier = SizeTier.TIER_1
                logger.info(f"[RISK-EVO] {pattern_key}: TIER-2 → TIER-1 (loss)")
            
            elif pattern.tier == SizeTier.TIER_1:
                # TIER-1: 2 consecutive losses → BASE
                if pattern.consecutive_losses >= 2:
                    pattern.tier = SizeTier.BASE
                    pattern.tier_reps = 0
                    logger.info(f"[RISK-EVO] {pattern_key}: TIER-1 → BASE (2 losses)")
    
    def check_mtd_drawdown(self, mtd_pnl_pct: float):
        """
        Apply monthly drawdown controls.
        
        Actions:
        - -2% MTD: Reduce all tiers by 1
        - -3% MTD: All patterns back to BASE
        - -5% MTD: Freeze trading for 48h
        """
        if mtd_pnl_pct <= -0.05:
            # 5% drawdown → 48h freeze
            self.frozen_until = time.time() + (48 * 3600)
            self._reset_all_tiers()
            logger.critical(f"[RISK-EVO] 5% MTD drawdown - 48h trading freeze!")
        
        elif mtd_pnl_pct <= -0.03:
            # 3% drawdown → all to BASE
            self._reset_all_tiers()
            logger.warning(f"[RISK-EVO] 3% MTD drawdown - all patterns to BASE")
        
        elif mtd_pnl_pct <= -0.02:
            # 2% drawdown → reduce by 1 tier
            self._reduce_all_tiers()
            logger.warning(f"[RISK-EVO] 2% MTD drawdown - all tiers reduced by 1")
    
    def _reset_all_tiers(self):
        """Reset all patterns to BASE tier."""
        for pattern in self.patterns.values():
            pattern.tier = SizeTier.BASE
            pattern.tier_reps = 0
    
    def _reduce_all_tiers(self):
        """Reduce all patterns by 1 tier."""
        tier_order = [SizeTier.BASE, SizeTier.TIER_1, SizeTier.TIER_2, SizeTier.TIER_3]
        for pattern in self.patterns.values():
            current_idx = tier_order.index(pattern.tier)
            if current_idx > 0:
                pattern.tier = tier_order[current_idx - 1]
    
    def reset_month(self):
        """Reset monthly stats (call at month start)."""
        self.mtd_pnl = 0.0
        self.mtd_trades = 0
    
    def get_stats(self) -> Dict:
        """Get ladder statistics."""
        tier_counts = {tier.value: 0 for tier in SizeTier}
        for pattern in self.patterns.values():
            tier_counts[pattern.tier.value] += 1
        
        return {
            "total_patterns": len(self.patterns),
            "tier_distribution": tier_counts,
            "mtd_pnl": self.mtd_pnl,
            "mtd_trades": self.mtd_trades,
            "frozen": time.time() < self.frozen_until,
            "phase": self.risk_calc.phase,
        }


# ============================================================================
# SINGLETON
# ============================================================================

_ladder_instance: Optional[SizeLadderManager] = None


def get_size_ladder(phase: str = "MONTH_2_3") -> SizeLadderManager:
    """Get or create the size ladder singleton."""
    global _ladder_instance
    if _ladder_instance is None:
        _ladder_instance = SizeLadderManager(phase)
    return _ladder_instance


def reset_size_ladder():
    """Reset the singleton (for testing)."""
    global _ladder_instance
    _ladder_instance = None
