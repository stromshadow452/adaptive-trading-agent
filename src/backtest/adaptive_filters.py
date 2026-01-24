"""
Adaptive Filters V1 - Data-Aware Threshold Adjustment

Provides LEARNING vs CONFIRMATION mode switching and
adaptive threshold computation based on data sufficiency.

Key Features:
    - Mode detection based on available data bars
    - Threshold relaxation for sparse data
    - Exploration budget management
    - Threshold lookup by stage

Author: SCOPUS Adaptive Trading System
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ModeConfig:
    """Configuration for each operating mode."""
    # Confidence thresholds
    ml_buy_threshold: float
    ml_sell_threshold: float
    meta_primary_threshold: float
    meta_finrl_threshold: float
    
    # Risk adjustments
    max_position_multiplier: float  # 1.0 = normal, 0.3 = 30% of normal
    danger_regime_action: str       # "block" or "reduce"
    danger_size_reduction: float    # If action = "reduce", reduce by this factor
    
    # Exploration
    exploration_trades_per_day: int
    exploration_size_cap: float     # Max size for exploration trades


# Predefined mode configurations
LEARNING_CONFIG = ModeConfig(
    ml_buy_threshold=0.35,
    ml_sell_threshold=0.35,
    meta_primary_threshold=0.30,
    meta_finrl_threshold=0.50,
    max_position_multiplier=0.30,
    danger_regime_action="reduce",
    danger_size_reduction=0.50,
    exploration_trades_per_day=2,
    exploration_size_cap=0.30
)

CONFIRMATION_CONFIG = ModeConfig(
    ml_buy_threshold=0.60,
    ml_sell_threshold=0.60,
    meta_primary_threshold=0.50,
    meta_finrl_threshold=0.70,
    max_position_multiplier=1.0,
    danger_regime_action="block",
    danger_size_reduction=0.0,
    exploration_trades_per_day=0,
    exploration_size_cap=0.0
)


# =============================================================================
# Adaptive Filter Engine
# =============================================================================

class AdaptiveFilterEngine:
    """
    Manages adaptive thresholds based on data sufficiency.
    
    Features:
        - Mode switching (LEARNING ↔ CONFIRMATION)
        - Threshold interpolation
        - Exploration budget tracking
    """
    
    # Data bar thresholds for mode switching (M5 timeframe)
    BARS_MIN = 20               # Absolute minimum
    BARS_LEARNING_FULL = 100    # Full LEARNING capacity
    BARS_TRANSITION = 300       # Start transitioning to CONFIRMATION
    BARS_CONFIRMATION = 500     # Full CONFIRMATION mode
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # Override defaults from config
        self.bars_min = self.config.get('bars_min', self.BARS_MIN)
        self.bars_confirmation = self.config.get('bars_confirmation', self.BARS_CONFIRMATION)
        
        # State tracking
        self.current_mode: str = "LEARNING"
        self.data_sufficiency: float = 0.0
        self.exploration_trades_today: int = 0
        self.exploration_count: int = 0  # Total exploration trades counter
        self.current_date: Optional[str] = None
        
        logger.info("AdaptiveFilterEngine initialized")
    
    def compute_mode(self, n_bars: int) -> tuple:
        """
        Determine operating mode and data sufficiency from bar count.
        
        Args:
            n_bars: Number of available data bars
            
        Returns:
            (mode, data_sufficiency, mode_config)
        """
        if n_bars < self.bars_min:
            self.current_mode = "LEARNING"
            self.data_sufficiency = 0.0
            return self.current_mode, 0.0, LEARNING_CONFIG
        
        if n_bars >= self.BARS_CONFIRMATION:
            self.current_mode = "CONFIRMATION"
            self.data_sufficiency = 1.0
            return self.current_mode, 1.0, CONFIRMATION_CONFIG
        
        # Logarithmic interpolation
        ratio = n_bars / self.BARS_LEARNING_FULL
        sufficiency = min(1.0, math.log(ratio + 1) / math.log(6))
        self.data_sufficiency = sufficiency
        
        # Mode transition
        if n_bars >= self.BARS_TRANSITION:
            self.current_mode = "CONFIRMATION"
            return self.current_mode, sufficiency, self._interpolate_config(sufficiency)
        else:
            self.current_mode = "LEARNING"
            return self.current_mode, sufficiency, self._interpolate_config(sufficiency)
    
    def _interpolate_config(self, sufficiency: float) -> ModeConfig:
        """Interpolate between LEARNING and CONFIRMATION configs."""
        def lerp(a, b, t):
            return a + (b - a) * t
        
        return ModeConfig(
            ml_buy_threshold=lerp(
                LEARNING_CONFIG.ml_buy_threshold,
                CONFIRMATION_CONFIG.ml_buy_threshold,
                sufficiency
            ),
            ml_sell_threshold=lerp(
                LEARNING_CONFIG.ml_sell_threshold,
                CONFIRMATION_CONFIG.ml_sell_threshold,
                sufficiency
            ),
            meta_primary_threshold=lerp(
                LEARNING_CONFIG.meta_primary_threshold,
                CONFIRMATION_CONFIG.meta_primary_threshold,
                sufficiency
            ),
            meta_finrl_threshold=lerp(
                LEARNING_CONFIG.meta_finrl_threshold,
                CONFIRMATION_CONFIG.meta_finrl_threshold,
                sufficiency
            ),
            max_position_multiplier=lerp(
                LEARNING_CONFIG.max_position_multiplier,
                CONFIRMATION_CONFIG.max_position_multiplier,
                sufficiency
            ),
            danger_regime_action=LEARNING_CONFIG.danger_regime_action if sufficiency < 0.8 else CONFIRMATION_CONFIG.danger_regime_action,
            danger_size_reduction=lerp(
                LEARNING_CONFIG.danger_size_reduction,
                CONFIRMATION_CONFIG.danger_size_reduction,
                sufficiency
            ),
            exploration_trades_per_day=round(lerp(
                LEARNING_CONFIG.exploration_trades_per_day,
                CONFIRMATION_CONFIG.exploration_trades_per_day,
                sufficiency
            )),
            exploration_size_cap=lerp(
                LEARNING_CONFIG.exploration_size_cap,
                CONFIRMATION_CONFIG.exploration_size_cap,
                sufficiency
            )
        )
    
    def adapt_threshold(self, base_threshold: float, n_bars: int) -> float:
        """
        Compute adaptive threshold based on data availability.
        
        At sparse data: threshold = base * 0.5 (50% reduction)
        At full data: threshold = base (full strictness)
        
        Args:
            base_threshold: The threshold for full data
            n_bars: Available data bars
            
        Returns:
            Adjusted threshold
        """
        if n_bars < self.bars_min:
            return base_threshold * 0.4  # Very relaxed
        
        sufficiency = min(1.0, math.log(n_bars / self.bars_min + 1) / math.log(6))
        return base_threshold * (0.5 + 0.5 * sufficiency)
    
    def get_exploration_budget(self) -> float:
        """Get current exploration budget as position size multiplier."""
        if self.current_mode == "CONFIRMATION":
            return 0.0
        
        # More budget when data is sparse
        budget = LEARNING_CONFIG.exploration_size_cap * (1.0 - self.data_sufficiency * 0.5)
        return max(0.05, min(LEARNING_CONFIG.exploration_size_cap, budget))
    
    def can_explore(self, mode_config: Optional[ModeConfig] = None) -> bool:
        """Check if exploration trade is allowed."""
        cfg = mode_config or (LEARNING_CONFIG if self.current_mode == "LEARNING" else CONFIRMATION_CONFIG)
        return self.exploration_trades_today < cfg.exploration_trades_per_day
    
    def record_exploration(self):
        """Record an exploration trade."""
        self.exploration_trades_today += 1
        self.exploration_count += 1
        logger.info(f"[EXPLORATION] Recorded trade #{self.exploration_count} (today: {self.exploration_trades_today})")
    
    def reset_daily(self, date: str):
        """Reset daily counters if date changed."""
        if date != self.current_date:
            self.current_date = date
            self.exploration_trades_today = 0
            logger.debug(f"Daily reset for {date}")
    
    def should_block_danger(self, mode_config: Optional[ModeConfig] = None) -> bool:
        """Check if DANGER regime should block trades."""
        cfg = mode_config or (LEARNING_CONFIG if self.current_mode == "LEARNING" else CONFIRMATION_CONFIG)
        return cfg.danger_regime_action == "block"
    
    def get_danger_size_reduction(self, mode_config: Optional[ModeConfig] = None) -> float:
        """Get size reduction factor for DANGER regime."""
        cfg = mode_config or (LEARNING_CONFIG if self.current_mode == "LEARNING" else CONFIRMATION_CONFIG)
        return cfg.danger_size_reduction


# =============================================================================
# Threshold Lookup Functions (Convenience)
# =============================================================================

def get_ml_threshold(mode: str, signal_type: str = "buy") -> float:
    """Get ML Brain threshold for given mode and signal type."""
    cfg = LEARNING_CONFIG if mode == "LEARNING" else CONFIRMATION_CONFIG
    return cfg.ml_buy_threshold if signal_type == "buy" else cfg.ml_sell_threshold


def get_meta_thresholds(mode: str) -> tuple:
    """Get meta-gating thresholds (primary, finrl)."""
    cfg = LEARNING_CONFIG if mode == "LEARNING" else CONFIRMATION_CONFIG
    return cfg.meta_primary_threshold, cfg.meta_finrl_threshold


def get_max_position_mult(mode: str) -> float:
    """Get maximum position size multiplier."""
    cfg = LEARNING_CONFIG if mode == "LEARNING" else CONFIRMATION_CONFIG
    return cfg.max_position_multiplier


# =============================================================================
# Singleton Instance (Optional)
# =============================================================================

_default_engine: Optional[AdaptiveFilterEngine] = None

def get_adaptive_filter_engine(config: Optional[Dict] = None) -> AdaptiveFilterEngine:
    """Get or create the default AdaptiveFilterEngine instance."""
    global _default_engine
    if _default_engine is None:
        _default_engine = AdaptiveFilterEngine(config)
    return _default_engine


def reset_adaptive_filter_engine():
    """Reset the singleton instance (for testing)."""
    global _default_engine
    _default_engine = None
