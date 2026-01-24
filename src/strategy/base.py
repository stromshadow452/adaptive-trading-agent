"""
WEAPON SYSTEM: Strategy Interface & Base Classes
=================================================

Core components:
- Strategy interface (abstract base)
- WeaponClass enum (SCALPEL/RIFLE/SNIPER/SHIELD)
- Data classes (MarketSnapshot, AgentContext, StrategySignal)

Philosophy: "Agent is the commander. Strategies are tools."
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS
# ============================================================================

class Signal(Enum):
    """Trading signal direction."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class WeaponClass(Enum):
    """Strategy weapon classification."""
    SCALPEL = "SCALPEL"   # Micro trades, 10-15% size
    RIFLE = "RIFLE"       # Standard trades, 100% size
    SNIPER = "SNIPER"     # A-grade only, DISABLED
    SHIELD = "SHIELD"     # Defensive, no trade


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class MarketSnapshot:
    """
    Immutable market state at decision time.
    Passed to strategies for evaluation.
    """
    timestamp: datetime
    symbol: str
    price: float
    atr: float
    atr_avg: float
    volume_ratio: float = 1.0
    spread: float = 0.0
    
    # Recent price levels (for micro strategies)
    high_20: float = 0.0    # 20-bar high
    low_20: float = 0.0     # 20-bar low
    sma_20: float = 0.0     # 20-bar SMA
    bb_upper: float = 0.0   # Bollinger upper
    bb_lower: float = 0.0   # Bollinger lower


@dataclass
class AgentContext:
    """
    Agent-level context passed to strategies.
    Contains all decision-making state.
    """
    # ML Brain
    ml_confidence: float      # 0.0-1.0
    ml_prediction: str        # BUY/SELL/HOLD
    
    # EDGE_SCORE (MARK-3)
    edge_score: float         # 0.0-1.0
    edge_tier: str            # C/B/A
    
    # Regime
    regime: str               # RANGE/TREND/DANGER
    regime_strength: float    # 0.0-1.0
    
    # Session
    session: str              # TOKYO/LONDON/NEW_YORK/SYDNEY/OFF
    
    # MARK-2 Survival
    mark2_can_trade: bool
    mark2_cooldown_min: float = 0.0
    memory_mod: float = 1.0
    ego_mod: float = 1.0
    memory_pain_level: float = 0.0  # 0.0-1.0
    
    # Risk
    base_size: float = 0.0
    rr_ratio: float = 1.5


@dataclass
class StrategySignal:
    """
    Output from a strategy evaluation.
    This is a SUGGESTION only - Router decides final action.
    """
    signal: Signal
    confidence: float         # 0.0-1.0
    suggested_sl: float       # Price level
    suggested_tp: float       # Price level
    reason: str               # Human-readable
    strategy_name: str = ""   # Filled by strategy
    weapon_class: str = ""    # Filled by strategy
    size_multiplier: float = 1.0  # Strategy-specific size


@dataclass
class StrategyStats:
    """Per-strategy performance tracking."""
    name: str
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    total_r: float = 0.0
    consecutive_losses: int = 0
    best_session: str = ""
    best_regime: str = ""
    disabled_until: Optional[datetime] = None
    
    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.wins / self.total_trades
    
    @property
    def avg_r(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.total_r / self.total_trades
    
    @property
    def is_disabled(self) -> bool:
        if self.disabled_until is None:
            return False
        return datetime.utcnow() < self.disabled_until


# ============================================================================
# STRATEGY INTERFACE
# ============================================================================

class Strategy(ABC):
    """
    Base strategy interface.
    
    RULES:
    1. Strategies are STATELESS at execution time
    2. Strategies NEVER trade by themselves
    3. Strategies only SUGGEST signals
    4. Router decides final action
    5. MARK-2 has absolute veto power
    """
    
    def __init__(self):
        self._enabled = True
        self.stats = StrategyStats(name=self.name)
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique strategy identifier."""
        pass
    
    @property
    @abstractmethod
    def weapon_class(self) -> WeaponClass:
        """Which weapon class this strategy belongs to."""
        pass
    
    @property
    def enabled(self) -> bool:
        """Whether strategy is currently enabled."""
        if self.stats.is_disabled:
            return False
        return self._enabled
    
    @enabled.setter
    def enabled(self, value: bool):
        self._enabled = value
    
    @abstractmethod
    def evaluate(
        self,
        snapshot: MarketSnapshot,
        context: AgentContext,
    ) -> Optional[StrategySignal]:
        """
        Evaluate market conditions and return signal suggestion.
        
        Returns None if strategy has no opinion.
        Returns StrategySignal if strategy suggests action.
        
        NOTE: This is a SUGGESTION only. Router decides final action.
        """
        pass
    
    @abstractmethod
    def get_size_multiplier(self) -> float:
        """Return size multiplier for this strategy (0.1-1.0)."""
        pass
    
    def is_allowed(self, context: AgentContext) -> bool:
        """
        Check if strategy is allowed to trade in current context.
        Override in subclasses for specific conditions.
        """
        # Base checks
        if not self.enabled:
            return False
        if not context.mark2_can_trade:
            return False
        if context.regime == 'DANGER':
            return False
        if context.session == 'OFF':
            return False
        return True
    
    def record_trade(self, r_multiple: float, session: str, regime: str):
        """Record trade outcome for stats tracking."""
        self.stats.total_trades += 1
        self.stats.total_r += r_multiple
        
        if r_multiple > 0:
            self.stats.wins += 1
            self.stats.consecutive_losses = 0
        else:
            self.stats.losses += 1
            self.stats.consecutive_losses += 1
        
        # Update best session/regime
        # (Simplified - real implementation would track per-session stats)
        logger.info(
            f"[STRATEGY] {self.name}: Trade recorded R={r_multiple:.2f}, "
            f"WR={self.stats.win_rate:.1%}, AvgR={self.stats.avg_r:.2f}"
        )


# ============================================================================
# SHIELD STRATEGY (Defensive)
# ============================================================================

class ShieldStrategy(Strategy):
    """
    Defensive strategy - always returns HOLD.
    Used when conditions are unfavorable.
    """
    
    @property
    def name(self) -> str:
        return "shield"
    
    @property
    def weapon_class(self) -> WeaponClass:
        return WeaponClass.SHIELD
    
    def evaluate(
        self,
        snapshot: MarketSnapshot,
        context: AgentContext,
    ) -> StrategySignal:
        return StrategySignal(
            signal=Signal.HOLD,
            confidence=1.0,
            suggested_sl=0,
            suggested_tp=0,
            reason="Shield active - no trade",
            strategy_name=self.name,
            weapon_class=self.weapon_class.value,
            size_multiplier=0.0
        )
    
    def get_size_multiplier(self) -> float:
        return 0.0
