"""
WEAPON SYSTEM: Strategy Router
===============================

Central decision point for strategy selection.
Implements the IF/ELSE decision tree.

Decision Flow:
1. SURVIVAL VETO (MARK-2)
2. DANGER REGIME check
3. OFF SESSION check
4. TRY PRIMARY (RIFLE)
5. TRY SCALPEL (MICRO)
6. NO TRADE

Philosophy: "Agent is the commander. Strategies are tools."
"""

from typing import Optional, Tuple, List, Dict, Any
from datetime import datetime, timedelta
import logging

from src.strategy.base import (
    Strategy, WeaponClass, Signal,
    MarketSnapshot, AgentContext, StrategySignal,
    ShieldStrategy, StrategyStats
)
from src.strategy.scalpel import RangeMicroMR, LiquiditySweepFade
from src.strategy.rifle import MLPrimaryStrategy
from src.strategy.rsi_range_reversion import RSIRangeReversion

logger = logging.getLogger(__name__)


# ============================================================================
# STRATEGY MANAGER
# ============================================================================

class StrategyManager:
    """
    Manages strategy lifecycle and retirement.
    Tracks per-strategy stats and handles auto-disable.
    """
    
    RETIREMENT_THRESHOLDS = {
        'min_trades': 20,           # Min trades before evaluation
        'min_win_rate': 0.40,       # Below this = retire
        'min_avg_r': 0.05,          # Below this = retire
        'max_consecutive_loss': 6,  # Auto-disable after 6 losses
        'cooldown_hours': 48,       # 48h cooldown after retire
    }
    
    def __init__(self):
        self.stats: Dict[str, StrategyStats] = {}
    
    def register_strategy(self, strategy: Strategy):
        """Register a strategy for tracking."""
        self.stats[strategy.name] = strategy.stats
    
    def record_trade(
        self,
        strategy_name: str,
        r_multiple: float,
        session: str,
        regime: str
    ):
        """Record trade outcome for strategy."""
        if strategy_name not in self.stats:
            return
        
        stats = self.stats[strategy_name]
        stats.total_trades += 1
        stats.total_r += r_multiple
        
        if r_multiple > 0:
            stats.wins += 1
            stats.consecutive_losses = 0
        else:
            stats.losses += 1
            stats.consecutive_losses += 1
        
        # Check retirement conditions
        self._check_retirement(strategy_name)
    
    def _check_retirement(self, strategy_name: str):
        """Check if strategy should be retired."""
        stats = self.stats[strategy_name]
        thresholds = self.RETIREMENT_THRESHOLDS
        
        # Not enough trades
        if stats.total_trades < thresholds['min_trades']:
            return
        
        # Consecutive loss limit
        if stats.consecutive_losses >= thresholds['max_consecutive_loss']:
            self._retire_strategy(strategy_name, "Consecutive loss limit")
            return
        
        # Poor win rate
        if stats.win_rate < thresholds['min_win_rate']:
            self._retire_strategy(strategy_name, "Low win rate")
            return
        
        # Negative expectancy
        if stats.avg_r < thresholds['min_avg_r']:
            self._retire_strategy(strategy_name, "Low expectancy")
            return
    
    def _retire_strategy(self, name: str, reason: str):
        """Disable strategy temporarily."""
        stats = self.stats[name]
        cooldown = timedelta(hours=self.RETIREMENT_THRESHOLDS['cooldown_hours'])
        stats.disabled_until = datetime.utcnow() + cooldown
        logger.warning(
            f"[STRATEGY] {name} RETIRED: {reason}, "
            f"cooldown until {stats.disabled_until}"
        )


# ============================================================================
# STRATEGY ROUTER
# ============================================================================

class StrategyRouter:
    """
    Central decision point for strategy selection.
    
    RULES:
    1. Can return NO TRADE at any point
    2. MARK-2 survival has final veto
    3. Primary strategy (RIFLE) has priority
    4. Micro strategies only in specific conditions
    5. Never override safety constraints
    """
    
    def __init__(self, enable_micro: bool = True):
        """
        Initialize router with strategies.
        
        Args:
            enable_micro: Whether to enable micro (scalpel) strategies
        """
        self.enable_micro = enable_micro
        
        # Initialize strategies
        self.shield = ShieldStrategy()
        self.primary = MLPrimaryStrategy()
        self.scalpels: List[Strategy] = [
            RSIRangeReversion(),   # Check first in RANGE (locked RSI weapon)
            RangeMicroMR(),
            LiquiditySweepFade(),
        ]
        
        # Strategy manager for tracking
        self.manager = StrategyManager()
        self._register_strategies()
        
        # Stats
        self.routing_stats = {
            'total_calls': 0,
            'no_trade': 0,
            'rifle_trades': 0,
            'scalpel_trades': 0,
            'rsi_trades': 0,
            'blocked_by_mark2': 0,
            'blocked_by_danger': 0,
            'blocked_by_session': 0,
        }
        
        logger.info(
            f"[ROUTER] Initialized with: "
            f"primary={self.primary.name}, "
            f"scalpels={[s.name for s in self.scalpels]}, "
            f"enable_micro={enable_micro}"
        )
    
    def _register_strategies(self):
        """Register all strategies with manager."""
        self.manager.register_strategy(self.primary)
        for s in self.scalpels:
            self.manager.register_strategy(s)
    
    def route(
        self,
        snapshot: MarketSnapshot,
        context: AgentContext,
    ) -> Tuple[Optional[Strategy], Optional[StrategySignal]]:
        """
        Route to appropriate strategy.
        
        Decision Tree:
        1. SURVIVAL VETO (MARK-2)
        2. DANGER REGIME
        3. OFF SESSION
        4. TRY RIFLE (primary)
        5. TRY SCALPEL (micro)
        6. NO TRADE
        
        Returns:
            (Strategy, StrategySignal) if trade should happen
            (None, None) if NO TRADE
        """
        self.routing_stats['total_calls'] += 1
        
        # ===== STAGE 1: SURVIVAL VETO =====
        if not context.mark2_can_trade:
            self.routing_stats['blocked_by_mark2'] += 1
            self._log_decision("BLOCKED", "MARK-2 veto", context)
            return (None, None)
        
        if context.mark2_cooldown_min > 0:
            self.routing_stats['blocked_by_mark2'] += 1
            self._log_decision("BLOCKED", f"MARK-2 cooldown ({context.mark2_cooldown_min:.1f}m)", context)
            return (None, None)
        
        # ===== STAGE 2: DANGER REGIME =====
        if context.regime == 'DANGER':
            self.routing_stats['blocked_by_danger'] += 1
            self._log_decision("BLOCKED", "DANGER regime", context)
            return (None, None)
        
        # ===== STAGE 3: OFF SESSION =====
        if context.session == 'OFF':
            self.routing_stats['blocked_by_session'] += 1
            self._log_decision("BLOCKED", "OFF session", context)
            return (None, None)
        
        # ===== STAGE 4: TRY PRIMARY (RIFLE) =====
        if self._should_use_rifle(context):
            signal = self.primary.evaluate(snapshot, context)
            if signal and signal.signal != Signal.HOLD:
                self.routing_stats['rifle_trades'] += 1
                self._log_decision("RIFLE", signal.reason, context)
                return (self.primary, signal)
        
        # ===== STAGE 5: TRY SCALPEL (MICRO) =====
        if self.enable_micro and self._should_try_scalpel(context):
            for scalpel in self.scalpels:
                if not scalpel.enabled:
                    continue
                signal = scalpel.evaluate(snapshot, context)
                if signal and signal.signal != Signal.HOLD:
                    # Track RSI trades separately
                    if scalpel.name == 'rsi_range_reversion':
                        self.routing_stats['rsi_trades'] += 1
                        self._log_decision("RSI-SCALPEL", signal.reason, context)
                    else:
                        self.routing_stats['scalpel_trades'] += 1
                        self._log_decision("SCALPEL", signal.reason, context)
                    return (scalpel, signal)
        
        # ===== STAGE 6: NO TRADE =====
        self.routing_stats['no_trade'] += 1
        return (None, None)
    
    def _should_use_rifle(self, context: AgentContext) -> bool:
        """Check if conditions favor primary strategy."""
        return (
            context.ml_confidence >= 0.60 and
            context.edge_tier in ('A', 'B') and
            context.regime != 'DANGER' and
            context.ml_prediction != 'HOLD'
        )
    
    def _should_try_scalpel(self, context: AgentContext) -> bool:
        """Check if conditions allow micro strategies."""
        return (
            context.ml_confidence >= 0.45 and
            context.ml_confidence <= 0.70 and
            context.edge_tier in ('B', 'C') and
            context.regime in ('RANGE', 'TREND') and
            context.regime_strength < 0.75 and
            context.memory_pain_level < 0.5
        )
    
    def _log_decision(self, decision: str, reason: str, context: AgentContext):
        """Log routing decision."""
        logger.info(
            f"[ROUTER] {decision} | {reason} | "
            f"conf={context.ml_confidence:.2f}, EDGE={context.edge_tier}, "
            f"regime={context.regime}, session={context.session}"
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get routing statistics."""
        total = max(1, self.routing_stats['total_calls'])
        return {
            **self.routing_stats,
            'rifle_rate': self.routing_stats['rifle_trades'] / total,
            'scalpel_rate': self.routing_stats['scalpel_trades'] / total,
            'no_trade_rate': self.routing_stats['no_trade'] / total,
        }
    
    def print_stats(self):
        """Print routing statistics."""
        stats = self.get_stats()
        print("\n" + "=" * 50)
        print("STRATEGY ROUTER STATS")
        print("=" * 50)
        print(f"Total Calls: {stats['total_calls']}")
        print(f"Rifle Trades: {stats['rifle_trades']} ({stats['rifle_rate']*100:.1f}%)")
        print(f"Scalpel Trades: {stats['scalpel_trades']} ({stats['scalpel_rate']*100:.1f}%)")
        print(f"No Trade: {stats['no_trade']} ({stats['no_trade_rate']*100:.1f}%)")
        print(f"\nBlocked by MARK-2: {stats['blocked_by_mark2']}")
        print(f"Blocked by DANGER: {stats['blocked_by_danger']}")
        print(f"Blocked by Session: {stats['blocked_by_session']}")
        print("=" * 50 + "\n")
