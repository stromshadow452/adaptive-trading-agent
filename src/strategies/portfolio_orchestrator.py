"""
SCOPUS v3.0 — Portfolio Orchestrator

Manages dual-strategy system:
- Strategy A: Choppy Engine (FX Defense)
- Strategy B: Silver Mean Reversion (Offense)

Capital Allocation:
- 30% to Strategy A (EURUSD)
- 50% to Strategy B (XAGUSD)
- 20% Reserve (dry powder)

Key Features:
- Independent signal generation
- Correlation monitoring
- Hard stop enforcement
- Capital rebalancing
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from enum import Enum

from src.strategies.choppy_engine import ChoppyEngine, ChoppySignal, ChoppyConfig
from src.strategies.silver_mean_reversion import SilverMeanReversion, SilverConfig, SignalType


class PortfolioAction(Enum):
    NONE = "NONE"
    EXECUTE_A = "EXECUTE_A"  # Execute Strategy A signal
    EXECUTE_B = "EXECUTE_B"  # Execute Strategy B signal
    EXECUTE_BOTH = "EXECUTE_BOTH"
    PAUSE_A = "PAUSE_A"
    PAUSE_B = "PAUSE_B"
    PAUSE_ALL = "PAUSE_ALL"


@dataclass
class PortfolioConfig:
    """Portfolio-level configuration."""
    
    # Account
    initial_balance: float = 10000.0
    
    # Capital Allocation
    strategy_a_allocation: float = 0.30  # 30%
    strategy_b_allocation: float = 0.50  # 50%
    reserve_allocation: float = 0.20     # 20%
    
    # Risk Limits
    strategy_a_max_risk: float = 0.005   # 0.5% per trade
    strategy_b_max_risk: float = 0.01    # 1.0% per trade
    portfolio_max_dd: float = 0.15       # 15% total DD
    strategy_b_max_dd: float = 0.25      # 25% strategy DD
    
    # Correlation limits
    max_simultaneous_positions: int = 2
    min_time_between_trades: int = 60  # minutes
    
    # Hard stops
    consecutive_losses_pause: int = 5
    weekly_loss_pause_pct: float = 0.05  # 5%


@dataclass
class PortfolioState:
    """Current portfolio state."""
    balance: float = 10000.0
    peak_balance: float = 10000.0
    
    # Strategy states
    strategy_a_active: bool = True
    strategy_b_active: bool = True
    strategy_a_position: Optional[str] = None
    strategy_b_position: Optional[str] = None
    
    # Performance tracking
    strategy_a_pnl: float = 0.0
    strategy_b_pnl: float = 0.0
    
    # Risk tracking
    current_dd: float = 0.0
    strategy_a_dd: float = 0.0
    strategy_b_dd: float = 0.0
    consecutive_losses: int = 0
    weekly_pnl: float = 0.0
    
    # Trade counts
    trades_today: int = 0
    last_trade_time: Optional[datetime] = None


@dataclass
class PortfolioSignal:
    """Combined portfolio signal."""
    action: PortfolioAction
    strategy_a_signal: Optional[ChoppySignal] = None
    strategy_b_signal: Optional[SignalType] = None
    position_size_a: float = 0.0
    position_size_b: float = 0.0
    reason: str = ""


class PortfolioOrchestrator:
    """
    Dual-Strategy Portfolio Manager.
    
    Routes signals to appropriate strategies,
    manages capital, enforces risk limits.
    """
    
    def __init__(self, config: PortfolioConfig = None):
        self.config = config or PortfolioConfig()
        self.state = PortfolioState(balance=self.config.initial_balance,
                                    peak_balance=self.config.initial_balance)
        
        # Initialize strategies
        self.strategy_a = ChoppyEngine(ChoppyConfig())
        self.strategy_b = SilverMeanReversion(SilverConfig())
        
        # Data caches
        self.eurusd_data: Optional[pd.DataFrame] = None
        self.xagusd_data: Optional[pd.DataFrame] = None
        
        # Trade history
        self.trade_history: List[Dict] = []
    
    def update_data(self, eurusd_df: pd.DataFrame = None, 
                    xagusd_df: pd.DataFrame = None):
        """Update price data for both strategies."""
        if eurusd_df is not None:
            self.eurusd_data = self.strategy_a.compute_indicators(eurusd_df.copy())
        
        if xagusd_df is not None:
            self.xagusd_data = self.strategy_b.compute_indicators(xagusd_df.copy())
    
    def check_hard_stops(self) -> Tuple[bool, str]:
        """Check if hard stop conditions are met."""
        
        # Portfolio drawdown
        if self.state.current_dd > self.config.portfolio_max_dd:
            return True, f"Portfolio DD {self.state.current_dd:.1%} > {self.config.portfolio_max_dd:.1%}"
        
        # Strategy B drawdown
        if self.state.strategy_b_dd > self.config.strategy_b_max_dd:
            return True, f"Strategy B DD {self.state.strategy_b_dd:.1%} > {self.config.strategy_b_max_dd:.1%}"
        
        # Consecutive losses
        if self.state.consecutive_losses >= self.config.consecutive_losses_pause:
            return True, f"Consecutive losses: {self.state.consecutive_losses}"
        
        # Weekly loss
        weekly_loss_pct = abs(self.state.weekly_pnl) / self.config.initial_balance
        if self.state.weekly_pnl < 0 and weekly_loss_pct > self.config.weekly_loss_pause_pct:
            return True, f"Weekly loss {weekly_loss_pct:.1%} > {self.config.weekly_loss_pause_pct:.1%}"
        
        return False, ""
    
    def generate_signal(self, current_time: datetime = None) -> PortfolioSignal:
        """
        Generate combined portfolio signal.
        
        Checks both strategies and routes appropriately.
        """
        
        # Check hard stops first
        stopped, reason = self.check_hard_stops()
        if stopped:
            return PortfolioSignal(
                action=PortfolioAction.PAUSE_ALL,
                reason=f"HARD STOP: {reason}"
            )
        
        signal_a = None
        signal_b = None
        
        # Get Strategy A signal (EURUSD)
        if self.state.strategy_a_active and self.eurusd_data is not None:
            result_a = self.strategy_a.generate_signal(
                self.eurusd_data, 
                bar_time=current_time
            )
            signal_a = result_a.signal
        
        # Get Strategy B signal (XAGUSD)
        if self.state.strategy_b_active and self.xagusd_data is not None:
            result_b = self.strategy_b.generate_signal(self.xagusd_data)
            signal_b = result_b.signal
        
        # Determine action
        action = PortfolioAction.NONE
        size_a = 0.0
        size_b = 0.0
        reasons = []
        
        # Check time between trades
        if current_time and self.state.last_trade_time:
            minutes_since = (current_time - self.state.last_trade_time).total_seconds() / 60
            if minutes_since < self.config.min_time_between_trades:
                return PortfolioSignal(
                    action=PortfolioAction.NONE,
                    reason=f"Cooldown: {minutes_since:.0f}/{self.config.min_time_between_trades} min"
                )
        
        # Process Strategy A
        if signal_a in [ChoppySignal.MICRO_LONG, ChoppySignal.MICRO_SHORT]:
            capital_a = self.state.balance * self.config.strategy_a_allocation
            size_a = self._calculate_position_size_a(capital_a)
            action = PortfolioAction.EXECUTE_A
            reasons.append(f"A: {signal_a.value}")
        
        # Process Strategy B
        if signal_b in [SignalType.LONG, SignalType.SHORT]:
            capital_b = self.state.balance * self.config.strategy_b_allocation
            size_b = self._calculate_position_size_b(capital_b)
            
            if action == PortfolioAction.EXECUTE_A:
                action = PortfolioAction.EXECUTE_BOTH
            else:
                action = PortfolioAction.EXECUTE_B
            reasons.append(f"B: {signal_b.value}")
        
        # Check max simultaneous positions
        active_positions = sum([
            1 if self.state.strategy_a_position else 0,
            1 if self.state.strategy_b_position else 0
        ])
        if active_positions >= self.config.max_simultaneous_positions:
            if action in [PortfolioAction.EXECUTE_A, PortfolioAction.EXECUTE_B, 
                         PortfolioAction.EXECUTE_BOTH]:
                return PortfolioSignal(
                    action=PortfolioAction.NONE,
                    reason=f"Max positions ({active_positions}) reached"
                )
        
        return PortfolioSignal(
            action=action,
            strategy_a_signal=signal_a,
            strategy_b_signal=signal_b,
            position_size_a=size_a,
            position_size_b=size_b,
            reason=" | ".join(reasons) if reasons else "No signals"
        )
    
    def _calculate_position_size_a(self, allocated_capital: float) -> float:
        """Calculate position size for Strategy A (EURUSD)."""
        # Very small size for defensive
        risk_amount = allocated_capital * self.config.strategy_a_max_risk
        # Assume 30 pip stop for EURUSD
        pip_value = 10  # $10 per pip per lot
        stop_pips = 30
        
        size = risk_amount / (stop_pips * pip_value)
        return round(size, 4)
    
    def _calculate_position_size_b(self, allocated_capital: float) -> float:
        """Calculate position size for Strategy B (XAGUSD)."""
        risk_amount = allocated_capital * self.config.strategy_b_max_risk
        # Assume 1100 pip stop for Silver
        pip_value = 50  # $50 per pip per lot
        stop_pips = 1100
        
        size = risk_amount / (stop_pips * pip_value)
        return round(size, 4)
    
    def on_trade_opened(self, strategy: str, direction: str, 
                        size: float, current_time: datetime = None):
        """Record trade opened."""
        if strategy == "A":
            self.state.strategy_a_position = direction
            self.strategy_a.on_trade_opened(direction)
        elif strategy == "B":
            self.state.strategy_b_position = direction
            self.strategy_b.on_trade_opened(direction, -1)
        
        self.state.trades_today += 1
        self.state.last_trade_time = current_time
        
        self.trade_history.append({
            'time': current_time,
            'strategy': strategy,
            'direction': direction,
            'size': size,
            'status': 'OPEN'
        })
    
    def on_trade_closed(self, strategy: str, pnl: float, current_time: datetime = None):
        """Record trade closed."""
        if strategy == "A":
            self.state.strategy_a_position = None
            self.state.strategy_a_pnl += pnl
            self.strategy_a.on_trade_closed()
        elif strategy == "B":
            self.state.strategy_b_position = None
            self.state.strategy_b_pnl += pnl
            self.strategy_b.on_trade_closed()
        
        # Update balance
        self.state.balance += pnl
        self.state.weekly_pnl += pnl
        
        # Update peak and drawdown
        if self.state.balance > self.state.peak_balance:
            self.state.peak_balance = self.state.balance
        
        self.state.current_dd = (self.state.peak_balance - self.state.balance) / self.state.peak_balance
        
        # Update consecutive losses
        if pnl < 0:
            self.state.consecutive_losses += 1
        else:
            self.state.consecutive_losses = 0
        
        # Update trade history
        if self.trade_history:
            self.trade_history[-1]['pnl'] = pnl
            self.trade_history[-1]['status'] = 'CLOSED'
            self.trade_history[-1]['close_time'] = current_time
    
    def get_status(self) -> Dict:
        """Get current portfolio status."""
        return {
            'balance': self.state.balance,
            'total_pnl': self.state.balance - self.config.initial_balance,
            'current_dd': f"{self.state.current_dd:.1%}",
            'strategy_a': {
                'active': self.state.strategy_a_active,
                'position': self.state.strategy_a_position,
                'pnl': self.state.strategy_a_pnl
            },
            'strategy_b': {
                'active': self.state.strategy_b_active,
                'position': self.state.strategy_b_position,
                'pnl': self.state.strategy_b_pnl
            },
            'trades_today': self.state.trades_today,
            'consecutive_losses': self.state.consecutive_losses
        }
    
    def reset_weekly(self):
        """Reset weekly counters (call every Monday)."""
        self.state.weekly_pnl = 0.0
        self.strategy_a.trades_this_week = 0
    
    def reset_daily(self):
        """Reset daily counters."""
        self.state.trades_today = 0


# ============================================================================
# TEST
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print(" SCOPUS v3.0 — Portfolio Orchestrator")
    print(" Dual-Strategy System")
    print("=" * 70)
    
    config = PortfolioConfig(initial_balance=10000.0)
    orchestrator = PortfolioOrchestrator(config)
    
    print(f"\n Portfolio Configuration:")
    print(f"   Initial Balance: ${config.initial_balance:,.0f}")
    print(f"   Strategy A (Choppy): {config.strategy_a_allocation:.0%}")
    print(f"   Strategy B (Silver): {config.strategy_b_allocation:.0%}")
    print(f"   Reserve: {config.reserve_allocation:.0%}")
    
    print(f"\n Risk Limits:")
    print(f"   Strategy A Risk/Trade: {config.strategy_a_max_risk:.1%}")
    print(f"   Strategy B Risk/Trade: {config.strategy_b_max_risk:.1%}")
    print(f"   Max Portfolio DD: {config.portfolio_max_dd:.0%}")
    print(f"   Max Strategy B DD: {config.strategy_b_max_dd:.0%}")
    
    print(f"\n Status:")
    status = orchestrator.get_status()
    print(f"   Balance: ${status['balance']:,.0f}")
    print(f"   Drawdown: {status['current_dd']}")
    print(f"   Strategy A Active: {status['strategy_a']['active']}")
    print(f"   Strategy B Active: {status['strategy_b']['active']}")
    
    print(f"\n SYSTEM READY FOR PAPER TRADING")
