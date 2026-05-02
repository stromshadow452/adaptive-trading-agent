"""
Portfolio State Manager

Tracks all positions, capital, exposure, and P/L across the portfolio.
Core state container for the Portfolio Shadow Engine.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
import numpy as np


class PositionSide(Enum):
    """Position direction."""
    LONG = 'LONG'
    SHORT = 'SHORT'


class PositionStatus(Enum):
    """Position lifecycle status."""
    OPEN = 'OPEN'
    CLOSED = 'CLOSED'


@dataclass
class Position:
    """Individual position in the portfolio."""
    symbol: str
    side: PositionSide
    entry_price: float
    entry_time: datetime
    size_pct: float  # Risk as % of capital
    sl_price: float
    tp_price: float
    brain_type: str  # 'TREND' or 'CHOPPY'
    
    # Updated during lifecycle
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    status: PositionStatus = PositionStatus.OPEN
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    realized_pnl: float = 0.0
    exit_reason: str = ""
    
    def update_price(self, price: float):
        """Update current price and unrealized P/L."""
        self.current_price = price
        
        if self.side == PositionSide.LONG:
            pnl_mult = (price - self.entry_price) / self.entry_price
        else:
            pnl_mult = (self.entry_price - price) / self.entry_price
        
        # P/L relative to risk taken
        sl_dist = abs(self.entry_price - self.sl_price)
        if sl_dist > 0:
            r_multiple = (price - self.entry_price) / sl_dist
            if self.side == PositionSide.SHORT:
                r_multiple = -r_multiple
            self.unrealized_pnl = self.size_pct * r_multiple
        else:
            self.unrealized_pnl = 0.0
    
    def close(self, exit_price: float, exit_time: datetime, reason: str):
        """Close the position."""
        self.status = PositionStatus.CLOSED
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.exit_reason = reason
        
        # Calculate realized P/L
        sl_dist = abs(self.entry_price - self.sl_price)
        if sl_dist > 0:
            r_multiple = (exit_price - self.entry_price) / sl_dist
            if self.side == PositionSide.SHORT:
                r_multiple = -r_multiple
            self.realized_pnl = self.size_pct * r_multiple
        else:
            self.realized_pnl = 0.0


@dataclass
class DailySnapshot:
    """Daily portfolio state snapshot."""
    date: datetime
    starting_capital: float
    ending_capital: float
    open_positions: int
    trades_opened: int
    trades_closed: int
    realized_pnl: float
    unrealized_pnl: float
    total_exposure: float
    max_drawdown: float
    danger_level: str
    protection_actions: List[str]


@dataclass
class PortfolioState:
    """
    Complete portfolio state container.
    
    Tracks all positions, capital, exposure, and performance metrics.
    """
    
    # Configuration
    initial_capital: float = 10000.0
    max_total_exposure: float = 0.03  # 3% max total risk
    max_per_asset_exposure: float = 0.015  # 1.5% max per asset
    max_open_positions: int = 3
    max_daily_trades: int = 5
    
    # Current state
    capital: float = field(default=10000.0)
    peak_capital: float = field(default=10000.0)
    
    # Positions
    open_positions: Dict[str, Position] = field(default_factory=dict)
    closed_positions: List[Position] = field(default_factory=list)
    
    # Daily tracking
    daily_snapshots: List[DailySnapshot] = field(default_factory=list)
    today_trades_opened: int = 0
    today_trades_closed: int = 0
    today_realized_pnl: float = 0.0
    
    # Performance
    max_drawdown: float = 0.0
    max_drawdown_date: Optional[datetime] = None
    
    # Danger state
    danger_level: str = 'SAFE'
    trading_halted: bool = False
    halt_until: Optional[datetime] = None
    
    def __post_init__(self):
        """Initialize capital from initial_capital."""
        self.capital = self.initial_capital
        self.peak_capital = self.initial_capital
    
    # =========================================================================
    # EXPOSURE CALCULATIONS
    # =========================================================================
    
    def get_total_exposure(self) -> float:
        """Get total portfolio risk exposure."""
        return sum(p.size_pct for p in self.open_positions.values())
    
    def get_asset_exposure(self, symbol: str) -> float:
        """Get exposure for a specific asset."""
        if symbol in self.open_positions:
            return self.open_positions[symbol].size_pct
        return 0.0
    
    def get_currency_exposure(self, currency: str) -> Dict[str, float]:
        """Get exposure by currency (e.g., JPY, USD)."""
        long_exposure = 0.0
        short_exposure = 0.0
        
        for pos in self.open_positions.values():
            if currency in pos.symbol:
                if pos.side == PositionSide.LONG:
                    # If currency is quote (e.g., EURJPY long = JPY short)
                    if pos.symbol.endswith(currency):
                        short_exposure += pos.size_pct
                    else:
                        long_exposure += pos.size_pct
                else:
                    if pos.symbol.endswith(currency):
                        long_exposure += pos.size_pct
                    else:
                        short_exposure += pos.size_pct
        
        return {'long': long_exposure, 'short': short_exposure}
    
    # =========================================================================
    # POSITION MANAGEMENT
    # =========================================================================
    
    def can_open_position(self, symbol: str, size_pct: float) -> tuple[bool, str]:
        """Check if a new position can be opened."""
        # Check if trading halted
        if self.trading_halted:
            return False, "Trading halted"
        
        # Check max positions
        if len(self.open_positions) >= self.max_open_positions:
            return False, f"Max positions ({self.max_open_positions}) reached"
        
        # Check if already have position in this asset
        if symbol in self.open_positions:
            return False, f"Already have position in {symbol}"
        
        # Check total exposure
        new_total = self.get_total_exposure() + size_pct
        if new_total > self.max_total_exposure:
            return False, f"Would exceed max exposure ({new_total:.1%} > {self.max_total_exposure:.1%})"
        
        # Check daily trade limit
        if self.today_trades_opened >= self.max_daily_trades:
            return False, f"Daily trade limit ({self.max_daily_trades}) reached"
        
        return True, "Approved"
    
    def open_position(self, position: Position):
        """Open a new position."""
        self.open_positions[position.symbol] = position
        self.today_trades_opened += 1
    
    def close_position(
        self, 
        symbol: str, 
        exit_price: float, 
        exit_time: datetime, 
        reason: str
    ) -> Optional[Position]:
        """Close an open position."""
        if symbol not in self.open_positions:
            return None
        
        position = self.open_positions.pop(symbol)
        position.close(exit_price, exit_time, reason)
        self.closed_positions.append(position)
        
        # Update capital
        self.capital += self.capital * position.realized_pnl
        self.today_trades_closed += 1
        self.today_realized_pnl += position.realized_pnl
        
        # Update peak and drawdown
        if self.capital > self.peak_capital:
            self.peak_capital = self.capital
        
        dd = (self.peak_capital - self.capital) / self.peak_capital
        if dd > self.max_drawdown:
            self.max_drawdown = dd
            self.max_drawdown_date = exit_time
        
        return position
    
    def close_all_positions(self, current_prices: Dict[str, float], 
                            exit_time: datetime, reason: str):
        """Close all open positions."""
        symbols = list(self.open_positions.keys())
        for symbol in symbols:
            if symbol in current_prices:
                self.close_position(symbol, current_prices[symbol], exit_time, reason)
    
    # =========================================================================
    # DAILY OPERATIONS
    # =========================================================================
    
    def update_prices(self, prices: Dict[str, float]):
        """Update all position prices."""
        for symbol, price in prices.items():
            if symbol in self.open_positions:
                self.open_positions[symbol].update_price(price)
    
    def get_unrealized_pnl(self) -> float:
        """Get total unrealized P/L."""
        return sum(p.unrealized_pnl for p in self.open_positions.values())
    
    def get_current_dd(self) -> float:
        """Get current drawdown including unrealized."""
        current_equity = self.capital + self.capital * self.get_unrealized_pnl()
        if current_equity > self.peak_capital:
            return 0.0
        return (self.peak_capital - current_equity) / self.peak_capital
    
    def end_of_day(self, date: datetime, prices: Dict[str, float]):
        """Process end of day - create snapshot and reset daily counters."""
        self.update_prices(prices)
        
        snapshot = DailySnapshot(
            date=date,
            starting_capital=self.capital - self.capital * self.today_realized_pnl,
            ending_capital=self.capital,
            open_positions=len(self.open_positions),
            trades_opened=self.today_trades_opened,
            trades_closed=self.today_trades_closed,
            realized_pnl=self.today_realized_pnl,
            unrealized_pnl=self.get_unrealized_pnl(),
            total_exposure=self.get_total_exposure(),
            max_drawdown=self.max_drawdown,
            danger_level=self.danger_level,
            protection_actions=[],
        )
        
        self.daily_snapshots.append(snapshot)
        
        # Reset daily counters
        self.today_trades_opened = 0
        self.today_trades_closed = 0
        self.today_realized_pnl = 0.0
        
        return snapshot
    
    def reset_for_new_month(self):
        """Reset state for a new month simulation."""
        self.capital = self.initial_capital
        self.peak_capital = self.initial_capital
        self.open_positions.clear()
        self.closed_positions.clear()
        self.daily_snapshots.clear()
        self.today_trades_opened = 0
        self.today_trades_closed = 0
        self.today_realized_pnl = 0.0
        self.max_drawdown = 0.0
        self.max_drawdown_date = None
        self.danger_level = 'SAFE'
        self.trading_halted = False
        self.halt_until = None
    
    # =========================================================================
    # REPORTING
    # =========================================================================
    
    def get_monthly_stats(self) -> dict:
        """Get monthly performance statistics."""
        if not self.closed_positions:
            return {}
        
        wins = [p for p in self.closed_positions if p.realized_pnl > 0]
        losses = [p for p in self.closed_positions if p.realized_pnl < 0]
        
        gross_profit = sum(p.realized_pnl for p in wins) if wins else 0
        gross_loss = abs(sum(p.realized_pnl for p in losses)) if losses else 0.001
        
        net_return = (self.capital - self.initial_capital) / self.initial_capital
        
        # Best/worst day
        daily_pnls = [(s.date, s.realized_pnl) for s in self.daily_snapshots]
        if daily_pnls:
            worst_day = min(daily_pnls, key=lambda x: x[1])
            best_day = max(daily_pnls, key=lambda x: x[1])
        else:
            worst_day = (None, 0)
            best_day = (None, 0)
        
        return {
            'starting_capital': self.initial_capital,
            'ending_capital': self.capital,
            'net_return_pct': net_return * 100,
            'gross_profit': gross_profit * self.initial_capital,
            'gross_loss': gross_loss * self.initial_capital,
            'profit_factor': gross_profit / gross_loss if gross_loss > 0 else 999,
            'total_trades': len(self.closed_positions),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': len(wins) / len(self.closed_positions) * 100,
            'max_drawdown_pct': self.max_drawdown * 100,
            'max_drawdown_date': self.max_drawdown_date,
            'worst_day': worst_day,
            'best_day': best_day,
        }
