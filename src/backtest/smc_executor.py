"""
SCOPUS v2.0 — SMC Entry Executor

Entry logic based on Smart Money Concepts (SMC).
Entries are LOCATION-BASED, not signal-based.

Entry Requirements:
    1. Valid range detected
    2. Price in correct zone (premium/discount)
    3. Liquidity swept
    4. Rejection candle confirmed
    5. Trade limits respected
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from datetime import datetime, time
from enum import Enum

from src.backtest.structure_engine import (
    MarketStructureEngine, 
    RangeInfo, 
    LiquidityPool, 
    LiquidityType,
    Zone,
    StructureType,
)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class TradeSetup:
    """Complete trade setup with entry, SL, TP."""
    valid: bool
    direction: str = ""  # "BUY" or "SELL"
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit_1: float = 0.0
    take_profit_2: float = 0.0
    risk_pips: float = 0.0
    reward_pips: float = 0.0
    rr_ratio: float = 0.0
    reason: str = ""
    zone: Zone = Zone.EQUILIBRIUM
    sweep_level: float = 0.0


@dataclass
class TradeState:
    """Tracks trade frequency and state."""
    trades_today: Dict[str, int] = None  # symbol -> count
    last_trade_time: Dict[str, datetime] = None
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    loss_streak: int = 0
    
    def __post_init__(self):
        self.trades_today = self.trades_today or {}
        self.last_trade_time = self.last_trade_time or {}


# ============================================================================
# NO-TRADE CONDITIONS
# ============================================================================

class NoTradeFilter:
    """
    Filters for conditions where we should NOT trade.
    """
    
    # Session hours (UTC)
    VALID_SESSION_START = 7   # 07:00 UTC = London open
    VALID_SESSION_END = 21    # 21:00 UTC = NY close
    
    # Days
    FRIDAY_CUTOFF_HOUR = 17   # No new trades after 17:00 UTC Friday
    SUNDAY_START_HOUR = 22    # No trades before 22:00 UTC Sunday
    
    # Limits
    MAX_TRADES_PER_DAY_PER_SYMBOL = 2
    MAX_TRADES_PER_DAY_TOTAL = 4
    MIN_SPREAD_RATIO = 3.0    # Skip if spread > 3x normal
    
    def __init__(self, normal_spread_pips: float = 1.0):
        self.normal_spread = normal_spread_pips
    
    def check_session(self, timestamp: datetime) -> Tuple[bool, str]:
        """Check if current time is within valid trading session."""
        hour = timestamp.hour
        weekday = timestamp.weekday()  # 0=Monday, 6=Sunday
        
        # Sunday check
        if weekday == 6 and hour < self.SUNDAY_START_HOUR:
            return False, "Sunday before market open"
        
        # Friday check
        if weekday == 4 and hour >= self.FRIDAY_CUTOFF_HOUR:
            return False, "Friday after cutoff"
        
        # Session hours
        if hour < self.VALID_SESSION_START or hour >= self.VALID_SESSION_END:
            return False, f"Outside session hours ({self.VALID_SESSION_START}-{self.VALID_SESSION_END} UTC)"
        
        return True, "Session valid"
    
    def check_spread(self, current_spread: float) -> Tuple[bool, str]:
        """Check if spread is acceptable."""
        if current_spread > self.normal_spread * self.MIN_SPREAD_RATIO:
            return False, f"Spread too wide ({current_spread:.1f} vs normal {self.normal_spread:.1f})"
        return True, "Spread acceptable"
    
    def check_trade_limits(self, state: TradeState, symbol: str) -> Tuple[bool, str]:
        """Check trade frequency limits."""
        symbol_trades = state.trades_today.get(symbol, 0)
        total_trades = sum(state.trades_today.values())
        
        if symbol_trades >= self.MAX_TRADES_PER_DAY_PER_SYMBOL:
            return False, f"Max trades reached for {symbol} ({symbol_trades})"
        
        if total_trades >= self.MAX_TRADES_PER_DAY_TOTAL:
            return False, f"Max daily trades reached ({total_trades})"
        
        return True, "Trade limits OK"
    
    def check_loss_streak(self, state: TradeState) -> Tuple[bool, str]:
        """Check loss streak cooldown."""
        if state.loss_streak >= 2:
            return False, f"Loss streak cooldown ({state.loss_streak} consecutive losses)"
        return True, "No loss streak"
    
    def apply_all_filters(self, timestamp: datetime, spread: float, 
                          state: TradeState, symbol: str) -> Tuple[bool, str]:
        """Apply all no-trade filters."""
        
        checks = [
            self.check_session(timestamp),
            self.check_spread(spread),
            self.check_trade_limits(state, symbol),
            self.check_loss_streak(state),
        ]
        
        for passed, reason in checks:
            if not passed:
                return False, reason
        
        return True, "All filters passed"


# ============================================================================
# REJECTION DETECTOR
# ============================================================================

class RejectionDetector:
    """
    Detects rejection candles (confirmation for entry).
    
    Rejection = wick > 50% of total candle range
    """
    
    MIN_WICK_RATIO = 0.40  # Wick must be > 40% of range (was 50%)
    
    def detect_bullish_rejection(self, candle: pd.Series) -> bool:
        """
        Bullish rejection: Long lower wick, closes upper half.
        
        Indicates sellers tried to push down but buyers rejected.
        """
        total_range = candle['high'] - candle['low']
        if total_range <= 0:
            return False
        
        lower_wick = min(candle['open'], candle['close']) - candle['low']
        upper_wick = candle['high'] - max(candle['open'], candle['close'])
        body = abs(candle['close'] - candle['open'])
        
        # Lower wick should be dominant
        wick_ratio = lower_wick / total_range
        
        # Close should be in upper half
        close_position = (candle['close'] - candle['low']) / total_range
        
        return wick_ratio >= self.MIN_WICK_RATIO and close_position > 0.5
    
    def detect_bearish_rejection(self, candle: pd.Series) -> bool:
        """
        Bearish rejection: Long upper wick, closes lower half.
        
        Indicates buyers tried to push up but sellers rejected.
        """
        total_range = candle['high'] - candle['low']
        if total_range <= 0:
            return False
        
        lower_wick = min(candle['open'], candle['close']) - candle['low']
        upper_wick = candle['high'] - max(candle['open'], candle['close'])
        
        # Upper wick should be dominant
        wick_ratio = upper_wick / total_range
        
        # Close should be in lower half
        close_position = (candle['close'] - candle['low']) / total_range
        
        return wick_ratio >= self.MIN_WICK_RATIO and close_position < 0.5


# ============================================================================
# SMC EXECUTOR
# ============================================================================

class SMCExecutor:
    """
    Smart Money Concepts Entry Executor.
    
    Entry Logic:
        BUY: Discount zone + EQL swept + bullish rejection
        SELL: Premium zone + EQH swept + bearish rejection
    
    All entries are LOCATION-BASED.
    """
    
    # Risk parameters
    SL_BUFFER_PIPS = 5.0      # Buffer beyond sweep level
    TP1_ZONE = "equilibrium"   # First TP at equilibrium
    TP2_ZONE = "opposite"      # Second TP at opposite zone
    
    def __init__(self, pip_value: float = 0.0001):
        self.pip_value = pip_value
        self.structure_engine = MarketStructureEngine(pip_value)
        self.rejection_detector = RejectionDetector()
        self.no_trade_filter = NoTradeFilter()
    
    def evaluate_setup(self, df: pd.DataFrame, symbol: str, 
                       state: TradeState, spread: float = 1.0) -> TradeSetup:
        """
        Evaluate if current bar presents a valid trade setup.
        
        Returns TradeSetup with all details.
        """
        # Get current bar info
        current_bar = df.iloc[-1]
        timestamp = df.index[-1] if hasattr(df.index, '__getitem__') else datetime.now()
        
        # Apply no-trade filters
        can_trade, filter_reason = self.no_trade_filter.apply_all_filters(
            timestamp, spread, state, symbol
        )
        if not can_trade:
            return TradeSetup(valid=False, reason=filter_reason)
        
        # Analyze market structure
        analysis = self.structure_engine.analyze(df)
        
        # Check if tradeable
        if not analysis['tradeable']:
            return TradeSetup(valid=False, reason=analysis['reason'])
        
        range_info = analysis['range']
        zone = analysis['zone']
        liquidity = analysis['liquidity']
        
        # Find swept pools
        swept_pools = [p for p in liquidity if p.swept]
        if not swept_pools:
            return TradeSetup(valid=False, reason="No liquidity swept")
        
        # Determine direction based on zone
        if zone == Zone.DISCOUNT:
            return self._evaluate_buy_setup(
                current_bar, range_info, swept_pools, spread
            )
        elif zone == Zone.PREMIUM:
            return self._evaluate_sell_setup(
                current_bar, range_info, swept_pools, spread
            )
        else:
            return TradeSetup(valid=False, reason="Price in equilibrium zone")
    
    def _evaluate_buy_setup(self, bar: pd.Series, range_info: RangeInfo,
                            pools: List[LiquidityPool], spread: float) -> TradeSetup:
        """Evaluate BUY setup in discount zone."""
        
        # Check for swept EQL
        eql_pools = [p for p in pools if p.liquidity_type == LiquidityType.EQL]
        if not eql_pools:
            return TradeSetup(valid=False, reason="No EQL swept for BUY")
        
        # Check for bullish rejection
        if not self.rejection_detector.detect_bullish_rejection(bar):
            return TradeSetup(valid=False, reason="No bullish rejection candle")
        
        # Calculate entry, SL, TP
        sweep_level = min(p.level for p in eql_pools)
        entry_price = bar['close']
        stop_loss = sweep_level - (self.SL_BUFFER_PIPS * self.pip_value)
        
        # TP1 at equilibrium (midpoint)
        equilibrium = (range_info.high + range_info.low) / 2
        tp1 = equilibrium
        
        # TP2 at premium zone
        tp2 = range_info.high - (range_info.high - range_info.low) * 0.1
        
        risk_pips = (entry_price - stop_loss) / self.pip_value
        reward_pips = (tp1 - entry_price) / self.pip_value
        rr_ratio = reward_pips / risk_pips if risk_pips > 0 else 0
        
        # Validate R:R
        if rr_ratio < 1.2:
            return TradeSetup(valid=False, reason=f"R:R too low ({rr_ratio:.2f})")
        
        return TradeSetup(
            valid=True,
            direction="BUY",
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit_1=tp1,
            take_profit_2=tp2,
            risk_pips=risk_pips,
            reward_pips=reward_pips,
            rr_ratio=rr_ratio,
            reason="Discount zone + EQL swept + bullish rejection",
            zone=Zone.DISCOUNT,
            sweep_level=sweep_level,
        )
    
    def _evaluate_sell_setup(self, bar: pd.Series, range_info: RangeInfo,
                             pools: List[LiquidityPool], spread: float) -> TradeSetup:
        """Evaluate SELL setup in premium zone."""
        
        # Check for swept EQH
        eqh_pools = [p for p in pools if p.liquidity_type == LiquidityType.EQH]
        if not eqh_pools:
            return TradeSetup(valid=False, reason="No EQH swept for SELL")
        
        # Check for bearish rejection
        if not self.rejection_detector.detect_bearish_rejection(bar):
            return TradeSetup(valid=False, reason="No bearish rejection candle")
        
        # Calculate entry, SL, TP
        sweep_level = max(p.level for p in eqh_pools)
        entry_price = bar['close']
        stop_loss = sweep_level + (self.SL_BUFFER_PIPS * self.pip_value)
        
        # TP1 at equilibrium (midpoint)
        equilibrium = (range_info.high + range_info.low) / 2
        tp1 = equilibrium
        
        # TP2 at discount zone
        tp2 = range_info.low + (range_info.high - range_info.low) * 0.1
        
        risk_pips = (stop_loss - entry_price) / self.pip_value
        reward_pips = (entry_price - tp1) / self.pip_value
        rr_ratio = reward_pips / risk_pips if risk_pips > 0 else 0
        
        # Validate R:R
        if rr_ratio < 1.2:
            return TradeSetup(valid=False, reason=f"R:R too low ({rr_ratio:.2f})")
        
        return TradeSetup(
            valid=True,
            direction="SELL",
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit_1=tp1,
            take_profit_2=tp2,
            risk_pips=risk_pips,
            reward_pips=reward_pips,
            rr_ratio=rr_ratio,
            reason="Premium zone + EQH swept + bearish rejection",
            zone=Zone.PREMIUM,
            sweep_level=sweep_level,
        )


# ============================================================================
# RISK MANAGER
# ============================================================================

class SMCRiskManager:
    """
    Risk management for SMC system.
    
    Fixed risk per trade, strict limits, kill-switches.
    """
    
    RISK_PER_TRADE = 0.005    # 0.5% of account
    MAX_DAILY_LOSS = 0.015    # 1.5% daily kill-switch
    MAX_WEEKLY_LOSS = 0.03    # 3% weekly kill-switch
    
    def __init__(self, account_balance: float):
        self.account_balance = account_balance
        self.state = TradeState()
    
    def calculate_position_size(self, setup: TradeSetup, symbol: str) -> float:
        """
        Calculate position size based on fixed risk.
        
        Position size = (Account * Risk%) / (SL in account currency)
        """
        if not setup.valid:
            return 0.0
        
        risk_amount = self.account_balance * self.RISK_PER_TRADE
        
        # SL distance in pips
        sl_pips = setup.risk_pips
        
        # Standard lot = 100,000 units, 1 pip = $10 for standard lot
        # Position size in lots
        pip_value_per_lot = 10.0  # USD pairs
        
        position_size = risk_amount / (sl_pips * pip_value_per_lot)
        
        # Round to 0.01 (micro lots)
        return round(position_size, 2)
    
    def check_kill_switch(self) -> Tuple[bool, str]:
        """Check if any kill-switch is triggered."""
        
        daily_loss_pct = abs(self.state.daily_pnl) / self.account_balance
        weekly_loss_pct = abs(self.state.weekly_pnl) / self.account_balance
        
        if self.state.daily_pnl < 0 and daily_loss_pct >= self.MAX_DAILY_LOSS:
            return True, f"Daily kill-switch triggered ({daily_loss_pct:.1%} loss)"
        
        if self.state.weekly_pnl < 0 and weekly_loss_pct >= self.MAX_WEEKLY_LOSS:
            return True, f"Weekly kill-switch triggered ({weekly_loss_pct:.1%} loss)"
        
        return False, "Kill-switches OK"
    
    def record_trade(self, symbol: str, pnl: float):
        """Record a completed trade."""
        # Update trade count
        self.state.trades_today[symbol] = self.state.trades_today.get(symbol, 0) + 1
        
        # Update P&L
        self.state.daily_pnl += pnl
        self.state.weekly_pnl += pnl
        
        # Update loss streak
        if pnl < 0:
            self.state.loss_streak += 1
        else:
            self.state.loss_streak = 0
    
    def reset_daily(self):
        """Reset daily counters."""
        self.state.trades_today = {}
        self.state.daily_pnl = 0.0
    
    def reset_weekly(self):
        """Reset weekly counters."""
        self.state.weekly_pnl = 0.0
        self.state.loss_streak = 0


# ============================================================================
# TEST
# ============================================================================

if __name__ == '__main__':
    print("SMC Entry Executor v2.0")
    print("=" * 50)
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2023-06-15 10:00', periods=200, freq='15min')
    
    # Create range-bound price with a sweep
    base = 1.1000
    prices = np.zeros(200)
    prices[0] = base
    
    for i in range(1, 200):
        # Mean reverting random walk
        prices[i] = prices[i-1] + np.random.randn() * 0.0008
        # Keep in range
        prices[i] = np.clip(prices[i], 1.0950, 1.1050)
    
    # Add a sweep at the end (price goes below then reverses)
    prices[-5:] = [1.0955, 1.0948, 1.0945, 1.0960, 1.0980]  # Sweep and reject
    
    df = pd.DataFrame({
        'open': prices,
        'high': prices + abs(np.random.randn(200)) * 0.0005,
        'low': prices - abs(np.random.randn(200)) * 0.0005,
        'close': prices + np.random.randn(200) * 0.0002,
    }, index=dates)
    
    # Fix last bar to show rejection
    df.loc[df.index[-1], 'low'] = 1.0945  # Lower wick
    df.loc[df.index[-1], 'close'] = 1.0980  # Close higher
    df.loc[df.index[-1], 'high'] = 1.0985
    
    # Test executor
    executor = SMCExecutor(pip_value=0.0001)
    state = TradeState()
    
    setup = executor.evaluate_setup(df, "EURUSD", state, spread=1.0)
    
    print(f"\nSetup Valid: {setup.valid}")
    print(f"Direction: {setup.direction}")
    print(f"Entry: {setup.entry_price:.5f}")
    print(f"SL: {setup.stop_loss:.5f}")
    print(f"TP1: {setup.take_profit_1:.5f}")
    print(f"TP2: {setup.take_profit_2:.5f}")
    print(f"Risk: {setup.risk_pips:.1f} pips")
    print(f"R:R: {setup.rr_ratio:.2f}")
    print(f"Reason: {setup.reason}")
    
    # Test risk manager
    risk_mgr = SMCRiskManager(account_balance=10000)
    position_size = risk_mgr.calculate_position_size(setup, "EURUSD")
    print(f"\nPosition Size: {position_size:.2f} lots")
