"""
SCOPUS v3.0 — Tokyo Session Mean Reversion

Session-based mean reversion strategy for Asian session.
Trades JPY pairs within the Tokyo range using statistical extremes.

Session Window: 00:00 - 07:00 UTC (Prime: 03:00 - 07:00)
Assets: USDJPY, AUDJPY, NZDJPY, AUDUSD
Logic: Mean reversion within Asian range at extreme deviations

Key Characteristics:
- Low volatility accumulation phase
- BoJ intervention creates floor/ceiling
- Corporate hedging creates mean-reverting flow
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
from datetime import datetime, time
from enum import Enum


class SessionSignal(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    EXIT_LONG = "EXIT_LONG"
    EXIT_SHORT = "EXIT_SHORT"
    HOLD = "HOLD"
    SESSION_CLOSED = "SESSION_CLOSED"


@dataclass
class TokyoConfig:
    """Tokyo Session configuration."""
    
    # Session times (UTC)
    session_start_hour: int = 0   # 00:00 UTC
    session_end_hour: int = 7     # 07:00 UTC
    prime_start_hour: int = 3     # Prime trading: 03:00
    
    # Range parameters
    range_formation_hours: int = 2  # Wait 2 hours for range to form
    min_range_pips: float = 15.0    # Minimum range required
    max_range_pips: float = 60.0    # Maximum range (avoid volatile days)
    
    # Entry parameters
    entry_threshold_pct: float = 0.80  # Enter at 80% of range extreme
    exit_target_pct: float = 0.50      # Exit at 50% (middle of range)
    
    # Risk management
    atr_period: int = 14
    atr_sl_multiplier: float = 1.5   # Tighter stops for range trading
    risk_per_trade: float = 0.005    # 0.5% per trade
    
    # Filters
    min_atr_pips: float = 8.0        # Skip dead market
    max_atr_pips: float = 30.0       # Skip volatile market
    max_trades_per_session: int = 2   # Limit overtrading


@dataclass
class TokyoRange:
    """Asian session range data."""
    high: float = 0.0
    low: float = 0.0
    mid: float = 0.0
    range_pips: float = 0.0
    formed: bool = False
    formation_time: datetime = None


@dataclass  
class TokyoSignalResult:
    """Signal result with context."""
    signal: SessionSignal
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    reason: str = ""
    range_info: TokyoRange = None
    confidence: float = 0.0


class TokyoSessionEngine:
    """
    Tokyo Session Mean Reversion Engine.
    
    Strategy Logic:
    1. Wait for session to form range (first 2 hours)
    2. Mark Asian High and Low
    3. Trade mean reversion at range extremes
    4. Exit at range midpoint
    """
    
    def __init__(self, config: TokyoConfig = None, pip_value: float = 0.01):
        self.config = config or TokyoConfig()
        self.pip_value = pip_value
        
        # Session state
        self.current_range = TokyoRange()
        self.position = None
        self.trades_this_session = 0
        self.current_session_date = None
    
    def is_tokyo_session(self, bar_time: datetime) -> bool:
        """Check if current time is in Tokyo session."""
        hour = bar_time.hour
        return self.config.session_start_hour <= hour < self.config.session_end_hour
    
    def is_prime_trading_time(self, bar_time: datetime) -> bool:
        """Check if in prime trading window (after range forms)."""
        hour = bar_time.hour
        return self.config.prime_start_hour <= hour < self.config.session_end_hour
    
    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute required indicators."""
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        
        # ATR
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift(1))
        tr3 = abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = tr.ewm(span=self.config.atr_period, adjust=False).mean()
        
        # Shift to prevent look-ahead
        df['atr'] = df['atr'].shift(1)
        
        return df
    
    def update_session_range(self, df: pd.DataFrame, bar_time: datetime):
        """
        Update or reset Asian session range.
        Range is formed from session start to prime_start_hour.
        """
        current_date = bar_time.date()
        
        # New session - reset range
        if current_date != self.current_session_date:
            self.current_session_date = current_date
            self.current_range = TokyoRange()
            self.trades_this_session = 0
            self.position = None
        
        # If in range formation period (00:00 - 03:00)
        if bar_time.hour < self.config.prime_start_hour:
            # Get today's bars so far
            today_mask = df.index.date == current_date
            session_mask = df.index.hour < self.config.prime_start_hour
            range_bars = df[today_mask & session_mask]
            
            if len(range_bars) > 0:
                self.current_range.high = range_bars['high'].max()
                self.current_range.low = range_bars['low'].min()
                self.current_range.mid = (self.current_range.high + self.current_range.low) / 2
                self.current_range.range_pips = (self.current_range.high - self.current_range.low) / self.pip_value
        
        # Mark range as formed when entering prime time
        if bar_time.hour >= self.config.prime_start_hour and not self.current_range.formed:
            if (self.current_range.range_pips >= self.config.min_range_pips and
                self.current_range.range_pips <= self.config.max_range_pips):
                self.current_range.formed = True
                self.current_range.formation_time = bar_time
    
    def generate_signal(self, df: pd.DataFrame, bar_idx: int = -1) -> TokyoSignalResult:
        """Generate trading signal based on Tokyo range."""
        
        bar = df.iloc[bar_idx]
        bar_time = df.index[bar_idx]
        close = bar['close']
        atr = bar['atr']
        
        # Check session
        if not self.is_tokyo_session(bar_time):
            # Exit any positions at session end
            if self.position is not None:
                return TokyoSignalResult(
                    signal=SessionSignal.EXIT_LONG if self.position == "LONG" else SessionSignal.EXIT_SHORT,
                    reason="Session ended",
                    range_info=self.current_range
                )
            return TokyoSignalResult(
                signal=SessionSignal.SESSION_CLOSED,
                reason="Outside Tokyo session",
                range_info=self.current_range
            )
        
        # Update range
        self.update_session_range(df, bar_time)
        
        # Not in prime trading time yet
        if not self.is_prime_trading_time(bar_time):
            return TokyoSignalResult(
                signal=SessionSignal.HOLD,
                reason="Range forming",
                range_info=self.current_range
            )
        
        # Range not formed properly
        if not self.current_range.formed:
            return TokyoSignalResult(
                signal=SessionSignal.HOLD,
                reason=f"Range invalid: {self.current_range.range_pips:.1f} pips",
                range_info=self.current_range
            )
        
        # Check ATR filter
        atr_pips = atr / self.pip_value
        if pd.isna(atr) or atr_pips < self.config.min_atr_pips:
            return TokyoSignalResult(
                signal=SessionSignal.HOLD,
                reason="Low volatility (dead market)",
                range_info=self.current_range
            )
        
        if atr_pips > self.config.max_atr_pips:
            return TokyoSignalResult(
                signal=SessionSignal.HOLD,
                reason="High volatility (skip session)",
                range_info=self.current_range
            )
        
        # Trade limit
        if self.trades_this_session >= self.config.max_trades_per_session:
            return TokyoSignalResult(
                signal=SessionSignal.HOLD,
                reason="Session trade limit reached",
                range_info=self.current_range
            )
        
        # Exit check for open positions
        if self.position == "LONG":
            if close >= self.current_range.mid:
                return TokyoSignalResult(
                    signal=SessionSignal.EXIT_LONG,
                    take_profit=self.current_range.mid,
                    reason="Target hit (range midpoint)",
                    range_info=self.current_range
                )
        
        if self.position == "SHORT":
            if close <= self.current_range.mid:
                return TokyoSignalResult(
                    signal=SessionSignal.EXIT_SHORT,
                    take_profit=self.current_range.mid,
                    reason="Target hit (range midpoint)",
                    range_info=self.current_range
                )
        
        # Already in position
        if self.position is not None:
            return TokyoSignalResult(
                signal=SessionSignal.HOLD,
                reason="In position",
                range_info=self.current_range
            )
        
        # Entry calculations
        range_size = self.current_range.high - self.current_range.low
        upper_threshold = self.current_range.low + (range_size * self.config.entry_threshold_pct)
        lower_threshold = self.current_range.low + (range_size * (1 - self.config.entry_threshold_pct))
        
        # SHORT at range high
        if close >= upper_threshold:
            stop_loss = self.current_range.high + (atr * self.config.atr_sl_multiplier)
            
            # Calculate confidence based on proximity to extreme
            proximity = (close - self.current_range.mid) / (self.current_range.high - self.current_range.mid)
            confidence = min(proximity, 1.0)
            
            return TokyoSignalResult(
                signal=SessionSignal.SHORT,
                entry_price=close,
                stop_loss=stop_loss,
                take_profit=self.current_range.mid,
                reason=f"SHORT: Price at {proximity*100:.0f}% of upper range",
                range_info=self.current_range,
                confidence=confidence
            )
        
        # LONG at range low
        if close <= lower_threshold:
            stop_loss = self.current_range.low - (atr * self.config.atr_sl_multiplier)
            
            proximity = (self.current_range.mid - close) / (self.current_range.mid - self.current_range.low)
            confidence = min(proximity, 1.0)
            
            return TokyoSignalResult(
                signal=SessionSignal.LONG,
                entry_price=close,
                stop_loss=stop_loss,
                take_profit=self.current_range.mid,
                reason=f"LONG: Price at {proximity*100:.0f}% of lower range",
                range_info=self.current_range,
                confidence=confidence
            )
        
        return TokyoSignalResult(
            signal=SessionSignal.HOLD,
            reason="Price in middle of range",
            range_info=self.current_range
        )
    
    def enter_position(self, direction: str):
        """Record position entry."""
        self.position = direction
        self.trades_this_session += 1
    
    def exit_position(self):
        """Record position exit."""
        self.position = None
    
    def reset(self):
        """Reset engine state."""
        self.current_range = TokyoRange()
        self.position = None
        self.trades_this_session = 0
        self.current_session_date = None


# ============================================================================
# POSITION SIZING
# ============================================================================

def calculate_position_size(account_balance: float, risk_per_trade: float,
                            entry_price: float, stop_loss: float,
                            pip_value: float = 0.01) -> float:
    """Calculate position size for JPY pairs."""
    PIP_VALUE_PER_LOT = 1000  # JPY pairs: ~$9.30 per pip per lot
    
    risk_amount = account_balance * risk_per_trade
    sl_pips = abs(entry_price - stop_loss) / pip_value
    
    if sl_pips <= 0:
        return 0.0
    
    position_size = risk_amount / (sl_pips * PIP_VALUE_PER_LOT / 100)
    
    return round(position_size, 4)


# ============================================================================
# TEST
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print(" SCOPUS v3.0 — Tokyo Session Mean Reversion")
    print(" Asian Range Trading Strategy")
    print("=" * 70)
    
    config = TokyoConfig()
    
    print(f"\n Session Window:")
    print(f"   Start: {config.session_start_hour:02d}:00 UTC")
    print(f"   Prime: {config.prime_start_hour:02d}:00 UTC")
    print(f"   End:   {config.session_end_hour:02d}:00 UTC")
    
    print(f"\n Range Parameters:")
    print(f"   Min Range: {config.min_range_pips} pips")
    print(f"   Max Range: {config.max_range_pips} pips")
    print(f"   Entry at: {config.entry_threshold_pct*100}% of range")
    print(f"   Exit at: {config.exit_target_pct*100}% (midpoint)")
    
    print(f"\n Risk Management:")
    print(f"   ATR Stop: {config.atr_sl_multiplier}x")
    print(f"   Risk/Trade: {config.risk_per_trade*100}%")
    print(f"   Max Trades/Session: {config.max_trades_per_session}")
    
    print(f"\n Assets: USDJPY, AUDJPY, NZDJPY, AUDUSD")
    print(f"\n Status: READY FOR BACKTEST")
