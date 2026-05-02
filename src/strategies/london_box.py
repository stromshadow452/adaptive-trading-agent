"""
SCOPUS v3.0 — London Box Breakout

EMA-filtered Asian range breakout strategy for London session.
Trades ONLY confirmed trend breakouts with acceptance bar.

Session Window: 07:30 - 10:30 UTC
Asset: GBPUSD only
Logic: EMA trend filter + Asian box breakout + acceptance confirmation

Key Rules:
- Price above EMA50 with rising slope → LONG only
- Price below EMA50 with falling slope → SHORT only
- Breakout candle must close outside Asian range
- Next bar must HOLD outside range (acceptance)
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from datetime import datetime, time
from enum import Enum


class BoxSignal(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    PENDING_LONG = "PENDING_LONG"
    PENDING_SHORT = "PENDING_SHORT"
    EXIT_LONG = "EXIT_LONG"
    EXIT_SHORT = "EXIT_SHORT"
    HOLD = "HOLD"
    SESSION_CLOSED = "SESSION_CLOSED"


@dataclass
class LondonBoxConfig:
    """London Box Breakout configuration."""
    
    # Session times (UTC hours)
    asian_start_hour: int = 0
    asian_end_hour: int = 7
    london_start_hour: int = 7
    london_start_minute: int = 30
    london_end_hour: int = 10
    london_end_minute: int = 30
    max_entry_hour: int = 9
    max_entry_minute: int = 30
    
    # Asian box requirements
    min_asian_range_pips: float = 15.0   # Reduced from 25
    max_asian_range_pips: float = 80.0   # Increased from 60
    
    # EMA filter
    ema_period: int = 50
    ema_slope_lookback: int = 5
    min_ema_slope_pips: float = 0.5  # Relaxed - EMA must move at least 0.5 pips
    
    # Breakout requirements
    min_breakout_pips: float = 5.0   # Reduced from 10 to 5
    require_body_close: bool = True
    
    # Acceptance bar
    acceptance_hold_pct: float = 0.5  # 50% of bar outside range
    
    # Risk management
    risk_reward: float = 2.0
    max_trades_per_day: int = 1
    no_counter_trade: bool = True
    
    # ATR for validation
    atr_period: int = 14


@dataclass
class AsianBox:
    """Asian session range data."""
    high: float = 0.0
    low: float = 0.0
    range_pips: float = 0.0
    valid: bool = False


@dataclass
class BoxSignalResult:
    """Signal result with context."""
    signal: BoxSignal
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    reason: str = ""
    asian_box: AsianBox = None
    ema_value: float = 0.0


class LondonBoxEngine:
    """
    London Box Breakout Engine.
    
    Strategy Flow:
    1. Calculate Asian session high/low (00:00-07:00)
    2. Check EMA filter (price vs EMA50, slope direction)
    3. Wait for breakout close outside Asian range
    4. Confirm with acceptance bar (holds outside)
    5. Enter with stop at opposite side of range
    """
    
    def __init__(self, config: LondonBoxConfig = None, pip_value: float = 0.0001):
        self.config = config or LondonBoxConfig()
        self.pip_value = pip_value
        
        # State
        self.asian_box = AsianBox()
        self.position = None
        self.trades_today = 0
        self.current_date = None
        self.pending_breakout = None  # 'LONG' or 'SHORT'
        self.breakout_bar_idx = None
    
    def is_london_session(self, bar_time: datetime) -> bool:
        """Check if in London session window."""
        start = time(self.config.london_start_hour, self.config.london_start_minute)
        end = time(self.config.london_end_hour, self.config.london_end_minute)
        bar_t = bar_time.time()
        return start <= bar_t < end
    
    def can_still_enter(self, bar_time: datetime) -> bool:
        """Check if within entry window."""
        max_entry = time(self.config.max_entry_hour, self.config.max_entry_minute)
        return bar_time.time() < max_entry
    
    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute EMA and ATR."""
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        
        # EMA 50
        df['ema50'] = df['close'].ewm(span=self.config.ema_period, adjust=False).mean()
        
        # ATR
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift(1))
        tr3 = abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = tr.ewm(span=self.config.atr_period, adjust=False).mean()
        
        # Shift to prevent look-ahead
        df['ema50'] = df['ema50'].shift(1)
        df['atr'] = df['atr'].shift(1)
        
        return df
    
    def _calculate_asian_box(self, df: pd.DataFrame, current_date):
        """Calculate Asian session high/low."""
        today_mask = df.index.date == current_date
        asian_mask = df.index.hour < self.config.asian_end_hour
        asian_bars = df[today_mask & asian_mask]
        
        if len(asian_bars) >= 4:
            self.asian_box.high = asian_bars['high'].max()
            self.asian_box.low = asian_bars['low'].min()
            self.asian_box.range_pips = (self.asian_box.high - self.asian_box.low) / self.pip_value
            
            self.asian_box.valid = (
                self.asian_box.range_pips >= self.config.min_asian_range_pips and
                self.asian_box.range_pips <= self.config.max_asian_range_pips
            )
        else:
            self.asian_box.valid = False
    
    def _check_ema_filter(self, df: pd.DataFrame, bar_idx: int) -> Tuple[str, bool]:
        """
        Check EMA direction filter.
        Returns (allowed_direction, is_valid)
        """
        close = df.iloc[bar_idx]['close']
        ema = df.iloc[bar_idx]['ema50']
        
        if pd.isna(ema):
            return None, False
        
        # Check slope
        lookback = self.config.ema_slope_lookback
        if bar_idx < lookback:
            return None, False
        
        ema_now = df.iloc[bar_idx]['ema50']
        ema_prev = df.iloc[bar_idx - lookback]['ema50']
        
        if pd.isna(ema_prev):
            return None, False
        
        slope_pips = (ema_now - ema_prev) / self.pip_value
        
        # LONG: Price above EMA AND EMA rising
        if close > ema and slope_pips >= self.config.min_ema_slope_pips:
            return "LONG", True
        
        # SHORT: Price below EMA AND EMA falling
        if close < ema and slope_pips <= -self.config.min_ema_slope_pips:
            return "SHORT", True
        
        return None, False
    
    def _check_breakout(self, bar: pd.Series, allowed_direction: str) -> Optional[str]:
        """Check for breakout close outside Asian range."""
        close = bar['close']
        
        if allowed_direction == "LONG":
            if close > self.asian_box.high:
                breakout_pips = (close - self.asian_box.high) / self.pip_value
                if breakout_pips >= self.config.min_breakout_pips:
                    return "LONG"
        
        elif allowed_direction == "SHORT":
            if close < self.asian_box.low:
                breakout_pips = (self.asian_box.low - close) / self.pip_value
                if breakout_pips >= self.config.min_breakout_pips:
                    return "SHORT"
        
        return None
    
    def _check_acceptance(self, bar: pd.Series) -> bool:
        """Check if bar holds outside Asian range (acceptance)."""
        if self.pending_breakout == "LONG":
            # For LONG: check how much of bar is above Asian high
            if bar['low'] >= self.asian_box.high:
                return True  # Entire bar above
            
            bar_range = bar['high'] - bar['low']
            if bar_range > 0:
                outside_portion = max(0, bar['high'] - self.asian_box.high) / bar_range
                return outside_portion >= self.config.acceptance_hold_pct
        
        elif self.pending_breakout == "SHORT":
            # For SHORT: check how much of bar is below Asian low
            if bar['high'] <= self.asian_box.low:
                return True  # Entire bar below
            
            bar_range = bar['high'] - bar['low']
            if bar_range > 0:
                outside_portion = max(0, self.asian_box.low - bar['low']) / bar_range
                return outside_portion >= self.config.acceptance_hold_pct
        
        return False
    
    def generate_signal(self, df: pd.DataFrame, bar_idx: int = -1) -> BoxSignalResult:
        """Generate trading signal."""
        
        bar = df.iloc[bar_idx]
        bar_time = df.index[bar_idx]
        close = bar['close']
        ema = bar['ema50']
        
        current_date = bar_time.date()
        
        # New day - reset state
        if current_date != self.current_date:
            self.current_date = current_date
            self.asian_box = AsianBox()
            self.trades_today = 0
            self.position = None
            self.pending_breakout = None
            self.breakout_bar_idx = None
        
        # Calculate Asian box during Asian session
        if bar_time.hour < self.config.asian_end_hour:
            self._calculate_asian_box(df, current_date)
            return BoxSignalResult(
                signal=BoxSignal.HOLD,
                reason="Asian session - building range",
                asian_box=self.asian_box,
                ema_value=ema
            )
        
        # Outside London session
        if not self.is_london_session(bar_time):
            if self.position is not None:
                return BoxSignalResult(
                    signal=BoxSignal.EXIT_LONG if self.position == "LONG" else BoxSignal.EXIT_SHORT,
                    reason="Session ended",
                    asian_box=self.asian_box,
                    ema_value=ema
                )
            return BoxSignalResult(
                signal=BoxSignal.SESSION_CLOSED,
                reason="Outside London session",
                asian_box=self.asian_box,
                ema_value=ema
            )
        
        # Asian box not valid
        if not self.asian_box.valid:
            return BoxSignalResult(
                signal=BoxSignal.HOLD,
                reason=f"Asian range invalid: {self.asian_box.range_pips:.0f} pips",
                asian_box=self.asian_box,
                ema_value=ema
            )
        
        # Trade limit reached
        if self.trades_today >= self.config.max_trades_per_day:
            return BoxSignalResult(
                signal=BoxSignal.HOLD,
                reason="Daily trade limit",
                asian_box=self.asian_box,
                ema_value=ema
            )
        
        # Check for exit (if in position)
        if self.position == "LONG":
            # Hit target
            target = self.asian_box.high + (self.asian_box.high - self.asian_box.low) * self.config.risk_reward
            if bar['high'] >= target:
                return BoxSignalResult(
                    signal=BoxSignal.EXIT_LONG,
                    take_profit=target,
                    reason="Target hit",
                    asian_box=self.asian_box,
                    ema_value=ema
                )
        
        if self.position == "SHORT":
            target = self.asian_box.low - (self.asian_box.high - self.asian_box.low) * self.config.risk_reward
            if bar['low'] <= target:
                return BoxSignalResult(
                    signal=BoxSignal.EXIT_SHORT,
                    take_profit=target,
                    reason="Target hit",
                    asian_box=self.asian_box,
                    ema_value=ema
                )
        
        # Already in position
        if self.position is not None:
            return BoxSignalResult(
                signal=BoxSignal.HOLD,
                reason="In position",
                asian_box=self.asian_box,
                ema_value=ema
            )
        
        # Check for acceptance (if pending breakout)
        if self.pending_breakout is not None:
            if self._check_acceptance(bar):
                # Acceptance confirmed - ENTRY
                if self.pending_breakout == "LONG":
                    stop = self.asian_box.low
                    entry = close
                    target = entry + (entry - stop) * self.config.risk_reward
                    
                    return BoxSignalResult(
                        signal=BoxSignal.LONG,
                        entry_price=entry,
                        stop_loss=stop,
                        take_profit=target,
                        reason="LONG: Breakout accepted",
                        asian_box=self.asian_box,
                        ema_value=ema
                    )
                else:
                    stop = self.asian_box.high
                    entry = close
                    target = entry - (stop - entry) * self.config.risk_reward
                    
                    return BoxSignalResult(
                        signal=BoxSignal.SHORT,
                        entry_price=entry,
                        stop_loss=stop,
                        take_profit=target,
                        reason="SHORT: Breakout accepted",
                        asian_box=self.asian_box,
                        ema_value=ema
                    )
            else:
                # Acceptance failed
                self.pending_breakout = None
                return BoxSignalResult(
                    signal=BoxSignal.HOLD,
                    reason="Acceptance failed - fake breakout",
                    asian_box=self.asian_box,
                    ema_value=ema
                )
        
        # Too late to enter
        if not self.can_still_enter(bar_time):
            return BoxSignalResult(
                signal=BoxSignal.HOLD,
                reason="Past entry cutoff",
                asian_box=self.asian_box,
                ema_value=ema
            )
        
        # Check EMA filter
        allowed_direction, ema_valid = self._check_ema_filter(df, bar_idx)
        
        if not ema_valid:
            return BoxSignalResult(
                signal=BoxSignal.HOLD,
                reason="EMA filter blocked",
                asian_box=self.asian_box,
                ema_value=ema
            )
        
        # Check for breakout
        breakout = self._check_breakout(bar, allowed_direction)
        
        if breakout:
            self.pending_breakout = breakout
            self.breakout_bar_idx = bar_idx
            
            return BoxSignalResult(
                signal=BoxSignal.PENDING_LONG if breakout == "LONG" else BoxSignal.PENDING_SHORT,
                reason=f"Breakout {breakout} - waiting for acceptance",
                asian_box=self.asian_box,
                ema_value=ema
            )
        
        return BoxSignalResult(
            signal=BoxSignal.HOLD,
            reason="No breakout",
            asian_box=self.asian_box,
            ema_value=ema
        )
    
    def enter_position(self, direction: str):
        """Record position entry."""
        self.position = direction
        self.trades_today += 1
        self.pending_breakout = None
    
    def exit_position(self):
        """Record position exit."""
        self.position = None
    
    def reset(self):
        """Reset engine state."""
        self.asian_box = AsianBox()
        self.position = None
        self.trades_today = 0
        self.current_date = None
        self.pending_breakout = None
        self.breakout_bar_idx = None


if __name__ == '__main__':
    print("=" * 70)
    print(" SCOPUS v3.0 — London Box Breakout")
    print(" EMA-Filtered Asian Range Breakout")
    print("=" * 70)
    
    config = LondonBoxConfig()
    
    print(f"\n Session Window:")
    print(f"   Asian: {config.asian_start_hour:02d}:00 - {config.asian_end_hour:02d}:00 UTC")
    print(f"   London: {config.london_start_hour:02d}:{config.london_start_minute:02d} - {config.london_end_hour:02d}:{config.london_end_minute:02d} UTC")
    print(f"   Max Entry: {config.max_entry_hour:02d}:{config.max_entry_minute:02d} UTC")
    
    print(f"\n Asian Box:")
    print(f"   Range: {config.min_asian_range_pips} - {config.max_asian_range_pips} pips")
    
    print(f"\n EMA Filter:")
    print(f"   Period: {config.ema_period}")
    print(f"   Slope Lookback: {config.ema_slope_lookback} bars")
    print(f"   Min Slope: {config.min_ema_slope_pips} pips")
    
    print(f"\n Breakout:")
    print(f"   Min Size: {config.min_breakout_pips} pips")
    print(f"   Acceptance: {config.acceptance_hold_pct*100}% outside") 
    
    print(f"\n Risk:")
    print(f"   R:R = 1:{config.risk_reward}")
    print(f"   Max Trades/Day: {config.max_trades_per_day}")
    
    print(f"\n Asset: GBPUSD only")
    print(f"\n Status: READY FOR BACKTEST")
