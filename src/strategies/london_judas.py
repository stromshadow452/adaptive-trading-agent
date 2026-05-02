"""
SCOPUS v3.0 — London Session Judas Swing

Session-based liquidity sweep strategy for London open.
Trades the "trap" after Asian range liquidity grab.

Session Window: 07:00 - 12:00 UTC (Prime: 07:00 - 10:00)
Assets: GBPUSD (primary), EURUSD
Logic: Wait for sweep of Asian range, enter on reversal

Key Characteristics:
- Highest global liquidity
- Institutional manipulation common
- Retail stops clustered beyond Asian range
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from datetime import datetime, time, timedelta
from enum import Enum


class JudasSignal(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    EXIT_LONG = "EXIT_LONG"
    EXIT_SHORT = "EXIT_SHORT"
    HOLD = "HOLD"
    SESSION_CLOSED = "SESSION_CLOSED"


@dataclass
class LondonConfig:
    """London Judas Swing configuration."""
    
    # Session times (UTC)
    asian_start_hour: int = 0      # Asian session start
    asian_end_hour: int = 7        # London open
    london_end_hour: int = 12      # End of London prime
    
    # Sweep detection
    min_sweep_pips: float = 5.0    # Minimum sweep beyond range
    max_sweep_pips: float = 25.0   # Maximum sweep (avoid genuine breakout)
    max_reentry_bars: int = 8      # Must re-enter within 8 bars (2 hours on M15)
    
    # Asian range requirements
    min_asian_range_pips: float = 25.0
    max_asian_range_pips: float = 80.0
    
    # Entry confirmation
    require_engulfing: bool = False  # Require engulfing pattern
    
    # Risk management
    atr_period: int = 14
    atr_sl_multiplier: float = 2.0   # Stop beyond sweep high/low
    risk_per_trade: float = 0.01     # 1% per trade
    
    # Filters
    max_trades_per_day: int = 1      # ONE trade per day


@dataclass
class AsianRange:
    """Asian session range data."""
    high: float = 0.0
    low: float = 0.0
    formed: bool = False
    sweep_type: str = None  # 'HIGH' or 'LOW'
    sweep_price: float = 0.0
    sweep_time: datetime = None
    reentry_confirmed: bool = False


@dataclass
class JudasSignalResult:
    """Signal result with context."""
    signal: JudasSignal
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    reason: str = ""
    asian_range: AsianRange = None
    confidence: float = 0.0


class LondonJudasEngine:
    """
    London Judas Swing Engine.
    
    Strategy Logic:
    1. Mark Asian High and Low (00:00 - 07:00 UTC)
    2. At London open, wait for liquidity sweep
    3. Sweep = price exceeds Asian H/L by 5-25 pips
    4. Confirmation = price re-enters Asian range
    5. Entry = OPPOSITE direction of sweep
    """
    
    def __init__(self, config: LondonConfig = None, pip_value: float = 0.0001):
        self.config = config or LondonConfig()
        self.pip_value = pip_value
        
        # Session state
        self.asian_range = AsianRange()
        self.position = None
        self.trades_today = 0
        self.current_date = None
        self.bars_since_sweep = 0
    
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
        df['atr'] = df['atr'].shift(1)
        
        return df
    
    def _update_asian_range(self, df: pd.DataFrame, bar_time: datetime):
        """Calculate Asian session range."""
        current_date = bar_time.date()
        
        # New day - reset
        if current_date != self.current_date:
            self.current_date = current_date
            self.asian_range = AsianRange()
            self.trades_today = 0
            self.position = None
            self.bars_since_sweep = 0
        
        # Build Asian range during Asian session
        if bar_time.hour < self.config.asian_end_hour:
            today_mask = df.index.date == current_date
            asian_mask = df.index.hour < self.config.asian_end_hour
            asia_bars = df[today_mask & asian_mask]
            
            if len(asia_bars) >= 4:  # At least 1 hour of data
                self.asian_range.high = asia_bars['high'].max()
                self.asian_range.low = asia_bars['low'].min()
                
                range_pips = (self.asian_range.high - self.asian_range.low) / self.pip_value
                self.asian_range.formed = (
                    range_pips >= self.config.min_asian_range_pips and
                    range_pips <= self.config.max_asian_range_pips
                )
    
    def _detect_sweep(self, bar: pd.Series, bar_time: datetime):
        """Detect liquidity sweep."""
        if self.asian_range.sweep_type is not None:
            return  # Already detected sweep
        
        high = bar['high']
        low = bar['low']
        close = bar['close']
        
        # Check for HIGH sweep (price exceeds Asian High)
        if high > self.asian_range.high:
            sweep_distance = (high - self.asian_range.high) / self.pip_value
            
            if (self.config.min_sweep_pips <= sweep_distance <= self.config.max_sweep_pips):
                self.asian_range.sweep_type = "HIGH"
                self.asian_range.sweep_price = high
                self.asian_range.sweep_time = bar_time
                self.bars_since_sweep = 0
        
        # Check for LOW sweep
        elif low < self.asian_range.low:
            sweep_distance = (self.asian_range.low - low) / self.pip_value
            
            if (self.config.min_sweep_pips <= sweep_distance <= self.config.max_sweep_pips):
                self.asian_range.sweep_type = "LOW"
                self.asian_range.sweep_price = low
                self.asian_range.sweep_time = bar_time
                self.bars_since_sweep = 0
    
    def _check_reentry(self, bar: pd.Series) -> bool:
        """Check if price re-enters Asian range after sweep."""
        if self.asian_range.sweep_type is None:
            return False
        
        close = bar['close']
        
        # Price must return inside the Asian range
        if self.asian_range.low <= close <= self.asian_range.high:
            return True
        
        return False
    
    def generate_signal(self, df: pd.DataFrame, bar_idx: int = -1) -> JudasSignalResult:
        """Generate trading signal."""
        
        bar = df.iloc[bar_idx]
        bar_time = df.index[bar_idx]
        close = bar['close']
        atr = bar['atr']
        
        # Update Asian range
        self._update_asian_range(df, bar_time)
        
        # Outside London session
        if bar_time.hour < self.config.asian_end_hour or bar_time.hour >= self.config.london_end_hour:
            if self.position is not None:
                return JudasSignalResult(
                    signal=JudasSignal.EXIT_LONG if self.position == "LONG" else JudasSignal.EXIT_SHORT,
                    reason="Session ended",
                    asian_range=self.asian_range
                )
            return JudasSignalResult(
                signal=JudasSignal.SESSION_CLOSED,
                reason="Outside London session",
                asian_range=self.asian_range
            )
        
        # Asian range not valid
        if not self.asian_range.formed:
            return JudasSignalResult(
                signal=JudasSignal.HOLD,
                reason="Asian range invalid",
                asian_range=self.asian_range
            )
        
        # Trade limit
        if self.trades_today >= self.config.max_trades_per_day:
            return JudasSignalResult(
                signal=JudasSignal.HOLD,
                reason="Daily trade limit reached",
                asian_range=self.asian_range
            )
        
        # Detect new sweep
        self._detect_sweep(bar, bar_time)
        
        # Track bars since sweep
        if self.asian_range.sweep_type is not None:
            self.bars_since_sweep += 1
            
            # Timeout - sweep didn't result in trap
            if self.bars_since_sweep > self.config.max_reentry_bars:
                self.asian_range.sweep_type = None  # Reset, wait for new sweep
                return JudasSignalResult(
                    signal=JudasSignal.HOLD,
                    reason="Sweep timeout - no reentry",
                    asian_range=self.asian_range
                )
        
        # Exit check
        if self.position == "LONG":
            asian_mid = (self.asian_range.high + self.asian_range.low) / 2
            if close >= asian_mid:
                return JudasSignalResult(
                    signal=JudasSignal.EXIT_LONG,
                    take_profit=asian_mid,
                    reason="Target hit (Asian mid)",
                    asian_range=self.asian_range
                )
        
        if self.position == "SHORT":
            asian_mid = (self.asian_range.high + self.asian_range.low) / 2
            if close <= asian_mid:
                return JudasSignalResult(
                    signal=JudasSignal.EXIT_SHORT,
                    take_profit=asian_mid,
                    reason="Target hit (Asian mid)",
                    asian_range=self.asian_range
                )
        
        # Already in position
        if self.position is not None:
            return JudasSignalResult(
                signal=JudasSignal.HOLD,
                reason="In position",
                asian_range=self.asian_range
            )
        
        # Check for entry after sweep + reentry
        if self.asian_range.sweep_type is not None and self._check_reentry(bar):
            self.asian_range.reentry_confirmed = True
            
            asian_mid = (self.asian_range.high + self.asian_range.low) / 2
            
            if self.asian_range.sweep_type == "HIGH":
                # Swept high, enter SHORT
                stop_loss = self.asian_range.sweep_price + (atr * self.config.atr_sl_multiplier)
                
                return JudasSignalResult(
                    signal=JudasSignal.SHORT,
                    entry_price=close,
                    stop_loss=stop_loss,
                    take_profit=asian_mid,
                    reason=f"Judas SHORT: High swept, reentry confirmed",
                    asian_range=self.asian_range,
                    confidence=0.7
                )
            
            else:  # LOW sweep
                # Swept low, enter LONG
                stop_loss = self.asian_range.sweep_price - (atr * self.config.atr_sl_multiplier)
                
                return JudasSignalResult(
                    signal=JudasSignal.LONG,
                    entry_price=close,
                    stop_loss=stop_loss,
                    take_profit=asian_mid,
                    reason=f"Judas LONG: Low swept, reentry confirmed",
                    asian_range=self.asian_range,
                    confidence=0.7
                )
        
        # Waiting for setup
        if self.asian_range.sweep_type is not None:
            return JudasSignalResult(
                signal=JudasSignal.HOLD,
                reason=f"Sweep detected ({self.asian_range.sweep_type}), waiting for reentry",
                asian_range=self.asian_range
            )
        
        return JudasSignalResult(
            signal=JudasSignal.HOLD,
            reason="No sweep detected",
            asian_range=self.asian_range
        )
    
    def enter_position(self, direction: str):
        """Record position entry."""
        self.position = direction
        self.trades_today += 1
    
    def exit_position(self):
        """Record position exit."""
        self.position = None
    
    def reset(self):
        """Reset engine state."""
        self.asian_range = AsianRange()
        self.position = None
        self.trades_today = 0
        self.current_date = None
        self.bars_since_sweep = 0


# ============================================================================
# TEST
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print(" SCOPUS v3.0 — London Session Judas Swing")
    print(" Liquidity Sweep Trap Strategy")
    print("=" * 70)
    
    config = LondonConfig()
    
    print(f"\n Session Window:")
    print(f"   Asian: {config.asian_start_hour:02d}:00 - {config.asian_end_hour:02d}:00 UTC")
    print(f"   London: {config.asian_end_hour:02d}:00 - {config.london_end_hour:02d}:00 UTC")
    
    print(f"\n Sweep Parameters:")
    print(f"   Min Sweep: {config.min_sweep_pips} pips")
    print(f"   Max Sweep: {config.max_sweep_pips} pips")
    print(f"   Reentry Window: {config.max_reentry_bars} bars")
    
    print(f"\n Risk Management:")
    print(f"   ATR Stop: {config.atr_sl_multiplier}x")
    print(f"   Risk/Trade: {config.risk_per_trade*100}%")
    print(f"   Max Trades/Day: {config.max_trades_per_day}")
    
    print(f"\n Assets: GBPUSD, EURUSD")
    print(f"\n Status: READY FOR BACKTEST")
