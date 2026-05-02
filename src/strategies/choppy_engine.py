"""
SCOPUS v3.0 — Choppy Engine (Strategy A)

Defensive strategy for FX majors.
Primary goal: Capital preservation, NOT high returns.

Purpose:
- Operates in EURUSD (most efficient market)
- Detects low-trend/choppy regimes
- BLOCKS most trades (defensive posture)
- Only allows micro-fades at extreme conditions

Philosophy:
- FX majors have NO retail edge
- Best strategy is to NOT trade
- Occasional micro-trades to stay engaged
- Prevents boredom-driven gambling
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional
from enum import Enum


class ChoppySignal(Enum):
    BLOCK = "BLOCK"           # Do not trade
    MICRO_LONG = "MICRO_LONG"  # Tiny long position
    MICRO_SHORT = "MICRO_SHORT"  # Tiny short position
    EXIT = "EXIT"              # Close any position


@dataclass
class ChoppyConfig:
    """Configuration - designed to be VERY restrictive."""
    
    # ADX for trend detection
    adx_period: int = 14
    adx_choppy_threshold: float = 20.0  # Below = choppy
    
    # Bollinger Bands (tight)
    bb_period: int = 20
    bb_std: float = 2.0
    
    # ATR for volatility
    atr_period: int = 14
    atr_too_low: float = 0.0003  # Below = dead market
    atr_too_high: float = 0.0015  # Above = trending
    
    # RSI for micro-fades
    rsi_period: int = 14
    rsi_extreme_high: float = 85.0  # Very extreme
    rsi_extreme_low: float = 15.0   # Very extreme
    
    # Risk Management
    max_risk_per_trade: float = 0.005  # 0.5% (half of normal)
    max_trades_per_week: int = 2  # Very limited
    
    # Position sizing
    micro_position_multiplier: float = 0.25  # 25% of normal size


@dataclass
class ChoppySignalResult:
    signal: ChoppySignal
    reason: str
    regime: str = "CHOPPY"
    suggested_size_multiplier: float = 0.0


class ChoppyEngine:
    """
    Choppy Market Engine for FX Majors.
    
    Core philosophy:
    - The market is ALWAYS right
    - Retail has NO edge in FX majors
    - Best trade is often NO trade
    - Only micro-fade extreme conditions
    """
    
    def __init__(self, config: ChoppyConfig = None):
        self.config = config or ChoppyConfig()
        self.position = None
        self.trades_this_week = 0
        self.current_week = None
    
    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute regime detection indicators."""
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        
        # ADX
        df['adx'] = self._calculate_adx(df)
        
        # Bollinger Bands
        df['bb_sma'] = df['close'].rolling(self.config.bb_period).mean()
        df['bb_std'] = df['close'].rolling(self.config.bb_period).std()
        df['bb_upper'] = df['bb_sma'] + (self.config.bb_std * df['bb_std'])
        df['bb_lower'] = df['bb_sma'] - (self.config.bb_std * df['bb_std'])
        
        # ATR
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift(1))
        tr3 = abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = tr.ewm(span=self.config.atr_period, adjust=False).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(span=self.config.rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(span=self.config.rsi_period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi'] = df['rsi'].fillna(50)
        
        # Shift to prevent look-ahead
        for col in ['adx', 'bb_sma', 'bb_upper', 'bb_lower', 'atr', 'rsi']:
            df[col] = df[col].shift(1)
        
        return df
    
    def _calculate_adx(self, df: pd.DataFrame) -> pd.Series:
        """Calculate ADX."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        period = self.config.adx_period
        atr = pd.Series(tr).ewm(span=period, adjust=False).mean()
        plus_di = 100 * pd.Series(plus_dm).ewm(span=period, adjust=False).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).ewm(span=period, adjust=False).mean() / atr
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.ewm(span=period, adjust=False).mean()
        
        return adx.fillna(25)
    
    def analyze_regime(self, df: pd.DataFrame) -> str:
        """Determine current market regime."""
        bar = df.iloc[-1]
        adx = bar['adx']
        atr = bar['atr']
        
        if pd.isna(adx) or pd.isna(atr):
            return "UNKNOWN"
        
        # Trending market - DO NOT TRADE
        if adx > 25:
            return "TRENDING"
        
        # Dead market - no volatility
        if atr < self.config.atr_too_low:
            return "DEAD"
        
        # High volatility but no trend - dangerous
        if atr > self.config.atr_too_high:
            return "VOLATILE"
        
        # True choppy - our territory (but still mostly block)
        return "CHOPPY"
    
    def generate_signal(self, df: pd.DataFrame, bar_time=None) -> ChoppySignalResult:
        """
        Generate trading signal.
        
        Default: BLOCK (do not trade)
        Only micro-trade at EXTREME conditions.
        """
        
        bar = df.iloc[-1]
        regime = self.analyze_regime(df)
        
        # Reset weekly counter
        if bar_time:
            week = bar_time.isocalendar()[1]
            if week != self.current_week:
                self.current_week = week
                self.trades_this_week = 0
        
        # Check weekly limit
        if self.trades_this_week >= self.config.max_trades_per_week:
            return ChoppySignalResult(
                signal=ChoppySignal.BLOCK,
                reason="Weekly trade limit reached",
                regime=regime
            )
        
        # BLOCK in non-choppy regimes
        if regime != "CHOPPY":
            return ChoppySignalResult(
                signal=ChoppySignal.BLOCK,
                reason=f"Regime is {regime}, not choppy",
                regime=regime
            )
        
        # Exit check
        if self.position is not None:
            close = bar['close']
            bb_sma = bar['bb_sma']
            
            # Exit when price returns to mean
            if self.position == "LONG" and close >= bb_sma:
                return ChoppySignalResult(
                    signal=ChoppySignal.EXIT,
                    reason="Price returned to mean",
                    regime=regime
                )
            
            if self.position == "SHORT" and close <= bb_sma:
                return ChoppySignalResult(
                    signal=ChoppySignal.EXIT,
                    reason="Price returned to mean",
                    regime=regime
                )
        
        # Already in position
        if self.position is not None:
            return ChoppySignalResult(
                signal=ChoppySignal.BLOCK,
                reason="Already in position",
                regime=regime
            )
        
        # MICRO-FADE only at EXTREME conditions
        rsi = bar['rsi']
        close = bar['close']
        bb_upper = bar['bb_upper']
        bb_lower = bar['bb_lower']
        
        # Extreme overbought
        if rsi > self.config.rsi_extreme_high and close > bb_upper:
            return ChoppySignalResult(
                signal=ChoppySignal.MICRO_SHORT,
                reason=f"Extreme overbought: RSI={rsi:.1f}",
                regime=regime,
                suggested_size_multiplier=self.config.micro_position_multiplier
            )
        
        # Extreme oversold
        if rsi < self.config.rsi_extreme_low and close < bb_lower:
            return ChoppySignalResult(
                signal=ChoppySignal.MICRO_LONG,
                reason=f"Extreme oversold: RSI={rsi:.1f}",
                regime=regime,
                suggested_size_multiplier=self.config.micro_position_multiplier
            )
        
        # Default: BLOCK
        return ChoppySignalResult(
            signal=ChoppySignal.BLOCK,
            reason="No extreme condition",
            regime=regime
        )
    
    def on_trade_opened(self, direction: str):
        """Record trade opened."""
        self.position = direction
        self.trades_this_week += 1
    
    def on_trade_closed(self):
        """Record trade closed."""
        self.position = None
    
    def reset(self):
        """Reset engine state."""
        self.position = None
        self.trades_this_week = 0
        self.current_week = None


# ============================================================================
# TEST
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print(" SCOPUS v3.0 — Choppy Engine (Strategy A)")
    print(" FX Majors Defense Strategy")
    print("=" * 60)
    
    config = ChoppyConfig()
    
    print(f"\n Configuration:")
    print(f"   ADX Threshold: {config.adx_choppy_threshold}")
    print(f"   RSI Extreme: {config.rsi_extreme_high}/{config.rsi_extreme_low}")
    print(f"   Max Trades/Week: {config.max_trades_per_week}")
    print(f"   Position Multiplier: {config.micro_position_multiplier}x")
    
    print(f"\n Philosophy:")
    print(f"   - Default action: BLOCK trades")
    print(f"   - Only micro-trade at extremes")
    print(f"   - Preserve capital, don't pursue returns")
    
    print(f"\n Status: READY")
