"""
SCOPUS MTF System - M15 Executor

Tactical execution engine on M15 timeframe.
ONLY trades when H4 Brain grants permission.
ONLY trades in H4's direction.

Entry filters:
1. Pullback to EMA
2. RSI reset
3. Structure support/resistance
4. Minimum volatility
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
from enum import Enum

from src.backtest.h4_brain import H4Decision, Direction, Permission, Confidence


# =============================================================================
# ENTRY SIGNAL
# =============================================================================

@dataclass
class EntrySignal:
    """Entry signal from M15 Executor."""
    direction: Direction
    entry_price: float
    sl_price: float
    tp_price: float
    atr: float
    confidence: float
    reason: str
    timestamp: pd.Timestamp


# =============================================================================
# M15 EXECUTOR
# =============================================================================

class M15Executor:
    """
    Tactical executor operating on M15 timeframe.
    
    Rules:
    1. If H4 = BLOCK → silent (no signals)
    2. If H4 = BUY → only look for BUY entries
    3. If H4 = SELL → only look for SELL entries
    4. Never trade counter to H4 direction
    
    Entry conditions:
    - Price pulled back to EMA (not chasing)
    - RSI reset (not overbought/oversold)
    - Near structure support/resistance
    - Sufficient volatility
    """
    
    # Entry filters - TIGHTENED for quality
    PULLBACK_TOLERANCE = 0.002    # Within 0.2% of EMA (was 0.5%)
    RSI_BUY_MAX = 55              # Don't buy if RSI > 55 (was 60)
    RSI_SELL_MIN = 45             # Don't sell if RSI < 45 (was 40)
    MIN_ATR_RATIO = 0.8           # Minimum volatility (was 0.7)
    
    # Risk parameters - Better R:R
    SL_ATR_MULT = 1.5             # Tighter SL (was 2.0)
    TP_ATR_MULT = 3.0             # Same TP = 2.0 R:R (was 1.5 R:R)
    
    # Session filter (UTC hours)
    LONDON_START = 7
    NY_END = 21
    
    def __init__(self):
        self.last_signal: Optional[EntrySignal] = None
    
    def find_entry(
        self,
        h4_decision: H4Decision,
        m15_features: Dict[str, float],
        m15_df: pd.DataFrame,
        timestamp: pd.Timestamp,
    ) -> Optional[EntrySignal]:
        """
        Look for entry aligned with H4 direction.
        
        Returns None if no valid entry found.
        """
        # Rule 1: Check permission
        if h4_decision.permission != Permission.ALLOW:
            return None
        
        # Rule 2: Only trade during active sessions
        hour = timestamp.hour
        if hour < self.LONDON_START or hour > self.NY_END:
            return None
        
        direction = h4_decision.direction
        
        if direction == Direction.NO_TRADE:
            return None
        
        # Get current price data
        if len(m15_df) < 20:
            return None
        
        close = m15_df['close'].iloc[-1]
        ema_21 = m15_features.get('ema_21', close)
        rsi = m15_features.get('rsi_14', 50)
        atr = m15_features.get('atr_14', 0.001)
        atr_ratio = m15_features.get('atr_ratio_14_50', 1.0)
        
        # Check minimum volatility
        if atr_ratio < self.MIN_ATR_RATIO:
            return None
        
        # Direction-specific checks
        if direction == Direction.BUY:
            return self._check_buy_entry(
                close, ema_21, rsi, atr, m15_df, timestamp, h4_decision.confidence
            )
        elif direction == Direction.SELL:
            return self._check_sell_entry(
                close, ema_21, rsi, atr, m15_df, timestamp, h4_decision.confidence
            )
        
        return None
    
    def _check_buy_entry(
        self,
        close: float,
        ema_21: float,
        rsi: float,
        atr: float,
        m15_df: pd.DataFrame,
        timestamp: pd.Timestamp,
        h4_confidence: Confidence,
    ) -> Optional[EntrySignal]:
        """Check for valid BUY entry."""
        # Pullback check: price near or below EMA (not chasing)
        pullback_ratio = (close - ema_21) / ema_21
        
        if pullback_ratio > self.PULLBACK_TOLERANCE:
            return None  # Price too far above EMA, chasing
        
        # RSI check: not overbought
        if rsi > self.RSI_BUY_MAX:
            return None
        
        # Structure support: price above recent lows
        recent_low = m15_df['low'].rolling(20).min().iloc[-1]
        if close < recent_low:
            return None  # Breaking down, not a pullback
        
        # Valid entry
        entry_price = close
        sl_price = entry_price - self.SL_ATR_MULT * atr
        tp_price = entry_price + self.TP_ATR_MULT * atr
        
        # Confidence based on H4 + pullback quality
        confidence = 0.5
        if h4_confidence == Confidence.HIGH:
            confidence += 0.2
        if pullback_ratio < 0:  # Price below EMA = better pullback
            confidence += 0.1
        if rsi < 50:
            confidence += 0.1
        
        signal = EntrySignal(
            direction=Direction.BUY,
            entry_price=entry_price,
            sl_price=sl_price,
            tp_price=tp_price,
            atr=atr,
            confidence=confidence,
            reason="Pullback to EMA in H4 uptrend",
            timestamp=timestamp,
        )
        
        self.last_signal = signal
        return signal
    
    def _check_sell_entry(
        self,
        close: float,
        ema_21: float,
        rsi: float,
        atr: float,
        m15_df: pd.DataFrame,
        timestamp: pd.Timestamp,
        h4_confidence: Confidence,
    ) -> Optional[EntrySignal]:
        """Check for valid SELL entry."""
        # Pullback check: price near or above EMA
        pullback_ratio = (ema_21 - close) / ema_21
        
        if pullback_ratio > self.PULLBACK_TOLERANCE:
            return None  # Price too far below EMA
        
        # RSI check: not oversold
        if rsi < self.RSI_SELL_MIN:
            return None
        
        # Structure resistance: price below recent highs
        recent_high = m15_df['high'].rolling(20).max().iloc[-1]
        if close > recent_high:
            return None  # Breaking up, not a pullback
        
        # Valid entry
        entry_price = close
        sl_price = entry_price + self.SL_ATR_MULT * atr
        tp_price = entry_price - self.TP_ATR_MULT * atr
        
        # Confidence
        confidence = 0.5
        if h4_confidence == Confidence.HIGH:
            confidence += 0.2
        if pullback_ratio < 0:
            confidence += 0.1
        if rsi > 50:
            confidence += 0.1
        
        signal = EntrySignal(
            direction=Direction.SELL,
            entry_price=entry_price,
            sl_price=sl_price,
            tp_price=tp_price,
            atr=atr,
            confidence=confidence,
            reason="Pullback to EMA in H4 downtrend",
            timestamp=timestamp,
        )
        
        self.last_signal = signal
        return signal


# =============================================================================
# M15 FEATURE COMPUTER
# =============================================================================

class M15FeatureComputer:
    """Compute M15-specific features for execution."""
    
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute M15 features with 1-bar lag."""
        result = df.copy()
        
        # EMAs
        result['ema_8'] = df['close'].ewm(span=8, adjust=False).mean()
        result['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
        result['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        
        # ATR
        tr = pd.concat([
            df['high'] - df['low'],
            (df['high'] - df['close'].shift(1)).abs(),
            (df['low'] - df['close'].shift(1)).abs(),
        ], axis=1).max(axis=1)
        
        result['atr_14'] = tr.rolling(14).mean()
        result['atr_50'] = tr.rolling(50).mean()
        result['atr_ratio_14_50'] = result['atr_14'] / result['atr_50']
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        result['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Apply 1-bar lag
        feature_cols = ['ema_8', 'ema_21', 'ema_50', 'atr_14', 'atr_50', 'atr_ratio_14_50', 'rsi_14']
        for col in feature_cols:
            result[col] = result[col].shift(1)
        
        return result
    
    def get_features_at_bar(self, features_df: pd.DataFrame, idx: int) -> Dict[str, float]:
        """Get features at specific bar index."""
        if idx >= len(features_df):
            return {}
        
        row = features_df.iloc[idx]
        return {
            'ema_8': row.get('ema_8'),
            'ema_21': row.get('ema_21'),
            'ema_50': row.get('ema_50'),
            'atr_14': row.get('atr_14'),
            'atr_50': row.get('atr_50'),
            'atr_ratio_14_50': row.get('atr_ratio_14_50'),
            'rsi_14': row.get('rsi_14'),
        }


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    from src.backtest.h4_brain import H4Brain, Regime
    
    print("M15 Executor test")
    
    # Simulated H4 decision
    h4_decision = H4Decision(
        direction=Direction.BUY,
        permission=Permission.ALLOW,
        regime=Regime.TREND,
        confidence=Confidence.HIGH,
        trend_strength=1.8,
        timestamp=pd.Timestamp.now(),
    )
    
    # Simulated M15 features
    m15_features = {
        'ema_21': 1.0850,
        'rsi_14': 45,
        'atr_14': 0.0015,
        'atr_ratio_14_50': 1.1,
    }
    
    # Simulated M15 dataframe
    m15_df = pd.DataFrame({
        'close': [1.0845, 1.0848, 1.0852, 1.0849, 1.0847],
        'high': [1.0855, 1.0858, 1.0862, 1.0859, 1.0857],
        'low': [1.0840, 1.0843, 1.0847, 1.0844, 1.0842],
    })
    
    executor = M15Executor()
    signal = executor.find_entry(
        h4_decision,
        m15_features,
        m15_df,
        pd.Timestamp.now().replace(hour=10),  # London session
    )
    
    if signal:
        print(f"Entry: {signal.direction.value}")
        print(f"Price: {signal.entry_price:.5f}")
        print(f"SL: {signal.sl_price:.5f}")
        print(f"TP: {signal.tp_price:.5f}")
        print(f"Confidence: {signal.confidence:.2f}")
    else:
        print("No entry signal")
