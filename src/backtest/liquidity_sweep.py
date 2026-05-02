"""
Liquidity Sweep Detection

Detects when price sweeps a recent high/low (liquidity grab) and reverses.
This is a key Smart Money Concept (SMC) indicator for mean reversion.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class SweepResult:
    """Result of liquidity sweep detection."""
    swept_high: bool
    swept_low: bool
    sweep_direction: str  # 'BULL', 'BEAR', or 'NONE'
    sweep_distance_atr: float
    range_high: float
    range_low: float
    details: Dict[str, float]


class LiquiditySweepDetector:
    """
    Detects liquidity sweeps (stop hunts) in price data.
    
    A liquidity sweep occurs when:
    1. Price wick extends beyond a recent high/low
    2. Price closes back inside the range
    3. This triggers stops and creates reversal opportunity
    """
    
    def __init__(
        self,
        lookback: int = 20,
        sweep_threshold_atr: float = 0.2,
        close_inside_tolerance: float = 0.1
    ):
        """
        Args:
            lookback: Number of bars to look back for range
            sweep_threshold_atr: Minimum sweep distance as ATR multiple
            close_inside_tolerance: How far inside range close must be (ATR)
        """
        self.lookback = lookback
        self.sweep_threshold_atr = sweep_threshold_atr
        self.close_inside_tolerance = close_inside_tolerance
    
    def detect(
        self,
        high: float,
        low: float,
        close: float,
        atr: float,
        recent_highs: np.ndarray,
        recent_lows: np.ndarray
    ) -> SweepResult:
        """
        Detect liquidity sweep on current bar.
        
        Args:
            high: Current bar high
            low: Current bar low
            close: Current bar close
            atr: Current ATR value
            recent_highs: Array of recent high prices (excluding current)
            recent_lows: Array of recent low prices (excluding current)
            
        Returns:
            SweepResult with detection results
        """
        # Calculate range
        range_high = np.max(recent_highs)
        range_low = np.min(recent_lows)
        
        # Calculate sweep threshold
        sweep_min = self.sweep_threshold_atr * atr
        close_tolerance = self.close_inside_tolerance * atr
        
        # Detect HIGH sweep (bearish signal)
        # Wick above range high, close back inside
        high_exceeded = high > range_high + sweep_min
        close_below_high = close < range_high - close_tolerance
        swept_high = high_exceeded and close_below_high
        
        # Detect LOW sweep (bullish signal)
        # Wick below range low, close back inside
        low_exceeded = low < range_low - sweep_min
        close_above_low = close > range_low + close_tolerance
        swept_low = low_exceeded and close_above_low
        
        # Calculate sweep distance
        high_distance = (high - range_high) / atr if swept_high else 0
        low_distance = (range_low - low) / atr if swept_low else 0
        sweep_distance = max(high_distance, low_distance)
        
        # Determine direction
        if swept_low and not swept_high:
            direction = 'BULL'  # Swept lows = likely to go up
        elif swept_high and not swept_low:
            direction = 'BEAR'  # Swept highs = likely to go down
        elif swept_high and swept_low:
            direction = 'NONE'  # Both swept = unclear
        else:
            direction = 'NONE'
        
        return SweepResult(
            swept_high=swept_high,
            swept_low=swept_low,
            sweep_direction=direction,
            sweep_distance_atr=sweep_distance,
            range_high=range_high,
            range_low=range_low,
            details={
                'high_exceeded_by': (high - range_high) / atr if atr > 0 else 0,
                'low_exceeded_by': (range_low - low) / atr if atr > 0 else 0,
                'close_from_high': (range_high - close) / atr if atr > 0 else 0,
                'close_from_low': (close - range_low) / atr if atr > 0 else 0,
            }
        )
    
    def detect_from_df(
        self,
        df: pd.DataFrame,
        atr_col: str = 'atr_14'
    ) -> SweepResult:
        """
        Detect sweep from DataFrame (convenience method).
        
        Args:
            df: DataFrame with OHLC and ATR columns
            atr_col: Name of ATR column
            
        Returns:
            SweepResult for the last bar
        """
        if len(df) < self.lookback + 1:
            return SweepResult(
                swept_high=False,
                swept_low=False,
                sweep_direction='NONE',
                sweep_distance_atr=0,
                range_high=0,
                range_low=0,
                details={}
            )
        
        # Get current bar
        high = df['high'].iloc[-1]
        low = df['low'].iloc[-1]
        close = df['close'].iloc[-1]
        atr = df[atr_col].iloc[-1] if atr_col in df.columns else 0.001
        
        # Get recent bars (excluding current)
        recent_highs = df['high'].iloc[-(self.lookback + 1):-1].values
        recent_lows = df['low'].iloc[-(self.lookback + 1):-1].values
        
        return self.detect(high, low, close, atr, recent_highs, recent_lows)


class RangeStructureDetector:
    """
    Detects range structure and price position within range.
    """
    
    def __init__(self, lookback: int = 50):
        self.lookback = lookback
    
    def detect(
        self,
        close: float,
        atr: float,
        recent_highs: np.ndarray,
        recent_lows: np.ndarray
    ) -> Dict:
        """
        Detect range structure.
        
        Returns:
            Dict with range metrics
        """
        range_high = np.max(recent_highs)
        range_low = np.min(recent_lows)
        range_size = range_high - range_low
        
        # Normalize to ATR
        range_atr = range_size / atr if atr > 0 else 0
        
        # Price position (0 = at low, 1 = at high)
        if range_size > 0:
            position = (close - range_low) / range_size
        else:
            position = 0.5
        
        # Clamp position to 0-1
        position = max(0, min(1, position))
        
        # Determine zone
        if position < 0.33:
            zone = 'LOWER_THIRD'
        elif position > 0.67:
            zone = 'UPPER_THIRD'
        else:
            zone = 'MIDDLE_THIRD'
        
        # Range midpoint
        range_mid = (range_high + range_low) / 2
        
        return {
            'range_high': range_high,
            'range_low': range_low,
            'range_mid': range_mid,
            'range_size': range_size,
            'range_atr': range_atr,
            'position_in_range': position,
            'zone': zone,
        }
    
    def detect_from_df(self, df: pd.DataFrame, atr_col: str = 'atr_14') -> Dict:
        """Detect range from DataFrame."""
        if len(df) < self.lookback:
            return {
                'range_high': 0,
                'range_low': 0,
                'range_mid': 0,
                'range_size': 0,
                'range_atr': 0,
                'position_in_range': 0.5,
                'zone': 'MIDDLE_THIRD',
            }
        
        close = df['close'].iloc[-1]
        atr = df[atr_col].iloc[-1] if atr_col in df.columns else 0.001
        recent_highs = df['high'].iloc[-self.lookback:].values
        recent_lows = df['low'].iloc[-self.lookback:].values
        
        return self.detect(close, atr, recent_highs, recent_lows)


class FairValueGapDetector:
    """
    Detects Fair Value Gaps (FVGs) - imbalance zones from aggressive moves.
    """
    
    def __init__(self, min_gap_atr: float = 0.3):
        """
        Args:
            min_gap_atr: Minimum gap size as ATR multiple
        """
        self.min_gap_atr = min_gap_atr
    
    def detect_from_df(self, df: pd.DataFrame, atr_col: str = 'atr_14') -> Dict:
        """
        Detect FVG from last 3 candles.
        
        Returns:
            Dict with FVG detection results
        """
        if len(df) < 3:
            return {'bullish_fvg': False, 'bearish_fvg': False, 'gap_size_atr': 0}
        
        atr = df[atr_col].iloc[-1] if atr_col in df.columns else 0.001
        min_gap = self.min_gap_atr * atr
        
        # Candle 1 (oldest), 2 (middle), 3 (current)
        c1_high = df['high'].iloc[-3]
        c1_low = df['low'].iloc[-3]
        c3_high = df['high'].iloc[-1]
        c3_low = df['low'].iloc[-1]
        
        # Bullish FVG: Gap between C1 high and C3 low (price jumped up)
        bullish_gap = c3_low - c1_high
        bullish_fvg = bullish_gap > min_gap
        
        # Bearish FVG: Gap between C1 low and C3 high (price dropped)
        bearish_gap = c1_low - c3_high
        bearish_fvg = bearish_gap > min_gap
        
        gap_size = max(bullish_gap, bearish_gap) / atr if atr > 0 else 0
        
        return {
            'bullish_fvg': bullish_fvg,
            'bearish_fvg': bearish_fvg,
            'gap_size_atr': gap_size,
            'bullish_gap_atr': bullish_gap / atr if atr > 0 else 0,
            'bearish_gap_atr': bearish_gap / atr if atr > 0 else 0,
        }


# ============================================================================
# Convenience functions
# ============================================================================

def detect_sweep(df: pd.DataFrame, lookback: int = 20) -> SweepResult:
    """Quick sweep detection."""
    detector = LiquiditySweepDetector(lookback=lookback)
    return detector.detect_from_df(df)


def detect_range(df: pd.DataFrame, lookback: int = 50) -> Dict:
    """Quick range detection."""
    detector = RangeStructureDetector(lookback=lookback)
    return detector.detect_from_df(df)


def detect_fvg(df: pd.DataFrame) -> Dict:
    """Quick FVG detection."""
    detector = FairValueGapDetector()
    return detector.detect_from_df(df)
