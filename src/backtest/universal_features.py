"""
SCOPUS Phase-1: Universal Features

Asset-agnostic feature computation using Z-scores, ATR normalization,
and percentile ranking. The agent never sees raw prices.

Design Principles:
- All price-based features normalized to Z-scores or ATR multiples
- All distances measured in ATR units
- All momentum features percentile-ranked or Z-scored
- Session time encoded cyclically (sin/cos)
- No symbol names, no raw prices, no absolute values
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def zscore(series: pd.Series, lookback: int = 50) -> pd.Series:
    """
    Compute rolling Z-score.
    Returns: How many standard deviations from rolling mean.
    """
    mean = series.rolling(lookback, min_periods=10).mean()
    std = series.rolling(lookback, min_periods=10).std()
    return (series - mean) / std.replace(0, np.nan)


def percentile_rank(series: pd.Series, lookback: int = 100) -> pd.Series:
    """
    Compute rolling percentile rank (0 to 1).
    Returns: Where current value ranks in lookback window.
    """
    def _pct_rank(x):
        if len(x) < 2:
            return 0.5
        return (x.argsort().argsort()[-1]) / (len(x) - 1)
    return series.rolling(lookback, min_periods=10).apply(_pct_rank, raw=False)


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=span, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(period).mean()


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range."""
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (0-100)."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def roc(series: pd.Series, period: int = 10) -> pd.Series:
    """Rate of Change (percentage)."""
    return series.pct_change(period) * 100


def stoch_k(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Stochastic K (0-100)."""
    low_min = df['low'].rolling(period).min()
    high_max = df['high'].rolling(period).max()
    denom = (high_max - low_min).replace(0, np.nan)
    return ((df['close'] - low_min) / denom) * 100


def bb_width(df: pd.DataFrame, period: int = 20, num_std: float = 2.0) -> pd.Series:
    """Bollinger Band width as percentage of price."""
    mid = sma(df['close'], period)
    std = df['close'].rolling(period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return (upper - lower) / mid * 100


def macd_histogram(series: pd.Series) -> pd.Series:
    """MACD histogram."""
    ema12 = ema(series, 12)
    ema26 = ema(series, 26)
    macd_line = ema12 - ema26
    signal = ema(macd_line, 9)
    return macd_line - signal


# ============================================================================
# UNIVERSAL FEATURE COMPUTER
# ============================================================================

class UniversalFeatureComputer:
    """
    Compute asset-agnostic features.
    
    The agent sees:
    - Z-scores (trend, momentum, structure)
    - ATR-normalized distances
    - Percentile ranks
    - Cyclical time encoding
    
    The agent NEVER sees:
    - Raw prices
    - Symbol names
    - Pip values
    - Absolute distances
    """
    
    # Features that will be computed
    FEATURE_LIST = [
        # Trend features
        'trend_20_zscore', 'trend_50_zscore', 'trend_200_zscore',
        'ema_cross_8_21', 'ema_cross_21_50',
        'slope_20_atr', 'slope_50_atr',
        
        # Momentum features
        'rsi_14', 'rsi_zscore',
        'stoch_k', 'stoch_zscore',
        'roc_10_zscore', 'roc_20_zscore',
        'macd_hist_atr',
        
        # Volatility features
        'atr_ratio_14_50', 'atr_expansion',
        'bb_width_pct', 'bb_width_zscore',
        'range_ratio',
        
        # Structure features
        'price_position_20', 'price_position_50',
        'dist_high_atr', 'dist_low_atr',
        
        # Session features
        'hour_sin', 'hour_cos',
        'dow_sin', 'dow_cos',
        'is_london', 'is_ny', 'is_overlap',
    ]
    
    def __init__(self, lookback: int = 50):
        """
        Args:
            lookback: Default lookback for Z-scores and percentiles
        """
        self.lookback = lookback
    
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all universal features.
        
        Args:
            df: OHLCV DataFrame with 'open', 'high', 'low', 'close', 'volume'
                Index should be DatetimeIndex.
        
        Returns:
            DataFrame with normalized features (no raw prices)
        """
        f = pd.DataFrame(index=df.index)
        
        # Cache common calculations
        close = df['close']
        ema_20 = ema(close, 20)
        ema_50 = ema(close, 50)
        ema_200 = ema(close, 200)
        atr_14 = atr(df, 14)
        atr_50 = atr(df, 50)
        rsi_14 = rsi(close, 14)
        
        # === TREND FEATURES (Z-scored) ===
        f['trend_20_zscore'] = zscore(close - ema_20, self.lookback) / atr_14 * atr_14.mean()
        f['trend_50_zscore'] = zscore(close - ema_50, self.lookback) / atr_14 * atr_14.mean()
        f['trend_200_zscore'] = zscore(close - ema_200, self.lookback) / atr_14 * atr_14.mean()
        
        # EMA crosses (distance in ATR)
        f['ema_cross_8_21'] = (ema(close, 8) - ema(close, 21)) / atr_14
        f['ema_cross_21_50'] = (ema(close, 21) - ema_50) / atr_14
        
        # Slope (change over 5 bars normalized by ATR)
        f['slope_20_atr'] = (ema_20 - ema_20.shift(5)) / atr_14
        f['slope_50_atr'] = (ema_50 - ema_50.shift(10)) / atr_14
        
        # === MOMENTUM FEATURES ===
        f['rsi_14'] = rsi_14 / 100  # Normalize to 0-1
        f['rsi_zscore'] = zscore(rsi_14, self.lookback)
        
        stoch = stoch_k(df, 14)
        f['stoch_k'] = stoch / 100  # Normalize to 0-1
        f['stoch_zscore'] = zscore(stoch, self.lookback)
        
        f['roc_10_zscore'] = zscore(roc(close, 10), self.lookback)
        f['roc_20_zscore'] = zscore(roc(close, 20), self.lookback)
        
        macd_hist = macd_histogram(close)
        f['macd_hist_atr'] = macd_hist / atr_14
        
        # === VOLATILITY FEATURES ===
        f['atr_ratio_14_50'] = atr_14 / atr_50  # Current vs historical vol
        f['atr_expansion'] = (atr(df, 5) - atr(df, 20)) / atr(df, 20)  # Vol expansion
        
        bb_w = bb_width(df, 20)
        f['bb_width_pct'] = percentile_rank(bb_w, 100)
        f['bb_width_zscore'] = zscore(bb_w, self.lookback)
        
        f['range_ratio'] = (df['high'] - df['low']) / atr_14  # Bar range vs ATR
        
        # === STRUCTURE FEATURES ===
        # Position within range (0 = at low, 1 = at high)
        high_20 = df['high'].rolling(20).max()
        low_20 = df['low'].rolling(20).min()
        range_20 = (high_20 - low_20).replace(0, np.nan)
        f['price_position_20'] = (close - low_20) / range_20
        
        high_50 = df['high'].rolling(50).max()
        low_50 = df['low'].rolling(50).min()
        range_50 = (high_50 - low_50).replace(0, np.nan)
        f['price_position_50'] = (close - low_50) / range_50
        
        # Distance to swing high/low in ATR
        f['dist_high_atr'] = (high_50 - close) / atr_14
        f['dist_low_atr'] = (close - low_50) / atr_14
        
        # === SESSION FEATURES (Cyclical Encoding) ===
        if isinstance(df.index, pd.DatetimeIndex):
            hour = df.index.hour
            f['hour_sin'] = np.sin(2 * np.pi * hour / 24)
            f['hour_cos'] = np.cos(2 * np.pi * hour / 24)
            
            dow = df.index.dayofweek
            f['dow_sin'] = np.sin(2 * np.pi * dow / 5)
            f['dow_cos'] = np.cos(2 * np.pi * dow / 5)
            
            # Session flags (UTC-based)
            f['is_london'] = ((hour >= 8) & (hour < 16)).astype(float)
            f['is_ny'] = ((hour >= 13) & (hour < 21)).astype(float)
            f['is_overlap'] = ((hour >= 13) & (hour < 16)).astype(float)
        else:
            # Default values if no datetime index
            f['hour_sin'] = 0.0
            f['hour_cos'] = 1.0
            f['dow_sin'] = 0.0
            f['dow_cos'] = 1.0
            f['is_london'] = 0.0
            f['is_ny'] = 0.0
            f['is_overlap'] = 0.0
        
        # Clip extreme values
        f = f.clip(-10, 10)
        
        # Fill NaN with 0 (neutral)
        f = f.fillna(0)
        
        return f
    
    def compute_single(self, df: pd.DataFrame, idx: int = -1) -> Dict[str, float]:
        """
        Compute features for a single bar.
        
        Args:
            df: Full OHLCV DataFrame
            idx: Index of bar (-1 for latest)
        
        Returns:
            Dict of feature name -> value
        """
        features_df = self.compute(df)
        return features_df.iloc[idx].to_dict()


# ============================================================================
# POSITION SIZING NORMALIZER
# ============================================================================

class UniversalPositionSizer:
    """
    ATR-based position sizing that works for any asset.
    
    Size is determined by:
    1. Account risk % (e.g., 1%)
    2. Distance to SL in ATR
    3. Asset class adjustments
    """
    
    # Asset class size multipliers
    CLASS_MULTIPLIERS = {
        'FX_MAJOR': 1.0,
        'FX_VOLATILE': 0.7,
        'CRYPTO_MAJOR': 0.3,
        'INDEX': 0.8,
        'UNKNOWN': 0.5,
    }
    
    def __init__(self, base_risk_pct: float = 0.01):
        """
        Args:
            base_risk_pct: Base risk per trade (e.g., 0.01 = 1%)
        """
        self.base_risk_pct = base_risk_pct
    
    def calculate_size(
        self,
        equity: float,
        atr: float,
        sl_atr_mult: float,
        price: float,
        pip_value: float = 1.0,
        asset_class: str = 'FX_MAJOR',
        quality_mult: float = 1.0,
    ) -> Tuple[float, float]:
        """
        Calculate position size in lots.
        
        Args:
            equity: Current account equity
            atr: Current ATR value
            sl_atr_mult: SL distance as ATR multiple
            price: Current price (for position value calc)
            pip_value: Value per pip/point
            asset_class: Asset class for size adjustment
            quality_mult: Quality scorer multiplier (0-1.15)
        
        Returns:
            (position_size, risk_amount)
        """
        # Base risk amount
        risk_amount = equity * self.base_risk_pct
        
        # Apply asset class adjustment
        class_mult = self.CLASS_MULTIPLIERS.get(asset_class, 0.5)
        
        # Apply quality adjustment
        adjusted_risk = risk_amount * class_mult * quality_mult
        
        # SL distance in price terms
        sl_distance = atr * sl_atr_mult
        
        # Position size (simplified - actual calc depends on instrument)
        position_size = adjusted_risk / (sl_distance * pip_value)
        
        return max(0.01, round(position_size, 2)), adjusted_risk


# ============================================================================
# SL/TP NORMALIZER
# ============================================================================

class UniversalSLTPCalculator:
    """
    ATR-based SL/TP that works for any asset.
    No fixed pip values - everything in ATR multiples.
    """
    
    # Default ATR multiples by asset class
    DEFAULT_PARAMS = {
        'FX_MAJOR': {'sl_atr': 1.5, 'tp_atr': 2.5, 'min_rr': 1.5},
        'FX_VOLATILE': {'sl_atr': 2.0, 'tp_atr': 3.0, 'min_rr': 1.5},
        'CRYPTO_MAJOR': {'sl_atr': 2.5, 'tp_atr': 4.0, 'min_rr': 1.5},
        'INDEX': {'sl_atr': 1.5, 'tp_atr': 2.5, 'min_rr': 1.5},
        'UNKNOWN': {'sl_atr': 2.0, 'tp_atr': 3.0, 'min_rr': 1.5},
    }
    
    def calculate(
        self,
        entry_price: float,
        atr: float,
        direction: str,  # 'BUY' or 'SELL'
        asset_class: str = 'FX_MAJOR',
        override_sl_atr: float = None,
        override_tp_atr: float = None,
    ) -> Dict[str, float]:
        """
        Calculate SL and TP levels.
        
        Args:
            entry_price: Entry price
            atr: Current ATR
            direction: Trade direction
            asset_class: For default parameters
            override_sl_atr: Override default SL ATR multiple
            override_tp_atr: Override default TP ATR multiple
        
        Returns:
            Dict with 'sl', 'tp', 'sl_atr', 'tp_atr', 'rr_ratio'
        """
        params = self.DEFAULT_PARAMS.get(asset_class, self.DEFAULT_PARAMS['UNKNOWN'])
        
        sl_atr = override_sl_atr or params['sl_atr']
        tp_atr = override_tp_atr or params['tp_atr']
        
        sl_distance = atr * sl_atr
        tp_distance = atr * tp_atr
        
        if direction == 'BUY':
            sl = entry_price - sl_distance
            tp = entry_price + tp_distance
        else:  # SELL
            sl = entry_price + sl_distance
            tp = entry_price - tp_distance
        
        return {
            'sl': sl,
            'tp': tp,
            'sl_distance': sl_distance,
            'tp_distance': tp_distance,
            'sl_atr': sl_atr,
            'tp_atr': tp_atr,
            'rr_ratio': tp_atr / sl_atr,
        }


# ============================================================================
# FEATURE VALIDATOR
# ============================================================================

def validate_no_price_leakage(features: Dict[str, float]) -> bool:
    """
    Validate that no raw prices leaked into features.
    
    Returns True if clean, raises ValueError if contaminated.
    """
    FORBIDDEN_PATTERNS = ['close', 'open', 'high', 'low', 'price', 'pip']
    
    for key, value in features.items():
        # Check for forbidden names
        for pattern in FORBIDDEN_PATTERNS:
            if pattern in key.lower() and 'position' not in key.lower():
                if 'zscore' not in key and 'atr' not in key and 'ratio' not in key:
                    raise ValueError(f"Potential price leakage in feature: {key}")
        
        # Check for values that look like prices (large absolute numbers)
        if isinstance(value, (int, float)):
            if abs(value) > 100 and 'pct' not in key:
                # Could be a raw price - suspicious
                pass  # Warning only, don't block
    
    return True


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def compute_universal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to compute universal features.
    
    Args:
        df: OHLCV DataFrame
    
    Returns:
        Normalized feature DataFrame
    """
    computer = UniversalFeatureComputer()
    return computer.compute(df)


def get_feature_for_bar(df: pd.DataFrame, bar_idx: int = -1) -> Dict[str, float]:
    """
    Get features for a specific bar.
    
    Args:
        df: OHLCV DataFrame
        bar_idx: Bar index (-1 for latest)
    
    Returns:
        Feature dict
    """
    computer = UniversalFeatureComputer()
    return computer.compute_single(df, bar_idx)
