"""
SCOPUS Data Layer - Multi-Timeframe Context Builder

Aligns multiple timeframes and computes derived features.
"""

from typing import List, Dict, Tuple, Optional
from datetime import timedelta
import pandas as pd
import numpy as np
import logging

from .types import Symbol, Timeframe, MTFFrame
from .store import MarketDataStore

logger = logging.getLogger(__name__)


# ============================================================================
# HTF CACHE - Avoid reloading D1/H4/H1 on every candle
# ============================================================================
# Key: (symbol.value, timeframe.value, date_str) -> DataFrame
_HTF_CACHE: Dict[Tuple[str, str, str], pd.DataFrame] = {}
_HTF_CACHE_STATS = {'hits': 0, 'misses': 0}


def _get_cached_htf(
    store: MarketDataStore,
    symbol: Symbol,
    tf: Timeframe,
    start,
    end,
) -> pd.DataFrame:
    """
    Get HTF data with caching.
    
    For D1: Load ~1 year of data once and cache.
    For H4/H1: Load ~1 month at a time and cache.
    Massive speedup for backtests (100x+ for D1).
    """
    from datetime import timedelta
    
    # Only cache higher timeframes (H1 and above)
    if tf not in [Timeframe.H1, Timeframe.H4, Timeframe.D1, Timeframe.W1]:
        return store.load_ohlcv(symbol, tf, start, end)
    
    # Determine cache granularity and load range
    if tf == Timeframe.D1:
        # D1: Cache by year - load 2 years of data to cover any backtest
        cache_start = start.replace(month=1, day=1) if hasattr(start, 'replace') else start
        cache_end = end.replace(month=12, day=31) if hasattr(end, 'replace') else end
        cache_key = (symbol.value, tf.value, str(cache_start.year) if hasattr(cache_start, 'year') else 'all')
    elif tf in [Timeframe.H4, Timeframe.H1]:
        # H4/H1: Cache by month
        cache_start = start.replace(day=1) if hasattr(start, 'replace') else start
        cache_end = (end + timedelta(days=31)).replace(day=1) if hasattr(end, 'replace') else end
        cache_key = (symbol.value, tf.value, f"{cache_start.year}-{cache_start.month:02d}" if hasattr(cache_start, 'year') else 'all')
    else:
        # W1: Cache all
        cache_start = start - timedelta(days=365*2)
        cache_end = end + timedelta(days=365)
        cache_key = (symbol.value, tf.value, 'all')
    
    if cache_key in _HTF_CACHE:
        _HTF_CACHE_STATS['hits'] += 1
        cached_df = _HTF_CACHE[cache_key]
        # Filter to requested range from cache
        try:
            return cached_df[(cached_df.index >= start) & (cached_df.index <= end)]
        except:
            return cached_df
    
    # Cache miss - load larger range and store
    _HTF_CACHE_STATS['misses'] += 1
    
    # Load the expanded range for caching
    df = store.load_ohlcv(symbol, tf, cache_start, cache_end)
    
    if not df.empty:
        _HTF_CACHE[cache_key] = df.copy()
        
        # Log cache stats periodically
        if _HTF_CACHE_STATS['misses'] <= 10 or _HTF_CACHE_STATS['misses'] % 50 == 0:
            hits = _HTF_CACHE_STATS['hits']
            misses = _HTF_CACHE_STATS['misses']
            ratio = hits / (hits + misses) * 100 if (hits + misses) > 0 else 0
            logger.info(f"HTF Cache [{tf.value}]: {hits} hits, {misses} misses ({ratio:.1f}% hit rate)")
    
    # Return filtered to requested range
    try:
        return df[(df.index >= start) & (df.index <= end)]
    except:
        return df


def clear_htf_cache():
    """Clear the HTF cache (call between symbols or at start of backtest)."""
    global _HTF_CACHE
    _HTF_CACHE = {}
    _HTF_CACHE_STATS['hits'] = 0
    _HTF_CACHE_STATS['misses'] = 0
    logger.info("HTF cache cleared")



def _pad_for_timeframe(tf: Timeframe) -> timedelta:
    """
    Calculate padding needed for indicator computation.
    
    Returns enough historical data to compute indicators like RSI(14), SMA(200), etc.
    """
    padding_map = {
        Timeframe.M1: timedelta(days=3),
        Timeframe.M5: timedelta(days=7),
        Timeframe.M15: timedelta(days=14),
        Timeframe.M30: timedelta(days=21),
        Timeframe.H1: timedelta(days=30),
        Timeframe.H4: timedelta(days=60),
        Timeframe.D1: timedelta(days=250),  # ~1 year for SMA(200)
        Timeframe.W1: timedelta(days=500),
        Timeframe.MN1: timedelta(days=1000),
    }
    return padding_map.get(tf, timedelta(days=30))


def build_mtf_context(
    store: MarketDataStore,
    symbol: Symbol,
    base_timeframe: Timeframe,
    higher_timeframes: List[Timeframe],
    start,
    end,
) -> MTFFrame:
    """
    Build multi-timeframe context frame.
    
    Creates a DataFrame indexed by base_timeframe with columns from all timeframes.
    Each higher timeframe is aligned using merge_asof (backward direction).
    
    Args:
        store: MarketDataStore instance
        symbol: Trading symbol
        base_timeframe: Base timeframe for index
        higher_timeframes: List of higher timeframes to include
        start: Start datetime (UTC)
        end: End datetime (UTC)
    
    Returns:
        MTFFrame with columns prefixed by timeframe (e.g., M5_open, H1_rsi_14)
    
    Example:
        >>> mtf = build_mtf_context(
        ...     store=store,
        ...     symbol=Symbol.EURUSD,
        ...     base_timeframe=Timeframe.M5,
        ...     higher_timeframes=[Timeframe.M15, Timeframe.H1, Timeframe.H4, Timeframe.D1],
        ...     start=datetime(2024, 1, 1),
        ...     end=datetime(2024, 12, 31),
        ... )
        >>> mtf.columns
        ['M5_open', 'M5_high', 'M5_low', 'M5_close', 'M5_volume',
         'M15_open', 'M15_close', 'H1_open', 'H1_close', 'D1_open', 'D1_close']
    """
    logger.info(f"Building MTF context for {symbol.value} {base_timeframe.value}")
    
    # 1. Load base timeframe
    base_df = store.load_ohlcv(symbol, base_timeframe, start, end)
    
    if base_df.empty:
        logger.warning(f"No base data for {symbol.value} {base_timeframe.value}")
        return pd.DataFrame()
    
    # Rename columns with base timeframe prefix
    base_df = base_df.rename(
        columns={col: f"{base_timeframe.value}_{col}" for col in base_df.columns if col != "source"}
    )
    
    # Drop source column (not needed in features)
    if f"{base_timeframe.value}_source" in base_df.columns:
        base_df = base_df.drop(columns=[f"{base_timeframe.value}_source"])
    
    # Initialize context frame
    ctx_df = base_df.copy()
    
    # 2. Align higher timeframes
    for tf in higher_timeframes:
        logger.debug(f"Aligning {tf.value}...")
        
        # Load with padding for indicators
        padded_start = start - _pad_for_timeframe(tf)
        
        # Use cached loader for HTF (D1/H4/H1) - 100x+ speedup
        tf_df = _get_cached_htf(store, symbol, tf, padded_start, end)
        
        if tf_df.empty:
            logger.debug(f"No data for {symbol.value} {tf.value}, skipping")
            continue

        
        # Rename columns with timeframe prefix (excluding source)
        tf_df = tf_df.rename(
            columns={col: f"{tf.value}_{col}" for col in tf_df.columns if col != "source"}
        )
        
        # Drop source column to avoid merge conflicts
        if "source" in tf_df.columns:
            tf_df = tf_df.drop(columns=["source"])
        if f"{tf.value}_source" in tf_df.columns:
            tf_df = tf_df.drop(columns=[f"{tf.value}_source"])
        
        # Merge asof: for each base timestamp, get the last higher-TF bar
        ctx_df = pd.merge_asof(
            left=ctx_df.reset_index().sort_values("timestamp"),
            right=tf_df.reset_index().sort_values("timestamp"),
            on="timestamp",
            direction="backward",
        ).set_index("timestamp").sort_index()
    
    logger.info(f"MTF context built: {len(ctx_df)} rows, {len(ctx_df.columns)} columns")
    
    return ctx_df


def add_mtf_features(
    df: MTFFrame,
    base_timeframe: Timeframe,
    higher_timeframes: List[Timeframe],
) -> MTFFrame:
    """
    Add derived features to MTF context frame.
    
    Computes technical indicators for each timeframe:
    - RSI(14)
    - ATR(14)
    - ADX(14)
    - ATR percentile
    - Bollinger Z-score
    - Trend flag (simple SMA crossover)
    - Volatility regime
    
    Args:
        df: MTF context frame from build_mtf_context()
        base_timeframe: Base timeframe
        higher_timeframes: List of higher timeframes
    
    Returns:
        MTFFrame with additional feature columns
    """
    logger.info("Adding MTF features...")
    
    all_timeframes = [base_timeframe] + higher_timeframes
    
    for tf in all_timeframes:
        close_col = f"{tf.value}_close"
        
        if close_col not in df.columns:
            continue
        
        # RSI(14)
        rsi_col = f"{tf.value}_rsi_14"
        df[rsi_col] = compute_rsi(df[close_col], period=14)
        
        # ATR(14)
        if f"{tf.value}_high" in df.columns and f"{tf.value}_low" in df.columns:
            atr_col = f"{tf.value}_atr_14"
            df[atr_col] = compute_atr(
                df[f"{tf.value}_high"],
                df[f"{tf.value}_low"],
                df[close_col],
                period=14
            )
            atr_pctile_col = f"{tf.value}_atr_pctile"
            df[atr_pctile_col] = compute_atr_percentile(df[atr_col], window=100)

            adx_col = f"{tf.value}_adx_14"
            df[adx_col] = compute_adx(
                df[f"{tf.value}_high"],
                df[f"{tf.value}_low"],
                df[close_col],
                period=14
            )

        bb_middle_col = f"{tf.value}_bb_middle"
        bb_std_col = f"{tf.value}_bb_std"
        boll_z_col = f"{tf.value}_boll_z"
        rolling_mean = df[close_col].rolling(window=20).mean()
        rolling_std = df[close_col].rolling(window=20).std()
        df[bb_middle_col] = rolling_mean
        df[bb_std_col] = rolling_std
        df[boll_z_col] = compute_bollinger_zscore(df[close_col], rolling_mean, rolling_std)
        
        # Trend flag (SMA crossover)
        trend_col = f"{tf.value}_trend_flag"
        df[trend_col] = compute_trend_flag(df[close_col])

        # Sanitize indicator columns so downstream routing gets real floats, not NaN.
        for feature_col in [rsi_col, f"{tf.value}_atr_14", f"{tf.value}_atr_pctile", f"{tf.value}_adx_14", boll_z_col]:
            if feature_col in df.columns:
                df[feature_col] = (
                    pd.to_numeric(df[feature_col], errors="coerce")
                    .replace([np.inf, -np.inf], np.nan)
                    .ffill()
                    .bfill()
                )
    
    logger.info(f"Features added: {len(df.columns)} total columns")
    
    return df


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI indicator"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Compute Average True Range"""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    return atr


def compute_atr_percentile(atr: pd.Series, window: int = 100) -> pd.Series:
    """Compute rolling percentile rank of ATR in [0, 1]."""
    def _pct_rank(values: np.ndarray) -> float:
        valid = values[~np.isnan(values)]
        if len(valid) == 0:
            return np.nan
        current = valid[-1]
        return float(np.mean(valid <= current))

    return atr.rolling(window=window, min_periods=max(20, window // 5)).apply(_pct_rank, raw=True)


def compute_bollinger_zscore(close: pd.Series, middle: pd.Series, std: pd.Series) -> pd.Series:
    """Compute Bollinger Z-score using a 20-period mean and std."""
    denom = std.replace(0, np.nan)
    return ((close - middle) / denom).replace([np.inf, -np.inf], np.nan)


def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Compute Wilder-style ADX(14) in the 0-100 range."""
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    tr = pd.concat(
        [
            (high - low).abs(),
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.rolling(window=period, min_periods=period).mean()
    plus_di = 100 * plus_dm.rolling(window=period, min_periods=period).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.rolling(window=period, min_periods=period).mean() / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.rolling(window=period, min_periods=period).mean().clip(lower=0, upper=100)


def compute_trend_flag(close: pd.Series) -> pd.Series:
    """
    Simple trend flag based on SMA crossover.
    
    Returns:
        1 = uptrend (fast > slow)
        -1 = downtrend (fast < slow)
        0 = neutral
    """
    sma_fast = close.rolling(window=20).mean()
    sma_slow = close.rolling(window=50).mean()
    
    trend = pd.Series(0, index=close.index)
    trend[sma_fast > sma_slow] = 1
    trend[sma_fast < sma_slow] = -1
    
    return trend
