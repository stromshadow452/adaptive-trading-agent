# src/features/common_features.py
"""
Shared 20-feature builder (training + live) — exactly matches the model pipeline.

Output columns (order fixed):
['close','ret','sma5','sma20','sma_ratio','sma50','sma100','sma_ratio_long',
 'atr14','hl_range','body','ret_5','ret_20','rsi14','boll_z','atr_pct',
 'vol_norm','hod','dow','adx14']

NOTE: adx14 is computed using true Wilder's ADX (+DI/-DI) — NOT an ATR proxy.
"""

from __future__ import annotations
import re
from typing import List
import numpy as np
import pandas as pd

# Single source of truth for feature order
FEATURE_LIST: List[str] = [
    "close",
    "ret",
    "sma5",
    "sma20",
    "sma_ratio",
    "sma50",
    "sma100",
    "sma_ratio_long",
    "atr14",
    "hl_range",
    "body",
    "ret_5",
    "ret_20",
    "rsi14",
    "boll_z",
    "atr_pct",
    "vol_norm",
    "hod",
    "dow",
    "adx14",  # True Wilder's ADX — added for regime detection
]

def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize incoming frame to columns: open, high, low, close, volume; index=UTC datetime if available.
    Accepts many header styles and BOMs.
    """
    if df is None or df.empty:
        raise ValueError("Empty dataframe passed to _normalize_ohlcv")

    # map lowercase/stripped names back to original
    lut = {re.sub(r"[<>]", "", str(c)).replace("\ufeff", "").strip().lower(): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n in lut:
                return lut[n]
        return None

    c_open  = pick("open", "o", "bidopen", "askopen")
    c_high  = pick("high", "h", "bidhigh", "askhigh")
    c_low   = pick("low", "l", "bidlow", "asklow")
    c_close = pick("close", "c", "bidclose", "askclose", "last", "price", "closeprice")
    c_vol   = pick("volume", "vol", "tickvol", "tick_volume", "tickvolume", "v", "tick vol", "tickvol.")

    if c_open is None or c_high is None or c_low is None or c_close is None:
        raise ValueError(f"CSV missing OHLC columns; have={list(df.columns)}")

    def _col(name):
        """Return column as a plain 1-D Series (safe against duplicate cols)."""
        val = df[name]
        if isinstance(val, pd.DataFrame):
            val = val.iloc[:, 0]  # take first if duplicated
        return val.squeeze()

    out = pd.DataFrame(index=df.index.copy())
    out["open"]  = pd.to_numeric(_col(c_open),  errors="coerce")
    out["high"]  = pd.to_numeric(_col(c_high),  errors="coerce")
    out["low"]   = pd.to_numeric(_col(c_low),   errors="coerce")
    out["close"] = pd.to_numeric(_col(c_close), errors="coerce")
    out["volume"] = pd.to_numeric(_col(c_vol), errors="coerce") if c_vol else 1.0

    # Try to build datetime index if present
    date_col = None
    time_col = None
    ts_col = None
    for c in df.columns:
        cl = str(c).lower()
        if ("date" in cl) and date_col is None:
            date_col = c
        if (("time" in cl) or ("timestamp" in cl)) and time_col is None:
            time_col = c
        if cl == "timestamp":
            ts_col = c

    try:
        if date_col is not None and time_col is not None:
            dt = pd.to_datetime(df[date_col].astype(str) + " " + df[time_col].astype(str), errors="coerce", utc=True)
            out.index = dt
        elif ts_col is not None:
            out.index = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
    except Exception:
        # keep original index if parsing fails
        pass

    out = out.dropna(subset=["open", "high", "low", "close"])
    return out

def _compute_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average True Range from normalized OHLCV."""
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr = tr.rolling(window=window, min_periods=window).mean()
    return atr.bfill().fillna(0.0)


def calculate_wilder_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    True Wilder's ADX using correct +DI / -DI directional movement.

    This is the CORRECT implementation. It measures DIRECTIONAL STRENGTH,
    not volatility magnitude. ATR/price ratio is NOT a substitute for ADX.

    Args:
        df: DataFrame with 'high', 'low', 'close' columns (normalized)
        period: ADX period (Wilder default = 14)

    Returns:
        pd.Series of ADX values in range [0, 100]
    """
    high  = df["high"]
    low   = df["low"]
    close = df["close"]

    # True Range
    tr = pd.concat([
        (high - low).abs(),
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    # Raw Directional Movement
    up_move   = high.diff()
    down_move = -(low.diff())

    plus_dm  = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    # Wilder exponential smoothing (alpha = 1/period)
    def _wilder(s: pd.Series) -> pd.Series:
        return s.ewm(alpha=1.0 / period, adjust=False).mean()

    atr_w    = _wilder(tr)
    plus_di  = 100.0 * _wilder(plus_dm)  / (atr_w + 1e-10)
    minus_di = 100.0 * _wilder(minus_dm) / (atr_w + 1e-10)

    dx  = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx = _wilder(dx)

    return adx.clip(0, 100).bfill().fillna(0.0)

def compute_features_from_ohlcv(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the 19 features from raw OHLCV (any header style).
    Returns DataFrame with columns in FEATURE_LIST order.
    """
    df = _normalize_ohlcv(raw_df)

    feat = pd.DataFrame(index=df.index)
    close = df["close"]

    # Base price/ret
    feat["close"] = close
    feat["ret"] = close.pct_change(fill_method=None).fillna(0.0)

    # Simple moving averages (short)
    feat["sma5"]  = close.rolling(5,  min_periods=5).mean()
    feat["sma20"] = close.rolling(20, min_periods=20).mean()
    feat["sma_ratio"] = (feat["sma5"] / (feat["sma20"].replace(0, np.nan))).ffill().fillna(1.0)

    # Long MAs & ratio
    feat["sma50"]  = close.rolling(50,  min_periods=50).mean()
    feat["sma100"] = close.rolling(100, min_periods=100).mean()
    feat["sma_ratio_long"] = (feat["sma50"] / (feat["sma100"].replace(0, np.nan))).ffill().fillna(1.0)

    # Volatility / range
    feat["atr14"]    = _compute_atr(df, window=14)
    feat["hl_range"] = (df["high"] - df["low"]).abs()
    feat["body"]     = (close - df["open"]).abs()

    # Momentum-ish
    feat["ret_5"]  = close.pct_change(5,  fill_method=None).fillna(0.0)
    feat["ret_20"] = close.pct_change(20, fill_method=None).fillna(0.0)

    # RSI(14)
    up = close.diff().clip(lower=0)
    down = (-close.diff().clip(upper=0)).abs()
    avg_gain = up.rolling(14, min_periods=14).mean()
    avg_loss = down.rolling(14, min_periods=14).mean().replace(0, np.nan)
    rs = (avg_gain / avg_loss).replace([np.inf, -np.inf], np.nan)
    feat["rsi14"] = (100 - 100 / (1 + rs)).fillna(50)

    # Bollinger Z (20, 2)
    mid = close.rolling(20, min_periods=20).mean()
    std = close.rolling(20, min_periods=20).std()
    feat["boll_z"] = ((close - mid) / std.replace(0, np.nan)).fillna(0)

    # ATR as percent of price
    feat["atr_pct"] = (feat["atr14"] / close).clip(lower=1e-8)

    # Volume normalization (z over 20)
    v = df["volume"].replace([np.inf, -np.inf], np.nan).ffill().fillna(1.0)
    feat["vol_norm"] = (v - v.rolling(20, min_periods=1).mean()) / (
        v.rolling(20, min_periods=1).std().replace(0, np.nan)
    )
    feat["vol_norm"] = feat["vol_norm"].fillna(0.0)

    # Session features
    try:
        idx = pd.DatetimeIndex(feat.index)
        feat["hod"] = idx.hour.astype(float)
        feat["dow"] = idx.dayofweek.astype(float)
    except Exception:
        feat["hod"] = 0.0
        feat["dow"] = 0.0

    # True Wilder's ADX(14) — regime detection quality signal
    feat["adx14"] = calculate_wilder_adx(df, period=14)

    # Fill & order
    feat = feat.ffill().bfill()
    feat = feat.reindex(columns=FEATURE_LIST)
    return feat.fillna(0.0)
