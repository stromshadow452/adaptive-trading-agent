# src/regime_detector.py
"""
Regime detector CLI for Adaptive Trading Agent.

Usage:
    python -m src.regime_detector --csv data/datasets/EURUSD_H1_processed.csv --out reports/regime_eurusd_h1.csv
    python -m src.regime_detector --csv data/datasets/EURUSD_M15_processed.csv

What it does:
- Reads a CSV (auto-detects datetime column among common names)
- Ensures Datetime index (tz-aware)
- Computes simple regime features:
    * rolling volatility (ATR-like using close returns)
    * trend strength (ema slope ratio)
    * optional ADX-like proxy using rolling directional movement
- Assigns regime labels:
    - trending_high_vol
    - trending_low_vol
    - range_high_vol
    - range_low_vol
- Saves row-level regime column to output CSV and prints a short summary.

Dependencies: pandas, numpy
"""

from __future__ import annotations
import argparse
import os
import sys
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd


COMMON_DATE_COLS = ["Datetime", "datetime", "DATE", "Date", "date", "time", "Time"]


def find_datetime_column(df: pd.DataFrame) -> Optional[str]:
    # If index is already datetime-like, return None (we'll use index)
    if isinstance(df.index, pd.DatetimeIndex):
        return None
    for col in COMMON_DATE_COLS:
        if col in df.columns:
            return col
    # try any column with dtype object but parseable as datetime
    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna().astype(str).head(10)
            try:
                pd.to_datetime(sample, utc=True)
                return col
            except Exception:
                continue
    return None


def read_csv_flex(path: str, tz_aware: bool = True) -> pd.DataFrame:
    """
    Read CSV and auto-detect datetime column. Returns DataFrame with Datetime index.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    # read first 2000 rows fast to detect columns / dtypes
    df = pd.read_csv(path, low_memory=False)
    dt_col = find_datetime_column(df)
    if dt_col is None:
        # attempt to parse combined Date + Time columns
        if "Date" in df.columns and "Time" in df.columns:
            dt = df["Date"].astype(str).str.strip() + " " + df["Time"].astype(str).str.strip()
            df.insert(0, "Datetime", dt)
            dt_col = "Datetime"
        else:
            # if index-like CSV (first column is datetime)
            first_col = df.columns[0]
            try:
                parsed = pd.to_datetime(df[first_col], errors="coerce", utc=True)
                if parsed.notna().sum() > 0:
                    df.insert(0, "Datetime", parsed.astype(str))
                    dt_col = "Datetime"
            except Exception:
                dt_col = None

    if dt_col:
        # parse and set index
        df["Datetime"] = pd.to_datetime(df[dt_col], errors="coerce", utc=tz_aware)
        if df["Datetime"].isna().any():
            # try to coerce with dateutil fallback (already used by pandas)
            df["Datetime"] = pd.to_datetime(df[dt_col], errors="coerce", utc=tz_aware)
        # drop rows that couldn't be parsed
        n_bad = df["Datetime"].isna().sum()
        if n_bad:
            print(f"Warning: {n_bad} rows have NaT after datetime parsing — they will be dropped.")
        df = df.dropna(subset=["Datetime"])
        df = df.set_index("Datetime")
    else:
        # assume index already contains datetimes or no datetime available
        if not isinstance(df.index, pd.DatetimeIndex):
            # try to parse first column
            first_col = df.columns[0]
            try:
                parsed = pd.to_datetime(df[first_col], errors="coerce", utc=tz_aware)
                if parsed.notna().sum() > 0:
                    df.index = parsed
                    df.index.name = "Datetime"
                else:
                    raise ValueError("Could not find datetime column.")
            except Exception as e:
                raise ValueError(f"Failed to detect/parse datetime column: {e}")
    # ensure index is sorted
    df = df.sort_index()
    return df


def compute_atr_proxy(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    ATR-like using high-low and previous close.
    Requires columns: High, Low, Close (case-insensitive)
    Falls back to using pct returns std if OHLC not available.
    """
    cols = {c.lower(): c for c in df.columns}
    if {"high", "low", "close"}.issubset(set(cols.keys())):
        H = df[cols["high"]]
        L = df[cols["low"]]
        C = df[cols["close"]]
        tr1 = (H - L).abs()
        tr2 = (H - C.shift(1)).abs()
        tr3 = (L - C.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window, min_periods=1).mean()
        return atr
    else:
        # fallback: rolling std of log returns scaled
        close_col = None
        for k in ("close", "Close", "Close"):
            if k in df.columns:
                close_col = k
                break
        if close_col:
            logret = np.log(df[close_col]).diff().fillna(0)
            # scale to price by multiplying by current price magnitude (approx)
            atr_proxy = logret.rolling(window, min_periods=1).std().fillna(0)
            # convert to price-like metric by multiplying by mean price (approx)
            scale = df[close_col].abs().replace(0, 1).mean()
            return (atr_proxy * scale).fillna(0)
        else:
            # ultimate fallback: std of close values
            return df.iloc[:, 0].rolling(window, min_periods=1).std().fillna(0)


def compute_trend_strength(df: pd.DataFrame, span_short: int = 8, span_long: int = 21) -> pd.Series:
    """
    Simple trend strength measure: normalized EMA slope ratio.
    Returns a positive number if short EMA > long EMA and gives magnitude.
    """
    # pick close column
    close = None
    for c in df.columns:
        if c.lower() == "close":
            close = df[c]
            break
    if close is None:
        close = df.iloc[:, 0]  # fallback to first column
    ema_s = close.ewm(span=span_short, adjust=False).mean()
    ema_l = close.ewm(span=span_long, adjust=False).mean()
    diff = (ema_s - ema_l)
    # normalize by rolling std to get non-dimensional trend strength
    denom = close.rolling(span_long, min_periods=1).std().replace(0, np.nan)
    strength = (diff / denom).fillna(0).abs()
    # sign: positive if ema_s > ema_l else negative
    sign = np.where(ema_s >= ema_l, 1.0, -1.0)
    return pd.Series(strength * sign, index=df.index)


def compute_adx_proxy(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Lightweight ADX-like proxy (directional movement index).
    Uses high/low/close columns if present, otherwise returns trend_strength absolute.
    """
    cols = {c.lower(): c for c in df.columns}
    if {"high", "low", "close"}.issubset(set(cols.keys())):
        H = df[cols["high"]]
        L = df[cols["low"]]
        C = df[cols["close"]]
        up = H.diff()
        down = -L.diff()
        plus_dm = np.where((up > down) & (up > 0), up, 0.0)
        minus_dm = np.where((down > up) & (down > 0), down, 0.0)
        tr1 = (H - L).abs()
        tr2 = (H - C.shift(1)).abs()
        tr3 = (L - C.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        # smoothed moving average (Wilder)
        atr = tr.rolling(window, min_periods=1).mean()
        plus = pd.Series(plus_dm, index=df.index).rolling(window, min_periods=1).mean()
        minus = pd.Series(minus_dm, index=df.index).rolling(window, min_periods=1).mean()
        plus_di = 100 * (plus / (atr.replace(0, np.nan)))
        minus_di = 100 * (minus / (atr.replace(0, np.nan)))
        dx = 100 * (np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan))
        adx = dx.rolling(window, min_periods=1).mean().fillna(0)
        return adx
    else:
        # fallback: abs(trend_strength)
        return compute_trend_strength(df).abs()


def assign_regime_label(vol: pd.Series, trend: pd.Series, adx: pd.Series,
                        vol_thresh: float = None, trend_thresh: float = None, adx_thresh: float = None) -> pd.Series:
    """
    Assign regimes using thresholds.
    - vol: rolling volatility (price units)
    - trend: signed strength (positive => uptrend)
    - adx: trend strength proxy (0..100)
    """
    # automatic thresholds if None
    vol_med = vol.median() if vol_thresh is None else vol_thresh
    vol_high = vol.median() * 1.5
    if trend_thresh is None:
        trend_thresh = trend.abs().median() * 0.5
    if adx_thresh is None:
        adx_thresh = max(20.0, adx.median())  # ADX typical threshold ~20-25

    labels = []
    for v, t, a in zip(vol.values, trend.values, adx.values):
        trending = (abs(t) >= trend_thresh) or (a >= adx_thresh)
        high_vol = v >= vol_high
        if trending and high_vol:
            labels.append("trending_high_vol")
        elif trending and not high_vol:
            labels.append("trending_low_vol")
        elif (not trending) and high_vol:
            labels.append("range_high_vol")
        else:
            labels.append("range_low_vol")
    return pd.Series(labels, index=vol.index)


def summarize_labels(labels: pd.Series) -> Dict[str, Dict[str, float]]:
    total = len(labels)
    res = {}
    for lab, grp in labels.groupby(labels):
        res[lab] = {"count": int(len(grp)), "pct": float(len(grp) / total)}
    return {"total_bars": total, "by_regime": res}


def main():
    p = argparse.ArgumentParser(description="Regime detector (simple vol/trend/ADX proxy)")
    p.add_argument("--csv", required=True, help="Input CSV file (OHLCV or processed dataset)")
    p.add_argument("--out", required=False, help="Output CSV with regime column. If omitted, uses <in>_regimes.csv")
    p.add_argument("--vol_window", type=int, default=14, help="Window for volatility/ATR")
    p.add_argument("--trend_short", type=int, default=8, help="Short EMA span")
    p.add_argument("--trend_long", type=int, default=21, help="Long EMA span")
    p.add_argument("--adx_window", type=int, default=14, help="ADX proxy window")
    args = p.parse_args()

    in_csv = args.csv
    out_csv = args.out or (os.path.splitext(in_csv)[0] + "_regimes.csv")

    try:
        df = read_csv_flex(in_csv, tz_aware=True)
    except Exception as e:
        print("ERROR reading CSV:", e)
        sys.exit(2)

    # compute features
    try:
        atr = compute_atr_proxy(df, window=args.vol_window)
        trend = compute_trend_strength(df, span_short=args.trend_short, span_long=args.trend_long)
        adx = compute_adx_proxy(df, window=args.adx_window)
    except Exception as e:
        print("ERROR computing features:", e)
        sys.exit(3)

    # assign regimes
    regimes = assign_regime_label(atr, trend, adx)

    # attach columns and save
    out_df = df.copy()
    out_df["_atr"] = atr
    out_df["_trend_strength"] = trend
    out_df["_adx_proxy"] = adx
    out_df["regime"] = regimes

    try:
        out_df.to_csv(out_csv, index=True)
        print(f"Saved regimes -> {out_csv}")
    except Exception as e:
        print("ERROR saving output:", e)
        sys.exit(4)

    summary = summarize_labels(regimes)
    print("Summary:", summary)


if __name__ == "__main__":
    main()
