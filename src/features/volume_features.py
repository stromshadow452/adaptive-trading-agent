"""
src/features/volume_features.py
=================================
SCOPUS Volume Structure Features — Week 7 (Category G).

8 volume analysis features:
    vol2_obv_zscore      — On-Balance Volume z-score (20-bar)
    vol2_vwap_dist       — Distance from 20-bar VWAP in ATR units
    vol2_up_ratio        — Up-volume / total volume (20-bar ratio)
    vol2_pct_rank_5      — Volume percentile rank (5-bar window)
    vol2_pct_rank_20     — Volume percentile rank (20-bar window)
    vol2_vol_surge       — Binary: volume > 2× 20-bar average
    vol2_price_vol_corr  — Rolling corr(return, volume) 20 bars
    vol2_vwap_side       — Close relative to VWAP: +1=above, -1=below

Prefix vol2_ to avoid clash with volatility vol_* features.

Promotion status: SHADOW_ONLY
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List

VOLUME_FEATURE_LIST: List[str] = [
    "vol2_obv_zscore",
    "vol2_vwap_dist",
    "vol2_up_ratio",
    "vol2_pct_rank_5",
    "vol2_pct_rank_20",
    "vol2_vol_surge",
    "vol2_price_vol_corr",
    "vol2_vwap_side",
]


def _rolling_zscore(s: pd.Series, window: int) -> pd.Series:
    mu  = s.rolling(window, min_periods=window // 2).mean()
    sig = s.rolling(window, min_periods=window // 2).std().replace(0, np.nan)
    return ((s - mu) / sig).fillna(0.0)


def _rolling_pctile(s: pd.Series, window: int) -> pd.Series:
    def _rank(x):
        if len(x) < 2:
            return 0.5
        return float(np.sum(x[:-1] <= x[-1])) / max(len(x) - 1, 1)
    return s.rolling(window, min_periods=max(2, window // 2)).apply(_rank, raw=True)


def compute_volume_features(raw: pd.DataFrame, atr14: pd.Series) -> pd.DataFrame:
    """
    Compute 8 volume structure features.

    Args:
        raw  : OHLCV DataFrame with columns open/high/low/close/volume.
        atr14: ATR(14) series aligned with raw index.

    Returns:
        DataFrame with VOLUME_FEATURE_LIST columns.
    """
    f     = pd.DataFrame(index=raw.index)
    close = raw["close"]
    vol   = raw.get("volume", pd.Series(1.0, index=raw.index))
    atr   = atr14.replace(0, np.nan)
    ret   = close.pct_change(fill_method=None).fillna(0.0)

    # Clean volume
    vol = vol.replace([np.inf, -np.inf], np.nan).ffill().fillna(1.0)
    vol = vol.clip(lower=1e-8)

    # On-Balance Volume (OBV)
    direction = np.sign(ret).replace(0, 0)
    obv       = (direction * vol).cumsum()
    f["vol2_obv_zscore"]     = _rolling_zscore(obv, 20)

    # VWAP (20-bar, approximated using typical price)
    typical   = (raw["high"] + raw["low"] + close) / 3.0
    cum_tp_v  = (typical * vol).rolling(20, min_periods=10).sum()
    cum_v     = vol.rolling(20, min_periods=10).sum().replace(0, np.nan)
    vwap      = (cum_tp_v / cum_v)
    f["vol2_vwap_dist"]      = ((close - vwap) / atr).fillna(0.0).clip(-10, 10)
    f["vol2_vwap_side"]      = np.sign(close - vwap).fillna(0.0)

    # Up-volume ratio (volume on up-bars / total over 20 bars)
    up_vol   = (vol * (ret > 0).astype(float))
    up_sum   = up_vol.rolling(20, min_periods=10).sum()
    tot_sum  = vol.rolling(20, min_periods=10).sum().replace(0, np.nan)
    f["vol2_up_ratio"]       = (up_sum / tot_sum).fillna(0.5)

    # Volume percentile ranks
    f["vol2_pct_rank_5"]     = _rolling_pctile(vol, 5)
    f["vol2_pct_rank_20"]    = _rolling_pctile(vol, 20)

    # Volume surge flag
    avg_vol  = vol.rolling(20, min_periods=10).mean().replace(0, np.nan)
    f["vol2_vol_surge"]      = (vol > 2.0 * avg_vol).astype(float)

    # Rolling price-volume correlation (momentum confirmation or divergence)
    f["vol2_price_vol_corr"] = (
        ret.rolling(20, min_periods=10)
           .corr(vol)
           .fillna(0.0)
    )

    return f[VOLUME_FEATURE_LIST].fillna(0.0)
