"""
src/features/structure_features.py
====================================
SCOPUS Support / Resistance Proximity Features — Week 7 (Category F).

8 structure features measuring distance from key price levels:
    sr_dist_5d_high_atr    — distance from 5-day high in ATR units
    sr_dist_5d_low_atr     — distance from 5-day low in ATR units
    sr_dist_20d_high_atr   — distance from 20-day high in ATR units
    sr_dist_20d_low_atr    — distance from 20-day low in ATR units
    sr_pct_in_range_20     — % position in 20-day H-L range [0,1]
    sr_pct_in_range_5      — % position in 5-day H-L range [0,1]
    sr_high_proximity      — 1 if < 0.3 ATR from recent high
    sr_low_proximity       — 1 if < 0.3 ATR from recent low

All distances in ATR — asset-agnostic, no raw pip values.

Promotion status: SHADOW_ONLY
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List

STRUCTURE_FEATURE_LIST: List[str] = [
    "sr_dist_5d_high_atr",
    "sr_dist_5d_low_atr",
    "sr_dist_20d_high_atr",
    "sr_dist_20d_low_atr",
    "sr_pct_in_range_20",
    "sr_pct_in_range_5",
    "sr_high_proximity",
    "sr_low_proximity",
]

_PROXIMITY_ATR_THRESHOLD = 0.3   # within 0.3 ATR = "near" a level


def compute_structure_features(raw: pd.DataFrame, atr14: pd.Series) -> pd.DataFrame:
    """
    Compute 8 S/R proximity features.

    Args:
        raw  : OHLCV DataFrame with columns open/high/low/close.
        atr14: ATR(14) series aligned with raw index.

    Returns:
        DataFrame with STRUCTURE_FEATURE_LIST columns.
    """
    f     = pd.DataFrame(index=raw.index)
    close = raw["close"]
    high  = raw["high"]
    low   = raw["low"]
    atr   = atr14.replace(0, np.nan)

    # --- 5-day lookback S/R ---
    bars_5 = 5 * 24   # 5 days × 24 bars (H1)
    bars_5 = min(bars_5, len(raw))
    h5     = high.rolling(bars_5, min_periods=max(1, bars_5 // 4)).max()
    l5     = low.rolling(bars_5,  min_periods=max(1, bars_5 // 4)).min()
    range5 = (h5 - l5).replace(0, np.nan)

    f["sr_dist_5d_high_atr"] = ((h5 - close) / atr).fillna(0.0).clip(0, 10)
    f["sr_dist_5d_low_atr"]  = ((close - l5) / atr).fillna(0.0).clip(0, 10)
    f["sr_pct_in_range_5"]   = ((close - l5) / range5).fillna(0.5).clip(0, 1)

    # --- 20-day lookback S/R ---
    bars_20 = 20 * 24
    bars_20 = min(bars_20, len(raw))
    h20     = high.rolling(bars_20, min_periods=max(1, bars_20 // 4)).max()
    l20     = low.rolling(bars_20,  min_periods=max(1, bars_20 // 4)).min()
    range20 = (h20 - l20).replace(0, np.nan)

    f["sr_dist_20d_high_atr"] = ((h20 - close) / atr).fillna(0.0).clip(0, 10)
    f["sr_dist_20d_low_atr"]  = ((close - l20) / atr).fillna(0.0).clip(0, 10)
    f["sr_pct_in_range_20"]   = ((close - l20) / range20).fillna(0.5).clip(0, 1)

    # --- Proximity flags (binary) ---
    f["sr_high_proximity"] = (f["sr_dist_20d_high_atr"] < _PROXIMITY_ATR_THRESHOLD).astype(float)
    f["sr_low_proximity"]  = (f["sr_dist_20d_low_atr"]  < _PROXIMITY_ATR_THRESHOLD).astype(float)

    return f[STRUCTURE_FEATURE_LIST].fillna(0.0)
