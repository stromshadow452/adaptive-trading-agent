"""
src/features/session_features.py
=================================
SCOPUS Session & Calendar Features — Week 7 (Category E).

10 session/calendar features using cyclical encoding:
    sess_hour_sin, sess_hour_cos   — cyclical hour (24h)
    sess_dow_sin,  sess_dow_cos    — cyclical day-of-week (5 days)
    sess_is_london                 — 08:00–16:00 UTC
    sess_is_ny                     — 13:00–21:00 UTC
    sess_is_tokyo                  — 00:00–08:00 UTC
    sess_is_overlap                — London/NY overlap 13:00–16:00 UTC
    sess_is_monday                 — Monday open effect flag
    sess_is_friday_close           — Friday close effect flag

Cyclical encoding prevents the model seeing "hour 23" and "hour 0" as
far apart when they are actually adjacent.

Promotion status: SHADOW_ONLY
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List

SESSION_FEATURE_LIST: List[str] = [
    "sess_hour_sin",
    "sess_hour_cos",
    "sess_dow_sin",
    "sess_dow_cos",
    "sess_is_london",
    "sess_is_ny",
    "sess_is_tokyo",
    "sess_is_overlap",
    "sess_is_monday",
    "sess_is_friday_close",
]


def compute_session_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 10 session/calendar features from a DataFrame with DatetimeIndex.

    Args:
        df: Any DataFrame whose index is a DatetimeIndex (UTC preferred).
            Can be raw OHLCV or the base feature DataFrame.

    Returns:
        DataFrame with SESSION_FEATURE_LIST columns, same length as df.
        All values in [-1, 1] or {0, 1}.
    """
    f = pd.DataFrame(index=df.index)

    try:
        idx = pd.DatetimeIndex(df.index)
        hour = idx.hour.astype(float)
        dow  = idx.dayofweek.astype(float)   # 0=Mon…4=Fri

        # Cyclical hour encoding (24h period)
        f["sess_hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
        f["sess_hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

        # Cyclical day-of-week encoding (5 trading days)
        f["sess_dow_sin"]  = np.sin(2 * np.pi * dow / 5.0)
        f["sess_dow_cos"]  = np.cos(2 * np.pi * dow / 5.0)

        # Session flags (UTC-based)
        f["sess_is_london"]       = ((hour >= 8)  & (hour < 16)).astype(float)
        f["sess_is_ny"]           = ((hour >= 13) & (hour < 21)).astype(float)
        f["sess_is_tokyo"]        = ((hour >= 0)  & (hour < 8)).astype(float)
        f["sess_is_overlap"]      = ((hour >= 13) & (hour < 16)).astype(float)

        # Calendar effects
        f["sess_is_monday"]       = (dow == 0).astype(float)
        f["sess_is_friday_close"] = ((dow == 4) & (hour >= 17)).astype(float)

    except Exception:
        # Non-datetime index — return neutral values
        for col in SESSION_FEATURE_LIST:
            f[col] = 0.0

    return f[SESSION_FEATURE_LIST]
