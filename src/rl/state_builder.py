# src/rl/state_builder.py
from __future__ import annotations
from typing import Dict, Any, Sequence
import numpy as np
import pandas as pd

DEFAULT_FEATURES: Sequence[str] = (
    "close","volume","rsi","ema_fast","ema_slow","macd","macd_signal","macd_hist",
    "adx","di_plus","di_minus","atr","supertrend_dir","vol_ratio","trend_slope","clv",
)

def build_obs_from_row(row: pd.Series, feature_list: Sequence[str] = DEFAULT_FEATURES) -> np.ndarray:
    vals = []
    for k in feature_list:
        vals.append(float(row.get(k, 0.0)))
    return np.asarray(vals, dtype=np.float32)

def make_finrl_observation(df_with_inds: pd.DataFrame, lookback: int = 32,
                           feature_list: Sequence[str] = DEFAULT_FEATURES) -> np.ndarray:
    """
    Stack last `lookback` rows of features (time-major) → shape (lookback * F,).
    If not enough rows, left-pad with the first available row.
    """
    df = df_with_inds.copy()
    if len(df) == 0:
        return np.zeros((lookback * len(feature_list),), dtype=np.float32)

    tail = df.tail(lookback)
    if len(tail) < lookback:
        first = df.iloc[0]
        pad_rows = [first] * (lookback - len(tail))
        tail = pd.concat([pd.DataFrame([r]) for r in pad_rows] + [tail], ignore_index=True)

    frames = []
    for _, r in tail.iterrows():
        frames.append(build_obs_from_row(r, feature_list))
    mat = np.stack(frames, axis=0)  # (L, F)
    return mat.reshape(-1).astype(np.float32)

def to_single_step_obs(last_row: pd.Series, feature_list: Sequence[str] = DEFAULT_FEATURES) -> np.ndarray:
    return build_obs_from_row(last_row, feature_list).reshape(1, -1)
