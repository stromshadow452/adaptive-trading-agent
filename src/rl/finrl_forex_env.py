#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simple FinRL-style Forex environment with 3 actions: 0=FLAT, 1=LONG, 2=SHORT.
- Loads multi-timeframe CSVs standardized as: {SYMBOL}_{TF}.csv with a 'Date' column.
- Builds technical features on fast/base/slow frames and joins onto base index.
- Reward = position * return - trade_cost - regularizers.
- Regularizers:
    * pos_bias_penalty: per-bar penalty when in a position (reduces "always-in-position" drift)
    * flat_penalty: per-bar penalty when flat (keep 0.0 unless you want to discourage flat)
"""

from __future__ import annotations

import os
import math
import numpy as np
import pandas as pd

# Gym/Gymnasium compatibility
try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # pragma: no cover
    import gym
    from gym import spaces


# ---------- Utilities ----------

def _load_tf_csv(data_root: str, symbol: str, tf: str) -> pd.DataFrame:
    """
    Load a standardized timeframe CSV:
      - path: f"{data_root}/{SYMBOL}_{TF}.csv"
      - must contain 'Date' column; others flexible but need 'Close'.
    """
    p = os.path.join(str(data_root), f"{symbol.upper()}_{tf.upper()}.csv")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    df = pd.read_csv(p)
    if "Date" not in df.columns:
        raise ValueError(f"{p} must contain a 'Date' column")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True)
    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
    # normalize column names
    ren = {c: c.capitalize() for c in df.columns}
    df = df.rename(columns=ren)
    if "Close" not in df.columns:
        # try lowercase
        if "close" in df.columns:
            df = df.rename(columns={"close": "Close"})
        else:
            raise ValueError(f"{p} missing 'Close' column")
    # Keep numeric cols only to avoid merges choking
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Close"])
    return df


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.clip(lower=0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / (loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.ffill().bfill().fillna(50.0)


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    # requires High/Low/Close (if not present, approximate with Close)
    h = df["High"] if "High" in df.columns else df["Close"]
    l = df["Low"] if "Low" in df.columns else df["Close"]
    c = df["Close"]
    prev_c = c.shift(1)
    tr = pd.concat([
        (h - l).abs(),
        (h - prev_c).abs(),
        (l - prev_c).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    return atr.ffill().bfill()


def _feats(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Compute a compact feature set on a given timeframe."""
    out = pd.DataFrame(index=df.index)
    out[f"{prefix}ret_1"]  = df["Close"].pct_change().fillna(0.0)
    out[f"{prefix}ma_5"]   = df["Close"].rolling(5).mean().pct_change().fillna(0.0)
    out[f"{prefix}ma_20"]  = df["Close"].rolling(20).mean().pct_change().fillna(0.0)
    out[f"{prefix}rsi_14"] = _rsi(df["Close"], 14) / 100.0  # scale to [0,1]
    out[f"{prefix}atr_14"] = (_atr(df, 14) / df["Close"]).clip(upper=0.2).fillna(0.0)
    return out


# ---------- Environment ----------

class ForexEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        data_root: str,
        symbol: str,
        tf_fast: str,
        tf_base: str,
        tf_slow: str,
        start: str | None = None,
        end: str | None = None,
        cost_bps: float = 1.0,
        pos_bias_penalty: float = 1e-5,
        flat_penalty: float = 0.0,
        max_episode_steps: int | None = None,
        **kwargs
    ):
        super().__init__()
        self.symbol = symbol.upper()
        self.data_root = data_root
        self.tf_fast = tf_fast.upper()
        self.tf_base = tf_base.upper()
        self.tf_slow = tf_slow.upper()
        self.cost_bps = float(cost_bps)
        self.pos_bias_penalty = float(pos_bias_penalty)
        self.flat_penalty = float(flat_penalty)
        self.max_episode_steps = max_episode_steps

        # Load raw frames
        base = _load_tf_csv(self.data_root, self.symbol, self.tf_base)
        fast = _load_tf_csv(self.data_root, self.symbol, self.tf_fast)
        slow = _load_tf_csv(self.data_root, self.symbol, self.tf_slow)

        # Date filtering
        if start:
            base = base.loc[pd.to_datetime(start, utc=True):]
        if end:
            base = base.loc[:pd.to_datetime(end, utc=True)]

        # Build features & join on base index
        bfe = _feats(base, "b_")
        ffe = _feats(fast, "f_").reindex(base.index, method="ffill")
        sfe = _feats(slow, "s_").reindex(base.index, method="ffill")

        feats = bfe.join(ffe, how="left").join(sfe, how="left")
        feats = feats.ffill().bfill()
        self.prices = base["Close"].astype(float)
        self.features = feats.astype(float)

        # Observation space (flat vector of features)
        self.feat_cols = list(self.features.columns)
        n_obs = len(self.feat_cols)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float32)
        # Action space: 0=FLAT, 1=LONG, 2=SHORT
        self.action_space = spaces.Discrete(3)

        # Internal state
        self.idx = 0
        self.position = 0  # -1,0,1
        self._done = False
        self._max_steps_real = len(self.features) - 2  # need t and t+1
        if self.max_episode_steps is None:
            self.max_episode_steps = self._max_steps_real
        self._steps = 0

    # --- helpers ---
    def _obs(self) -> np.ndarray:
        return self.features.iloc[self.idx].values.astype(np.float32)

    def _step_price_ret(self) -> float:
        px0 = float(self.prices.iloc[self.idx])
        px1 = float(self.prices.iloc[self.idx + 1])
        return (px1 / px0) - 1.0

    # --- API ---
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.idx = 0
        self.position = 0
        self._done = False
        self._steps = 0
        obs = self._obs()
        info = {}
        return obs, info

    def step(self, action: int):
        if self._done:
            obs = self._obs()
            return obs, 0.0, True, False, {}

        action = int(action)
        prev_pos = self.position
        # map action to target position
        if action == 0:
            self.position = 0
        elif action == 1:
            self.position = 1
        else:
            self.position = -1

        # market move on this step (idx -> idx+1)
        ret = self._step_price_ret()

        # trading cost if position changed
        trade_cost = 0.0
        if self.position != prev_pos:
            # cost applied on notional change (going from 0->1 charges once, 1->-1 charges twice)
            notional_turn = abs(self.position - prev_pos)
            trade_cost = notional_turn * (self.cost_bps * 1e-4)

        # PnL of holding the position over this bar
        pnl = ret * float(self.position)

        # reward with regularizers
        reward = pnl - trade_cost
        if self.position != 0:
            reward -= self.pos_bias_penalty
        else:
            reward -= self.flat_penalty

        # advance time
        self.idx += 1
        self._steps += 1
        terminated = (self.idx >= self._max_steps_real) or (self._steps >= self.max_episode_steps)
        truncated = False
        self._done = terminated or truncated

        obs = self._obs()
        info = {
            "ret": ret,
            "pnl": pnl,
            "trade_cost": trade_cost,
            "position": self.position
        }
        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self):  # optional
        pass
