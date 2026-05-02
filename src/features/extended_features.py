"""
src/features/extended_features.py
===================================
SCOPUS Extended Feature Set — Week 7 (Adaptive Upgrade Phase 1).

Adds 52 new features across 4 categories:
  A. Lagged signals      (12 features) — lag_*
  B. Volatility regime   (14 features) — vol_*
  C. Interaction         (16 features) — ix_*
  D. Micro-momentum      (10 features) — mom_*

Usage:
    from src.features.common_features import compute_features_from_ohlcv
    from src.features.extended_features import ExtendedFeatures

    base_df  = compute_features_from_ohlcv(raw_ohlcv)       # 20 features
    ext_df   = ExtendedFeatures.compute(raw_ohlcv, base_df) # +52 features
    full_df  = pd.concat([base_df, ext_df], axis=1)         # 72 features total

Promotion status: SHADOW_ONLY
  → Requires 30 shadow days with PSI < 0.15 before production promotion.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Optional

# ---------------------------------------------------------------------------
# Feature catalogue (single source of truth)
# ---------------------------------------------------------------------------

LAGGED_FEATURES: List[str] = [
    "lag_ret1", "lag_ret2", "lag_ret3",
    "lag_rsi3", "lag_rsi5",
    "lag_atr5", "lag_atr10",
    "lag_sma_ratio3", "lag_sma_ratio5",
    "lag_boll_z2", "lag_boll_z5",
    "lag_vol_norm3",
]

VOLATILITY_FEATURES: List[str] = [
    "vol_atr_ratio_5_20",
    "vol_atr_ratio_14_50",
    "vol_atr_pctile_50",
    "vol_atr_pctile_100",
    "vol_hv5",
    "vol_hv20",
    "vol_hv_ratio",
    "vol_garman_klass",
    "vol_parkinson",
    "vol_of_vol",
    "vol_atr_zscore_20",
    "vol_atr_zscore_50",
    "vol_bb_squeeze",
    "vol_regime",
]

INTERACTION_FEATURES: List[str] = [
    "ix_rsi_vol",
    "ix_trend_mom",
    "ix_body_vol",
    "ix_vol_trend",
    "ix_rsi_boll",
    "ix_adx_trend",
    "ix_adx_vol",
    "ix_rsi_lag_div",
    "ix_mom_align",
    "ix_vol_regime_rsi",
    "ix_squeeze_mom",
    "ix_hod_vol",
    "ix_dow_trend",
    "ix_adx_rsi",
    "ix_body_hl",
    "ix_spread_vol",
]

MICRO_MOMENTUM_FEATURES: List[str] = [
    "mom_1",
    "mom_3",
    "mom_10",
    "mom_accel",
    "mom_sign",
    "mom_streak",
    "mom_zscore",
    "mom_skew",
    "mom_kurt",
    "mom_rsi_roc",
]

# Combined list for shadow promotion tracking
EXTENDED_FEATURE_LIST: List[str] = (
    LAGGED_FEATURES
    + VOLATILITY_FEATURES
    + INTERACTION_FEATURES
    + MICRO_MOMENTUM_FEATURES
)

# Promotion gate metadata
PROMOTION_GATE = {
    "status":          "shadow",   # 'research' | 'shadow' | 'production'
    "min_shadow_days": 30,
    "max_psi":         0.15,
    "max_corr":        0.85,
    "min_gain_rank":   0.50,       # top 50% of features by model gain
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _safe_div(a: pd.Series, b: pd.Series, fill: float = 0.0) -> pd.Series:
    return (a / b.replace(0, np.nan)).fillna(fill)


def _rolling_pctile(s: pd.Series, window: int) -> pd.Series:
    """Rolling percentile rank [0, 1]."""
    def _rank(x):
        if len(x) < 2:
            return 0.5
        return float(np.sum(x[:-1] <= x[-1])) / max(len(x) - 1, 1)
    return s.rolling(window, min_periods=max(5, window // 2)).apply(_rank, raw=True)


def _rolling_zscore(s: pd.Series, window: int) -> pd.Series:
    mu  = s.rolling(window, min_periods=window // 2).mean()
    sig = s.rolling(window, min_periods=window // 2).std().replace(0, np.nan)
    return ((s - mu) / sig).fillna(0.0)


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([
        (h - l).abs(),
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period // 2).mean().fillna(0.0)


# ---------------------------------------------------------------------------
# Category A: Lagged Features
# ---------------------------------------------------------------------------

def _compute_lagged(base: pd.DataFrame) -> pd.DataFrame:
    """
    12 lagged-state features from base FEATURE_LIST columns.
    base: DataFrame with common_features.FEATURE_LIST columns.
    """
    f = pd.DataFrame(index=base.index)

    ret  = base.get("ret",         pd.Series(0.0, index=base.index))
    rsi  = base.get("rsi14",       pd.Series(50.0, index=base.index))
    atr  = base.get("atr14",       pd.Series(0.0, index=base.index))
    smar = base.get("sma_ratio",   pd.Series(1.0, index=base.index))
    boll = base.get("boll_z",      pd.Series(0.0, index=base.index))
    vol  = base.get("vol_norm",    pd.Series(0.0, index=base.index))

    f["lag_ret1"]       = ret.shift(1).fillna(0.0)
    f["lag_ret2"]       = ret.shift(2).fillna(0.0)
    f["lag_ret3"]       = ret.shift(3).fillna(0.0)
    f["lag_rsi3"]       = rsi.shift(3).fillna(50.0)
    f["lag_rsi5"]       = rsi.shift(5).fillna(50.0)
    f["lag_atr5"]       = atr.shift(5).fillna(0.0)
    f["lag_atr10"]      = atr.shift(10).fillna(0.0)
    f["lag_sma_ratio3"] = smar.shift(3).fillna(1.0)
    f["lag_sma_ratio5"] = smar.shift(5).fillna(1.0)
    f["lag_boll_z2"]    = boll.shift(2).fillna(0.0)
    f["lag_boll_z5"]    = boll.shift(5).fillna(0.0)
    f["lag_vol_norm3"]  = vol.shift(3).fillna(0.0)

    return f[LAGGED_FEATURES]


# ---------------------------------------------------------------------------
# Category B: Volatility Regime Features
# ---------------------------------------------------------------------------

def _compute_volatility_regime(raw: pd.DataFrame) -> pd.DataFrame:
    """
    14 volatility-regime features from raw OHLCV.
    """
    f     = pd.DataFrame(index=raw.index)
    close = raw["close"]

    atr5  = _atr(raw, 5)
    atr14 = _atr(raw, 14)
    atr20 = _atr(raw, 20)
    atr50 = _atr(raw, 50)

    # ATR ratios
    f["vol_atr_ratio_5_20"]  = _safe_div(atr5,  atr20,  1.0)
    f["vol_atr_ratio_14_50"] = _safe_div(atr14, atr50,  1.0)

    # ATR percentile rank
    f["vol_atr_pctile_50"]  = _rolling_pctile(atr14, 50)
    f["vol_atr_pctile_100"] = _rolling_pctile(atr14, 100)

    # Realized volatility (close-to-close log returns std)
    log_ret = np.log(close / close.shift(1).replace(0, np.nan))
    f["vol_hv5"]  = log_ret.rolling(5,  min_periods=3).std().fillna(0.0)
    f["vol_hv20"] = log_ret.rolling(20, min_periods=10).std().fillna(0.0)
    f["vol_hv_ratio"] = _safe_div(f["vol_hv5"], f["vol_hv20"], 1.0)

    # Garman-Klass volatility estimator (more efficient than close-to-close)
    ln_hl = np.log((raw["high"] / raw["low"].replace(0, np.nan)))
    ln_co = np.log((close / raw["open"].replace(0, np.nan)))
    gk    = (0.5 * ln_hl**2 - (2 * np.log(2) - 1) * ln_co**2)
    f["vol_garman_klass"] = gk.rolling(14, min_periods=7).mean().fillna(0.0)

    # Parkinson estimator (high-low range)
    pk = (1.0 / (4.0 * np.log(2))) * ln_hl**2
    f["vol_parkinson"] = pk.rolling(14, min_periods=7).mean().fillna(0.0)

    # Vol of vol (second-order uncertainty)
    f["vol_of_vol"] = f["vol_hv20"].rolling(20, min_periods=10).std().fillna(0.0)

    # Z-scored ATR
    f["vol_atr_zscore_20"] = _rolling_zscore(atr14, 20)
    f["vol_atr_zscore_50"] = _rolling_zscore(atr14, 50)

    # Bollinger squeeze (BB width < 20th percentile over 100 bars)
    mid = close.rolling(20, min_periods=20).mean()
    std = close.rolling(20, min_periods=20).std()
    bb_width = (2 * 2 * std / mid.replace(0, np.nan)).fillna(0.0)
    bb_pctile = _rolling_pctile(bb_width, 100)
    f["vol_bb_squeeze"] = (bb_pctile < 0.20).astype(float)

    # Volatility regime binary (1=HV, 0=LV) based on 80th percentile
    f["vol_regime"] = (f["vol_atr_pctile_100"] > 0.80).astype(float)

    return f[VOLATILITY_FEATURES]


# ---------------------------------------------------------------------------
# Category C: Interaction Features
# ---------------------------------------------------------------------------

def _compute_interactions(base: pd.DataFrame, vol_df: pd.DataFrame) -> pd.DataFrame:
    """
    16 interaction features combining base and volatility features.
    """
    f = pd.DataFrame(index=base.index)

    rsi       = base.get("rsi14",       pd.Series(50.0, index=base.index))
    atr_pct   = base.get("atr_pct",     pd.Series(0.0,  index=base.index))
    sma_ratio = base.get("sma_ratio",   pd.Series(1.0,  index=base.index))
    ret_5     = base.get("ret_5",        pd.Series(0.0,  index=base.index))
    vol_norm  = base.get("vol_norm",    pd.Series(0.0,  index=base.index))
    boll_z    = base.get("boll_z",      pd.Series(0.0,  index=base.index))
    adx14     = base.get("adx14",       pd.Series(20.0, index=base.index))
    body      = base.get("body",        pd.Series(0.0,  index=base.index))
    hl_range  = base.get("hl_range",    pd.Series(0.0,  index=base.index))
    hod       = base.get("hod",         pd.Series(0.0,  index=base.index))
    dow       = base.get("dow",         pd.Series(0.0,  index=base.index))
    rsi_lag5  = base.get("rsi14",       rsi).shift(5).fillna(50.0)  # use base RSI lag

    vol_regime = vol_df.get("vol_regime", pd.Series(0.0, index=base.index))
    bb_squeeze = vol_df.get("vol_bb_squeeze", pd.Series(0.0, index=base.index))

    # RSI × volatility state
    f["ix_rsi_vol"]      = (rsi / 100.0) * atr_pct

    # Trend × momentum alignment (both same direction = stronger)
    f["ix_trend_mom"]    = (sma_ratio - 1.0) * ret_5

    # Bar candle efficiency (body / ATR)
    atr14_s = base.get("atr14", pd.Series(1e-6, index=base.index))
    f["ix_body_vol"]     = _safe_div(body, atr14_s.replace(0, np.nan), 0.0)

    # Volume-confirmed trend
    f["ix_vol_trend"]    = vol_norm * (sma_ratio - 1.0)

    # Double confirmation: RSI × Bollinger Z
    f["ix_rsi_boll"]     = ((rsi - 50) / 50.0) * boll_z

    # ADX × trend direction
    f["ix_adx_trend"]    = (adx14 / 100.0) * (sma_ratio - 1.0)

    # ADX × volatility state
    f["ix_adx_vol"]      = (adx14 / 100.0) * atr_pct

    # RSI rate of change (momentum acceleration)
    f["ix_rsi_lag_div"]  = (rsi - rsi_lag5).fillna(0.0)

    # Short/long momentum alignment (-1, 0, +1)
    f["ix_mom_align"]    = np.sign(ret_5) * np.sign(base.get("ret_20",
                           pd.Series(0.0, index=base.index)))

    # Regime-conditional RSI (RSI value only matters in correct regime)
    f["ix_vol_regime_rsi"] = vol_regime * ((rsi - 50) / 50.0)

    # Squeeze × momentum (breakout potential)
    f["ix_squeeze_mom"]  = bb_squeeze * ret_5

    # Time-of-day × volatility
    f["ix_hod_vol"]      = (hod / 23.0) * atr_pct

    # Day-of-week × trend
    f["ix_dow_trend"]    = (dow / 4.0) * (sma_ratio - 1.0)

    # ADX × RSI direction score
    f["ix_adx_rsi"]      = (adx14 / 100.0) * ((rsi - 50) / 50.0)

    # Bar quality: body proportion of full range
    f["ix_body_hl"]      = _safe_div(body, hl_range.replace(0, np.nan), 0.0)

    # Range vs expected (bar volatility vs ATR)
    f["ix_spread_vol"]   = _safe_div(hl_range, atr14_s.replace(0, np.nan), 1.0)

    return f[INTERACTION_FEATURES]


# ---------------------------------------------------------------------------
# Category D: Micro-Momentum Features
# ---------------------------------------------------------------------------

def _compute_micro_momentum(base: pd.DataFrame) -> pd.DataFrame:
    """
    10 short-term momentum features.
    """
    f = pd.DataFrame(index=base.index)
    ret = base.get("ret", pd.Series(0.0, index=base.index))
    rsi = base.get("rsi14", pd.Series(50.0, index=base.index))

    f["mom_1"]     = ret.copy()
    f["mom_3"]     = ret.rolling(3, min_periods=1).sum()
    f["mom_10"]    = ret.rolling(10, min_periods=5).sum()

    # Momentum acceleration: current 1-bar vs 5-bar average
    avg_5          = ret.rolling(5, min_periods=3).mean()
    f["mom_accel"] = (ret - avg_5).fillna(0.0)

    # Directional consistency: rolling sum of sign(ret) over 5 bars (-1 to 1)
    sign_ret       = np.sign(ret).fillna(0.0)
    f["mom_sign"]  = (sign_ret.rolling(5, min_periods=3).sum() / 5.0)

    # Consecutive bar streak (+ for up, - for down)
    def _streak(x):
        if len(x) == 0:
            return 0.0
        streak = 0
        direction = np.sign(x[-1])
        for v in reversed(x):
            if np.sign(v) == direction and direction != 0:
                streak += direction
            else:
                break
        return float(streak)
    f["mom_streak"] = ret.rolling(10, min_periods=2).apply(_streak, raw=True).fillna(0.0)

    # Z-scored 1-bar return
    f["mom_zscore"] = _rolling_zscore(ret, 30)

    # Return skewness (fat-tail awareness)
    f["mom_skew"]   = ret.rolling(30, min_periods=15).skew().fillna(0.0)

    # Return kurtosis
    f["mom_kurt"]   = ret.rolling(30, min_periods=15).kurt().fillna(0.0)

    # RSI rate of change (RSI momentum)
    f["mom_rsi_roc"] = (rsi - rsi.shift(5)).fillna(0.0)

    return f[MICRO_MOMENTUM_FEATURES]


# ---------------------------------------------------------------------------
# Main compute function
# ---------------------------------------------------------------------------

class ExtendedFeatures:
    """
    Compute all 4 extended feature categories (52 new features).

    Input:
        raw   : raw OHLCV DataFrame (same format as common_features.py expects)
        base  : output of compute_features_from_ohlcv() — FEATURE_LIST columns

    Output:
        DataFrame with EXTENDED_FEATURE_LIST columns only.
        Caller should pd.concat([base_df, ext_df], axis=1) to get full 72-feature set.
    """

    @staticmethod
    def compute(raw: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all extended features.

        Args:
            raw : normalized OHLCV DataFrame (open/high/low/close/volume)
            base: output of compute_features_from_ohlcv() — has FEATURE_LIST columns

        Returns:
            DataFrame of shape (len(base), 52) with EXTENDED_FEATURE_LIST columns.
        """
        if raw is None or raw.empty or base is None or base.empty:
            return pd.DataFrame(columns=EXTENDED_FEATURE_LIST)

        try:
            lag_df = _compute_lagged(base)
        except Exception as e:
            lag_df = pd.DataFrame(0.0, index=base.index, columns=LAGGED_FEATURES)

        try:
            vol_df = _compute_volatility_regime(raw)
        except Exception as e:
            vol_df = pd.DataFrame(0.0, index=base.index, columns=VOLATILITY_FEATURES)

        try:
            ix_df = _compute_interactions(base, vol_df)
        except Exception as e:
            ix_df = pd.DataFrame(0.0, index=base.index, columns=INTERACTION_FEATURES)

        try:
            mom_df = _compute_micro_momentum(base)
        except Exception as e:
            mom_df = pd.DataFrame(0.0, index=base.index, columns=MICRO_MOMENTUM_FEATURES)

        result = pd.concat([lag_df, vol_df, ix_df, mom_df], axis=1)
        result = result.clip(-10, 10).fillna(0.0)
        return result[EXTENDED_FEATURE_LIST]

    @staticmethod
    def feature_names() -> List[str]:
        return list(EXTENDED_FEATURE_LIST)

    @staticmethod
    def feature_count() -> int:
        return len(EXTENDED_FEATURE_LIST)
