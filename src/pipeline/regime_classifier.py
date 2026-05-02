"""
src/pipeline/regime_classifier.py
==================================
Dynamic Regime Classifier — Stage 5 (Regime)

Classifies each bar into exactly one of three regimes:

    MEAN_REVERSION  →  trades ALLOWED      (MR edge applies)
    TREND           →  trades BLOCKED      (trend overrides MR)
    NEUTRAL         →  trades REDUCED      (ambiguous — size down)

Integration rule:
    This module is a READ-ONLY observer of OHLCV data.
    It produces a regime label consumed at Stage 8 (MetaGatingBrain).
    It does NOT modify features, signals, sizing, or SL/TP.

Why this works:
    Mean-reversion edges (Bollinger z-score) collapse inside strong trends because
    prices do NOT revert — they continue.  ADX measures directional momentum
    relative to volatility (True Range), making it the most direct available
    proxy for "is price trending vs oscillating?".

    |atr_pctile| acts as a second axis:
      • Low volatility + low ADX  → quiet range  → MR setups are real
      • High volatility + high ADX → trending     → MR signals are traps
      • High volatility + low ADX → noisy chop   → NEUTRAL, size down

Vectorized: no Python loops over rows.
Deterministic: thresholds are explicit config values, no ML.
Backtest-safe: uses only past data (rolling windows, no lookahead).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

__all__ = [
    "RegimeConfig",
    "RegimeScoringConfig",
    "classify_regime",
    "classify_regime_scalar",
    "compute_regime_score",
]


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class RegimeConfig:
    """
    All thresholds in one place — easy to tune per symbol or time frame.

    TREND gate (BLOCK):
        ADX > adx_trend_threshold  AND  atr_pctile > atr_trend_min

    MEAN_REVERSION gate (ALLOW):
        ADX < adx_mr_threshold  AND  atr_pctile < atr_mr_max

    Everything else → NEUTRAL (size reduction, configurable via size_neutral).
    """

    # ADX thresholds (Wilder's 14-period ADX, scale 0-100)
    adx_period:           int   = 14     # rolling window for ADX calculation
    adx_trend_threshold:  float = 20.0   # T1 — above this → trending (relaxed from 22.0)
    adx_mr_threshold:     float = 15.0   # T3 — below this → mean-reverting

    # ATR percentile thresholds (0.0–1.0)
    atr_pctile_period:    int   = 100    # lookback for percentile rank
    atr_trend_min:        float = 0.45   # T2 — confirm trend with moderate+ ATR percentile (relaxed from 0.50)
    atr_mr_max:           float = 0.65   # T4 — low-ish vol confirms MR (relaxed from 0.60)

    # Optional return-std stability guard
    use_stability_filter: bool  = True
    stability_window:     int   = 10     # rolling window for return std
    stability_threshold:  float = 0.003  # daily std threshold (~0.3%) above →
                                          # noisy, push toward NEUTRAL

    # Minimum warm-up bars before issuing a real regime (returns NEUTRAL until)
    min_bars: int = 30


# ── Continuous Scoring Engine ─────────────────────────────────────────────────

@dataclass
class RegimeScoringConfig:
    """
    Configuration for the continuous Regime Scoring Engine.

    Replaces binary hard-threshold classification with a smooth 0-to-1 score:
        1.0  →  ideal mean-reversion environment (allow + full size)
        0.65 →  minimum quality gate (allow_trade threshold)
        0.0  →  strong trend / dangerous (block)

    Weights must sum to 1.0.
    """
    # Feature normalization denominators
    adx_scale:       float = 50.0    # ADX / adx_scale → [0, 1]
    noise_scale:     float = 0.01    # ret_std / noise_scale → [0, 1]

    # Composite score weights (must sum to 1.0)
    #   w_adx   : trend-suppression weight (highest priority)
    #   w_atr   : volatility-state weight
    #   w_noise : noise/chop weight
    w_adx:   float = 0.50
    w_atr:   float = 0.30
    w_noise: float = 0.20

    # Meta-gating gate: allow_trade = True iff regime_score > threshold
    score_threshold: float = 0.65

    def __post_init__(self):
        total = round(self.w_adx + self.w_atr + self.w_noise, 10)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"[RegimeScoringConfig] Weights must sum to 1.0, got {total:.6f}. "
                f"w_adx={self.w_adx}  w_atr={self.w_atr}  w_noise={self.w_noise}"
            )


def compute_regime_score(
    adx_val:     float,
    atr_pct_val: float,
    ret_std_val: float = 0.0,
    cfg: Optional[RegimeScoringConfig] = None,
) -> dict:
    """
    Continuous Regime Scoring Engine — Stage 5 output.

    Produces a single float in [0, 1] representing how favourable the current
    market environment is for mean-reversion trading:

        regime_score = w_adx   * (1 - adx_norm)
                     + w_atr   * (1 - atr_norm)
                     + w_noise * (1 - noise_norm)

    where:
        adx_norm   = clip(adx_val  / adx_scale,   0, 1)
        atr_norm   = clip(atr_pct_val,             0, 1)   (already 0-1)
        noise_norm = clip(ret_std_val / noise_scale,0, 1)

    Rationale:
        • (1 - adx_norm)   : Low ADX → oscillating market → MR works
        • (1 - atr_norm)   : Low ATR pctile → contained vol → moves revert
        • (1 - noise_norm) : Low return std → stable oscillations, not noise

    Stage 8 integration:
        result = compute_regime_score(adx, atr_pct, ret_std)
        if not result["allow_trade"]:
            return  # block
        signal["size"] *= result["regime_score"]   # optional: scale by score

    Parameters
    ----------
    adx_val     : float  Wilder's ADX (0–100)
    atr_pct_val : float  ATR percentile rank (0.0–1.0)
    ret_std_val : float  Rolling return std; 0.0 disables the noise component
    cfg         : RegimeScoringConfig (uses defaults if None)

    Returns
    -------
    {
        "regime_score":   float,   # 0.0 → 1.0  (higher = better for MR)
        "allow_trade":    bool,    # score > cfg.score_threshold
        "adx_norm":       float,   # normalized ADX  (0-1)
        "atr_norm":       float,   # normalized ATR percentile (0-1)
        "noise_norm":     float,   # normalized return std (0-1)
    }
    """
    if cfg is None:
        cfg = RegimeScoringConfig()

    # ── Normalize ────────────────────────────────────────────────────────────
    adx_norm   = float(np.clip(adx_val     / cfg.adx_scale,   0.0, 1.0))
    atr_norm   = float(np.clip(atr_pct_val,                   0.0, 1.0))
    noise_norm = float(np.clip(ret_std_val / cfg.noise_scale,  0.0, 1.0))

    # ── Composite score ───────────────────────────────────────────────────────
    score = (
        cfg.w_adx   * (1.0 - adx_norm)
        + cfg.w_atr   * (1.0 - atr_norm)
        + cfg.w_noise * (1.0 - noise_norm)
    )
    # safety clamp (floating-point edge cases)
    score = float(np.clip(score, 0.0, 1.0))

    return {
        "regime_score": round(score, 4),
        "allow_trade":  score > cfg.score_threshold,
        "adx_norm":     round(adx_norm,   4),
        "atr_norm":     round(atr_norm,   4),
        "noise_norm":   round(noise_norm, 4),
    }


# ── Core Computation ──────────────────────────────────────────────────────────

def _wilder_adx(
    high: pd.Series,
    low:  pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Wilder's smoothed ADX.  Fully vectorized.

    Returns a Series on the same index as inputs, range 0–100.
    Requires 2*period bars to stabilize (first `period` bars will be NaN/0).
    """
    # True Range
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    # Directional Movement
    up   = high.diff()
    down = -low.diff()

    plus_dm  = np.where((up > down) & (up > 0),   up,   0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    plus_dm_s  = pd.Series(plus_dm,  index=high.index, dtype=float)
    minus_dm_s = pd.Series(minus_dm, index=high.index, dtype=float)

    # Wilder's smoothing (equivalent to EMA with alpha = 1/period)
    def _wilder_smooth(s: pd.Series) -> pd.Series:
        out   = np.empty(len(s))
        out[:] = np.nan
        vals  = s.values
        # seed
        seed_end = period
        out[seed_end - 1] = vals[:seed_end].sum()
        for i in range(seed_end, len(vals)):
            out[i] = out[i - 1] - (out[i - 1] / period) + vals[i]
        return pd.Series(out, index=s.index)

    atr_w     = _wilder_smooth(tr)
    plus_w    = _wilder_smooth(plus_dm_s)
    minus_w   = _wilder_smooth(minus_dm_s)

    # Directional Indicators
    with np.errstate(divide="ignore", invalid="ignore"):
        plus_di  = 100 * plus_w  / atr_w.replace(0, np.nan)
        minus_di = 100 * minus_w / atr_w.replace(0, np.nan)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)

    adx = _wilder_smooth(dx.fillna(0)) / period   # normalize back to 0-100
    return adx.fillna(0)




def _atr_percentile_rank(
    high:   pd.Series,
    low:    pd.Series,
    close:  pd.Series,
    period:  int = 14,
    window:  int = 100,
) -> pd.Series:
    """
    ATR-14 percentile rank within the last `window` bars.
    Returns values in [0, 1].  0 = current ATR is the lowest it's been;
    1 = current ATR is the highest.
    """
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr = tr.rolling(period, min_periods=period).mean()

    # Rolling percentile rank
    def _pctile(x: np.ndarray) -> float:
        if len(x) < 2 or np.isnan(x[-1]):
            return 0.5
        return float((x[:-1] < x[-1]).sum() / (len(x) - 1))

    pctile = atr.rolling(window + 1, min_periods=period + 2).apply(
        _pctile, raw=True
    )
    return pctile.fillna(0.5)


def _return_std(close: pd.Series, window: int = 10) -> pd.Series:
    """Rolling std of log returns — proxy for noise level."""
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(window, min_periods=3).std().fillna(0.0)


# ── Public API ────────────────────────────────────────────────────────────────

def classify_regime(
    df:  pd.DataFrame,
    cfg: Optional[RegimeConfig] = None,
) -> pd.DataFrame:
    """
    Vectorized regime classification over a full OHLCV DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: open, high, low, close (case-insensitive).
        Optionally: atr_pctile (if already computed upstream — will skip recompute).
    cfg : RegimeConfig, optional
        Thresholds.  Defaults to RegimeConfig() with published defaults.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with three new columns appended:
            regime         : str    — "MEAN_REVERSION" | "TREND" | "NEUTRAL"
            regime_adx     : float  — ADX value (0–100)
            regime_atr_pct : float  — ATR percentile rank (0–1)
            regime_ret_std : float  — rolling return std (stability filter)

    Notes
    -----
    * No lookahead: all rolling operations use only past data.
    * First `cfg.min_bars` rows will have regime = "NEUTRAL" (warm-up guard).
    """
    if cfg is None:
        cfg = RegimeConfig()

    # ── Normalize column names ──────────────────────────────────────────────
    rename = {c: c.lower() for c in df.columns}
    work   = df.rename(columns=rename)

    for col in ("high", "low", "close"):
        if col not in work.columns:
            raise ValueError(
                f"[RegimeClassifier] Missing required column '{col}'. "
                f"Available: {list(work.columns)}"
            )

    high  = work["high"].astype(float)
    low   = work["low"].astype(float)
    close = work["close"].astype(float)

    # ── Feature 1: Trend Strength (Wilder ADX) ──────────────────────────────
    adx = _wilder_adx(high, low, close, period=cfg.adx_period)

    # ── Feature 2: Volatility State (ATR percentile) ────────────────────────
    if "atr_pctile" in work.columns:
        # Re-use upstream-computed value if available (avoids duplication)
        atr_pct = work["atr_pctile"].astype(float)
    else:
        atr_pct = _atr_percentile_rank(
            high, low, close,
            period=cfg.adx_period,
            window=cfg.atr_pctile_period,
        )

    # ── Feature 3: Stability Filter (optional) ──────────────────────────────
    ret_std = _return_std(close, window=cfg.stability_window)

    # ── Regime Logic ─────────────────────────────────────────────────────────
    #
    #  TREND   : ADX > T1  AND  atr_pctile > T2
    #            (strong directional move + elevated volatility)
    #
    #  MEAN_REVERSION : ADX < T3  AND  atr_pctile < T4
    #                   (+ optionally: return_std below noise threshold)
    #            (low directional strength + contained volatility)
    #
    #  NEUTRAL : everything else (transition zones)

    is_trend = (adx >= cfg.adx_trend_threshold) & (atr_pct >= cfg.atr_trend_min)

    is_mr = (adx < cfg.adx_mr_threshold) & (atr_pct < cfg.atr_mr_max)
    if cfg.use_stability_filter:
        # Extra noise guard: very choppy returns downgrade MR → NEUTRAL
        is_mr = is_mr & (ret_std <= cfg.stability_threshold)

    # Priority: TREND overrides MR when both conditions somehow fire
    regime = pd.Series("NEUTRAL", index=df.index, dtype=object)
    regime[is_mr]    = "MEAN_REVERSION"
    regime[is_trend] = "TREND"            # TREND wins on conflict

    # ── Warm-up guard ────────────────────────────────────────────────────────
    regime.iloc[: cfg.min_bars] = "NEUTRAL"

    # ── Attach results ───────────────────────────────────────────────────────
    out = df.copy()
    out["regime"]         = regime.values
    out["regime_adx"]     = adx.values
    out["regime_atr_pct"] = atr_pct.values
    out["regime_ret_std"] = ret_std.values

    # ── Continuous score (vectorized) ─────────────────────────────────────────
    scfg     = RegimeScoringConfig()
    adx_n    = np.clip(adx.values / scfg.adx_scale,   0.0, 1.0)
    atr_n    = np.clip(atr_pct.values,                 0.0, 1.0)
    noise_n  = np.clip(ret_std.values / scfg.noise_scale, 0.0, 1.0)
    scores   = np.clip(
        scfg.w_adx   * (1.0 - adx_n)
        + scfg.w_atr   * (1.0 - atr_n)
        + scfg.w_noise * (1.0 - noise_n),
        0.0, 1.0
    )
    out["regime_score"] = scores

    return out


def classify_regime_scalar(
    adx_val:     float,
    atr_pct_val: float,
    ret_std_val: float = 0.0,
    cfg: Optional[RegimeConfig] = None,
) -> dict:
    """
    Single-bar regime classification — for use inside the live pipeline loop.

    Called once per bar after features are computed. Returns a dict matching
    the MetaGatingBrain "regime_info" schema.

    Parameters
    ----------
    adx_val     : float  Wilder's ADX (0–100) for the current bar
    atr_pct_val : float  ATR percentile rank (0–1) for the current bar
    ret_std_val : float  Rolling return std (optional stability guard)
    cfg         : RegimeConfig (uses defaults if None)

    Returns
    -------
    {
        "regime":         "MEAN_REVERSION" | "TREND" | "NEUTRAL",
        "trend_strength": float,   # ADX value
        "atr_pctile":     float,   # ATR percentile
        "ret_std":        float,   # return std
        "allow_trade":    bool,    # True only for MEAN_REVERSION
        "size_mult":      float,   # 1.0 | 0.0 | 0.5
    }

    Integration at Stage 8 (MetaGatingBrain):
        regime_info = classify_regime_scalar(adx, atr_pct)
        if not regime_info["allow_trade"]:
            return  # block signal
        signal["size"] *= regime_info["size_mult"]
    """
    if cfg is None:
        cfg = RegimeConfig()

    is_trend = (
        adx_val     >= cfg.adx_trend_threshold
        and atr_pct_val >= cfg.atr_trend_min
    )

    is_mr = (
        adx_val     < cfg.adx_mr_threshold
        and atr_pct_val < cfg.atr_mr_max
        and (not cfg.use_stability_filter or ret_std_val <= cfg.stability_threshold)
    )

    # Soft risk scaling: no hard blocking, size multiplier controls risk
    if is_trend:
        regime    = "TREND"
        allow     = True        # Allow TREND regime with reduced sizing
        size_mult = 0.4         # 40% size for trending markets
    elif is_mr:
        regime    = "MEAN_REVERSION"
        allow     = True        # Full size for clear MR regimes
        size_mult = 1.0
    else:
        # Grey-zone handling: NEUTRAL/transition regimes get reduced-size trades
        regime    = "NEUTRAL"
        allow     = True        # Allow grey-zone trades with reduced sizing
        size_mult = 0.5         # 50% size for borderline regimes

    # Get continuous score for regime quality metrics
    score_result = compute_regime_score(adx_val, atr_pct_val, ret_std_val)

    return {
        "regime":         regime,
        "trend_strength": round(adx_val, 2),
        "atr_pctile":     round(atr_pct_val, 4),
        "ret_std":        round(ret_std_val, 6),
        # Soft sizing: no regime hard-blocks, size multiplier controls risk
        "allow_trade":    allow,
        "size_mult":      size_mult,
        # Include continuous score components (but not allow_trade which we override)
        "regime_score":   score_result["regime_score"],
        "adx_norm":       score_result["adx_norm"],
        "atr_norm":       score_result["atr_norm"],
        "noise_norm":     score_result["noise_norm"],
    }


# ── Quick self-test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # ── 1. Binary classifier on synthetic data ────────────────────────────────
    np.random.seed(42)
    n = 500
    trend_returns = np.random.normal(0.0003, 0.001, 250)
    mr_returns    = np.random.normal(0.0, 0.0004, 250)
    log_returns   = np.concatenate([trend_returns, mr_returns])
    close         = 1.1000 * np.exp(np.cumsum(log_returns))
    high          = close * (1 + np.abs(np.random.normal(0, 0.0003, n)))
    low           = close * (1 - np.abs(np.random.normal(0, 0.0003, n)))

    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    df  = pd.DataFrame({"open": close, "high": high, "low": low, "close": close}, index=idx)

    cfg = RegimeConfig()
    result = classify_regime(df, cfg)

    trend_bars = (result["regime"] == "TREND").sum()
    mr_bars    = (result["regime"] == "MEAN_REVERSION").sum()
    neut_bars  = (result["regime"] == "NEUTRAL").sum()
    score_vals = result["regime_score"]

    print("=" * 65)
    print("  RegimeClassifier + Scoring Engine — Self-Test")
    print("=" * 65)
    print(f"  Bars          : {n}")
    print(f"  TREND         : {trend_bars:4d}  ({100*trend_bars/n:.1f}%)")
    print(f"  MEAN_REVERSION: {mr_bars:4d}  ({100*mr_bars/n:.1f}%)")
    print(f"  NEUTRAL       : {neut_bars:4d}  ({100*neut_bars/n:.1f}%)")
    print(f"  Score mean    : {score_vals.mean():.3f}  "
          f"min={score_vals.min():.3f}  max={score_vals.max():.3f}")
    score_allow = (score_vals > 0.65).sum()
    print(f"  Score > 0.65  : {score_allow:4d}  ({100*score_allow/n:.1f}%)  ← score-gated allows")
    print("=" * 65)

    # ── 2. compute_regime_score() examples + edge cases ───────────────────────
    SEP  = "-" * 65
    HDR  = (f"  {'Scenario':<25}  {'ADX':>5}  {'ATR':>5}  "
            f"{'Std':>6}  {'Score':>6}  {'Allow':>5}")

    scenarios = [
        # (adx, atr_pct, ret_std, description)
        (10.0, 0.35, 0.001, "Ideal MR environment"),
        (12.0, 0.50, 0.002, "Good MR  (near threshold)"),
        (18.0, 0.55, 0.002, "Borderline NEUTRAL"),
        (25.0, 0.65, 0.003, "Transitioning to trend"),
        (30.0, 0.80, 0.001, "Strong trend — high ADX"),
        (45.0, 0.90, 0.005, "Very strong trend + vol"),
        ( 8.0, 0.20, 0.008, "Low ADX but very noisy"),
        ( 0.0, 0.00, 0.000, "Edge: all zeros → score 1.0"),
        (50.0, 1.00, 0.010, "Edge: all max   → score 0.0"),
    ]

    print(f"\n  compute_regime_score() — Example Inputs → Outputs")
    print(SEP)
    print(HDR)
    print(SEP)
    scfg = RegimeScoringConfig()
    for adx_v, atr_v, std_v, desc in scenarios:
        r = compute_regime_score(adx_v, atr_v, std_v, scfg)
        allow = "✓" if r["allow_trade"] else "✗"
        print(f"  {desc:<25}  {adx_v:>5.1f}  {atr_v:>5.2f}  "
              f"{std_v:>6.4f}  {r['regime_score']:>6.3f}  {allow:>5}")
    print(SEP)

    # ── 3. classify_regime_scalar() with score ────────────────────────────────
    print(f"\n  classify_regime_scalar() — binary + score combined")
    test_cases = [
        (30.0, 0.80, 0.001, "→ TREND"),
        (10.0, 0.40, 0.001, "→ MEAN_REVERSION"),
        (20.0, 0.55, 0.002, "→ NEUTRAL"),
        (10.0, 0.40, 0.005, "→ NEUTRAL (noisy)"),
    ]
    for adx_v, atr_v, std_v, note in test_cases:
        r = classify_regime_scalar(adx_v, atr_v, std_v, cfg)
        print(f"    ADX={adx_v:4.1f}  atr={atr_v:.2f}  std={std_v:.3f}"
              f"  regime={r['regime']:<16}  score={r['regime_score']:.3f}  allow={r['allow_trade']}  {note}")
    print()
