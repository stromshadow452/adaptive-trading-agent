# src/indicators_engine.py
from __future__ import annotations
"""
Indicators Engine (pure pandas/numpy)
-------------------------------------
Computes TradingView-like indicators WITHOUT external libraries (works on Python 3.14).
Inputs: DataFrame with columns: ["time","open","high","low","close","volume"]
Outputs: same DF + columns (rsi, macd_hist, adx, atr, ema_fast/slow, supertrend_dir, vol_ratio, trend_slope, clv)
         and a combined `tech_score` in approx range [-3, +3].

Usage in code:
    from src.indicators_engine import enrich_ohlcv, last_tech_snapshot, build_candidate_from_snapshot
"""

from dataclasses import dataclass
from typing import Dict, Any
import numpy as np
import pandas as pd


# ----------------------- config -----------------------

@dataclass
class IndiConfig:
    rsi_len: int = 14
    ema_fast: int = 20
    ema_slow: int = 50
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    adx_len: int = 14
    atr_len: int = 14
    supertrend_len: int = 10
    supertrend_mult: float = 3.0
    vol_ratio_lookback: int = 20
    trend_slope_lookback: int = 20
    # scoring thresholds
    rsi_buy: int = 55
    rsi_sell: int = 45
    adx_trend: int = 20


# ----------------------- math helpers -----------------------

def _ema(s: pd.Series, length: int) -> pd.Series:
    return s.ewm(alpha=2/(length+1.0), adjust=False, min_periods=length).mean()

def _rma(s: pd.Series, length: int) -> pd.Series:
    # Wilder's smoothing
    alpha = 1.0 / float(length)
    return s.ewm(alpha=alpha, adjust=False, min_periods=length).mean()

def _rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = _rma(gain, length)
    avg_loss = _rma(loss, length)
    rs = avg_gain / (avg_loss + 1e-12)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def _macd(close: pd.Series, fast: int, slow: int, signal: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd = ema_fast - ema_slow
    macd_signal = _ema(macd, signal)
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist

def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr

def _atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    tr = _true_range(high, low, close)
    return _rma(tr, length)

def _adx(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)
    atr = _atr(high, low, close, length)
    plus_di = 100.0 * _rma(plus_dm, length) / (atr + 1e-12)
    minus_di = 100.0 * _rma(minus_dm, length) / (atr + 1e-12)
    dx = 100.0 * (plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-12)
    adx = _rma(dx, length)
    return adx, plus_di, minus_di

def _supertrend(high: pd.Series, low: pd.Series, close: pd.Series, length: int, mult: float):
    """
    Returns: direction (+1 / -1), upper_band, lower_band
    Implementation follows common Supertrend logic with ATR-based bands.
    """
    atr = _atr(high, low, close, length)
    hl2 = (high + low) / 2.0
    upper_basic = hl2 + mult * atr
    lower_basic = hl2 - mult * atr

    upper_band = upper_basic.copy()
    lower_band = lower_basic.copy()

    # refine bands (carry previous values if they would tighten against trend)
    for i in range(1, len(close)):
        upper_band.iloc[i] = upper_basic.iloc[i] if (upper_basic.iloc[i] < upper_band.iloc[i-1] or close.iloc[i-1] > upper_band.iloc[i-1]) else upper_band.iloc[i-1]
        lower_band.iloc[i] = lower_basic.iloc[i] if (lower_basic.iloc[i] > lower_band.iloc[i-1] or close.iloc[i-1] < lower_band.iloc[i-1]) else lower_band.iloc[i-1]

    # trend direction & final band selection
    st_dir = pd.Series(index=close.index, dtype=float)
    st = pd.Series(index=close.index, dtype=float)

    st_dir.iloc[0] = 1.0  # start as up by default
    st.iloc[0] = lower_band.iloc[0]

    for i in range(1, len(close)):
        prev_dir = st_dir.iloc[i-1]
        if st.iloc[i-1] == upper_band.iloc[i-1]:
            st_dir.iloc[i] = -1.0 if close.iloc[i] > upper_band.iloc[i] else -1.0
        elif st.iloc[i-1] == lower_band.iloc[i-1]:
            st_dir.iloc[i] = 1.0 if close.iloc[i] >= lower_band.iloc[i] else 1.0

        # flip logic
        if prev_dir == -1.0 and close.iloc[i] > upper_band.iloc[i]:
            st_dir.iloc[i] = 1.0
        elif prev_dir == 1.0 and close.iloc[i] < lower_band.iloc[i]:
            st_dir.iloc[i] = -1.0
        else:
            st_dir.iloc[i] = prev_dir

        st.iloc[i] = lower_band.iloc[i] if st_dir.iloc[i] > 0 else upper_band.iloc[i]

    return st_dir, upper_band, lower_band


# ----------------------- core compute -----------------------

def enrich_ohlcv(df: pd.DataFrame, cfg: IndiConfig | None = None) -> pd.DataFrame:
    """
    Add indicators to an OHLCV dataframe (expects columns: open, high, low, close, volume).
    Returns a new dataframe with extra indicator columns and a `tech_score`.
    """
    cfg = cfg or IndiConfig()

    data = df.copy()
    cols = {c.lower(): c for c in data.columns}
    for needed in ["open","high","low","close","volume"]:
        if needed not in cols and needed not in data.columns:
            raise ValueError(f"OHLCV column '{needed}' missing")

    # lower-case views
    o = data[cols.get("open","open")]
    h = data[cols.get("high","high")]
    l = data[cols.get("low","low")]
    c = data[cols.get("close","close")]
    v = data[cols.get("volume","volume")]

    # indicators
    data["rsi"] = _rsi(c, length=cfg.rsi_len)

    data["ema_fast"] = _ema(c, cfg.ema_fast)
    data["ema_slow"] = _ema(c, cfg.ema_slow)

    macd, macd_signal, macd_hist = _macd(c, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal)
    data["macd"] = macd
    data["macd_signal"] = macd_signal
    data["macd_hist"] = macd_hist

    adx, di_plus, di_minus = _adx(h, l, c, cfg.adx_len)
    data["adx"] = adx
    data["di_plus"] = di_plus
    data["di_minus"] = di_minus

    data["atr"] = _atr(h, l, c, cfg.atr_len)

    st_dir, st_ub, st_lb = _supertrend(h, l, c, cfg.supertrend_len, cfg.supertrend_mult)
    data["supertrend_dir"] = st_dir
    data["supertrend_ub"] = st_ub
    data["supertrend_lb"] = st_lb

    vol_ma = v.rolling(cfg.vol_ratio_lookback).mean()
    data["vol_ratio"] = (v / (vol_ma + 1e-12)).clip(0, None)

    ema_slow = data["ema_slow"]
    slope = ema_slow.diff(cfg.trend_slope_lookback) / max(1, cfg.trend_slope_lookback)
    data["trend_slope"] = slope / (data["atr"] + 1e-12)

    rng = (h - l).replace(0, np.nan)
    data["clv"] = (((c - l) - (h - c)) / rng).fillna(0)

    # final score
    data["tech_score"] = _score_block(data, cfg).clip(-3.0, 3.0)

    return data


def _score_block(df: pd.DataFrame, cfg: IndiConfig) -> pd.Series:
    s = pd.Series(0.0, index=df.index)

    rsi = df["rsi"]
    macd_h = df["macd_hist"]
    ema_f = df["ema_fast"]; ema_s = df["ema_slow"]
    adx = df["adx"]
    st_dir = df["supertrend_dir"]
    volr = df["vol_ratio"]
    clv = df["clv"]
    slope = df["trend_slope"]

    # RSI bias
    s += np.where(rsi >= cfg.rsi_buy, 0.4, 0.0)
    s += np.where(rsi <= cfg.rsi_sell, -0.4, 0.0)

    # MACD histogram sign
    s += np.where(macd_h > 0, 0.3, 0.0)
    s += np.where(macd_h < 0, -0.3, 0.0)

    # EMA regime
    s += np.where(ema_f > ema_s, 0.5, -0.5)

    # Trend slope normalized (bounded)
    s += np.clip(slope, -0.6, 0.6)

    # ADX trend filter (only add if regime aligned)
    strong = (adx >= cfg.adx_trend).astype(float)
    s += strong * np.where(ema_f > ema_s, 0.3, -0.3)

    # Supertrend direction
    s += np.where(st_dir > 0, 0.3, 0.0)
    s += np.where(st_dir < 0, -0.3, 0.0)

    # Volume & CLV small nudges
    s += np.clip((volr - 1.0) * 0.1, -0.2, 0.4)
    s += np.clip(clv * 0.2, -0.2, 0.2)

    return s


# ----------------------- convenience -----------------------

def last_tech_snapshot(df_with_inds: pd.DataFrame) -> Dict[str, float]:
    row = df_with_inds.iloc[-1]
    keys = [
        "rsi","macd","macd_signal","macd_hist",
        "ema_fast","ema_slow","adx","di_plus","di_minus",
        "atr","supertrend_dir","vol_ratio","trend_slope","clv",
        "tech_score"
    ]
    return {k: float(row.get(k, np.nan)) for k in keys}

def build_candidate_from_snapshot(pair: str, tf: str, snap: Dict[str, float], price: float | None = None) -> Dict[str, Any]:
    return {
        "pair": pair,
        "tf": tf,
        "tech_score": float(snap.get("tech_score", 0.0)),
        "tech_sharpe": 0.0,
        "tech_trades": 0.0,
        "adx": float(snap.get("adx", 0.0)),
        "atr": float(snap.get("atr", 0.0)),
        "vol_ratio": float(snap.get("vol_ratio", 1.0)),
        "close": float(price) if price is not None else None,
        "ema_fast": float(snap.get("ema_fast", 0.0)),
        "ema_slow": float(snap.get("ema_slow", 0.0)),
        "supertrend_dir": float(snap.get("supertrend_dir", 0.0)),
        "rsi": float(snap.get("rsi", 50.0)),
    }
