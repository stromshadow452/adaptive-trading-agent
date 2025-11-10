# tools/multi_pair_screener.py
from __future__ import annotations

# --- make project root importable (so `src/` resolves) ---
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

"""
Multi Pair Screener
-------------------
Reads OHLCV (from CSVs or mock generator), computes indicators, and writes
candidate JSON files for intraday & swing modes.

Usage:
  python tools/multi_pair_screener.py --config config/screener.yaml --out reports/screener
  # or package mode (from project root):
  python -m tools.multi_pair_screener --config config/screener.yaml --out reports/screener

Config (YAML) example (all fields optional):
  pairs: [AUDUSD, EURUSD, GBPUSD, USDJPY]
  intraday_tfs: [M15]
  swing_tfs: [H4, D1]
  bars: 500
  ohlcv_dir: data/ohlcv   # expects files: <PAIR>_<TF>.csv with time,open,high,low,close,volume
  mock: false             # if true, always generate synthetic data
  seed: 42

Output:
  reports/screener/<YYYYMMDD>_intraday.json
  reports/screener/<YYYYMMDD>_swing.json
"""

import argparse
import json
import warnings
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List

import numpy as np
import pandas as pd

# silence pandas future warnings (optional)
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

# Optional YAML
try:
    import yaml  # type: ignore
except Exception:
    yaml = None

# Indicators engine (pure pandas version you added)
from src.indicators_engine import enrich_ohlcv


# ---------------- defaults ----------------

DEFAULTS = {
    "pairs": ["AUDUSD", "EURUSD", "GBPUSD", "USDJPY"],
    "intraday_tfs": ["M15"],
    "swing_tfs": ["H4"],
    "bars": 500,
    "ohlcv_dir": "data/ohlcv",
    "mock": False,
    "seed": 123,
}


# ---------------- helpers ----------------

def now_utc_date_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")

def load_config(path: str | None) -> Dict[str, Any]:
    cfg = dict(DEFAULTS)
    if not path:
        return cfg
    p = Path(path)
    if not p.exists() or yaml is None:
        return cfg
    try:
        with open(p, "r", encoding="utf-8") as f:
            y = yaml.safe_load(f) or {}
        if isinstance(y, dict):
            cfg.update(y)
    except Exception:
        pass
    return cfg

def csv_path_for(ohlcv_dir: str, pair: str, tf: str) -> Path:
    return Path(ohlcv_dir) / f"{pair}_{tf}.csv"

def has_columns(df: pd.DataFrame, needed=("open","high","low","close","volume")) -> bool:
    cols = set([c.lower() for c in df.columns])
    return all(n in cols for n in needed)

def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with lower-case OHLCV columns and a 'time' index if present."""
    # soft rename
    rename_map = {}
    for c in df.columns:
        lc = c.lower()
        if lc in {"time","date","datetime","timestamp"} and "time" not in rename_map.values():
            rename_map[c] = "time"
        elif lc in {"open","high","low","close","volume"}:
            rename_map[c] = lc
    df = df.rename(columns=rename_map)

    # try parse time
    if "time" in df.columns:
        try:
            df["time"] = pd.to_datetime(df["time"])
            df = df.sort_values("time").reset_index(drop=True)
        except Exception:
            pass
    return df

def load_ohlcv_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        df = normalize_ohlcv(df)
        if not has_columns(df):
            return None
        # keep recent window (extra for indicator warmup)
        return df.tail(DEFAULTS["bars"] * 2)
    except Exception:
        return None

def generate_mock_ohlcv(bars: int, seed: int = 123, start_price: float = 1.1000) -> pd.DataFrame:
    """Simple GBM-like generator so the pipeline always runs."""
    rng = np.random.default_rng(seed)
    times = pd.date_range(end=pd.Timestamp.utcnow(), periods=bars, freq="min", tz="UTC")
    drift = 0.0
    vol = 0.0008
    returns = rng.normal(drift, vol, bars)
    price = start_price * np.exp(np.cumsum(returns))
    close = pd.Series(price, index=times)
    high = close * (1.0 + rng.uniform(0.0, 0.0008, bars))
    low = close * (1.0 - rng.uniform(0.0, 0.0008, bars))
    open_ = close.shift(1).fillna(close.iloc[0])
    volu = rng.integers(100, 2000, bars)
    df = pd.DataFrame({
        "time": times,
        "open": open_.values,
        "high": np.maximum.reduce([open_.values, high.values, close.values]),
        "low": np.minimum.reduce([open_.values, low.values, close.values]),
        "close": close.values,
        "volume": volu,
    })
    return df

def resample_to_tf(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Resample minute-level mock or raw data to target TF. Assumes df.time exists."""
    if "time" not in df.columns:
        return df
    out = df.set_index("time")
    rule = tf.upper()
    # normalize TF strings (M15/H1/H4/D1) → pandas-friendly rules
    if rule.startswith("M"):
        n = int(rule[1:])
        rule = f"{n}min"
    elif rule.startswith("H"):
        n = int(rule[1:])
        rule = f"{n}h"
    elif rule in ("D1","1D","D"):
        rule = "1D"
    # ohlcv resample
    o = out["open"].resample(rule).first()
    h = out["high"].resample(rule).max()
    l = out["low"].resample(rule).min()
    c = out["close"].resample(rule).last()
    v = out["volume"].resample(rule).sum()
    res = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v}).dropna()
    return res.reset_index().rename(columns={"time":"time"})

def detect_session(ts_utc: pd.Timestamp | None) -> str:
    """Rough session by UTC hour."""
    if ts_utc is None or pd.isna(ts_utc):
        return "UNKNOWN"
    hour = ts_utc.hour
    if 23 <= hour or hour < 8:
        return "ASIA"
    if 7 <= hour < 16:
        return "LONDON"
    if 12 <= hour < 21:
        return "NEWYORK"
    return "OFF"

def build_candidate(pair: str, tf: str, df_inds: pd.DataFrame) -> Dict[str, Any]:
    """Take enriched OHLCV and produce a candidate dict expected by DecisionEngine."""
    row = df_inds.iloc[-1]
    last_ts = df_inds["time"].iloc[-1] if "time" in df_inds.columns else None
    last_ts = pd.to_datetime(last_ts, utc=True, errors="coerce") if last_ts is not None else None

    cand = {
        "pair": pair,
        "tf": tf,
        "close": float(row["close"]),
        "price": float(row["close"]),
        "score": float(row.get("tech_score", 0.0)),   # legacy key used earlier
        "tech_score": float(row.get("tech_score", 0.0)),
        "tech_sharpe": 0.0,
        "tech_trades": 0.0,
        "adx": float(row.get("adx", 0.0)),
        "atr": float(row.get("atr", 0.0)),
        "vol_ratio": float(row.get("vol_ratio", 1.0)),
        # placeholders for fundamentals; your event module can fill these later
        "impact": 0.0,
        "impact_score": 0.0,
        "surprise_norm": 0.0,
        "fund_score": 0.0,
        "within_1h_event": False,
        "session": detect_session(last_ts),
    }
    return cand


# ---------------- pipeline ----------------

def process_pair_tf(pair: str, tf: str, cfg: Dict[str, Any]) -> Dict[str, Any] | None:
    bars = int(cfg.get("bars", DEFAULTS["bars"]))
    seed = int(cfg.get("seed", DEFAULTS["seed"]))
    ohlcv_dir = cfg.get("ohlcv_dir", DEFAULTS["ohlcv_dir"])
    force_mock = bool(cfg.get("mock", DEFAULTS["mock"]))

    df: pd.DataFrame | None = None
    if not force_mock:
        p = csv_path_for(ohlcv_dir, pair, tf)
        df = load_ohlcv_csv(p)

    if df is None:
        # generate mock minute bars then resample to requested TF
        df_min = generate_mock_ohlcv(max(bars * 5, 500), seed=seed, start_price=1.1000 + (hash(pair) % 1000)/10000.0)
        df = resample_to_tf(df_min, tf)

    df = df.tail(bars).reset_index(drop=True)
    df_inds = enrich_ohlcv(df)
    return build_candidate(pair, tf, df_inds)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/screener.yaml")
    ap.add_argument("--out", default="reports/screener")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    date_str = now_utc_date_str()

    intraday_candidates: List[Dict[str, Any]] = []
    swing_candidates: List[Dict[str, Any]] = []

    pairs = cfg.get("pairs", DEFAULTS["pairs"])
    itfs = cfg.get("intraday_tfs", DEFAULTS["intraday_tfs"])
    stfs = cfg.get("swing_tfs", DEFAULTS["swing_tfs"])

    # intraday set
    for pair in pairs:
        for tf in itfs:
            try:
                cand = process_pair_tf(pair, tf, cfg)
                if cand: intraday_candidates.append(cand)
            except Exception as e:
                print(f"[skip] {pair} {tf}: {e}")

    # swing set
    for pair in pairs:
        for tf in stfs:
            try:
                cand = process_pair_tf(pair, tf, cfg)
                if cand: swing_candidates.append(cand)
            except Exception as e:
                print(f"[skip] {pair} {tf}: {e}")

    # write outputs
    intraday_path = out_dir / f"{date_str}_intraday.json"
    swing_path = out_dir / f"{date_str}_swing.json"
    with open(intraday_path, "w", encoding="utf-8") as f:
        json.dump(intraday_candidates, f, indent=2)
    with open(swing_path, "w", encoding="utf-8") as f:
        json.dump(swing_candidates, f, indent=2)

    print(f"Saved: {intraday_path}")
    print(f"Saved: {swing_path}")

if __name__ == "__main__":
    main()
