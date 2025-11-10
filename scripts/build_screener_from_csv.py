#!/usr/bin/env python3
"""
Build screener candidates from your local CSV files (multi-timeframe).

Accepts either:
1) One CSV per symbol (e.g., EURUSD.csv) -> resamples to requested TFs
2) One CSV per symbol-timeframe (e.g., EURUSD_M15.csv) -> uses as-is

CSV format (flexible headers, case-insensitive, autodetected):
  timestamp, open, high, low, close, volume
Timestamp must be parseable (UTC preferred).

Outputs:
  reports/screener/YYYYMMDD_intraday.json
Structure (list of candidates):
  {
    "symbol": "EURUSD",
    "pair": "EURUSD",
    "tf": "M15",
    "price": 1.06543,
    "atr14": 0.0012,
    "rsi14": 48.2,
    "vol_ratio": 0.85,
    "adx_proxy": 0.62,
    "trend_strength": 0.62,
    "regime": "trend" | "meanrev" | "calm" | "volatile",
    "score": 0.34,
    "confidence": 0.65
  }

Usage examples (PowerShell):

# One CSV per symbol; resample to multiple TFs
python scripts/build_screener_from_csv.py `
  --data_dir data/csv `
  --tfs M5 M15 H1 `
  --out reports/screener

# Already-split files like EURUSD_M15.csv, BTCUSD_H1.csv (auto-detect)
python scripts/build_screener_from_csv.py `
  --data_dir data/split `
  --tfs M15 H1 `
  --out reports/screener
"""

from __future__ import annotations
import argparse, os, json, math, re
from typing import Dict, Any, List, Tuple
from datetime import datetime, timezone
import pandas as pd
import numpy as np

# ---------- helpers ----------
TF_MAP = {"M1":"T1","M5":"T5","M15":"T15","M30":"T30","H1":"H","H4":"4H","D1":"D"}

def to_pandas_tf(tf: str) -> str:
    tf = tf.upper()
    return TF_MAP.get(tf, tf)

def parse_filename(path: str) -> Tuple[str, str | None]:
    """
    Returns (symbol, tf) parsed from filename like 'EURUSD_M15.csv' or 'EURUSD.csv'
    """
    name = os.path.splitext(os.path.basename(path))[0]
    m = re.match(r"^([A-Za-z0-9\-]+)_([Mm]\d+|[Hh]\d+|[Dd]1)$", name)
    if m:
        sym = m.group(1).upper()
        tf = m.group(2).upper()
        return sym, tf
    return name.upper(), None

def find_cols(df: pd.DataFrame) -> Dict[str,str]:
    cols = {c.lower(): c for c in df.columns}
    def pick(*names):
        for n in names:
            if n in cols: return cols[n]
        return None
    return {
        "ts": pick("timestamp","time","date","datetime"),
        "open": pick("open","o"),
        "high": pick("high","h"),
        "low":  pick("low","l"),
        "close":pick("close","c","adjclose","adj_close"),
        "vol":  pick("volume","vol","v")
    }

def ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    cols = find_cols(df)
    ts_col = cols["ts"]; o=cols["open"]; h=cols["high"]; l=cols["low"]; c=cols["close"]; v=cols["vol"]
    if not ts_col or not o or not h or not l or not c:
        raise ValueError("Missing required columns (need timestamp/open/high/low/close).")
    out = pd.DataFrame({
        "timestamp": pd.to_datetime(df[ts_col], utc=True, errors="coerce"),
        "open": pd.to_numeric(df[o], errors="coerce"),
        "high": pd.to_numeric(df[h], errors="coerce"),
        "low":  pd.to_numeric(df[l], errors="coerce"),
        "close":pd.to_numeric(df[c], errors="coerce"),
        "volume": pd.to_numeric(df[v], errors="coerce") if v else 0.0
    }).dropna(subset=["timestamp","open","high","low","close"])
    out = out.sort_values("timestamp")
    out = out.set_index("timestamp")
    return out

def resample_ohlcv(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    rule = to_pandas_tf(tf)
    o = df["open"].resample(rule).first()
    h = df["high"].resample(rule).max()
    l = df["low"].resample(rule).min()
    c = df["close"].resample(rule).last()
    v = df["volume"].resample(rule).sum(min_count=1)
    out = pd.concat([o,h,l,c,v], axis=1)
    out.columns = ["open","high","low","close","volume"]
    out = out.dropna()
    return out

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h,l,c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    up = np.where(delta>0, delta, 0.0)
    dn = np.where(delta<0, -delta, 0.0)
    up_ema = pd.Series(up, index=series.index).ewm(alpha=1/n, adjust=False).mean()
    dn_ema = pd.Series(dn, index=series.index).ewm(alpha=1/n, adjust=False).mean()
    rs = up_ema / (dn_ema.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)

def adx_proxy(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """A light directional strength proxy: normalize |close - EMA(close,n)| by ATR."""
    ema = df["close"].ewm(span=n, adjust=False).mean()
    strength = (df["close"] - ema).abs()
    atr14 = atr(df, n).replace(0, np.nan)
    proxy = (strength / atr14).clip(0, 5)
    return (proxy / 5.0).fillna(0.0)  # 0..1

def vol_ratio(df: pd.DataFrame, short: int = 14, long: int = 50) -> pd.Series:
    atr_s = atr(df, short)
    atr_l = atr(df, long).replace(0, np.nan)
    vr = (atr_s / atr_l).clip(lower=0, upper=3)
    return (vr / 3.0).fillna(1.0)  # ~1 normal, >1 volatile, <1 calm

def regime_label(vr: float, adx_p: float) -> str:
    if vr >= 0.9 and adx_p >= 0.55: return "trend"
    if vr >= 0.9 and adx_p <  0.55: return "volatile"
    if vr <  0.9 and adx_p >= 0.55: return "trend"
    if vr <  0.9 and adx_p <  0.55: return "meanrev"
    return "calm"

def score_from_features(close: float, rsi14: float, adx_p: float, vr: float) -> float:
    """
    Simple directional score in [-1, 1]:
      - overbought + strong trend -> sell bias
      - oversold  + strong trend -> buy bias
      - mean-rev -> fade extremes
    """
    trend = adx_p  # 0..1
    # center RSI around 50
    rsi_dev = (rsi14 - 50.0) / 50.0  # -1..+1
    # base trend-following
    score_trend = trend * ( -rsi_dev )  # high rsi -> negative (sell), low rsi -> positive (buy)
    # mean-rev adjustment when volatility low
    meanrev_weight = max(0.0, 0.4 * (1.0 - vr))
    score_meanrev = meanrev_weight * ( rsi_dev * -1 )  # fade extremes
    score = score_trend + score_meanrev
    return float(max(-1.0, min(1.0, score)))

def last_features(df: pd.DataFrame) -> Dict[str, float]:
    close = float(df["close"].iloc[-1])
    atr14 = float(atr(df, 14).iloc[-1])
    rsi14 = float(rsi(df["close"], 14).iloc[-1])
    adx_p = float(adx_proxy(df, 14).iloc[-1])
    vr = float(vol_ratio(df, 14, 50).iloc[-1])
    return {"price": close, "atr14": atr14, "rsi14": rsi14, "adx_proxy": adx_p, "vol_ratio": vr}

def build_candidate(symbol: str, tf: str, feats: Dict[str, float]) -> Dict[str, Any]:
    vr = feats["vol_ratio"]; adx_p = feats["adx_proxy"]; rsi14 = feats["rsi14"]; price = feats["price"]
    regime = regime_label(vr, adx_p)
    score = score_from_features(price, rsi14, adx_p, vr)
    confidence = float(min(1.0, 0.5 + 0.5 * max(adx_p, vr)))  # 0.5..1.0
    return {
        "symbol": symbol,
        "pair": symbol,
        "tf": tf,
        "price": round(price, 8),
        "atr14": round(feats["atr14"], 8),
        "rsi14": round(rsi14, 4),
        "vol_ratio": round(vr, 4),
        "adx_proxy": round(adx_p, 4),
        "trend_strength": round(adx_p, 4),
        "regime": regime,
        "score": round(score, 6),
        "confidence": round(confidence, 4)
    }

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return ensure_ohlcv(df)

def produce_candidates(data_dir: str, tfs: List[str]) -> List[Dict[str, Any]]:
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.lower().endswith(".csv")]
    results: List[Dict[str, Any]] = []
    for fp in files:
        sym, tf_in_file = parse_filename(fp)
        try:
            raw = load_csv(fp)
        except Exception as e:
            print(f"[skip] {fp}: {e}")
            continue

        if tf_in_file:  # already split per TF
            if tf_in_file not in [tf.upper() for tf in tfs]:
                continue
            df_tf = raw  # use as-is
            if len(df_tf) < 100:  # too short
                continue
            feats = last_features(df_tf)
            cand = build_candidate(sym, tf_in_file, feats)
            results.append(cand)
        else:
            # resample for each requested TF
            for tf in tfs:
                try:
                    df_tf = resample_ohlcv(raw, tf)
                    if len(df_tf) < 100:
                        continue
                    feats = last_features(df_tf)
                    cand = build_candidate(sym, tf, feats)
                    results.append(cand)
                except Exception as e:
                    print(f"[skip] {sym} {tf}: {e}")
                    continue
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Folder with CSV files")
    ap.add_argument("--tfs", nargs="+", default=["M15","H1"], help="List of timeframes (e.g., M5 M15 H1)")
    ap.add_argument("--out", default="reports/screener", help="Output folder")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    cands = produce_candidates(args.data_dir, args.tfs)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d")
    out_path = os.path.join(args.out, f"{ts}_intraday.json")

    # write as either list or {"candidates": [...]}; decision_engine supports both
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"candidates": cands}, f, indent=2)

    print(f"Wrote {len(cands)} candidates -> {out_path}")

if __name__ == "__main__":
    main()
