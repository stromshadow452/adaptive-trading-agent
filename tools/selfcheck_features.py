# --- repo path shim so "src/..." works when running from tools/ ---
import os, sys, glob, json
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from typing import List
from src.model_guard import load_sidecar_meta, assert_feature_names, feature_list_hash
import pandas as pd

def _find_csv(symbol: str, roots: List[str]) -> str | None:
    stems = [
        f"{symbol}_M15*.csv", f"{symbol}_M5*.csv",
        f"{symbol}*.csv".replace("/", "").replace("-", "_"),
    ]
    for r in roots:
        for pat in stems:
            hits = glob.glob(os.path.join(r, "**", pat), recursive=True)
            if hits:
                hits.sort(key=lambda p: os.path.getmtime(p))
                return hits[-1]
    return None

def _normalize_df(path: str) -> pd.DataFrame:
    # tolerant reader (sep=None) & rename cols to lower
    df = pd.read_csv(path, sep=None, engine="python")
    lc = {c.lower().strip(): c for c in df.columns}
    def pick(*names):
        for n in names:
            if n in lc: return lc[n]
        return None
    c_open  = pick("open","o","bidopen","askopen")
    c_high  = pick("high","h","bidhigh","askhigh")
    c_low   = pick("low","l","bidlow","asklow")
    c_close = pick("close","c","bidclose","askclose","last","price","closeprice")
    c_vol   = pick("volume","vol","tickvol","tick_volume","tickvolume","tick vol","tickvol.")
    c_date  = pick("date")
    c_time  = pick("time","timestamp")

    if not all([c_open, c_high, c_low, c_close]):
        raise RuntimeError("CSV missing OHLC columns")

    out = pd.DataFrame()
    out["open"]  = pd.to_numeric(df[c_open], errors="coerce")
    out["high"]  = pd.to_numeric(df[c_high], errors="coerce")
    out["low"]   = pd.to_numeric(df[c_low], errors="coerce")
    out["close"] = pd.to_numeric(df[c_close], errors="coerce")
    out["volume"] = pd.to_numeric(df[c_vol], errors="coerce") if c_vol else 1.0

    if c_date and c_time:
        ts = pd.to_datetime(df[c_date].astype(str) + " " + df[c_time].astype(str), errors="coerce", utc=True)
    elif c_time:
        ts = pd.to_datetime(df[c_time], errors="coerce", utc=True)
    else:
        ts = pd.Series(pd.NaT, index=df.index)
    out["timestamp"] = ts
    out = out.dropna(subset=["open","high","low","close"]).tail(400).reset_index(drop=True)
    return out

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Model & features self-check")
    ap.add_argument("--model", default="models/fx_bin_19f.joblib")
    ap.add_argument("--csv_price_dir", action="append", default=["data/raw/M15_only","data/raw/forex_backup_2020_2025"])
    ap.add_argument("--symbol", default="EURUSD")
    args = ap.parse_args()

    meta = load_sidecar_meta(args.model)
    expected = meta.get("feature_names") or []
    print("== Meta ==")
    print(json.dumps({
        "model_name": meta.get("model_name"),
        "version": meta.get("version"),
        "timeframe": meta.get("timeframe"),
        "feature_count": len(expected),
        "feature_order_hash": meta.get("feature_order_hash"),
        "recomputed_hash": feature_list_hash(expected),
    }, indent=2))

    # Try to build live features using executor’s path (same logic/columns)
    # We’ll reuse the executor’s 19-feature builder via a local copy to avoid importing tools/*
    from tools.executor import _normalize_ohlcv, _compute_features_from_df  # uses same 19f recipe

    csvp = _find_csv(args.symbol.upper(), args.csv_price_dir)
    if not csvp:
        print(f"[warn] No CSV found for {args.symbol} under {args.csv_price_dir}; meta-only check done.")
        return 0

    raw = _normalize_df(csvp)
    feats = _compute_features_from_df(raw)
    live_cols = list(feats.columns)
    print(f"[info] CSV: {os.path.basename(csvp)}  rows={len(raw)}  features={len(live_cols)}")
    # guard
    assert_feature_names(expected, live_cols)
    print("[ok] Feature names & order match metadata.")
    # show last row sample
    sample = feats.iloc[-1].to_dict()
    small = {k: float(sample[k]) for k in expected[:min(5, len(expected))]}
    print("[ok] Sample (first 5 features):", small)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
