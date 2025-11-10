#!/usr/bin/env python3
"""
Filter a screener candidates JSON to only keep symbols for which we have CSV data.
Also optionally rewrite the timeframe to a known-available TF from a universe map.

Usage examples:
  # Auto-pick latest *_intraday.json from reports/screener and keep M15,H4
  python tools/filter_candidates_by_data.py --out reports/screener/latest_intraday.filtered.json --tfs M15,H4

  # Or point to a specific file
  python tools/filter_candidates_by_data.py --in reports/screener/20251025_intraday.json --out reports/screener/20251025_intraday.filtered.json --tfs M15,H4

  # Optional: use universe map (from tools/universe_from_data.py) to fallback TFs
  python tools/filter_candidates_by_data.py --out reports/screener/latest_intraday.filtered.json --tfs M15,H4 --universe-map config/screener_universe.yaml
"""
import argparse, json, os, glob
from pathlib import Path
from typing import Dict, Any, List, Set, Optional, Tuple

# Light classifiers
FX_CCYS = {"USD","EUR","JPY","GBP","AUD","NZD","CAD","CHF"}
CRYPTO_BASES = {"BTC","ETH","SOL","ADA","XRP","DOGE","BNB","DOT","LTC"}

TF_SYNONYMS = {
    "M1":  {"M1","1m","1min","60s"},
    "M5":  {"M5","5m","5min","300s"},
    "M15": {"M15","15m","15min","900s"},
    "M30": {"M30","30m","30min","1800s"},
    "H1":  {"H1","1H","60m"},
    "H4":  {"H4","4H","240m"},
    "D1":  {"D1","1D","1440m","Daily"},
}

def classify_symbol(sym: str) -> str:
    s = sym.upper()
    if s.endswith("USD") and s[:3] in CRYPTO_BASES:
        return "crypto"
    if len(s) == 6 and s[:3] in FX_CCYS and s[3:] in FX_CCYS:
        return "forex"
    return "other"

def tf_candidates(tf: str) -> Set[str]:
    t = tf.upper()
    out: Set[str] = {t}
    for canon, syns in TF_SYNONYMS.items():
        if t == canon or t in syns:
            out |= {canon}
            out |= set(syns)
    return out

def csv_exists(data_root: Path, sym: str, tf: str) -> bool:
    for t in tf_candidates(tf):
        # Search anywhere under data_root
        matches = list(data_root.rglob(f"{sym.upper()}_{t}.csv"))
        if matches:
            return True
    return False

def pick_first_available_tf(data_root: Path, sym: str, allowed_tfs: List[str]) -> Optional[str]:
    """Pick the first TF from allowed_tfs that exists on disk for this symbol."""
    for tf in allowed_tfs:
        if csv_exists(data_root, sym, tf):
            return tf
    return None

def load_universe_map(path: str) -> Dict[str, Any]:
    if not path:
        return {}
    if not os.path.exists(path):
        return {}
    if path.lower().endswith((".yaml", ".yml")):
        try:
            import yaml  # type: ignore
        except Exception:
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def latest_screener_file(folder: str = "reports/screener") -> Optional[str]:
    files = sorted(glob.glob(os.path.join(folder, "*_intraday.json")), key=os.path.getmtime)
    if files:
        return files[-1]
    files = sorted(glob.glob(os.path.join(folder, "*.json")), key=os.path.getmtime)
    return files[-1] if files else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", help="candidates JSON; if omitted, auto-picks latest in reports/screener")
    ap.add_argument("--out", required=True, help="filtered JSON output path")
    ap.add_argument("--data", default="data", help="data root")
    ap.add_argument("--tfs", default="M15,H4", help="canonical TFs to keep/fallback to (comma sep)")
    ap.add_argument("--universe-map", default="", help="config/screener_universe.yaml (optional) for TF fallback")
    args = ap.parse_args()

    data_root = Path(args.data)
    keep_tfs = [t.strip().upper() for t in args.tfs.split(",") if t.strip()]

    # Locate input file
    in_file = args.inp or latest_screener_file()
    if not in_file or not os.path.exists(in_file):
        raise SystemExit(f"[ERR] candidates JSON not found. Looked at: {in_file or '(auto)'}")

    # Load universe map (optional) for symbol_timeframes fallback
    uni = load_universe_map(args.universe_map)
    symbol_timeframes: Dict[str, List[str]] = (uni.get("symbol_timeframes") or {}) if isinstance(uni, dict) else {}

    # Load candidates
    with open(in_file, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict) and "candidates" in raw:
        candidates = raw["candidates"]
    elif isinstance(raw, list):
        candidates = raw
    else:
        raise SystemExit("[ERR] Unexpected candidate file structure.")

    kept: List[Dict[str, Any]] = []
    meta = {"total_in": len(candidates), "kept": 0, "skipped": 0, "retimeframed": 0}

    for c in candidates:
        sym = (c.get("pair") or c.get("symbol") or c.get("asset") or "").upper()
        tf  = (c.get("tf")   or c.get("timeframe") or "UNK").upper()
        if not sym or tf == "UNK":
            meta["skipped"] += 1
            continue

        clazz = classify_symbol(sym)
        if clazz not in {"forex", "crypto"}:
            meta["skipped"] += 1
            continue

        # If we have the requested TF on disk, keep as-is
        if csv_exists(data_root, sym, tf):
            kept.append(c)
            continue

        # Otherwise, try allowed TFs (e.g., M15,H4) then universe-map fallback
        preferred = pick_first_available_tf(data_root, sym, keep_tfs)
        if preferred:
            c2 = dict(c)
            c2["tf"] = preferred
            kept.append(c2)
            meta["retimeframed"] += 1
            continue

        # Try universe map symbol_timeframes (if provided)
        tf_opts = symbol_timeframes.get(sym, [])
        if tf_opts:
            preferred2 = pick_first_available_tf(data_root, sym, tf_opts)
            if preferred2:
                c2 = dict(c)
                c2["tf"] = preferred2
                kept.append(c2)
                meta["retimeframed"] += 1
                continue

        # No data for this symbol/TF combo
        meta["skipped"] += 1

    meta["kept"] = len(kept)
    out = {"candidates": kept, "meta": meta, "source": os.path.abspath(in_file)}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[OK] kept={meta['kept']} skipped={meta['skipped']} retimeframed={meta['retimeframed']} -> {args.out}")
    print(f"[SRC] {in_file}")

if __name__ == "__main__":
    main()
