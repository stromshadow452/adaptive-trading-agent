#!/usr/bin/env python3
"""
Scan ./data/** for <SYMBOL>_<TF>.csv and write a universe YAML/JSON with only available assets.
Usage:
  python tools/universe_from_data.py --out config/screener_universe.yaml --classes forex,crypto --tfs M15,H4
"""
import argparse, re, json, os
from pathlib import Path
from typing import Dict, Set, List

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data", help="root data folder")
    ap.add_argument("--out", required=True, help="output file (.yaml or .json)")
    ap.add_argument("--classes", default="forex,crypto", help="which classes to keep (comma sep)")
    ap.add_argument("--tfs", default="M15,H4", help="timeframes to include (canonical tokens)")
    args = ap.parse_args()

    data_root = Path(args.data)
    keep_classes = {c.strip() for c in args.classes.split(",") if c.strip()}
    keep_tfs = [t.strip().upper() for t in args.tfs.split(",") if t.strip()]
    # expand synonyms
    all_tf_tokens: Set[str] = set()
    for t in keep_tfs:
        all_tf_tokens |= TF_SYNONYMS.get(t, {t})
        all_tf_tokens.add(t)

    # find files like SYMBOL_TF.csv anywhere under data_root
    pat = re.compile(r"^([A-Za-z0-9\-]+)_([A-Za-z0-9]+)\.csv$")
    found: Dict[str, Set[str]] = {}
    for p in data_root.rglob("*.csv"):
        m = pat.match(p.name)
        if not m:
            continue
        sym, tf = m.group(1).upper(), m.group(2).upper()
        # keep only desired TFs (consider synonyms)
        if tf not in all_tf_tokens:
            continue
        # canonicalize TF to the requested canonical where possible
        canonical_tf = None
        for canon, syns in TF_SYNONYMS.items():
            if tf in syns or tf == canon:
                if canon in keep_tfs:
                    canonical_tf = canon
                    break
        if canonical_tf is None:
            continue
        # classify
        clazz = classify_symbol(sym)
        if clazz not in keep_classes:
            continue
        found.setdefault(sym, set()).add(canonical_tf)

    # split by class
    by_class: Dict[str, List[str]] = {"forex": [], "crypto": [], "other": []}
    tfs_for: Dict[str, List[str]] = {}
    for sym, tset in sorted(found.items()):
        clazz = classify_symbol(sym)
        by_class[clazz].append(sym)
        tfs_for[sym] = sorted(tset)

    # output
    out = {
        "universe": {
            "forex": sorted(by_class["forex"]),
            "crypto": sorted(by_class["crypto"]),
            # omit others from screener
        },
        "timeframes": keep_tfs,
        "symbol_timeframes": tfs_for,
    }

    # write yaml or json
    if args.out.lower().endswith(".yaml") or args.out.lower().endswith(".yml"):
        try:
            import yaml  # type: ignore
        except Exception:
            # fallback to json if PyYAML missing
            args.out = os.path.splitext(args.out)[0] + ".json"
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)
            print(f"[OK] wrote JSON -> {args.out} (PyYAML not installed)")
            return
        with open(args.out, "w", encoding="utf-8") as f:
            yaml.safe_dump(out, f, sort_keys=False)
        print(f"[OK] wrote YAML -> {args.out}")
    else:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"[OK] wrote JSON -> {args.out}")

if __name__ == "__main__":
    main()
