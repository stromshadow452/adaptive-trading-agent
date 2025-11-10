#!/usr/bin/env python3
"""
Multi-source Screener (FX + Crypto)
- Auto-discovers all CSV roots under data/raw/
- Normalizes symbols (BTC-USDT -> BTCUSD; USDT/BUSD/USDC -> USD)
- Classifies kind: forex | crypto
- Deduplicates per (symbol, tf) keeping newest CSV
- Writes a merged candidates file

Usage (no need to list dirs):
  python scripts/build_screener_from_csv_multi.py ^
    --root data/raw ^
    --tfs M5 M15 H1 ^
    --out reports/screener
"""

from __future__ import annotations
import os, glob, json, argparse, re
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional

SEP_RE       = re.compile(r"[^A-Za-z]")   # strip -, _, /, etc.
QUOTE_MAP    = {"USDT": "USD", "BUSD": "USD", "USDC": "USD"}
FX_CCY       = {"USD","EUR","JPY","GBP","AUD","NZD","CAD","CHF"}
CRYPTO_BASES = {"BTC","ETH","SOL","XRP","BNB","ADA","DOGE","DOT","LTC","XMR","AVAX","SHIB","TRX","LINK","UNI","MATIC"}
METALS       = {"XAU","XAG"}

def now_utc_str() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00","Z")

def ts_for_filename() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

def normalize_symbol(sym: str) -> str:
    """'BTC-USDT'->'BTCUSD'; 'eth/usdt'->'ETHUSD'; 'EURUSD' stays."""
    if not sym: return ""
    s = SEP_RE.sub("", sym.upper())
    if len(s) < 6: return s
    base = s[:3]
    quote_raw = s[3:]
    quote = QUOTE_MAP.get(quote_raw, quote_raw[:3])
    return f"{base}{quote}"

def classify_symbol(sym: str) -> str:
    s = normalize_symbol(sym)
    if len(s) < 6: return "other"
    base, quote = s[:3], s[3:6]
    if base in CRYPTO_BASES: return "crypto"
    if (base in FX_CCY or base in METALS) and (quote in FX_CCY): return "forex"
    return "other"

def last_close_from_csv(path: str) -> Optional[float]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            header = f.readline().strip().split(",")
            lower  = [h.strip().lower() for h in header]
            if "close" not in lower: return None
            idx = lower.index("close")
            last = None
            for line in f:
                if line.strip():
                    last = line
            if not last: return None
            parts = last.strip().split(",")
            return float(parts[idx])
    except Exception:
        return None

def discover_csv_roots(root: str) -> List[str]:
    roots = []
    for entry in os.scandir(root):
        if entry.is_dir():
            roots.append(entry.path)
    return roots

def symbol_from_filename(fname: str) -> str:
    """EURUSD_M1.csv -> EURUSD; BTC-USDT_1m.csv -> BTC-USDT -> normalized later."""
    stem = os.path.basename(fname).rsplit(".", 1)[0]
    # cut at first underscore to drop TF fragments if any
    return stem.split("_")[0]

def scan_one_root(root: str) -> List[Tuple[str, str]]:
    """Return list of (csv_path, raw_symbol) for this root."""
    out: List[Tuple[str, str]] = []
    for path in glob.glob(os.path.join(root, "**", "*.csv"), recursive=True):
        raw = symbol_from_filename(path)
        out.append((path, raw))
    return out

def build_candidates(roots: List[str], tfs: List[str]) -> List[Dict]:
    # keep newest CSV per normalized symbol
    newest: Dict[str, Tuple[str, float]] = {}  # sym -> (path, mtime)
    for r in roots:
        for path, raw in scan_one_root(r):
            sym_norm = normalize_symbol(raw)
            if not sym_norm or len(sym_norm) < 6:
                continue
            mtime = os.path.getmtime(path)
            prev = newest.get(sym_norm)
            if (not prev) or (mtime > prev[1]):
                newest[sym_norm] = (path, mtime)

    cands: List[Dict] = []
    for sym, (csv_path, _) in newest.items():
        price = last_close_from_csv(csv_path)
        if price is None:
            continue
        kind = classify_symbol(sym)
        if kind not in {"forex","crypto"}:
            continue
        for tf in tfs:
            cands.append({
                "symbol": sym,
                "price": price,
                "tf": tf,
                "kind": kind,
                "csv_path": csv_path,
                "timestamp": now_utc_str()
            })
    return cands

def main():
    ap = argparse.ArgumentParser(description="Multi-source screener (FX+Crypto)")
    ap.add_argument("--root", default="data/raw", help="Top folder containing CSV subfolders (auto-discovers)")
    ap.add_argument("--tfs", nargs="+", default=["M5","M15","H1"])
    ap.add_argument("--out", default="reports/screener")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    roots = discover_csv_roots(args.root)
    if not roots:
        print(f"[ERR] No CSV roots found under {args.root}")
        return 2

    cands = build_candidates(roots, args.tfs)
    out_path = os.path.join(args.out, f"{ts_for_filename()}_multi.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"candidates": cands}, f, indent=2)
    print(f"[OK] {len(cands)} candidates saved -> {out_path}")
    print("[roots]", *roots, sep="\n- ")

if __name__ == "__main__":
    main()
