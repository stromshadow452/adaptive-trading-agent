#!/usr/bin/env python3
"""
tools/resample_tf.py

Resample existing unified OHLCV store timeframes (e.g. 5m -> 15m, 15m -> 1h).
Reads per-symbol / per-tf year-split files under <root>/<SYMBOL>/<TF>/<YEAR>.(parquet|csv)
Writes resampled files to <root>/<SYMBOL>/<TARGET_TF>/<YEAR>.(parquet|csv)

Usage examples:
  python tools/resample_tf.py --root data/unified --from_tf 5m --to_tf 15m --symbols EURUSD,GBPUSD --write parquet
  python tools/resample_tf.py --root data/unified --from_tf 15m --to_tf 1h --overwrite

Notes:
 - Input files may be parquet or csv. Parquet preferred.
 - Target timeframe must be a multiple of source timeframe (e.g. 5m -> 15m, not 7m -> 30m).
 - Ensure timestamps are timezone-aware (UTC) or naive but consistent.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from datetime import timedelta
from typing import List, Optional

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
    TQDM = True
except Exception:
    TQDM = False


TF_ALIAS = {
    "1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min",
    "1h": "1h", "4h": "4h", "1d": "1d"
}

PANDAS_RULE = {
    "1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min",
    "1h": "1h", "4h": "4h", "1d": "1d"
}


def has_parquet_engine() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except Exception:
        try:
            import fastparquet  # noqa: F401
            return True
        except Exception:
            return False


def list_symbols(root: str) -> List[str]:
    # top-level dirs under root are symbols
    syms = []
    if not os.path.isdir(root):
        return syms
    for s in sorted(os.listdir(root)):
        sp = os.path.join(root, s)
        if os.path.isdir(sp):
            syms.append(s)
    return syms


def available_timeframes_for_symbol(root: str, sym: str) -> List[str]:
    p = os.path.join(root, sym)
    if not os.path.isdir(p):
        return []
    return [d for d in sorted(os.listdir(p)) if os.path.isdir(os.path.join(p, d))]


def year_files(root: str, sym: str, tf: str) -> List[str]:
    dirp = os.path.join(root, sym, tf)
    if not os.path.isdir(dirp):
        return []
    # pick both parquet and csv
    files = glob.glob(os.path.join(dirp, "*.*"))
    # prefer parquet first when both exist for same year
    files_by_year = {}
    for f in files:
        name = os.path.basename(f)
        year = os.path.splitext(name)[0]
        try:
            int(year)
        except Exception:
            # try to extract year token from filename
            import re
            m = re.search(r"(20\d{2})", name)
            year = m.group(1) if m else name
        # choose parquet over csv
        ext = os.path.splitext(f)[1].lower()
        existing = files_by_year.get(year)
        if existing is None:
            files_by_year[year] = f
        else:
            # choose parquet if present
            if existing.endswith(".csv") and ext in (".parquet", ".pq"):
                files_by_year[year] = f
    return [files_by_year[k] for k in sorted(files_by_year.keys())]


def read_frame(path: str) -> pd.DataFrame:
    if path.lower().endswith(".parquet") or path.lower().endswith(".pq"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def write_frame(df: pd.DataFrame, path: str, write_mode: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if write_mode == "parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def tf_to_rule(tf: str) -> str:
    tfn = tf.lower()
    if tfn in PANDAS_RULE:
        return PANDAS_RULE[tfn]
    # naive mapping: if endswith 'm' -> minutes
    if tfn.endswith("m"):
        n = tfn[:-1]
        return f"{n}min"
    return tfn


def is_multiple(src: str, tgt: str) -> bool:
    # check whether target is a multiple of source (minutes/hours/days)
    def to_minutes(tf: str) -> Optional[int]:
        s = tf.lower()
        if s.endswith("min"):
            try:
                return int(s.replace("min", ""))
            except Exception:
                return None
        if s.endswith("m") and s[:-1].isdigit():
            return int(s[:-1])
        if s.endswith("h"):
            return int(s.replace("h", "")) * 60
        if s.endswith("d"):
            return int(s.replace("d", "")) * 60 * 24
        return None

    src_min = to_minutes(tf_to_rule(src))
    tgt_min = to_minutes(tf_to_rule(tgt))
    if src_min is None or tgt_min is None:
        return False
    return tgt_min % src_min == 0


def resample_dataframe(df: pd.DataFrame, src_rule: str, tgt_rule: str) -> pd.DataFrame:
    # expects df has columns: timestamp, open, high, low, close, volume, symbol, timeframe, asset
    if "timestamp" not in df.columns:
        raise ValueError("timestamp column missing")
    # ensure datetime index
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp")
    df = df.set_index("timestamp")

    # OHLC aggregation
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    }

    # resample with label='right' and closed='right' so intervals like 00:00-00:14 -> 00:15
    res = df.resample(tgt_rule, label="right", closed="right").agg(agg)
    # carry forward symbol/asset/timeframe by taking last non-null from original (they're constant per group)
    last_meta = df[["symbol", "timeframe", "asset"]].resample(tgt_rule, label="right", closed="right").last()
    out = res.join(last_meta)
    out = out.dropna(subset=["open", "high", "low", "close"])
    # reset index to column
    out = out.reset_index()
    # ensure proper columns order
    cols = ["timestamp", "open", "high", "low", "close", "volume", "symbol", "timeframe", "asset"]
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    return out[cols]


def run_resample(root: str, from_tf: str, to_tf: str, symbols: Optional[List[str]], write_mode: str, overwrite: bool, min_rows: int, dry_run: bool):
    if not is_multiple(from_tf, to_tf):
        print(f"ERROR: target timeframe {to_tf} is not a multiple of source timeframe {from_tf}. Aborting.")
        return 2

    syms = symbols or list_symbols(root)
    if not syms:
        print(f"No symbols found under {root}")
        return 3

    total_tasks = 0
    tasks = []
    for s in syms:
        if symbols and s.upper() not in [x.upper() for x in symbols]:
            continue
        files = year_files(root, s, from_tf)
        for f in files:
            tasks.append((s, f))
            total_tasks += 1

    it = tqdm(tasks, desc="Resampling", unit="file") if TQDM else tasks
    for sym, fp in it:
        try:
            input_df = read_frame(fp)
            if input_df.shape[0] < min_rows:
                print(f"skip {fp} (rows<{min_rows})")
                continue

            # resample (per-file year) - preserve asset & symbol
            tgt_rule = tf_to_rule(to_tf)
            src_rule = tf_to_rule(from_tf)
            out_df = resample_dataframe(input_df, src_rule, tgt_rule)
            if out_df.empty:
                print(f"no rows after resample: {fp}")
                continue

            # write into symbol/target_tf/year.ext
            year = pd.to_datetime(out_df["timestamp"].iloc[0], utc=True).year
            out_dir = os.path.join(root, sym, to_tf)
            ext = ".parquet" if write_mode == "parquet" else ".csv"
            out_path = os.path.join(out_dir, f"{year}{ext}")
            if os.path.exists(out_path) and not overwrite:
                # merge with existing target file (append & dedupe)
                existing = read_frame(out_path)
                merged = pd.concat([existing, out_df], ignore_index=True)
                merged = merged.sort_values("timestamp").drop_duplicates(subset=["timestamp", "symbol", "timeframe"])
                if dry_run:
                    print(f"[dry] would merge -> {out_path} (rows {existing.shape[0]} + {out_df.shape[0]} => {merged.shape[0]})")
                else:
                    write_frame(merged, out_path, write_mode)
                    print(f"merged -> {out_path} ({merged.shape[0]} rows)")
            else:
                if dry_run:
                    print(f"[dry] would write -> {out_path} ({out_df.shape[0]} rows)")
                else:
                    write_frame(out_df, out_path, write_mode)
                    print(f"wrote -> {out_path} ({out_df.shape[0]} rows)")

        except Exception as e:
            print(f"ERROR processing {fp}: {e}", file=sys.stderr)
            continue

    return 0


def parse_args():
    p = argparse.ArgumentParser(description="Resample OHLCV timeframe in unified store")
    p.add_argument("--root", default="data/unified", help="unified store root")
    p.add_argument("--from_tf", required=True, help="source timeframe (e.g. 5m,15m,1h)")
    p.add_argument("--to_tf", required=True, help="target timeframe (e.g. 15m,1h,4h)")
    p.add_argument("--symbols", default="", help="comma-separated symbols to process (default=all)")
    p.add_argument("--write", choices=["parquet", "csv"], default="parquet", help="output format")
    p.add_argument("--overwrite", action="store_true", help="overwrite target files instead of merging")
    p.add_argument("--min_rows", type=int, default=10, help="skip source files with fewer than min_rows")
    p.add_argument("--dry_run", action="store_true", help="show actions but don't write files")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    syms = [s.strip() for s in args.symbols.split(",") if s.strip()] if args.symbols else None
    # if parquet chosen but engine missing, fallback to csv
    write_mode = args.write
    if write_mode == "parquet" and not has_parquet_engine():
        print("[warn] parquet engine not found; falling back to CSV writes.")
        write_mode = "csv"

    rc = run_resample(root=args.root, from_tf=args.from_tf, to_tf=args.to_tf, symbols=syms,
                      write_mode=write_mode, overwrite=args.overwrite, min_rows=args.min_rows, dry_run=args.dry_run)
    sys.exit(rc)
