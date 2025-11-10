from __future__ import annotations

import argparse
import glob
import os
import re
from typing import List, Optional, Tuple

import pandas as pd

from src.data_loader import read_csv
from src.indicators import ema, rsi, atr
from src.backtest import simple_backtest


# ------------------------
# Helpers
# ------------------------

def normalize_tf(tf: str) -> str:
    """Accept '1h', '15m' etc and normalize to 'H1', 'M15'."""
    t = tf.strip().lower()
    m = re.fullmatch(r"(\d+)\s*([mh])", t)
    if m:
        num, unit = m.groups()
        return ("M" if unit == "m" else "H") + num
    return tf.strip().upper()


def infer_tf_from_filename(path: str) -> Optional[str]:
    """Try to infer timeframe token from filename like EURUSD_H1_2020.csv."""
    base = os.path.basename(path).upper()
    m = re.search(r"_(H\d+|M\d+)(?:[ _\.])", base)
    return m.group(1) if m else None


def _glob_many(patterns: List[str]) -> List[str]:
    files: List[str] = []
    for pat in patterns:
        files.extend(glob.glob(pat, recursive=True))
    return sorted(set(files))


def find_forex_files(symbol: str, tf: str, base_path: str) -> List[str]:
    """Find CSVs for a symbol/timeframe under a base path.

    Looks for both .../SYMBOL_TF_*.csv and .../SYMBOL_TF.csv.
    """
    tf_fx = normalize_tf(tf)
    sym = symbol.upper()
    patterns = [
        os.path.join(base_path, "**", f"{sym}_{tf_fx}_*.csv"),
        os.path.join(base_path, "**", f"{sym}_{tf_fx}.csv"),
    ]
    return _glob_many(patterns)


def load_and_prepare(files: List[str]) -> pd.DataFrame:
    if not files:
        raise FileNotFoundError("No files passed to load_and_prepare().")
    dfs = []
    for p in files:
        df = read_csv(p)
        # standardize column names: Open, High, Low, Close, (Volume optional)
        df = df.rename(columns={c: c.strip().title() for c in df.columns})
        required = {"Open", "High", "Low", "Close"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing columns {missing} in: {p}. Have: {list(df.columns)}"
            )
        dfs.append(df)
    merged = pd.concat(dfs, axis=0)
    merged = merged[~merged.index.duplicated(keep="last")]
    merged = merged.sort_index()
    return merged


def run_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["ema8"] = ema(df["Close"], span=8)
    df["ema21"] = ema(df["Close"], span=21)
    df["rsi14"] = rsi(df["Close"], period=14)
    df["atr14"] = atr(df, period=14)
    required = ["Open", "High", "Low", "Close", "ema8", "ema21", "rsi14", "atr14"]
    df = df.dropna(subset=[c for c in required if c in df.columns])
    return df


def resolve_tf_auto(symbol: str, base_path: str) -> str:
    """If --tf auto, try to detect timeframe from available files."""
    # Prefer H1 if available; else grab first matching file
    probe = find_forex_files(symbol, "H1", base_path)
    if not probe:
        probe = _glob_many(
            [
                os.path.join(base_path, "**", f"{symbol.upper()}_*.csv"),
                os.path.join(base_path, "**", f"{symbol.upper()}.csv"),
            ]
        )
    if not probe:
        raise FileNotFoundError(
            f"No files found under {base_path!r} for {symbol.upper()}."
        )
    detected = infer_tf_from_filename(probe[0])
    if not detected:
        raise RuntimeError(
            "Could not infer timeframe from filenames; please pass --tf explicitly."
        )
    return detected


# ------------------------
# Modes
# ------------------------

def smoke(symbol: str, tf: str, base_path: str):
    """Your original quick sanity run, now with configurable base_path."""
    if tf.strip().lower() == "auto":
        tf = resolve_tf_auto(symbol, base_path)

    files = find_forex_files(symbol, tf, base_path)

    print("CWD:", os.getcwd())
    print("Data path:", os.path.abspath(base_path))
    print("Symbol:", symbol.upper(), "TF:", normalize_tf(tf))
    print(
        "Glob pattern(s):",
        os.path.join(base_path, "**", f"{symbol.upper()}_{normalize_tf(tf)}_*.csv"),
        "and",
        os.path.join(base_path, "**", f"{symbol.upper()}_{normalize_tf(tf)}.csv"),
    )
    print(f"Matched files: {len(files)}")
    for f in files[:20]:
        print(" -", os.path.basename(f))
    if len(files) > 20:
        print(f" ... (+{len(files)-20} more)")

    if not files:
        raise FileNotFoundError(
            f"No CSV files found for {symbol}_{normalize_tf(tf)} under {base_path!r}"
        )

    df = load_and_prepare(files)
    print(f"Loaded total rows: {len(df):,} (from {len(files)} files)")

    df = run_indicators(df)
    # Compatibility: if indicator names are capitalized elsewhere
    for name in ["ema8", "ema21", "rsi14", "atr14"]:
        if name not in df.columns:
            alt = name[0].upper() + name[1:]
            if alt in df.columns:
                df[name] = df[alt]

    metrics, signals, equity = simple_backtest(df, {})
    print("Smoke metrics:", metrics)
    print("\nPreview (last 5 rows):")
    cols = [
        c
        for c in ["Open", "High", "Low", "Close", "ema8", "ema21", "rsi14", "atr14"]
        if c in df.columns
    ]
    print(df[cols].tail(5))


def backtest(symbol: str, tf: str, base_path: str):
    """Full backtest mode: same pipeline as smoke, with clearer reporting."""
    if tf.strip().lower() == "auto":
        tf = resolve_tf_auto(symbol, base_path)

    files = find_forex_files(symbol, tf, base_path)
    if not files:
        raise FileNotFoundError(
            f"No CSV files found for {symbol}_{normalize_tf(tf)} under {base_path!r}"
        )

    print("=== Backtest ===")
    print("CWD:", os.getcwd())
    print("Data path:", os.path.abspath(base_path))
    print("Symbol:", symbol.upper(), "TF:", normalize_tf(tf))
    print(f"Files: {len(files)} matched")

    df = load_and_prepare(files)
    df = run_indicators(df)

    metrics, signals, equity = simple_backtest(df, {})
    print("\nBacktest metrics:")
    for k, v in metrics.items():
        print(f" - {k}: {v}")

    print("\nEquity curve tail:")
    try:
        print(equity.tail(5))
    except Exception:
        print("(equity data unavailable)")

    print("\nSignals tail:")
    try:
        print(signals.tail(5))
    except Exception:
        print("(signals data unavailable)")


# ------------------------
# CLI
# ------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        default="smoke",
        choices=["smoke", "backtest"],
        help="Run mode. 'smoke' = quick check, 'backtest' = full run.",
    )
    ap.add_argument("--symbol", default="EURUSD")
    ap.add_argument(
        "--tf",
        default="auto",
        help="e.g., H1/H4/M15 or 1h/4h/15m or 'auto' to detect from files",
    )
    ap.add_argument(
        "--data_path",
        default="data/raw/forex",
        help="Root folder to search for CSVs (e.g. data/raw/forex_kaggle)",
    )
    args = ap.parse_args()

    if args.mode == "smoke":
        smoke(args.symbol, args.tf, args.data_path)
    elif args.mode == "backtest":
        backtest(args.symbol, args.tf, args.data_path)
    else:
        raise SystemExit("mode not implemented yet")


if __name__ == "__main__":
    main()
