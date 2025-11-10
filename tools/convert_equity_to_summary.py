#!/usr/bin/env python
"""
Convert paper equity CSV files into *_summary.json that train_meta_selector.py can use.

Input CSV expectation:
  - columns include at least: timestamp, equity  (optionally price)
Output JSON fields:
  - pair, tf (best-effort parse from filename)
  - sharpe (daily-like; simple mean/std of pct returns * sqrt(252) if intraday else sqrt(52) if weekly-ish)
  - total_return (final_equity / first_equity - 1)
  - bars (number of rows)
Usage:
  python tools/convert_equity_to_summary.py --csv_dir reports/daily_logs
"""
import argparse, glob, json, re
from pathlib import Path
import pandas as pd
import numpy as np

def parse_pair_tf_from_name(stem: str):
    # e.g. AUDUSD_H4_paper_20251020_171137_equity -> pair=AUDUSD, tf=H4
    m = re.match(r"([A-Za-z0-9]+)_([A-Za-z0-9]+)_paper_", stem)
    if m:
        return m.group(1), m.group(2)
    return None, None

def infer_sharpe_scale(tf: str):
    if not tf:
        return np.sqrt(252)  # default trading days
    tf = tf.lower()
    if tf.endswith("m"):  # minutes
        return np.sqrt(252)  # rough day-based scaling
    if tf.endswith("h"):  # hours
        return np.sqrt(252)
    if tf.endswith("d"):  # daily
        return np.sqrt(252)
    if tf.endswith("w"):  # weekly
        return np.sqrt(52)
    if tf.endswith("mth") or tf.endswith("mo"):  # monthly-ish
        return np.sqrt(12)
    return np.sqrt(252)

def process_file(path: Path):
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[skip] {path.name}: {e}")
        return None

    if "equity" not in df.columns:
        print(f"[skip] {path.name}: missing 'equity' column")
        return None

    eq = df["equity"].astype(float)
    bars = int(len(eq))
    if bars < 2:
        print(f"[skip] {path.name}: too few rows")
        return None

    ret = eq.pct_change().dropna()
    scale = infer_sharpe_scale(path.stem.split("_")[1] if "_" in path.stem else None)
    sharpe = float(np.nanmean(ret) / (np.nanstd(ret) + 1e-12) * scale) if len(ret) > 1 else 0.0
    total_return = float(eq.iloc[-1] / eq.iloc[0] - 1.0)

    pair, tf = parse_pair_tf_from_name(path.stem)
    out = {
        "pair": pair,
        "tf": tf,
        "sharpe": sharpe,
        "total_return": total_return,
        "bars": bars,
        "source_equity_csv": str(path)
    }
    out_name = path.with_name(path.stem.replace("_equity", "") + "_summary.json")
    with open(out_name, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("Saved summary:", out_name)
    return out_name

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_dir", default="reports/daily_logs")
    args = ap.parse_args()
    csvs = glob.glob(str(Path(args.csv_dir) / "*_equity.csv"))
    if not csvs:
        print("No *_equity.csv found in", args.csv_dir)
        return
    for c in csvs:
        process_file(Path(c))

if __name__ == "__main__":
    main()
