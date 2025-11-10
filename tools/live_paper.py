# tools/live_paper.py
import argparse
import csv
import os
from datetime import datetime, UTC

import numpy as np
import pandas as pd

# We import from tools.paper_trader for consistency;
# if your environment can’t resolve package paths, you can inline load_df/run_paper.
from tools.paper_trader import load_df, run_paper, LOG_DIR  # type: ignore


SUMMARY_PATH = os.path.join(LOG_DIR, "production_summary.csv")
os.makedirs(LOG_DIR, exist_ok=True)


def _append_summary_row(stamp: str, symbol: str, tf: str, metrics: dict, log_rel_path: str):
    """Append a row to production_summary.csv (create if needed)."""
    header = ["stamp", "symbol", "tf", "sharpe", "total_return", "bars", "log"]
    exists = os.path.exists(SUMMARY_PATH)
    mode = "a" if exists else "w"
    with open(SUMMARY_PATH, mode, newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(header)
        w.writerow([
            stamp,
            symbol,
            tf,
            metrics.get("sharpe", 0.0),
            metrics.get("total_return", 0.0),
            metrics.get("bars", 0),
            log_rel_path.replace("\\", "/"),
        ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True, type=str)
    ap.add_argument("--tf", required=True, type=str)
    ap.add_argument("--initial_equity", type=float, default=10_000.0)
    args = ap.parse_args()

    symbol = args.symbol.upper()
    tf = args.tf.upper()

    # load latest processed data
    df, _ = load_df(symbol, tf)

    # run the paper strategy (uses EMA/RSI if present)
    metrics, log_df = run_paper(df, initial_equity=args.initial_equity, tf=tf)

    # save a production log: Date, price, ret, signal, pos, strat_ret, equity
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out_name = f"{symbol}_{tf}_processed__prod_{stamp}.csv"
    out_path = os.path.join(LOG_DIR, out_name)
    log_df.to_csv(out_path, index=False)

    # update a summary CSV
    _append_summary_row(stamp, symbol, tf, metrics, os.path.join(LOG_DIR, out_name))

    print(f"✅ Prod log: {out_path}")
    print(metrics)


if __name__ == "__main__":
    main()
