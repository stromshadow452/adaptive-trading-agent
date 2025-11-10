#!/usr/bin/env python3
"""
tools/paper_deploy.py
Simple paper deploy simulator: for each candidate, run simple_backtest() and save trades/equity summary.
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse, json, os
from pathlib import Path
from datetime import datetime
import pandas as pd

from src.data_loader import read_csv
from src.indicators import ema, rsi, atr
from src.backtest import simple_backtest

def load_candidates(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def prepare_df_for_backtest(pair, tf, raw_root="data/raw"):
    # try common processed path first
    cand_path = Path(f"data/datasets/{pair}_{tf}_processed.csv")
    if cand_path.exists():
        df = pd.read_csv(cand_path, index_col=0, parse_dates=True)
        return df
    # fallback to raw
    for p in Path(raw_root).rglob(f"*{pair}*{tf}*.csv"):
        df = read_csv(str(p))
        return df
    raise FileNotFoundError(f"No data for {pair} {tf}")

def add_features(df):
    df = df.copy()
    df["ema8"] = ema(df["Close"], span=8)
    df["ema21"] = ema(df["Close"], span=21)
    df["rsi14"] = rsi(df["Close"], period=14)
    df["atr14"] = atr(df, period=14)
    df = df.dropna()
    return df

def run_paper(candidates_json, top=1, equity=10000, out_dir="reports/daily_logs"):
    cand = load_candidates(candidates_json)
    if isinstance(cand, dict) and "candidates" in cand:
        cand = cand["candidates"]
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    results = []
    for item in cand[:top]:
        pair = item.get("pair")
        tf = item.get("tf")
        try:
            df = prepare_df_for_backtest(pair, tf)
            df = add_features(df)
            metrics, sig, equity_series = simple_backtest(df, {})
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            prefix = f"{pair}_{tf}_paper_{ts}"
            # save equity series
            eq_path = Path(out_dir) / f"{prefix}_equity.csv"
            equity_series.to_csv(eq_path, index=True, header=["equity"])
            # save metrics summary
            summary_path = Path(out_dir) / f"{prefix}_summary.json"
            summary_path.write_text(json.dumps(metrics, indent=2))
            results.append({"pair":pair, "tf":tf, "metrics":metrics, "equity_csv":str(eq_path)})
            print("Saved paper result:", eq_path)
        except Exception as e:
            print("Error for", item, e)
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True, help="JSON file from screener / predeploy")
    ap.add_argument("--top", type=int, default=1)
    ap.add_argument("--equity", type=float, default=10000)
    ap.add_argument("--out", default="reports/daily_logs")
    args = ap.parse_args()
    run_paper(args.candidates, top=args.top, equity=args.equity, out_dir=args.out)

if __name__ == "__main__":
    main()
