#!/usr/bin/env python3
"""
Aggregate daily & monthly returns from an equity curve.

Usage:
  python tools/aggregate_returns.py --equity_csv reports/equity.csv

Outputs:
  - reports/returns_daily.csv
  - reports/returns_monthly.csv
  - reports/returns_summary.json
"""

import argparse
import pandas as pd
import numpy as np
import json
import os
import sys
from datetime import datetime


def read_equity(path: str) -> pd.DataFrame:
    """Read equity CSV and ensure timestamp/equity columns exist."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if "timestamp" not in df.columns or "equity" not in df.columns:
        raise ValueError("equity_csv must contain columns: timestamp,equity")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")
    df = df.set_index("timestamp")
    return df


def compute_daily_monthly(df: pd.DataFrame):
    """Compute daily & monthly returns DataFrames."""
    daily = df["equity"].resample("1D").last().dropna()
    daily_ret = daily.pct_change().dropna()
    daily_df = daily_ret.to_frame("ret").reset_index()
    daily_df["ret"] = daily_df["ret"].round(6)

    # ✅ FIXED: use "ME" instead of deprecated "M"
    monthly = df["equity"].resample("ME").last().dropna()
    monthly_ret = monthly.pct_change().dropna()
    monthly_df = monthly_ret.to_frame("ret").reset_index()
    monthly_df["month"] = monthly_df["timestamp"].dt.to_period("M").astype(str)
    monthly_df["ret"] = monthly_df["ret"].round(6)

    return daily_df, monthly_df


def sharpe_ratio(returns: pd.Series, rf=0.0, periods_per_year=252):
    """Compute annualized Sharpe ratio."""
    if returns.std(ddof=0) == 0 or len(returns) < 2:
        return 0.0
    excess = returns - rf / periods_per_year
    return float(np.sqrt(periods_per_year) * excess.mean() / excess.std(ddof=0))


def max_drawdown(equity: pd.Series) -> float:
    """Compute max drawdown from equity curve."""
    rollmax = equity.cummax()
    dd = equity / rollmax - 1.0
    return float(dd.min())


def save_csvs_and_summary(daily_df, monthly_df, df, out_dir="reports"):
    os.makedirs(out_dir, exist_ok=True)

    daily_path = os.path.join(out_dir, "returns_daily.csv")
    monthly_path = os.path.join(out_dir, "returns_monthly.csv")
    summary_path = os.path.join(out_dir, "returns_summary.json")

    daily_df.to_csv(daily_path, index=False)
    monthly_df.to_csv(monthly_path, index=False)

    # Summary stats
    daily_ret = daily_df["ret"]
    hit_rate = float((daily_ret > 0).sum() / len(daily_ret)) if len(daily_ret) > 0 else 0.0
    total_return = (df["equity"].iloc[-1] / df["equity"].iloc[0]) - 1.0
    sharpe = sharpe_ratio(daily_ret)
    mdd = max_drawdown(df["equity"])

    summary = {
        "days": int(len(daily_df)),
        "cum_return": float(total_return),
        "ann_sharpe": float(sharpe),
        "hit_rate": float(hit_rate),
        "max_drawdown": float(mdd),
        "daily_csv": daily_path.replace("/", "\\"),
        "monthly_csv": monthly_path.replace("/", "\\")
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser(description="Aggregate daily & monthly returns")
    ap.add_argument("--equity_csv", type=str, required=True,
                    help="CSV with columns [timestamp,equity]")
    ap.add_argument("--out_dir", type=str, default="reports",
                    help="Output folder for aggregated returns")
    args = ap.parse_args()

    try:
        df = read_equity(args.equity_csv)
        daily_df, monthly_df = compute_daily_monthly(df)
        save_csvs_and_summary(daily_df, monthly_df, df, args.out_dir)
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
