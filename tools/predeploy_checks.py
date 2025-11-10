#!/usr/bin/env python3
"""
Pre-deploy Safety Gate
- Computes: Sharpe, Max Drawdown, Turnover, Win Rate, Exposure, Days
- Inputs:  equity CSV (timestamp,equity) OR trades CSV (timestamp/pnl or entry_time/exit_time/size/price)
- Output:  JSON report + process exit code (0=PASS, 2=FAIL, 3=bad args)

Example:
  python tools/predeploy_checks.py \
    --equity_csv reports/equity.csv \
    --trades_csv reports/last_trades.csv \
    --min_sharpe 1.2 --max_dd -0.2 --min_winrate 0.45 --max_exposure 0.6 --min_days 60
"""

import argparse
import json
import math
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd


# ---------- IO helpers ----------
def read_csv_auto(path: str) -> pd.DataFrame:
    if not path:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def ensure_datetime(col: pd.Series) -> pd.Series:
    return pd.to_datetime(col, errors="coerce")


# ---------- Metrics ----------
def compute_equity_from_trades(trades: pd.DataFrame, start_equity: float = 100000.0) -> pd.DataFrame:
    """
    Build a daily equity curve from trade-level PnL.
    Required columns:
      - 'timestamp' (or 'time')
      - 'pnl' (or one of ['profit','pl','return','ret'])
    """
    df = trades.copy()

    # timestamp
    if "timestamp" not in df.columns:
        if "time" in df.columns:
            df = df.rename(columns={"time": "timestamp"})
        else:
            raise ValueError("trades must contain 'timestamp' or 'time' column")
    df["timestamp"] = ensure_datetime(df["timestamp"])

    # pnl
    if "pnl" not in df.columns:
        alt = [c for c in ["profit", "pl", "return", "ret"] if c in df.columns]
        if not alt:
            raise ValueError("trades must contain 'pnl' or one of ['profit','pl','return','ret']")
        df["pnl"] = df[alt[0]]

    # sort / dropna / equity
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df["equity"] = float(start_equity) + df["pnl"].astype(float).cumsum()

    # resample to daily last
    daily = (
        df.set_index("timestamp")["equity"]
        .resample("1D")
        .last()
        .ffill()
        .dropna()
    )
    return daily.to_frame(name="equity").reset_index(names="timestamp")


def max_drawdown(equity: pd.Series) -> float:
    rollmax = equity.cummax()
    dd = equity / rollmax - 1.0
    return float(dd.min())


def sharpe_ratio(returns: pd.Series, rf: float = 0.0, periods_per_year: int = 252) -> float:
    if returns is None or len(returns) < 2:
        return 0.0
    vol = returns.std(ddof=0)
    if vol == 0 or np.isnan(vol):
        return 0.0
    excess = returns - rf / periods_per_year
    return float(np.sqrt(periods_per_year) * excess.mean() / vol)


def turnover(trades: pd.DataFrame, equity0: float = 100000.0) -> float:
    """
    Approximate turnover as total notional traded / avg equity.
    Accepts:
      - 'notional' directly
      - or derive from 'size' * 'price'
      - or fallback from |pnl| * factor if nothing else exists
    """
    if trades is None or trades.empty:
        return 0.0

    df = trades.copy()
    if "notional" not in df.columns:
        if {"size", "price"}.issubset(df.columns):
            df["notional"] = (pd.to_numeric(df["size"], errors="coerce").abs()
                              * pd.to_numeric(df["price"], errors="coerce").abs())
        elif "pnl" in df.columns:
            # crude fallback
            df["notional"] = pd.to_numeric(df["pnl"], errors="coerce").abs() * 10.0
        else:
            return 0.0

    total_notional = pd.to_numeric(df["notional"], errors="coerce").fillna(0.0).abs().sum()
    avg_eq = equity0 if equity0 > 0 else 1.0
    return float(total_notional / avg_eq)


def win_rate(trades: pd.DataFrame) -> float:
    if trades is None or trades.empty:
        return 0.0
    if "pnl" not in trades.columns:
        return 0.0
    s = pd.to_numeric(trades["pnl"], errors="coerce")
    s = s.dropna()
    n = len(s)
    if n == 0:
        return 0.0
    return float((s > 0).sum() / n)


def exposure_pct(trades: pd.DataFrame) -> float:
    """
    Fraction of time in position over the evaluated span.

    Preferred calculation:
      Sum of (exit_time - entry_time) / (max(exit_time) - min(entry_time))

    Fallbacks:
      - If lifecycle times unavailable, use 'timestamp' range and a crude proxy
        (fraction of rows with non-zero 'size').
    """
    if trades is None or trades.empty:
        return 0.0
    df = trades.copy()

    # Preferred: lifecycle timestamps
    has_lifecycle = {"entry_time", "exit_time"}.issubset(df.columns)
    if has_lifecycle:
        et = ensure_datetime(df["entry_time"])
        xt = ensure_datetime(df["exit_time"])
        valid = et.notna() & xt.notna()
        if valid.any():
            hold = (xt[valid] - et[valid]).dt.total_seconds().clip(lower=0)
            total_hold = float(hold.sum())

            tmin = et[valid].min()
            tmax = xt[valid].max()
            total_span = (tmax - tmin).total_seconds()
            if total_span <= 0:
                return 0.0
            # clip to [0, 1.0] because overlapping trades can over-sum holds
            return float(np.clip(total_hold / total_span, 0.0, 1.0))

    # Fallback: use timestamp span + simple activity proxy
    if "timestamp" in df.columns:
        ts = ensure_datetime(df["timestamp"]).dropna()
        if ts.empty:
            return 0.0
        total_span = (ts.max() - ts.min()).total_seconds()
        if total_span <= 0:
            return 0.0
        if "size" in df.columns:
            in_pos = (pd.to_numeric(df["size"], errors="coerce").fillna(0.0) != 0).sum()
            return float(np.clip(in_pos / len(df), 0.0, 1.0))
        return 0.0

    return 0.0


# ---------- CLI / main ----------
def load_thresholds(args) -> dict:
    return {
        "min_sharpe": args.min_sharpe,
        "max_drawdown": args.max_dd,
        "max_turnover": args.max_turnover,
        "min_winrate": args.min_winrate,
        "max_exposure": args.max_exposure,
        "min_days": args.min_days,
    }


def main():
    ap = argparse.ArgumentParser(description="Pre-deploy safety checks")
    ap.add_argument("--trades_csv", type=str, default=None,
                    help="CSV of trades; should include timestamp & pnl (or profit/pl/return/ret). "
                         "Optional lifecycle columns: entry_time, exit_time, size, price, notional.")
    ap.add_argument("--equity_csv", type=str, default=None,
                    help="CSV with columns [timestamp,equity] (daily or higher frequency OK).")
    ap.add_argument("--report_json", type=str, default="reports/predeploy_report.json")

    ap.add_argument("--start_equity", type=float, default=100000.0,
                    help="Used if equity must be derived from trades.")
    ap.add_argument("--rf", type=float, default=0.0, help="Risk-free rate (annualized).")

    # thresholds
    ap.add_argument("--min_sharpe", type=float, default=1.2)
    ap.add_argument("--max_dd", type=float, default=-0.15,
                    help="Maximum allowed drawdown (negative value, e.g., -0.20).")
    ap.add_argument("--max_turnover", type=float, default=20.0)
    ap.add_argument("--min_winrate", type=float, default=0.45)
    ap.add_argument("--max_exposure", type=float, default=0.60)
    ap.add_argument("--min_days", type=int, default=60)

    # optional: relax win-rate if Sharpe is strong
    ap.add_argument("--min_winrate_when_sharpe_ok", type=float, default=None,
                    help="If set, and Sharpe passes, allow this lower win-rate threshold.")

    args = ap.parse_args()

    trades = None
    equity_df = None

    if args.trades_csv:
        trades = read_csv_auto(args.trades_csv)

    if args.equity_csv:
        equity_df = read_csv_auto(args.equity_csv)
        # normalize equity CSV
        if "timestamp" not in equity_df.columns or "equity" not in equity_df.columns:
            raise ValueError("equity_csv must have columns: timestamp,equity")
        equity_df["timestamp"] = ensure_datetime(equity_df["timestamp"])
        equity_df = equity_df.dropna(subset=["timestamp", "equity"]).sort_values("timestamp")

    if equity_df is None:
        if trades is None:
            print("ERROR: Provide either --equity_csv or --trades_csv", file=sys.stderr)
            sys.exit(3)
        equity_df = compute_equity_from_trades(trades, start_equity=args.start_equity)

    # daily equity & returns
    daily = (
        equity_df.set_index("timestamp")["equity"]
        .resample("1D")
        .last()
        .dropna()
    )
    daily_ret = daily.pct_change().dropna()
    days = int(daily_ret.shape[0])

    # metrics
    metrics = {
        "days": days,
        "sharpe": sharpe_ratio(daily_ret, rf=args.rf, periods_per_year=252),
        "max_drawdown": max_drawdown(daily),
    }
    if trades is not None:
        metrics["turnover"] = turnover(trades, equity0=float(daily.iloc[0]) if len(daily) else float(args.start_equity))
        metrics["win_rate"] = win_rate(trades)
        metrics["exposure"] = exposure_pct(trades)
    else:
        metrics["turnover"] = 0.0
        metrics["win_rate"] = 0.0
        metrics["exposure"] = 0.0

    thresholds = load_thresholds(args)

    checks = {
        "days_ok": metrics["days"] >= thresholds["min_days"],
        "sharpe_ok": metrics["sharpe"] >= thresholds["min_sharpe"],
        "dd_ok": metrics["max_drawdown"] >= thresholds["max_drawdown"],
        "turnover_ok": metrics["turnover"] <= thresholds["max_turnover"],
        "winrate_ok": metrics["win_rate"] >= thresholds["min_winrate"],
        "exposure_ok": metrics["exposure"] <= thresholds["max_exposure"],
    }

    # optional rule: if Sharpe passes, allow a lower win-rate floor
    if args.min_winrate_when_sharpe_ok is not None and checks["sharpe_ok"]:
        if metrics["win_rate"] >= float(args.min_winrate_when_sharpe_ok):
            checks["winrate_ok"] = True

    all_ok = all(checks.values())

    # write report
    os.makedirs(os.path.dirname(args.report_json), exist_ok=True)
    report = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "inputs": {
            "equity_csv": args.equity_csv,
            "trades_csv": args.trades_csv,
        },
        "metrics": metrics,
        "thresholds": thresholds | ({"min_winrate_when_sharpe_ok": args.min_winrate_when_sharpe_ok}
                                    if args.min_winrate_when_sharpe_ok is not None else {}),
        "checks": checks,
        "result": "PASS" if all_ok else "FAIL",
    }
    with open(args.report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # console echo (ASCII-safe)
    print(json.dumps(report, indent=2))

    # exit code
    sys.exit(0 if all_ok else 2)


if __name__ == "__main__":
    main()
