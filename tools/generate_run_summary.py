#!/usr/bin/env python3
"""
tools/generate_run_summary.py

Aggregate run health from executions and write:
- reports/aggregate/aggregate_table.csv
- reports/aggregate/aggregate_summary.json

It prefers realized executions at:
  reports/executions/executions.csv
(or executions_mark2close.csv if present)

If nothing is realized, counts will show 0 and PF will be 0 or inf as appropriate.

Usage:
  python tools/generate_run_summary.py --reports reports/daily_logs --out reports/aggregate --verbose
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Dict, List

EXEC_DIR = Path("reports/executions")
PREF_FILES = [
    EXEC_DIR / "executions_mark2close.csv",
    EXEC_DIR / "executions.csv",
]

def to_bool(x) -> bool:
    return str(x).strip().lower() in ("true", "1", "yes", "y")

def to_float(x, default=None):
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default

def find_executions_csv() -> Path | None:
    for p in PREF_FILES:
        if p.exists():
            return p
    return None

def read_realized_trades(path: Path) -> List[Dict]:
    out = []
    with path.open(newline="", encoding="utf-8") as fh:
        rdr = csv.DictReader(fh)
        for r in rdr:
            side = (r.get("side") or "").lower()
            if side not in ("buy", "sell"):
                continue
            if not to_bool(r.get("executed")):
                continue
            pnl = to_float(r.get("pnl"), None)
            if pnl is None:
                # allow immediate mark2close variants (may miss pnl if not computed)
                price = to_float(r.get("price"), None)
                exitp = to_float(r.get("exit_price"), None)
                qty = to_float(r.get("qty") or r.get("size"), None)
                if None not in (price, exitp, qty):
                    pnl = (exitp - price) * (qty if side == "buy" else -qty)
            if pnl is None:
                continue
            out.append({"symbol": r.get("symbol"), "pnl": pnl})
    return out

def compute_metrics(pnls: List[float]) -> Dict:
    trades = len(pnls)
    wins   = sum(1 for x in pnls if x > 0)
    losses = sum(1 for x in pnls if x < 0)

    gp = sum(x for x in pnls if x > 0)
    gl = -sum(x for x in pnls if x < 0)

    if gl > 0:
        pf = gp / gl
    else:
        pf = float("inf") if gp > 0 else 0.0

    if trades >= 2:
        try:
            mu = statistics.mean(pnls)
            sd = statistics.pstdev(pnls) or 1e-12
            sharpe = mu / sd
        except statistics.StatisticsError:
            sharpe = 0.0
    else:
        sharpe = 0.0

    # max drawdown on cumulative equity from pnl sequence
    eq = []
    s = 0.0
    peak = 0.0
    mdd = 0.0
    for x in pnls:
        s += x
        eq.append(s)
        peak = max(peak, s)
        mdd = max(mdd, peak - s)

    out = {
        "winrate": {"value": (wins / trades) if trades else 0.0, "ok": (wins / trades) >= 0.5 if trades else False},
        "profit_factor": {"value": pf, "ok": pf >= 1.2 if math.isfinite(pf) else False},
        "sharpe": {"value": sharpe, "ok": sharpe >= 0.5},
        "max_drawdown": {"value": mdd, "ok": mdd <= 0.0},  # zero or very small for intraday mark2close
        "counts": {"trades": trades, "wins": wins, "losses": losses},
    }
    return out

def write_table(out_dir: Path, trades: List[Dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / "aggregate_table.csv"
    with fp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["symbol", "pnl"])
        for t in trades:
            w.writerow([t["symbol"], f"{t['pnl']:.10f}"])
    print(f"Saved CSV -> {fp.as_posix()}")

def write_summary(out_dir: Path, summary: Dict) -> None:
    fp = out_dir / "aggregate_summary.json"
    with fp.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Saved JSON -> {fp.as_posix()}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", default="reports/daily_logs", help="(kept for compatibility; not used if executions CSV present)")
    ap.add_argument("--out", default="reports/aggregate")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    exec_csv = find_executions_csv()
    if not exec_csv:
        # No executions at all -> empty output
        empty = {
            "winrate": {"value": 0.0, "ok": False},
            "profit_factor": {"value": float("nan"), "ok": False},
            "sharpe": {"value": 0.0, "ok": False},
            "max_drawdown": {"value": 0.0, "ok": True},
            "counts": {"trades": 0, "wins": 0, "losses": 0},
        }
        write_table(out_dir, [])
        write_summary(out_dir, empty)
        if args.verbose:
            print("Aggregate health summary:")
            print(json.dumps(empty, indent=2))
        return

    trades = read_realized_trades(exec_csv)
    pnls = [t["pnl"] for t in trades]
    summary = compute_metrics(pnls)

    # For backwards compatibility include raw_trades if using mark2close pipeline
    summary["counts"]["raw_trades"] = 0

    write_table(out_dir, trades)
    write_summary(out_dir, summary)

    if args.verbose:
        print("Aggregate health summary:")
        print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
