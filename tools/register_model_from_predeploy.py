#!/usr/bin/env python3
"""
Register a trained model into the paper pool using the latest predeploy report.

Examples
--------
# Minimal: use predeploy report only
python tools/register_model_from_predeploy.py \
  --name ppo_EURUSD_M15_20251016_1206

# With checkpoint copy (if you have a trained folder)
python tools/register_model_from_predeploy.py \
  --name ppo_EURUSD_M15_20251016_1206 \
  --copy_from "models/finrl/ppo_EURUSD_M15"

# Also compute total PnL/days directly from equity CSV
python tools/register_model_from_predeploy.py \
  --name ppo_EURUSD_M15_20251016_1206 \
  --equity_csv reports/equity.csv
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from datetime import datetime

try:
    import pandas as pd  # optional, only needed if --equity_csv given
except Exception:
    pd = None


def load_json(path: Path):
    """Robust JSON loader (handles UTF-8 with/without BOM)."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        try:
            return json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception as e:
            print(f"❌ Could not parse JSON: {path} ({e})", file=sys.stderr)
            return None


def compute_metrics_from_equity(equity_csv: Path):
    """Compute days and total_pnl from an equity curve CSV (timestamp,equity)."""
    if equity_csv is None or not equity_csv.exists():
        return None

    if pd is None:
        print("⚠️ pandas not available; cannot compute metrics from equity CSV.", file=sys.stderr)
        return None

    df = pd.read_csv(equity_csv)
    if "timestamp" not in df.columns or "equity" not in df.columns:
        print("⚠️ equity_csv must contain 'timestamp,equity' columns.", file=sys.stderr)
        return None

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")
    if df.shape[0] < 2:
        return None

    total_pnl = float(df["equity"].iloc[-1] - df["equity"].iloc[0])

    # daily resample to count days robustly
    daily = (
        df.set_index("timestamp")["equity"]
          .resample("1D")
          .last()
          .dropna()
    )
    days = int(max(len(daily) - 1, 0))

    return {"total_pnl": total_pnl, "days": days}


def main():
    ap = argparse.ArgumentParser(description="Register model from predeploy report into paper pool.")
    ap.add_argument("--report", default="reports/predeploy_report.json",
                    help="Path to predeploy report JSON (default: reports/predeploy_report.json)")
    ap.add_argument("--equity_csv", default=None,
                    help="Optional equity CSV to compute total_pnl and days (columns: timestamp,equity)")
    ap.add_argument("--name", required=True,
                    help="Target model name under models/paper_pool/<name>")
    ap.add_argument("--copy_from", default=None,
                    help="Optional path to trained model/checkpoint folder to copy under paper_pool/<name>/checkpoint")
    ap.add_argument("--tags", nargs="*", default=None,
                    help="Optional tags to store in metrics.json (e.g., --tags EURUSD M15 PPO)")
    ap.add_argument("--overwrite", action="store_true",
                    help="If set, overwrite existing model directory if it exists")
    args = ap.parse_args()

    report_path = Path(args.report)
    equity_path = Path(args.equity_csv) if args.equity_csv else None
    pool_dir = Path("models/paper_pool") / args.name

    # Load predeploy report (optional but recommended)
    rep = None
    if report_path.exists():
        rep = load_json(report_path)
        if rep is None:
            print("⚠️ Proceeding without report metrics (failed to parse).", file=sys.stderr)
    else:
        print(f"⚠️ Report file not found: {report_path} — continuing without it.", file=sys.stderr)

    # Base metrics from report (if available)
    metrics_from_report = rep.get("metrics", {}) if isinstance(rep, dict) else {}

    sharpe = float(metrics_from_report.get("sharpe", 0.0))
    max_drawdown = float(metrics_from_report.get("max_drawdown", 0.0))
    days = int(metrics_from_report.get("days", 0))
    total_pnl = float(metrics_from_report.get("cum_return", 0.0)) if "cum_return" in metrics_from_report else 0.0

    # If equity CSV is provided, compute PnL/days from equity (authoritative)
    computed = compute_metrics_from_equity(equity_path) if equity_path else None
    if computed:
        total_pnl = float(computed.get("total_pnl", total_pnl))
        days = int(computed.get("days", days))

    # Prepare destination directory
    if pool_dir.exists() and args.overwrite:
        shutil.rmtree(pool_dir)
    pool_dir.mkdir(parents=True, exist_ok=True)

    # Build metrics.json content
    metrics_out = {
        "name": args.name,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "total_pnl": total_pnl,
        "days": days,
        "sources": {
            "report": str(report_path) if report_path else None,
            "equity_csv": str(equity_path) if equity_path else None
        }
    }
    if args.tags:
        metrics_out["tags"] = list(args.tags)

    # Write metrics.json
    (pool_dir / "metrics.json").write_text(json.dumps(metrics_out, indent=2), encoding="utf-8")

    # Optionally copy checkpoint directory
    if args.copy_from:
        src = Path(args.copy_from)
        dst = pool_dir / "checkpoint"
        if not src.exists():
            print(f"⚠️ copy_from not found: {src}. Skipping checkpoint copy.", file=sys.stderr)
        else:
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)

    # Output
    print(f"✅ Registered new model at: {pool_dir}")
    print(json.dumps(metrics_out, indent=2))
    sys.exit(0)


if __name__ == "__main__":
    main()
