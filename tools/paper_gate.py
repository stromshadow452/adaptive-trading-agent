#!/usr/bin/env python3
"""
paper_gate.py
One-shot pipeline: paper-trade -> predeploy gate -> aggregate -> register -> (optional) promote.

Usage example:
  python tools/paper_gate.py --symbol EURUSD --tf M15 \
    --data_root data/raw/forex_kaggle_multiTF \
    --model_dir models/live \
    --start 2025-06-01 --end 2025-08-31 \
    --cost_bps 1 --min_sharpe 1.2 --max_dd -0.2 --min_winrate 0.30 \
    --promote_if_pass --prefer_newer

Outputs:
  - reports/<auto>_equity.csv, _trades.csv, _summary.json  (from paper_trader)
  - reports/predeploy_report.json                          (from predeploy_checks)
  - reports/returns_daily.csv, returns_monthly.csv         (from aggregate_returns)
  - models/paper_pool/<name>/metrics.json                  (from register_model_from_predeploy)
  - models/live -> promoted model (if PASS + --promote_if_pass)
  - reports/deploy_summary.json                            (this script)
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

def run(cmd: list[str], check=True) -> subprocess.CompletedProcess:
    print(f"\n🚀 Running: {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, cwd=str(ROOT), check=check)

def last_created(pattern: str) -> Path | None:
    """Return most-recently modified file under reports matching glob pattern."""
    files = sorted(REPORTS.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None

def main():
    ap = argparse.ArgumentParser(description="Paper->Gate->(Register)->(Promote)")
    # paper-trader inputs
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--tf", required=True)
    ap.add_argument("--data_root", default="data/raw/forex_kaggle_multiTF")
    ap.add_argument("--model_dir", required=True, help="models/live or explicit checkpoint zip/folder")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--cost_bps", type=float, default=1.0)
    ap.add_argument("--initial_equity", type=float, default=100000.0)
    ap.add_argument("--epsilon", type=float, default=0.0, help="ε-greedy exploration during paper trading (0..1)")
    ap.add_argument("--deterministic", action="store_true")

    # gate thresholds
    ap.add_argument("--min_sharpe", type=float, default=1.2)
    ap.add_argument("--max_dd", type=float, default=-0.2, help="negative value (e.g., -0.2)")
    ap.add_argument("--min_winrate", type=float, default=0.45)
    ap.add_argument("--max_exposure", type=float, default=0.6)
    ap.add_argument("--max_turnover", type=float, default=20.0)
    ap.add_argument("--min_days", type=int, default=60)

    # promotion controls
    ap.add_argument("--promote_if_pass", action="store_true", help="Register + promote automatically when gate PASSes")
    ap.add_argument("--prefer_newer", action="store_true", help="Hint to promoter to tie-break in favor of newer")
    ap.add_argument("--force_name", default=None, help="Force promotion of this candidate name after PASS")
    ap.add_argument("--name_prefix", default=None, help="Candidate name prefix; default auto")
    args = ap.parse_args()

    # 1) PAPER-TRADE
    pt_cmd = [
        sys.executable, "tools/paper_trader.py",
        "--symbol", args.symbol, "--tf", args.tf,
        "--data_root", args.data_root, "--model_dir", args.model_dir,
        "--start", args.start, "--end", args.end,
        "--cost_bps", str(args.cost_bps),
        "--initial_equity", str(args.initial_equity),
    ]
    if args.deterministic:
        pt_cmd.append("--deterministic")
    if args.epsilon and args.epsilon > 0:
        pt_cmd += ["--epsilon", str(args.epsilon)]
    run(pt_cmd)

    # get latest outputs from paper trader
    eq = last_created(f"paper_{args.symbol}_{args.tf}_*_equity.csv")
    tr = last_created(f"paper_{args.symbol}_{args.tf}_*_trades.csv")
    if not (eq and tr and eq.exists() and tr.exists()):
        print("❌ Could not find paper-trader outputs.", file=sys.stderr)
        sys.exit(2)

    # 2) GATE (predeploy)
    gate_cmd = [
        sys.executable, "tools/predeploy_checks.py",
        "--equity_csv", str(eq),
        "--trades_csv", str(tr),
        "--min_sharpe", str(args.min_sharpe),
        "--max_dd", str(args.max_dd),
        "--min_winrate", str(args.min_winrate),
        "--max_exposure", str(args.max_exposure),
        "--max_turnover", str(args.max_turnover),
        "--min_days", str(args.min_days),
    ]
    cp = run(gate_cmd, check=False)
    # predeploy_checks non-zero exit means FAIL; still produce report JSON
    report_path = REPORTS / "predeploy_report.json"
    if not report_path.exists():
        print("❌ predeploy_report.json not found (gate step).", file=sys.stderr)
        sys.exit(3)

    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    result = report.get("result", "FAIL")
    passed = (result == "PASS")

    # 3) Aggregate returns (for dashboards/journaling)
    run([sys.executable, "tools/aggregate_returns.py", "--equity_csv", str(eq)], check=False)

    summary = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "paper": {"equity_csv": str(eq), "trades_csv": str(tr)},
        "gate": report,
        "promoted": False,
        "promoted_name": None,
        "live_path": None,
        "notes": "",
    }

    # 4) If PASS & requested, register + promote
    if passed and args.promote_if_pass:
        # candidate name
        tstamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        name_prefix = args.name_prefix or f"ppo_{args.symbol}_{args.tf}_paper"
        cand_name = f"{name_prefix}_{tstamp}"

        # register model pointing to current model_dir (so pool has the files)
        reg_cmd = [
            sys.executable, "tools/register_model_from_predeploy.py",
            "--name", cand_name,
            "--equity_csv", str(eq),
            "--copy_from", args.model_dir,
        ]
        run(reg_cmd)

        # promote
        prom_cmd = [sys.executable, "tools/promote_model.py", "--require_predeploy_pass"]
        if args.prefer_newer:
            prom_cmd.append("--prefer_newer")
        if args.force_name:
            prom_cmd += ["--force", args.force_name]
        run(prom_cmd)

        # live marker
        live_path = ROOT / "models" / "live"
        summary["promoted"] = True
        summary["promoted_name"] = args.force_name or cand_name
        summary["live_path"] = str(live_path)

        print(f"\n✅ DEPLOYED: {summary['promoted_name']} -> models/live")

    else:
        if not passed:
            summary["notes"] = "Gate FAILED; not promoted."
            print("\n⚠️ Gate FAILED; not promoted.")

    # 5) Write deploy summary JSON (for display)
    deploy_json = REPORTS / "deploy_summary.json"
    with open(deploy_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n📝 Summary saved: {deploy_json}")

    # concise display line (for dashboards / CI logs)
    if summary["promoted"]:
        print(f"\n=== LIVE STRATEGY: {summary['promoted_name']} (PASS) ===")
    else:
        print(f"\n=== LIVE UNCHANGED (Gate {result}) ===")

    # Return non-zero if FAIL so CI can notice
    sys.exit(0 if passed or not args.promote_if_pass else 0)

if __name__ == "__main__":
    main()
