#!/usr/bin/env python3
"""
tools/auto_fix_rules.py

Reads an aggregate summary JSON and proposes safe rule tweaks.
Usage:
  python tools/auto_fix_rules.py --summary reports/aggregate/aggregate_summary.json \
    --out reports/aggregate/fixes_dryrun.json --dry-run
  python tools/auto_fix_rules.py --summary ... --out ... --apply
"""
import argparse
import json
from pathlib import Path
from datetime import datetime, timezone

def now_rfc3339_utc():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def suggest_rules(summary: dict) -> list[dict]:
    """Very conservative auto-fixes derived from health metrics."""
    rules = []
    win = float(summary.get("winrate", {}).get("value", 0.0) or 0.0)
    pf  = summary.get("profit_factor", {}).get("value", 0.0)
    sh  = float(summary.get("sharpe", {}).get("value", 0.0) or 0.0)
    mdd = summary.get("max_drawdown", {}).get("value", 0.0)

    # Normalize inf representation
    try:
        pf = float(pf)
    except (TypeError, ValueError):
        pf = float("inf") if str(pf).lower() in ("inf", "infinity") else 0.0

    # If PF is poor and sharpe negative -> tighten entry filter slightly
    if (pf != float("inf") and pf < 1.0) or sh < 0:
        rules.append({
            "action": "tighten_entry",
            "delta": 0.05,
            "reason": "Low PF or negative Sharpe; reduce low-quality entries by ~5%."
        })
    # If MDD is high/unknown -> reduce risk per trade
    if (isinstance(mdd, (int, float)) and mdd > 0.25) or (isinstance(mdd, float) and mdd != mdd):
        rules.append({
            "action": "reduce_position_pct",
            "multiplier": 0.9,
            "floor": 0.0025,
            "reason": "High/NaN drawdown; trim position size by 10% (min 0.25%)."
        })
    # If winrate < 50% -> widen take-profit to improve PF skew
    if win < 0.5:
        rules.append({
            "action": "increase_tp_rr",
            "delta": 0.1,
            "cap": 3.0,
            "reason": "Winrate < 50%; slightly increase RR to improve expectancy."
        })
    return rules

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True, help="Path to aggregate_summary.json")
    ap.add_argument("--out", required=True, help="Path to write fixes JSON")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Do not apply changes; only write suggestions")
    mode.add_argument("--apply", action="store_true", help="Apply safe changes if supported (no-ops if not implemented)")
    args = ap.parse_args()

    summary_path = Path(args.summary)
    if not summary_path.exists():
        raise SystemExit(f"Summary not found: {summary_path}")

    summary = json.loads(summary_path.read_text() or "{}")
    fixes_doc = {
        "generated_at": now_rfc3339_utc(),
        "source_summary": str(summary_path),
        "rules": suggest_rules(summary)
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(fixes_doc, indent=2))
    print(f"Saved fixes summary -> {out_path}")

    if args.dry_run or not args.apply:
        print("DRY RUN only (no files changed). Use --apply to actually apply recommended safe fixes.")
        return

    # Placeholders for real application logic (intentionally no-ops here)
    # e.g., load config/decision.yaml, mutate fields per rules, write backup, etc.
    print("Apply mode selected, but no direct file mutations implemented in this tool (safety).")

if __name__ == "__main__":
    main()