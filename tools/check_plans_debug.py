#!/usr/bin/env python3
"""
check_plans_debug.py

Validate a list of plan JSON files (approved paths), print OK/skip reasons,
and perform max_position gating based on executions CSV.

Usage:
  python tools/check_plans_debug.py --approved approved_paths.json --executions executions.csv [--max_position 2]

Outputs to stdout. Returns 0 on success.

This is intentionally defensive and prints clear skip reasons.
"""
from __future__ import annotations
import argparse
import json
import csv
import sys
from pathlib import Path
from typing import Dict, List, Any

def load_paths(path: Path) -> List[str]:
    txt = path.read_text(encoding="utf-8-sig")
    try:
        data = json.loads(txt)
    except Exception as e:
        raise RuntimeError(f"Failed to parse approved paths JSON {path}: {e}")
    if not isinstance(data, list):
        raise RuntimeError(f"Approved JSON must be a list of file paths. Got {type(data)}")
    return data

def load_executions_open_counts(executions_csv: Path) -> Dict[str,int]:
    counts: Dict[str,int] = {}
    if not executions_csv.exists():
        return counts
    with executions_csv.open(newline='', encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            try:
                status = str(r.get("status","")).strip().upper()
                symbol = str(r.get("symbol","")).strip()
            except Exception:
                continue
            if status == "OPEN" and symbol:
                counts[symbol] = counts.get(symbol, 0) + 1
    return counts

def read_plan_json(p: Path) -> Any:
    # open with utf-8-sig to handle BOMs
    txt = p.read_text(encoding="utf-8-sig")
    return json.loads(txt)

def symbol_from_path(p: str) -> str:
    # Expect filenames like EURUSD_M15_plan_YYYY...
    basename = Path(p).name
    # split on underscore, first chunk should be symbol
    parts = basename.split("_")
    return parts[0] if parts else basename

def check_plan(p: Path) -> List[str]:
    """
    Return list of problems (empty -> OK).
    """
    problems: List[str] = []
    try:
        j = read_plan_json(p)
    except Exception as e:
        problems.append(f"invalid_json: {e}")
        return problems

    # Normalize decision vs top-level
    decision = {}
    if isinstance(j.get("decision"), dict):
        decision = j["decision"].copy()
    # overlay top-level fields if not present in decision
    for key in ("enter","side","size","price","sl","tp"):
        if key not in decision and key in j:
            decision[key] = j[key]

    # Helpful raw candidate fields
    candidate = j.get("candidate", {})

    # Basic checks
    enter = decision.get("enter", False)
    if not bool(enter):
        problems.append("enter=False")
    side = decision.get("side") or j.get("side") or ""
    side = str(side).lower() if side is not None else ""
    if side not in ("buy","sell"):
        problems.append(f"side_invalid_or_missing: '{side}'")
    # size might be named 'size' or 'order_size' or be nested; try to coerce
    size_raw = decision.get("size", j.get("size", 0))
    try:
        size = float(size_raw or 0)
    except Exception:
        size = 0.0
    if size <= 0:
        problems.append(f"size_invalid_or_zero: {size_raw}")
    # price check
    price_raw = decision.get("price", j.get("price", 0))
    try:
        price = float(price_raw or 0)
    except Exception:
        price = 0.0
    # If candidate.price exists and positive, use that as a hint
    cand_price = candidate.get("price") if isinstance(candidate, dict) else None
    if (price <= 0) and cand_price:
        try:
            price = float(cand_price)
        except Exception:
            price = price
    if price <= 0:
        problems.append(f"price_missing_or_zero (decision.price/top-level.price/candidate.price)")

    # If final_score or decision.final_score present, ensure it meets minimum threshold? (optional)
    # We don't fail on score but include in notes if present
    # Add some context like generated_at etc.
    generated_at = j.get("generated_at")
    if generated_at:
        # not a failure, but include in problems list as context? skip for now
        pass

    return problems

def main():
    ap = argparse.ArgumentParser(description="Debug-check approved plan paths")
    ap.add_argument("--approved", required=True, type=Path, help="JSON file with list of plan file paths")
    ap.add_argument("--executions", required=False, type=Path, help="executions CSV path (for OPEN count)")
    ap.add_argument("--max_position", required=False, type=int, default=9999, help="max OPEN positions per symbol")
    args = ap.parse_args()

    try:
        paths = load_paths(args.approved)
    except Exception as e:
        print(f"[ERROR] loading approved paths: {e}", file=sys.stderr)
        sys.exit(2)

    open_counts = {}
    if args.executions:
        open_counts = load_executions_open_counts(args.executions)

    total = len(paths)
    ok = 0
    skipped = 0
    missing = 0
    invalid_json = 0

    print(f"[INFO] Loaded {total} plan paths. max_position={args.max_position}")
    for pth in paths:
        sym = symbol_from_path(pth)
        open_for_sym = open_counts.get(sym, 0)
        if open_for_sym >= args.max_position:
            print(f"[SKIP] {sym} -> {pth}")
            print(f"       - reason: open_positions={open_for_sym} >= max_position={args.max_position}")
            skipped += 1
            continue

        p = Path(pth)
        if not p.exists():
            print(f"[MISSING] {sym} -> {pth}")
            missing += 1
            continue

        problems = check_plan(p)
        if not problems:
            print(f"[OK] {sym} -> {pth}")
            ok += 1
        else:
            # Print the top-level problems and helpful candidate info
            print(f"[SKIP] {sym} -> {pth}")
            for pr in problems:
                print(f"       - {pr}")
            skipped += 1

    print("\nSUMMARY COUNTS:")
    print(f"  total: {total}")
    print(f"  ok: {ok}")
    print(f"  skipped: {skipped}")
    print(f"  missing: {missing}")
    print(f"  invalid_json: {invalid_json}")

    # exit code 0 even if some skipped (works as diagnostic)
    sys.exit(0)

if __name__ == "__main__":
    main()
