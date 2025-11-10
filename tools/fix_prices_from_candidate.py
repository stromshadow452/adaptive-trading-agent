#!/usr/bin/env python3
"""
fix_prices_from_candidate.py

Scan an "approved paths" JSON (array of plan file paths), open each plan,
and if the plan has missing or zero price but candidate.price exists and > 0,
set plan['price'] and plan['decision']['price'] to candidate.price and save.

Usage:
    python tools/fix_prices_from_candidate.py \
        --approved reports/daily/approved_paths_actionable_filtered.json

Options:
    --approved PATH       Path to JSON array of plan file paths (required).
    --backup-dir PATH     Directory to copy original plan files before editing (default: reports/daily/backups).
    --dry-run             Don't write files, only print what would change.
    --verbose             More verbose logging.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Optional

def try_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, (float, int)):
        return float(v)
    if isinstance(v, str):
        s = v.strip().strip('"').strip("'")
        try:
            return float(s)
        except ValueError:
            return None
    return None

def load_json_file(p: Path):
    # Support files that may include BOM
    text = p.read_text(encoding="utf-8-sig")
    return json.loads(text)

def write_json_file(p: Path, obj):
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

def ensure_backup(orig: Path, backup_dir: Path, verbose: bool=False):
    backup_dir.mkdir(parents=True, exist_ok=True)
    dest = backup_dir / orig.name
    if not dest.exists():
        shutil.copy2(orig, dest)
        if verbose:
            print(f"[BACKUP] {orig} -> {dest}")

def fix_plan_file(plan_path: Path, do_write: bool, backup_dir: Path, verbose: bool=False):
    try:
        j = load_json_file(plan_path)
    except Exception as e:
        print(f"[ERROR] Failed to read JSON {plan_path}: {e}")
        return False, "read_error"

    # Top-level price
    top_price = try_float(j.get("price"))
    # decision.price if present
    decision = j.get("decision") if isinstance(j.get("decision"), dict) else {}
    decision_price = try_float(decision.get("price") if isinstance(decision.get("price"), (str,int,float)) else None)

    # candidate.price
    candidate = j.get("candidate") if isinstance(j.get("candidate"), dict) else {}
    candidate_price = try_float(candidate.get("price") if "price" in candidate else candidate.get("close") if "close" in candidate else None)

    # Determine if plan needs fix:
    if (top_price is not None and top_price > 0) or (decision_price is not None and decision_price > 0):
        if verbose:
            print(f"[SKIP] {plan_path} -> already has price (top={top_price}, decision={decision_price})")
        return False, "already_has_price"

    if candidate_price is None or candidate_price <= 0:
        if verbose:
            print(f"[SKIP] {plan_path} -> candidate.price missing or zero (candidate={candidate_price})")
        return False, "candidate_missing_or_zero"

    # At this point candidate_price > 0 and plan price is missing/zero -> patch
    if do_write:
        try:
            # Backup original file first
            ensure_backup(plan_path, backup_dir, verbose=verbose)

            # Set top-level price
            j["price"] = candidate_price

            # Ensure decision dict exists
            if not isinstance(j.get("decision"), dict):
                j["decision"] = {}

            j["decision"]["price"] = candidate_price

            # If size is missing/zero or decision.enter is missing etc we don't touch other fields.
            write_json_file(plan_path, j)
            if verbose:
                print(f"[FIXED] {plan_path} -> set price={candidate_price}")
            return True, "fixed"
        except Exception as e:
            print(f"[ERROR] Failed to write {plan_path}: {e}")
            return False, "write_error"
    else:
        print(f"[DRY-RUN] Would set price={candidate_price} in {plan_path}")
        return True, "would_fix"

def main():
    ap = argparse.ArgumentParser(description="Fix plan files' price from candidate.price when missing or zero.")
    ap.add_argument("--approved", required=True, help="Path to JSON array of plan file paths (approved_paths_*.json)")
    ap.add_argument("--backup-dir", default="reports/daily/backups", help="Directory to copy original plan files before editing")
    ap.add_argument("--dry-run", action="store_true", help="Don't modify files; only report")
    ap.add_argument("--verbose", action="store_true", help="Verbose logs")
    args = ap.parse_args()

    approved_path = Path(args.approved)
    if not approved_path.exists():
        print(f"[FATAL] approved file not found: {approved_path}", file=sys.stderr)
        sys.exit(2)

    try:
        approved_list = load_json_file(approved_path)
    except Exception as e:
        print(f"[FATAL] Failed to load approved JSON list {approved_path}: {e}", file=sys.stderr)
        sys.exit(3)

    if not isinstance(approved_list, list):
        print(f"[FATAL] Expected an array of file paths in {approved_path}", file=sys.stderr)
        sys.exit(4)

    backup_dir = Path(args.backup_dir)
    total = len(approved_list)
    fixed_count = 0
    would_fix = 0
    skipped = 0
    errors = 0
    details = {}

    for p in approved_list:
        path = Path(p)
        if not path.exists():
            print(f"[MISSING] {path}")
            skipped += 1
            details[str(path)] = "missing"
            continue

        ok, reason = fix_plan_file(path, do_write=not args.dry_run, backup_dir=backup_dir, verbose=args.verbose)
        if ok and not args.dry_run:
            fixed_count += 1
        elif ok and args.dry_run:
            would_fix += 1
        else:
            if reason in ("read_error", "write_error"):
                errors += 1
            else:
                skipped += 1
        details[str(path)] = reason

    print("\nSUMMARY:")
    print(f"  total paths processed: {total}")
    if args.dry_run:
        print(f"  would_fix: {would_fix}")
    else:
        print(f"  fixed_count: {fixed_count}")
    print(f"  skipped (no candidate price / already had price / missing): {skipped}")
    print(f"  errors: {errors}")

    # Optionally write a small report next to approved file
    report_path = approved_path.with_name(approved_path.stem + "_fix_report.json")
    try:
        report = {
            "approved_list": str(approved_path),
            "total": total,
            "fixed": fixed_count,
            "would_fix": would_fix,
            "skipped": skipped,
            "errors": errors,
            "details": details,
        }
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nReport written: {report_path}")
    except Exception as e:
        print(f"[WARN] Failed to write report file: {e}")

if __name__ == "__main__":
    main()
