#!/usr/bin/env python3
"""
tools/fix_plans_and_make_actionable.py

Scan reports/daily/*plan_*.json, try to fix common JSON issues, backup originals,
and produce reports/daily/approved_paths_actionable.json containing absolute
paths to actionable plans (side buy|sell and size > 0).

Run from repo root:
  python tools/fix_plans_and_make_actionable.py
"""

import json
import re
import sys
from pathlib import Path
from shutil import copy2

ROOT = Path.cwd()
PLANS_DIR = ROOT / "reports" / "daily"
BACKUP_DIR = PLANS_DIR / "_backup_plans"
OUT_ACTIONABLE = PLANS_DIR / "approved_paths_actionable.json"

# conservative fixes to try (applied in order)
def fix_normalize_python_tokens(s: str) -> str:
    # replace Python booleans/None with JSON equivalents (word-boundaries)
    s = re.sub(r'\bTrue\b', 'true', s)
    s = re.sub(r'\bFalse\b', 'false', s)
    s = re.sub(r'\bNone\b', 'null', s)
    return s

def fix_remove_trailing_commas(s: str) -> str:
    # remove trailing commas before closing } or ]
    s = re.sub(r',\s*(\}|\])', r'\1', s)
    return s

def fix_single_to_double_quotes(s: str) -> str:
    # last-resort: convert simple single-quoted JSON-ish tokens to double quotes.
    # Only replace single quotes that appear to wrap keys or short values (no newlines).
    # This is conservative to avoid changing apostrophes inside long text.
    def repl(m):
        inner = m.group(1)
        # escape any double quotes already inside
        inner_esc = inner.replace('"', '\\"')
        return f'"{inner_esc}"'
    # Replace keys: 'key':  -> "key":
    s = re.sub(r"'([A-Za-z0-9_\- ]{1,120})'\s*:", repl, s)
    # Replace values: : 'value'[,}\]]
    s = re.sub(r":\s*'([^'\n]{0,200})'(\s*[,\}\]])", lambda m: f': "{m.group(1)}"{m.group(2)}', s)
    return s

def try_load(s: str):
    try:
        return json.loads(s)
    except Exception:
        return None

def backup_and_write(p: Path, content: str):
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    backup = BACKUP_DIR / (p.name + ".bak")
    if not backup.exists():
        copy2(p, backup)  # copy original only once
    p.write_text(content, encoding="utf-8")

def main():
    if not PLANS_DIR.exists():
        print("reports/daily not found. Run from repo root.", file=sys.stderr)
        raise SystemExit(2)

    plan_files = sorted(PLANS_DIR.glob("*plan_*.json"))
    actionable = []
    stats = {"total": 0, "valid": 0, "fixed": 0, "invalid": 0, "actionable": 0, "skipped": 0}

    for p in plan_files:
        stats["total"] += 1
        raw = p.read_text(encoding="utf-8")
        obj = try_load(raw)
        fixed = False

        if obj is None:
            # attempt fixes in order; after each try loading
            attempts = [
                fix_normalize_python_tokens,
                lambda s: fix_remove_trailing_commas(fix_normalize_python_tokens(s)),
                lambda s: fix_single_to_double_quotes(fix_remove_trailing_commas(fix_normalize_python_tokens(s))),
            ]
            for fn in attempts:
                cand = fn(raw)
                obj = try_load(cand)
                if obj is not None:
                    # backup original and write fixed version (pretty-printed)
                    pretty = json.dumps(obj, indent=2, ensure_ascii=False)
                    backup_and_write(p, pretty)
                    stats["fixed"] += 1
                    fixed = True
                    print(f"[FIXED] {p.name} (fix applied: {fn.__name__})")
                    break

        else:
            stats["valid"] += 1

        if obj is None:
            stats["invalid"] += 1
            print(f"[INVALID] {p.name} — could not parse or fix (left untouched).")
            continue

        # At this point obj is a Python object parsed from the plan
        # defensive field lookups
        side = None
        size = 0.0

        # prefer decision.* then top-level
        dec = obj.get("decision") if isinstance(obj, dict) else None
        if isinstance(dec, dict):
            side = dec.get("side") or side
            size = dec.get("size") or size

        side = side or obj.get("side")
        size = size or obj.get("size") or 0.0

        # normalize side string if present
        if isinstance(side, str):
            side_str = side.lower()
        else:
            side_str = None

        # numeric cast for size
        try:
            size_num = float(size)
        except Exception:
            size_num = 0.0

        if side_str in ("buy", "sell") and size_num > 0.0:
            abs_path = str(p.resolve())
            actionable.append(abs_path)
            stats["actionable"] += 1
            print(f"[KEEP] {p.name} -> side={side_str} size={size_num}")
        else:
            stats["skipped"] += 1
            print(f"[SKIP] {p.name} -> side={side_str or '<none>'} size={size_num}")

    # write actionable JSON if any
    if actionable:
        OUT_ACTIONABLE.write_text(json.dumps(actionable, indent=2), encoding="utf-8")
        print()
        print(f"Saved {len(actionable)} actionable paths -> {OUT_ACTIONABLE}")
    else:
        print()
        print("No actionable plans found; approved_paths_actionable.json not created.")

    print()
    print("SUMMARY:", stats)

if __name__ == "__main__":
    main()
