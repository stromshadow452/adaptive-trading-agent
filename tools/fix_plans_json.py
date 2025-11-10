#!/usr/bin/env python3
"""
fix_plans_json.py
- Scans reports/daily/*plan*.json
- Tries to repair common non-JSON (Python dicts, None/True/False, single quotes).
- Writes backup originals to reports/daily/_backup_plans/
- Overwrites file with valid JSON (pretty-printed).
"""

import os, json, ast, pathlib, re, sys

ROOT = pathlib.Path("reports/daily")
BACKUP = ROOT / "_backup_plans"
BACKUP.mkdir(parents=True, exist_ok=True)

def try_json_load(text, path):
    try:
        return json.loads(text)
    except Exception as e:
        return None

def try_ast_eval(text):
    try:
        return ast.literal_eval(text)
    except Exception:
        return None

def repair_simple_tokens(text):
    # Note: apply conservative replacements only
    # Replace Python None/True/False with JSON null/true/false (word boundaries)
    t = re.sub(r'\bNone\b', 'null', text)
    t = re.sub(r'\bTrue\b', 'true', t)
    t = re.sub(r'\bFalse\b', 'false', t)
    # If single quotes used as string quote, replace them with double quotes only where safe:
    # - This is heuristic: replace keys like 'key': -> "key":
    t = re.sub(r"(?P<q>')(?P<k>[A-Za-z0-9_\-]+)(?P=q)\s*:", r'"\g<k>":', t)
    # - replace single-quoted values 'value' -> "value" (avoid apostrophes inside)
    t = re.sub(r":\s*'([^']*)'", r': "\1"', t)
    # Remove trailing commas before closing } or ]
    t = re.sub(r",\s*([\]\}])", r"\1", t)
    return t

def process_file(p: pathlib.Path):
    s = p.read_text(encoding="utf-8", errors="replace")
    # 1) try JSON
    obj = try_json_load(s, p)
    if obj is not None:
        return ("ok-json", p)

    # 2) try ast.literal_eval (python dict)
    obj = try_ast_eval(s)
    if obj is not None:
        # convert to JSON
        backup = BACKUP / (p.name + ".orig")
        p.replace(backup) if p.exists() else None  # move file to backup (atomic)
        # But above p.replace will error if backup exists: do safer copy
    # We'll implement safe backup below and continue

    # Next: attempt token-repair then json.loads
    repaired = repair_simple_tokens(s)
    obj2 = try_json_load(repaired, p)
    if obj2 is not None:
        # backup original and write repaired JSON
        backup_path = BACKUP / (p.name + ".orig")
        if not backup_path.exists():
            p.rename(backup_path)
        p.write_text(json.dumps(obj2, indent=2, ensure_ascii=False), encoding="utf-8")
        return ("repaired-text", p)
    # last attempt: ast on repaired text
    obj3 = try_ast_eval(repaired)
    if obj3 is not None:
        backup_path = BACKUP / (p.name + ".orig")
        if not backup_path.exists():
            p.rename(backup_path)
        p.write_text(json.dumps(obj3, indent=2, ensure_ascii=False), encoding="utf-8")
        return ("repaired-ast", p)

    return ("failed", p)

def main():
    plan_files = sorted(ROOT.glob("*plan*.json"))
    if not plan_files:
        print("No plan files found under reports/daily.")
        return 1
    stats = {"ok-json":0, "repaired-text":0, "repaired-ast":0, "repaired-ast-original":0, "failed":0}
    for p in plan_files:
        status, path = process_file(p)
        print(status.upper(), p.name)
        stats[status] = stats.get(status, 0) + 1
    print("SUMMARY:", stats)
    return 0

if __name__ == "__main__":
    sys.exit(main())
