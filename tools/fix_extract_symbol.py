#!/usr/bin/env python3
# fix_extract_symbol.py
# Replace broken _extract_symbol_from_plan in tools/executor.py with robust version.

from pathlib import Path
import re, shutil, sys

FP = Path("tools/executor.py")
if not FP.exists():
    print("ERROR: tools/executor.py not found", file=sys.stderr); sys.exit(2)

BAK = FP.with_suffix(".py.bak")
shutil.copy2(FP, BAK)
print("Backup written to:", BAK)

orig = FP.read_text(encoding="utf-8")

replacement = r'''
def _extract_symbol_from_plan(plan: dict, verbose: bool=False) -> str:
    """
    Robust symbol extractor.
    Prefer explicit plan fields, then nested candidate, then globals()['plan'], then filename heuristics.
    Returns normalized symbol (e.g., EURUSD) or "UNKNOWN".
    """
    try:
        # 1) explicit top-level keys
        if isinstance(plan, dict):
            for key in ("symbol", "pair", "asset"):
                raw = plan.get(key)
                if raw and isinstance(raw, str):
                    sym = normalize_symbol(raw).upper()
                    if verbose: print(f"[sym] from field '{key}' -> {sym}")
                    return sym
            # 1b) nested candidate dict
            cand = plan.get("candidate") if isinstance(plan.get("candidate"), dict) else {}
            for key in ("symbol", "pair", "asset"):
                raw = cand.get(key)
                if raw and isinstance(raw, str):
                    sym = normalize_symbol(raw).upper()
                    if verbose: print(f"[sym] from nested candidate '{key}' -> {sym}")
                    return sym

        # 2) fallback to globals().get("plan")
        g = globals().get("plan", {}) if isinstance(globals().get("plan", {}), dict) else {}
        cand_g = g.get("candidate") if isinstance(g.get("candidate"), dict) else {}
        for key in ("symbol", "pair", "asset"):
            raw = g.get(key) or cand_g.get(key)
            if raw and isinstance(raw, str):
                sym = normalize_symbol(raw).upper()
                if verbose: print(f"[sym] from globals field '{key}' -> {sym}")
                return sym

        # 3) hints (base/ticker/instrument) from combined candidate
        plan_cand = plan.get("candidate") if isinstance(plan, dict) and isinstance(plan.get("candidate"), dict) else {}
        combined = {**(plan_cand or {}), **(cand_g or {})}
        hints = []
        for k in ("base", "ticker", "instrument"):
            v = combined.get(k)
            if isinstance(v, str):
                hints.append(v)
        if hints:
            candidate_sym = normalize_symbol("".join(hints)).upper()
            if len(candidate_sym) >= 6:
                if verbose: print(f"[sym] from nested hints -> {candidate_sym}")
                return candidate_sym

        # 4) plan filename heuristics
        guess_path = None
        if isinstance(plan, dict):
            guess_path = plan.get("_plan_path") or plan.get("plan_file")
        if not guess_path:
            guess_path = g.get("_plan_path") or g.get("plan_file")
        if guess_path:
            guess = _symbol_from_plan_path(guess_path)
            if guess:
                sym = normalize_symbol(guess).upper()
                if verbose: print(f"[sym] from filename -> {sym}")
                return sym

    except Exception:
        # swallow — return UNKNOWN below
        pass

    if verbose: print("[sym] default -> UNKNOWN]")
    return "UNKNOWN"
'''

# Replace the existing function block. We look for "def _extract_symbol_from_plan(" until the next
# top-level "def " or "class " or EOF. This is safer than fragile regexes that failed earlier.
m = re.search(r'(?s)(^def\s+_extract_symbol_from_plan\s*\(.*?)(?=^\s*(def |class )|\Z)', orig, flags=re.MULTILINE)
if m:
    start, end = m.start(1), m.end(1)
    new_text = orig[:start] + replacement + orig[end:]
    FP.write_text(new_text, encoding="utf-8")
    print("Patched _extract_symbol_from_plan in", FP)
else:
    # Append replacement if not found
    FP.write_text(orig + "\n\n" + replacement, encoding="utf-8")
    print("Appended _extract_symbol_from_plan to EOF in", FP)

print("Done. Please re-run the inspector or executor (dry_run) to confirm.")
