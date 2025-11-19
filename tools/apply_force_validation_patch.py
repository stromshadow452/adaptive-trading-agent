from pathlib import Path
fp = Path('tools/executor.py')
txt = fp.read_text(encoding='utf-8')

needle = 'if side not in {\"buy\",\"sell\"} or size_plan<=0:'

if needle not in txt:
    print('NEEDLE_NOT_FOUND: search for the exact validation line failed. Aborting.')
    raise SystemExit(2)

replacement = '''# --- FORCE: derive side/size directly from original plan (defensive) ---
try:
    # prefer explicit values already present, else pull straight from plan
    _raw_side = None
    if isinstance(plan, dict):
        _raw_side = plan.get("side") or plan.get("action") or plan.get("direction") or None
    # ensure side is normalized lower-case string
    side = (side or "") if (isinstance(side, str) and side) else ""
    if (not side or str(side).strip()=="") and _raw_side:
        try:
            side = str(_raw_side).strip().lower()
        except Exception:
            side = ""
    # size fallback (float)
    if (not isinstance(size_plan, (int,float))) or size_plan <= 0:
        try:
            if isinstance(plan, dict):
                size_plan = float(plan.get("size") or plan.get("qty") or plan.get("amount") or 0)
        except Exception:
            size_plan = 0.0
except Exception:
    # be conservative — leave existing values if anything fails
    pass

# debug
if args.verbose:
    try:
        print(f"[FORCE_VALIDATION] side={side!r}, size_plan={size_plan!r}, plan_side={_raw_side!r}")
    except Exception:
        pass

# perform the original validation guard
if side not in {"buy","sell"} or size_plan<=0:
'''

# perform a single substitution
new_txt = txt.replace(needle, replacement + "\n" + needle, 1)
fp.write_text(new_txt, encoding='utf-8')
print('Patched executor.py: inserted FORCE_VALIDATION block before validation guard.')
