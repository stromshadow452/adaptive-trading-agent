from pathlib import Path
fp = Path('tools/executor.py')
txt = fp.read_text(encoding='utf-8')

needle = 'if side not in {\"buy\",\"sell\"} or size_plan<=0:'

if needle not in txt:
    print('NEEDLE_NOT_FOUND: validation line not found. Aborting.')
    raise SystemExit(2)

# NOTE: this block does NOT include the 'if ...' validation line.
force_block = '''# --- FORCE: derive side/size directly from original plan (defensive) ---
try:
    # prefer explicit values already present, else pull straight from plan
    _raw_side = None
    if isinstance(plan, dict):
        _raw_side = plan.get("side") or plan.get("action") or plan.get("direction") or None
    # ensure side is normalized lower-case string (keep whatever 'side' already is if present)
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
    # conservative fallback: leave values as-is
    pass

# debug
if args.verbose:
    try:
        print(f"[FORCE_VALIDATION] side={side!r}, size_plan={size_plan!r}, plan_side={_raw_side!r}")
    except Exception:
        pass

'''
# Insert the force_block immediately BEFORE the existing 'if' line (so the original body remains intact)
new_txt = txt.replace(needle, force_block + "\\n" + needle, 1)
fp.write_text(new_txt, encoding='utf-8')
print('Patched executor.py: inserted FORCE_VALIDATION block BEFORE validation guard (no duplicate if).')
