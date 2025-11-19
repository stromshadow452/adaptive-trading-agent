from pathlib import Path
fp = Path('tools/executor.py')
txt = fp.read_text(encoding='utf-8')

needle = 'if side not in {\"buy\",\"sell\"} or size_plan<=0:'

if needle not in txt:
    print('NEEDLE_NOT_FOUND: validation line not found. Aborting.')
    raise SystemExit(2)

# This block contains REAL newlines (no escaped \\n)
force_block = (
    '# --- FORCE: derive side/size directly from original plan (defensive) ---\n'
    'try:\n'
    '    # prefer explicit values already present, else pull straight from plan\n'
    '    _raw_side = None\n'
    '    if isinstance(plan, dict):\n'
    '        _raw_side = plan.get(\"side\") or plan.get(\"action\") or plan.get(\"direction\") or None\n'
    '    # ensure side is normalized lower-case string (keep whatever \"side\" already is if present)\n'
    '    side = (side or \"\") if (isinstance(side, str) and side) else \"\"\n'
    '    if (not side or str(side).strip()==\"\") and _raw_side:\n'
    '        try:\n'
    '            side = str(_raw_side).strip().lower()\n'
    '        except Exception:\n'
    '            side = \"\"\n'
    '    # size fallback (float)\n'
    '    if (not isinstance(size_plan, (int,float))) or size_plan <= 0:\n'
    '        try:\n'
    '            if isinstance(plan, dict):\n'
    '                size_plan = float(plan.get(\"size\") or plan.get(\"qty\") or plan.get(\"amount\") or 0)\n'
    '        except Exception:\n'
    '            size_plan = 0.0\n'
    'except Exception:\n'
    '    # conservative fallback: leave values as-is\n'
    '    pass\n'
    '\n'
    '# debug\n'
    'if args.verbose:\n'
    '    try:\n'
    '        print(f\"[FORCE_VALIDATION] side={side!r}, size_plan={size_plan!r}, plan_side={_raw_side!r}\")\n'
    '    except Exception:\n'
    '        pass\n'
    '\n'
)

# Insert the force_block immediately BEFORE the existing 'if' line (so the original body remains intact)
new_txt = txt.replace(needle, force_block + needle, 1)
fp.write_text(new_txt, encoding='utf-8')
print('Patched executor.py: inserted FORCE_VALIDATION block BEFORE validation guard (with real newlines).')
