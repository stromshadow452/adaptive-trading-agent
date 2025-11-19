# insert_coerce_side_size.py -- patch tools/executor.py to coerce side/size from plan
from pathlib import Path
fp = Path("tools/executor.py")
txt = fp.read_text(encoding="utf-8")
needle = "if side not in {\"buy\",\"sell\"} or size_plan<=0:"
if needle in txt:
    insert_block = """
        # --- COERCE: if executor lost side/size, try to recover from original plan ---
        try:
            if (not side or str(side).strip()==\"\") and isinstance(plan, dict):
                side = str(plan.get(\"side\") or plan.get(\"action\") or \"\").lower()
            if (not isinstance(size_plan, (int, float))) or size_plan <= 0:
                try:
                    size_plan = float(plan.get(\"size\") or plan.get(\"qty\") or 0)
                except Exception:
                    size_plan = 0.0
        except Exception:
            # conservative fallback, leave values as-is
            pass
        if args.verbose:
            try:
                print(f"[DEBUG_COERCE] post-coerce side={side!r}, size_plan={size_plan!r}")
            except Exception:
                pass
        # --- end COERCE ---
"""
    new = txt.replace(needle, insert_block + "\n        " + needle, 1)
    fp.write_text(new, encoding="utf-8")
    print("Inserted COERCE block before validation.")
else:
    print("Validation needle not found; no change. Please search for validation line and patch manually.")
