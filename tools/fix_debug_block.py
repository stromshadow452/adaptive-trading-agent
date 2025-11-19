# write the fixer file

#!/usr/bin/env python3
# tools/fix_debug_block.py
# Safe repair: replace broken DEBUG_PLAN_BEFORE_VALIDATION insertion with correct block.

from pathlib import Path
import shutil
import sys

FP = Path("tools/executor.py")
if not FP.exists():
    print("ERROR: tools/executor.py not found", file=sys.stderr)
    sys.exit(2)

BAK = FP.with_suffix(".fixbak")
shutil.copy2(FP, BAK)
print(f"Backup written to: {BAK}")

orig_lines = FP.read_text(encoding="utf-8").splitlines()

# Find a line that contains our debug marker (either escaped or not)
marker = "DEBUG_PLAN_BEFORE_VALIDATION"
idx = None
for i,l in enumerate(orig_lines):
    if marker in l:
        idx = i
        break

if idx is None:
    print("Marker not found. No changes made.")
    sys.exit(0)

# We'll replace a small window around that marker (previous line and next few lines)
start = max(0, idx-1)
end = min(len(orig_lines), idx+6)

# The clean replacement block we want (exact)
replacement_block = [
"        size_plan = _lot_clamp(size_plan, min_lot=args.broker_min_lot, lot_step=args.broker_lot_step, min_size_cfg=args.decision_min_size)",
"        if args.verbose: print(f\"[DEBUG_PLAN_BEFORE_VALIDATION] plan={plan!r}, side_var={side!r}, size_plan_var={size_plan!r}\")",
"        # soft fallback: coerce side/size from original plan if validation sees them missing",
"        if (not side or str(side).strip()==\"\") and isinstance(plan, dict):",
"            side = str(plan.get(\"side\") or plan.get(\"action\") or \"\").lower()",
"        if (not isinstance(size_plan, (int, float))) or size_plan <= 0:",
"            try:",
"                size_plan = float(plan.get(\"size\") or plan.get(\"qty\") or 0)",
"            except Exception:",
"                size_plan = 0.0",
"",
]

# Replace window
new_lines = orig_lines[:start] + replacement_block + orig_lines[end:]

FP.write_text("\n".join(new_lines), encoding="utf-8")
print(f"Replaced broken debug block around line {idx+1}. File updated.")
print("Please run your executor dry-run and paste the last 40 lines of output (or the DEBUG lines).")


# run fixer with venv python

