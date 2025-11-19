# tools/insert_cleanup.py
# Run with: .\.venv\Scripts\python.exe tools\insert_cleanup.py
from pathlib import Path
import re, sys

FP = Path("tools/executor.py")
if not FP.exists():
    print("ERROR: tools/executor.py not found.")
    sys.exit(2)

txt = FP.read_text(encoding="utf-8")

# Idempotency: do nothing if cleanup already present
if 'del globals()[\"plan\"]' in txt or "del globals()['plan']" in txt:
    print("Cleanup already present in tools/executor.py — nothing to do.")
    sys.exit(0)

# Pattern: find the dry-run guard header 'if args.dry_run and DRY_WRITE_ALLOWED'
pat = re.compile(r'(?m)^(\s*)if args\.dry_run and DRY_WRITE_ALLOWED')

m = pat.search(txt)
if not m:
    print("Could not locate 'if args.dry_run and DRY_WRITE_ALLOWED' in file. Aborting.")
    sys.exit(2)

indent = m.group(1) or ""
# Build cleanup block matching surrounding indentation (4 spaces deeper than typical loop-body)
cleanup = (
    indent + "try:\n"
    + indent + "    del globals()['plan']\n"
    + indent + "except Exception:\n"
    + indent + "    pass\n\n"
)

# Insert the cleanup block immediately before the matched guard (first occurrence only)
new_txt = txt[: m.start()] + cleanup + txt[m.start():]

# Write atomically
bak = FP.with_name(FP.name + ".pre_cleanup.bak")
bak.write_text(txt, encoding="utf-8")
FP.write_text(new_txt, encoding="utf-8")

print(f"Inserted cleanup block before dry-run guard and backed up original to: {bak}")
