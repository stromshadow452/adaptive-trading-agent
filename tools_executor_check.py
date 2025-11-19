import py_compile, traceback, glob, os, sys
print("1) Trying to compile tools/executor.py ...")
try:
    py_compile.compile("tools/executor.py", doraise=True)
    print(">>> SYNTAX OK: tools/executor.py compiles cleanly.")
except Exception:
    print(">>> COMPILE ERROR (traceback below):")
    traceback.print_exc()

print("\n2) Listing likely backup candidates (most recent first)...")
patterns = [
    "tools\\executor.py.bak*",
    "tools\\executor.py.fix*.bak*",
    "tools\\executor.py.*bak*",
    "tools\\executor.py.*",
    "tools\\executor.py.autofix.bak",
    "tools\\executor.py.fixdup.tf.bak",
    "tools\\executor.py.fix_topplan.bak",
    "tools\\executor.py.fix*.bak",
]
cands = []
for pat in patterns:
    cands += glob.glob(pat)
cands = sorted(set(cands), key=lambda p: os.path.getmtime(p) if os.path.exists(p) else 0, reverse=True)
if not cands:
    print("No backups found with common names in repo/tools.")
else:
    for i,p in enumerate(cands[:20],1):
        print(f"{i:2d}. {p}   (mtime: {os.path.getmtime(p)})")
    latest = cands[0]
    print("\\n3) Suggested restore command (PowerShell):")
    print(f"   Copy-Item -Path '{latest}' -Destination 'tools\\executor.py' -Force")
    print("   # after restore: .\\.venv\\Scripts\\python.exe -c \"import py_compile; py_compile.compile('tools/executor.py', doraise=True)\"")
