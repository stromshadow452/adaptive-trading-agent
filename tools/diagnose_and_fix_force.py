# tools/diagnose_and_fix_force.py
from pathlib import Path
import shutil, sys, re, traceback, textwrap

FP = Path("tools/executor.py")
BAK = Path("tools/executor.py.bak")

if not FP.exists():
    print("ERROR: tools/executor.py not found in repo root. Run from repo root.")
    sys.exit(2)

# backup
shutil.copy2(FP, BAK)
print(f"Backup created -> {BAK}")

txt = FP.read_text(encoding="utf-8")
# quick compile check to get exact syntax error if present
def try_compile(src):
    try:
        compile(src, str(FP), 'exec')
        return None
    except SyntaxError as se:
        return se
    except Exception as e:
        return e

se = try_compile(txt)
if se is None:
    print("No SyntaxError: file compiles cleanly. Exiting.")
    sys.exit(0)

print("Compilation error detected.")
if isinstance(se, SyntaxError):
    print(f"SyntaxError: msg='{se.msg}'  lineno={se.lineno}  offset={se.offset}")
else:
    print("Non-SyntaxError during compile: ", repr(se))

# print context lines with visible whitespace
ln = se.lineno if isinstance(se, SyntaxError) and se.lineno else None
def show_context(src, lineno, context=8):
    lines = src.splitlines()
    L = len(lines)
    if lineno is None:
        rng = range(max(0, L-20), L)
    else:
        start = max(1, lineno - context)
        end = min(L, lineno + context)
        rng = range(start, end+1)
    print("\n--- CONTEXT (visible whitespace) ---")
    for i in rng:
        s = lines[i-1]
        vs = s.replace("\t", "→").replace(" ", "·")
        marker = ">>" if lineno and i == lineno else "  "
        print(f"{marker} {i:4d}: {vs!r}")
    print("--- END CONTEXT ---\n")

show_context(txt, ln, context=8)

# robustly find start_marker ignoring leading whitespace
start_re = re.compile(r'^\s*#\s*---\s*FORCE:.*derive side/size.*$', re.MULTILINE | re.IGNORECASE)
guard_re = re.compile(r'^\s*if\s+side\s+not\s+in\s+\{\s*\"buy\"\s*,\s*\"sell\"\s*\}\s+or\s+size_plan\s*<=\s*0\s*:', re.MULTILINE)

start_m = start_re.search(txt)
guard_m = guard_re.search(txt, start_m.end() if start_m else 0)

print("Marker search results:")
print("  start_marker found:", bool(start_m), f"at {start_m.start() if start_m else 'N/A'}")
print("  guard found after start:", bool(guard_m), f"at {guard_m.start() if guard_m else 'N/A'}")

if not start_m or not guard_m:
    print("\nCould not find both start marker and guard (maybe they were edited).")
    print("I'll print the first 3000 chars and the last 3000 chars to help you paste into chat if you need further help.\n")
    print("--- HEAD (first 3000 chars) ---")
    print(txt[:3000])
    print("--- TAIL (last 3000 chars) ---")
    print(txt[-3000:])
    print("\nPaste the context above here if you want me to craft a manual patch.")
    sys.exit(3)

# Determine guard indentation (whitespace at beginning of the guard line)
guard_line_start = txt.rfind("\n", 0, guard_m.start()) + 1
guard_indent = re.match(r'^\s*', txt[guard_line_start:guard_m.start()]).group(0)
print(f"Detected guard indentation as {len(guard_indent)} bytes (repr: {guard_indent!r})")

# Build the fixed force block with the same indent as guard
indent = guard_indent
fb = []
fb.append(indent + "# --- FORCE: derive side/size directly from original plan (defensive) ---\n")
fb.append(indent + "try:\n")
fb.append(indent + "    # prefer explicit values already present, else pull straight from plan\n")
fb.append(indent + "    _raw_side = None\n")
fb.append(indent + "    if isinstance(plan, dict):\n")
fb.append(indent + "        _raw_side = plan.get('side') or plan.get('action') or plan.get('direction') or None\n")
fb.append(indent + "    # keep existing side if present; otherwise use plan's\n")
fb.append(indent + "    side = (side or '') if (isinstance(side, str) and side) else ''\n")
fb.append(indent + "    if (not side or str(side).strip()=='') and _raw_side:\n")
fb.append(indent + "        try:\n")
fb.append(indent + "            side = str(_raw_side).strip().lower()\n")
fb.append(indent + "        except Exception:\n")
fb.append(indent + "            side = ''\n")
fb.append(indent + "    # size fallback (float)\n")
fb.append(indent + "    if (not isinstance(size_plan, (int,float))) or size_plan <= 0:\n")
fb.append(indent + "        try:\n")
fb.append(indent + "            if isinstance(plan, dict):\n")
fb.append(indent + "                size_plan = float(plan.get('size') or plan.get('qty') or plan.get('amount') or 0)\n")
fb.append(indent + "        except Exception:\n")
fb.append(indent + "            size_plan = 0.0\n")
fb.append(indent + "except Exception:\n")
fb.append(indent + "    # conservative fallback: leave values as-is\n")
fb.append(indent + "    pass\n")
fb.append("\n")
fb.append(indent + "# debug\n")
fb.append(indent + "if args.verbose:\n")
fb.append(indent + "    try:\n")
fb.append(indent + "        print(f\"[FORCE_VALIDATION] side={side!r}, size_plan={size_plan!r}, plan_side={_raw_side!r}\")\n")
fb.append(indent + "    except Exception:\n")
fb.append(indent + "        pass\n")
fb.append("\n")

force_block = "".join(fb)

# Replace the span from start_m.start() up to guard_m.start() (exclusive) with our force_block
pre = txt[:start_m.start()]
post = txt[guard_m.start():]
newtxt = pre + force_block + post

# Try compiling new text
se2 = try_compile(newtxt)
if se2 is None:
    FP.write_text(newtxt, encoding="utf-8")
    print("SUCCESS: Applied replacement and new file compiles cleanly. Wrote changes to tools/executor.py")
    sys.exit(0)
else:
    print("Attempted replacement but compile still fails.")
    print("Compile error after replacement:", repr(se2))
    # print context around previous syntax error and the guard region for debugging
    print("\n--- Guard-line context (20 lines around guard) ---")
    guard_lineno = newtxt[:guard_m.start()].count("\n") + 1
    show_context(newtxt, guard_lineno, context=12)
    print("\n--- End ---")
    # do not write newtxt in this failure case (we kept backup)
    sys.exit(4)
