from pathlib import Path
import re, sys, shutil

fp = Path("tools/executor.py")
bak = Path("tools/executor.py.repair_bak")
if not fp.exists():
    print("ERROR: tools/executor.py not found"); sys.exit(2)
shutil.copy2(fp, bak)
print(f"Backup written -> {bak}")

txt = fp.read_text(encoding="utf-8")

# Normalize indentation for leading tabs to spaces on every line (preserve inner spacing)
lines = txt.splitlines(True)
for i,l in enumerate(lines):
    # replace leading tabs with 8 spaces (match file's block style)
    m = re.match(r'^(\t+)', l)
    if m:
        tabs = len(m.group(1))
        lines[i] = (' ' * 8 * tabs) + l[len(m.group(1)):]
txt = "".join(lines)

# Locate markers
start_marker = "# --- FORCE: derive side/size directly from original plan (defensive) ---"
guard = "if side not in {\"buy\",\"sell\"} or size_plan<=0:"

si = txt.find(start_marker)
gi = txt.find(guard, si if si!=-1 else 0)

if si == -1 or gi == -1:
    print("Could not locate start_marker or validation guard. Aborting. (file may differ)")
    sys.exit(3)

# Determine indentation level to use: use indentation of the guard line
# find start of guard line
guard_line_start = txt.rfind('\n', 0, gi) + 1
guard_indent = re.match(r'^\s*', txt[guard_line_start:gi]).group(0)

# Build a clean FORCE block indented to same level as guard (so block sits just before guard)
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

# Now replace from start_marker up to guard with the force_block + guard (guard kept from original)
pre = txt[:si]
post = txt[gi:]  # includes guard and remaining
newtxt = pre + force_block + post

# As a safety: check there are no duplicate start markers left (we wanted single insertion)
if newtxt.count(start_marker) > 1:
    print("Warning: multiple start markers remain after insertion. Cleaning duplicates.")
    # remove subsequent markers
    first = newtxt.find(start_marker)
    rest = newtxt[first+1:]
    rest_clean = rest.replace(start_marker, "")
    newtxt = newtxt[:first+1] + rest_clean

fp.write_text(newtxt, encoding="utf-8")
print("Repair applied: normalized indentation & replaced FORCE block cleanly.")
