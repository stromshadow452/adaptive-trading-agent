from pathlib import Path
fp = Path("tools/executor.py")
txt = fp.read_text(encoding="utf-8")

# Find the problematic unindented FORCE block start (unique marker)
start_marker = "# --- FORCE: derive side/size directly from original plan (defensive) ---"
if start_marker not in txt:
    print("Marker not found - aborting. Did file change?"); raise SystemExit(2)

# Build corrected, properly indented block (8 spaces indentation to match surrounding code)
force_block = (
"        # --- FORCE: derive side/size directly from original plan (defensive) ---\n"
"        try:\n"
"            # prefer explicit values already present, else pull straight from plan\n"
"            _raw_side = None\n"
"            if isinstance(plan, dict):\n"
"                _raw_side = plan.get(\"side\") or plan.get(\"action\") or plan.get(\"direction\") or None\n"
"            # ensure side is normalized lower-case string (keep whatever 'side' already is if present)\n"
"            side = (side or \"\") if (isinstance(side, str) and side) else \"\"\n"
"            if (not side or str(side).strip()==\"\") and _raw_side:\n"
"                try:\n"
"                    side = str(_raw_side).strip().lower()\n"
"                except Exception:\n"
"                    side = \"\"\n"
"            # size fallback (float)\n"
"            if (not isinstance(size_plan, (int,float))) or size_plan <= 0:\n"
"                try:\n"
"                    if isinstance(plan, dict):\n"
"                        size_plan = float(plan.get(\"size\") or plan.get(\"qty\") or plan.get(\"amount\") or 0)\n"
"                except Exception:\n"
"                    size_plan = 0.0\n"
"        except Exception:\n"
"            # conservative fallback: leave values as-is\n"
"            pass\n"
"\n"
"        # debug\n"
"        if args.verbose:\n"
"            try:\n"
"                print(f\"[FORCE_VALIDATION] side={side!r}, size_plan={size_plan!r}, plan_side={_raw_side!r}\")\n"
"            except Exception:\n"
"                pass\n"
"\n"
)
# Replace only the first occurrence of the old FORCE block (unindented). We locate the start_marker and replace up to the existing 'if side not in' guard.
idx = txt.find(start_marker)
# find the subsequent 'if side not in' that corresponds to the validation guard
guard = "if side not in {\"buy\",\"sell\"} or size_plan<=0:"
gidx = txt.find(guard, idx)
if idx == -1 or gidx == -1:
    print("Could not locate FORCE block or validation guard; aborting")
    raise SystemExit(2)

# splice: everything before start_marker stays, then insert indented force_block, then insert the guard and the rest
new_txt = txt[:idx] + force_block + txt[gidx:]
fp.write_text(new_txt, encoding="utf-8")
print("Applied Option A: conservative FORCE block (indented).")
