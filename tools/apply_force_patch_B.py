from pathlib import Path
fp = Path("tools/executor.py")
txt = fp.read_text(encoding="utf-8")

start_marker = "# --- FORCE: derive side/size directly from original plan (defensive) ---"
if start_marker not in txt:
    print("Marker not found - aborting."); raise SystemExit(2)

force_block = (
"        # --- FORCE (AGGRESSIVE): override side/size from plan if present ---\n"
"        try:\n"
"            # If plan explicitly sets side, force it (aggressive fallback)\n"
"            if isinstance(plan, dict) and plan.get(\"side\"):\n"
"                try:\n"
"                    side = str(plan.get(\"side\")).strip().lower()\n"
"                except Exception:\n"
"                    side = (side or \"\").lower()\n"
"            # also accept action/direction\n"
"            if (not side or str(side).strip()==\"\") and isinstance(plan, dict):\n"
"                _raw_side = plan.get(\"action\") or plan.get(\"direction\") or None\n"
"                if _raw_side:\n"
"                    try: side = str(_raw_side).strip().lower()\n"
"                    except Exception: pass\n"
"            # Force size if plan specifies it\n"
"            try:\n"
"                if isinstance(plan, dict) and plan.get(\"size\") is not None:\n"
"                    size_plan = float(plan.get(\"size\") or 0.0)\n" 
"            except Exception:\n"
"                pass\n"
"        except Exception:\n"
"            pass\n"
"\n"
"        if args.verbose:\n"
"            try:\n"
"                print(f\"[FORCE_VALIDATION-AGG] side={side!r}, size_plan={size_plan!r}, plan_size={plan.get('size',None)!r}\")\n"
"            except Exception:\n"
"                pass\n"
"\n"
)
# splice like Option A
idx = txt.find(start_marker)
guard = "if side not in {\"buy\",\"sell\"} or size_plan<=0:"
gidx = txt.find(guard, idx)
if idx == -1 or gidx == -1:
    print("Could not locate FORCE block or validation guard; aborting")
    raise SystemExit(2)
new_txt = txt[:idx] + force_block + txt[gidx:]
fp.write_text(new_txt, encoding="utf-8")
print("Applied Option B: aggressive FORCE block (indented).")
