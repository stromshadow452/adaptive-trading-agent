import importlib.util, pathlib, sys, json, traceback
P = pathlib.Path("reports/screener/approved_smoke.json")
print("approved exists:", P.exists())
print("approved preview:", P.read_text(encoding="utf-8")[:400])

# load executor module from file (so we inspect same code it runs)
spec = importlib.util.spec_from_file_location("tools_executor","tools/executor.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

print("MODULE LOADED FROM:", getattr(spec,'origin',None))
# load plans via module helper
try:
    plans = mod._load_approved(str(P))
except Exception as e:
    print("ERROR: _load_approved failed:", repr(e))
    traceback.print_exc()
    sys.exit(2)

print("loaded_plans_type:", type(plans).__name__, "count:", len(plans))
for i,plan in enumerate(plans):
    print("\\n--- PLAN", i, "repr ---")
    # pretty print keys & values
    try:
        import pprint
        pprint.pprint(plan)
    except Exception:
        print(repr(plan))
    # raw fields
    print(" raw side:", repr(plan.get("side")), " raw size:", repr(plan.get("size")), " raw symbol:", repr(plan.get("symbol")))
    # call executor's extractor
    try:
        sym = mod._extract_symbol_from_plan(plan, verbose=True)
    except Exception as e:
        sym = "ERR:" + repr(e)
    print(" _extract_symbol_from_plan ->", sym)
    # also run normalize_symbol if present
    if hasattr(mod, "normalize_symbol"):
        try:
            ns = mod.normalize_symbol(plan.get("symbol") or "")
            print(" normalize_symbol(plan['symbol']) ->", repr(ns))
        except Exception as e:
            print(" normalize_symbol error:", repr(e))
print("\\nINSPECTOR DONE")
