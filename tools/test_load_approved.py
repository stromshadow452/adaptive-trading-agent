import pathlib, importlib.util, json, sys
P = pathlib.Path("reports/screener/approved_smoke.json")
print("exists:", P.exists())
print("content (first 400 chars):")
print(P.read_text()[:400])
# load executor module from file (so we don't import wrong module)
spec = importlib.util.spec_from_file_location("tools_executor","tools/executor.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
# call the internal loader
try:
    plans = mod._load_approved(str(P))
    print("loaded_plans_type:", type(plans).__name__)
    print("loaded_count:", len(plans))
    print("sample:", plans[:2])
except Exception as e:
    print("ERROR calling _load_approved:", repr(e))
    raise
