import importlib.util, pathlib, sys, re, pprint
spec = importlib.util.spec_from_file_location("tools_executor","tools/executor.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
P = pathlib.Path("reports/screener/approved_smoke.json")
plans = mod._load_approved(str(P))
print("plans loaded:", len(plans))
for i, plan in enumerate(plans):
    print("==== plan", i, "raw ====")
    pprint.pprint(plan)
    # show normalized values if code has helpers
    side = plan.get("side")
    size = plan.get("size")
    try:
        # try any helper the module exposes for order normalization
        if hasattr(mod, "_normalize_order_params"):
            print("_normalize_order_params ->", mod._normalize_order_params(plan))
    except Exception as e:
        print("normalize helper error:", e)
    # attempt to call the internal place that formats execution (best-effort)
    # search for a small helper named like _build_order or similar
    cand_names = [n for n in dir(mod) if re.search(r'(build|make|format).*order|order.*make|execution', n, re.I)]
    print("candidate helper names:", cand_names[:10])
print("done")