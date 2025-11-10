import json, sys, os

def load_json(path):
    try:
        return json.load(open(path))
    except Exception as e:
        print(f"ERR loading {path}: {e}")
        return None

def inspect_plan_file(path):
    if not os.path.exists(path):
        print(f" MISSING: {path}")
        return
    data = load_json(path)
    if not data:
        print(f" EMPTY/INVALID JSON: {path}")
        return
    # If file is a plan object or list, handle both
    entries = data if isinstance(data, list) else [data]
    for i,e in enumerate(entries):
        sym = e.get("symbol") or e.get("candidate",{}).get("pair") or e.get("candidate",{}).get("symbol")
        tf  = e.get("tf") or e.get("candidate",{}).get("tf")
        dec = e.get("decision") or e
        enter = dec.get("enter") if isinstance(dec, dict) else None
        size  = dec.get("size") if isinstance(dec, dict) else e.get("size")
        price = e.get("price") or (e.get("candidate",{}).get("price"))
        side  = dec.get("side") if isinstance(dec, dict) else e.get("side")
        sl_type = dec.get("sl_type") if isinstance(dec, dict) else e.get("sl_type")
        print(f" PLAN: {os.path.basename(path)}  idx={i}  symbol={sym} tf={tf} enter={enter} size={size} price={price} side={side} sl_type={sl_type}")

def try_inspect(approved_path):
    obj = load_json(approved_path)
    if obj is None: return
    # if approved is list of paths
    if isinstance(obj, list) and all(isinstance(x, str) for x in obj):
        print(f"Approved is PATH list ({len(obj)} items)")
        for p in obj:
            inspect_plan_file(p)
        return
    # if approved is list of objects
    if isinstance(obj, list) and len(obj)>0 and isinstance(obj[0], dict):
        print(f"Approved is OBJECT list ({len(obj)} items) — inspecting embedded plan-like objects")
        for i,e in enumerate(obj[:50]):
            # write temp file for each object then inspect
            tmp = f".tmp_plan_{i}.json"
            json.dump(e, open(tmp,"w"), indent=2)
            inspect_plan_file(tmp)
            os.remove(tmp)
        return
    # fallback: single object
    print("Approved file content unknown type — printing raw:")
    print(obj)

if __name__ == '__main__':
    # try common approved file names if none arg passed
    candidates = sys.argv[1:] or ["reports/daily/approved_paths.json", "reports/daily/approved_filtered.json", "reports/daily/approved.json"]
    for c in candidates:
        print("\n>>> Inspecting", c)
        if os.path.exists(c):
            try_inspect(c)
        else:
            print(" NOT FOUND:", c)
