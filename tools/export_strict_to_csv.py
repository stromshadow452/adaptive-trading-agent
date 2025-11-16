import json, csv, pathlib, sys

agg = pathlib.Path("reports/executions/strict_agg.json")
out = pathlib.Path("reports/executions/executions_run_strict.csv")
out.parent.mkdir(parents=True, exist_ok=True)

if not agg.exists():
    print("ERROR: missing aggregate file:", agg)
    sys.exit(1)

with agg.open("r", encoding="utf-8") as f:
    data = json.load(f)

# accept list or dict {executions|plans|orders:[...]}
rows = []
if isinstance(data, list):
    rows = data
elif isinstance(data, dict):
    for key in ("executions", "plans", "orders"):
        if isinstance(data.get(key), list):
            rows = data[key]
            break

def pick(d, *ks):
    for k in ks:
        if k in d and d[k] is not None:
            return d[k]
    return ""

flat = []
for r in rows:
    flat.append({
        "timestamp":    pick(r, "timestamp", "time", "ts"),
        "symbol":       pick(r, "symbol", "sym"),
        "side":         pick(r, "side", "action", "dir"),
        "size":         pick(r, "size", "qty", "quantity"),
        "price":        pick(r, "price", "px"),
        "reason":       pick(r, "reason", "why"),
        "source":       pick(r, "source", "stage"),
        "enter":        pick(r, "enter"),
        "primary_conf": pick(r, "primary_conf", "primary", "p_conf"),
        "finrl_conf":   pick(r, "finrl_conf", "rl_conf", "f_conf"),
    })

fields = ["timestamp", "symbol", "side", "size", "price", "reason", "source", "enter", "primary_conf", "finrl_conf"]
with out.open("w", newline="", encoding="utf-8") as g:
    wr = csv.DictWriter(g, fieldnames=fields)
    wr.writeheader()
    for r in flat:
        wr.writerow({k: r.get(k, "") for k in fields})

print("WROTE", out.resolve(), "rows=", len(flat))
