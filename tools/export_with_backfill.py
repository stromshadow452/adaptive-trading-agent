import json, csv, pathlib, sys
import pandas as pd

DATA_DIR = pathlib.Path("data/raw/synthetic_M15")
AGG      = pathlib.Path("reports/executions/strict_agg.json")
OUT      = pathlib.Path("reports/executions/executions_run_strict.csv")

if not AGG.exists():
    print("ERROR: missing aggregate file:", AGG); sys.exit(1)

# 1) Build symbol -> last_ts map from CSVs
sym_last_ts = {}
for p in DATA_DIR.glob("*_M15_*.csv"):
    try:
        df = pd.read_csv(p, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(p, encoding="utf-8-sig")
    if df.empty: 
        continue
    cols = {c.lower(): c for c in df.columns}
    ts_col = cols.get("timestamp") or cols.get("time")
    if not ts_col:
        continue
    last_ts = str(df[ts_col].iloc[-1])
    if "T" not in last_ts or not last_ts.endswith("Z"):
        last_ts = last_ts.replace(" ", "T")
        if not last_ts.endswith("Z"): last_ts += "Z"
    sym = p.stem.split("_")[0]
    sym_last_ts[sym] = last_ts

# 2) Load aggregate and flatten
data = json.loads(AGG.read_text(encoding="utf-8"))
if isinstance(data, list):
    rows = data
elif isinstance(data, dict):
    rows = data.get("executions") or data.get("plans") or data.get("orders") or []
else:
    rows = []

def pick(d, *ks):
    for k in ks:
        if k in d and d[k] is not None:
            return d[k]
    return ""

flat = []
for r in rows:
    sym = pick(r, "symbol", "sym")
    ts  = pick(r, "timestamp", "time", "ts") or sym_last_ts.get(sym, "")
    flat.append({
        "timestamp":    ts,
        "symbol":       sym,
        "side":         pick(r,"side","action","dir"),
        "size":         pick(r,"size","qty","quantity"),
        "price":        pick(r,"price","px"),
        "reason":       pick(r,"reason","why"),
        "source":       pick(r,"source","stage"),
        "enter":        pick(r,"enter"),
        "primary_conf": pick(r,"primary_conf","primary","p_conf"),
        "finrl_conf":   pick(r,"finrl_conf","rl_conf","f_conf"),
    })

OUT.parent.mkdir(parents=True, exist_ok=True)
fields = ["timestamp","symbol","side","size","price","reason","source","enter","primary_conf","finrl_conf"]
with OUT.open("w", newline="", encoding="utf-8") as g:
    wr = csv.DictWriter(g, fieldnames=fields)
    wr.writeheader()
    for r in flat:
        wr.writerow({k: r.get(k, "") for k in fields})

print("WROTE", OUT.resolve(), "rows=", len(flat))
