# tools/make_live_selection.py
import os, glob, json, pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPORTS = os.path.join(ROOT, "reports"); os.makedirs(REPORTS, exist_ok=True)

def latest(pattern):
    f = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not f: raise FileNotFoundError(pattern)
    return f[-1]

wf  = latest(os.path.join(REPORTS, "wf_summary_*.csv"))
df  = pd.read_csv(wf).copy()

df = df.sort_values(['wf_sharpe_mean','wf_ret_mean'], ascending=False)
out = []
per_family_cap = 10
family_count = {}

for _, r in df.iterrows():
    fam = r['family']
    if family_count.get(fam, 0) >= per_family_cap:
        continue
    out.append(r)
    family_count[fam] = family_count.get(fam, 0) + 1
    if len(out) >= 40:
        break

sel = pd.DataFrame(out)
json_obj = sel.to_dict(orient="records")
path_csv  = os.path.join(REPORTS, "live_selection.csv")
path_json = os.path.join(REPORTS, "live_selection.json")
sel.to_csv(path_csv, index=False)
with open(path_json, "w", encoding="utf-8") as f:
    json.dump(json_obj, f, indent=2)
print(f"✅ Live selection written:\n- {path_csv}\n- {path_json}\nCounts per family:", sel['family'].value_counts().to_dict())
