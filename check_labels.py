import glob, json, numpy as np

files = glob.glob(r"reports/daily_logs\*_summary.json")
y = []
for f in files:
    try:
        j = json.load(open(f, encoding="utf-8"))
        y.append(1 if j.get("total_return", 0) > 0 else 0)
    except Exception:
        pass

y = np.array(y)
print("samples:", y.size, "| positives:", int((y==1).sum()), "| negatives:", int((y==0).sum()))
