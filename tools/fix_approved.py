import json, sys, os

p = "reports/daily/approved.json"
if not os.path.exists(p):
    print("File not found:", p)
    sys.exit(1)

with open(p, "r", encoding="utf8") as f:
    data = json.load(f)

# handle two possible shapes:
# 1) { "approved": [ ... ] }  OR 2) [ {...}, {...} ]
if isinstance(data, dict) and "approved" in data and isinstance(data["approved"], list):
    L = data["approved"]
    wrapper = True
else:
    L = data if isinstance(data, list) else []
    wrapper = False

updated = 0
for e in L:
    # keep existing side/size if present; only fill missing/invalid
    side = e.get("side")
    if not isinstance(side, str) or side.strip() == "":
        e["side"] = "buy"
        updated += 1
    try:
        size = float(e.get("size", 0) or 0)
    except Exception:
        size = 0.0
    if size <= 0.0:
        e["size"] = 0.05
        updated += 1

out = data
if wrapper:
    out = {"generated_at": data.get("generated_at"), "count": len(L), "approved": L}

with open(p, "w", encoding="utf8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)

print(f"Updated {p} — entries: {len(L)}, fields updated: {updated}")
