# tools/plan_summary.py
#!/usr/bin/env python3
import json, sys
from pathlib import Path
import csv

approved = Path("reports/daily/approved_paths_actionable_filtered.json")
if not approved.exists():
    print("approved list not found:", approved); sys.exit(1)

paths = json.loads(approved.read_text(encoding="utf-8-sig"))
rows = []
for p in paths:
    j = json.loads(Path(p).read_text(encoding="utf-8-sig"))
    decision = j.get("decision", {}) or {}
    # overlay top-level
    for k in ("enter","side","size","price"):
        if k not in decision and k in j:
            decision[k] = j[k]
    candidate = j.get("candidate", {})
    rows.append({
      "file": str(p),
      "symbol": Path(p).name.split("_",1)[0],
      "enter": decision.get("enter", False),
      "side": decision.get("side") or j.get("side") or "",
      "size": decision.get("size") or j.get("size") or 0,
      "price": decision.get("price") or j.get("price") or candidate.get("price") or 0,
      "candidate_price": candidate.get("price") or ""
    })

w = csv.DictWriter(sys.stdout, fieldnames=["file","symbol","enter","side","size","price","candidate_price"])
w.writeheader()
for r in rows:
    w.writerow(r)
