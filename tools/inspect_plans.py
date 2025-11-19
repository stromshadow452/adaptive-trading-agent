# tools/inspect_plans.py
import json, pathlib
from pprint import pprint

P = pathlib.Path("reports/screener/approved_smoke.json")
try:
    obj = json.loads(P.read_text(encoding="utf-8"))
except Exception as e:
    print("failed to read JSON:", e); raise SystemExit(2)

def get_symbol(record):
    def _norm(s): return (s or "").upper().replace("-","").replace("_","").replace("/","").replace("USDT","USD")
    def _extract(d, *paths, default=None):
        for p in paths:
            cur = d
            try:
                for k in p.split("."):
                    if isinstance(cur, dict) and k in cur:
                        cur = cur[k]
                    else:
                        cur = None; break
                if cur not in (None,""): return cur
            except Exception:
                pass
        return default
    if isinstance(record, dict):
        cand = record.get("candidate", {}) if isinstance(record.get("candidate"), dict) else {}
        raw = _extract(record, "symbol", "pair", "asset",
                       "candidate.symbol", "candidate.pair", "candidate.asset", default="UNKNOWN")
        sym = _norm(raw)
        if sym: return sym
    return "UNKNOWN"

if isinstance(obj, list):
    for i, item in enumerate(obj):
        print("ITEM", i, "type", type(item).__name__)
        if isinstance(item, dict):
            print("  keys:", list(item.keys()))
        pprint(item)
        print("  -> extracted symbol:", get_symbol(item))
        print("-"*60)
else:
    print("Top-level not list/dict. repr:", repr(obj)[:400])
