#!/usr/bin/env python3
import json, sys, math
from datetime import datetime

IN  = r"reports/daily/approved_fixed.json"
OUT = r"reports/daily/approved_top1.json"
MODE = "score"  # "score" or "latest"

def sfloat(x, d=0.0):
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return d
        return v
    except Exception:
        return d

def norm_symbol(p):
    # robust symbol extraction + normalization
    s = (
        p.get("symbol")
        or p.get("pair")
        or (p.get("candidate", {}) or {}).get("symbol")
        or (p.get("candidate", {}) or {}).get("pair")
        or ""
    )
    s = str(s).upper().replace("-", "").replace("_", "").replace("/", "")
    s = s.replace("USDT", "USD")
    return s

def plan_side(p):
    sd = (p.get("side") or p.get("direction") or p.get("bias") or "").strip().lower()
    if sd in ("buy","long","b"):  return "buy"
    if sd in ("sell","short","s"): return "sell"
    return "hold"

def is_enterable(p):
    # be lenient: side buy/sell -> enterable; else if 'enter' True use that
    sd = plan_side(p)
    if sd in ("buy","sell"):
        return True
    ent = p.get("enter")
    return bool(ent)

def parse_dt(s):
    if not s: return None
    # support "20251030T183706Z" or ISO
    try:
        if isinstance(s, str) and s.endswith("Z") and "T" in s and len(s)>=16 and s[8]=="T"[0]:
            # compact Zulu format
            return datetime.strptime(s, "%Y%m%dT%H%M%SZ")
    except Exception:
        pass
    try:
        return datetime.fromisoformat(s.replace("Z","+00:00"))
    except Exception:
        return None

with open(IN, "r", encoding="utf-8-sig") as f:
    obj = json.load(f)

plans = obj.get("approved", obj)

# 1) Filter only candidates we can trade (by side or explicit enter flag)
eligible = []
for p in plans:
    sym = norm_symbol(p)
    if not sym:
        continue
    if not is_enterable(p):
        continue
    sd = plan_side(p)
    if sd not in ("buy","sell"):
        continue
    eligible.append(p)

if not eligible:
    # Fallback: if nothing eligible (due to flags), try picking latest per symbol regardless,
    # then executor will apply risk gates.
    bysym = {}
    for p in plans:
        sym = norm_symbol(p)
        if not sym:
            continue
        dt = parse_dt(p.get("generated_at")) or parse_dt((p.get("candidate",{}) or {}).get("generated_at"))
        prev = bysym.get(sym)
        if prev is None:
            bysym[sym] = (dt, p)
        else:
            if dt and (prev[0] is None or dt > prev[0]):
                bysym[sym] = (dt, p)
    kept = [pp for _, pp in bysym.values()]
else:
    # 2) Choose top-1 per symbol by final_score (abs or raw? Use raw to prefer directionally stronger)
    bysym = {}
    for p in eligible:
        sym = norm_symbol(p)
        sc = sfloat(p.get("final_score"), 0.0)
        prev = bysym.get(sym)
        if prev is None or sc > sfloat(prev.get("final_score"), 0.0):
            bysym[sym] = p
    kept = list(bysym.values())

out = {"approved": kept} if isinstance(obj, dict) and "approved" in obj else kept

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)

# Debug summary
syms = sorted({norm_symbol(p) for p in kept})
print(f"Symbols kept: {len(kept)} -> {OUT}")
print("Kept symbols:", ", ".join(syms) if syms else "(none)")
