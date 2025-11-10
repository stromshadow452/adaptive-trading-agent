#!/usr/bin/env python3
import json, os, glob, csv, math
from collections import defaultdict

APPROVED_IN   = r"reports/daily/approved.json"
CSV_PRICE_DIR = r"data/raw/forex_backup_2020_2025"
APPROVED_OUT  = r"reports/daily/approved_fixed.json"
VERBOSE       = True  # debug prints

def sfloat(x, d=0.0):
    try:
        v = float(x)
        return d if (math.isnan(v) or math.isinf(v)) else v
    except Exception:
        return d

def pick_close_from_row(row: dict):
    # normalize keys
    lk = {k.lower().strip(): k for k in row.keys() if k is not None}
    # direct close keys
    for k in ("close","c","closeprice","last","price","settle","adj_close","adjclose"):
        if k in lk:
            v = row.get(lk[k])
            if v not in (None, ""):
                try:
                    return float(v)
                except Exception:
                    pass
    # bid/ask mid fallback
    bid = ask = None
    for k in ("bid","bidclose","b"):
        if k in lk:
            try: bid = float(row[lk[k]])
            except: pass
    for k in ("ask","askclose","a","offer"):
        if k in lk:
            try: ask = float(row[lk[k]])
            except: pass
    if bid is not None and ask is not None:
        return (bid + ask) / 2.0
    return None

def read_last_price_from_csv(path: str):
    """
    Safe reader: open once, sniff delimiter, iterate rows, return last valid close.
    DOES NOT return a DictReader outside the 'with' block (no closed-file bug).
    """
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            sample = f.read(8192)
            f.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=[",",";","\t","|"])
            except Exception:
                dialect = csv.get_dialect("excel")
            reader = csv.DictReader(f, dialect=dialect)
            last_px = None
            for row in reader:
                px = pick_close_from_row(row)
                if px is not None:
                    last_px = px
            return last_px
    except KeyboardInterrupt:
        raise
    except Exception as e:
        if VERBOSE:
            print(f"[warn] read error {os.path.basename(path)}: {e}")
        return None

def last_close_from_csv(symbol: str):
    """
    Look for files like SYMBOL_*.csv or SYMBOL*.csv; prefer lexicographically newest.
    Read last valid price from the newest file that yields a value.
    """
    pats = [
        os.path.join(CSV_PRICE_DIR, f"{symbol}_*.csv"),
        os.path.join(CSV_PRICE_DIR, f"{symbol}*.csv"),
    ]
    files = []
    for p in pats:
        files.extend(glob.glob(p))
    if not files:
        return None
    files.sort()  # last is "newest" by name (your names include year ranges)
    for path in reversed(files):
        px = read_last_price_from_csv(path)
        if px is not None:
            if VERBOSE:
                print(f"[price] {symbol}: {px}  <- {os.path.basename(path)}")
            return px
    return None

def recompute_size(plan: dict, rcfg: dict) -> float:
    risk_per_trade = sfloat(rcfg.get("risk_per_trade"), 0.01)
    equity = sfloat(plan.get("equity"), sfloat(rcfg.get("default_equity"), 10000.0))
    price  = sfloat(plan.get("price"), 0.0)
    atr    = sfloat(plan.get("atr") or plan.get("atr14"), 0.0)
    stop_mult = sfloat(rcfg.get("stop_atr_mult"), 1.5)
    sl_pct = sfloat(rcfg.get("sl_pct"), 0.01)
    eps = 1e-9
    if price <= eps:
        return 0.0
    stop_dist = max(atr * stop_mult, eps) if atr > eps else max(price * sl_pct, eps)
    notional = equity * risk_per_trade
    size = notional / (stop_dist * price)
    size = max(sfloat(rcfg.get("min_size"), 0.001), min(sfloat(rcfg.get("max_size"), 0.05), size))
    return max(0.0, size)

def compute_sl_tp(plan: dict, rcfg: dict):
    price = sfloat(plan.get("price"), 0.0)
    atr   = sfloat(plan.get("atr") or plan.get("atr14"), 0.0)
    side  = (plan.get("side") or "").lower()
    stop_mult = sfloat(rcfg.get("stop_atr_mult"), 1.5)
    rr = sfloat(rcfg.get("takeprofit_rr"), 1.5)
    sl_pct = sfloat(rcfg.get("sl_pct"), 0.01)
    tp_pct = sfloat(rcfg.get("tp_pct"), 0.02)
    eps = 1e-9
    if price <= eps:
        return None, None, "none"
    if atr > eps:
        sd = atr * stop_mult
        tp = sd * rr
        if side == "buy":  return price - sd, price + tp, "atr"
        if side == "sell": return price + sd, price - tp, "atr"
    if side == "buy":  return price * (1 - sl_pct), price * (1 + tp_pct), "pct"
    if side == "sell": return price * (1 + sl_pct), price * (1 - tp_pct), "pct"
    return None, None, "none"

# --- load approved + risk config (with defaults) ---
with open(APPROVED_IN, "r", encoding="utf-8-sig") as f:
    obj = json.load(f)

plans = obj.get("approved", obj)  # support {"approved": [...]} or plain list

risk_cfg = {
    "risk_per_trade": 0.01, "default_equity": 10000.0,
    "stop_atr_mult": 1.5, "takeprofit_rr": 1.5,
    "min_size": 0.001, "max_size": 0.05,
    "sl_pct": 0.01, "tp_pct": 0.02
}
try:
    import yaml
    if os.path.exists("config/decision.yaml"):
        with open("config/decision.yaml","r",encoding="utf-8") as f:
            y = yaml.safe_load(f) or {}
            risk_cfg.update(y.get("decision", y) or {})
except Exception:
    pass

sym_stats = defaultdict(lambda: {"seen":0,"priced":0,"sized":0})
fixed = 0
missing_csv = set()

for p in plans:
    sym = (p.get("symbol") or p.get("candidate", {}).get("symbol") or "").upper().replace("USDT","USD")
    if not sym:
        continue
    sym_stats[sym]["seen"] += 1

    price = sfloat(p.get("price"), 0.0)
    size  = sfloat(p.get("size"), 0.0)

    # price backfill
    if price <= 0.0:
        px = last_close_from_csv(sym)
        if px and px > 0:
            p["price"] = px
            price = px
            sym_stats[sym]["priced"] += 1
        else:
            missing_csv.add(sym)
            if VERBOSE:
                print(f"[miss] {sym}: no price found in CSV dir")

    # size recompute
    if (size <= 0.0) and price > 0 and (p.get("side") in ("buy", "sell")):
        p["size"] = recompute_size(p, risk_cfg)
        if sfloat(p["size"], 0.0) > 0.0:
            sym_stats[sym]["sized"] += 1

    # SL/TP fill if missing
    if p.get("sl") is None or p.get("tp") is None:
        sl, tp, _ = compute_sl_tp(p, risk_cfg)
        if p.get("sl") is None: p["sl"] = sl
        if p.get("tp") is None: p["tp"] = tp

    if sfloat(p.get("size"), 0.0) > 0.0:
        fixed += 1

out = {"approved": plans} if isinstance(obj, dict) and "approved" in obj else plans
os.makedirs(os.path.dirname(APPROVED_OUT), exist_ok=True)
with open(APPROVED_OUT, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)

print(f"\nFixed plans with size>0: {fixed}")
if VERBOSE:
    print("\nPer-symbol stats:")
    for sym, st in sorted(sym_stats.items()):
        print(f"  {sym}: seen={st['seen']} priced={st['priced']} sized={st['sized']}")
    if missing_csv:
        print("\nSymbols with no CSV-derived price found:", ", ".join(sorted(missing_csv)))
print(f"Wrote: {APPROVED_OUT}")
