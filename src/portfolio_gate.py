#!/usr/bin/env python3
"""
Portfolio Gate (standalone)

Filters trade plans to enforce portfolio-level discipline:
- same-side cap
- total positions cap
- net exposure cap (approx by sum(size*sign))
- per-symbol limit
- per-bucket-per-side caps (fx/crypto/equity)
- drawdown stop (from reports/aggregate/aggregate_summary.json)

Usage examples (PowerShell-friendly):

# Use the default preset on ONLY the latest 8 plans
python scripts/portfolio_gate.py `
  --plans_dir reports/daily `
  --latest 8 `
  --out_approved reports/daily/approved.json `
  --out_rejected reports/daily/rejected.json

# Use tight preset
python scripts/portfolio_gate.py `
  --plans_dir reports/daily_run `
  --profile tight `
  --out_approved reports/daily/approved.json `
  --out_rejected reports/daily/rejected.json

# Use loose preset + custom overrides from config/portfolio.json
python scripts/portfolio_gate.py `
  --plans_dir reports/daily `
  --profile loose `
  --config config/portfolio.json `
  --out_approved reports/daily/approved.json `
  --out_rejected reports/daily/rejected.json
"""

from __future__ import annotations
import os, json, glob, argparse, math
from typing import Any, Dict, List, Tuple

# -------- presets (match your JSONs) ----------
PRESETS = {
    "default": {
        "max_same_side": 3,
        "max_total_positions": 6,
        "max_net_exposure": 0.40,
        "max_bucket_per_side": {"fx": 2, "crypto": 2, "equity": 2},
        "max_drawdown_stop": 0.05,
        "per_symbol_limit": 1,
    },
    "tight":   {  # fewer trades / tighter risk
        "max_same_side": 2,
        "max_total_positions": 4,
        "max_net_exposure": 0.30,
        "per_symbol_limit": 1,
        # bucket caps/drawdown fall back to defaults unless provided by --config
    },
    "loose":   {  # more trades / looser
        "max_same_side": 3,
        "max_total_positions": 6,
        "max_net_exposure": 0.45,
        "per_symbol_limit": 1,
    }
}

def safe_float(x: Any, d: float = 0.0) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return d
        return v
    except Exception:
        return d

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

def list_plan_files(plans_dir: str) -> List[str]:
    # only consider *_plan_*.json to avoid picking up approved/rejected
    files = glob.glob(os.path.join(plans_dir, "*_plan_*.json"))
    files.sort(key=os.path.getmtime, reverse=True)
    return files

def infer_bucket(symbol: str) -> str:
    s = (symbol or "").upper()
    # crude but effective
    if s.endswith("USD") or s.startswith("USD") or any(k in s for k in ["EUR","JPY","GBP","AUD","NZD","CAD","CHF"]):
        return "fx"
    if "BTC" in s or "ETH" in s or s.endswith("BTC") or s.endswith("ETH"):
        return "crypto"
    return "equity"

def load_equity_state(aggregate_path: str) -> Dict[str, float]:
    """
    Looks for aggregate_summary.json with:
    { "equity": 12345.0, "peak_equity": 13000.0 }
    Returns dict with equity, peak_equity, dd (drawdown fraction 0..1)
    """
    if not aggregate_path or not os.path.exists(aggregate_path):
        return {"equity": 10000.0, "peak_equity": 10000.0, "dd": 0.0}
    try:
        data = load_json(aggregate_path)
        eq = safe_float(data.get("equity"), 10000.0)
        pk = max(eq, safe_float(data.get("peak_equity"), eq))
        dd = 0.0 if pk <= 0 else max(0.0, (pk - eq) / pk)
        return {"equity": eq, "peak_equity": pk, "dd": dd}
    except Exception:
        return {"equity": 10000.0, "peak_equity": 10000.0, "dd": 0.0}

def merge_config(profile: str, config_path: str | None) -> Dict[str, Any]:
    # start from default preset to ensure bucket/drawdown exist
    cfg = dict(PRESETS["default"])
    # overlay chosen preset
    if profile in PRESETS and profile != "default":
        cfg.update(PRESETS[profile])
    # overlay custom config if provided
    if config_path and os.path.exists(config_path):
        try:
            user_cfg = load_json(config_path)
            if isinstance(user_cfg, dict):
                cfg.update(user_cfg)
        except Exception:
            pass
    # ensure nested dict defaults for bucket caps
    if "max_bucket_per_side" not in cfg:
        cfg["max_bucket_per_side"] = {"fx": 2, "crypto": 2, "equity": 2}
    return cfg

def sign(side: str) -> int:
    s = (side or "").lower()
    if s == "buy": return +1
    if s == "sell": return -1
    return 0

def apply_portfolio_gate(plans: List[Dict[str, Any]], cfg: Dict[str, Any], eq_state: Dict[str, float]) -> Tuple[List[Dict[str,Any]], List[Dict[str,Any]]]:
    """
    Returns (approved, rejected_with_reason)
    """
    if not plans:
        return [], []

    # Drawdown hard stop
    dd = safe_float(eq_state.get("dd"), 0.0)
    dd_stop = safe_float(cfg.get("max_drawdown_stop", 0.05), 0.05)
    if dd >= dd_stop:
        rejected = []
        for p in plans:
            if p.get("enter", False):
                q = dict(p)
                q["gate_reason"] = {"mode": "drawdown_stop", "dd": dd, "limit": dd_stop}
                rejected.append(q)
        return [], rejected

    max_same_side = int(cfg.get("max_same_side", 3))
    max_total = int(cfg.get("max_total_positions", 6))
    max_net_exp = safe_float(cfg.get("max_net_exposure", 0.40))
    per_symbol_limit = int(cfg.get("per_symbol_limit", 1))
    bucket_caps = cfg.get("max_bucket_per_side", {"fx": 2, "crypto": 2, "equity": 2})

    # Counters
    counts_side = {"buy": 0, "sell": 0}
    counts_total = 0
    counts_symbol: Dict[str, int] = {}
    counts_bucket = {"fx": {"buy": 0, "sell": 0}, "crypto": {"buy": 0, "sell": 0}, "equity": {"buy": 0, "sell": 0}}
    net_exposure = 0.0

    # Sort by absolute final_score (confidence) descending
    def score_key(p):
        try:
            return abs(float(p.get("final_score", 0.0)))
        except Exception:
            return 0.0

    approved: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []

    for p in sorted(plans, key=score_key, reverse=True):
        if not p.get("enter", False):
            # skip non-entries silently
            continue

        symbol = (p.get("symbol") or "").upper()
        side_str = (p.get("side") or "hold").lower()
        if side_str not in ("buy", "sell"):
            q = dict(p); q["gate_reason"] = {"mode": "hold_side"}
            rejected.append(q); continue

        size = safe_float(p.get("size"), 0.0)
        bucket = infer_bucket(symbol)

        # per-symbol
        if counts_symbol.get(symbol, 0) >= per_symbol_limit:
            q = dict(p); q["gate_reason"] = {"mode": "per_symbol_limit", "symbol": symbol, "limit": per_symbol_limit}
            rejected.append(q); continue

        # same-side
        if counts_side[side_str] >= max_same_side:
            q = dict(p); q["gate_reason"] = {"mode": "same_side_cap", "side": side_str, "limit": max_same_side}
            rejected.append(q); continue

        # total cap
        if counts_total >= max_total:
            q = dict(p); q["gate_reason"] = {"mode": "max_total_positions", "limit": max_total}
            rejected.append(q); continue

        # bucket cap
        bucket_limit = int(bucket_caps.get(bucket, 2))
        if counts_bucket[bucket][side_str] >= bucket_limit:
            q = dict(p); q["gate_reason"] = {"mode": "bucket_side_cap", "bucket": bucket, "side": side_str, "limit": bucket_limit}
            rejected.append(q); continue

        # net exposure cap (approximate)
        trial = net_exposure + (size * sign(side_str))
        if abs(trial) > max_net_exp:
            q = dict(p); q["gate_reason"] = {"mode": "net_exposure_cap", "trial": trial, "limit": max_net_exp}
            rejected.append(q); continue

        # approve and update counts
        approved.append(p)
        counts_side[side_str] += 1
        counts_total += 1
        counts_symbol[symbol] = counts_symbol.get(symbol, 0) + 1
        counts_bucket[bucket][side_str] += 1
        net_exposure = trial

    return approved, rejected

def main():
    ap = argparse.ArgumentParser(description="Portfolio Gate (standalone)")
    ap.add_argument("--plans_dir", required=True, help="Folder with *_plan_*.json")
    ap.add_argument("--out_approved", required=True, help="Path to write approved.json")
    ap.add_argument("--out_rejected", required=True, help="Path to write rejected.json")
    ap.add_argument("--aggregate", default="reports/aggregate/aggregate_summary.json", help="Equity/DD summary JSON")
    ap.add_argument("--profile", choices=list(PRESETS.keys()), default="default", help="Preset: default|tight|loose")
    ap.add_argument("--config", default=None, help="Optional JSON config that overrides the preset")
    ap.add_argument("--latest", type=int, default=0, help="Only consider latest N plan files (0=all)")
    args = ap.parse_args()

    files = list_plan_files(args.plans_dir)
    if args.latest and args.latest > 0:
        files = files[:args.latest]

    plans: List[Dict[str, Any]] = []
    for f in files:
        try:
            plans.append(load_json(f))
        except Exception:
            # ignore unreadable
            pass

    cfg = merge_config(args.profile, args.config)
    eq_state = load_equity_state(args.aggregate)
    approved, rejected = apply_portfolio_gate(plans, cfg, eq_state)

    save_json(args.out_approved, approved)
    save_json(args.out_rejected, rejected)

    # summary
    reasons: Dict[str, int] = {}
    for r in rejected:
        m = (r.get("gate_reason") or {}).get("mode", "unknown")
        reasons[m] = reasons.get(m, 0) + 1

    print(f"Plans in: {len(plans)} | Approved: {len(approved)} | Rejected: {len(rejected)}")
    if rejected:
        print("Reject reasons:", reasons)
    print("Config used:", json.dumps(cfg, indent=2))
    print("Equity state:", json.dumps(eq_state, indent=2))

if __name__ == "__main__":
    main()
