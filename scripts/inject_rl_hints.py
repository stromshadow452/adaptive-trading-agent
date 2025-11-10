#!/usr/bin/env python3
"""
Inject RL hints (rl_action/rl_conf or rl_prob_long/rl_prob_short)
into an existing screener JSON so two-layer fusion can run immediately.
"""

import argparse, json, random, math, os
from typing import Any, Dict, List

def safe_float(x: Any, d: float = 0.0) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return d
        return v
    except Exception:
        return d

def sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except Exception:
        return 0.5

def infer_side(score: float) -> str:
    return "long" if score >= 0 else "short"

def inject_rl(cand: Dict[str, Any], mode: str = "action") -> Dict[str, Any]:
    sc = safe_float(cand.get("score"), 0.0)
    rsi = safe_float(cand.get("rsi14"), 50.0)
    regime = (cand.get("regime") or "").lower()

    base_conf = sigmoid(abs(sc) * 2.0)
    noise = random.uniform(-0.08, 0.08)
    conf = max(0.0, min(1.0, base_conf + noise))

    side = infer_side(sc)
    if 40 <= rsi <= 60 and regime in ("meanrev", "range"):
        conf = max(0.0, min(1.0, conf - 0.05))

    if mode == "prob":
        if side == "long":
            cand["rl_prob_long"] = round(conf, 4)
            cand["rl_prob_short"] = round(1.0 - conf, 4)
        else:
            cand["rl_prob_long"] = round(1.0 - conf, 4)
            cand["rl_prob_short"] = round(conf, 4)
    else:
        cand["rl_action"] = side
        cand["rl_conf"] = round(conf, 4)

    return cand

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--mode", choices=["action", "prob"], default="action")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed or 0)

    with open(args.inp, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict) and "candidates" in raw:
        wrap = True
        cands = raw["candidates"]
    else:
        wrap = False
        cands = raw

    out_cands = [inject_rl(dict(c), args.mode) for c in cands]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"candidates": out_cands} if wrap else out_cands, f, indent=2)

    print(f"Injected RL hints into {len(out_cands)} candidates -> {args.out}")

if __name__ == "__main__":
    main()
