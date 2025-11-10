
#!/usr/bin/env python
"""
Minimal paper executor:
- Loads a plan JSON (with entry/SL/TP)
- Simulates a simple equity curve for one hypothetical trade
- Saves *_equity.csv into --out (default: reports/daily_logs)

Usage:
  python tools/paper_executor.py --plan reports/daily/PAIR_TF_plan_....json --equity 10000 --out reports/daily_logs
"""

from __future__ import annotations
import argparse, json, hashlib, math
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd

def utc_ts():
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

def seeded_rng(seed_str: str):
    h = int(hashlib.sha256(seed_str.encode("utf-8")).hexdigest(), 16) % (2**32 - 1)
    return np.random.default_rng(h)

def simulate_trade(entry: float, sl: float | None, tp: float | None, action: str, atr: float, steps: int = 200):
    """
    Very simple 1D price path:
      - If action=buy, drift positive; if sell, drift negative.
      - Vol ~ atr fraction
      - Stops when SL/TP reached or last step
    Returns: prices(np.array), hit("tp"/"sl"/"none")
    """
    if entry <= 0 or math.isnan(entry):
        entry = 1.0
    vol = max(1e-9, atr) * 0.2  # softer than ATR
    drift = (0.10 / steps) * (1 if action == "buy" else -1 if action == "sell" else 0)

    rng = seeded_rng(f"{entry}-{sl}-{tp}-{action}-{atr}-{steps}")
    prices = [entry]
    hit = "none"
    for _ in range(1, steps):
        shock = rng.normal(0.0, vol)
        next_price = prices[-1] * (1 + drift + shock / max(1e-9, prices[-1]))
        prices.append(next_price)
        if tp is not None and ((action == "buy" and next_price >= tp) or (action == "sell" and next_price <= tp)):
            hit = "tp"; break
        if sl is not None and ((action == "buy" and next_price <= sl) or (action == "sell" and next_price >= sl)):
            hit = "sl"; break
    return np.array(prices), hit

def equity_from_prices(prices: np.ndarray, equity0: float, size_pct: float, entry: float, action: str):
    """
    Mark-to-market PnL on notional = equity0 * size_pct.
    For simplicity, 1 unit per price point. Adjust as needed for FX contract sizes.
    """
    notional = equity0 * max(0.0, float(size_pct))
    if notional <= 0:
        notional = equity0 * 0.01
    direction = 1 if action == "buy" else -1 if action == "sell" else 0
    pnl_series = direction * (prices - entry)  # 1 unit
    equity = equity0 + pnl_series
    equity[equity < 0] = 0.0
    return equity

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True, help="path to plan JSON")
    ap.add_argument("--equity", type=float, default=10000.0)
    ap.add_argument("--out", default="reports/daily_logs")
    args = ap.parse_args()

    plan_path = Path(args.plan)
    if not plan_path.exists():
        print(f"Plan not found: {plan_path}")
        return

    plan = json.load(open(plan_path, "r", encoding="utf-8"))
    pair = plan.get("pair", "UNKNOWN")
    tf = plan.get("tf", "TF")
    entry = float(plan.get("entry", 0.0) or 0.0)
    sl = plan.get("sl", None)
    sl = None if sl is None else float(sl)
    tp = plan.get("tp", None)
    tp = None if tp is None else float(tp)
    action = plan.get("action", "hold")
    atr = float(plan.get("raw_candidate", {}).get("atr", 0.0) or 0.0)
    size_pct = float(plan.get("size_pct", 0.01) or 0.01)

    # If hold/no SL/TP, still simulate a flat-ish path so we generate equity
    if action == "hold":
        # Nudge tiny drift by sign of final_score to create label variety
        drift_hint = float(plan.get("final_score", 0.0))
        prices = [entry if entry > 0 else 1.0]
        rng = seeded_rng(json.dumps(plan, sort_keys=True))
        vol = max(1e-9, atr) * 0.1
        for _ in range(200):
            shock = rng.normal(drift_hint * 1e-4, vol)
            prices.append(max(1e-9, prices[-1] + shock))
        prices = np.array(prices)
        hit = "none"
    else:
        prices, hit = simulate_trade(entry, sl, tp, action, atr, steps=200)

    eq = equity_from_prices(prices, args.equity, size_pct, entry if entry > 0 else prices[0], action)

    # Build a simple time index (1-min bars)
    t0 = datetime.now(timezone.utc)
    idx = pd.date_range(t0, periods=len(eq), freq="1min")
    df = pd.DataFrame({"timestamp": idx, "equity": eq, "price": prices})
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    ts = utc_ts()
    out_csv = out_dir / f"{pair}_{tf}_paper_{ts}_equity.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved equity: {out_csv} (hit={hit})")

if __name__ == "__main__":
    main()
