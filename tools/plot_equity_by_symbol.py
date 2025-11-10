#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict
from pathlib import Path
import math
import matplotlib.pyplot as plt

EXEC = Path("reports/executions/executions_mark2close.csv")
OUT  = Path("reports/aggregate/equity_by_symbol.png")

def to_bool(x): return str(x).strip().lower() in ("true","1","yes","y")
def to_float(x, d=None):
    try:
        v = float(x); import math as m
        return d if (m.isnan(v) or m.isinf(v)) else v
    except Exception: return d

def load_trades():
    if not EXEC.exists():
        raise SystemExit(f"Missing: {EXEC} — run mark2close first.")
    trades_by_symbol = defaultdict(list)
    with EXEC.open(newline="", encoding="utf-8") as fh:
        rdr = csv.DictReader(fh)
        for r in rdr:
            side = (r.get("side") or "").lower()
            if side not in ("buy","sell"): continue
            if not to_bool(r.get("executed")): continue
            sym = (r.get("symbol") or "").upper()
            pnl = to_float(r.get("pnl"), None)
            if pnl is None: continue
            trades_by_symbol[sym].append(pnl)
    return trades_by_symbol

def make_equity(pnls):
    s = 0.0; eq = []
    for p in pnls: s += p; eq.append(s)
    return eq

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ncols", type=int, default=3)
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    data = load_trades()
    if not data:
        raise SystemExit("No realized trades found to plot.")

    # sort by total PnL (best -> worst)
    symbols = sorted(data.keys(), key=lambda s: sum(data[s]), reverse=True)

    n = len(symbols); ncols = max(1, int(args.ncols)); nrows = (n + ncols - 1) // ncols
    plt.figure(figsize=(5*ncols, 3.2*nrows), dpi=args.dpi)

    for i, sym in enumerate(symbols, start=1):
        pnls = data[sym]; eq = make_equity(pnls)
        xs = list(range(len(eq)))
        wins_x = [k for k,p in enumerate(pnls) if p>0]; wins_y = [eq[k] for k in wins_x]
        loss_x = [k for k,p in enumerate(pnls) if p<=0]; loss_y = [eq[k] for k in loss_x]
        ax = plt.subplot(nrows, ncols, i)
        ax.plot(xs, eq, linewidth=2, label=sym)
        ax.scatter(wins_x, wins_y, s=60, color='green', label="Win")
        ax.scatter(loss_x, loss_y, s=60, color='red', label="Loss")
        ax.set_title(sym); ax.set_xlabel("Trade #"); ax.set_ylabel("Cumulative PnL")
        ax.grid(alpha=0.3); ax.legend(loc="best", fontsize=8)

    plt.suptitle("Equity by Symbol (Cumulative PnL)", y=0.995)
    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, bbox_inches="tight")
    print(f"[OK] saved -> {OUT.as_posix()}")

if __name__ == "__main__":
    main()
