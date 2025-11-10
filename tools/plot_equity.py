#!/usr/bin/env python3
import csv
from pathlib import Path
import matplotlib.pyplot as plt

CURVE = Path("reports/aggregate/equity_curve.csv")
EXEC  = Path("reports/executions/executions_mark2close.csv")
OUT   = Path("reports/aggregate/equity_curve_annotated.png")

def to_bool(x):
    s = str(x).strip().lower()
    return s in ("true", "1", "yes", "y")

def to_float(x, d=0.0):
    try:
        v = float(x)
        if v != v:  # NaN
            return d
        return v
    except Exception:
        return d

def load_equity():
    if not CURVE.exists():
        raise SystemExit(f"Missing equity curve: {CURVE} (run tools/equity_curve.py first)")
    xs, ys = [], []
    with CURVE.open() as f:
        next(f)  # skip header
        for line in f:
            i, v = line.strip().split(",")
            xs.append(int(i))
            ys.append(float(v))
    return xs, ys

def load_pnl_sequence():
    if not EXEC.exists():
        return []
    pnl = []
    with EXEC.open(newline="", encoding="utf-8") as fh:
        rdr = csv.DictReader(fh)
        for r in rdr:
            side = (r.get("side") or "").lower()
            if side not in ("buy", "sell"):
                continue
            if not to_bool(r.get("executed")):
                continue
            pnl.append(to_float(r.get("pnl"), None))
    return [x for x in pnl if x is not None]

def main():
    xs, equity = load_equity()
    pnl = load_pnl_sequence()

    n = min(len(equity), len(pnl))
    if n == 0:
        raise SystemExit("No trades to plot.")

    xs = list(range(n))
    equity = equity[:n]
    pnl = pnl[:n]

    win_x = [i for i, v in enumerate(pnl) if v > 0]
    lose_x = [i for i, v in enumerate(pnl) if v <= 0]
    base = equity[0]
    eq_norm = [v - base for v in equity]
    win_y = [eq_norm[i] for i in win_x]
    lose_y = [eq_norm[i] for i in lose_x]

    plt.figure(figsize=(8, 5))
    plt.plot(xs, eq_norm, linewidth=2, label="Equity")
    plt.scatter(win_x, win_y, s=80, zorder=5, label="Wins", color="green")
    plt.scatter(lose_x, lose_y, s=80, zorder=5, label="Losses", color="red")
    plt.title("Equity Curve with Trade Outcomes")
    plt.xlabel("Trade #")
    plt.ylabel("Equity Change")
    plt.grid(alpha=0.3)
    plt.legend()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"[OK] Enhanced equity chart saved -> {OUT.as_posix()}")

if __name__ == "__main__":
    main()
