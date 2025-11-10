#!/usr/bin/env python3
import csv, math, statistics, json
from pathlib import Path

INP = Path("reports/executions/executions_mark2close.csv")
OUT_EC = Path("reports/aggregate/equity_curve.csv")
OUT_SUM = Path("reports/aggregate/equity_summary.json")

def to_float(x, d=0.0):
    try:
        v=float(x)
        return d if math.isnan(v) or math.isinf(v) else v
    except: return d

def main(start_capital=10000.0):
    if not INP.exists():
        raise SystemExit(f"Missing: {INP}")
    with INP.open(newline="", encoding="utf-8") as fh:
        rdr = csv.DictReader(fh)
        pnl = [to_float(r.get("pnl"), None)
               for r in rdr
               if (str(r.get("executed","")).lower()=="true"
                   and str(r.get("side","")).lower() in ("buy","sell"))]
        pnl = [x for x in pnl if x is not None]

    eq = []
    e = start_capital
    peak = e
    mdd_abs = 0.0
    for x in pnl:
        e += x
        eq.append(e)
        peak = max(peak, e)
        mdd_abs = max(mdd_abs, peak - e)

    OUT_EC.parent.mkdir(parents=True, exist_ok=True)
    with OUT_EC.open("w", newline="", encoding="utf-8") as fh:
        fh.write("idx,equity\n")
        for i, v in enumerate(eq):
            fh.write(f"{i},{v}\n")

    trades = len(pnl)
    wins = sum(1 for x in pnl if x > 0)
    losses = sum(1 for x in pnl if x < 0)
    gp = sum(x for x in pnl if x > 0)
    gl = -sum(x for x in pnl if x < 0)
    pf = (gp/gl) if gl>0 else (float("inf") if gp>0 else 0.0)
    try:
        mu = statistics.mean(pnl); sd = statistics.pstdev(pnl) or 1e-12; sharpe = mu/sd
    except statistics.StatisticsError:
        sharpe = 0.0
    mdd_pct = (mdd_abs/peak) if peak>0 else 0.0

    summary = {
        "start_capital": start_capital,
        "end_capital": eq[-1] if eq else start_capital,
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "winrate": (wins/trades) if trades else 0.0,
        "profit_factor": pf,
        "sharpe_like": sharpe,
        "max_drawdown_abs": mdd_abs,
        "max_drawdown_pct": mdd_pct
    }
    with OUT_SUM.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[OK] equity -> {OUT_EC.as_posix()}")
    print(f"[OK] summary -> {OUT_SUM.as_posix()}")
    print(summary)

if __name__ == "__main__":
    main()
