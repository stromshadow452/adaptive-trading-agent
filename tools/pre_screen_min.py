# tools/pre_screen_min.py
import os, json, traceback
from datetime import datetime
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd

print(">> START pre_screen_min", flush=True)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REG  = os.path.join(ROOT, "strategy_bank", "normalized_yaml", "_registry.json")
REPORTS = os.path.join(ROOT, "reports"); os.makedirs(REPORTS, exist_ok=True)

print("ROOT:", ROOT, flush=True)
print("REG exists:", os.path.isfile(REG), flush=True)

reg = json.load(open(REG, "r", encoding="utf-8"))
print("Registry size:", len(reg), flush=True)

from backtesting.test import GOOG
prices = GOOG.rename(columns=str.title)
print("Price rows:", len(prices), flush=True)

class TrendMA(Strategy):
    n_fast = 10; n_slow = 30
    def init(self):
        p = self.data.Close
        self.a = self.I(pd.Series(p).rolling(self.n_fast).mean)
        self.b = self.I(pd.Series(p).rolling(self.n_slow).mean)
    def next(self):
        if crossover(self.a, self.b): self.position.close(); self.buy()
        if crossover(self.b, self.a): self.position.close(); self.sell()

rows, ok = [], 0
N = min(50, len(reg))
print("Running", N, "quick tests...", flush=True)
for i in range(N):
    s = reg[i]
    try:
        bt = Backtest(prices, TrendMA, cash=10000, commission=.0005, finalize_trades=True)
        st = bt.run()
        rows.append({
            "id": s["id"], "vendor": s["source_vendor"], "name": s["name"],
            "return_pct": float(st.get("Return [%]", 0)),
            "sharpe": float(st.get("Sharpe Ratio", 0)),
            "trades": int(st.get("# Trades", 0)),
        })
        ok += 1
    except Exception as e:
        print(f"[WARN] {s['id']} failed: {e}", flush=True)
        traceback.print_exc()
    if (i+1) % 10 == 0:
        print(f"... {i+1}/{N} done", flush=True)

ts = datetime.now().strftime("%Y%m%d_%H%M")
out = os.path.join(REPORTS, f"prescreen_demo_{ts}.csv")
pd.DataFrame(rows).to_csv(out, index=False)
print(f">> DONE. Wrote: {out} (rows={len(rows)}, ok={ok}/{N})", flush=True)
