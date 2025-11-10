# tools/pre_screen.py
import os, json, glob, re
import numpy as np
import pandas as pd
from datetime import datetime
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# ========== Config ==========
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REG  = os.path.join(ROOT, "strategy_bank", "normalized_yaml", "_registry.json")
REPORTS_DIR = os.path.join(ROOT, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

# Point this to your MT4/MT5 EURUSD Daily files
CUSTOM_GLOB = r"E:\adaptive-trading-agent (2) (1)\adaptive-trading-agent (2)\data\raw\forex\forex\EURUSD_Daily*.csv"

CASH = 10_000
COMMISSION = 0.0005
FINALIZE_TRADES = True

# ========= Robust Data Loading =========
def _normalize_cols(cols):
    norm = {}
    for c in cols:
        k = c.lower().strip()
        k = re.sub(r"[<>]", "", k)    # <DATE> -> DATE
        k = re.sub(r"\s+", "", k)     # remove spaces/tabs
        norm[k] = c
    return norm

def _pick(norm_map, *names):
    for n in names:
        key = re.sub(r"[<>\s]+", "", n.lower())
        if key in norm_map:
            return norm_map[key]
    return None

def _read_ohlc_any(path):
    import pandas as pd
    df = None

    # Try comma, semicolon, then whitespace regex
    for kws in ({"sep": ","}, {"sep": ";"}, {"sep": r"\s+"}):
        try:
            df = pd.read_csv(path, **kws, engine="python")
            if df is not None and df.shape[1] >= 5:
                break
        except Exception:
            df = None

    if df is None:
        import os as _os
        raise ValueError(f"Cannot read CSV: {_os.path.basename(path)}")

    norm = _normalize_cols(df.columns)
    t_date = _pick(norm, "DATE")
    t_time = _pick(norm, "TIME")
    t_stamp = _pick(norm, "DATETIME", "TIMESTAMP")

    o = _pick(norm, "OPEN", "PRICEOPEN", "O")
    h = _pick(norm, "HIGH", "PRICEHIGH", "H")
    l = _pick(norm, "LOW", "PRICELOW", "L")
    c = _pick(norm, "CLOSE", "PRICECLOSE", "ADJCLOSE", "C", "LAST")
    v = _pick(norm, "VOLUME", "VOL", "TICKVOL", "TICKVOLUME", "V")

    if t_stamp:
        dt = pd.to_datetime(df[t_stamp], errors="coerce")
    elif t_date and t_time:
        dt = pd.to_datetime(df[t_date].astype(str) + " " + df[t_time].astype(str), errors="coerce")
    elif t_date:
        dt = pd.to_datetime(df[t_date], errors="coerce")
    else:
        import os as _os
        raise AssertionError(f"Time/Date columns not found in {_os.path.basename(path)} | cols={list(df.columns)}")

    needed = [o, h, l, c]
    if any(x is None for x in needed):
        import os as _os
        raise AssertionError(f"OHLC columns not found in {_os.path.basename(path)} | cols={list(df.columns)}")

    out = pd.DataFrame({
        "Date":  dt,
        "Open":  pd.to_numeric(df[o], errors="coerce"),
        "High":  pd.to_numeric(df[h], errors="coerce"),
        "Low":   pd.to_numeric(df[l], errors="coerce"),
        "Close": pd.to_numeric(df[c], errors="coerce"),
    })
    if v and v in df:
        out["Volume"] = pd.to_numeric(df[v], errors="coerce")

    out = out.dropna(subset=["Date", "Open", "High", "Low", "Close"])
    out = out.sort_values("Date").set_index("Date")
    return out

def load_prices():
    files = sorted(glob.glob(CUSTOM_GLOB))
    if files:
        dfs = []
        for f in files:
            try:
                dfs.append(_read_ohlc_any(f))
            except Exception as e:
                print(f"[WARN] Skip {os.path.basename(f)}: {e}")
        if dfs:
            out = pd.concat(dfs).sort_index()
            print(f">> Loaded EURUSD from {len(files)} files | rows={len(out)}")
            return out

    from backtesting.test import GOOG
    out = GOOG.rename(columns=str.title)
    print(f">> Fallback GOOG | rows={len(out)}")
    return out

# ========= Family-Proxy Strategies (fast templates) =========
class TrendMA(Strategy):
    n_fast = 5; n_slow = 20  # daily-friendly
    def init(self):
        p = self.data.Close
        self.ma_fast = self.I(pd.Series(p).rolling(self.n_fast).mean)
        self.ma_slow = self.I(pd.Series(p).rolling(self.n_slow).mean)
    def next(self):
        if crossover(self.ma_fast, self.ma_slow):
            self.position.close(); self.buy()
        elif crossover(self.ma_slow, self.ma_fast):
            self.position.close(); self.sell()

class MeanRSI(Strategy):
    rsi_n = 14; lo = 25; hi = 75
    def init(self):
        close = pd.Series(self.data.Close)
        delta = close.diff()
        up, down = delta.clip(lower=0), -delta.clip(upper=0)
        ru = up.ewm(alpha=1/self.rsi_n, adjust=False).mean()
        rd = down.ewm(alpha=1/self.rsi_n, adjust=False).mean()
        rs = ru/rd
        self.rsi = self.I(lambda: 100 - (100/(1+rs)))
    def next(self):
        if self.rsi[-1] < self.lo and not self.position.is_long:
            self.position.close(); self.buy()
        if self.rsi[-1] > self.hi and not self.position.is_short:
            self.position.close(); self.sell()

class BreakoutATR(Strategy):
    n = 20
    def init(self):
        hi = pd.Series(self.data.High); lo = pd.Series(self.data.Low)
        self.max_n = self.I(lambda: hi.rolling(self.n).max())
        self.min_n = self.I(lambda: lo.rolling(self.n).min())
    def next(self):
        if self.data.Close[-1] > self.max_n[-2]:
            self.position.close(); self.buy()
        elif self.data.Close[-1] < self.min_n[-2]:
            self.position.close(); self.sell()

FAMILY_TEMPLATE = {
    "trend": TrendMA,
    "mean_reversion": MeanRSI,
    "breakout": BreakoutATR,
}
def pick_strategy(fam: str):
    return FAMILY_TEMPLATE.get(fam, TrendMA)

# ========= Run =========
def main():
    with open(REG, "r", encoding="utf-8") as f:
        reg = json.load(f)

    prices = load_prices()
    print(f">> Registry size: {len(reg)}  |  Price rows: {len(prices)}", flush=True)

    rows = []
    for i, s in enumerate(reg, 1):
        StrategyCls = pick_strategy(s.get("family","unknown"))
        try:
            bt = Backtest(
                prices, StrategyCls, cash=CASH, commission=COMMISSION,
                finalize_trades=FINALIZE_TRADES
            )
            st = bt.run()
            rows.append({
                "id": s["id"], "vendor": s["source_vendor"], "name": s["name"],
                "family": s.get("family","unknown"),
                "return_pct": float(st.get("Return [%]", 0)),
                "sharpe": float(st.get("Sharpe Ratio", 0)),
                "trades": int(st.get("# Trades", 0)),
                "winrate_pct": float(st.get("Win Rate [%]", 0)),
                "maxdd_pct": float(st.get("Max. Drawdown [%]", 0)),
            })
        except Exception as e:
            rows.append({
                "id": s["id"], "vendor": s["source_vendor"], "name": s["name"],
                "family": s.get("family","unknown"),
                "return_pct": np.nan, "sharpe": np.nan, "trades": 0,
                "winrate_pct": np.nan, "maxdd_pct": np.nan, "error": str(e),
            })
        if i % 100 == 0:
            print(f"... processed {i}/{len(reg)}", flush=True)

    df = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)

    # Softer, daily-friendly pre-screen
    df["ok"] = (
        df["trades"].ge(5) &
        df["sharpe"].fillna(-9).gt(0.2) &
        df["maxdd_pct"].abs().lt(60)
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    all_path = os.path.join(REPORTS_DIR, f"prescreen_all_{ts}.csv")
    top_path = os.path.join(REPORTS_DIR, f"prescreen_top_{ts}.csv")

    df.to_csv(all_path, index=False)
    df[df["ok"]].sort_values(["sharpe","return_pct"], ascending=False).head(150).to_csv(top_path, index=False)

    print(f"✅ Prescreen done.\nAll: {all_path}\nTop: {top_path}\nPassed: {int(df['ok'].sum())}/{len(df)}", flush=True)

if __name__ == "__main__":
    main()
