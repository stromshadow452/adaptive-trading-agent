# tools/walk_forward.py
import os, glob, json, re
import numpy as np
import pandas as pd
from datetime import datetime
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# ===== Config =====
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REG  = os.path.join(ROOT, "strategy_bank", "normalized_yaml", "_registry.json")
REPORTS = os.path.join(ROOT, "reports"); os.makedirs(REPORTS, exist_ok=True)

CUSTOM_GLOB = r"E:\adaptive-trading-agent (2) (1)\adaptive-trading-agent (2)\data\raw\forex\forex\EURUSD_Daily*.csv"
CASH = 10_000
COMMISSION = 0.0005
FINALIZE_TRADES = True

# Walk-forward parameters (tuned for Daily data ~1300 rows)
FOLDS = 4
TEST_LEN = 120          # ~6 months per fold
MIN_TRAIN = 252         # ~1 year minimum training lookback (for param-based strats; proxies don’t fit but it spaces splits well)
MIN_TEST_BARS = 60

# ===== Helpers =====
def latest_top_csv():
    files = sorted(glob.glob(os.path.join(REPORTS, "prescreen_top_*.csv")), key=os.path.getmtime)
    if not files:
        raise FileNotFoundError("prescreen_top_*.csv not found in reports/")
    return files[-1]

def _normalize_cols(cols):
    norm = {}
    for c in cols:
        k = c.lower().strip()
        k = re.sub(r"[<>]", "", k)
        k = re.sub(r"\s+", "", k)
        norm[k] = c
    return norm

def _pick(norm_map, *names):
    for n in names:
        key = re.sub(r"[<>\s]+", "", n.lower())
        if key in norm_map:
            return norm_map[key]
    return None

def _read_ohlc_any(path):
    import pandas as pd, os as _os
    df = None
    for kws in ({"sep": ","}, {"sep": ";"}, {"sep": r"\s+"}):
        try:
            df = pd.read_csv(path, **kws, engine="python")
            if df is not None and df.shape[1] >= 5:
                break
        except Exception:
            df = None
    if df is None:
        raise ValueError(f"Cannot read CSV: {_os.path.basename(path)}")

    norm = _normalize_cols(df.columns)
    t_date = _pick(norm, "DATE"); t_time = _pick(norm, "TIME"); t_stamp = _pick(norm, "DATETIME","TIMESTAMP")
    o = _pick(norm, "OPEN","PRICEOPEN","O")
    h = _pick(norm, "HIGH","PRICEHIGH","H")
    l = _pick(norm, "LOW","PRICELOW","L")
    c = _pick(norm, "CLOSE","PRICECLOSE","ADJCLOSE","C","LAST")
    v = _pick(norm, "VOLUME","VOL","TICKVOL","TICKVOLUME","V")

    import pandas as pd
    if t_stamp:
        dt = pd.to_datetime(df[t_stamp], errors="coerce")
    elif t_date and t_time:
        dt = pd.to_datetime(df[t_date].astype(str) + " " + df[t_time].astype(str), errors="coerce")
    elif t_date:
        dt = pd.to_datetime(df[t_date], errors="coerce")
    else:
        raise AssertionError(f"Time/Date columns not found in {_os.path.basename(path)}")

    need = [o,h,l,c]
    if any(x is None for x in need):
        raise AssertionError(f"OHLC columns not found in {_os.path.basename(path)}")

    out = pd.DataFrame({
        "Date":  dt,
        "Open":  pd.to_numeric(df[o], errors="coerce"),
        "High":  pd.to_numeric(df[h], errors="coerce"),
        "Low":   pd.to_numeric(df[l], errors="coerce"),
        "Close": pd.to_numeric(df[c], errors="coerce"),
    })
    if v and v in df:
        out["Volume"] = pd.to_numeric(df[v], errors="coerce")

    out = out.dropna(subset=["Date","Open","High","Low","Close"]).sort_values("Date").set_index("Date")
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

# ===== Family proxy strategies =====
class TrendMA(Strategy):
    n_fast=5; n_slow=20
    def init(self):
        p=self.data.Close
        self.a=self.I(pd.Series(p).rolling(self.n_fast).mean)
        self.b=self.I(pd.Series(p).rolling(self.n_slow).mean)
    def next(self):
        if crossover(self.a,self.b): self.position.close(); self.buy()
        elif crossover(self.b,self.a): self.position.close(); self.sell()

class MeanRSI(Strategy):
    rsi_n=14; lo=25; hi=75
    def init(self):
        c=pd.Series(self.data.Close); d=c.diff()
        up,down=d.clip(lower=0), -d.clip(upper=0)
        ru=up.ewm(alpha=1/self.rsi_n, adjust=False).mean()
        rd=down.ewm(alpha=1/self.rsi_n, adjust=False).mean()
        rs=ru/rd
        self.rsi=self.I(lambda: 100-(100/(1+rs)))
    def next(self):
        if self.rsi[-1]<self.lo and not self.position.is_long: self.position.close(); self.buy()
        if self.rsi[-1]>self.hi and not self.position.is_short: self.position.close(); self.sell()

class BreakoutATR(Strategy):
    n=20
    def init(self):
        hi=pd.Series(self.data.High); lo=pd.Series(self.data.Low)
        self.maxn=self.I(lambda: hi.rolling(self.n).max())
        self.minn=self.I(lambda: lo.rolling(self.n).min())
    def next(self):
        if self.data.Close[-1]>self.maxn[-2]: self.position.close(); self.buy()
        elif self.data.Close[-1]<self.minn[-2]: self.position.close(); self.sell()

FAMILY_TEMPLATE={"trend":TrendMA,"mean_reversion":MeanRSI,"breakout":BreakoutATR}
def pick_strategy(fam): return FAMILY_TEMPLATE.get(fam, TrendMA)

# ===== Walk-forward splits (fixed-length tests) =====
def wf_splits(n, folds=FOLDS, test_len=TEST_LEN, min_train=MIN_TRAIN):
    """Return [(train_slice, test_slice), ...] using fixed test_len bars per fold."""
    idx = []
    total_test = folds * test_len
    start_train_end = max(min_train, n - total_test)
    for k in range(folds):
        train_end = start_train_end + k * test_len
        test_start = train_end
        test_end = min(test_start + test_len, n)
        if test_end - test_start >= MIN_TEST_BARS:
            idx.append((slice(0, train_end), slice(test_start, test_end)))
    return idx

def run_bt(df, StrategyCls):
    bt = Backtest(df, StrategyCls, cash=CASH, commission=COMMISSION, finalize_trades=FINALIZE_TRADES)
    st = bt.run()
    return {
        "ret": float(st.get("Return [%]", 0)),
        "sharpe": float(st.get("Sharpe Ratio", 0)),
        "trades": int(st.get("# Trades", 0)),
        "maxdd": float(st.get("Max. Drawdown [%]", 0))
    }

def zscore(a):
    a = np.asarray(a, dtype=float)
    mu = np.nanmean(a); sd = np.nanstd(a) + 1e-9
    return (a - mu) / sd

def main():
    top_path = latest_top_csv()
    top = pd.read_csv(top_path)
    with open(REG, "r", encoding="utf-8") as f:
        reg = {r["id"]: r for r in json.load(f)}

    df = load_prices()
    n = len(df)
    splits = wf_splits(n)
    print(f">> Using {top_path} | prices={n} | folds={len(splits)} (TEST_LEN={TEST_LEN}, MIN_TRAIN={MIN_TRAIN})")

    rows = []
    for i, row in enumerate(top.itertuples(index=False), 1):
        sid = getattr(row, "id")
        fam = getattr(row, "family", "trend")
        StrategyCls = pick_strategy(fam)

        fold_stats = []
        for tr, te in splits:
            test_df = df.iloc[te]
            st = run_bt(test_df, StrategyCls)
            fold_stats.append(st)

        # aggregate
        if fold_stats:
            ret = np.mean([fs["ret"] for fs in fold_stats])
            shp = np.mean([fs["sharpe"] for fs in fold_stats])
            trd = np.mean([fs["trades"] for fs in fold_stats])
            dd  = np.mean([abs(fs["maxdd"]) for fs in fold_stats])
        else:
            ret = shp = trd = dd = np.nan

        rows.append({
            "id": sid,
            "vendor": getattr(row, "vendor", ""),
            "name": getattr(row, "name", ""),
            "family": fam,
            "wf_ret_mean": ret,
            "wf_sharpe_mean": shp,
            "wf_trades_mean": trd,
            "wf_maxdd_mean": dd,
            "folds": len(fold_stats)
        })

        if i % 25 == 0:
            print(f"... WFA {i}/{len(top)}")

    out = pd.DataFrame(rows)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    wf_path = os.path.join(REPORTS, f"wf_summary_{ts}.csv")
    out.to_csv(wf_path, index=False)
    print(f"✅ WFA done: {wf_path}")

    # --- Scoring (no hard pass/fail) ---
    # Fill NaNs with medians before z-scoring
    out2 = out.copy()
    for col in ["wf_sharpe_mean", "wf_ret_mean", "wf_maxdd_mean"]:
        med = np.nanmedian(out2[col])
        out2[col] = out2[col].fillna(med)

    z_shp = zscore(out2["wf_sharpe_mean"].values)
    z_ret = zscore(out2["wf_ret_mean"].values)
    z_dd  = zscore(np.abs(out2["wf_maxdd_mean"].values))  # penalize larger DD

    score = 0.6 * z_shp + 0.4 * z_ret - 0.2 * z_dd
    out2["score"] = score
    out2["nonzero_trades"] = out2["wf_trades_mean"].fillna(0) >= 1

    # Preference to those that actually trade at least a bit, but don't hard fail
    ranked = (out2.sort_values(["nonzero_trades", "score"], ascending=[False, False])
                 .reset_index(drop=True))

    # Build shortlist
    TOP_K = 60
    shortlist = ranked.head(TOP_K)

    spath = os.path.join(REPORTS, f"shortlist_{ts}.csv")
    shortlist.to_csv(spath, index=False)
    print(f"⭐ Shortlist written: {spath} (rows={len(shortlist)})")

    # Diagnostics
    diag = {
        "wf_summary": wf_path,
        "shortlist": spath,
        "stats": {
            "ret_mean": float(np.nanmean(out["wf_ret_mean"])),
            "sharpe_mean": float(np.nanmean(out["wf_sharpe_mean"])),
            "dd_mean_abs": float(np.nanmean(np.abs(out["wf_maxdd_mean"]))),
            "trades_mean": float(np.nanmean(out["wf_trades_mean"])),
        }
    }
    dpath = os.path.join(REPORTS, f"wf_diag_{ts}.json")
    with open(dpath, "w", encoding="utf-8") as f:
        json.dump(diag, f, indent=2)
    print(f"📊 Diag written: {dpath}")

if __name__ == "__main__":
    main()
