# tools/pre_screen_full.py
"""
Run REAL strategies that subclass backtesting.Strategy from your vendor tree,
instead of proxy templates.

Usage examples (PowerShell / cmd):
    python -u tools\pre_screen_full.py --vendor backtesting_py
    python -u tools\pre_screen_full.py --vendor backtesting_py --limit 200
    python -u tools\pre_screen_full.py --vendor backtesting_py --glob "strategy_bank/external_raw/backtesting_py/**/*.py"

Outputs:
    reports/prescreen_full_<vendor>_<timestamp>.csv   # per-class backtest stats
    reports/prescreen_full_errors_<vendor>_<timestamp>.csv  # load/run errors (if any)
"""

import os, sys, glob, re, json, hashlib, argparse, traceback
import numpy as np
import pandas as pd
from datetime import datetime
from importlib.util import spec_from_file_location, module_from_spec

# ---------- Repo layout ----------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPORTS = os.path.join(ROOT, "reports"); os.makedirs(REPORTS, exist_ok=True)

# ---------- Data: robust EURUSD Daily loader (your MT4/MT5 files) ----------
CUSTOM_GLOB = r"E:\adaptive-trading-agent (2) (1)\adaptive-trading-agent (2)\data\raw\forex\forex\EURUSD_Daily*.csv"

def _normalize_cols(cols):
    norm = {}
    for c in cols:
        k = c.lower().strip()
        k = re.sub(r"[<>]", "", k)
        k = re.sub(r"\s+", "", k)
        norm[k] = c
    return norm

def _pick(norm, *names):
    for n in names:
        key = re.sub(r"[<>\s]+", "", n.lower())
        if key in norm:
            return norm[key]
    return None

def _read_ohlc_any(path):
    df = None
    # try comma, semicolon, whitespace
    for kws in ({"sep": ","}, {"sep": ";"}, {"sep": r"\s+"}):
        try:
            df = pd.read_csv(path, **kws, engine="python")
            if df is not None and df.shape[1] >= 5:
                break
        except Exception:
            df = None
    if df is None:
        raise ValueError(f"Cannot read CSV: {os.path.basename(path)}")
    norm = _normalize_cols(df.columns)
    t_date = _pick(norm, "DATE"); t_time = _pick(norm, "TIME"); t_stamp = _pick(norm, "DATETIME","TIMESTAMP")
    o = _pick(norm, "OPEN","PRICEOPEN","O")
    h = _pick(norm, "HIGH","PRICEHIGH","H")
    l = _pick(norm, "LOW","PRICELOW","L")
    c = _pick(norm, "CLOSE","PRICECLOSE","ADJCLOSE","C","LAST")
    v = _pick(norm, "VOLUME","VOL","TICKVOL","TICKVOLUME","V")

    if t_stamp:
        dt = pd.to_datetime(df[t_stamp], errors="coerce")
    elif t_date and t_time:
        dt = pd.to_datetime(df[t_date].astype(str) + " " + df[t_time].astype(str), errors="coerce")
    elif t_date:
        dt = pd.to_datetime(df[t_date], errors="coerce")
    else:
        raise AssertionError(f"Time/Date columns not found in {os.path.basename(path)}")

    need = [o,h,l,c]
    if any(x is None for x in need):
        raise AssertionError(f"OHLC columns not found in {os.path.basename(path)} | cols={list(df.columns)}")

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

# ---------- Dynamic import helpers ----------
def _safe_mod_name(path):
    """Make a unique, importable module name from a file path."""
    base = re.sub(r'[^a-zA-Z0-9_]', '_', os.path.relpath(path, ROOT))
    h = hashlib.sha1(path.encode()).hexdigest()[:8]
    return f"vendor_{base}_{h}"

def import_module_from_path(py_path):
    """Dynamically import a module from file path, without polluting sys.modules names."""
    name = _safe_mod_name(py_path)
    spec = spec_from_file_location(name, py_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"spec loader failed for {py_path}")
    mod = module_from_spec(spec)
    # Temporarily extend sys.path so relative imports inside vendor files have a chance
    vendor_dir = os.path.dirname(py_path)
    sys.path.insert(0, vendor_dir)
    try:
        spec.loader.exec_module(mod)  # type: ignore
    finally:
        # Remove the path entry we added
        try:
            sys.path.remove(vendor_dir)
        except ValueError:
            pass
    return mod

def find_strategy_classes(mod):
    """
    Return list of (cls_name, cls) for classes that look like backtesting.Strategy subclasses.
    We detect by duck-typing to avoid import-time circulars:
      - class has attribute 'next' (callable)
      - and has 'init' or uses indicators; if base class available, we also check subclassing.
    """
    from types import ModuleType
    from inspect import isclass, isfunction

    # optional: if backtesting.Strategy is imported, check subclass
    try:
        from backtesting import Strategy as BTStrategy
    except Exception:
        BTStrategy = None

    out = []
    for attr_name in dir(mod):
        obj = getattr(mod, attr_name)
        if not isclass(obj):
            continue
        # Heuristics
        has_next = callable(getattr(obj, "next", None))
        has_init = callable(getattr(obj, "init", None))
        ok = has_next and (has_init or True)
        if BTStrategy is not None:
            try:
                if issubclass(obj, BTStrategy) and obj is not BTStrategy:
                    ok = True
            except Exception:
                pass
        if ok:
            out.append((attr_name, obj))
    return out

# ---------- Backtesting runner ----------
from backtesting import Backtest

def run_one_strategy(prices: pd.DataFrame, StrategyCls, cash=10_000, commission=0.0005):
    bt = Backtest(prices, StrategyCls, cash=cash, commission=commission, finalize_trades=True)
    st = bt.run()
    # Extract common stats names; fill if missing
    def g(k, default=np.nan):
        try:
            return float(st.get(k, default))
        except Exception:
            return default
    def gi(k, default=0):
        try:
            return int(st.get(k, default))
        except Exception:
            return default
    return {
        "Return[%]": g("Return [%]"),
        "Sharpe": g("Sharpe Ratio"),
        "Trades": gi("# Trades"),
        "WinRate[%]": g("Win Rate [%]"),
        "MaxDD[%]": g("Max. Drawdown [%]"),
        "SQN": g("SQN"),
        "Expectancy": g("Expectancy [%]"),
        "AvgTrade[%]": g("Avg. Trade [%]"),
    }

# ---------- Main pipeline ----------
def main():
    ap = argparse.ArgumentParser(description="Run REAL vendor strategies (Backtesting.py) for prescreen")
    ap.add_argument("--vendor", default="backtesting_py", help="Vendor folder name under strategy_bank/external_raw")
    ap.add_argument("--glob", default="", help="Override glob for python files")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of files (0 = no limit)")
    ap.add_argument("--max-classes-per-file", type=int, default=5, help="Safety cap per file")
    ap.add_argument("--cash", type=float, default=10_000)
    ap.add_argument("--commission", type=float, default=0.0005)
    args = ap.parse_args()

    vendor = args.vendor
    cash = args.cash
    commission = args.commission

    # Source globs
    if args.glob:
        patt = os.path.join(ROOT, args.glob)
    else:
        patt = os.path.join(ROOT, "strategy_bank", "external_raw", vendor, "**", "*.py")

    files = sorted(glob.glob(patt, recursive=True))
    if args.limit and args.limit > 0:
        files = files[:args.limit]

    if not files:
        print(f"[ERROR] No .py files found for vendor={vendor} pattern={patt}")
        sys.exit(2)

    prices = load_prices()
    print(f">> Running vendor={vendor} | files={len(files)} | rows={len(prices)}")

    rows = []
    errs = []

    for idx, fpath in enumerate(files, 1):
        rel = os.path.relpath(fpath, ROOT).replace("\\", "/")
        try:
            mod = import_module_from_path(fpath)
        except Exception as e:
            errs.append({"file": rel, "error": f"IMPORT: {e}", "trace": traceback.format_exc()})
            if idx % 20 == 0:
                print(f"... {idx}/{len(files)} files (import errors so far: {len(errs)})")
            continue

        try:
            cls_list = find_strategy_classes(mod)
        except Exception as e:
            errs.append({"file": rel, "error": f"SCAN: {e}", "trace": traceback.format_exc()})
            continue

        if not cls_list:
            # not a strategy file; skip quietly
            continue

        # safety cap per file
        cls_list = cls_list[: args.max_classes_per_file]

        for cls_name, StrategyCls in cls_list:
            sid_base = f"{vendor}|{rel}|{cls_name}"
            sid = hashlib.sha1(sid_base.encode()).hexdigest()[:12]
            sid_full = f"{vendor}_{sid}"

            try:
                stats = run_one_strategy(prices, StrategyCls, cash=cash, commission=commission)
                rows.append({
                    "id": sid_full,
                    "vendor": vendor,
                    "file": rel,
                    "class": cls_name,
                    "return_pct": stats["Return[%]"],
                    "sharpe": stats["Sharpe"],
                    "trades": stats["Trades"],
                    "winrate_pct": stats["WinRate[%]"],
                    "maxdd_pct": stats["MaxDD[%]"],
                    "sqn": stats["SQN"],
                    "expectancy_pct": stats["Expectancy"],
                    "avg_trade_pct": stats["AvgTrade[%]"],
                })
            except Exception as e:
                errs.append({
                    "file": rel,
                    "class": cls_name,
                    "error": f"RUN: {e}",
                    "trace": traceback.format_exc()
                })

        if idx % 20 == 0:
            print(f"... processed {idx}/{len(files)} files | results={len(rows)} | errors={len(errs)}")

    # Write reports
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out_csv = os.path.join(REPORTS, f"prescreen_full_{vendor}_{ts}.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"✅ Wrote: {out_csv} (rows={len(rows)})")

    if errs:
        err_csv = os.path.join(REPORTS, f"prescreen_full_errors_{vendor}_{ts}.csv")
        pd.DataFrame(errs).to_csv(err_csv, index=False)
        print(f"⚠️  Errors logged: {err_csv} (rows={len(errs)})")

if __name__ == "__main__":
    # imports needed by loader
    import pandas as pd
    main()
