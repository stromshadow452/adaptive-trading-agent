# tools/generate_dashboard_usdjpy.py
"""
Usage:
  python tools/generate_dashboard_usdjpy.py \
    --csv reports/daily_logs/USDJPY_H4_processed__prod_20251013_091505.csv \
    --out reports/figures/usdjpy_h4_dashboard.html

Requires: pandas, plotly
"""
import argparse
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def compute_drawdown(equity: pd.Series):
    h = equity.cummax()
    dd = equity / h - 1.0
    return dd

def summary_metrics(equity: pd.Series, strat_ret: pd.Series):
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    # crude ann factor for display only
    avg = strat_ret.mean()
    std = strat_ret.std(ddof=0) + 1e-12
    sharpe = (avg / std) * np.sqrt(252) if std > 0 else 0.0
    maxdd = float(compute_drawdown(equity).min())
    return {
        "total_return": total_return,
        "sharpe": float(sharpe),
        "max_drawdown": maxdd,
        "bars": int(len(strat_ret))
    }

def try_parse_csv(path: str) -> pd.DataFrame:
    # Read without forcing parse_dates — safer for varied headers
    df = pd.read_csv(path, low_memory=False)
    # normalize common date-like column names
    date_cols = [c for c in df.columns if c.lower() in ("date","datetime","time","timestamp")]
    if date_cols:
        # try: prefer Datetime/Date then others
        pref = None
        for cand in ("Datetime","Date","datetime","date"):
            if cand in df.columns:
                pref = cand; break
        if pref is None:
            pref = date_cols[0]
        # if there's also a Time column and Date separate, merge
        if "Date" in df.columns and "Time" in df.columns and "Datetime" not in df.columns:
            df["Datetime"] = df["Date"].astype(str).str.strip() + " " + df["Time"].astype(str).str.strip()
            pref = "Datetime"
        try:
            df[pref] = pd.to_datetime(df[pref], errors="coerce", utc=True)
            df = df.set_index(pref).sort_index()
            return df
        except Exception:
            pass
    # fallback: try parsing index or any column with datetime-like content
    for col in df.columns:
        try:
            parsed = pd.to_datetime(df[col], errors="coerce", utc=True)
            if parsed.notna().sum() > 0.4 * len(parsed):  # heuristic: >40% parse success
                df[col] = parsed
                df = df.set_index(col).sort_index()
                return df
        except Exception:
            continue
    # last resort: try to parse first column values as combined date/time strings
    try:
        df.index = pd.to_datetime(df.iloc[:,0], errors="coerce", utc=True)
        df = df.dropna(how="all", axis=0)
        df = df.sort_index()
        return df
    except Exception:
        raise RuntimeError("Could not parse any datetime column in CSV: " + path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to prod strategy CSV")
    ap.add_argument("--out", default="reports/figures/usdjpy_h4_dashboard.html")
    args = ap.parse_args()

    df = try_parse_csv(args.csv)

    # try common equity/strat_ret columns
    if "equity" in df.columns:
        equity = df["equity"].astype(float)
    elif "Equity" in df.columns:
        equity = df["Equity"].astype(float)
    else:
        if "strat_ret" in df.columns:
            strat = df["strat_ret"].astype(float).fillna(0.0)
        elif "ret" in df.columns:
            strat = df["ret"].astype(float).fillna(0.0)
        else:
            # try compute per-row return from price if available
            price = None
            for cand in ["price","Price","Close","close"]:
                if cand in df.columns:
                    price = df[cand].astype(float)
                    break
            if price is None:
                raise SystemExit("CSV missing 'equity' or 'strat_ret'/'ret' or price column.")
            strat = price.pct_change().fillna(0.0)
        equity = (1 + strat).cumprod() * 1.0

    price_col = None
    for cand in ["price","Price","Close","close"]:
        if cand in df.columns:
            price_col = cand; break

    signal_col = None
    for cand in ["signal","Signal","pos","position"]:
        if cand in df.columns:
            signal_col = cand; break

    strat_ret = df["strat_ret"].astype(float) if "strat_ret" in df.columns else strat

    metrics = summary_metrics(equity, strat_ret)

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.35,0.45,0.20],
                        vertical_spacing=0.02,
                        specs=[[{"secondary_y": False}],
                               [{"secondary_y": False}],
                               [{"secondary_y": False}]])

    if price_col:
        fig.add_trace(go.Scatter(x=df.index, y=df[price_col].astype(float), name="Price (Close)", line=dict(width=1)), row=1, col=1)

    if signal_col and price_col:
        sig = df[signal_col].astype(float).fillna(0.0)
        buys = sig[sig>0]
        sells = sig[sig<0]
        if not buys.empty:
            fig.add_trace(go.Scatter(x=buys.index, y=df.loc[buys.index, price_col].astype(float),
                                     mode="markers", name="Buy", marker=dict(symbol="triangle-up", size=8, color="green")), row=1, col=1)
        if not sells.empty:
            fig.add_trace(go.Scatter(x=sells.index, y=df.loc[sells.index, price_col].astype(float),
                                     mode="markers", name="Sell", marker=dict(symbol="triangle-down", size=8, color="red")), row=1, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)

    fig.add_trace(go.Scatter(x=equity.index, y=equity.values, name="Equity", line=dict(width=2)), row=2, col=1)
    fig.update_yaxes(title_text="Equity", row=2, col=1)

    dd = compute_drawdown(equity)
    fig.add_trace(go.Scatter(x=dd.index, y=dd.values, name="Drawdown", line=dict(width=1), fill='tozeroy'), row=3, col=1)
    fig.update_yaxes(title_text="Drawdown", row=3, col=1)

    ann_text = f"Total Return: {metrics['total_return']*100:.2f}%<br>Sharpe: {metrics['sharpe']:.3f}<br>Max DD: {metrics['max_drawdown']*100:.2f}%<br>Bars: {metrics['bars']}"
    fig.add_annotation(text=ann_text, xref="paper", yref="paper", x=0.99, y=0.98, showarrow=False, bordercolor="black", bgcolor="white", font=dict(size=12))

    fig.update_layout(height=900, width=1200, title_text="USDJPY H4 - Strategy Dashboard", template="plotly_white")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.write_html(args.out, include_plotlyjs='cdn')
    print("Saved dashboard:", args.out)

if __name__ == "__main__":
    main()
