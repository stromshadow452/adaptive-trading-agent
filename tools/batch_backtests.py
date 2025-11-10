import os, glob
import pandas as pd
from src.backtest import simple_backtest

OUT_DIR = "reports/backtests"
os.makedirs(OUT_DIR, exist_ok=True)

rows = []
for p in sorted(glob.glob("data/datasets/*_processed.csv")):
    try:
        df = pd.read_csv(p, parse_dates=["Date"]).set_index("Date").sort_index()
        m, _, _ = simple_backtest(df, {})
        m["file"] = os.path.basename(p)
        rows.append(m)
        print("OK:", os.path.basename(p), m)
    except Exception as e:
        print("ERR:", os.path.basename(p), "->", e)

if rows:
    out = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    out_path = os.path.join(OUT_DIR, "summary.csv")
    out.to_csv(out_path, index=False)
    print("✅ Saved:", out_path)
    print(out.head(10))
else:
    print("⚠️ No results produced.")
