# kotha file — paste new
import argparse, os
import numpy as np
import pandas as pd
from tools.paper_trader import load_df, run_paper

OUT_DIR = "reports/backtests"
os.makedirs(OUT_DIR, exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--tf", required=True)
    ap.add_argument("--fee_bps", type=float, default=1.0)
    ap.add_argument("--slip_bps", type=float, default=2.0)
    args = ap.parse_args()

    df, _ = load_df(args.symbol, args.tf, None)

    rsi_bands = [None, (45,55), (40,60), (35,65)]
    atr_sets  = [None, (0.2,0.8), (0.3,0.7), (0.4,0.8), (0.5,0.8), (0.6,0.9)]
    modes     = ["trend","quiet"]

    rows=[]
    for rb in rsi_bands:
        for ar in atr_sets:
            for md in modes:
                try:
                    _, m = run_paper(df, 10000.0, args.fee_bps, args.slip_bps,
                                     rsi_block=rb, atr_regime=ar, regime_mode=md)
                    rows.append({
                        "rsi_block": rb, "atr_regime": ar, "mode": md,
                        "sharpe": m["sharpe"], "total_return": m["total_return"], "bars": m["bars"]
                    })
                except Exception as e:
                    rows.append({"rsi_block": rb, "atr_regime": ar, "mode": md, "sharpe": -999, "total_return": -999, "bars": 0, "err": str(e)})

    out = pd.DataFrame(rows).sort_values(["sharpe","total_return"], ascending=[False, False])
    path = os.path.join(OUT_DIR, f"sweep_{args.symbol}_{args.tf}.csv")
    out.to_csv(path, index=False)
    print("✅ Saved sweep:", path)
    print(out.head(15))

if __name__ == "__main__":
    main()
