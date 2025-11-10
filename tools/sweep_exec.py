import os, itertools, pandas as pd
from tools.paper_trader import load_df, run_paper

OUT_DIR = "reports/backtests"
os.makedirs(OUT_DIR, exist_ok=True)

def main():
    symbol = "USDJPY"
    tf = "H4"
    df, _ = load_df(symbol, tf, None)

    atr_pos_k_list = [0.0, 0.2, 0.3, 0.5]
    sl_atr_list = [0.0, 1.0, 1.5, 2.0]
    tp_atr_list = [0.0, 1.5, 2.0, 3.0]
    max_hold_list = [0, 50, 100, 150]

    rows = []
    for atr_pos_k, sl_atr, tp_atr, max_hold in itertools.product(
        atr_pos_k_list, sl_atr_list, tp_atr_list, max_hold_list
    ):
        try:
            _, m = run_paper(df, init_eq=15000, atr_pos_k=atr_pos_k,
                             sl_atr=sl_atr, tp_atr=tp_atr, max_hold=max_hold)
            rows.append({
                "atr_pos_k": atr_pos_k, "sl_atr": sl_atr, "tp_atr": tp_atr,
                "max_hold": max_hold, "sharpe": m["sharpe"], "total_return": m["total_return"]
            })
        except Exception as e:
            rows.append({
                "atr_pos_k": atr_pos_k, "sl_atr": sl_atr, "tp_atr": tp_atr,
                "max_hold": max_hold, "sharpe": -999, "total_return": -999, "err": str(e)
            })

    out = pd.DataFrame(rows).sort_values(["sharpe", "total_return"], ascending=[False, False])
    path = os.path.join(OUT_DIR, f"sweep_exec_{symbol}_{tf}.csv")
    out.to_csv(path, index=False)
    print("✅ Saved sweep:", path)
    print(out.head(20))

if __name__ == "__main__":
    main()
