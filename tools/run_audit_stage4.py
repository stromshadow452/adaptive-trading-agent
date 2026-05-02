import sys, json, math
import numpy as np
import pandas as pd
from pathlib import Path

# Load data
sys.path.insert(0, ".")
from tools.fast_shadow import load_csv
df = load_csv("data/raw/forex_backup_2020_2025/EURUSD_H1_2024_to_2025.csv")
N = len(df)
SIM_LEN = 2160 # 90 days * 24 hours
sim_df = df.iloc[N - SIM_LEN:].copy().reset_index(drop=False)

for c in ["open", "high", "low", "close"]: sim_df[c] = sim_df[c].astype(float)

# Features
sim_df["atr14"] = (sim_df["high"] - sim_df["low"]).rolling(14, min_periods=1).mean()
sim_df["sma5"] = sim_df["close"].rolling(5, min_periods=1).mean()
sim_df["sma50"] = sim_df["close"].rolling(50, min_periods=1).mean()
sim_df["sma_ratio"] = sim_df["sma5"] / (sim_df["sma50"] + 1e-9)
sim_df["trend_str"] = (sim_df["sma_ratio"] - 1.0).abs()
sim_df["atr_max"] = sim_df["atr14"].rolling(100, min_periods=1).max()
sim_df["atr_min"] = sim_df["atr14"].rolling(100, min_periods=1).min()
sim_df["atr_pctile"] = ((sim_df["atr14"] - sim_df["atr_min"]) / (sim_df["atr_max"] - sim_df["atr_min"] + 1e-9))

sim_df["sma20"] = sim_df["close"].rolling(20, min_periods=1).mean()
sim_df["std20"] = sim_df["close"].rolling(20, min_periods=1).std()
sim_df["boll_z"] = (sim_df["close"] - sim_df["sma20"]) / (sim_df["std20"] + 1e-9)
sim_df["bar_pos"] = np.arange(len(sim_df)) / len(sim_df)
sim_df["period"] = sim_df["bar_pos"].apply(lambda x: "0-30" if x < 1/3 else ("30-60" if x < 2/3 else "60-90"))

# Simple iterrows backtester
def run_backtest(bollz_thresh, atr_thresh):
    trades = []
    in_trade = False
    entry_price = 0
    sl = 0
    tp = 0
    side = ""
    pip = 0.0001
    trade_size = 100000
    
    for idx in range(20, len(sim_df)):
        row = sim_df.iloc[idx]
        
        # Check Exits first
        if in_trade:
            hit_tp = False
            hit_sl = False
            exit_px = 0
            if side == "buy":
                if row["low"] <= sl: hit_sl = True; exit_px = sl
                if row["high"] >= tp: hit_tp = True; exit_px = tp
            else:
                if row["high"] >= sl: hit_sl = True; exit_px = sl
                if row["low"] <= tp: hit_tp = True; exit_px = tp
                
            if hit_sl or hit_tp:
                if hit_sl and hit_tp: exit_px = sl # assume stop loss hit first
                raw_pnl = (exit_px - entry_price) * trade_size if side == "buy" else (entry_price - exit_px) * trade_size
                pnl = raw_pnl - (0.6 * pip * trade_size) # 0.6 pip total friction
                
                sl_d = abs(entry_price - sl)
                raw_r = raw_pnl / (sl_d * trade_size) if sl_d > 0 else 0
                
                trades.append({
                    "pnl": pnl,
                    "win": pnl > 0,
                    "period": sim_df.iloc[entry_idx]["period"],
                    "r_mult": raw_r
                })
                in_trade = False
            continue
            
        # Check Entries
        bz = row["boll_z"]
        atr_p = row["atr_pctile"]
        ts = row["trend_str"]
        atr_val = row["atr14"]
        
        if abs(bz) >= bollz_thresh and atr_p >= atr_thresh and ts < 0.003:
            side = "buy" if bz < 0 else "sell"
            entry_price = row["close"]
            entry_idx = idx
            
            if side == "buy":
                sl = entry_price - (atr_val * 1.5)
                tp = entry_price + (atr_val * 2.0)
            else:
                sl = entry_price + (atr_val * 1.5)
                tp = entry_price - (atr_val * 2.0)
            in_trade = True

    return pd.DataFrame(trades)

def get_metrics(df_t):
    if len(df_t) == 0: return 0.0, 0.0, 0, 0.0, 0.0
    gp = df_t[df_t["win"]]["pnl"].sum()
    gl = abs(df_t[~df_t["win"]]["pnl"].sum())
    pf = gp / gl if gl > 0 else float('inf')
    wr = df_t["win"].mean()
    n = len(df_t)
    eq = df_t["pnl"].cumsum()
    dd = (eq - eq.cummax()).min() if n > 0 else 0
    avg_r = df_t["r_mult"].mean()
    return pf, wr, n, dd, avg_r

variants = [
    ("A", 1.2, 0.25),
    ("B", 1.25, 0.25),
    ("C", 1.2, 0.30)
]

print("=== TASK 1: CONTROLLED RELAXATION TEST ===")
results = {}
for name, bz_t, atr_t in variants:
    td = run_backtest(bz_t, atr_t)
    pf, wr, n, dd, avgr = get_metrics(td)
    dd_pct = (abs(dd) / 10000.0) * 100
    results[name] = td
    print(f"Variant {name}: PF={pf:.2f}, WR={wr:.1%}, Trades={n}, DD={dd_pct:.1f}%, AvgR={avgr:.2f}")

print("\n=== TASK 2: MINIMUM VIABILITY CHECK ===")
survivors = []
for name, bz_t, atr_t in variants:
    td = results[name]
    pf, wr, n, dd, avgr = get_metrics(td)
    dd_pct = (abs(dd) / 10000.0) * 100
    if n >= 50 and pf >= 1.30 and dd_pct <= 10.0:
        survivors.append(name)
        print(f"Variant {name} -> PASS")
    else:
        print(f"Variant {name} -> FAIL (Trades={n}, PF={pf:.2f}, DD={dd_pct:.1f}%)")

print("\n=== TASK 3: STABILITY CHECK (30/30/30) ===")
survivors_t3 = []
for name in survivors:
    td = results[name]
    reject = False
    print(f"Variant {name}:")
    for p in ["0-30", "30-60", "60-90"]:
        td_p = td[td["period"] == p]
        pf_p, wr_p, n_p, _, _ = get_metrics(td_p)
        print(f"  {p}: PF={pf_p:.2f}, Trades={n_p}")
        if n_p == 0 or pf_p < 1.0:
            reject = True
    if not reject: survivors_t3.append(name)

if not survivors_t3:
    print("No variants survived TASK 3")

print("\n=== TASK 4: TRADE DISTRIBUTION QUALITY ===")
survivors_t4 = []
for name in survivors_t3:
    td = results[name]
    tot_pnl = td["pnl"].sum()
    top5_pct = td.sort_values("pnl", ascending=False).head(5)["pnl"].sum() / tot_pnl if tot_pnl > 0 else 1.0
    
    streaks, cur = 0, 0
    if len(td) > 0:
        for w in td["win"]:
            if not w: cur += 1; streaks = max(streaks, cur)
            else: cur = 0
                
    avg_win = td[td["win"]]["pnl"].mean() if len(td[td["win"]]) else 0
    avg_loss = abs(td[~td["win"]]["pnl"].mean()) if len(td[~td["win"]]) else 0
    wl_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')
    
    print(f"Variant {name}: Top5={top5_pct:.1%}, MaxSeqLoss={streaks}, W/L Ratio={wl_ratio:.2f}")
    if top5_pct <= 0.70 and streaks <= 6 and wl_ratio >= 1.1:
        survivors_t4.append(name)

if not survivors_t4:
    print("No variants survived TASK 4")
