import sys, json, math
import numpy as np
import pandas as pd
from pathlib import Path

# Load data
sys.path.insert(0, ".")
from tools.fast_shadow import load_csv
df = load_csv("data/raw/forex_backup_2020_2025/EURUSD_H1_2024_to_2025.csv")
N = len(df)
SIM_LEN = 2160
sim_df = df.iloc[N - SIM_LEN:].copy().reset_index(drop=False)
sim_df.columns = ["orig_ts" if c == "index" else c for c in sim_df.columns]
for c in ["open", "high", "low", "close"]: sim_df[c] = sim_df[c].astype(float)

# Features
sim_df["atr14"] = (sim_df["high"] - sim_df["low"]).rolling(14, min_periods=1).mean()
sim_df["sma5"] = sim_df["close"].rolling(5, min_periods=1).mean()
sim_df["sma50"] = sim_df["close"].rolling(50, min_periods=1).mean()
sim_df["sma_ratio"] = sim_df["sma5"] / sim_df["sma50"]
sim_df["trend_str"] = (sim_df["sma_ratio"] - 1.0).abs()
sim_df["atr_max"] = sim_df["atr14"].rolling(100, min_periods=1).max()
sim_df["atr_min"] = sim_df["atr14"].rolling(100, min_periods=1).min()
sim_df["atr_pctile"] = ((sim_df["atr14"] - sim_df["atr_min"]) / (sim_df["atr_max"] - sim_df["atr_min"] + 1e-9))
sim_df["bar_pos"] = np.arange(len(sim_df)) / len(sim_df)

# Matches
records = []
for line in Path("logs/shadow/fills.jsonl").read_text(encoding="utf-8", errors="replace").splitlines():
    try:
        r = json.loads(line.strip())
        if r.get("event") == "CLOSE" and r.get("status") == "closed":
            records.append(r)
    except Exception: pass

pip = 0.0001
opens = sim_df["open"].values
rows = []
for i, r in enumerate(records):
    fp = r.get("fill_px", 0)
    slip = r.get("slippage_pips", 0.3)
    side = r.get("side", "buy")
    entry_est = fp - slip * pip if side == "buy" else fp + slip * pip
    idx = int(np.argmin(np.abs(opens - entry_est)))
    bar = sim_df.iloc[idx]
    
    # Store trade parameters for stress testing
    sl = r.get("sl") or fp
    tp = r.get("tp") or fp
    sl_d = abs(fp - sl)
    tp_d = abs(fp - tp)
    r_mult = r.get("r_multiple", 0) or 0
    
    rows.append({
        "seq": i,
        "bar_pos": float(bar["bar_pos"]),
        "pnl": float(r.get("pnl_usd", 0)),
        "win": float(r.get("pnl_usd", 0)) > 0,
        "atr_pctile": float(bar["atr_pctile"]),
        "trend_str": float(bar["trend_str"]),
        "r_mult": r_mult,
        "base_sl_d": sl_d,
        "base_tp_d": tp_d,
        "base_slip_pips": slip,
        "side": side,
        "trade_size": r.get("size", 100000)
    })

tf = pd.DataFrame(rows)
tf["period"] = tf["bar_pos"].apply(
    lambda x: "0-30" if x < 1/3 else ("30-60" if x < 2/3 else "60-90"))

# Apply Filters
tf_filtered = tf[(tf["atr_pctile"] >= 0.30) & (tf["trend_str"] < 0.003)].copy()
tf_filtered.reset_index(drop=True, inplace=True)

def calc_metrics(d):
    pf_val = d[d["pnl"] > 0]["pnl"].sum() / abs(d[d["pnl"] <= 0]["pnl"].sum()) if abs(d[d["pnl"] <= 0]["pnl"].sum())>0 else float('inf')
    eq = d["pnl"].cumsum()
    dd = (eq - eq.cummax()).min() if len(eq)>0 else 0
    return pf_val, d["win"].mean() if len(d)>0 else 0, len(d), dd

# TASK 1
pf_val, wr_val, trades, max_dd = calc_metrics(tf_filtered)
avg_r = tf_filtered["r_mult"].mean()

print("\n--- TASK 1 ---")
print(f"PF: {pf_val:.3f}")
print(f"WR: {wr_val:.1%}")
print(f"Trades: {trades}")
print(f"Max DD: {max_dd:.2f}")
print(f"Avg R: {avg_r:.3f}")

# TASK 2
print("\n--- TASK 2 ---")
for p in ["0-30", "30-60", "60-90"]:
    pf_p, wr_p, tr_p, _ = calc_metrics(tf_filtered[tf_filtered["period"] == p])
    print(f"{p}: PF={pf_p:.3f}, WR={wr_p:.1%}, Trades={tr_p}")

# TASK 3
print("\n--- TASK 3 ---")
tot_pnl = tf_filtered["pnl"].sum()
top5_pnl = tf_filtered.sort_values(by="pnl", ascending=False).head(5)["pnl"].sum()
top5_pct = top5_pnl / tot_pnl if tot_pnl > 0 else 0
print(f"Top 5 trades PnL = {top5_pnl:.2f}")
print(f"Total PnL = {tot_pnl:.2f}")
print(f"Top 5 % contribution = {top5_pct:.1%}")

# TASK 4
print("\n--- TASK 4 ---")
avg_win = tf_filtered[tf_filtered["win"]]["pnl"].mean() if trades > 0 and len(tf_filtered[tf_filtered["win"]]) else 0
avg_loss = tf_filtered[~tf_filtered["win"]]["pnl"].mean() if trades > 0 and len(tf_filtered[~tf_filtered["win"]]) else 0
print(f"Avg Win: {avg_win:.2f}")
print(f"Avg Loss: {avg_loss:.2f}")
if avg_loss != 0:
    print(f"Win/Loss Size Ratio: {abs(avg_win/avg_loss):.2f}")

# Max losing streak
streaks, cur = 0, 0
for w in tf_filtered["win"]:
    if not w:
        cur += 1
        streaks = max(streaks, cur)
    else: cur = 0
print(f"Max Losing Streak: {streaks}")

# TASK 5: Stress Test
print("\n--- TASK 5 ---")
# 1. Slip +50% -> means entry is worse by 1.5x slip. So pnl decreases by size * (0.5 * slip) * pip.
# 2. TP -10% -> winning trades make 10% less.
# 3. SL +10% -> losing trades lose 10% more.
# Let's adjust PnL based on this.

stress_pnls = []
for _, row in tf_filtered.iterrows():
    p = row["pnl"]
    win = row["win"]
    size = row["trade_size"]
    slip_penalty = size * (row["base_slip_pips"] * 0.5) * pip
    if win:
        p = (p * 0.90) - slip_penalty
    else:
        p = (p * 1.10) - slip_penalty
    stress_pnls.append(p)

tf_filtered["stress_pnl"] = stress_pnls
stress_gp = tf_filtered[tf_filtered["stress_pnl"]>0]["stress_pnl"].sum()
stress_gl = abs(tf_filtered[tf_filtered["stress_pnl"]<=0]["stress_pnl"].sum())
stress_pf = stress_gp / stress_gl if stress_gl > 0 else float('inf')
print(f"Stress PF: {stress_pf:.3f}")
