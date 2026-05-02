import sys, json, math
import numpy as np
import pandas as pd
from pathlib import Path

# Load data as done in generalization_audit.py
sys.path.insert(0, ".")
from tools.fast_shadow import load_csv
df = load_csv("data/raw/forex_backup_2020_2025/EURUSD_H1_2024_to_2025.csv")
N = len(df)
SIM_LEN = 2160
sim_df = df.iloc[N - SIM_LEN:].copy().reset_index(drop=False)
sim_df.columns = ["orig_ts" if c == "index" else c for c in sim_df.columns]

sim_df["open"] = sim_df["open"].astype(float)
sim_df["high"] = sim_df["high"].astype(float)
sim_df["low"]  = sim_df["low"].astype(float)
sim_df["close"]= sim_df["close"].astype(float)

sim_df["atr14"] = (sim_df["high"] - sim_df["low"]).rolling(14, min_periods=1).mean()
sim_df["sma5"] = sim_df["close"].rolling(5, min_periods=1).mean()
sim_df["sma50"] = sim_df["close"].rolling(50, min_periods=1).mean()
sim_df["sma_ratio"] = sim_df["sma5"] / sim_df["sma50"]
sim_df["trend_str"] = (sim_df["sma_ratio"] - 1.0).abs()
sim_df["atr_max"] = sim_df["atr14"].rolling(100, min_periods=1).max()
sim_df["atr_min"] = sim_df["atr14"].rolling(100, min_periods=1).min()
sim_df["atr_pctile"] = ((sim_df["atr14"] - sim_df["atr_min"]) / (sim_df["atr_max"] - sim_df["atr_min"] + 1e-9))
sim_df["bar_pos"] = np.arange(len(sim_df)) / len(sim_df)
sim_df["ret_5"] = sim_df["close"].pct_change(5)

# Calculate RSI(14)
delta = sim_df["close"].diff()
gain = (delta.where(delta > 0, 0)).rolling(14, min_periods=1).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
rs = gain / (loss + 1e-9)
sim_df["rsi14"] = 100 - (100 / (1 + rs))

# Calculate ADX(14) (approximated Wilder's Smoothing if not exact, we can use simple SMA for TR/DM for speed)
high_diff = sim_df["high"].diff()
low_diff = -sim_df["low"].diff()
tr = pd.concat([sim_df["high"] - sim_df["low"],
                (sim_df["high"] - sim_df["close"].shift()).abs(),
                (sim_df["low"] - sim_df["close"].shift()).abs()], axis=1).max(axis=1)

plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0)
minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0)

tr14 = pd.Series(tr).rolling(14, min_periods=1).sum()
plus_di14 = 100 * pd.Series(plus_dm).rolling(14, min_periods=1).sum() / (tr14 + 1e-9)
minus_di14 = 100 * pd.Series(minus_dm).rolling(14, min_periods=1).sum() / (tr14 + 1e-9)

dx = 100 * (abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14 + 1e-9))
sim_df["adx14"] = dx.rolling(14, min_periods=1).mean()

records = []
for line in Path("logs/shadow/fills.jsonl").read_text(encoding="utf-8", errors="replace").splitlines():
    try:
        r = json.loads(line.strip())
        if r.get("event") == "CLOSE" and r.get("status") == "closed":
            records.append(r)
    except Exception:
        pass

pip = 0.0001
opens = sim_df["open"].values

matched_indices = []
for r in records:
    fp = r.get("fill_px", 0)
    slip = r.get("slippage_pips", 0.3)
    side = r.get("side", "buy")
    entry_est = fp - slip * pip if side == "buy" else fp + slip * pip
    idx = int(np.argmin(np.abs(opens - entry_est)))
    matched_indices.append(idx)

rows = []
for i, (r, mi) in enumerate(zip(records, matched_indices)):
    bar = sim_df.iloc[mi]
    rows.append({
        "seq": i,
        "bar_pos": float(bar["bar_pos"]),
        "pnl": r.get("pnl_usd", 0),
        "win": r.get("pnl_usd", 0) > 0,
        "atr_pips": float(bar["atr14"]) * 10000,
        "atr_pctile": float(bar["atr_pctile"]),
        "trend_str": float(bar["trend_str"]),
        "rsi14": float(bar["rsi14"]),
        "adx14": float(bar["adx14"]),
    })

tf = pd.DataFrame(rows)
tf["period"] = tf["bar_pos"].apply(
    lambda x: "D01-30" if x < 1/3 else ("D31-60" if x < 2/3 else "D61-90"))

def pf(d):
    if len(d) == 0: return float("nan")
    gp = d[d["pnl"] > 0]["pnl"].sum()
    gl = abs(d[d["pnl"] <= 0]["pnl"].sum())
    return gp / gl if gl > 0 else float("inf")

# BASELINE
base_pf = pf(tf)
base_wr = tf["win"].mean()
base_n = len(tf)
equity = tf["pnl"].cumsum()
peak = equity.cummax()
base_dd = (equity - peak).min()

# FILTER
tf_filtered = tf[tf["atr_pctile"] >= 0.30].copy()
filt_pf = pf(tf_filtered)
filt_wr = tf_filtered["win"].mean() if len(tf_filtered) else 0
filt_n = len(tf_filtered)
eq_f = tf_filtered["pnl"].cumsum()
pk_f = eq_f.cummax()
filt_dd = (eq_f - pk_f).min() if len(eq_f) else 0

print(f"=== TASK 1: VALIDATE NEW RULE ===")
print("Baseline (Only |boll_z| >= 1.3):")
print(f"  PF: {base_pf:.3f}")
print(f"  Win Rate: {base_wr:.1%}")
print(f"  Total Trades: {base_n}")
print(f"  Drawdown: {base_dd:.2f}")

print("\nFilter (|boll_z| >= 1.3 AND atr_pctile >= 0.30):")
print(f"  PF: {filt_pf:.3f}")
print(f"  Win Rate: {filt_wr:.1%}")
print(f"  Total Trades: {filt_n}")
print(f"  Drawdown: {filt_dd:.2f}")


print("\n=== TASK 2: TIME-BASED ROBUSTNESS ===")
for per in ["D01-30", "D31-60", "D61-90"]:
    p = tf_filtered[tf_filtered["period"] == per]
    p_pf = pf(p)
    p_wr = p["win"].mean() if len(p) else 0
    p_n = len(p)
    print(f"{per} -> PF: {p_pf:.3f}, Win Rate: {p_wr:.1%}, Trades: {p_n}")


print("\n=== TASK 3: FAILURE CONDITION CHECK ===")
# worst 10% trades
worst_n = max(1, int(len(tf_filtered) * 0.1))
worst_trades = tf_filtered.sort_values(by="pnl").head(worst_n)
print("Worst Trades (After Filter):")
for i, row in worst_trades.iterrows():
    print(f"  PnL: {row['pnl']:.2f} | ADX: {row['adx14']:.1f} | RSI: {row['rsi14']:.1f} | ATR(p): {row['atr_pips']:.1f} | Trend: {row['trend_str']:.5f}")


print("\n=== OVERFITTING & RECOMMENDATION METRICS ===")
trade_drop = (base_n - filt_n) / base_n
print(f"Trade count drop: {trade_drop:.1%}")
