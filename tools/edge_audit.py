"""
tools/edge_audit.py
====================
Statistical edge validation — READ ONLY.
Tasks: shuffle test, OOS split, top-trade dependency, streak analysis, execution realism.
"""
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path

FILLS_PATH = "logs/shadow/fills.jsonl"

# ─── Load closed trades ────────────────────────────────────────────────────
fills = []
with open(FILLS_PATH, "r") as f:
    for line in f:
        t = json.loads(line.strip())
        if t.get("event") == "CLOSE" and t.get("pnl_usd") is not None:
            fills.append(t)

df = pd.DataFrame(fills)
df["pnl"]  = df["pnl_usd"].astype(float)
df["r"]    = df["r_multiple"].astype(float)
df["ts"]   = pd.to_datetime(df["requested_at"])
df["side"] = df.get("side", pd.Series(["?"] * len(df)))
df = df.sort_values("ts").reset_index(drop=True)

N  = len(df)
gp = df[df["pnl"] > 0]["pnl"].sum()
gl = abs(df[df["pnl"] <= 0]["pnl"].sum())
base_pf = gp / gl if gl > 0 else float("inf")


def pf_of(series):
    g = series[series > 0].sum()
    l = abs(series[series <= 0].sum())
    return g / l if l > 0 else float("inf")


SEP = "=" * 60

# ═══════════════════════════════════════════════════════════
# TASK 1 — SHUFFLE TEST
# ═══════════════════════════════════════════════════════════
print(SEP)
print("  TASK 1 — SHUFFLE TEST (5 runs)")
print(SEP)
print(f"  Baseline PF : {base_pf:.4f}  (N={N})")

pnl_vals = df["pnl"].tolist()
shuffle_pfs = []
random.seed(42)
for i in range(5):
    shuffled = pnl_vals.copy()
    random.shuffle(shuffled)
    s = pd.Series(shuffled)
    shuffle_pfs.append(pf_of(s))
    print(f"  Shuffle {i+1}   : PF = {shuffle_pfs[-1]:.4f}")

avg_shuffle = np.mean(shuffle_pfs)
pf_drop_pct = (base_pf - avg_shuffle) / base_pf * 100
print(f"\n  Avg shuffle PF : {avg_shuffle:.4f}")
print(f"  PF drop        : {pf_drop_pct:.1f}%")
if avg_shuffle < 1.0:
    print("  ✓ SHUFFLE COLLAPSES → Edge is sequence-dependent (good sign)")
elif abs(base_pf - avg_shuffle) < 0.02:
    print("  ⚠ SHUFFLE STABLE  → Outcomes not sequence-dependent (random-like)")
else:
    print(f"  ~ PARTIAL DROP    → Some sequence dependency")

# ═══════════════════════════════════════════════════════════
# TASK 2 — OUT-OF-SAMPLE SPLIT (first 45 / last 45 days)
# ═══════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  TASK 2 — OUT-OF-SAMPLE SPLIT")
print(SEP)

date_min = df["ts"].min()
date_max = df["ts"].max()
total_span = (date_max - date_min).total_seconds()
mid_ts = date_min + pd.Timedelta(seconds=total_span / 2)

first_half = df[df["ts"] <= mid_ts]
last_half  = df[df["ts"] >  mid_ts]

def half_stats(sub, label):
    n  = len(sub)
    if n == 0:
        print(f"  {label}: 0 trades")
        return
    _pf = pf_of(sub["pnl"])
    _wr = (sub["pnl"] > 0).mean() * 100
    _ar = sub["r"].mean()
    print(f"  {label}: trades={n:3d}  PF={_pf:.3f}  WR={_wr:.1f}%  AvgR={_ar:+.3f}  PnL={sub['pnl'].sum():+.2f}")

half_stats(first_half, "First half (in-sample) ")
half_stats(last_half,  "Last half  (OOS test)  ")

pf1 = pf_of(first_half["pnl"]) if len(first_half) > 0 else 0
pf2 = pf_of(last_half["pnl"])  if len(last_half)  > 0 else 0
oos_delta = pf2 - pf1
print(f"\n  OOS PF delta : {oos_delta:+.3f}")
if pf2 >= pf1 * 0.80:
    print("  ✓ OOS STABLE  → Edge holds on unseen data")
elif pf2 >= 1.0:
    print("  ~ OOS WEAKER  → Edge degrades but still profitable OOS")
else:
    print("  ✗ OOS FAILS   → Profitability disappears on unseen data")

# ═══════════════════════════════════════════════════════════
# TASK 3 — TOP-TRADE DEPENDENCY
# ═══════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  TASK 3 — TOP-TRADE DEPENDENCY (remove top 5 winners)")
print(SEP)

top5_idx   = df["pnl"].nlargest(5).index
top5_pnl   = df.loc[top5_idx, "pnl"].sum()
top5_pct   = top5_pnl / df["pnl"].sum() * 100 if df["pnl"].sum() != 0 else 0
df_ex_top5 = df.drop(top5_idx)
pf_ex_top5 = pf_of(df_ex_top5["pnl"])

print(f"  Full PF       : {base_pf:.4f}  ({N} trades)")
print(f"  Top-5 PnL     : {top5_pnl:+.2f}  ({top5_pct:.1f}% of total)")
print(f"  PF ex-top-5   : {pf_ex_top5:.4f}  ({len(df_ex_top5)} trades)")
print(f"  PF collapse   : {(base_pf - pf_ex_top5):.4f} ({(base_pf - pf_ex_top5)/base_pf*100:.1f}%)")

if pf_ex_top5 >= 1.0:
    print("  ✓ ROBUST      → Profitable even without best trades")
elif pf_ex_top5 >= 0.90:
    print("  ~ BORDERLINE  → Marginally unprofitable ex-top-5")
else:
    print("  ✗ FRAGILE     → Profitability entirely driven by few lucky trades")

# Also test removing top 10
top10_idx   = df["pnl"].nlargest(10).index
df_ex_top10 = df.drop(top10_idx)
pf_ex_top10 = pf_of(df_ex_top10["pnl"])
print(f"  PF ex-top-10  : {pf_ex_top10:.4f}  ({len(df_ex_top10)} trades)")

# ═══════════════════════════════════════════════════════════
# TASK 4 — STREAK ANALYSIS
# ═══════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  TASK 4 — STREAK ANALYSIS")
print(SEP)

outcomes = (df["pnl"] > 0).astype(int).tolist()  # 1=win, 0=loss

max_loss_streak = 0
max_win_streak  = 0
cur_loss = 0
cur_win  = 0
loss_streak_lengths = []
win_streak_lengths  = []

for o in outcomes:
    if o == 0:
        cur_loss += 1
        cur_win   = 0
    else:
        if cur_loss > 0:
            loss_streak_lengths.append(cur_loss)
        cur_win  += 1
        cur_loss  = 0
    max_loss_streak = max(max_loss_streak, cur_loss)
    max_win_streak  = max(max_win_streak, cur_win)
if cur_loss > 0:
    loss_streak_lengths.append(cur_loss)

print(f"  Max losing streak : {max_loss_streak}")
print(f"  Max winning streak: {max_win_streak}")
print(f"  Avg loss streak   : {np.mean(loss_streak_lengths):.2f}" if loss_streak_lengths else "  No loss streaks")
print(f"  Loss streak dist  : {sorted(loss_streak_lengths, reverse=True)[:10]}")

# Clustering: are losses bunched or spread?
loss_indices = [i for i, o in enumerate(outcomes) if o == 0]
if len(loss_indices) > 1:
    gaps = np.diff(loss_indices)
    print(f"  Avg gap between losses: {np.mean(gaps):.2f} trades")
    print(f"  Losses clustered (gap<3): {(gaps < 3).sum()} occurrences")
    if np.mean(gaps) < 2.5:
        print("  ⚠ LOSSES CLUSTER  → Bad streaks likely")
    else:
        print("  ✓ LOSSES SPREAD   → No dangerous clustering")

# ═══════════════════════════════════════════════════════════
# TASK 5 — EXECUTION REALISM
# ═══════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  TASK 5 — EXECUTION REALISM")
print(SEP)

issues = []

# Check 1: Fill prices vs close price (should differ — next-bar open)
all_events = []
with open(FILLS_PATH, "r") as f:
    for line in f:
        t = json.loads(line.strip())
        all_events.append(t)

open_events  = [t for t in all_events if t.get("event") == "OPEN"]
close_events = [t for t in all_events if t.get("event") == "CLOSE"]

print(f"  Total events    : {len(all_events)} ({len(open_events)} opens, {len(close_events)} closes)")

# Check 2: SL_HIT with positive PnL (violation)
sl_positive = [t for t in close_events if t.get("close_reason") == "sl_hit" and float(t.get("pnl_usd", 0)) > 0]
tp_negative = [t for t in close_events if t.get("close_reason") == "tp_hit" and float(t.get("pnl_usd", 0)) < 0]

if sl_positive:
    issues.append(f"SL_HIT with positive PnL: {len(sl_positive)} trades")
    for t in sl_positive[:3]:
        print(f"  ⚠ SL_HIT+PnL>0: pnl={t['pnl_usd']} close_reason={t.get('close_reason')}")
else:
    print("  ✓ No SL_HIT with positive PnL")

if tp_negative:
    issues.append(f"TP_HIT with negative PnL: {len(tp_negative)} trades")
    for t in tp_negative[:3]:
        print(f"  ⚠ TP_HIT+PnL<0: pnl={t['pnl_usd']}")
else:
    print("  ✓ No TP_HIT with negative PnL")

# Check 3: Close reasons distribution
from collections import Counter
reasons = Counter(t.get("close_reason", "unknown") for t in close_events)
print(f"  Close reasons   : {dict(reasons)}")

# Check 4: R-multiple sanity — should be close to ±1.0 or TP ratio
r_vals = df["r"].values
r_wins  = r_vals[r_vals > 0]
r_losses = r_vals[r_vals <= 0]
if len(r_wins) > 0:
    print(f"  Avg R (wins)    : {r_wins.mean():+.3f}  (expect ≈ +1.5 for 1.5 RR)")
if len(r_losses) > 0:
    print(f"  Avg R (losses)  : {r_losses.mean():+.3f}  (expect ≈ -1.0)")

# Check 5: any duplicate timestamps (same-bar entry+exit)
open_ts  = {t.get("requested_at") for t in open_events}
close_ts = {t.get("requested_at") for t in close_events}
same_bar = open_ts & close_ts
if same_bar:
    issues.append(f"Intra-bar open+close: {len(same_bar)} events")
    print(f"  ⚠ Same-bar open+close timestamps: {len(same_bar)}")
else:
    print("  ✓ No same-bar open+close (no intra-bar cheating)")

# ═══════════════════════════════════════════════════════════
# FINAL VERDICT
# ═══════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  FINAL VERDICT")
print(SEP)

score = 0
max_score = 5

# Score 1: shuffle collapses
if avg_shuffle < base_pf * 0.90:
    score += 1
    print("  [+] Shuffle: edge is non-random")
else:
    print("  [-] Shuffle: outcomes appear random")

# Score 2: OOS holds
if pf2 >= 1.0:
    score += 1
    print(f"  [+] OOS: profitable on unseen data (PF={pf2:.3f})")
else:
    print(f"  [-] OOS: unprofitable on unseen data (PF={pf2:.3f})")

# Score 3: not top-trade dependent
if pf_ex_top5 >= 0.95:
    score += 1
    print(f"  [+] Dependency: robust ex-top-5 (PF={pf_ex_top5:.3f})")
else:
    print(f"  [-] Dependency: collapses ex-top-5 (PF={pf_ex_top5:.3f})")

# Score 4: no execution violations
if not issues:
    score += 1
    print("  [+] Execution: no realism violations")
else:
    print(f"  [-] Execution: {len(issues)} issues found")

# Score 5: OOS PF >= baseline * 0.8
if pf2 >= base_pf * 0.80:
    score += 1
    print(f"  [+] OOS stability: within 20% of in-sample (OOS={pf2:.3f} base={base_pf:.3f})")
else:
    print(f"  [-] OOS degrades >20% vs in-sample")

print(f"\n  Score: {score}/{max_score}")

if score >= 4:
    verdict = "REAL EDGE"
    note    = "System shows consistent, non-random profitability across tests."
elif score == 3:
    verdict = "WEAK EDGE"
    note    = "Edge exists but is marginal or dependent on conditions. Monitor closely."
else:
    verdict = "FAKE EDGE"
    note    = "Profitability is fragile, random, or execution-inflated. Do not scale."

print(f"\n  ╔{'═'*56}╗")
print(f"  ║  VERDICT: {verdict:45s}║")
print(f"  ╚{'═'*56}╝")
print(f"\n  {note}")
print(SEP)
