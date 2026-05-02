"""
tools/deep_audit.py
====================
Deep edge leakage audit: loss clustering, sizing bias, SL/TP geometry.
"""
import json, sys
import numpy as np
from pathlib import Path
from collections import defaultdict

FILLS = "logs/shadow/fills.jsonl"
records = []
for line in Path(FILLS).read_text(encoding="utf-8", errors="replace").splitlines():
    try:
        r = json.loads(line.strip())
        if r.get("event") == "CLOSE" and r.get("status") == "closed":
            records.append(r)
    except Exception:
        continue

if not records:
    print("No closed trades found."); sys.exit(1)

wins   = [t for t in records if (t.get("pnl_usd") or 0) > 0]
losses = [t for t in records if (t.get("pnl_usd") or 0) <= 0]

def pf(trades):
    gp = sum(t["pnl_usd"] for t in trades if t.get("pnl_usd",0) > 0)
    gl = abs(sum(t["pnl_usd"] for t in trades if t.get("pnl_usd",0) <= 0))
    return round(gp / gl, 3) if gl > 0 else float("inf")

def avg(vals): return round(float(np.mean(vals)), 4) if vals else 0.0

print(f"\n{'='*62}")
print(f"  SCOPUS Deep Edge Audit  —  {len(records)} trades  "
      f"(W={len(wins)} L={len(losses)})")
print(f"{'='*62}")

# ─────────────────────────────────────────────────────────────────
# 1. TREND vs UNCERTAIN PF — full breakdown
# ─────────────────────────────────────────────────────────────────
print("\n[ 1 ] TREND vs UNCERTAIN DEEP BREAKDOWN")
by_regime = defaultdict(list)
for t in records:
    by_regime[t.get("regime","?")].append(t)

for regime, trades in sorted(by_regime.items(), key=lambda x: -len(x[1])):
    w = [t for t in trades if t.get("pnl_usd",0) > 0]
    l = [t for t in trades if t.get("pnl_usd",0) <= 0]
    r_w = avg([t.get("r_multiple",0) for t in w])
    r_l = avg([abs(t.get("r_multiple",1)) for t in l])
    pnl = sum(t.get("pnl_usd",0) for t in trades)
    sizes_w = [t.get("size",0) for t in w]
    sizes_l = [t.get("size",0) for t in l]
    print(f"\n  {regime}  n={len(trades)}  PF={pf(trades):.3f}  PnL=${pnl:+.2f}")
    print(f"    WR={len(w)/len(trades):.0%}   avg_win_R={r_w:+.2f}   avg_loss_R=-{r_l:.2f}")
    print(f"    avg lot WINS={avg(sizes_w):.3f}   avg lot LOSSES={avg(sizes_l):.3f}", end="")
    if avg(sizes_l) > avg(sizes_w) * 1.15:
        print("  ⚠️  BIGGER SIZE ON LOSERS")
    elif avg(sizes_w) > avg(sizes_l) * 1.15:
        print("  ✅ bigger size on winners")
    else:
        print("  (sizing flat)")

# ─────────────────────────────────────────────────────────────────
# 2. LOSS CLUSTERING — consecutive loss runs & signal conditions
# ─────────────────────────────────────────────────────────────────
print("\n\n[ 2 ] LOSS CLUSTERING ANALYSIS")

# Sequential run detection
outcomes = [1 if t.get("pnl_usd",0) > 0 else -1 for t in records]
max_run = 0; cur_run = 0; run_sizes = []
for o in outcomes:
    if o == -1:
        cur_run += 1
        max_run = max(max_run, cur_run)
    else:
        if cur_run >= 2:
            run_sizes.append(cur_run)
        cur_run = 0
if cur_run >= 2:
    run_sizes.append(cur_run)

print(f"  Max consecutive losses: {max_run}")
print(f"  Loss-run lengths (≥2): {sorted(run_sizes, reverse=True)[:10]}")
n_isolated = sum(1 for r in run_sizes if r == 1) + (len(losses) - sum(run_sizes))
print(f"  Avg loss-run length:    {avg(run_sizes):.1f}" if run_sizes else "  No runs ≥2")

# Loss by side
loss_buy  = [t for t in losses if t.get("side") == "buy"]
loss_sell = [t for t in losses if t.get("side") == "sell"]
win_buy   = [t for t in wins   if t.get("side") == "buy"]
win_sell  = [t for t in wins   if t.get("side") == "sell"]
print(f"\n  Side breakdown:")
print(f"    BUY:   {len(win_buy)} wins  {len(loss_buy)} losses  "
      f"WR={len(win_buy)/max(len(win_buy)+len(loss_buy),1):.0%}  "
      f"PF={pf(win_buy+loss_buy):.3f}")
print(f"    SELL:  {len(win_sell)} wins  {len(loss_sell)} losses  "
      f"WR={len(win_sell)/max(len(win_sell)+len(loss_sell),1):.0%}  "
      f"PF={pf(win_sell+loss_sell):.3f}")

# Loss size vs win size overall
loss_sizes = [t.get("size",0) for t in losses]
win_sizes  = [t.get("size",0) for t in wins]
size_bias  = avg(loss_sizes) / max(avg(win_sizes), 1e-6) - 1.0
print(f"\n  Sizing bias (loss lot / win lot - 1):  {size_bias:+.1%}")
if size_bias > 0.15:
    print(f"  ⚠️  SIZING HURTS: avg loss lot={avg(loss_sizes):.3f} > avg win lot={avg(win_sizes):.3f}")
elif size_bias < -0.10:
    print(f"  ✅ Sizing helps: bigger lots on winners")
else:
    print(f"  Sizing neutral (avg loss={avg(loss_sizes):.3f}  avg win={avg(win_sizes):.3f})")

# ─────────────────────────────────────────────────────────────────
# 3. SL / TP GEOMETRY ANALYSIS
# ─────────────────────────────────────────────────────────────────
print("\n\n[ 3 ] SL / TP GEOMETRY ANALYSIS")

sl_dists, tp_dists, actual_exits = [], [], []
for t in records:
    fp   = t.get("fill_px", 0)
    sl   = t.get("sl")
    tp   = t.get("tp")
    cp   = t.get("close_px", 0)
    side = t.get("side","buy")
    if fp and sl and tp:
        sl_d = abs(fp - sl)
        tp_d = abs(fp - tp)
        sl_dists.append(sl_d)
        tp_dists.append(tp_d)
        # Actual exit distance from entry
        if side == "buy":
            actual_exits.append(cp - fp)
        else:
            actual_exits.append(fp - cp)

if sl_dists:
    rr_ratio = avg(tp_dists) / avg(sl_dists)
    print(f"  Avg SL distance:  {avg(sl_dists)*10000:.1f} pips")
    print(f"  Avg TP distance:  {avg(tp_dists)*10000:.1f} pips")
    print(f"  Configured R:R:   1:{rr_ratio:.2f}")

if actual_exits:
    act_wins  = [e for e in actual_exits if e > 0]
    act_loss  = [e for e in actual_exits if e < 0]
    print(f"\n  Actual exit (wins):   +{avg(act_wins)*10000:.1f} pips  "
          f"(TP was {avg(tp_dists)*10000:.1f} pips → reaching {avg(act_wins)/avg(tp_dists)*100:.0f}% of TP)")
    print(f"  Actual exit (losses): -{abs(avg(act_loss))*10000:.1f} pips  "
          f"(SL was {avg(sl_dists)*10000:.1f} pips → full stop)")

    # MFE proxy: winners that stopped early vs full TP
    tp_hit_distances  = [e for t,e in zip(records, actual_exits)
                         if t.get("close_reason")=="tp_hit" and e>0]
    sl_hit_distances  = [e for t,e in zip(records, actual_exits)
                         if t.get("close_reason")=="sl_hit" and e<0]
    print(f"\n  Avg TP hit distance:  {avg(tp_hit_distances)*10000:.1f} pips")
    print(f"  Avg SL hit distance:  {abs(avg(sl_hit_distances))*10000:.1f} pips")

    # Are we stopping at exactly -SL (no slippage) or more?
    if abs(avg(sl_hit_distances)) > avg(sl_dists) * 1.05:
        print(f"  ⚠️  SL SLIPPAGE: actual stop loss worse than set SL by "
              f"{(abs(avg(sl_hit_distances))/avg(sl_dists)-1)*100:.1f}%")

# ─────────────────────────────────────────────────────────────────
# 4. ADAPTIVE SIZING vs DRAWDOWN
# ─────────────────────────────────────────────────────────────────
print("\n\n[ 4 ] ADAPTIVE SIZING — DOES IT HURT?")

# Check if sizing correlates with trade outcome
sizes = [t.get("size", 0) for t in records]
pnls  = [t.get("pnl_usd", 0) for t in records]
r_mults = [t.get("r_multiple", 0) or 0 for t in records]

if len(sizes) > 5:
    corr = np.corrcoef(sizes, pnls)[0,1]
    print(f"  Correlation(lot_size, pnl_usd):  {corr:+.3f}", end="")
    if corr < -0.15:
        print("  ⚠️  NEGATIVE — bigger lots → worse PnL (sizing hurts)")
    elif corr > 0.15:
        print("  ✅ POSITIVE — bigger lots → better PnL (sizing helps)")
    else:
        print("  NEUTRAL")

    corr2 = np.corrcoef(sizes, r_mults)[0,1]
    print(f"  Correlation(lot_size, R_multiple): {corr2:+.3f}", end="")
    if corr2 < -0.15:
        print("  ⚠️  Bigger lots enter at lower-confidence moments")
    elif corr2 > 0.15:
        print("  ✅ Bigger lots on high-R trades")
    else:
        print("  NEUTRAL")

# Quartile analysis: are big lots losing?
sorted_by_size = sorted(records, key=lambda x: x.get("size",0))
q1 = sorted_by_size[:len(records)//4]    # smallest lots
q4 = sorted_by_size[3*len(records)//4:]  # biggest lots
print(f"\n  Bottom 25% lot sizes → PF={pf(q1):.3f}  WR={sum(1 for t in q1 if t.get('pnl_usd',0)>0)/len(q1):.0%}")
print(f"  Top 25% lot sizes    → PF={pf(q4):.3f}  WR={sum(1 for t in q4 if t.get('pnl_usd',0)>0)/len(q4):.0%}")

if pf(q4) < pf(q1) * 0.85:
    print(f"  ⚠️  SIZING DESTROYS EDGE: large lots underperform small lots by "
          f"{(1-pf(q4)/pf(q1))*100:.0f}%")
elif pf(q4) > pf(q1) * 1.15:
    print(f"  ✅ Adaptive sizing ADDS value: large lots outperform small lots")
else:
    print(f"  Sizing effect: NEUTRAL (within 15% band)")

# ─────────────────────────────────────────────────────────────────
# VERDICT
# ─────────────────────────────────────────────────────────────────
print(f"\n\n{'='*62}")
print(f"  NEXT BIGGEST EDGE LEAK & ONE FIX")
print(f"{'='*62}")

# Determine primary leak
trend_pf = pf(by_regime.get("TREND",[]))
uncert_pf = pf(by_regime.get("UNCERTAIN",[]))
buy_pf   = pf(win_buy + loss_buy)
sell_pf  = pf(win_sell + loss_sell)

leaks = []

if abs(buy_pf - sell_pf) > 0.25:
    worse_side = "BUY" if buy_pf < sell_pf else "SELL"
    leaks.append(("DIRECTION_BIAS", abs(buy_pf - sell_pf),
        f"{worse_side} side has PF={min(buy_pf,sell_pf):.3f} vs {max(buy_pf,sell_pf):.3f}",
        f"Filter or disable {worse_side} signals when regime is TREND (trend-following, not MR):\n"
        f"     In _collect_strategy_signals: skip {worse_side.lower()} if momentum index confirms trend direction"))

if size_bias > 0.15:
    leaks.append(("SIZING_BIAS", size_bias,
        f"Adaptive sizing puts {size_bias:.0%} MORE capital on losing trades",
        f"Cap position size at fixed 0.5% risk per trade regardless of confidence:\n"
        f"     In weapon_system.yaml: max_risk_per_trade_pct: 0.005\n"
        f"     In risk_sizer.py: clamp(size, min_lots, equity * 0.005 / sl_pips)"))

if pf(q4) < pf(q1) * 0.85:
    leaks.append(("SIZE_QUARTILE", pf(q1)/max(pf(q4),0.01),
        f"Top-quartile lots PF={pf(q4):.3f} vs bottom-quartile PF={pf(q1):.3f}",
        f"Flatten lot sizing: remove confidence-scaling, use fixed fractional sizing:\n"
        f"     confidence * base_lots → fixed base_lots\n"
        f"     Only scale on verified winning conditions (WR > 55% for that regime)"))

if len(run_sizes) > 0 and max(run_sizes) >= 4:
    leaks.append(("LOSS_CLUSTERING", max(run_sizes),
        f"Max {max(run_sizes)} consecutive losses — signals cluster in unfavorable conditions",
        f"Add a 'cooling-off' circuit: after 3 consecutive losses, pause for 5 bars:\n"
        f"     Track consecutive_losses counter in pipeline state\n"
        f"     if consecutive_losses >= 3: skip next 5 bars for this symbol"))

if not leaks:
    leaks.append(("LOW_TP_RATE", 0.5 - (len(wins)/len(records)),
        f"TP rate {len(wins)/len(records):.0%} is the primary remaining drag",
        f"Raise entry bar: require BOTH RSI turning AND boll_z confirmation simultaneously:\n"
        f"     if rsi < 40 and rsi_turning_up AND boll_z < -0.5: buy  (dual filter)\n"
        f"     This raises WR at cost of fewer trades"))

# Sort by severity
leaks.sort(key=lambda x: -x[1])
top = leaks[0]

print(f"\n  ❌ LEAK: {top[0]}")
print(f"     EVIDENCE: {top[2]}")
print(f"     ONE FIX:  {top[3]}")
print()
