"""
SCOPUS Post-Regime-Fix Validation Audit
Tasks 1-7: Core metrics, regime breakdown, before/after, stability, risk, verdict.
READ-ONLY — no code changes.
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

# ─── Load fills ───
fills = []
with open('logs/shadow/fills.jsonl', 'r') as f:
    for line in f:
        t = json.loads(line.strip())
        if t.get('event') == 'CLOSE' and t.get('pnl_usd') is not None:
            fills.append(t)

df = pd.DataFrame(fills)
df['pnl'] = df['pnl_usd'].astype(float)
df['r'] = df['r_multiple'].astype(float)
df['ts'] = pd.to_datetime(df['requested_at'])
df = df.sort_values('ts').reset_index(drop=True)

# Reconstruct regime using same proxy as regime_analysis.py
all_events = []
with open('logs/shadow/fills.jsonl', 'r') as f:
    for line in f:
        t = json.loads(line.strip())
        all_events.append(t)

price_series = []
for t in sorted(all_events, key=lambda x: x.get('requested_at', '')):
    px = float(t.get('fill_px', t.get('requested_px', 0)))
    ts = pd.Timestamp(t.get('requested_at', t.get('ts', '')))
    if px > 0:
        price_series.append({'timestamp': ts, 'price': px})

pdf = pd.DataFrame(price_series).sort_values('timestamp').reset_index(drop=True)
pdf['sma_20'] = pdf['price'].rolling(20, min_periods=5).mean()
pdf['sma_ratio'] = pdf['price'] / pdf['sma_20']
pdf['pseudo_adx'] = (abs(pdf['sma_ratio'] - 1.0) / 0.01 * 25).clip(0, 50)

regimes = []
for _, row in df.iterrows():
    idx = (pdf['timestamp'] - row['ts']).abs().idxmin()
    pr = pdf.iloc[idx]
    adx = float(pr['pseudo_adx']) if pd.notna(pr['pseudo_adx']) else None
    sr = float(pr['sma_ratio']) if pd.notna(pr['sma_ratio']) else None
    if adx is not None and adx > 28 and sr is not None and abs(sr - 1.0) > 0.004:
        regimes.append('TREND')
    elif adx is None or adx < 20:
        regimes.append('RANGE')
    else:
        regimes.append('TRANSITION')

df['regime'] = regimes

N = len(df)
wins = df[df['pnl'] > 0]
losses = df[df['pnl'] <= 0]
gp = wins['pnl'].sum()
gl = abs(losses['pnl'].sum())

# ═══════════════════════════════════════════════════
# TASK 1: CORE METRICS
# ═══════════════════════════════════════════════════
pf = gp / gl if gl > 0 else float('inf')
wr = len(wins) / N * 100 if N > 0 else 0
net_pnl = df['pnl'].sum()
avg_r = df['r'].mean()

# Max drawdown (equity curve)
equity = df['pnl'].cumsum()
peak = equity.cummax()
dd = equity - peak
max_dd = dd.min()
max_dd_pct = (max_dd / (peak.max() if peak.max() > 0 else 1)) * 100

print("=" * 60)
print("  TASK 1 — CORE METRICS")
print("=" * 60)
print(f"  Total Trades  : {N}")
print(f"  Profit Factor : {pf:.2f}")
print(f"  Win Rate      : {wr:.1f}%")
print(f"  Net PnL       : {net_pnl:+.2f}")
print(f"  Max Drawdown  : {max_dd:.2f} ({max_dd_pct:.1f}%)")
print(f"  Avg R/trade   : {avg_r:+.3f}")
print(f"  Gross Profit  : {gp:.2f}")
print(f"  Gross Loss    : {gl:.2f}")

# ═══════════════════════════════════════════════════
# TASK 2: REGIME BREAKDOWN
# ═══════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  TASK 2 — REGIME BREAKDOWN")
print("=" * 60)

for r in ['RANGE', 'TREND', 'TRANSITION']:
    sub = df[df['regime'] == r]
    n = len(sub)
    if n == 0:
        print(f"\n  {r}: 0 trades")
        continue
    w = sub[sub['pnl'] > 0]
    l = sub[sub['pnl'] <= 0]
    _gp = w['pnl'].sum()
    _gl = abs(l['pnl'].sum())
    _pf = _gp / _gl if _gl > 0 else float('inf')
    _wr = len(w) / n * 100
    _ar = sub['r'].mean()
    print(f"\n  {r}:")
    print(f"    Trades   : {n}")
    print(f"    PnL      : {sub['pnl'].sum():+.2f}")
    print(f"    PF       : {_pf:.2f}")
    print(f"    WinRate  : {_wr:.1f}%")
    print(f"    Avg R    : {_ar:+.3f}")

# ═══════════════════════════════════════════════════
# TASK 3: BEFORE vs AFTER
# ═══════════════════════════════════════════════════
OLD_PF = 1.24
OLD_TRADES = 111
OLD_PNL = 1408.08
OLD_TRANSITION_LOSS = -710.55

print("\n" + "=" * 60)
print("  TASK 3 — BEFORE vs AFTER COMPARISON")
print("=" * 60)
print(f"                  {'OLD':>10s}  {'NEW':>10s}  {'DELTA':>10s}")
print(f"  Trades        : {OLD_TRADES:10d}  {N:10d}  {N - OLD_TRADES:+10d}")
print(f"  PnL           : {OLD_PNL:+10.2f}  {net_pnl:+10.2f}  {net_pnl - OLD_PNL:+10.2f}")
print(f"  PF            : {OLD_PF:10.2f}  {pf:10.2f}  {pf - OLD_PF:+10.2f}")
transition_now = df[df['regime'] == 'TRANSITION']['pnl'].sum() if len(df[df['regime'] == 'TRANSITION']) > 0 else 0
print(f"  Trans. Loss   : {OLD_TRANSITION_LOSS:+10.2f}  {transition_now:+10.2f}  {transition_now - OLD_TRANSITION_LOSS:+10.2f}")
trade_reduction = (1 - N / OLD_TRADES) * 100 if OLD_TRADES > 0 else 0
print(f"  Trade Change  : {trade_reduction:+.1f}%")

# ═══════════════════════════════════════════════════
# TASK 4: 30/30/30 STABILITY
# ═══════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  TASK 4 — 30/30/30 STABILITY CHECK")
print("=" * 60)

if N > 0:
    date_min = df['ts'].min()
    date_max = df['ts'].max()
    total_span = (date_max - date_min).days
    d1 = date_min + pd.Timedelta(days=total_span / 3)
    d2 = date_min + pd.Timedelta(days=2 * total_span / 3)

    periods = [
        ('First third', df[df['ts'] <= d1]),
        ('Middle third', df[(df['ts'] > d1) & (df['ts'] <= d2)]),
        ('Last third', df[df['ts'] > d2]),
    ]

    print(f"  Date range: {date_min.strftime('%Y-%m-%d')} → {date_max.strftime('%Y-%m-%d')} ({total_span} days)")
    print(f"  Splits at: {d1.strftime('%Y-%m-%d')}, {d2.strftime('%Y-%m-%d')}")
    print(f"\n  {'Period':20s} {'Trades':>7s} {'PF':>7s} {'WR':>7s} {'PnL':>10s}")
    print("  " + "-" * 55)
    for label, sub in periods:
        n = len(sub)
        if n == 0:
            print(f"  {label:20s} {0:>7d} {'N/A':>7s} {'N/A':>7s} {'0.00':>10s}")
            continue
        w = sub[sub['pnl'] > 0]
        l = sub[sub['pnl'] <= 0]
        _gp = w['pnl'].sum()
        _gl = abs(l['pnl'].sum())
        _pf = _gp / _gl if _gl > 0 else float('inf')
        _wr = len(w) / n * 100
        print(f"  {label:20s} {n:>7d} {_pf:>7.2f} {_wr:>6.1f}% {sub['pnl'].sum():>+10.2f}")

# ═══════════════════════════════════════════════════
# TASK 5: RISK CONCENTRATION
# ═══════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  TASK 5 — RISK CONCENTRATION")
print("=" * 60)

# Top 5 trades as % of total PnL
top5 = df.nlargest(5, 'pnl')
top5_pnl = top5['pnl'].sum()
top5_pct = (top5_pnl / net_pnl * 100) if net_pnl != 0 else 0
print(f"  Top 5 trades PnL  : {top5_pnl:+.2f} ({top5_pct:.1f}% of total)")

# Max losing streak
streak = 0
max_streak = 0
for p in df['pnl']:
    if p <= 0:
        streak += 1
        max_streak = max(max_streak, streak)
    else:
        streak = 0
print(f"  Max losing streak : {max_streak}")

# Largest loss vs avg loss
if len(losses) > 0:
    largest_loss = losses['pnl'].min()
    avg_loss = losses['pnl'].mean()
    print(f"  Largest loss      : {largest_loss:.2f}")
    print(f"  Average loss      : {avg_loss:.2f}")
    print(f"  Ratio (lg/avg)    : {largest_loss / avg_loss:.2f}x")

# Max winning streak
streak = 0
max_win_streak = 0
for p in df['pnl']:
    if p > 0:
        streak += 1
        max_win_streak = max(max_win_streak, streak)
    else:
        streak = 0
print(f"  Max winning streak: {max_win_streak}")

# ═══════════════════════════════════════════════════
# TASK 6: FAILURE DETECTION
# ═══════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  TASK 6 — FAILURE DETECTION")
print("=" * 60)

flags = []
trend_n = len(df[df['regime'] == 'TREND'])
range_sub = df[df['regime'] == 'RANGE']
range_w = range_sub[range_sub['pnl'] > 0]
range_l = range_sub[range_sub['pnl'] <= 0]
range_pf = range_w['pnl'].sum() / abs(range_l['pnl'].sum()) if abs(range_l['pnl'].sum()) > 0 else float('inf')

if trend_n < 10:
    flags.append(f"  ⚠ TREND trades = {trend_n} (< 10) → weak trend signal")
else:
    print(f"  ✓ TREND trades = {trend_n} (≥ 10)")

if range_pf < 1.2:
    flags.append(f"  ⚠ RANGE PF = {range_pf:.2f} (< 1.2) → MR edge degrading")
else:
    print(f"  ✓ RANGE PF = {range_pf:.2f} (≥ 1.2)")

for r in ['RANGE', 'TREND', 'TRANSITION']:
    sub = df[df['regime'] == r]
    if len(sub) > 0:
        w = sub[sub['pnl'] > 0]
        l = sub[sub['pnl'] <= 0]
        _pf = w['pnl'].sum() / abs(l['pnl'].sum()) if abs(l['pnl'].sum()) > 0 else float('inf')
        if _pf < 1.0:
            flags.append(f"  ⚠ {r} PF = {_pf:.2f} (< 1.0) → still leaking")
        else:
            print(f"  ✓ {r} PF = {_pf:.2f} (≥ 1.0)")

if flags:
    for f in flags:
        print(f)

# ═══════════════════════════════════════════════════
# TASK 7: VERDICT
# ═══════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  TASK 7 — FINAL VERDICT")
print("=" * 60)

# Decision logic
reject_reasons = []
borderline_reasons = []

if pf < 1.0:
    reject_reasons.append("PF < 1.0 — system unprofitable")
if N < 30:
    reject_reasons.append(f"Only {N} trades — insufficient sample")

transition_trades = len(df[df['regime'] == 'TRANSITION'])
if transition_trades > 0:
    t_pnl = df[df['regime'] == 'TRANSITION']['pnl'].sum()
    if t_pnl < -100:
        borderline_reasons.append(f"TRANSITION still active: {transition_trades} trades, {t_pnl:+.2f}")

if trend_n < 10:
    borderline_reasons.append(f"TREND sample too small ({trend_n})")
if range_pf < 1.2:
    borderline_reasons.append(f"RANGE PF degrading ({range_pf:.2f})")
if pf < 1.3:
    borderline_reasons.append(f"Overall PF marginal ({pf:.2f})")
if top5_pct > 80:
    borderline_reasons.append(f"Top-5 concentration too high ({top5_pct:.0f}%)")

if reject_reasons:
    verdict = "REJECT"
    reason_str = "; ".join(reject_reasons)
elif borderline_reasons:
    verdict = "BORDERLINE"
    reason_str = "; ".join(borderline_reasons)
else:
    verdict = "APPROVE"
    reason_str = "All checks passed"

print(f"\n  ╔{'═'*56}╗")
print(f"  ║  VERDICT: {verdict:45s}║")
print(f"  ╚{'═'*56}╝")
print(f"\n  Reason: {reason_str}")
print("=" * 60)
