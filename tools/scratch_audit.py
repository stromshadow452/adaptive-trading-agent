import json

with open('logs/shadow/fills_multi.jsonl', 'r') as f:
    lines = [json.loads(line) for line in f if line.strip()]

closes = [l for l in lines if l['event'] == 'CLOSE']
print(f'Total Closed Trades: {len(closes)}')

task1_violations = []
task7_violations = []
total_spread_cost = 0.0
total_slip_cost = 0.0
equity = 10000.0
peak_equity = 10000.0
max_dd = 0.0
task6_optimistic = []

for t in closes:
    ideal_pnl = t['ideal_pnl_usd']
    real_pnl = t['pnl_usd']
    spread_usd = t['spread_usd']
    slip_usd = t['slippage_usd']
    
    if spread_usd < 0 or slip_usd < 0:
        task7_violations.append(f"{t['symbol']}: Negative friction (spread={spread_usd}, slip={slip_usd})")
        
    if real_pnl > ideal_pnl:
        task1_violations.append(f"{t['symbol']}: Real PnL ({real_pnl}) > Ideal PnL ({ideal_pnl})")
        
    total_spread_cost += spread_usd
    total_slip_cost += slip_usd
    
    equity += real_pnl
    if equity > peak_equity:
        peak_equity = equity
    dd = (peak_equity - equity) / peak_equity
    if dd > max_dd:
        max_dd = dd
        
    # Task 6: Same candle exit?
    if t['requested_at'] == t['filled_at']:
        # This isn't the true bar timestamp, but hold_minutes gives it away
        pass
    if t.get('hold_minutes', 0) == 0:
        if t['close_reason'] == 'tp_hit':
            task6_optimistic.append(f"{t['symbol']} hit TP in same candle (0 min hold)")

print(f'Task 1 Violations (Execution improved PnL): {len(task1_violations)}')
if task1_violations: print(task1_violations[:5])

print(f'Task 7 Violations (Negative friction): {len(task7_violations)}')
if task7_violations: print(task7_violations[:5])

print(f'Task 6 Optimistic Fills (Same candle TP): {len(task6_optimistic)}')

print(f'Total Spread Cost: ${total_spread_cost:.2f}')
print(f'Total Slippage Cost: ${total_slip_cost:.2f}')
print(f'Real Max Drawdown: {max_dd*100:.4f}%')
