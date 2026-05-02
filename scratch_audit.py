import json
import pandas as pd
import numpy as np

log_path = 'logs/shadow/fills_multi.jsonl'
closes = []
opens = {}
events = []

with open(log_path, 'r') as f:
    for line in f:
        d = json.loads(line)
        if d.get('event') == 'FILL':
            if 'ts' in d:
                opens[f"{d.get('symbol')}_{d.get('side')}"] = pd.to_datetime(d['ts'])
                events.append((pd.to_datetime(d['ts']), 1))
        elif d.get('event') == 'CLOSE':
            d['ts'] = pd.to_datetime(d['ts'])
            closes.append(d)
            events.append((d['ts'], -1))

print(f'Total closed trades loaded: {len(closes)}')

df = pd.DataFrame(closes)
if not df.empty:
    df = df.sort_values('ts')
    df['pnl_usd'] = pd.to_numeric(df['pnl_usd'])

    # Task 1: Equity Curve Validation
    df['equity'] = 10000 + df['pnl_usd'].cumsum()
    peak = df['equity'].cummax()
    dd = (peak - df['equity']) / peak
    real_max_dd = dd.max() * 100
    does_equity_go_down = (df['pnl_usd'] < 0).any()

    # Task 2: Trade Overlap
    events.sort(key=lambda x: x[0])
    current_concurrent = 0
    max_concurrent = 0
    for t, val in events:
        current_concurrent += val
        if current_concurrent > max_concurrent:
            max_concurrent = current_concurrent

    # Task 3: Execution Timing
    same_bar_cheats = (df['hold_minutes'] == 0).sum() if 'hold_minutes' in df.columns else 0
    
    # Task 5: PnL Distribution
    wins = df[df['pnl_usd'] > 0]['pnl_usd']
    losses = df[df['pnl_usd'] < 0]['pnl_usd']
    avg_win = wins.mean() if not wins.empty else 0
    avg_loss = losses.mean() if not losses.empty else 0
    skew = df['pnl_usd'].skew()
    top_5_pnl = df['pnl_usd'].nlargest(5).sum()
    total_pnl = df['pnl_usd'].sum()
    outlier_dependency = (top_5_pnl / total_pnl) * 100 if total_pnl > 0 else 0
    
    # Check PF
    gross_profit = wins.sum()
    gross_loss = abs(losses.sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    print(f'REAL max drawdown: {real_max_dd:.4f}%')
    print(f'Does equity ever go down? {does_equity_go_down}')
    print(f'Max concurrent trades: {max_concurrent}')
    print(f'Same-bar entry+exit: {same_bar_cheats}')
    print(f'Average Win: ${avg_win:.2f}')
    print(f'Average Loss: ${avg_loss:.2f}')
    print(f'PnL Skew: {skew:.2f}')
    print(f'Top 5 trades account for {outlier_dependency:.2f}% of total PnL')
    print(f'Profit Factor: {pf:.2f}')
    
    # Check for unrealistic consistency
    consecutive_wins = (df['pnl_usd'] > 0).astype(int).groupby((df['pnl_usd'] < 0).cumsum()).sum().max()
    print(f'Max consecutive wins: {consecutive_wins}')
else:
    print('No trades found.')
