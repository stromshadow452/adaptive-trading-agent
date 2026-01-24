"""
Generate sample data for JARVIS Dashboard testing
"""
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta

# Create logs directory
os.makedirs("logs", exist_ok=True)
os.makedirs("logs/summary", exist_ok=True)

# Generate sample execution log
np.random.seed(42)
n_trades = 50

timestamps = [datetime.now() - timedelta(hours=i) for i in range(n_trades)]
timestamps.reverse()

data = {
    'timestamp': [t.isoformat() for t in timestamps],
    'symbol': np.random.choice(['EURUSD', 'GBPUSD', 'USDJPY'], n_trades),
    'side': np.random.choice(['buy', 'sell'], n_trades),
    'price': np.random.uniform(1.05, 1.15, n_trades),
    'size': np.random.uniform(0.01, 0.1, n_trades),
    'pnl': np.random.normal(5, 20, n_trades),
    'decision_source': np.random.choice(['PRIMARY', 'FINRL_FALLBACK', 'HEURISTIC'], n_trades, p=[0.6, 0.3, 0.1]),
    'regime': np.random.choice(['TREND', 'RANGE', 'UNCERTAIN'], n_trades, p=[0.4, 0.4, 0.2]),
    'status': 'EXECUTED'
}

df = pd.DataFrame(data)
df.to_csv('logs/executions.csv', index=False)
print(f"✅ Created logs/executions.csv with {n_trades} trades")

# Generate session summary
returns = df['pnl'].values
equity = 10000 + np.cumsum(returns)
running_max = np.maximum.accumulate(equity)
drawdown = (equity - running_max) / running_max
max_dd = abs(drawdown.min())

winning_trades = sum(1 for r in returns if r > 0)
total_trades = len(returns)

summary = {
    'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
    'total_trades': total_trades,
    'winning_trades': winning_trades,
    'losing_trades': total_trades - winning_trades,
    'winrate': winning_trades / total_trades,
    'total_pnl': returns.sum(),
    'avg_pnl_per_trade': returns.mean(),
    'total_return': (equity[-1] - 10000) / 10000,
    'max_drawdown': max_dd,
    'sharpe_ratio': (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0,
    'sortino_ratio': 1.2,
    'profit_factor': 1.5,
    'final_equity': equity[-1],
    'brain_usage': {
        'PRIMARY': int((df['decision_source'] == 'PRIMARY').sum()),
        'FINRL_FALLBACK': int((df['decision_source'] == 'FINRL_FALLBACK').sum()),
        'HEURISTIC': int((df['decision_source'] == 'HEURISTIC').sum())
    },
    'blocks': {
        'JARVIS_GUARD': 2,
        'CIRCUIT_BREAKER': 1,
        'REGIME_CRASH': 3,
        'LOW_CONFIDENCE': 5
    },
    'regime_counts': {
        'TREND': int((df['regime'] == 'TREND').sum()),
        'RANGE': int((df['regime'] == 'RANGE').sum()),
        'UNCERTAIN': int((df['regime'] == 'UNCERTAIN').sum())
    }
}

summary_file = f"logs/summary/session_{summary['timestamp']}.json"
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"✅ Created {summary_file}")
print(f"\n📊 Summary Stats:")
print(f"  Total Trades: {total_trades}")
print(f"  Winrate: {summary['winrate']*100:.1f}%")
print(f"  Total Return: {summary['total_return']*100:.2f}%")
print(f"  Max Drawdown: {summary['max_drawdown']*100:.2f}%")
print(f"  Sharpe Ratio: {summary['sharpe_ratio']:.2f}")
print(f"\n🚀 Now refresh your Streamlit dashboard!")
