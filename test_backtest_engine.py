"""
Quick test of SCOPUS backtest engine
"""
import sys
import os

# Add repo root to path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from datetime import datetime
from src.backtest.engine import run_backtest

# Test backtest
print("=" * 60)
print("SCOPUS Backtest Engine Test")
print("=" * 60)

result = run_backtest(
    symbols=['EURUSD', 'GBPUSD'],
    start_date=datetime(2025, 2, 1),
    end_date=datetime(2025, 3, 1),
    initial_capital=10000.0,
    enable_meta_gating=True,
    enable_portfolio_brain=True,
    enable_rl_fallback=True,
    csv_price_dir="temp_prices",
    output_dir="logs/backtest"
)

print("\n" + "=" * 60)
print("BACKTEST RESULTS")
print("=" * 60)
print(f"Execution Time: {result.execution_time:.2f}s")
print(f"Total Trades: {result.total_trades}")
print(f"Winrate: {result.winrate*100:.1f}%")
print(f"Total Return: {result.total_return*100:.2f}%")
print(f"Max Drawdown: {result.max_drawdown*100:.2f}%")
print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
print(f"Sortino Ratio: {result.sortino_ratio:.2f}")
print(f"Profit Factor: {result.profit_factor:.2f}")
print(f"Avg R-Multiple: {result.avg_r_multiple:.2f}R")
print(f"Exposure: {result.exposure_pct*100:.1f}%")

print("\n" + "=" * 60)
print("REGIME BREAKDOWN")
print("=" * 60)
for regime, stats in result.regime_breakdown.items():
    print(f"{regime}: {stats['trades']} trades, "
          f"WR={stats['winrate']*100:.1f}%, "
          f"Avg PnL=${stats['avg_pnl']:.2f}")

print("\n" + "=" * 60)
print("DECISION SOURCE BREAKDOWN")
print("=" * 60)
for source, stats in result.decision_source_breakdown.items():
    print(f"{source}: {stats['trades']} trades, "
          f"WR={stats['winrate']*100:.1f}%, "
          f"Avg PnL=${stats['avg_pnl']:.2f}")

print("\n" + "=" * 60)
print("FILES SAVED")
print("=" * 60)
print("✅ logs/backtest/trades.csv")
print("✅ logs/backtest/equity.csv")
print("✅ logs/backtest/summary.json")
print("=" * 60)
