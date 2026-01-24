"""
Phase-1 Audit: SHADOW Mode Test
Run SHADOW mode with same parameters as FAST mode for equivalence test.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.fast_mode import ShadowModeRunner, VectorizedFeatures
from src.backtest.execution_core import ExecutionCore
from src.market_data.unified_loader import load_unified
import json


def run_shadow_audit():
    print("=" * 70)
    print(" SHADOW MODE AUDIT")
    print("=" * 70)
    
    # Load same data
    symbol = "EURUSD"
    start = "2022-01-01"
    end = "2022-03-31"
    
    print(f"Loading {symbol} {start} to {end}...")
    df, _ = load_unified(symbol, "M15", start, end)
    print(f"Loaded {len(df):,} candles")
    
    # Run SHADOW
    print("\nRunning SHADOW mode...")
    shadow = ShadowModeRunner(initial_capital=10000.0)
    results = shadow.run(df, symbol)
    
    print(f"\nSHADOW Results:")
    print(f"  Final Equity: ${results['final_equity']:,.2f}")
    print(f"  Total Trades: {results['total_trades']}")
    print(f"  Win Rate: {results['winrate']*100:.1f}%")
    
    # Print trade details
    print("\nTrade Details:")
    for i, t in enumerate(results['trades']):
        print(f"  Trade {i+1}:")
        print(f"    Entry Index: {t['entry_candle_index']}")
        print(f"    Exit Index: {t['exit_candle_index']}")
        print(f"    Side: {t['side']}")
        print(f"    Entry: {t['entry_price']:.5f}")
        print(f"    Exit: {t['exit_price']:.5f}")
        print(f"    PnL: {t['pnl']:.2f}")
        print(f"    Exit Reason: {t['exit_reason']}")
        print(f"    Regime: {t['regime']}")
    
    # Save summary
    with open("shadow_mode_audit/summary.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to shadow_mode_audit/")
    
    return results


if __name__ == "__main__":
    import os
    os.makedirs("shadow_mode_audit", exist_ok=True)
    run_shadow_audit()
