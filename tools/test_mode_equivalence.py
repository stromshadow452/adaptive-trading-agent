"""
FAST vs SHADOW Equivalence Test
================================

This test PROVES that FAST and SHADOW modes produce IDENTICAL trades.

ACCEPTANCE CRITERIA:
- Same trade count
- Same entry timestamps
- Same exit reasons
- Same PnL

If this test fails, the architecture is BROKEN.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.fast_mode import FastModeRunner, ShadowModeRunner, VectorizedFeatures
from src.backtest.execution_core import ExecutionCore
import pandas as pd


def test_equivalence(symbol: str = "EURUSD", start: str = "2022-01-01", end: str = "2022-02-28"):
    """
    Test that FAST and SHADOW produce identical results.
    """
    print("=" * 70)
    print(" FAST vs SHADOW EQUIVALENCE TEST")
    print("=" * 70)
    print(f" Symbol: {symbol}")
    print(f" Period: {start} to {end}")
    print("=" * 70)
    print()
    
    # Load data once
    from src.market_data.unified_loader import load_unified
    df, _ = load_unified(symbol, "M15", start, end)
    print(f"Loaded {len(df):,} candles")
    print()
    
    # Run FAST mode
    print("Running FAST mode...")
    fast_runner = FastModeRunner(initial_capital=10000.0, output_dir="test_fast")
    
    # Use VectorizedFeatures
    features = VectorizedFeatures(df)
    fast_core = ExecutionCore(initial_equity=10000.0)
    
    for i in range(len(df)):
        row = df.iloc[i]
        feat = features.get(i)
        regime = features.get_regime(i)
        
        fast_core.on_candle(
            timestamp=row["timestamp"],
            symbol=symbol,
            open_price=row["open"],
            high_price=row["high"],
            low_price=row["low"],
            close_price=row["close"],
            candle_index=i,
            features=feat,
            regime=regime,
            mode="FAST",
        )
    
    fast_trades = fast_core.trades
    fast_results = fast_core.get_results()
    print(f"  FAST trades: {len(fast_trades)}")
    print(f"  FAST equity: ${fast_results['final_equity']:,.2f}")
    print()
    
    # Run SHADOW mode
    print("Running SHADOW mode...")
    shadow_runner = ShadowModeRunner(initial_capital=10000.0)
    shadow_results = shadow_runner.run(df, symbol)
    shadow_trades = shadow_runner.core.trades
    print(f"  SHADOW trades: {len(shadow_trades)}")
    print(f"  SHADOW equity: ${shadow_results['final_equity']:,.2f}")
    print()
    
    # Compare
    print("=" * 70)
    print(" COMPARISON")
    print("=" * 70)
    
    passed = True
    
    # Trade count
    if len(fast_trades) != len(shadow_trades):
        print(f"❌ TRADE COUNT MISMATCH: FAST={len(fast_trades)}, SHADOW={len(shadow_trades)}")
        passed = False
    else:
        print(f"✅ Trade count match: {len(fast_trades)}")
    
    # Compare each trade
    for i, (ft, st) in enumerate(zip(fast_trades, shadow_trades)):
        mismatches = []
        
        if ft.entry_candle_index != st.entry_candle_index:
            mismatches.append(f"entry_index: {ft.entry_candle_index} vs {st.entry_candle_index}")
        
        if ft.side != st.side:
            mismatches.append(f"side: {ft.side} vs {st.side}")
        
        if ft.exit_reason != st.exit_reason:
            mismatches.append(f"exit_reason: {ft.exit_reason} vs {st.exit_reason}")
        
        if abs(ft.pnl - st.pnl) > 0.01:
            mismatches.append(f"pnl: {ft.pnl:.4f} vs {st.pnl:.4f}")
        
        if mismatches:
            print(f"❌ Trade {i+1} MISMATCH: {', '.join(mismatches)}")
            passed = False
        else:
            print(f"✅ Trade {i+1}: {ft.side} @ idx {ft.entry_candle_index} → {ft.exit_reason} (pnl={ft.pnl:.2f})")
    
    # Final verdict
    print()
    print("=" * 70)
    if passed:
        print("✅ EQUIVALENCE TEST PASSED")
        print("   FAST ≡ SHADOW (trades are identical)")
    else:
        print("❌ EQUIVALENCE TEST FAILED")
        print("   FAST ≠ SHADOW (architecture is broken)")
    print("=" * 70)
    
    return passed


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2022-02-28")
    
    args = parser.parse_args()
    
    success = test_equivalence(args.symbol, args.start, args.end)
    sys.exit(0 if success else 1)
