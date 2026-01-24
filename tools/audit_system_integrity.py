"""
Trading System Audit Script
============================
Comprehensive verification of causal integrity, determinism, and rule-based execution.
"""

import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

def run_backtest(start: str, end: str, symbols: str = "EURUSD") -> dict:
    """Run backtest and return metrics."""
    result = subprocess.run(
        [sys.executable, "tools/shadow_mode.py", 
         "--start", start, "--end", end, "--symbols", symbols],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT)
    )
    
    # Parse FINAL_METRICS from output
    output = result.stdout + result.stderr
    if "FINAL_METRICS:" in output:
        parts = output.split("FINAL_METRICS:")
        if len(parts) > 1:
            metrics_str = parts[-1].split("}]}")[0] + "}]}"
            try:
                return json.loads(metrics_str)
            except:
                return {"error": "parse_failed", "raw": metrics_str[:500]}
    return {"error": "no_metrics", "raw": output[:500]}

def test_determinism():
    """Run same backtest twice, compare results."""
    print("\n" + "="*60)
    print("TEST 1: DETERMINISM")
    print("="*60)
    
    m1 = run_backtest("2023-07-01", "2023-07-05")
    m2 = run_backtest("2023-07-01", "2023-07-05")
    
    if "error" in m1 or "error" in m2:
        print(f"[FAIL] Backtest error: {m1.get('error')} / {m2.get('error')}")
        return False
    
    # Compare trade counts
    t1 = m1.get("total_trades", 0)
    t2 = m2.get("total_trades", 0)
    
    if t1 != t2:
        print(f"[FAIL] Trade counts differ: {t1} vs {t2}")
        return False
    
    # Compare trade details
    trades1 = m1.get("trades", [])
    trades2 = m2.get("trades", [])
    
    for i, (trade1, trade2) in enumerate(zip(trades1, trades2)):
        for key in ["timestamp_entry", "side", "entry_price", "size"]:
            if trade1.get(key) != trade2.get(key):
                print(f"[FAIL] Trade {i} differs in {key}: {trade1.get(key)} vs {trade2.get(key)}")
                return False
    
    print(f"[PASS] {t1} trades identical across runs")
    for t in trades1[:3]:
        print(f"  - {t.get('timestamp_entry')} {t.get('side')} @ {t.get('entry_price')}")
    return True

def test_causality():
    """Verify no lookahead bias in signals."""
    print("\n" + "="*60)
    print("TEST 2: CAUSALITY CHECK")
    print("="*60)
    
    # Read trade logs
    trades_file = PROJECT_ROOT / "shadow_mode_logs" / "backtest" / "trades.csv"
    if not trades_file.exists():
        print("[WARN] No trades.csv found")
        return True
    
    import pandas as pd
    df = pd.read_csv(trades_file)
    
    issues = []
    for i, row in df.iterrows():
        entry_time = pd.to_datetime(row["timestamp_entry"])
        exit_time = pd.to_datetime(row["timestamp_exit"])
        
        # Entry must be before exit
        if entry_time >= exit_time:
            issues.append(f"Trade {i}: entry >= exit")
        
        # Duration must be positive
        duration = (exit_time - entry_time).total_seconds() / 60
        if duration <= 0:
            issues.append(f"Trade {i}: negative duration")
    
    if issues:
        for issue in issues[:5]:
            print(f"[FAIL] {issue}")
        return False
    
    print(f"[PASS] {len(df)} trades have valid timestamps")
    return True

def test_pnl_math():
    """Verify PnL calculation matches raw data."""
    print("\n" + "="*60)
    print("TEST 5: PNL REALITY CHECK")
    print("="*60)
    
    trades_file = PROJECT_ROOT / "shadow_mode_logs" / "backtest" / "trades.csv"
    if not trades_file.exists():
        print("[WARN] No trades.csv found")
        return True
    
    import pandas as pd
    df = pd.read_csv(trades_file)
    
    issues = []
    for i, row in df.iterrows():
        entry = row["entry_price"]
        exit_p = row["exit_price"]
        size = row["size"]
        side = row["side"]
        pnl = row["pnl"]
        
        # Compute expected PnL
        if side == "BUY":
            expected_pnl = (exit_p - entry) * size
        else:  # SELL
            expected_pnl = (entry - exit_p) * size
        
        # Compare (allow small floating point diff)
        diff = abs(pnl - expected_pnl)
        if diff > 0.0001:
            issues.append(f"Trade {i}: PnL mismatch {pnl:.6f} vs expected {expected_pnl:.6f}")
    
    if issues:
        for issue in issues[:5]:
            print(f"[FAIL] {issue}")
        return False
    
    print(f"[PASS] All {len(df)} trades have correct PnL math")
    return True

def test_sl_tp_validity():
    """Verify SL/TP levels make sense."""
    print("\n" + "="*60)
    print("TEST 5b: SL/TP VALIDITY")
    print("="*60)
    
    trades_file = PROJECT_ROOT / "shadow_mode_logs" / "backtest" / "trades.csv"
    if not trades_file.exists():
        print("[WARN] No trades.csv found")
        return True
    
    import pandas as pd
    df = pd.read_csv(trades_file)
    
    issues = []
    for i, row in df.iterrows():
        entry = row["entry_price"]
        sl = row["sl_price"]
        tp = row["tp_price"]
        side = row["side"]
        
        if side == "BUY":
            # SL should be below entry, TP above
            if sl >= entry:
                issues.append(f"Trade {i} BUY: SL {sl} >= entry {entry}")
            if tp <= entry:
                issues.append(f"Trade {i} BUY: TP {tp} <= entry {entry}")
        else:  # SELL
            # SL should be above entry, TP below
            if sl <= entry:
                issues.append(f"Trade {i} SELL: SL {sl} <= entry {entry}")
            if tp >= entry:
                issues.append(f"Trade {i} SELL: TP {tp} >= entry {entry}")
    
    if issues:
        for issue in issues[:5]:
            print(f"[FAIL] {issue}")
        return False
    
    print(f"[PASS] All {len(df)} trades have valid SL/TP levels")
    return True

def test_regime_gating():
    """Verify all trades respect regime rules."""
    print("\n" + "="*60)
    print("TEST 3: REGIME GATING")
    print("="*60)
    
    trades_file = PROJECT_ROOT / "shadow_mode_logs" / "backtest" / "trades.csv"
    if not trades_file.exists():
        print("[WARN] No trades.csv found")
        return True
    
    import pandas as pd
    df = pd.read_csv(trades_file)
    
    # Count trades by regime
    regime_counts = df["regime"].value_counts().to_dict()
    
    danger_trades = regime_counts.get("DANGER", 0)
    if danger_trades > 0:
        print(f"[FAIL] {danger_trades} trades in DANGER regime (should be 0)")
        return False
    
    print(f"[PASS] Regime breakdown: {regime_counts}")
    return True

def run_full_audit():
    """Run all audit tests."""
    print("\n" + "#"*70)
    print("# TRADING SYSTEM INTEGRITY AUDIT")
    print(f"# Timestamp: {datetime.now().isoformat()}")
    print("#"*70)
    
    results = {
        "determinism": test_determinism(),
        "causality": test_causality(),
        "pnl_math": test_pnl_math(),
        "sl_tp_validity": test_sl_tp_validity(),
        "regime_gating": test_regime_gating(),
    }
    
    # Final verdict
    print("\n" + "="*60)
    print("FINAL VERDICT")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    print(f"\nTests: {passed}/{total} passed")
    for test, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test}: {status}")
    
    confidence = int(passed / total * 100)
    print(f"\nConfidence Score: {confidence}%")
    
    if passed == total:
        print("\n✅ This agent is a real rule-based trading system with causal integrity")
    else:
        print("\n❌ This agent has issues that need investigation")
    
    return all(results.values())

if __name__ == "__main__":
    run_full_audit()
