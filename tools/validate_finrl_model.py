"""
FinRL Model Validation Script for JARVIS Trading Agent

Validates trained PPO model and runs backtest to compute performance metrics.

Usage:
    python tools/validate_finrl_model.py --csv_path data/eurusd/EURUSD_M15.csv --model_path models/finrl/EURUSD_M15_policy.joblib
"""
import argparse
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

# Add repo root to path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.utils.model_signing import compute_feature_hash

# EXACT 19 features from Primary ML Brain (from executor.py TRAIN_FEATURES_DEFAULT)
REQUIRED_FEATURES = [
    "close", "ret", "sma5", "sma20", "sma_ratio", "sma50", "sma100", "sma_ratio_long",
    "atr14", "hl_range", "body", "ret_5", "ret_20", "rsi14", "boll_z", "atr_pct", "vol_norm", "hod", "dow"
]

# Expected feature hash
EXPECTED_FEATURE_HASH = "sha256:59207f8f82cf29243b0f580753de15c09632e3ca08092b2edda54e2fc3dc8c03"


def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    """Load CSV and compute 19 features (same as training script)"""
    print(f"[1/4] Loading data from {csv_path}...")
    
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df.set_index("timestamp")
    
    print(f"  Loaded {len(df)} candles")
    
    # Compute features using common_features.py
    print("[2/4] Computing features...")
    try:
        from src.features.common_features import compute_features_from_ohlcv
        df_features = compute_features_from_ohlcv(df)
        return df_features
    except ImportError as e:
        print(f"  [ERROR] Could not import compute_features_from_ohlcv: {e}")
        sys.exit(1)


def validate_model_metadata(model_path: str) -> dict:
    """
    Validate model metadata and signatures
    
    Returns:
        metadata dict
    """
    print(f"[3/4] Validating model metadata...")
    
    # Load model
    payload = joblib.load(model_path)
    
    if not isinstance(payload, dict):
        raise ValueError("Model payload must be a dict with 'model' and 'meta' keys")
    
    if "meta" not in payload:
        raise ValueError("Model payload missing 'meta' key")
    
    meta = payload["meta"]
    
    # Check required fields
    required_meta_fields = ["feature_hash", "sha256", "hmac"]
    missing = [f for f in required_meta_fields if f not in meta]
    if missing:
        raise ValueError(f"Metadata missing required fields: {missing}")
    
    # Verify feature hash
    feature_hash = meta["feature_hash"]
    print(f"  Feature hash: {feature_hash}")
    
    if feature_hash != EXPECTED_FEATURE_HASH:
        print(f"  [WARN] Feature hash mismatch!")
        print(f"    Expected: {EXPECTED_FEATURE_HASH}")
        print(f"    Got:      {feature_hash}")
        print(f"  This model will be REJECTED by JARVIS feature parity check.")
    else:
        print(f"  ✓ Feature hash matches Primary model")
    
    # Check signatures
    print(f"  SHA256: {meta['sha256'][:16]}...")
    print(f"  HMAC: {meta['hmac'][:16]}...")
    print(f"  ✓ Signatures present")
    
    return meta


def run_backtest(model, df: pd.DataFrame) -> dict:
    """
    Run backtest on validation data
    
    Returns:
        metrics dict
    """
    print(f"[4/4] Running backtest on {len(df)} candles...")
    
    # Initialize backtest state
    balance = 10000.0
    position = 0  # -1, 0, 1
    entry_price = 0.0
    trades = []
    equity_curve = [balance]
    
    # Run through data
    for i in range(len(df)):
        row = df.iloc[i]
        
        # Get observation (19 features)
        obs = np.array([
            row["open"], row["high"], row["low"], row["close"], row["volume"],
            row["rsi"], row["macd"], row["macd_signal"], row["macd_hist"],
            row["ema_20"], row["ema_50"], row["ema_200"],
            row["atr"], row["vwap"], row["returns"], row["volatility"],
            row["roll_max"], row["roll_min"], row["spread"]
        ], dtype=np.float32)
        
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        obs = obs.reshape(1, -1)
        
        # Get model action
        try:
            action, _states = model.predict(obs, deterministic=True)
            action = int(action[0]) if hasattr(action, '__len__') else int(action)
            action_mapped = action - 1  # 0->-1, 1->0, 2->+1
        except Exception as e:
            print(f"  [WARN] Model prediction failed at step {i}: {e}")
            action_mapped = 0
        
        current_price = row["close"]
        
        # Execute action
        if action_mapped == 1 and position == 0:  # BUY
            position = 1
            entry_price = current_price
        elif action_mapped == -1 and position == 0:  # SELL
            position = -1
            entry_price = current_price
        elif action_mapped == 0 and position != 0:  # CLOSE
            if position == 1:  # Close long
                profit_pct = (current_price - entry_price) / entry_price
            else:  # Close short
                profit_pct = (entry_price - current_price) / entry_price
            
            profit_abs = balance * profit_pct
            balance += profit_abs
            
            trades.append({
                "entry_price": entry_price,
                "exit_price": current_price,
                "position": "long" if position == 1 else "short",
                "profit_pct": profit_pct,
                "profit_abs": profit_abs
            })
            
            position = 0
            entry_price = 0.0
        
        equity_curve.append(balance)
    
    # Compute metrics
    total_trades = len(trades)
    
    if total_trades == 0:
        print("  [WARN] No trades executed during backtest")
        return {
            "total_trades": 0,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "winrate": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "reliability_p70": 0.0
        }
    
    # Total return
    total_return = (balance - 10000.0) / 10000.0
    
    # Max drawdown
    equity_curve = np.array(equity_curve)
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / running_max
    max_drawdown = abs(drawdown.min())
    
    # Winrate
    winning_trades = sum(1 for t in trades if t["profit_pct"] > 0)
    winrate = winning_trades / total_trades if total_trades > 0 else 0.0
    
    # Sharpe ratio (simplified)
    returns = [t["profit_pct"] for t in trades]
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    sharpe_ratio = mean_return / std_return if std_return > 0 else 0.0
    
    # Sortino ratio (downside deviation)
    downside_returns = [r for r in returns if r < 0]
    downside_std = np.std(downside_returns) if downside_returns else 1e-6
    sortino_ratio = mean_return / downside_std if downside_std > 0 else 0.0
    
    # Reliability (p >= 0.7) - percentage of trades with profit >= 0.7%
    reliable_trades = sum(1 for t in trades if t["profit_pct"] >= 0.007)
    reliability_p70 = reliable_trades / total_trades if total_trades > 0 else 0.0
    
    metrics = {
        "total_trades": total_trades,
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "winrate": winrate,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "reliability_p70": reliability_p70,
        "final_balance": balance
    }
    
    return metrics


def print_metrics(metrics: dict):
    """Print metrics to console"""
    print("\n" + "=" * 80)
    print("BACKTEST METRICS")
    print("=" * 80)
    print(f"Total Trades:       {metrics['total_trades']}")
    print(f"Total Return:       {metrics['total_return']*100:.2f}%")
    print(f"Max Drawdown:       {metrics['max_drawdown']*100:.2f}%")
    print(f"Winrate:            {metrics['winrate']*100:.2f}%")
    print(f"Sharpe Ratio:       {metrics['sharpe_ratio']:.3f}")
    print(f"Sortino Ratio:      {metrics['sortino_ratio']:.3f}")
    print(f"Reliability (p≥0.7%): {metrics['reliability_p70']*100:.2f}%")
    print(f"Final Balance:      ${metrics['final_balance']:.2f}")
    print("=" * 80)
    
    # Check requirements
    print("\nREQUIREMENT CHECKS:")
    checks = [
        ("Sample Size >= 500", metrics['total_trades'] >= 500),
        ("Total Return > 0", metrics['total_return'] > 0),
        ("Max Drawdown <= 20%", metrics['max_drawdown'] <= 0.20),
        ("Reliability >= 60%", metrics['reliability_p70'] >= 0.60)
    ]
    
    for check_name, passed in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {check_name}")
    
    all_passed = all(passed for _, passed in checks)
    if all_passed:
        print("\n✓ All requirements MET!")
    else:
        print("\n✗ Some requirements FAILED. Consider retraining with more data or tuning hyperparameters.")


def main():
    parser = argparse.ArgumentParser(description="Validate FinRL PPO model")
    parser.add_argument("--csv_path", required=True, help="Path to validation CSV")
    parser.add_argument("--model_path", required=True, help="Path to trained model .joblib")
    parser.add_argument("--metrics_out", default="logs/finrl/EURUSD_M15_metrics.json", help="Output metrics JSON")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("FinRL Model Validation for JARVIS Protocol")
    print("=" * 80)
    
    # Load data
    df = load_and_prepare_data(args.csv_path)
    
    # Validate metadata
    meta = validate_model_metadata(args.model_path)
    
    # Load model
    payload = joblib.load(args.model_path)
    model = payload["model"]
    
    # Run backtest
    metrics = run_backtest(model, df)
    
    # Print results
    print_metrics(metrics)
    
    # Save metrics
    os.makedirs(os.path.dirname(args.metrics_out), exist_ok=True)
    with open(args.metrics_out, "w") as f:
        json.dump({
            "metadata": meta,
            "metrics": metrics,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }, f, indent=2)
    
    print(f"\nMetrics saved to: {args.metrics_out}")
    print("=" * 80)


if __name__ == "__main__":
    main()
