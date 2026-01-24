# JARVIS Ironman Features - Executor Integration Guide

## Overview

This guide shows how to integrate the new Ironman features into `tools/executor.py`.

---

## Step 1: Add CLI Arguments

Add these arguments to the argparse section in `executor.py` (around line 1550):

```python
# Ironman Features
parser.add_argument("--enable_meta_gating", action="store_true",
                   help="Enable Meta-Gating Brain for regime detection")
parser.add_argument("--enable_portfolio_brain", action="store_true",
                   help="Enable Portfolio Brain for correlation-aware sizing")
parser.add_argument("--enable_slicer", action="store_true",
                   help="Enable Execution Slicer for large orders")
parser.add_argument("--slice_threshold", type=float, default=0.1,
                   help="Size threshold for order slicing")
parser.add_argument("--num_slices", type=int, default=5,
                   help="Number of slices for TWAP execution")
parser.add_argument("--multi_symbol", type=str, default=None,
                   help="Comma-separated list of symbols for multi-symbol mode")
```

---

## Step 2: Initialize Ironman Components

Add initialization in `main()` function (around line 1620):

```python
# Initialize Ironman components (opt-in)
meta_brain = None
portfolio_brain = None
slicer = None
metrics_collector = None

if args.enable_meta_gating:
    from src.decision.meta_gating import create_meta_gating_brain
    meta_brain = create_meta_gating_brain()
    logger.info("[IRONMAN] Meta-Gating Brain enabled")

if args.enable_portfolio_brain:
    from src.risk.portfolio import create_portfolio_brain
    portfolio_brain = create_portfolio_brain()
    logger.info("[IRONMAN] Portfolio Brain enabled")

if args.enable_slicer:
    from src.execution.slicer import create_execution_slicer
    slicer = create_execution_slicer(
        default_spread_bps=1.0,
        slippage_factor=0.5,
        min_slice_size=args.slice_threshold / 10
    )
    logger.info("[IRONMAN] Execution Slicer enabled")

# Always create metrics collector
from src.utils.metrics import create_metrics_collector
metrics_collector = create_metrics_collector()
```

---

## Step 3: Meta-Gating Integration

Add regime check in the decision loop (after RL fallback, around line 1380):

```python
# Stage 9: Meta-Gating Brain (Regime Detection)
regime_info = None
if meta_brain and args.enable_meta_gating:
    # Build feature dict from plan
    features = {
        'close': price,
        'atr14': plan.get('features', {}).get('atr14', 0.0),
        'rsi14': plan.get('features', {}).get('rsi14', 50.0),
    }
    
    regime_info = meta_brain.classify_regime(
        features=features,
        volatility=None,  # Will be calculated internally
        symbol=symbol
    )
    
    # Log regime
    metrics_collector.add_regime(regime_info['regime'])
    
    # Apply regime filter
    if regime_info['action'] == 'BLOCK':
        logger.warning(f"[META-GATING] Trade blocked: {regime_info['reason']}")
        metrics_collector.add_block(f"REGIME_{regime_info['regime']}")
        continue
    elif regime_info['action'] == 'REDUCE':
        size_factor *= regime_info['size_multiplier']
        logger.info(f"[META-GATING] Size reduced: {regime_info['reason']}")
```

---

## Step 4: Portfolio Brain Integration

Add portfolio sizing adjustment (before final size calculation, around line 1930):

```python
# Stage 7: Portfolio Brain (Correlation-Aware Sizing)
if portfolio_brain and args.enable_portfolio_brain:
    # Load price data for correlation (if multi-symbol mode)
    price_data = {}
    if args.multi_symbol:
        symbols = args.multi_symbol.split(',')
        for sym in symbols:
            csv_path = os.path.join(args.csv_price_dir, f"{sym}_M15.csv")
            if os.path.exists(csv_path):
                price_data[sym] = pd.read_csv(csv_path)
        
        # Update correlation matrix
        portfolio_brain.update_correlation_matrix(price_data)
    
    # Adjust size
    adjusted_size = portfolio_brain.adjust_size(
        symbol=symbol,
        base_size=size_plan * size_factor,
        open_positions=portfolio_brain.open_positions,
        csv_data=price_data.get(symbol)
    )
    
    final_size = _lot_clamp(adjusted_size, ...)
else:
    final_size = _lot_clamp(size_plan * size_factor, ...)
```

---

## Step 5: Execution Slicer Integration

Replace single execution with slicer logic (around line 1950):

```python
# Stage 11: Execution Reflex Engine (with Slicer)
if slicer and args.enable_slicer and final_size > args.slice_threshold:
    # Load future price data for TWAP simulation
    future_data = None
    if args.csv_price_dir:
        csv_path = os.path.join(args.csv_price_dir, f"{symbol}_M15.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, parse_dates=['timestamp'])
            # Get next N candles
            future_data = df.tail(args.num_slices)
    
    # Execute TWAP
    exec_result = slicer.execute_twap(
        symbol=symbol,
        side=side,
        size=final_size,
        csv_data=future_data,
        num_slices=args.num_slices,
        current_price=price
    )
    
    # Log sliced execution
    logger.info(f"[SLICER] Executed {exec_result.num_slices} slices, "
               f"avg_price={exec_result.avg_price:.5f}, "
               f"slippage={exec_result.total_slippage:.6f}")
    
    # Use average price for logging
    execution_price = exec_result.avg_price
else:
    # Current single-shot execution
    execution_price = price
```

---

## Step 6: Metrics Collection

Add metrics tracking throughout execution:

```python
# After successful execution
trade_record = {
    'symbol': symbol,
    'side': side,
    'size': final_size,
    'price': execution_price,
    'pnl': 0.0,  # Will be calculated on close
    'decision_source': decision_source,
    'regime': regime_info['regime'] if regime_info else 'UNKNOWN',
    'timestamp': datetime.now().isoformat()
}

metrics_collector.add_trade(trade_record)

# Track position in portfolio brain
if portfolio_brain:
    portfolio_brain.add_position(symbol, final_size, execution_price, side)
```

---

## Step 7: Session Summary

At the end of `main()`, save metrics:

```python
# Save session metrics
if metrics_collector:
    metrics_collector.print_summary()
    summary_path = metrics_collector.save_session_summary()
    logger.info(f"[METRICS] Session summary saved to {summary_path}")
```

---

## Complete Example Command

```powershell
$env:MODEL_HMAC_KEY = "your_key_here"
$env:PYTHONPATH = "$PWD"

python tools/executor.py `
  --plans tests/plans `
  --executions logs/ironman_exec.csv `
  --csv_price_dir temp_prices `
  --multi_symbol EURUSD,GBPUSD `
  --enable_meta_gating `
  --enable_portfolio_brain `
  --enable_slicer `
  --slice_threshold 0.1 `
  --num_slices 5 `
  --dry_run `
  --verbose
```

---

## Launch Dashboard

```powershell
streamlit run src/dashboard/app.py
```

---

## Verification

1. **Meta-Gating:** Check logs for `[META-GATING]` messages with regime labels
2. **Portfolio:** Check logs for `[PORTFOLIO]` size adjustments
3. **Slicer:** Check logs for `[SLICER]` execution details
4. **Metrics:** Check `logs/summary/session_*.json` for comprehensive metrics
5. **Dashboard:** Open browser to view equity curve and regime timeline

---

## Backward Compatibility

All features are **opt-in** via CLI flags. Without flags, executor behaves exactly as before.

**Default behavior preserved:**
- No `--enable_meta_gating` → No regime filtering
- No `--enable_portfolio_brain` → No correlation sizing
- No `--enable_slicer` → Single-shot execution
- Metrics always collected (minimal overhead)
