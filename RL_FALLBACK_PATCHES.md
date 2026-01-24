# RL Brain Fallback - Code Patches

## 1. Updated decide_and_execute Function

Replace the existing `decide_and_execute` function (lines 1321-1360) with:

```python
def decide_and_execute(plan: dict,
                       primary_model: Optional[Any],
                       finrl_model: Optional[Any],
                       primary_features: List[str],
                       csv_price_dirs: List[str],
                       primary_thresh: Optional[float],
                       finrl_thresh: Optional[float],
                       grey_zone: float,
                       primary_high: float = 0.70,
                       primary_block: float = 0.40,
                       rl_size_reduction: float = 0.4,
                       finrl_adapter: Optional[Any] = None,
                       verbose: bool=False) -> Tuple[str, Optional[float], Optional[str]]:
    """
    Stage 5: RL Brain Fallback Decision Logic
    
    Grey-Zone Logic:
    - primary_conf >= primary_high (0.70) → EXECUTE_PRIMARY
    - primary_conf <= primary_block (0.40) → SKIPPED_LOW_CONF
    - 0.40 < primary_conf < 0.70 → Try RL fallback if available
    """
    primary_conf, _ = get_confidences(
        plan, primary_model, finrl_model, primary_features, csv_price_dirs, verbose=verbose
    )
    sym_local = _extract_symbol_from_plan(plan, verbose=False)

    if verbose:
        print(f"[STAGE-5-RL] {sym_local}: primary_conf={primary_conf:.3f}, "
              f"thresholds: high={primary_high:.2f}, block={primary_block:.2f}")

    # Decision Logic
    if primary_conf >= primary_high:
        # High confidence - use PRIMARY only
        return "EXECUTE_PRIMARY", 1.0, f"primary_conf={primary_conf:.3f} >= high_thresh={primary_high:.3f}"
    
    elif primary_conf <= primary_block:
        # Too low - block trade
        return "SKIPPED_LOW_CONF", None, f"primary_conf={primary_conf:.3f} <= block_thresh={primary_block:.3f}"
    
    else:
        # Grey zone: 0.40 < primary_conf < 0.70
        # Try RL fallback if available
        if finrl_adapter and finrl_adapter.is_available():
            try:
                # Build feature vector for RL
                feature_vector = build_feature_vector_from_csv(
                    sym_local, csv_price_dirs, primary_features, verbose=verbose
                )
                
                if feature_vector is not None:
                    fv, _ = feature_vector
                    finrl_conf, finrl_action = finrl_adapter.predict_proba(fv)
                    
                    # Check RL confidence threshold
                    rl_min_thresh = finrl_thresh if finrl_thresh is not None else 0.65
                    
                    if finrl_conf >= rl_min_thresh and finrl_action != 0:
                        if verbose:
                            print(f"[RL-FALLBACK] {sym_local}: finrl_conf={finrl_conf:.3f}, "
                                  f"action={finrl_action}, size_reduction={rl_size_reduction}")
                        
                        return "EXECUTE_FINRL_FALLBACK", rl_size_reduction, (
                            f"primary_conf={primary_conf:.3f} in grey zone, "
                            f"finrl_conf={finrl_conf:.3f} >= {rl_min_thresh:.3f}, "
                            f"action={finrl_action}"
                        )
                    else:
                        if verbose:
                            print(f"[RL-FALLBACK] {sym_local}: RL conf too low "
                                  f"({finrl_conf:.3f} < {rl_min_thresh:.3f})")
                        
            except Exception as e:
                if verbose:
                    print(f"[RL-FALLBACK] {sym_local}: RL prediction failed: {e}")
        
        # Fallback: grey zone but RL unavailable or failed
        return "SKIPPED_LOW_CONF", None, (
            f"primary_conf={primary_conf:.3f} in grey zone but RL unavailable/failed"
        )
```

## 2. Initialize FinRLAdapter in main()

Add after PRIMARY_META is loaded (around line 1620):

```python
# Initialize FinRL Adapter (Stage 5: RL Brain Fallback)
finrl_adapter = None
if not args.disable_finrl and args.model_finrl_path:
    try:
        # Try to load FinRL model with JARVIS guards
        finrl_adapter = load_finrl_adapter(
            model_path=args.model_finrl_path,
            registry=ModelRegistry(registry_path=os.path.dirname(args.model_primary_path)),
            primary_meta=PRIMARY_META
        )
        
        if finrl_adapter and finrl_adapter.is_available():
            print(f"[RL-ADAPTER] FinRL fallback enabled: {args.model_finrl_path}")
        else:
            print(f"[RL-ADAPTER] FinRL model unavailable, fallback disabled")
            finrl_adapter = None
    except Exception as e:
        print(f"[RL-ADAPTER] Failed to load FinRL adapter: {e}")
        finrl_adapter = None
```

## 3. Update decide_and_execute Call

Update the call to `decide_and_execute` (around line 1853) to pass new parameters:

```python
decision, size_factor, reason = decide_and_execute(
    plan, primary_model, finrl_model_eff, primary_features, args.csv_price_dir,
    args.primary_thresh, args.finrl_thresh, args.grey_zone,
    primary_high=args.primary_high_threshold,
    primary_block=args.primary_block_threshold,
    rl_size_reduction=args.rl_size_reduction,
    finrl_adapter=finrl_adapter,
    verbose=args.verbose
)
```

## 4. Extend Execution Logging

Update execution row creation (around line 1865) to include RL metadata:

```python
row = {
    "ts": "", "mode": args.mode, "symbol": sym, "tf": tf, "side": side,
    "price": f"{price:.8f}", "size": f"{final_size:.8f}",
    "sl": "" if sl is None else f"{float(sl):.8f}",
    "tp": "" if tp is None else f"{float(tp):.8f}",
    "sl_type": sltype, "status": "OPEN", "order_id": order_id,
    "plan_file": globals().get("plan", {}).get("_plan_path") or (args.approved or args.plans),
    "decision_source": decision,  # NEW: "PRIMARY" or "FINRL_FALLBACK"
    "size_factor": f"{size_factor:.2f}" if size_factor else "1.00",  # NEW
}
```

Update EXEC_FIELDS (line 1363):

```python
EXEC_FIELDS = ["ts","mode","symbol","tf","side","price","size","sl","tp","sl_type",
               "status","order_id","plan_file","decision_source","size_factor"]
```

## 5. Console Logging Enhancement

Update console output (around line 1907 and 1942) to show decision source:

```python
if args.dry_run:
    print(f"[DRY OPEN] {sym} {tf} {side} price={price:.6f} size={final_size:.6f}  "
          f"-> {decision}: {reason}")
else:
    print(f"[OPEN] {sym} {tf} {side} price={price:.6f} size={final_size:.6f}  "
          f"-> {decision}: {reason}")
```

---

## Integration Summary

**Files Modified:**
1. `src/agents/finrl_adapter.py` - NEW (created)
2. `src/ml/registry.py` - Added `load_model_from_path()` method
3. `tools/executor.py` - Added:
   - FinRLAdapter import
   - CLI args: `--primary_high_threshold`, `--primary_block_threshold`, `--rl_size_reduction`
   - Updated `decide_and_execute()` function
   - FinRLAdapter initialization in `main()`
   - Extended logging (CSV + console)

**Pipeline Position:** Stage 5 (between Primary ML Brain and Volatility Brain)

**Safety Guarantees:**
- ✅ HMAC integrity check for RL models
- ✅ Feature parity validation
- ✅ Circuit breaker integration
- ✅ Graceful degradation if RL unavailable
- ✅ Size reduction (0.4x default)
- ✅ Grey-zone thresholds (0.40-0.70)
