# FinRL PPO Training - Quick Start Guide

## ✅ Files Created

1. **`src/utils/model_signing.py`** - HMAC signing helper
2. **`tools/train_finrl_ppo.py`** - Complete PPO training script
3. **`tools/validate_finrl_model.py`** - Model validation & backtest script

---

## 🔧 Fixed Issues

### Feature Hash Mismatch (RESOLVED)
- **Problem:** Initial feature list didn't match Primary model
- **Root Cause:** Used wrong features (rsi, macd, ema_20, etc.)
- **Solution:** Updated to use actual `TRAIN_FEATURES_DEFAULT` from executor.py:
  ```python
  ["close", "ret", "sma5", "sma20", "sma_ratio", "sma50", "sma100", "sma_ratio_long",
   "atr14", "hl_range", "body", "ret_5", "ret_20", "rsi14", "boll_z", "atr_pct", "vol_norm", "hod", "dow"]
  ```
- **Feature Computation:** Now uses `compute_features_from_ohlcv()` from `src/features/common_features.py`
- **Expected Hash:** `59207f8f82cf29243b0f580753de15c09632e3ca08092b2edda54e2fc3dc8c03`

---

## 🚀 Usage Instructions

### Step 1: Set HMAC Key
```powershell
$env:MODEL_HMAC_KEY = "your_secret_key_here"
```

### Step 2: Train Model
```powershell
$env:PYTHONPATH = "$PWD"
python tools/train_finrl_ppo.py `
  --csv_path data/eurusd/EURUSD_M15.csv `
  --model_out models/finrl/EURUSD_M15_policy.joblib `
  --timesteps 100000
```

**Expected Output:**
```
[FEATURE PARITY CHECK]
  Computed hash: 59207f8f82cf29243b0f580753de15c09632e3ca08092b2edda54e2fc3dc8c03
  Expected hash: 59207f8f82cf29243b0f580753de15c09632e3ca08092b2edda54e2fc3dc8c03
  ✓ Feature hash matches Primary model!
```

### Step 3: Validate Model
```powershell
python tools/validate_finrl_model.py `
  --csv_path data/eurusd/EURUSD_M15.csv `
  --model_path models/finrl/EURUSD_M15_policy.joblib
```

### Step 4: Use with Executor
```powershell
python tools/executor.py `
  --plans tests/plans `
  --executions logs/rl_exec.csv `
  --dry_run `
  --csv_price_dir temp_prices `
  --model_finrl_path models/finrl/EURUSD_M15_policy.joblib `
  --verbose
```

---

## 📋 Requirements

- ✅ stable-baselines3 (installed)
- ✅ gymnasium (installed)
- ✅ EURUSD M15 CSV data (your existing data)
- ✅ MODEL_HMAC_KEY environment variable

---

## 🎯 Next Steps

1. Run training script with your EURUSD data
2. Verify feature hash matches (should be `59207f8f...`)
3. Validate model performance metrics
4. Test with executor in grey-zone scenario

---

**All scripts ready to run!** 🚀
