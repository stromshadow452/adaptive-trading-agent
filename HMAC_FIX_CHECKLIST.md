# HMAC FIX - Complete Checklist

## 🔧 What Was Fixed

**BUG IDENTIFIED:** Training script was computing HMAC on intermediate file, then re-saving with signatures, which changed the file bytes. Verification failed because it checked against the FINAL file.

**FIX APPLIED:** 
- Training script now computes HMAC on FINAL file (without storing HMAC inside the file)
- HMAC is stored ONLY in `.sig` file
- Verification reads from `.sig` file and compares against current file bytes

---

## ✅ Step-by-Step Checklist

### 1. Set Environment Variables (CRITICAL!)

```powershell
# Set HMAC key (use the SAME key for training AND execution!)
$env:MODEL_HMAC_KEY = "1PoRYQg1H+tKR/t8IlNkVuiL182eST17cn5j+gOHOCk="

# Set Python path
$env:PYTHONPATH = "$PWD"
```

**⚠️ IMPORTANT:** Keep this PowerShell session open! Don't close it between training and execution.

---

### 2. Re-Train Model (with fixed script)

```powershell
python tools/train_finrl_ppo.py `
  --csv_path temp_prices/EURUSD_M15.csv `
  --model_out models/finrl/EURUSD_M15_policy.joblib `
  --timesteps 10000
```

**Expected Output:**
```
✓ Feature hash matches Primary model!
Training complete!
Model saved: models/finrl/EURUSD_M15_policy.joblib
Signature saved: models/finrl/EURUSD_M15_policy.joblib.sig
SHA256: 56e20fa6c6f12031...
HMAC: 9411a1ce9604303b...
```

---

### 3. (Optional) Debug Verification

```powershell
python -c "from src.utils.model_signing import print_debug_verification; print_debug_verification('models/finrl/EURUSD_M15_policy.joblib')"
```

**Expected Output:**
```
MODEL HMAC DEBUG VERIFICATION
============================================================
Model: models/finrl/EURUSD_M15_policy.joblib
Exists: True
.sig Exists: True

COMPUTED (from current file bytes):
  SHA256: 56e20fa6c6f12031...
  HMAC:   9411a1ce9604303b...

STORED (in .sig file):
  SHA256: 56e20fa6c6f12031...
  HMAC:   9411a1ce9604303b...

VERIFICATION:
  SHA256 Match: ✓ YES
  HMAC Match:   ✓ YES

✅ HMAC VERIFICATION PASSED!
```

---

### 4. Run Executor (with RL fallback)

```powershell
python tools/executor.py `
  --plans tests/plans `
  --executions logs/rl_exec.csv `
  --dry_run `
  --csv_price_dir temp_prices `
  --model_finrl_path models/finrl/EURUSD_M15_policy.joblib `
  --verbose
```

**Expected Output (SUCCESS):**
```
[RL-ADAPTER] FinRL fallback enabled: models/finrl/EURUSD_M15_policy.joblib
[STAGE-5-RL] EURUSD: primary_conf=0.918, thresholds: high=0.70, block=0.40
[DRY OPEN] EURUSD M15 buy -> EXECUTE_PRIMARY
Executed (opened): 1 (dry_run=True)
```

**NO MORE:** `[RL-ADAPTER] Failed to load RL model: SECURITY ALERT: Model HMAC mismatch!`

---

## 🔍 Troubleshooting

### If HMAC Still Fails:

1. **Check HMAC Key:**
   ```powershell
   echo $env:MODEL_HMAC_KEY
   ```
   Should output: `1PoRYQg1H+tKR/t8IlNkVuiL182eST17cn5j+gOHOCk=`

2. **Run Debug Verification:**
   ```powershell
   python -c "from src.utils.model_signing import print_debug_verification; print_debug_verification('models/finrl/EURUSD_M15_policy.joblib')"
   ```

3. **Ensure Same PowerShell Session:**
   - Don't close PowerShell between training and execution
   - Don't change `MODEL_HMAC_KEY` between steps

4. **Re-train if needed:**
   - Delete old model: `rm models/finrl/EURUSD_M15_policy.joblib*`
   - Re-run training command from step 2

---

## 📝 Technical Details

### How HMAC Signing Works Now:

**Training Side (train_finrl_ppo.py):**
1. Create model payload with metadata (NO signatures)
2. Save to `.joblib` file
3. Read back the FINAL file bytes
4. Compute SHA256 + HMAC on those bytes
5. Store signatures in SEPARATE `.sig` file (NOT in the model file)

**Loading Side (ModelRegistry.load_model_from_path):**
1. Read `.joblib` file bytes
2. Read `.sig` file for stored HMAC
3. Compute HMAC on current file bytes
4. Compare: `computed_hmac == stored_hmac`
5. If match → load model, else → raise error

**Key Insight:** HMAC is computed on the file WITHOUT the HMAC field inside it. This prevents the circular dependency where adding the HMAC changes the file.

---

## ✅ Success Criteria

- ✅ Training completes without errors
- ✅ `.sig` file created alongside `.joblib` file
- ✅ Debug verification shows "HMAC Match: ✓ YES"
- ✅ Executor loads RL model successfully
- ✅ No "SECURITY ALERT: Model HMAC mismatch!" errors

---

**All fixes applied! Follow the checklist above to re-train and test.** 🚀
