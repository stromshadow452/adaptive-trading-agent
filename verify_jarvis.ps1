Write-Host "=== JARVIS Protocol Verification ===" -ForegroundColor Cyan

# 1. Check files exist
Write-Host ""
Write-Host "[1/5] Checking JARVIS files..." -ForegroundColor Yellow
$files = @(
    "src/ml/registry.py",
    "src/features/incremental.py",
    "src/risk/circuit_breaker.py",
    "tools/executor.py",
    "tests/unit/test_jarvis.py"
)
foreach ($f in $files) {
    if (Test-Path $f) { 
        Write-Host "  OK $f" -ForegroundColor Green 
    }
    else { 
        Write-Host "  MISSING $f" -ForegroundColor Red 
    }
}

# 2. Check imports
Write-Host ""
Write-Host "[2/5] Testing imports..." -ForegroundColor Yellow
$env:PYTHONPATH = "$PWD"
python -c "from src.ml.registry import ModelRegistry; print('  OK ModelRegistry')"
python -c "from src.features.incremental import IncrementalFeatures; print('  OK IncrementalFeatures')"
python -c "from src.risk.circuit_breaker import CircuitBreaker; print('  OK CircuitBreaker')"

# 3. Check JARVIS guards in executor
Write-Host ""
Write-Host "[3/5] Checking JARVIS guards..." -ForegroundColor Yellow
$guards = Select-String -Path "tools/executor.py" -Pattern "JARVIS|_jarvis_guard" -Quiet
if ($guards) { 
    Write-Host "  OK JARVIS guards found in executor" -ForegroundColor Green 
}
else { 
    Write-Host "  MISSING JARVIS guards NOT found" -ForegroundColor Red 
}

# 4. Run executor
Write-Host ""
Write-Host "[4/5] Running executor..." -ForegroundColor Yellow
python tools/executor.py --plans tests/plans --executions logs/verify_exec.csv

# 5. Run deployment
Write-Host ""
Write-Host "[5/5] Running deployment script..." -ForegroundColor Yellow
powershell -ExecutionPolicy Bypass -File tools/deploy_finrl.ps1 -Env staging -Canary

Write-Host ""
Write-Host "=== Verification Complete ===" -ForegroundColor Cyan
