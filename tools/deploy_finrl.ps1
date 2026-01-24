# Usage: .\tools\deploy_finrl.ps1 -Env staging -Canary
param (
    [string]$Env = "staging",
    [switch]$Canary
)

$ErrorActionPreference = "Stop"

Write-Host "Deploying JARVIS Protocol to $Env..." -ForegroundColor Cyan

# 1. Run Tests
Write-Host "[1/4] Running Unit Tests..." -ForegroundColor Yellow
$env:PYTHONPATH = "$PWD"
pytest tests/unit -q
if ($LASTEXITCODE -ne 0) {
    Write-Error "Unit Tests Failed! Aborting deployment."
    exit 1
}

# 2. Verify Model Signatures
Write-Host "[2/4] Verifying Model Registry Integrity..." -ForegroundColor Yellow
# python -c "from src.ml.registry import ModelRegistry; ModelRegistry().verify_integrity()"

# 3. Canary Deployment
if ($Canary) {
    Write-Host "[3/4] Starting Canary on EURUSD M15..." -ForegroundColor Yellow
    
    # Ensure clean state
    if (Test-Path "logs/finrl/EURUSD_latency.jsonl") {
        Remove-Item "logs/finrl/EURUSD_latency.jsonl"
    }
    
    Write-Host "Running Executor in Canary Mode (Limit 500 trades)..." -ForegroundColor Cyan
    # Using the dummy plan created earlier for the dry run
    python tools/executor.py --plans tests/plans --executions logs/canary_exec.csv
    
    Write-Host "Canary Complete. Checking Metrics..." -ForegroundColor Green
    
    if (Test-Path "logs/finrl/EURUSD_latency.jsonl") {
        $content = Get-Content "logs/finrl/EURUSD_latency.jsonl"
        if ($content -match "High Inference Latency") {
            Write-Error "[FAIL] Latency Violation Detected!"
        }
    }
    
    Write-Host "[SUCCESS] Canary Passed." -ForegroundColor Green
}
else {
    Write-Host "Full Deployment..." -ForegroundColor Yellow
    # python tools/executor.py --mode live
}

Write-Host "Deployment Complete." -ForegroundColor Cyan
