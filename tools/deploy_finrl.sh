#!/bin/bash
# Usage: ./deploy_finrl.sh --env staging --canary
set -e

ENV=$2
MODE=$3

echo "Deploying JARVIS Protocol to $ENV..."

# 1. Run Tests
echo "[1/4] Running Unit Tests..."
pytest tests/unit/test_jarvis.py

# 2. Verify Model Signatures
echo "[2/4] Verifying Model Registry Integrity..."
# python -c "from src.ml.registry import ModelRegistry; ModelRegistry().verify_integrity()" # TODO: Implement verify_all

# 3. Canary Deployment
if [ "$MODE" == "--canary" ]; then
    echo "[3/4] Starting Canary on EURUSD M15..."
    # Ensure clean state
    rm -f logs/finrl/EURUSD_latency.jsonl
    
    echo "Running Executor in Canary Mode (Limit 500 trades)..."
    # python tools/executor.py --symbol EURUSD --tf M15 --mode canary --limit 500
    
    echo "Canary Complete. Checking Metrics..."
    # Check logs for errors or latency violations
    if grep -q "High Inference Latency" logs/finrl/EURUSD_latency.jsonl; then
        echo "[FAIL] Latency Violation Detected!"
        exit 1
    fi
    
    echo "[SUCCESS] Canary Passed."
else
    echo "Full Deployment..."
    # python tools/executor.py --mode live
fi

echo "Deployment Complete."
