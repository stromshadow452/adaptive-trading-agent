# --- JARVIS PROTOCOL HOTPATCH ---
from src.ml.registry import ModelRegistry
from src.risk.circuit_breaker import CircuitBreaker
from src.features.incremental import IncrementalFeatures
import time
import json
import hashlib

def jarvis_inference_guard(context, features):
    """
    Injects JARVIS safety checks before inference.
    """
    # 1. Circuit Breaker Check
    cb = CircuitBreaker()
    cb.check_gate(context.get('symbol', 'GLOBAL'))

    # 2. Feature Parity Check
    # Compute hash of current feature columns
    current_cols = sorted(features.columns.tolist())
    s = json.dumps(current_cols, sort_keys=True)
    current_hash = hashlib.sha256(s.encode()).hexdigest()
    
    expected_hash = context.get('model_meta', {}).get('feature_hash')
    
    if expected_hash and current_hash != expected_hash:
        cb.trip(reason=f"Feature Parity Mismatch. Exp: {expected_hash[:8]}, Got: {current_hash[:8]}")
        raise ValueError(f"Feature parity mismatch! Exp: {expected_hash}, Got: {current_hash}")

    # 3. Latency Guard
    start_ts = time.perf_counter()
    
    return start_ts

def jarvis_latency_check(start_ts, symbol):
    latency_ms = (time.perf_counter() - start_ts) * 1000
    
    # Log latency
    with open(f"logs/finrl/{symbol}_latency.jsonl", "a") as f:
        f.write(json.dumps({"ts": time.time(), "latency_ms": latency_ms}) + "\n")

    if latency_ms > 200:
        print(f"[WARN] High Inference Latency: {latency_ms:.2f}ms")
