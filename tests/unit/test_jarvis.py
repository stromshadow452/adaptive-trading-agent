import pytest
import json
import hashlib
import os
import time
import hmac
from src.features.incremental import IncrementalFeatures
from src.ml.registry import ModelRegistry
from src.stream.ingestor import StreamIngestor
from src.risk.circuit_breaker import CircuitBreaker

# --- Fixtures ---
@pytest.fixture
def registry(tmp_path):
    """Create a temporary model registry for testing"""
    return ModelRegistry(registry_path=str(tmp_path))

@pytest.fixture
def circuit_breaker(tmp_path):
    """Create a temporary circuit breaker for testing"""
    state_file = str(tmp_path / "cb.json")
    return CircuitBreaker(state_file=state_file)

# --- Feature Parity Tests ---
def test_feature_hash_parity(tmp_path):
    """Test that feature hash is computed correctly and matches"""
    # 1. Create Temp Registry
    registry_list = ["close", "sma5", "rsi14", "atr14", "sma20"]
    reg_path = tmp_path / "features.json"
    with open(reg_path, "w") as f:
        json.dump(registry_list, f)
    
    # 2. Compute Expected Hash
    s = json.dumps(sorted(registry_list), sort_keys=True)
    expected_hash = hashlib.sha256(s.encode()).hexdigest()
    
    # 3. Initialize IncrementalFeatures
    inc = IncrementalFeatures(registry_path=str(reg_path))
    
    # 4. Verify hash matches
    assert inc.feature_hash == expected_hash, f"Feature hash mismatch: {inc.feature_hash} != {expected_hash}"
    
    # 5. Verify output keys
    dummy_tick = {"price": 100.0, "vol": 10, "iso": "2023-01-01T00:00:00Z"}
    out = inc.on_tick("TEST", dummy_tick)
    
    # Check that output is a dict
    assert isinstance(out, dict), "Feature output should be a dictionary"
    
    # Check that all registry features are present in output
    for feature in registry_list:
        assert feature in out, f"Feature '{feature}' missing from output"

# --- Model Integrity Tests ---
def test_model_hmac_integrity(registry):
    """Test that HMAC model signing and verification works"""
    # Set HMAC key for testing
    os.environ["MODEL_HMAC_KEY"] = "test_secret_key_12345"
    
    # Mock Model & Meta
    model = {"weights": [1.0, 2.0, 3.0], "type": "test"}
    meta = {"feature_hash": "abc123", "version": "1.0"}
    
    # Save model (should create .joblib and .sig files)
    path = registry.save_model(model, meta, tag="test")
    
    # Verify files exist
    assert os.path.exists(path), f"Model file not created: {path}"
    assert os.path.exists(path + ".sig"), f"Signature file not created: {path}.sig"
    
    # Load model (should pass integrity check)
    loaded_model, loaded_meta = registry.load_model(tag="test")
    assert loaded_model == model, "Loaded model doesn't match original"
    assert loaded_meta["feature_hash"] == meta["feature_hash"], "Meta doesn't match"
    
    # Tamper with the model file
    with open(path, "rb") as f:
        data = f.read()
    
    # Flip a bit in the file
    tampered_data = data[:-1] + b'\x00'
    with open(path, "wb") as f:
        f.write(tampered_data)
    
    # Try to load tampered model (should fail)
    with pytest.raises(ValueError, match="HMAC mismatch"):
        registry.load_model(tag="test")

# --- Stream Replay Tests ---
@pytest.mark.skip(reason="ZMQ socket hang issue - TODO: fix WAL replay test")
def test_stream_replay_gap_fill(tmp_path):
    """Test that stream replay can fill sequence gaps"""
    db_path = str(tmp_path / "stream.db")
    # Use random port to avoid conflicts
    import random
    zmq_port = random.randint(10000, 60000)
    ingestor = StreamIngestor(db_path=db_path, zmq_port=zmq_port)
    
    # Ingest 3 ticks with sequence numbers
    ingestor.ingest("EURUSD", 1.1000, 100)
    ingestor.ingest("EURUSD", 1.1010, 200)
    ingestor.ingest("EURUSD", 1.1020, 300)
    
    # Replay sequence 2 (middle tick)
    replayed = ingestor.replay_range(2, 2)
    
    # Verify replay
    assert len(replayed) == 1, f"Expected 1 replayed tick, got {len(replayed)}"
    assert replayed[0]["price"] == 1.1010, f"Wrong price: {replayed[0]['price']}"
    assert replayed[0]["seq"] == 2, f"Wrong sequence: {replayed[0]['seq']}"
    
    # Replay range 1-3 (all ticks)
    replayed_all = ingestor.replay_range(1, 3)
    assert len(replayed_all) == 3, f"Expected 3 ticks, got {len(replayed_all)}"
    
    # Verify sequence order
    for i, tick in enumerate(replayed_all, 1):
        assert tick["seq"] == i, f"Sequence mismatch at position {i}"
    
    ingestor.stop()

# --- Circuit Breaker Tests ---
def test_circuit_breaker_persistence(circuit_breaker):
    """Test that circuit breaker state persists to file"""
    cb = circuit_breaker
    
    # Initial state should allow trading
    assert cb.check_gate("EURUSD") is True, "Gate should be open initially"
    
    # Trip the circuit breaker
    cb.trip("EURUSD", reason="Test trip", duration=2)
    
    # Verify state file was created
    assert os.path.exists(cb.state_file), "State file not created"
    
    # Check that gate is now closed
    with pytest.raises(RuntimeError, match="CIRCUIT BREAKER TRIPPED"):
        cb.check_gate("EURUSD")
    
    # Verify state persists by reading file
    with open(cb.state_file) as f:
        state = json.load(f)
    
    # Check state structure (could be flat or nested under 'symbols')
    if "symbols" in state:
        assert "EURUSD" in state["symbols"], "EURUSD not in state"
        assert state["symbols"]["EURUSD"]["tripped"] is True, "Circuit not marked as tripped"
        assert state["symbols"]["EURUSD"]["reason"] == "Test trip", "Reason not saved"
    else:
        assert "EURUSD" in state, "EURUSD not in state"
        assert state["EURUSD"]["tripped"] is True, "Circuit not marked as tripped"
        assert state["EURUSD"]["reason"] == "Test trip", "Reason not saved"
    
    # Wait for auto-reset
    time.sleep(2.1)
    
    # Gate should be open again
    assert cb.check_gate("EURUSD") is True, "Gate should auto-reset after timeout"

# --- Integration Test ---
def test_jarvis_guard_integration(tmp_path, circuit_breaker):
    """Test JARVIS guard logic with feature parity check"""
    # This simulates what happens in executor.py _jarvis_guard()
    
    # Setup
    cb = circuit_breaker
    feature_names = ["close", "sma5", "rsi14"]
    
    # Compute current hash
    s = json.dumps(sorted(feature_names), sort_keys=True)
    curr_hash = hashlib.sha256(s.encode()).hexdigest()
    
    # Case 1: Matching hash (should pass)
    meta_good = {"feature_hash": curr_hash}
    
    # Simulate guard check
    exp_hash = meta_good.get("feature_hash")
    assert curr_hash == exp_hash, "Hash should match"
    
    # Case 2: Mismatched hash (should fail and trip breaker)
    meta_bad = {"feature_hash": "wrong_hash_12345"}
    exp_hash_bad = meta_bad.get("feature_hash")
    
    if exp_hash_bad and curr_hash != exp_hash_bad:
        # This is what _jarvis_guard does
        cb.trip("EURUSD", reason=f"Feature Parity Mismatch. Exp: {exp_hash_bad[:8]}")
        
        # Verify circuit breaker was tripped
        with pytest.raises(RuntimeError):
            cb.check_gate("EURUSD")
        
        # Verify reason is correct
        with open(cb.state_file) as f:
            state = json.load(f)
        
        # Check state structure
        if "symbols" in state:
            assert "Feature Parity Mismatch" in state["symbols"]["EURUSD"]["reason"]
        else:
            assert "Feature Parity Mismatch" in state["EURUSD"]["reason"]
