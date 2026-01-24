"""
Model Signing Helper for JARVIS Protocol

Provides HMAC signing functionality for model integrity verification.
"""
import os
import hashlib
import hmac

# Get HMAC key from environment (set by user)
MODEL_HMAC_KEY = os.environ.get("MODEL_HMAC_KEY", "").encode("utf-8")

if not MODEL_HMAC_KEY:
    print("[WARN] MODEL_HMAC_KEY not set in environment. Using default (INSECURE for production).")
    MODEL_HMAC_KEY = b"default_insecure_key_change_me"


def sign_model_bytes(data: bytes) -> tuple[str, str]:
    """
    Sign model bytes with SHA256 and HMAC.
    
    Args:
        data: Raw model bytes
        
    Returns:
        (sha256_hex, hmac_hex) tuple
    """
    # SHA256 hash
    sha256_hash = hashlib.sha256(data).hexdigest()
    
    # HMAC signature
    hmac_sig = hmac.new(MODEL_HMAC_KEY, data, hashlib.sha256).hexdigest()
    
    return sha256_hash, hmac_sig


def compute_feature_hash(feature_names: list[str]) -> str:
    """
    Compute SHA256 hash of feature list.
    
    This MUST match the Primary model's feature hash for parity validation.
    Uses the EXACT same method as executor.py feature_list_hash():
    - Newline-separated feature names (NOT sorted, NOT JSON)
    - SHA256 hash with "sha256:" prefix
    
    Args:
        feature_names: List of feature names IN ORDER
        
    Returns:
        SHA256 hex string with "sha256:" prefix
    """
    # Join with newlines (EXACT same as executor.py)
    txt = "\n".join(feature_names)
    
    # SHA256 hash with prefix
    feature_hash = "sha256:" + hashlib.sha256(txt.encode("utf-8")).hexdigest()
    
    return feature_hash


def debug_verify_model(model_path: str) -> dict:
    """
    Debug function to verify model HMAC and show what's stored vs computed.
    
    Args:
        model_path: Path to .joblib model file
        
    Returns:
        dict with verification results
    """
    import json
    import joblib
    
    result = {
        "model_path": model_path,
        "exists": os.path.exists(model_path),
        "sig_exists": os.path.exists(model_path + ".sig"),
    }
    
    if not result["exists"]:
        result["error"] = "Model file not found"
        return result
    
    # Read model bytes
    with open(model_path, "rb") as f:
        model_bytes = f.read()
    
    # Compute current signatures
    computed_sha256, computed_hmac = sign_model_bytes(model_bytes)
    result["computed_sha256"] = computed_sha256
    result["computed_hmac"] = computed_hmac
    
    # Read stored signatures from .sig file
    sig_path = model_path + ".sig"
    if os.path.exists(sig_path):
        with open(sig_path) as f:
            stored_sigs = json.load(f)
        result["stored_sha256"] = stored_sigs.get("sha256", "N/A")
        result["stored_hmac"] = stored_sigs.get("hmac", "N/A")
        result["sha256_match"] = (result["computed_sha256"] == result["stored_sha256"])
        result["hmac_match"] = (result["computed_hmac"] == result["stored_hmac"])
    else:
        result["error"] = ".sig file not found"
    
    # Read metadata from model
    try:
        payload = joblib.load(model_path)
        meta = payload.get("meta", {})
        result["meta_sha256"] = meta.get("sha256", "N/A")
        result["meta_hmac"] = meta.get("hmac", "N/A")
    except Exception as e:
        result["meta_error"] = str(e)
    
    return result


def print_debug_verification(model_path: str):
    """Print debug verification results in readable format."""
    result = debug_verify_model(model_path)
    
    print("\n" + "=" * 60)
    print("MODEL HMAC DEBUG VERIFICATION")
    print("=" * 60)
    print(f"Model: {result['model_path']}")
    print(f"Exists: {result['exists']}")
    print(f".sig Exists: {result['sig_exists']}")
    
    if "error" in result:
        print(f"\n❌ ERROR: {result['error']}")
        return
    
    print(f"\nCOMPUTED (from current file bytes):")
    print(f"  SHA256: {result['computed_sha256'][:32]}...")
    print(f"  HMAC:   {result['computed_hmac'][:32]}...")
    
    print(f"\nSTORED (in .sig file):")
    print(f"  SHA256: {result.get('stored_sha256', 'N/A')[:32]}...")
    print(f"  HMAC:   {result.get('stored_hmac', 'N/A')[:32]}...")
    
    print(f"\nVERIFICATION:")
    print(f"  SHA256 Match: {'✓ YES' if result.get('sha256_match') else '✗ NO'}")
    print(f"  HMAC Match:   {'✓ YES' if result.get('hmac_match') else '✗ NO'}")
    
    if result.get('hmac_match'):
        print(f"\n✅ HMAC VERIFICATION PASSED!")
    else:
        print(f"\n❌ HMAC MISMATCH! Model may have been modified after signing.")
        print(f"\nTroubleshooting:")
        print(f"  1. Ensure MODEL_HMAC_KEY is the SAME during training and loading")
        print(f"  2. Re-train the model with the correct HMAC key")
        print(f"  3. Don't modify the .joblib file after training")
    
    print("=" * 60 + "\n")
