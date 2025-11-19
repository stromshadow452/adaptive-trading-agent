import os, json, joblib, sys
from pathlib import Path

# Try to import TRAIN_FEATURES_DEFAULT from likely modules; fallback to environment or local file
FEATURES = None
candidates = [
    "TRAIN_FEATURES_DEFAULT",
]
# try common locations
try:
    # try direct import from repo modules
    import tools.features as tf
    FEATURES = getattr(tf, "TRAIN_FEATURES_DEFAULT", None)
except Exception:
    pass

if FEATURES is None:
    # try reading config file
    fconf = Path("features/train_features.json")
    if fconf.exists():
        FEATURES = json.loads(fconf.read_text(encoding="utf-8"))
    else:
        # fallback hard-coded minimal safe list (you must replace with real training features if this is insufficient)
        FEATURES = ["open","high","low","close","volume","atr14","rsi14","ema9","ema21"]
        print("WARN: Using fallback FEATURES. Replace with TRAIN_FEATURES_DEFAULT from your training config for accuracy.", file=sys.stderr)

models_dir = Path("models")
if not models_dir.exists():
    print("No models/ directory found.", file=sys.stderr); sys.exit(2)

for path in models_dir.glob("*.joblib"):
    sidecar = path.with_suffix(path.suffix + "_metadata.json")
    meta = {}
    # try to load existing meta from model if available
    try:
        m = joblib.load(path)
        # try attributes useful for meta
        meta['timeframe'] = getattr(m, "timeframe", None)
        meta['model_name'] = getattr(m, "__class__", type(m)).__name__
    except Exception:
        # if joblib load fails, continue but still write sidecar
        print(f"NOTICE: could not joblib.load {path.name} (skipping model introspect). Will still write sidecar.", file=sys.stderr)
    meta.setdefault("feature_names", FEATURES)
    # compute a simple feature order hash (deterministic)
    import hashlib
    meta['feature_order_hash'] = hashlib.sha256(",".join(meta['feature_names']).encode("utf-8")).hexdigest()
    meta['generated_by'] = "populate_sidecars.py"
    meta['generated_at'] = int(__import__("time").time())
    sidecar.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"WROTE: {sidecar}")
