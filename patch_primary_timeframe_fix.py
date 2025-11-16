import os, joblib, json

p = os.path.join(os.getcwd(), "models", "fx_bin_19f_thresh55__pack_ok.joblib")
if not os.path.exists(p):
    print("ERROR: primary model not found at", p)
    raise SystemExit(1)

try:
    m = joblib.load(p)
    setattr(m, "timeframe", "M15")
    joblib.dump(m, p)
    print("Patched: set attribute timeframe='M15' on model:", p)
except Exception as e:
    meta = {"timeframe": "M15", "version": "patched"}
    meta_path = os.path.join(os.getcwd(), "models", "fx_bin_19f_thresh55__pack_ok.meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f)
    print("Fallback: wrote meta json ->", meta_path)
