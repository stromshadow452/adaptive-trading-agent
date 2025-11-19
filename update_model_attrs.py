import joblib, os, sys
p = "models/fx_bin_19f_thresh55__pack_ok.joblib"
if not os.path.exists(p):
    print("ERROR: model file not found:", p); sys.exit(2)

m = joblib.load(p)
print("BEFORE: type=", type(m), "has_meta=", hasattr(m, "meta"))
# set attributes the executor expects
try:
    setattr(m, "name", "fx_bin_19f_thresh55__pack_ok")
    setattr(m, "version", "0.0.0")
    if not hasattr(m, "meta") or not isinstance(m.meta, dict):
        m.meta = {"timeframe":"M15"}
    else:
        m.meta.setdefault("timeframe","M15")
    # write back
    joblib.dump(m, p)
    print("UPDATED model saved:", p)
    print("AFTER: type=", type(m), "meta=", getattr(m, "meta", None))
except Exception as e:
    print("EXCEPTION:", e)
    raise
