import os, joblib
p = os.path.join(os.getcwd(),"models","fx_bin_19f_thresh55__pack_ok.joblib")
m = joblib.load(p)
if isinstance(m, dict) and "model" in m:
    inner = m["model"]
    try:
        setattr(inner, "timeframe", "M15")
    except Exception as e:
        pass
    joblib.dump(inner, p)
    print("Replaced joblib with inner model. Done.")
else:
    print("Not a dict-with-model; no action taken.")
