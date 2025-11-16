import os, joblib, json
p = os.path.join(os.getcwd(), "models", "fx_bin_19f_thresh55__pack_ok.joblib")
try:
    m = joblib.load(p)
    setattr(m,"timeframe","M15")
    joblib.dump(m,p)
    print("patched primary timeframe attr")
except Exception as e:
    # fallback: write meta json alongside model
    meta = {"timeframe":"M15","version":"patched"}
    open(os.path.join(os.getcwd(),"models","fx_bin_19f_thresh55__pack_ok.meta.json"),"w").write(json.dumps(meta))
    print("wrote meta json fallback")
