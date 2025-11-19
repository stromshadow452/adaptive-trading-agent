import joblib, traceback, sys
try:
    joblib.load("models/fx_bin_19f_thresh55__pack_ok.joblib")
    print("MODEL LOADED OK")
except Exception:
    traceback.print_exc()
    sys.exit(2)
