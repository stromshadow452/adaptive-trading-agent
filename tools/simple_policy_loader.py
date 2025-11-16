import os, joblib

def load_policies(dir_path):
    out = {}
    for f in os.listdir(dir_path):
        if not f.endswith("_policy.joblib"):
            continue
        sym = f.replace("_M15","").replace("_policy.joblib","").upper()
        out[sym] = joblib.load(os.path.join(dir_path, f))
    return out
