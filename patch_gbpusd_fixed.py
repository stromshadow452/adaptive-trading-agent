import os, joblib, sys

# Minimal stub so pickle finds the class name during load.
class StubPolicyG:
    def __init__(self, *args, **kwargs):
        # keep minimal shape; real object will be populated by pickle
        pass

p = os.path.join(os.getcwd(),"models","finrl_policies","GBPUSD","ppo_v1.joblib")
if not os.path.exists(p):
    print("MISSING", p)
    sys.exit(2)

try:
    m = joblib.load(p)
except Exception as e:
    print("LOAD_FAILED", e)
    sys.exit(3)

# set timeframe and write back
setattr(m, "timeframe", "M15")
joblib.dump(m, p)
print("patched", p)
