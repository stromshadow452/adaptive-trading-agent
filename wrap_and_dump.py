import os, joblib
from finrl_wrapper_module import WrappedPolicy
for s in ["EURUSD","GBPUSD"]:
    p = os.path.join(os.getcwd(),"models","finrl_policies",s,"ppo_v1.joblib")
    if not os.path.exists(p):
        print("missing",p); continue
    m = joblib.load(p)
    wrapped = WrappedPolicy(m)
    joblib.dump(wrapped, p)
    print("wrapped saved:", p)
