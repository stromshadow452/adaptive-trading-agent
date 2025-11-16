import os, joblib
from finrl_stub_eurusd import StubPolicy
dst = os.path.join(os.getcwd(),"models","finrl_policies","EURUSD")
os.makedirs(dst, exist_ok=True)
joblib.dump(StubPolicy(), os.path.join(dst,"ppo_v1.joblib"))
print("wrote:", os.path.join(dst,"ppo_v1.joblib"))
