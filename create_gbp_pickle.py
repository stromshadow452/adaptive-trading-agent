import os, joblib
from finrl_stub_gbpusd import StubPolicyG
dst = os.path.join(os.getcwd(),"models","finrl_policies","GBPUSD")
os.makedirs(dst, exist_ok=True)
joblib.dump(StubPolicyG(), os.path.join(dst,"ppo_v1.joblib"))
print("wrote:", os.path.join(dst,"ppo_v1.joblib"))
