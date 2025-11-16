import os, joblib
p = os.path.join(os.getcwd(),"models","finrl_policies","GBPUSD","ppo_v1.joblib")
m = joblib.load(p)
setattr(m,"timeframe","M15")
joblib.dump(m,p)
print("patched",p)
