import os, joblib
from finrl_wrapper_module import WrappedPolicy
ok=[]
# EUR
try:
    from finrl_stub_eurusd import StubPolicy as EStub
    e = EStub()
    wrapped_e = WrappedPolicy(e)
    dst_e = os.path.join(os.getcwd(),"models","finrl_policies","EURUSD")
    os.makedirs(dst_e, exist_ok=True)
    joblib.dump(wrapped_e, os.path.join(dst_e,"ppo_v1.joblib"))
    print("recreated wrapped EUR ppo_v1.joblib")
    ok.append("EURUSD")
except Exception as e:
    print("EUR recreate failed:", e)
# GBP
try:
    from finrl_stub_gbpusd import StubPolicyG as GStub
    g = GStub()
    wrapped_g = WrappedPolicy(g)
    dst_g = os.path.join(os.getcwd(),"models","finrl_policies","GBPUSD")
    os.makedirs(dst_g, exist_ok=True)
    joblib.dump(wrapped_g, os.path.join(dst_g,"ppo_v1.joblib"))
    print("recreated wrapped GBP ppo_v1.joblib")
    ok.append("GBPUSD")
except Exception as e:
    print("GBP recreate failed:", e)
print("done; ok:", ok)
