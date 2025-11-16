import os, joblib
dst = os.path.join(os.getcwd(), "models","finrl_policies","EURUSD")
os.makedirs(dst, exist_ok=True)
class StubPolicy:
    def __init__(self): self.version='stub_v1'
    def act(self, features):
        # deterministic support for ML BUY; small ev/conf/costs
        return {'side':'BUY','er':0.0012,'conf':0.85,'costs':0.0001,'policy_id':self.version}
joblib.dump(StubPolicy(), os.path.join(dst,'ppo_v1.joblib'))
print("wrote", os.path.join(dst,'ppo_v1.joblib'))
