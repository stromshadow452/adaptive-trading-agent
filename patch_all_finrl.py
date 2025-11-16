import os, joblib
def patch(sym):
    p = os.path.join(os.getcwd(),"models","finrl_policies",sym,"ppo_v1.joblib")
    if not os.path.exists(p):
        print("missing",p); return
    m = joblib.load(p)
    # prefer attribute, else wrap
    try:
        setattr(m,"timeframe","M15")
        print("patched attr timeframe on",p)
        joblib.dump(m,p)
        return
    except Exception as e:
        pass
    # fallback: wrap
    class Wrapped:
        def __init__(self,model):
            self.model = model
            self.timeframe = "M15"
        def act(self,features):
            return getattr(self.model,"act",lambda f:None)(features)
        def predict(self,X):
            return getattr(self.model,"predict",lambda X:None)(X)
        def predict_proba(self,X):
            return getattr(self.model,"predict_proba",lambda X:None)(X)
    wrapped = Wrapped(m)
    joblib.dump(wrapped,p)
    print("wrapped+patched",p)

for s in ["EURUSD","GBPUSD"]:
    patch(s)
