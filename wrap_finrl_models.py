import os, joblib
def wrap(sym):
    p = os.path.join(os.getcwd(),"models","finrl_policies",sym,"ppo_v1.joblib")
    m = joblib.load(p)
    class Wrapped:
        def __init__(self,model):
            self.model = model
            self.timeframe = "M15"
            self.meta = {"timeframe":"M15","version":"wrapped_ppo_v1"}
        def act(self,features): return getattr(self.model,"act",lambda f:None)(features)
        def predict(self,X): return getattr(self.model,"predict",lambda X:None)(X)
        def predict_proba(self,X): return getattr(self.model,"predict_proba",lambda X:None)(X)
    joblib.dump(Wrapped(m),p)
    print("wrapped model saved:",p)
for s in ["EURUSD","GBPUSD"]:
    wrap(s)
