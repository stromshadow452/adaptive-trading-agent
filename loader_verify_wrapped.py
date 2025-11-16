import os, joblib
for s in ["EURUSD","GBPUSD"]:
    p = os.path.join(os.getcwd(),"models","finrl_policies",s,"ppo_v1.joblib")
    print("CHECK", s, "exists:", os.path.exists(p))
    if os.path.exists(p):
        m = joblib.load(p)
        print("TYPE", s, type(m), "TF:", getattr(m,"timeframe",None), "META:", getattr(m,"meta",None))
        print("HAS_ACT", s, hasattr(m,"act"))
        if hasattr(m,"act"):
            print("ACT->", m.act({"dummy":1}))
