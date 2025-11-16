import os, joblib
for s in ["EURUSD","GBPUSD"]:
    p = os.path.join(os.getcwd(),"models","finrl_policies",s,"ppo_v1.joblib")
    print("check:",s,os.path.exists(p))
    if os.path.exists(p):
        m = joblib.load(p)
        print("type:", type(m), "tf:", getattr(m,"timeframe",None))
        print("has_act:", hasattr(m,"act"))
        if hasattr(m,"act"):
            print("act->", m.act({"dummy":1}))
