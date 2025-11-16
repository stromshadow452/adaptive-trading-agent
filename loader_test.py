import os, joblib, json
p = os.path.join(os.getcwd(), "models","finrl_policies","EURUSD","ppo_v1.joblib")
print("path:", p)
print("exists:", os.path.exists(p))
try:
    m = joblib.load(p)
    print("loaded type:", type(m))
    print("has_act:", hasattr(m,"act"))
    if hasattr(m,"act"):
        out = m.act({"dummy":1})
        print("act output:", json.dumps(out))
except Exception as e:
    print("load error:", str(e))
