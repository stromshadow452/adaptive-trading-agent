import joblib, pprint, os, json
p = os.path.join(os.getcwd(), "models", "fx_bin_19f_thresh55__pack_ok.joblib")
m = joblib.load(p)
print("TYPE:", type(m))
if isinstance(m, dict):
    print("DICT keys:")
    pprint.pprint(list(m.keys()))
    print("meta ->", m.get("meta"))
    print("timeframe ->", m.get("timeframe"))
else:
    print("has attr timeframe?", getattr(m,"timeframe",None))
    print("has meta attr?", getattr(m,"meta",None))
