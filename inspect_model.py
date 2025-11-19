import joblib,sys,os
p = sys.argv[1] if len(sys.argv)>1 else 'models/fx_bin_19f_thresh55__pack_ok.joblib'
if not os.path.exists(p):
    print('MISSING:',p); sys.exit(2)
m = joblib.load(p)
print('LOADED:', p)
meta = getattr(m,'meta', None) or getattr(m,'metadata', None) or {}
print('TYPE:', type(m))
print('META:', meta)
