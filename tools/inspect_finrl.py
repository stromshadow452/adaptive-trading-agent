#!/usr/bin/env python3
# tools/inspect_finrl.py
import sys, os, glob, traceback

try:
    import joblib
except Exception:
    joblib = None

def _cands(sym, tf):
    pats = [
        f"{sym}_{tf}_policy.joblib",
        f"{sym}_{tf}.joblib",
        f"{sym}_{tf}_ppo.joblib",
        f"{sym}_{tf}_policy.pkl",
        f"{sym}_{tf}_*policy*.joblib",
        f"{sym}_{tf}_*policy*.pkl",
    ]
    return pats

def find_candidates(root, sym, tf):
    out = []
    for pat in _cands(sym, tf) + _cands(sym.lower(), tf.lower()):
        out += glob.glob(os.path.join(root, pat))
    out = sorted(set(out), key=lambda p: os.path.getmtime(p) if os.path.exists(p) else 0, reverse=True)
    return out

def inspect(path, feat_len=19):
    print("== FILE:", path)
    if joblib is None:
        print("  joblib not available in this environment.")
        return
    try:
        raw = joblib.load(path)
    except Exception:
        print("  LOAD FAILED:", traceback.format_exc(limit=1))
        return
    # try extract inner model if joblib payload wrapped in dict
    model = raw.get("model") if isinstance(raw, dict) and "model" in raw else raw
    print("  loaded type:", type(model))
    checks = [
        "predict_proba", "predict_confidence", "predict", "decision_function",
        "act", "predict_action", "forward", "__call__", "policy_output", "last_action"
    ]
    for c in checks:
        print(f"   has {c}: {hasattr(model, c)}")
    # try calling a few methods safely with dummy input
    X = [[0.0] * feat_len]
    for meth in ("predict_proba", "predict_confidence", "predict", "predict_action", "act"):
        fn = getattr(model, meth, None)
        if callable(fn):
            try:
                res = fn(X)
                print(f"   {meth}() -> type={type(res)} sample={str(res)[:200]}")
                break
            except Exception:
                print(f"   {meth}() call failed: {traceback.format_exc(limit=1)}")
    print()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tools/inspect_finrl.py <models_root> [SYMBOL] [TF]")
        sys.exit(1)
    root = sys.argv[1]
    sym = sys.argv[2] if len(sys.argv) > 2 else "EURUSD"
    tf  = sys.argv[3] if len(sys.argv) > 3 else "M15"
    print(f"Searching in: {root}  for symbol={sym} tf={tf}")
    files = find_candidates(root, sym, tf)
    if not files:
        print("No candidates found.")
        sys.exit(0)
    for f in files:
        inspect(f)
