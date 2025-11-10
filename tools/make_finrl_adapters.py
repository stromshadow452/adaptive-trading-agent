# tools/make_finrl_adapters.py
import os, glob, joblib
from tools.policy_adapter import SklearnPolicyAdapter  # <-- import from module

SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "USDCAD", "EURGBP"]
SRC = "models"
DST = "models/finrl_policies"
TIMEFRAME = "M15"
os.makedirs(DST, exist_ok=True)

def find_src(symbol):
    # Prefer per-symbol first; fall back to your general bins
    candidates = [
        f"{SRC}/fx_bin_{symbol}.joblib",
        f"{SRC}/fx_bin_{symbol.upper()}.joblib",
        f"{SRC}/fx_bin_19f_thresh55.joblib",
        f"{SRC}/fx_bin_19f.joblib",
        f"{SRC}/fx_binary.joblib",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    matches = [p for p in glob.glob(f"{SRC}/fx_bin_*{symbol}*.joblib")]
    return matches[0] if matches else None

def main():
    for sym in SYMBOLS:
        src = find_src(sym)
        if not src:
            print(f"[skip] no source model for {sym}")
            continue
        clf = joblib.load(src)
        adapter = SklearnPolicyAdapter(
            clf, proba_index=1,
            meta={"symbol": sym, "timeframe": TIMEFRAME, "framework": "sklearn"}
        )
        dst = f"{DST}/{sym}_{TIMEFRAME}_policy.joblib"
        joblib.dump(adapter, dst)
        print(f"[ok] wrote {dst}")

if __name__ == "__main__":
    main()
