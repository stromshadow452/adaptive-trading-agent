from typing import Dict, Any
import os, joblib, time, json

def load_finrl_policies(dir_path: str, log) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        import lightgbm  # type: ignore
    except Exception:
        lightgbm = None

    for fn in os.listdir(dir_path):
        if not fn.endswith("_M15_policy.joblib"):
            continue
        fpath = os.path.join(dir_path, fn)
        t0 = time.perf_counter()
        try:
            pol = joblib.load(fpath)
            symbol = fn.split("_")[0]  # EURUSD_M15_policy.joblib -> EURUSD
            out[symbol] = pol
            is_booster = False
            if lightgbm is not None:
                is_booster = isinstance(pol, lightgbm.Booster)
            rec = {
                "stage": "finrl_policy_load",
                "file": fpath,
                "type": str(type(pol)),
                "is_booster": is_booster,
                "lat_ms": round((time.perf_counter() - t0) * 1000, 3)
            }
            log(rec)
            print(json.dumps(rec))
        except Exception as e:
            rec = {
                "stage": "finrl_policy_load",
                "file": fpath,
                "status": "FAIL",
                "error": str(e)[:200]
            }
            log(rec)
            print(json.dumps(rec))
    rec = {"stage": "finrl_policy_summary", "count": len(out)}
    log(rec)
    print(json.dumps(rec))
    return out


