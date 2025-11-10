#!/usr/bin/env python3
import sys, joblib, os

def main():
    if len(sys.argv) != 3:
        print("Usage: python tools/export_primary_estimator.py <in_payload.joblib> <out_estimator.joblib>")
        sys.exit(2)
    src, dst = sys.argv[1], sys.argv[2]
    obj = joblib.load(src)
    # accept either payload dict or direct estimator
    if isinstance(obj, dict) and "model" in obj:
        est = obj["model"]
    else:
        est = obj
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    joblib.dump(est, dst)
    print(f"Exported estimator -> {dst}")

if __name__ == "__main__":
    sys.exit(main())
