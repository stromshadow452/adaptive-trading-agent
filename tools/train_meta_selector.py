# tools/train_meta_selector.py
"""
Train a small meta-selector that learns to weight technical vs fundamental.
Label: 1 if technical signal led to positive forward return, 0 otherwise.

Usage:
  python tools/train_meta_selector.py --logs reports/daily_logs --out models/meta_selector/meta_selector.joblib
"""

import argparse
import glob
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score
from sklearn.dummy import DummyClassifier
import joblib

FEATURES = [
    "tech_score", "impact", "surprise", "adx", "atr",
    "vol_ratio", "tech_sharpe", "trades"
]

def build_dataset_from_logs(log_dir: str) -> pd.DataFrame:
    files = glob.glob(f"{log_dir}/*_summary.json")
    rows = []
    for f in files:
        # Load summary JSON
        try:
            j = json.load(open(f, "r", encoding="utf-8"))
        except Exception:
            continue

        # Try to load a matching plan to grab raw candidate features
        base = Path(f).stem  # e.g. AUDUSD_H4_paper_20251020_123000_summary
        plan_files = glob.glob(f"{log_dir}/*{base.replace('_summary','')}*.json")
        raw = None
        for pf in plan_files:
            try:
                pj = json.load(open(pf, "r", encoding="utf-8"))
                raw = pj.get("raw_candidate") or pj
                break
            except Exception:
                pass

        tech_score = (raw or {}).get("tech_score", 0)
        impact = (raw or {}).get("impact_score", 0)
        surprise = (raw or {}).get("surprise_norm", 0)
        adx = (raw or {}).get("adx", 0)
        atr = (raw or {}).get("atr", 0)
        vol_ratio = (raw or {}).get("vol_ratio", 0)

        tech_sharpe = j.get("sharpe", 0)
        trades = j.get("bars", 0)

        # Label: did the tech-backed paper run make money?
        label = 1 if j.get("total_return", 0) > 0 else 0

        rows.append({
            "tech_score": tech_score,
            "impact": impact,
            "surprise": surprise,
            "adx": adx,
            "atr": atr,
            "vol_ratio": vol_ratio,
            "tech_sharpe": tech_sharpe,
            "trades": trades,
            "label": label
        })

    df = pd.DataFrame(rows).dropna()
    return df

def save_payload(clf, out_path: str, n: int, classes, counts):
    payload = {
        "model": clf,
        "feature_names": FEATURES,
        "model_type": clf.__class__.__name__,
        "n_samples": int(n),
        "class_counts": {int(c): int(k) for c, k in zip(classes, counts)}
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, out_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="reports/daily_logs", help="folder with *_summary.json")
    ap.add_argument("--out", default="models/meta_selector/meta_selector.joblib", help="output model path")
    ap.add_argument("--test_size", type=float, default=0.2)
    args = ap.parse_args()

    df = build_dataset_from_logs(args.logs)
    if df.empty:
        print("No training data found in", args.logs)
        return

    X = df[FEATURES].astype(float).values
    y = df["label"].astype(int).values

    n = len(y)
    classes, counts = np.unique(y, return_counts=True)
    print(f"Samples: {n} | Classes: {list(classes)} | Counts: {list(counts)}")

    # Edge case: only one sample
    if n < 2:
        print("Only one sample available. Saving a baseline DummyClassifier that always predicts the observed label.")
        clf = DummyClassifier(strategy="constant", constant=int(y[0]))
        clf.fit(X, y)
        save_payload(clf, args.out, n, classes, counts)
        print(f"Saved baseline model to {args.out}")
        return

    # Edge case: single class
    if len(classes) < 2:
        print("Only one class present in labels. Training a DummyClassifier (most_frequent).")
        clf = DummyClassifier(strategy="most_frequent")
        clf.fit(X, y)
        save_payload(clf, args.out, n, classes, counts)
        print(f"Saved baseline model to {args.out}")
        return

    # Small dataset: cross-validation with safe n_splits
    if n < 10:
        print("Small dataset detected. Using cross-validation for validation score.")
        min_class = int(counts.min())
        if min_class >= 2:
            n_splits = min(5, n, min_class)
            if n_splits < 2:
                n_splits = 2
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            cv_name = f"StratifiedKFold(n_splits={n_splits})"
        else:
            n_splits = min(5, n)
            if n_splits < 2:
                print("Too few samples for CV. Fitting on all data without CV.")
                clf = RandomForestClassifier(n_estimators=200, random_state=42)
                clf.fit(X, y)
                save_payload(clf, args.out, n, classes, counts)
                print(f"Saved model to {args.out}")
                return
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            cv_name = f"KFold(n_splits={n_splits})"

        clf = RandomForestClassifier(n_estimators=200, random_state=42)
        scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
        print(f"CV ({cv_name}) accuracy mean±std: {scores.mean():.3f} ± {scores.std():.3f}")
        clf.fit(X, y)  # fit on all for export
        save_payload(clf, args.out, n, classes, counts)
        print(f"Saved model to {args.out}")
        return

    # Normal split path (n >= 10)
    # Ensure at least one test sample
    test_size = max(args.test_size, 1.0 / n)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )
    clf = RandomForestClassifier(n_estimators=300, random_state=42)
    clf.fit(X_train, y_train)
    score = clf.score(X_val, y_val)
    print(f"Validation accuracy: {score:.3f}")

    save_payload(clf, args.out, n, classes, counts)
    print(f"Saved model to {args.out}")

if __name__ == "__main__":
    main()
