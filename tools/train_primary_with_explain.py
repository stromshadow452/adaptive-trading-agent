#!/usr/bin/env python3
"""
train_primary_with_explain.py (v1.5.5 — SSOT-aligned, extended adapter)

- Uses Single Source of Truth from src.features.common_features
- No local feature builders.
- Adapter tries these functions (in order) inside common_features:
    1) compute_all_features(df)
    2) compute_features_from_ohlcv(df)
    3) get_feature_frame(df)
    4) build_features(df)
    5) make_features(df)
- Enforces FEATURE_LIST order for X.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime, UTC
from glob import glob
from typing import List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
import hashlib
import datetime as dt

# --- repo path shim (so "src/..." imports work when running from tools/) ---
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
# --------------------------------------------------------------------------

# --- SHARED FEATURES & META GUARD (SSOT) ---
import src.features.common_features as CF
from src.features.common_features import FEATURE_LIST
from src.model_guard import write_model_metadata

# --- MODEL METADATA HELPERS ---
def _meta_path_for_model(model_path: str) -> str:
    base, _ = os.path.splitext(model_path)
    return f"{base}_metadata.json"

def _list_hash(names):
    h = hashlib.sha256()
    joined = "\n".join(names).encode("utf-8")
    h.update(joined)
    return f"sha256:{h.hexdigest()}"

def _tf_to_mt4_token(tf_norm: str) -> str:
    lut = {"5m": "M5", "15m": "M15", "1h": "H1", "4h": "H4", "1d": "D1"}
    return lut.get(tf_norm.lower(), tf_norm.upper())


# Model selection: LightGBM -> XGBoost -> RandomForest
ModelType = None
MODEL_NAME = None
try:
    import lightgbm as lgb  # type: ignore
    ModelType = "lgbm"
    MODEL_NAME = "LightGBM"
except Exception:
    try:
        import xgboost as xgb  # type: ignore
        ModelType = "xgb"
        MODEL_NAME = "XGBoost"
    except Exception:
        from sklearn.ensemble import RandomForestClassifier
        ModelType = "rf"
        MODEL_NAME = "RandomForest"

# SHAP optional
HAVE_SHAP = False
try:
    import shap  # type: ignore
    HAVE_SHAP = True
except Exception:
    pass

# -------------------------
# Global (for label flags from CLI inside gather_data)
# -------------------------
class _ArgsGlobal:
    binary: bool = False
    k_atr: float = 0.75
    proba_thresh: float = 0.5
args_global = _ArgsGlobal()

# -------------------------
# CSV Utilities & Normalization
# -------------------------

def _first_present(colset, lookup):
    for c in colset:
        if c in lookup:
            return lookup[c]
    return None

def read_csv_head(path: str, max_rows: Optional[int]) -> pd.DataFrame:
    """
    Robust delimiter sniffing (comma/semicolon/tab). Cleans BOM, trims whitespace, strips <> from headers.
    """
    df = None
    try:
        df = pd.read_csv(path, nrows=max_rows, sep=None, engine="python")
    except Exception:
        df = None

    if df is None or (len(df.columns) == 1 and isinstance(df.columns[0], str)
                      and ("\t" in df.columns[0] or ";" in df.columns[0])):
        for sep in ["\t", ";", ","]:
            try:
                df2 = pd.read_csv(path, nrows=max_rows, sep=sep, engine="python")
                if len(df2.columns) > 1:
                    df = df2
                    break
            except Exception:
                pass

    if df is None:
        df = pd.read_csv(path, nrows=max_rows)

    df.columns = [
        re.sub(r"[<>]", "", str(c)).replace("\ufeff", "").strip()
        for c in df.columns
    ]
    return df


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize OHLCV columns to: open, high, low, close, volume (lowercase).
    Tries to build a UTC datetime index from [Date, Time] or timestamp column.
    """
    clean_cols = []
    for c in df.columns:
        cc = str(c).replace("\ufeff", "")
        cc = re.sub(r"[<>]", "", cc).strip().lower()
        clean_cols.append((cc, c))
    col_lut = {lc: orig for lc, orig in clean_cols}

    OPEN_ALIASES  = ["open", "o", "bidopen", "askopen"]
    HIGH_ALIASES  = ["high", "h", "bidhigh", "askhigh"]
    LOW_ALIASES   = ["low", "l", "bidlow", "asklow"]
    CLOSE_ALIASES = ["close", "c", "bidclose", "askclose", "last", "price", "closeprice"]
    VOL_ALIASES   = ["volume", "vol", "tickvol", "tick_volume", "tickvolume", "v", "tickvol.", "tick vol"]

    def pick(aliases):
        return _first_present([a for a in aliases if a in col_lut], col_lut)

    src_open  = pick(OPEN_ALIASES)  or col_lut.get("open")
    src_high  = pick(HIGH_ALIASES)  or col_lut.get("high")
    src_low   = pick(LOW_ALIASES)   or col_lut.get("low")
    src_close = pick(CLOSE_ALIASES) or col_lut.get("close")
    src_vol   = pick(VOL_ALIASES)

    missing = [name for name, src in [("open", src_open), ("high", src_high),
                                      ("low", src_low), ("close", src_close)] if src is None]
    if missing:
        raise ValueError(
            f"normalize_ohlcv: Missing required OHLC columns: {missing}; available: {list(df.columns)}"
        )

    out = pd.DataFrame(index=df.index.copy())
    out["open"]  = pd.to_numeric(df[src_open], errors="coerce")
    out["high"]  = pd.to_numeric(df[src_high], errors="coerce")
    out["low"]   = pd.to_numeric(df[src_low], errors="coerce")
    out["close"] = pd.to_numeric(df[src_close], errors="coerce")
    out["volume"] = pd.to_numeric(df[src_vol], errors="coerce") if (src_vol and src_vol in df.columns) else 1.0

    # Build datetime index if possible
    date_col = None
    time_col = None
    for c in df.columns:
        cl = str(c).lower()
        if date_col is None and "date" in cl:
            date_col = c
        elif time_col is None and ("time" in cl or "timestamp" in cl):
            time_col = c
    try:
        if date_col and time_col:
            dtidx = pd.to_datetime(df[date_col].astype(str) + " " + df[time_col].astype(str), errors="coerce", utc=True)
            out.index = dtidx
        elif time_col:
            out.index = pd.to_datetime(df[time_col], errors="coerce", utc=True)
    except Exception:
        pass

    out = out.dropna(subset=["open", "high", "low", "close"])
    return out


# -------------------------
# SSOT Feature Adapter
# -------------------------

def _call_common_features(df_norm: pd.DataFrame) -> pd.DataFrame:
    """
    Call into src.features.common_features using the best available function.
    Tries, in order:
      - compute_all_features
      - compute_features_from_ohlcv
      - get_feature_frame
      - build_features
      - make_features
    """
    for fn_name in (
        "compute_all_features",
        "compute_features_from_ohlcv",
        "get_feature_frame",
        "build_features",
        "make_features",
    ):
        fn = getattr(CF, fn_name, None)
        if callable(fn):
            feat = fn(df_norm)
            if not isinstance(feat, pd.DataFrame):
                raise TypeError(f"common_features.{fn_name} must return a pandas.DataFrame, got {type(feat)}")
            return feat

    available = [n for n in dir(CF) if not n.startswith("_")]
    raise ImportError(
        "SSOT feature function not found in src.features.common_features. "
        "Please implement one of: compute_all_features(df), compute_features_from_ohlcv(df), "
        "get_feature_frame(df), build_features(df), make_features(df). "
        f"Available symbols: {available}"
    )


# -------------------------
# Data Gathering (SSOT features)
# -------------------------

def list_csvs(csv_dir: str, tf_filter: Optional[str], max_files: int, symbols: Optional[List[str]] = None) -> List[str]:
    patterns = [os.path.join(csv_dir, "**", "*.csv"), os.path.join(csv_dir, "*.csv")]
    files = []
    for p in patterns:
        files.extend(glob(p, recursive=True))
    if tf_filter:
        tf_lc = tf_filter.lower()
        files = [f for f in files if tf_lc in os.path.basename(f).lower()]
    if symbols:
        sy_tokens = []
        for s in symbols:
            s1 = s.replace("-", "").upper()
            s2 = s.upper()
            sy_tokens.extend([s1, s2])
        def keep(fname):
            base = os.path.basename(fname).upper()
            base_nodash = base.replace("-", "")
            return any(tok in base or tok in base_nodash for tok in sy_tokens)
        files = [f for f in files if keep(f)]
    files = sorted(dict.fromkeys(files))
    if max_files:
        files = files[:max_files]
    return files


def gather_data(
    csv_dir: str,
    max_files: int,
    horizon: int,
    thr: float,
    max_rows_per_file: Optional[int],
    sample_frac_per_file: Optional[float],
    tf_filter: Optional[str],
    symbols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    files = list_csvs(csv_dir, tf_filter, max_files, symbols)
    if not files:
        raise SystemExit(f"No CSVs found in '{csv_dir}' with filter '{tf_filter}' and symbols={symbols}")

    X_list, y_list, kept = [], [], 0
    print(f"Found {len(files)} csv files")
    for i, fpath in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {fpath}")
        try:
            raw = read_csv_head(fpath, max_rows_per_file)
            print(f"[normalize] columns in file: {list(raw.columns)}")
            if len(raw.columns) == 1:
                print("  !! Skipping file: could not parse columns after delimiter sniff.")
                continue

            df_norm = normalize_ohlcv(raw)
            feat = _call_common_features(df_norm)

            # Ensure ATR% for ATR-adaptive labels
            if "atr14" in feat.columns and "close" in feat.columns and "atr_pct" not in feat.columns:
                feat["atr_pct"] = (feat["atr14"] / feat["close"]).clip(lower=1e-8)

            y = make_labels(
                feat["close"], horizon=horizon, thr=thr,
                atr_pct=feat.get("atr_pct"), k=getattr(args_global, "k_atr", 0.75)
            )

            df = feat.dropna().copy()
            y = y.reindex(df.index).dropna()
            df = df.loc[y.index]

            if getattr(args_global, "binary", False):
                mask = y != 0
                df, y = df.loc[mask], y.loc[mask]
                y = (y == 1).astype(int)

            if sample_frac_per_file and 0.0 < sample_frac_per_file < 1.0:
                n_before = len(df)
                df = safe_sample(df, sample_frac_per_file)
                y = y.loc[df.index]
                print(f"  sampled {len(df)}/{n_before} rows ({sample_frac_per_file*100:.1f}%)")

            try:
                df.index = pd.to_datetime(df.index, utc=True)
                y.index = pd.to_datetime(y.index, utc=True)
            except Exception:
                pass

            df = df[~df.index.duplicated(keep="last")]
            y = y[~y.index.duplicated(keep="last")]

            X_list.append(df)
            y_list.append(y.astype(int))
            kept += 1

        except Exception as e:
            print(f"  !! Skipping file due to error: {e}")
            traceback.print_exc()

    if kept == 0:
        raise SystemExit("No usable files after normalization/feature prep.")

    X_all = pd.concat(X_list, axis=0)
    y_all = pd.concat(y_list, axis=0)

    mask = (~X_all.isna().any(axis=1)) & (~y_all.isna())
    X_all = X_all.loc[mask.index.intersection(X_all.index)]
    y_all = y_all.loc[mask.index.intersection(y_all.index)]
    mask2 = (~X_all.isna().any(axis=1)) & (~y_all.isna())
    X_all = X_all.loc[mask2]
    y_all = y_all.loc[mask2]

    try:
        X_all.index = pd.to_datetime(X_all.index, utc=True)
        y_all.index = pd.to_datetime(y_all.index, utc=True)
    except Exception:
        pass

    order = np.argsort(X_all.index.values)
    X_all = X_all.iloc[order]
    y_all = y_all.iloc[order]

    print(f"Total samples: {len(X_all)} (features: {X_all.shape[1]})")
    cls_counts = y_all.value_counts().to_dict()
    print(f"Label distribution: {cls_counts}")
    return X_all, y_all


# -------------------------
# Labeling & helpers
# -------------------------

def make_labels(close: pd.Series, horizon: int, thr: float,
                atr_pct: Optional[pd.Series] = None, k: float = 0.75) -> pd.Series:
    fwd_close = close.shift(-horizon)
    fwd_ret = (fwd_close - close) / close
    if atr_pct is not None:
        band = (k * atr_pct).reindex(close.index).fillna(thr)
    else:
        band = pd.Series(thr, index=close.index)
    y = pd.Series(0, index=close.index, dtype=int)
    y[fwd_ret > band]  = 1
    y[fwd_ret < -band] = -1
    return y

def safe_sample(df: pd.DataFrame, frac: float) -> pd.DataFrame:
    if frac is None or frac >= 1.0:
        return df
    if frac <= 0.0:
        return df
    return df.sample(frac=frac, random_state=42)


# -------------------------
# Modeling
# -------------------------

@dataclass
class TrainConfig:
    csv_dir: str
    out: str
    max_files: int
    max_rows_per_file: Optional[int]
    sample_frac_per_file: Optional[float]
    horizon: int
    thr: float
    tf_filter: Optional[str]
    shap: bool
    asset: str
    years: int
    tf: str
    symbols: str
    binary: bool
    k_atr: float
    proba_thresh: float


def compute_sample_weights(y: pd.Series) -> np.ndarray:
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.array(sorted(pd.unique(y)))
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y.values)
    w_map = {c: w for c, w in zip(classes, weights)}
    return y.map(w_map).astype(float).values


def train_model(X: pd.DataFrame, y: pd.Series, horizon: int, proba_thresh: float):
    global MODEL_NAME, ModelType
    n = len(X)
    split = int(n * 0.8)
    gap = max(2 * horizon, 40)
    test_start = min(n, split + gap)

    X_train, X_test = X.iloc[:split], X.iloc[test_start:]
    y_train, y_test = y.iloc[:split], y.iloc[test_start:]

    y_is_binary = (y_train.nunique() == 2 and set(pd.unique(y_train)) <= {0,1})
    if not y_is_binary:
        enc_map = {-1: 0, 0: 1, 1: 2}
        y_train_enc = y_train.map(enc_map).astype(int)
        sample_weight_train = compute_sample_weights(y_train_enc)
    else:
        sample_weight_train = compute_sample_weights(y_train)

    if ModelType == "lgbm":
        if y_is_binary:
            train_set = lgb.Dataset(X_train, label=y_train, weight=sample_weight_train)
            params = dict(
                objective="binary",
                learning_rate=0.05,
                num_leaves=63,
                feature_fraction=0.9,
                bagging_fraction=0.9,
                bagging_freq=1,
                min_data_in_leaf=20,
                max_depth=-1,
                metric="binary_logloss",
                verbosity=-1,
                seed=42,
            )
            model = lgb.train(params, train_set, num_boost_round=800)
            y_proba = model.predict(X_test)
            y_pred = pd.Series((y_proba >= proba_thresh).astype(int), index=X_test.index)
        else:
            train_set = lgb.Dataset(X_train, label=y_train, weight=sample_weight_train)
            params = dict(
                objective="multiclass",
                num_class=3,
                learning_rate=0.05,
                num_leaves=63,
                feature_fraction=0.9,
                bagging_fraction=0.9,
                bagging_freq=1,
                min_data_in_leaf=20,
                max_depth=-1,
                metric="multi_logloss",
                verbosity=-1,
                seed=42,
            )
            model = lgb.train(params, train_set, num_boost_round=800)
            y_pred_enc = np.argmax(model.predict(X_test), axis=1)
            dec_map = {0:-1, 1:0, 2:1}
            y_pred = pd.Series([dec_map[int(i)] for i in np.asarray(y_pred_enc)], index=X_test.index)

    elif ModelType == "xgb":
        if y_is_binary:
            import xgboost as xgb  # type: ignore
            model = xgb.XGBClassifier(
                n_estimators=800,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="binary:logistic",
                random_state=42,
                tree_method="hist",
                scale_pos_weight=float((len(y_train)-y_train.sum())/max(1,y_train.sum()))
            )
            model.fit(X_train, y_train)
            y_pred = pd.Series((model.predict_proba(X_test)[:,1] >= proba_thresh).astype(int), index=X_test.index)
        else:
            import xgboost as xgb  # type: ignore
            model = xgb.XGBClassifier(
                n_estimators=800,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="multi:softprob",
                num_class=3,
                random_state=42,
                tree_method="hist",
            )
            model.fit(X_train, y_train, sample_weight=sample_weight_train)
            y_pred_enc = np.argmax(model.predict_proba(X_test), axis=1)
            dec_map = {0:-1, 1:0, 2:1}
            y_pred = pd.Series([dec_map[int(i)] for i in np.asarray(y_pred_enc)], index=X_test.index)

    else:
        from sklearn.ensemble import RandomForestClassifier
        if y_is_binary:
            model = RandomForestClassifier(
                n_estimators=700, max_depth=None, n_jobs=-1, random_state=42, class_weight="balanced_subsample",
            )
            model.fit(X_train, y_train)
            y_pred = pd.Series(model.predict(X_test), index=X_test.index)
        else:
            model = RandomForestClassifier(
                n_estimators=700, max_depth=None, n_jobs=-1, random_state=42, class_weight="balanced_subsample",
            )
            enc_map = {-1:0, 0:1, 1:2}
            y_train_enc = y_train.map(enc_map).astype(int)
            model.fit(X_train, y_train_enc)
            y_pred_enc = model.predict(X_test)
            dec_map = {0:-1, 1:0, 2:1}
            y_pred = pd.Series([dec_map[int(i)] for i in np.asarray(y_pred_enc)], index=X_test.index)

    from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
    y_test_true = y_test
    if isinstance(y_pred, np.ndarray):
        y_pred = pd.Series(y_pred, index=X_test.index)
    common_idx = y_test_true.index.intersection(y_pred.index)
    y_test_true = y_test_true.loc[common_idx]
    y_pred = y_pred.loc[common_idx]

    acc = accuracy_score(y_test_true, y_pred)
    print(f"\nModel: {MODEL_NAME} ({'binary' if y_is_binary else 'ternary'})")
    print(f"Accuracy: {acc:.4f}")
    if y_is_binary:
        print("Confusion matrix:\n", confusion_matrix(y_test_true, y_pred, labels=[0,1]))
        print("\nClassification report:\n", classification_report(y_test_true, y_pred, labels=[0,1], digits=4))
    else:
        print("Confusion matrix:\n", confusion_matrix(y_test_true, y_pred, labels=[-1,0,1]))
        print("\nClassification report:\n", classification_report(y_test_true, y_pred, labels=[-1,0,1], digits=4))

    return model, (X_train, X_test, y_train, y_test)


def run_shap(model, X_train, X_test, out_dir: str, feature_names: List[str]):
    if not HAVE_SHAP:
        print("SHAP not available — skipping.")
        return
    try:
        os.makedirs(out_dir, exist_ok=True)
        print("Computing SHAP values (this may take a bit)…")
        explainer = shap.TreeExplainer(model)
        bg = shap.utils.sample(X_train, 200, random_state=42)
        shap_values = explainer.shap_values(bg)

        import matplotlib.pyplot as plt
        plt.figure()
        if isinstance(shap_values, list):
            sv_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
            order = np.argsort(-sv_abs)
            topk = min(20, len(feature_names))
            top_idx = order[:topk]
            plt.bar(range(topk), sv_abs[top_idx])
            plt.xticks(range(topk), [feature_names[i] for i in top_idx], rotation=45, ha="right")
            plt.title("Global Feature Importance (avg |SHAP|)")
        else:
            shap.summary_plot(shap_values, bg, show=False)
        plt.tight_layout()
        out_png = os.path.join(out_dir, "shap_importance.png")
        plt.savefig(out_png, dpi=160)
        plt.close()
        print(f"Saved SHAP summary: {out_png}")
    except Exception as e:
        print(f"SHAP failed: {e}")
        traceback.print_exc()


# -------------------------
# Main
# -------------------------

def parse_args() -> 'TrainConfig':
    p = argparse.ArgumentParser()
    p.add_argument("--csv_dir", default="data/raw", help="Root folder containing CSV files")
    p.add_argument("--out", required=True, help="Output .joblib path")
    p.add_argument("--max_files", type=int, default=200)
    p.add_argument("--max_rows_per_file", type=int, default=0, help="0 = no cap")
    p.add_argument("--sample_frac_per_file", type=float, default=1.0)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--thr", type=float, default=0.001)
    p.add_argument("--tf_filter", default="", help="Filter filenames by token, e.g., M15")
    p.add_argument("--shap", action="store_true", help="Compute SHAP plots")

    # Added windowing/asset/symbols
    p.add_argument("--asset", choices=["fx","crypto","both"], default="both")
    p.add_argument("--years", type=int, required=True)
    p.add_argument("--tf", default="15m")
    p.add_argument("--symbols", required=True)

    # Binary mode and ATR band
    p.add_argument("--binary", action="store_true", help="Train binary (up vs down), drop neutral (0)")
    p.add_argument("--k_atr", type=float, default=0.75, help="ATR band multiplier for ternary labels")
    p.add_argument("--proba_thresh", type=float, default=0.5, help="Probability threshold for binary predictions")

    a = p.parse_args()
    tf_norm = a.tf.lower().replace("m15","15m").replace("m5","5m").replace("h1","1h").replace("h4","4h").replace("d1","1d")
    return TrainConfig(
        csv_dir=a.csv_dir,
        out=a.out,
        max_files=a.max_files,
        max_rows_per_file=(None if (a.max_rows_per_file or 0) <= 0 else a.max_rows_per_file),
        sample_frac_per_file=(None if a.sample_frac_per_file >= 1.0 else float(a.sample_frac_per_file)),
        horizon=a.horizon,
        thr=a.thr,
        tf_filter=(a.tf_filter if a.tf_filter else None),
        shap=bool(a.shap),
        asset=a.asset,
        years=int(a.years),
        tf=tf_norm,
        symbols=a.symbols,
        binary=bool(a.binary),
        k_atr=float(a.k_atr),
        proba_thresh=float(a.proba_thresh),
    )


def main() -> int:
    global args_global

    args = parse_args()
    args_global.binary = args.binary
    args_global.k_atr = args.k_atr
    args_global.proba_thresh = args.proba_thresh

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    print(f"[{ts}] Starting training")
    print(json.dumps(asdict(args), indent=2))

    if "load_window" in globals():
        print("[patch-check] load_window found; but this script uses CSV scan. (Flow unchanged.)")
    else:
        print("[patch-check] load_window not found; skipping runtime check (flow unchanged).")

    symbols = [s.strip().upper() for s in str(args.symbols).split(",")]
    X, y = gather_data(
        csv_dir=args.csv_dir,
        max_files=args.max_files,
        horizon=args.horizon,
        thr=args.thr,
        max_rows_per_file=args.max_rows_per_file,
        sample_frac_per_file=args.sample_frac_per_file,
        tf_filter=args.tf_filter,
        symbols=symbols,
    )

    # === SSOT order enforcement ===
    missing = [c for c in FEATURE_LIST if c not in X.columns]
    if missing:
        raise SystemExit(f"SSOT mismatch: missing features in computed X: {missing}")
    X = X[FEATURE_LIST]
    feature_names = FEATURE_LIST
    # ==============================

    model, splits = train_model(X, y, args.horizon, args.proba_thresh)
    X_train, X_test, y_train, y_test = splits

    if args.shap:
        out_dir = os.path.join(os.path.dirname(args.out) or ".", "explain")
        os.makedirs(out_dir, exist_ok=True)
        run_shap(model, X_train, X_test, out_dir, feature_names)

    meta = dict(
        created_utc=ts,
        model=MODEL_NAME,
        features=feature_names,
        horizon=args.horizon,
        threshold=args.thr,
        tf_filter=args.tf_filter,
        csv_dir=os.path.abspath(args.csv_dir),
        version="1.5.5",
        label_scheme=("binary_updown" if args.binary else f"ternary_atr_adaptive(k={args.k_atr})"),
        asset=args.asset,
        years=args.years,
        tf=args.tf,
        symbols=symbols,
        embargo_gap=max(2*args.horizon, 40),
        dedup_timestamps=True,
        session_features=True,
        index_enforced_datetime=True,
        alignment="inner_join",
        proba_thresh=args.proba_thresh,
        k_atr=args.katr if hasattr(args, 'katr') else args.k_atr,
    )
    payload = dict(model=model, meta=meta)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    joblib.dump(payload, args.out)
    print(f"Saved model to: {args.out}")

    try:
        _feature_names_sidecar = list(payload.get("meta", {}).get("features", FEATURE_LIST))
        _symbols_sidecar = args.symbols.split(",") if isinstance(args.symbols, str) else args.symbols
        write_model_metadata(
            model_path=args.out,
            model_name=os.path.splitext(os.path.basename(args.out))[0],
            feature_names=_feature_names_sidecar,
            version=datetime.now(UTC).strftime("%Y.%m.%d"),
            extra={
                "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "timeframe": (args.tf_filter or "n/a"),
                "horizon": args.horizon,
                "label_scheme": ("binary_updown" if args.binary else f"ternary_atr_adaptive(k={args.k_atr})"),
                "symbols": _symbols_sidecar,
            },
        )
        print(f"[meta] wrote sidecar: {os.path.splitext(args.out)[0]}_metadata.json")
    except Exception as _e:
        print(f"[meta] sidecar write failed (non-fatal): {_e}")

    write_model_metadata(
        model_path=args.out,
        model_name=os.path.splitext(os.path.basename(args.out))[0],
        feature_names=FEATURE_LIST,
        version=dt.datetime.utcnow().strftime("%Y.%m.%d"),
        extra={
            "target": ("binary" if args.binary else "ternary"),
            "timeframe": (args.tf_filter or args.tf).upper() if args.tf_filter else args.tf.upper(),
            "train_window": f"{args.years}y_pool",
            "proba_thresh": args.proba_thresh,
            "k_atr": args.k_atr,
            "symbols": symbols,
        },
    )
    print(f"[meta] wrote standalone metadata: {os.path.splitext(args.out)[0]}_metadata.json")

    print("Metadata (embedded):\n", json.dumps(meta, indent=2))

    try:
        try:
            start = pd.to_datetime(X.index.min()).date().isoformat()
            end = pd.to_datetime(X.index.max()).date().isoformat()
            train_window = f"{start}..{end}"
        except Exception:
            train_window = None

        standalone_meta = {
            "model_name": os.path.splitext(os.path.basename(args.out))[0],
            "version": dt.datetime.utcnow().strftime("%Y.%m.%d"),
            "target": "binary" if args.binary else "ternary",
            "feature_names": feature_names,
            "feature_order_hash": _list_hash(feature_names),
            "scaler": None,
            "train_window": train_window,
            "timeframe": _tf_to_mt4_token(args.tf),
        }
        meta_path = _meta_path_for_model(args.out)
        with open(meta_path, "w", encoding="utf-8") as w:
            json.dump(standalone_meta, w, ensure_ascii=False, indent=2)
        print(f"[meta] wrote standalone metadata: {meta_path}")
    except Exception as e:
        print(f"[meta] failed to write standalone metadata: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
