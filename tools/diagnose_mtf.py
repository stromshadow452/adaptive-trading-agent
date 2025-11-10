#!/usr/bin/env python3
"""
tools/diagnose_mtf.py
Checks: files exist, datetime ok, NaNs/dupes, coverage windows, and a trial merge_asof.
Usage:
  python tools/diagnose_mtf.py ^
    --base data/datasets/EURUSD_M15_processed_clean.csv ^
    --others data/datasets/EURUSD_H1_processed_clean.csv data/datasets/EURUSD_D1_processed.csv ^
    --out data/datasets/EURUSD_M15_multiTF_DIAG.csv
"""
import argparse, os, pandas as pd, numpy as np

TF_TOL = {"M1":"30min","M5":"2h","M15":"4h","M30":"8h","H1":"3h","H4":"12h","D1":"2D","W1":"10D"}

def read_any(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path, low_memory=False)
    cols = [str(c).strip() for c in df.columns]
    df.columns = cols
    # build Datetime if needed
    if "Datetime" not in df.columns:
        if "Date" in df.columns and "Time" in df.columns:
            dt = pd.to_datetime(df["Date"].astype(str).str.strip()+" "+df["Time"].astype(str).str.strip(), errors="coerce", utc=True)
            df.insert(0,"Datetime",dt)
        else:
            first = df.columns[0]
            df.insert(0,"Datetime",pd.to_datetime(df[first], errors="coerce", utc=True))
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce", utc=True)
    df = df.dropna(subset=["Datetime"]).sort_values("Datetime").reset_index(drop=True)
    return df

def infer_tf_label(path):
    base = os.path.basename(path).upper().replace(".CSV","")
    for key in list(TF_TOL.keys())+["M1","M5","M15","M30","H1","H4","D1","W1"]:
        if f"_{key}_" in base or base.endswith(f"_{key}") or base.startswith(f"{key}_"):
            return key
    parts = base.split("_")
    return parts[1] if len(parts)>1 else parts[0]

def prefix_cols(df, tf, exclude=("Datetime","Open","High","Low","Close","Volume","Spread")):
    out = df.copy()
    for c in list(df.columns):
        if c in exclude: 
            continue
        out.rename(columns={c:f"{c}_{tf}"}, inplace=True)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--others", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print("== MTF Diagnose ==")
    print("Base :", args.base)
    print("Others:", args.others)

    base = read_any(args.base)
    print(f"[BASE] rows={len(base)} from {base['Datetime'].min()} → {base['Datetime'].max()}")
    base_dupes = base["Datetime"].duplicated().sum()
    print(f"[BASE] duplicate Datetime rows: {base_dupes}")
    if base_dupes>0:
        print("WARN: duplicates present in base—consider dedup before training.")

    for col in ["Open","High","Low","Close"]:
        if col not in base.columns:
            print(f"WARN: base missing {col}")

    reports = []
    merged = base.copy()
    for path in args.others:
        df = read_any(path)
        tf = infer_tf_label(path)
        tol = pd.Timedelta(TF_TOL.get(tf,"1D"))
        print(f"\n[OTHER:{tf}] file={os.path.basename(path)} rows={len(df)} "
              f"range={df['Datetime'].min()} → {df['Datetime'].max()} tol={tol}")

        # quick quality stats
        num_cols = [c for c in df.columns if c!="Datetime" and pd.api.types.is_numeric_dtype(df[c])]
        na_ratios = {c: float(df[c].isna().mean()) for c in num_cols}
        bad = [k for k,v in na_ratios.items() if v>0.1]
        if bad:
            print(f"INFO: {len(bad)} columns >10% NaN (showing up to 5): {bad[:5]}")

        # prefix & keep
        dfp = prefix_cols(df, tf)
        keep = ["Datetime"]+[c for c in dfp.columns if c.endswith(f"_{tf}")]
        dfk = dfp[keep].sort_values("Datetime")

        # trial merge_asof
        merged = pd.merge_asof(
            merged.sort_values("Datetime"),
            dfk, on="Datetime", direction="backward", tolerance=tol
        )

    # After all merges: report NaNs
    mcols = [c for c in merged.columns if c!="Datetime"]
    nan_after = {c: float(merged[c].isna().mean()) for c in mcols}
    any_nan = sum(v>0 for v in nan_after.values())
    print(f"\n[MERGED] rows={len(merged)} cols={len(merged.columns)} NaN-columns={any_nan}")
    # simple fill preview (not saving filled, just show potential)
    merged_fill = merged.fillna(method="ffill").fillna(method="bfill")
    any_nan_after_fill = sum(merged_fill[c].isna().any() for c in mcols)
    print(f"[MERGED] NaN-columns if ffill+bfill applied: {any_nan_after_fill}")

    # Save diagnostic merged (without fills so you can inspect true state)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    merged.to_csv(args.out, index=False)
    print(f"✅ Saved diagnostic merged CSV: {args.out}")

    # show a small sample of higher-TF columns if present
    sample_cols = ["Datetime","Close"]
    sample_cols += [c for c in merged.columns if c.endswith("_H1")][:5]
    sample_cols += [c for c in merged.columns if c.endswith("_D1")][:5]
    sample_cols = list(dict.fromkeys(sample_cols))  # unique
    print("\n[SAMPLE HEAD]")
    print(merged[sample_cols].head(5).to_string(index=False))

if __name__ == "__main__":
    main()
