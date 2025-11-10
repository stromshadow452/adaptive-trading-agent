#!/usr/bin/env python3
"""
Merge multi-timeframe CSVs from a folder for one symbol into a base-TF dataset.

Example:
  python tools/mtf_merge_folder.py ^
    --folder data/raw/forex_kaggle_multiTF ^
    --symbol EURUSD ^
    --base_tf M15 ^
    --others H1 D1 ^
    --out_dir data/datasets ^
    --indicators ^
    --ffill
"""
import argparse, os, glob
import pandas as pd
import numpy as np

TF_TOL = {
    "M1":"30min","M5":"2h","M15":"4h","M30":"8h",
    "H1":"3h","H4":"12h","D1":"2D","W1":"10D"
}

CORE = ["Datetime","Open","High","Low","Close","Volume","Spread"]

def read_any(path: str) -> pd.DataFrame:
    """Read a TF csv, ensure a single UTC Datetime column, dedupe & sort."""
    df = pd.read_csv(path, low_memory=False)
    # normalize headers & drop duplicate-named columns
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()].copy()

    if "Datetime" in df.columns:
        # Just coerce to UTC; DO NOT insert again
        df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce", utc=True)
    elif "Date" in df.columns and "Time" in df.columns:
        dt = pd.to_datetime(
            df["Date"].astype(str).str.strip() + " " + df["Time"].astype(str).str.strip(),
            errors="coerce", utc=True
        )
        df.insert(0, "Datetime", dt)
        # Optionally drop the raw Date/Time to avoid later confusion
        df = df.drop(columns=["Date","Time"], errors="ignore")
    else:
        # fallback: try first column as datetime
        first = df.columns[0]
        df["Datetime"] = pd.to_datetime(df[first], errors="coerce", utc=True)

    # keep only relevant columns that actually exist
    keep = [c for c in CORE if c in df.columns]
    # plus any other numeric feature columns the file might already have
    extra = [c for c in df.columns if c not in keep and c != "Datetime" and pd.api.types.is_numeric_dtype(df[c])]
    df = df[["Datetime"] + keep[1:] + extra]

    # clean & sort
    df = df.dropna(subset=["Datetime"]).sort_values("Datetime")
    df = df[~df["Datetime"].duplicated(keep="last")].reset_index(drop=True)
    return df

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # EMAs
    for n in (8, 21, 50):
        out[f"ema{n}"] = out["Close"].ewm(span=n, adjust=False).mean()
    # RSI14
    delta = out["Close"].diff()
    gain = (delta.clip(lower=0)).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean().replace(0, np.nan)
    rs = gain / loss
    out["rsi14"] = 100 - (100 / (1 + rs))
    # ATR14
    prev_close = out["Close"].shift(1)
    tr = pd.concat([
        (out["High"] - out["Low"]).abs(),
        (out["High"] - prev_close).abs(),
        (out["Low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    out["atr14"] = tr.ewm(alpha=1/14, adjust=False).mean()
    # Volume features (if available)
    if "Volume" in out.columns:
        out["vol_ema20"] = out["Volume"].ewm(span=20, adjust=False).mean()
        out["vol_ratio"] = out["Volume"] / out["vol_ema20"].replace(0, np.nan)
    # Returns
    out["ret"] = out["Close"].pct_change().fillna(0)
    out["logret"] = np.log(out["Close"]).diff().fillna(0)
    out["ema8_21"] = out["ema8"] - out["ema21"]
    return out

def prefix_cols(df: pd.DataFrame, tf_label: str,
                exclude=("Datetime","Open","High","Low","Close","Volume","Spread")) -> pd.DataFrame:
    out = df.copy()
    for c in list(df.columns):
        if c in exclude:
            continue
        out.rename(columns={c: f"{c}_{tf_label}"}, inplace=True)
    return out

def load_tf(folder: str, symbol: str, tf: str, indicators: bool=False) -> pd.DataFrame:
    patt = os.path.join(folder, f"{symbol}_{tf}.csv")
    matches = glob.glob(patt)
    if not matches:
        raise FileNotFoundError(f"Not found: {patt}")
    df = read_any(matches[0])
    if indicators:
        df = add_indicators(df)
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--base_tf", required=True, help="e.g., M15")
    ap.add_argument("--others", nargs="*", default=["H1","D1"])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--indicators", action="store_true", help="compute indicators per TF before merge")
    ap.add_argument("--ffill", action="store_true", help="forward/backward fill after merge")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    base = load_tf(args.folder, args.symbol, args.base_tf, indicators=args.indicators)
    merged = base.copy()

    for tf in args.others:
        right_df = load_tf(args.folder, args.symbol, tf, indicators=args.indicators)
        tol = pd.Timedelta(TF_TOL.get(tf.upper(), "1D"))
        dfp = prefix_cols(right_df, tf.upper())
        keep = ["Datetime"] + [c for c in dfp.columns if c.endswith(f"_{tf.upper()}")]
        right = dfp[keep].sort_values("Datetime")
        merged = pd.merge_asof(
            merged.sort_values("Datetime"),
            right,
            on="Datetime",
            direction="backward",
            tolerance=tol
        )

    if args.ffill:
        merged = merged.sort_values("Datetime").ffill().bfill()

    out_path = os.path.join(args.out_dir, f"{args.symbol}_{args.base_tf}_multiTF.csv")
    merged.to_csv(out_path, index=False)
    print(f"✅ Saved merged multi-TF: {out_path}")
    num_nan_cols = sum(merged[c].isna().any() for c in merged.columns if c != "Datetime")
    print(f"Rows={len(merged)}  Cols={len(merged.columns)}  NaN_cols={num_nan_cols}")

if __name__ == "__main__":
    main()
