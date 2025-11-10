#!/usr/bin/env python3
"""
make_daily_from_any.py
Convert any timeframe OHLCV CSV (M1, M5, M15, H1, etc.) into D1 candles.
Usage:
  python tools/make_daily_from_any.py \
      --in_csv data/raw/forex_kaggle_multiTF/EURUSD_H1.csv \
      --out_csv data/raw/forex_kaggle_multiTF/EURUSD_D1.csv
"""
import argparse, os, pandas as pd

def read_any(path):
    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    if "Datetime" in df.columns:
        dt = pd.to_datetime(df["Datetime"], errors="coerce", utc=True)
    elif "Date" in df.columns and "Time" in df.columns:
        dt = pd.to_datetime(
            df["Date"].astype(str).str.strip() + " " + df["Time"].astype(str).str.strip(),
            errors="coerce", utc=True
        )
    else:
        first = df.columns[0]
        dt = pd.to_datetime(df[first], errors="coerce", utc=True)
    df.insert(0, "Datetime", dt)
    keep = [c for c in ["Datetime","Open","High","Low","Close","Volume","Spread"] if c in df.columns]
    df = df[keep].dropna(subset=["Datetime"]).sort_values("Datetime")
    df = df[~df["Datetime"].duplicated(keep="last")].reset_index(drop=True)
    return df

def to_daily(df):
    agg = {"Open":"first","High":"max","Low":"min","Close":"last"}
    if "Volume" in df.columns: agg["Volume"]="sum"
    if "Spread" in df.columns: agg["Spread"]="mean"
    d1 = (df.set_index("Datetime")
            .resample("1D", label="right", closed="right")
            .agg(agg).dropna().reset_index())
    return d1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    print("Reading:", args.in_csv)
    df = read_any(args.in_csv)
    d1 = to_daily(df)

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    d1.to_csv(args.out_csv, index=False)
    print(f"✅ Saved {args.out_csv}  rows={len(d1)}  range={d1['Datetime'].min()} → {d1['Datetime'].max()}")

if __name__ == "__main__":
    main()
