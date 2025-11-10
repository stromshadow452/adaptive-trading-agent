#!/usr/bin/env python3
"""
Clean duplicate Date/Time columns and ensure a single UTC Datetime column.

Usage:
  python tools/clean_duplicate_date.py \
    --in_csv data/datasets/EURUSD_M15_processed.csv \
    --out_csv data/datasets/EURUSD_M15_processed_clean.csv \
    --tz_utc
"""
import argparse
import os
import pandas as pd

def _dedupe_same_name_columns(cols):
    """Rename duplicate column names (e.g., two 'Date' columns) to keep only the first intact."""
    seen = {}
    new_cols = []
    for c in cols:
        if c not in seen:
            seen[c] = 0
            new_cols.append(c)
        else:
            seen[c] += 1
            new_cols.append(f"{c}__dup{seen[c]}")
    return new_cols

def main():
    ap = argparse.ArgumentParser(description="Normalize CSV to have a single UTC 'Datetime' column.")
    ap.add_argument("--in_csv", required=True, help="input CSV path")
    ap.add_argument("--out_csv", required=True, help="output CSV path")
    ap.add_argument("--tz_utc", action="store_true", help="force Datetime to UTC timezone")
    args = ap.parse_args()

    print("Reading:", args.in_csv)
    df = pd.read_csv(args.in_csv, low_memory=False)

    # Make duplicate header names unique to avoid collisions (e.g., two 'Date' headers)
    df.columns = _dedupe_same_name_columns([str(c).strip() for c in df.columns])

    # If Datetime missing, try to build from Date + Time (prefer the FIRST Date/Time if multiple)
    if "Datetime" not in df.columns:
        # Find first suitable Date & Time columns
        date_col = None
        time_col = None
        for c in df.columns:
            if c.startswith("Date"):
                date_col = c
                break
        for c in df.columns:
            if c.startswith("Time"):
                time_col = c
                break

        if date_col is not None and time_col is not None:
            # Combine Date + Time → Datetime
            dt = pd.to_datetime(
                df[date_col].astype(str).str.strip() + " " + df[time_col].astype(str).str.strip(),
                errors="coerce", utc=True
            )
            df.insert(0, "Datetime", dt)
            # Drop ONLY the date/time columns used to build Datetime
            df = df.drop(columns=[date_col, time_col])
        else:
            # Fallback: try first column as datetime
            first = df.columns[0]
            df.insert(0, "Datetime", pd.to_datetime(df[first], errors="coerce", utc=True))

    # Coerce/standardize Datetime and ensure UTC
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce", utc=True)
    if args.tz_utc:
        # If already UTC, this is a no-op; if tz-naive, it becomes UTC
        df["Datetime"] = df["Datetime"].dt.tz_convert("UTC")

    # Drop any fully duplicated columns that might remain (same name after our dedupe fix won’t happen,
    # but this also removes columns with identical data to others)
    df = df.loc[:, ~df.columns.duplicated()].copy()

    # Clean and sort
    before = len(df)
    df = df.dropna(subset=["Datetime"]).sort_values("Datetime").reset_index(drop=True)
    after = len(df)

    # Save
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"✅ Saved clean CSV: {args.out_csv}  rows={after} (dropped {before - after} rows without valid Datetime)")

if __name__ == "__main__":
    main()
