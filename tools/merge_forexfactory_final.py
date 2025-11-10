# tools/merge_forexfactory_final.py
# -*- coding: utf-8 -*-
"""
Merge Forex Factory monthly CSVs (2007..N) into:
  1) one merged CSV
  2) per-year CSVs

Key points:
- Never default bad dates to "today".
- If a row's date is missing/unparseable, use folder YEAR/MONTH (day=1).
- Time parsing is strict; missing/bad time stays NaT (00:00 used only to build a timestamp).
- Adds impact_level = {Low:1, Medium:2, High:3, Non-Economic:0}.
- Compact, single-line progress bars (tqdm).

Example (PowerShell):
  python tools\\merge_forexfactory_final.py `
    --root_dir "data\\dataset-forexfactory-main" `
    --out_file_full "data\\forexfactory_2007_2025_full.csv" `
    --out_dir_yearly "data\\forexfactory_yearly_csv" `
    --min_year 2007 --max_year 2025
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


# --------- Helpers ---------
MONTH_FILE_RE = re.compile(r"[\\/](?P<year>\d{4})[\\/](?P<month>\d{1,2})[\\/][^\\/]+\.csv$", re.IGNORECASE)

CANON_COLS = {
    "date": "date",
    "time": "time",
    "currency": "currency",
    "impact": "impact",
    "actual": "actual",
    "forecast": "forecast",
    "previous": "previous",
    "event": "event",
}

IMPACT_MAP = {
    "high impact expected": 3,
    "high": 3,
    "high impact": 3,
    "medium impact expected": 2,
    "medium": 2,
    "low impact expected": 1,
    "low": 1,
    "non-economic": 0,
    "none": 0,
    "": 0,
}


def find_month_csvs(root: Path) -> List[Tuple[Path, int, int]]:
    """Find CSVs that live under YEAR/MONTH/filename.csv and return (path, year, month)."""
    out = []
    for p in root.rglob("*.csv"):
        m = MONTH_FILE_RE.search(str(p))
        if not m:
            continue
        year = int(m.group("year"))
        month = int(m.group("month"))
        out.append((p, year, month))
    # sort by year, month, path
    out.sort(key=lambda x: (x[1], x[2], str(x[0]).lower()))
    return out


def read_csv_safe(path: Path) -> pd.DataFrame:
    """Read a CSV tolerantly."""
    try:
        df = pd.read_csv(path, low_memory=False, encoding="utf-8", on_bad_lines="skip")
    except Exception:
        df = pd.read_csv(path, low_memory=False, encoding="latin-1", on_bad_lines="skip")
    # normalize headers
    df.columns = [str(c).strip().lower() for c in df.columns]
    # keep only known columns; if missing, create empty
    kept = {}
    for src, dst in CANON_COLS.items():
        if src in df.columns:
            kept[dst] = df[src]
        else:
            kept[dst] = pd.Series([np.nan] * len(df))
    return pd.DataFrame(kept)


def parse_date_strict(s: str):
    """Try to parse a date string using common formats. Return NaT on failure."""
    s = (str(s) if pd.notna(s) else "").strip()
    if not s:
        return pd.NaT
    # Common FF formats across years
    fmts = [
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%d-%b-%Y",
        "%b %d, %Y",
        "%d %b %Y",
        "%Y/%m/%d",
    ]
    for fmt in fmts:
        try:
            return pd.to_datetime(s, format=fmt, errors="raise")
        except Exception:
            pass
    # last resort: let pandas try (no default-to-today)
    try:
        return pd.to_datetime(s, errors="raise", dayfirst=True)
    except Exception:
        return pd.NaT


def parse_time_strict(s: str):
    """Parse time like '08:30', '10:00 AM'. Return NaT (not 00:00) when unknown."""
    s = (str(s) if pd.notna(s) else "").strip().lower()
    if not s or s in {"all day", "tentative", "n/a", "--"}:
        return pd.NaT
    fmts = ["%H:%M", "%I:%M %p", "%H:%M:%S"]
    for fmt in fmts:
        try:
            return pd.to_datetime(s, format=fmt, errors="raise").time()
        except Exception:
            pass
    return pd.NaT


def build_datetime(date_series: pd.Series, time_series: pd.Series) -> pd.Series:
    """Combine parsed date (Timestamp) and time (datetime.time) into pandas datetime64[ns]."""
    # base = date at midnight
    base = pd.to_datetime(date_series.dt.date.astype("string"), errors="coerce")
    # add time only when present
    t = pd.Series(time_series, index=time_series.index)
    has_time = t.notna()
    # hours/min/sec as integers or 0
    hours = np.where(has_time, [getattr(x, "hour", 0) for x in t], 0)
    mins = np.where(has_time, [getattr(x, "minute", 0) for x in t], 0)
    secs = np.where(has_time, [getattr(x, "second", 0) for x in t], 0)
    delta = pd.to_timedelta(hours, unit="h") + pd.to_timedelta(mins, unit="m") + pd.to_timedelta(secs, unit="s")
    return base + delta


# --------- Main ---------
def main():
    ap = argparse.ArgumentParser(description="Merge Forex Factory monthly CSVs into full + yearly CSVs.")
    ap.add_argument("--root_dir", required=True, help="Root folder that contains YEAR/MONTH/forex_factory.csv")
    ap.add_argument("--out_file_full", required=True, help="Path to write the merged CSV")
    ap.add_argument("--out_dir_yearly", required=True, help="Folder to write per-year CSVs")
    ap.add_argument("--min_year", type=int, default=2007, help="Minimum year to keep")
    ap.add_argument("--max_year", type=int, default=2025, help="Maximum year to keep")
    args = ap.parse_args()

    root = Path(args.root_dir)
    out_full = Path(args.out_file_full)
    out_yearly = Path(args.out_dir_yearly)
    out_yearly.mkdir(parents=True, exist_ok=True)

    print("🔍 Scanning for Forex Factory CSV files...")
    files = find_month_csvs(root)
    print(f"📁 Found {len(files)} monthly CSV files")

    frames = []
    for (path, yr, mo) in tqdm(files, desc="📊 Reading CSVs", ncols=100, leave=False):
        df = read_csv_safe(path)
        # add folder year/month
        df["year"] = yr
        df["month"] = mo
        # strict parsing
        df["date_parsed"] = df["date"].map(parse_date_strict)
        df["time_parsed"] = df["time"].map(parse_time_strict)

        # if date missing, use folder year/month/day=1
        mask_missing = df["date_parsed"].isna()
        if mask_missing.any():
            # Build yyyy-mm-01
            fix = pd.to_datetime(
                pd.DataFrame(
                    {
                        "year": np.where(mask_missing, yr, pd.NaT),
                        "month": np.where(mask_missing, mo, pd.NaT),
                        "day": np.where(mask_missing, 1, pd.NaT),
                    }
                )[mask_missing],
                errors="coerce",
            )
            df.loc[mask_missing, "date_parsed"] = fix.values

        # build datetime
        df["datetime"] = build_datetime(df["date_parsed"], df["time_parsed"])

        # normalize impact_level
        df["impact_level"] = (
            df["impact"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(lambda x: IMPACT_MAP.get(x, 0))
            .astype(int)
        )

        frames.append(df)

    full = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=list(CANON_COLS.values()))

    # Clean columns and order
    full = full[
        [
            "date",
            "time",
            "currency",
            "impact",
            "actual",
            "forecast",
            "previous",
            "event",
            "year",
            "month",
            "datetime",
            "impact_level",
        ]
    ]

    # Filter to requested year range and drop rows with no date_parsed at all
    # First, rebuild reliable date-only from datetime
    full["date_only"] = pd.to_datetime(full["datetime"], errors="coerce").dt.date
    full["keep_year"] = pd.to_datetime(full["datetime"], errors="coerce").dt.year

    before = len(full)
    full = full[full["keep_year"].between(args.min_year, args.max_year, inclusive="both")]
    full = full.drop(columns=["keep_year"])
    dropped = before - len(full)

    # Save merged
    print("💾 Saving merged full CSV...")
    full.to_csv(out_full, index=False)
    print(f"✅ Saved merged file: {out_full}")
    print(f"📊 Total rows: {len(full):,} (dropped: {dropped:,} outside {args.min_year}-{args.max_year} or bad dates)")

    # Write yearly
    if len(full):
        # Use the reliable year extracted from datetime
        full["year_safe"] = pd.to_datetime(full["datetime"], errors="coerce").dt.year
        years = sorted(y for y in full["year_safe"].dropna().astype(int).unique())
        for y in tqdm(years, desc="📆 Writing yearly files", ncols=100, leave=False):
            df_y = full[full["year_safe"] == y].drop(columns=["year_safe"])
            df_y.to_csv(out_yearly / f"forexfactory_{y}.csv", index=False)
        print(f"🎯 Yearly files saved to: {out_yearly}")
    else:
        print("ℹ️ No rows to write to yearly files.")


if __name__ == "__main__":
    pd.options.mode.copy_on_write = True
    main()
