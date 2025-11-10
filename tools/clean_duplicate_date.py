# tools/clean_duplicate_date.py
import pandas as pd
from pathlib import Path
import sys

p = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/datasets/EURUSD_H1_processed.csv")
out = Path(sys.argv[2]) if len(sys.argv) > 2 else p.with_name(p.stem + "_clean.csv")

print("Reading:", p)
df = pd.read_csv(p, dtype=str, low_memory=False)

# strip spaces in column names
df.columns = [c.strip() for c in df.columns]

# If duplicate column names exist, keep first occurrence for each name
if df.columns.duplicated().any():
    cols = []
    seen = set()
    for c in df.columns:
        if c not in seen:
            cols.append(c)
            seen.add(c)
    df = df.loc[:, cols]
    print("Dropped duplicate columns; now columns =", df.columns.tolist())

# If there is a Date + Time column, combine to Datetime
if "Time" in df.columns and "Date" in df.columns and "Datetime" not in df.columns:
    df["Datetime"] = df["Date"].astype(str).str.strip() + " " + df["Time"].astype(str).str.strip()
    # move Datetime to first column
    cols = ["Datetime"] + [c for c in df.columns if c != "Datetime"]
    df = df[cols]
    print("Combined Date+Time -> Datetime")

# Try to parse Datetime index
dt_col = None
for cand in ["Datetime", "Date", "Timestamp"]:
    if cand in df.columns:
        dt_col = cand
        break

if dt_col:
    try:
        df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce", utc=True)
        if df[dt_col].isna().all():
            print("Warning: parsed datetimes are all NaT; leaving index unchanged.")
        else:
            df = df.set_index(dt_col).sort_index()
            print("Set Datetime index from column:", dt_col)
    except Exception as e:
        print("Datetime parse failed:", e)

# Try convert numeric OHLCV columns
for col in ["Open","High","Low","Close","Volume","Spread","ret","logret","ema8","ema21","atr14"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col].str.replace(",","").str.strip(), errors="coerce")

df.to_csv(out, index=(not isinstance(df.index, pd.DatetimeIndex)))
print("Saved cleaned CSV →", out)
