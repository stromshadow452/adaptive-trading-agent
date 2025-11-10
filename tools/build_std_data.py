#!/usr/bin/env python3
"""
Merge scattered <SYMBOL>_<TF>* files into standardized CSVs:
  <out_dir>/<SYMBOL>_<TF>.csv with columns: Date, Open, High, Low, Close, Volume

Key features:
- Auto-detect delimiter (comma/semicolon/tab)
- Handle headers like <DATE> <TIME> <OPEN> <HIGH> ...
- Combine DATE + TIME to Date when needed
- Normalize to naive timestamps (no tz) before date filtering
- Accept multiple column name variants for OHLCV
"""

import argparse, sys, re
from pathlib import Path
import pandas as pd

DATE_CANDS  = ["DATE", "Date", "date", "timestamp", "TIME", "time", "Datetime", "datetime"]
TIME_CANDS  = ["TIME", "time"]
OPEN_CANDS  = ["OPEN","Open","open","o","Op"]
HIGH_CANDS  = ["HIGH","High","high","h","Hi"]
LOW_CANDS   = ["LOW","Low","low","l","Lo"]
CLOSE_CANDS = ["CLOSE","Close","close","c","Cl","Adj Close","adj_close","Adj_Close"]
VOL_CANDS   = ["VOLUME","Volume","volume","VOL","vol","Volume USD","tick_volume",
               "TickVolume","Ticks","ticks","TICKVOL","TICK_VOL","TICK_VOLUME","<TICKVOL>","<VOL>"]

def clean_cols(cols):
    out = []
    for c in cols:
        c = str(c).strip()
        # if entire header line stuck in one column (e.g., "<DATE>\t<TIME>...")
        # we'll handle later; for now just keep as is
        c = c.replace("<","").replace(">","")
        out.append(c)
    return out

def read_any(path: Path) -> pd.DataFrame:
    """Try robust reads: auto sep, then '\t' if needed."""
    # First attempt: sep=None (python engine) lets pandas sniff delimiters
    try:
        df = pd.read_csv(path, engine="python", sep=None)
        df.columns = clean_cols(df.columns)
        # If it collapsed into 1 column with embedded tabs, re-read as TSV
        if df.shape[1] == 1 and df.columns[0].count("\t") > 0:
            df = pd.read_csv(path, sep="\t", engine="python")
            df.columns = clean_cols(df.columns)
        return df
    except Exception:
        # Fallback to TSV
        df = pd.read_csv(path, sep="\t", engine="python")
        df.columns = clean_cols(df.columns)
        return df

def find_col(df, cands):
    # exact match
    for n in cands:
        if n in df.columns: return n
        # try uppercase
        if n.upper() in df.columns: return n.upper()
    # relaxed: collapse spaces/underscores/case
    norm = { re.sub(r"[ _]", "", c).lower(): c for c in df.columns }
    for n in cands:
        key = re.sub(r"[ _]", "", n).lower()
        if key in norm: return norm[key]
    return None

def build_datetime(df):
    # Try DATE+TIME first
    dcol = find_col(df, ["DATE","Date","date","timestamp","Datetime","datetime"])
    tcol = find_col(df, TIME_CANDS)

    if dcol is not None and tcol is not None and dcol != tcol:
        dt = pd.to_datetime(df[dcol].astype(str) + " " + df[tcol].astype(str), errors="coerce", utc=True)
        return dt

    # Otherwise single date-like column
    if dcol is None:
        # extremely weird cases where header row was read as a row:
        # try to detect with regex and split
        # but here we'll just fail cleanly; normalize_cols handles skip
        return pd.to_datetime(pd.Series([], dtype="object"))
    dt = pd.to_datetime(df[dcol], errors="coerce", utc=True)
    return dt

def normalize_one(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    # If there is a single column whose name itself contains tabs (header stuck), split columns
    if df.shape[1] == 1 and "\t" in df.columns[0]:
        parts = [c.strip().replace("<","").replace(">","") for c in df.columns[0].split("\t")]
        df = df.iloc[:,0].str.split("\t", expand=True)
        df.columns = parts

    # Clean column names
    df.columns = clean_cols(df.columns)

    # Build Date (UTC then drop tz)
    dt = build_datetime(df)
    df["Date"] = dt
    # Drop rows without valid datetime
    df = df.dropna(subset=["Date"])

    # Convert to naive (remove tz) for safe comparisons
    if getattr(df["Date"].dtype, "tz", None) is not None:
        df["Date"] = df["Date"].dt.tz_convert(None)
    else:
        # when parsed with utc=True, it’s tz-aware; otherwise remains naive
        try:
            df["Date"] = df["Date"].dt.tz_localize(None)
        except Exception:
            pass

    # Map OHLCV
    oc = find_col(df, OPEN_CANDS);  hc = find_col(df, HIGH_CANDS)
    lc = find_col(df, LOW_CANDS);   cc = find_col(df, CLOSE_CANDS)
    vc = find_col(df, VOL_CANDS)

    if oc is None or hc is None or lc is None or cc is None:
        raise ValueError(f"Missing required OHLC column(s). Found: Open={oc}, High={hc}, Low={lc}, Close={cc}")

    out = pd.DataFrame()
    out["Date"]  = df["Date"]
    out["Open"]  = pd.to_numeric(df[oc], errors="coerce")
    out["High"]  = pd.to_numeric(df[hc], errors="coerce")
    out["Low"]   = pd.to_numeric(df[lc], errors="coerce")
    out["Close"] = pd.to_numeric(df[cc], errors="coerce")
    if vc is not None:
        out["Volume"] = pd.to_numeric(df[vc], errors="coerce").fillna(0)
    else:
        out["Volume"] = 0

    out = out.dropna(subset=["Open","High","Low","Close"])
    out = out.sort_values("Date").drop_duplicates("Date")
    return out[["Date","Open","High","Low","Close","Volume"]]

def load_all(symbol, tf, sources):
    frames = []
    pat = re.compile(rf"^{re.escape(symbol)}_{re.escape(tf)}.*\.csv$", re.I)
    for src in sources:
        s = Path(src)
        if not s.exists(): continue
        for p in s.rglob("*.csv"):
            if pat.match(p.name):
                try:
                    df = read_any(p)
                    frames.append((p, df))
                except Exception as e:
                    print(f"⚠️ Read failed {p}: {e}")
    return frames

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--sources", nargs="+", required=True, help="Folders to search")
    ap.add_argument("--out_dir", default="data/raw/forex_kaggle_multiTF")
    ap.add_argument("--tfs", nargs="+", default=["M5","M15","H1","H4","D1"])
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    args = ap.parse_args()

    symbol = args.symbol.upper()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start_ts = pd.to_datetime(args.start) if args.start else None
    end_ts   = pd.to_datetime(args.end)   if args.end   else None

    for tf in args.tfs:
        tfu = tf.upper()
        found = load_all(symbol, tfu, args.sources)
        if not found:
            print(f"⚠️ No files found for {symbol}_{tfu} in sources: {args.sources}")
            continue

        merged = []
        for p, df in found:
            try:
                nd = normalize_one(df)
                merged.append(nd)
            except Exception as e:
                print(f"⚠️ Skipping {p.name}: {e}")

        if not merged:
            print(f"❌ Nothing usable for {symbol}_{tfu}")
            continue

        full = pd.concat(merged, ignore_index=True).sort_values("Date").drop_duplicates("Date")

        # Clip date range (now safe because we removed tz)
        if start_ts is not None:
            full = full[full["Date"] >= start_ts]
        if end_ts is not None:
            full = full[full["Date"] <= end_ts]

        out_path = out_dir / f"{symbol}_{tfu}.csv"
        full.to_csv(out_path, index=False)
        print(f"✅ Wrote {out_path}  rows={len(full)}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
