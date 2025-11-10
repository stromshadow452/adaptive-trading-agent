import os, sys, shutil, zipfile
from pathlib import Path
import pandas as pd
from datetime import datetime

# ====== CONFIG ======
PAIRS_FILE = "pairs.txt"          # oka pair per line (no slash)
YEARS = list(range(2003, datetime.utcnow().year + 1))  # HistData coverage (edit avachu)
BASE = Path.cwd()
DL_DIR = BASE / "data_raw_m1"     # downloads
OUT_DIR = BASE / "out"            # final outputs: per TF CSVs
OUT_M1 = BASE / "out_m1_merged"   # merged/normalized M1 per pair
# All timeframes you asked: 1min → 1month
RULES = {
    "M1":  "1T",
    "M5":  "5T",
    "M15": "15T",
    "M30": "30T",
    "H1":  "1H",
    "H4":  "4H",
    "D1":  "1D",
    "W1":  "1W",
    "MN1": "1ME",  # month-end
}
# =====================

DL_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_M1.mkdir(parents=True, exist_ok=True)

def run(cmd: str):
    print(">", cmd)
    code = os.system(cmd)
    if code != 0:
        print(f"⚠️ command failed: {cmd}")

def smart_read_csv(path: Path) -> pd.DataFrame:
    # HistData files are often ; separated, with columns like Date, Time, Open, High, Low, Close, Volume
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        first = f.readline()
    sep = ";" if ";" in first else ","
    df = pd.read_csv(path, sep=sep)
    # unify datetime
    cols = {c.lower(): c for c in df.columns}
    dt = None
    if "datetime" in cols:
        dt = cols["datetime"]
        dt_series = pd.to_datetime(df[dt], errors="coerce", utc=False)
    elif "date" in cols and "time" in cols:
        dt_series = pd.to_datetime(df[cols["date"]] + " " + df[cols["time"]], errors="coerce", utc=False)
    elif "date" in cols:
        dt_series = pd.to_datetime(df[cols["date"]], errors="coerce", utc=False)
    else:
        raise ValueError(f"Datetime columns not found in {path}")
    df["Date"] = dt_series

    # map OHLCV
    def pick(*names):
        for n in names:
            if n.lower() in cols: return cols[n.lower()]
        return None
    o = pick("open","bidopen","Open")
    h = pick("high","bidhigh","High")
    l = pick("low","bidlow","Low")
    c = pick("close","bidclose","Close")
    v = pick("volume","Volume","tickqty","ticks")

    keep = {"Date": df["Date"]}
    if o: keep["Open"] = df[o]
    if h: keep["High"] = df[h]
    if l: keep["Low"]  = df[l]
    if c: keep["Close"]= df[c]
    if v: keep["Volume"]=df[v] if v else 0

    out = pd.DataFrame(keep).dropna(subset=["Date"]).sort_values("Date")
    return out

def unzip_all(root: Path):
    for z in root.rglob("*.zip"):
        try:
            with zipfile.ZipFile(z, "r") as f:
                dest = z.parent
                f.extractall(dest)
            z.unlink(missing_ok=True)
        except Exception as e:
            print(f"⚠️ unzip failed {z}: {e}")

def merge_pair(pair: str) -> Path | None:
    files = sorted(DL_DIR.rglob(f"*{pair}*.csv"))
    if not files:
        return None
    dfs = []
    for f in files:
        try:
            dfs.append(smart_read_csv(f))
        except Exception as e:
            print(f"skip {f.name}: {e}")
    if not dfs:
        return None
    df = pd.concat(dfs, ignore_index=True).dropna().drop_duplicates(subset=["Date"])
    df = df.sort_values("Date").set_index("Date")
    # forward-fill missing OHLC if needed (optional)
    df = df[["Open","High","Low","Close","Volume"]].astype(float)
    out_path = OUT_M1 / f"{pair}_M1.csv"
    df.reset_index().to_csv(out_path, index=False)
    return out_path

def resample_all(pair: str, m1_path: Path):
    df = pd.read_csv(m1_path, parse_dates=["Date"]).set_index("Date")
    pair_dir = OUT_DIR / pair
    pair_dir.mkdir(parents=True, exist_ok=True)
    # Always (re)save normalized M1
    df.reset_index().to_csv(pair_dir / f"{pair}_M1.csv", index=False)

    # Resample OHLCV
    for tf, rule in RULES.items():
        if tf == "M1":
            continue
        o = df["Open"].resample(rule).first()
        h = df["High"].resample(rule).max()
        l = df["Low"].resample(rule).min()
        c = df["Close"].resample(rule).last()
        if "Volume" in df.columns:
            v = df["Volume"].resample(rule).sum()
        else:
            v = pd.Series(index=c.index, dtype=float)
        out = pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c, "Volume": v}).dropna()
        out.reset_index().to_csv(pair_dir / f"{pair}_{tf}.csv", index=False)
        print(f"✅ {pair} {tf}: {len(out):,} rows")

def main():
    pairs = [p.strip().upper().replace("/","") for p in open(PAIRS_FILE, "r").read().splitlines() if p.strip()]
    print(f"Pairs: {pairs}")

    # 1) Download M1 per pair/year via histdata CLI
    # histdata will place files under DL_DIR; we call it per pair/year
    for pair in pairs:
        for y in YEARS:
            # CLI tries both classic and new syntax
            cmd = f'histdata download --pair {pair} --year {y} --period m1 --format csv --out "{DL_DIR / pair / str(y)}"'
            run(cmd)

    # 2) Unzip everything
    unzip_all(DL_DIR)

    # 3) Merge each pair's CSVs into one clean M1
    for pair in pairs:
        merged = merge_pair(pair)
        if not merged:
            print(f"⚠️ No CSVs found for {pair}, skip.")
            continue
        # 4) Resample to all requested TFs
        resample_all(pair, merged)

    print(f"\nAll done ✅\nOutputs:\n- M1 merged: {OUT_M1}\n- All TFs per pair: {OUT_DIR}")

if __name__ == "__main__":
    main()
