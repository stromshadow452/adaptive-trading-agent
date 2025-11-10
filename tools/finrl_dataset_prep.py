# tools/finrl_dataset_prep.py
import os, glob, pandas as pd
from datetime import datetime

SYMBOL = "EURUSD"
TF = "H1"
os.makedirs("data/finrl", exist_ok=True)

# Locate source CSV
cands = sorted(glob.glob(f"data/raw/forex/**/*{SYMBOL}_{TF}_*.csv", recursive=True))
if not cands:
    raise FileNotFoundError("Source CSV not found under data/raw/forex/**/")
src = cands[-1]
print("Using source:", src)

def try_read(path):
    # 1) Let pandas auto-detect delimiter
    try:
        return pd.read_csv(path, engine="python", sep=None)
    except Exception:
        pass
    # 2) Common fallbacks
    for sep in ["\t", ";", ",", r"\s+"]:
        try:
            return pd.read_csv(path, engine="python", sep=sep)
        except Exception:
            continue
    # 3) Last resort: read raw, then split on tabs
    df = pd.read_csv(path, header=None)
    if df.shape[1] == 1:
        df = df[0].str.split("\t", expand=True)
        df.columns = df.iloc[0]
        df = df.iloc[1:].reset_index(drop=True)
    return df

df = try_read(src)

# Normalize column names: strip, lower, remove < >
def norm(c):
    return c.strip().lower().replace("<","").replace(">","").replace(" ", "_")
df.columns = [norm(str(c)) for c in df.columns]
print("Detected columns:", list(df.columns))

# Expect MT5: date, time, open, high, low, close, tickvol/vol/spread
date_col = None
# If already combined date exists
for cand in ["date_time","datetime","timestamp","date"]:
    if cand in df.columns:
        date_col = cand
        break

if date_col is None:
    if "date" in df.columns and "time" in df.columns:
        # Combine <date> + <time>
        df["date"] = pd.to_datetime(df["date"].astype(str).str.strip() + " " +
                                    df["time"].astype(str).str.strip(),
                                    errors="coerce", dayfirst=True)
        date_col = "date"
    else:
        raise ValueError("Need 'date'+'time' or a single datetime column.")

# Map prices
need_prices = ["open","high","low","close"]
missing = [c for c in need_prices if c not in df.columns]
if missing:
    raise ValueError(f"Missing price columns: {missing}")

# Volume: prefer tickvol, else vol, else 0
if "tickvol" in df.columns:
    df["volume"] = pd.to_numeric(df["tickvol"], errors="coerce").fillna(0)
elif "vol" in df.columns:
    df["volume"] = pd.to_numeric(df["vol"], errors="coerce").fillna(0)
elif "volume" in df.columns:
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
else:
    df["volume"] = 0

# Final frame
out = df.rename(columns={date_col: "date"})[["date","open","high","low","close","volume"]].copy()
# Cast numerics
for c in ["open","high","low","close"]:
    out[c] = pd.to_numeric(out[c], errors="coerce")
out = out.dropna().sort_values("date").reset_index(drop=True)

dst = f"data/finrl/{SYMBOL}_{TF}_finrl.csv"
out.to_csv(dst, index=False)
print(f"✅ Saved {dst} | rows={len(out)}")
