import os, glob, re
import pandas as pd
import numpy as np
from src.data_loader import read_csv, resample_df
from src.indicators import rsi, atr

RAW_DIRS = ["data/master", "data/raw/forex"]
OUT_DIR = "data/datasets"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- helpers ----------
def infer_tf_from_name(name: str) -> str | None:
    m = re.search(r"_(H\d+|M\d+)(?:[ _\.]|$)", name.upper())
    return m.group(1) if m else None

def tf_to_pandas_rule(tf_fx: str) -> str:
    tf = tf_fx.upper()
    u, n = tf[0], int(tf[1:])
    # pandas deprecates 'H'/'T' -> use 'h'/'min'
    return f"{n}h" if u == "H" else f"{n}min"

def infer_symbol_from_name(name: str) -> str | None:
    base = os.path.basename(name).upper()
    m = re.match(r"^([A-Z]{6})_", base)
    if m: return m.group(1)
    m = re.search(r"\b([A-Z]{6})\b", base)
    return m.group(1) if m else None

# ---------- features ----------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ret"]    = df["Close"].pct_change(fill_method=None).fillna(0.0)
    df["logret"] = np.log1p(df["ret"])
    df["ema8"]   = df["Close"].ewm(span=8, adjust=False).mean()
    df["ema21"]  = df["Close"].ewm(span=21, adjust=False).mean()
    df["ema50"]  = df["Close"].ewm(span=50, adjust=False).mean()
    df["rsi14"]  = rsi(df["Close"], period=14)
    df["atr14"]  = atr(df, period=14)
    df["ema8_21"]= df["ema8"] - df["ema21"]
    if "Volume" in df.columns:
        ve20 = df["Volume"].ewm(span=20, adjust=False).mean()
        df["vol_ema20"] = ve20
        df["vol_ratio"] = df["Volume"] / (ve20 + 1e-9)
    else:
        df["vol_ema20"] = 0.0
        df["vol_ratio"] = 0.0
    df = df.dropna()
    df.index.name = "Date"
    return df

def process_file(path: str):
    base = os.path.basename(path)
    try:
        df_raw = read_csv(path)
    except Exception as e:
        print("❌ Read error:", base, e); return

    tf_fx = infer_tf_from_name(base) or "NATIVE"
    try:
        if tf_fx == "NATIVE":
            df_tf = df_raw
            tf_out = "NATIVE"
        else:
            rule = tf_to_pandas_rule(tf_fx)
            df_tf = resample_df(df_raw, rule)
            tf_out = tf_fx
        df_feat = add_features(df_tf)
    except Exception as e:
        print("❌ Feature error:", base, e); return

    if len(df_feat) == 0:
        print(f"⚠️ Skipped (empty after resample/features): {base}")
        return

    sym = infer_symbol_from_name(base) or "UNKNOWN"
    out_name = f"{sym}_{tf_out}_processed.csv"
    out_path = os.path.join(OUT_DIR, out_name)
    df_feat.to_csv(out_path, index=True)
    print(f"✅ Saved: {out_path} (rows={len(df_feat)}, tf={tf_out})")

def main():
    files = []
    for d in RAW_DIRS:
        files += glob.glob(os.path.join(d, "**", "*.csv"), recursive=True)
    if not files:
        print("⚠️ No raw files in", RAW_DIRS); return
    for p in sorted(files):
        process_file(p)

if __name__ == "__main__":
    main()
