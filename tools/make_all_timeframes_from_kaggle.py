import os
import pandas as pd
from pathlib import Path

# === CONFIG ===
SRC_DIR = Path("data/raw/forex_kaggle_norm")        # normalized M1 CSVs
OUT_DIR = Path("data/raw/forex_kaggle_multiTF")     # output
# Use lowercase 'h' to avoid deprecation; 'D' is fine
TIMEFRAMES = {
    "M5":  "5min",
    "M15": "15min",
    "M30": "30min",
    "H1":  "1h",
    "H4":  "4h",
    "D1":  "1D",
}
# Optional: filter which symbols to run (comma separated). Example: GBPUSD,USDJPY
ONLY = set(s.strip().upper() for s in os.getenv("ONLY", "").split(",") if s.strip())
# If True, skip symbols that already have ALL TFs generated
SKIP_IF_DONE = True
# ===============

OUT_DIR.mkdir(parents=True, exist_ok=True)

def generate_timeframes(src_csv: Path):
    sym = src_csv.stem.split("_")[0].upper()
    if ONLY and sym not in ONLY:
        return

    # If all outputs exist and skipping is enabled, fast exit
    if SKIP_IF_DONE:
        if all((OUT_DIR / f"{sym}_{tf}.csv").exists() for tf in TIMEFRAMES):
            print(f"⏭️  {sym}: all TFs already exist, skipping.")
            return

    print(f"🔄 Processing {sym} ...")
    # Fast read: specific columns + parse dates
    df = pd.read_csv(
        src_csv,
        usecols=["Date", "Open", "High", "Low", "Close", "Volume"],
        parse_dates=["Date"],
        dtype={
            "Open": "float64",
            "High": "float64",
            "Low": "float64",
            "Close": "float64",
            "Volume": "float64",
        },
        engine="c",
    )
    # Clean index
    df = df.dropna(subset=["Date"]).sort_values("Date")
    df = df[~df["Date"].duplicated(keep="last")]
    df = df.set_index("Date")

    # For each TF, do a single resample with .agg (one pass)
    agg = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }

    for tf_name, rule in TIMEFRAMES.items():
        out_path = OUT_DIR / f"{sym}_{tf_name}.csv"
        if out_path.exists():
            # Small optimization: don't recompute if present
            # Comment this if you want to overwrite always
            # print(f"   ↪ {tf_name} exists, skipping.")
            continue
        # One pass downsampling
        tf_df = df.resample(rule, origin="start_day").agg(agg).dropna()
        # Ensure strictly increasing index & no gaps in columns
        tf_df = tf_df[["Open", "High", "Low", "Close", "Volume"]]
        tf_df.to_csv(out_path)
        print(f"✅ Saved {tf_name}: {out_path} ({len(tf_df)} rows)")

def main():
    sources = sorted(SRC_DIR.glob("*_M1.csv"))
    if not sources:
        print(f"⚠️ No M1 CSVs found in {SRC_DIR}")
        return

    # Optional: allow resume from a specific symbol via env RESUME
    resume_from = os.getenv("RESUME")
    resume_hit = resume_from is None

    for src in sources:
        sym = src.stem.split("_")[0].upper()
        if not resume_hit:
            if sym == resume_from.upper():
                resume_hit = True
            else:
                continue
        try:
            generate_timeframes(src)
        except KeyboardInterrupt:
            print("\n⛔ Interrupted by user. Safe to re-run; existing TF files are kept.")
            break
        except Exception as e:
            print(f"❌ {sym}: {e}")

if __name__ == "__main__":
    main()
