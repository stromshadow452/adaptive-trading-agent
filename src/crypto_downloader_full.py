# duka_downloader.py
import os
import time
import lzma
import struct
import requests
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# =========================================
# ✅ CONFIG
# =========================================
# Missing files list (25 files)
missing_files = {
    
    "USDJPY": ["M15_2020_2021", "M5_2020_2021", "M5_2021_2022", "M5_2022_2023", "M5_2023_2024"],
}

# Dukascopy base URL
BASE_URL = "https://datafeed.dukascopy.com/datafeed"

# Output directory
SAVE_DIR = "data_duka_missing"
os.makedirs(SAVE_DIR, exist_ok=True)

# =========================================
# 🧰 HTTP session with retries
# =========================================
def make_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    # Set a sensible UA to avoid being throttled
    session.headers.update({"User-Agent": "duka-downloader/1.0"})
    return session

SESSION = make_session()

# =========================================
# 📦 Download & Decode
# =========================================
def download_bi5(symbol: str, year: int, month: int, day: int, hour: int) -> bytes | None:
    """Download one .bi5 (hourly ticks). Returns bytes or None."""
    url = f"{BASE_URL}/{symbol}/{year}/{month:02d}/{day:02d}/{hour:02d}h_ticks.bi5"
    try:
        r = SESSION.get(url, timeout=30)
        if r.status_code == 200 and r.content:
            return r.content
        return None
    except requests.RequestException:
        return None

def _price_scale(symbol: str) -> float:
    """Price divisor for tick integers → float."""
    # Most FX pairs: 1e5, JPY-quoted pairs: 1e3
    return 1_000.0 if "JPY" in symbol.upper() else 100_000.0

def decode_bi5(content: bytes | None, year: int, month: int, day: int, hour: int, symbol: str) -> pd.DataFrame | None:
    """Decode .bi5 ticks → DataFrame with columns: timestamp, ask, bid, ask_volume, bid_volume."""
    if not content:
        return None
    try:
        raw = lzma.decompress(content)
    except lzma.LZMAError:
        return None

    # Each record: 20 bytes -> 5 big-endian int32: ms, ask, bid, ask_vol, bid_vol
    rows = []
    scale = _price_scale(symbol)
    base_dt = datetime(year, month, day, hour)

    # iter_unpack gracefully skips any trailing incomplete bytes
    for ms, ask_i, bid_i, ask_vol_i, bid_vol_i in struct.iter_unpack(">5i", raw):
        ts = base_dt + timedelta(milliseconds=ms)
        rows.append([ts, ask_i / scale, bid_i / scale, ask_vol_i, bid_vol_i])

    if not rows:
        return None

    return pd.DataFrame(rows, columns=["timestamp", "ask", "bid", "ask_volume", "bid_volume"])

# =========================================
# ⏱️ Resampling ticks → OHLCV
# =========================================
def ticks_to_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    Convert tick data to OHLCV bars with a mid-price ( (ask+bid)/2 ) OHLC
    and sum of volumes (ask_volume + bid_volume).
    """
    if df.empty:
        return df

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()

    mid = (df["ask"] + df["bid"]) / 2.0
    ohlc = mid.resample(rule).ohlc()
    vol = (df["ask_volume"] + df["bid_volume"]).resample(rule).sum()

    out = ohlc.join(vol.rename("volume")).dropna()
    out = out.reset_index()
    # Ensure naive timestamps in UTC or keep tz-aware; here we write ISO format
    return out

def tf_to_pandas_rule(tf_name: str) -> str | None:
    """Map 'M5' → '5T', 'M15' → '15T'. Return None for unknown (keep ticks)."""
    m = tf_name.upper()
    if m == "M5":
        return "5T"
    if m == "M15":
        return "15T"
    return None

# =========================================
# 📅 Range helpers
# =========================================
def daterange_days(start_year: int, end_year_exclusive: int):
    """
    Iterate days from Jan 1 of start_year up to (but not including) Jan 1 of end_year_exclusive.
    Example: 2020_2021 → all days of 2020.
    """
    start = datetime(start_year, 1, 1)
    end = datetime(end_year_exclusive, 1, 1)
    delta = (end - start).days
    for i in range(delta):
        yield start + timedelta(days=i)

# =========================================
# 🚚 Download a year-range and save
# =========================================
def download_year(symbol: str, tf_name: str, start_year: int, end_year_exclusive: int):
    """
    Download all ticks for the range and resample to tf_name (M5/M15) if applicable.
    Saves to: {SAVE_DIR}/{symbol}_{tf_name}_{start}_to_{end}.csv
    """
    out_path = os.path.join(SAVE_DIR, f"{symbol}_{tf_name}_{start_year}_to_{end_year_exclusive}.csv")

    # Skip if already exists (idempotent)
    if os.path.exists(out_path):
        print(f"⏭️  Exists, skipping: {out_path}")
        return

    all_chunks: list[pd.DataFrame] = []

    days = list(daterange_days(start_year, end_year_exclusive))
    with tqdm(total=len(days), desc=f"{symbol}-{tf_name}-{start_year}", ncols=100) as pbar:
        for d in days:
            # Optional: skip weekends quickly (Dukascopy often has no weekend ticks)
            if d.weekday() >= 5:  # 5=Sat, 6=Sun
                pbar.update(1)
                continue

            for hour in range(24):
                bi5 = download_bi5(symbol, d.year, d.month, d.day, hour)
                if not bi5:
                    continue
                df = decode_bi5(bi5, d.year, d.month, d.day, hour, symbol)
                if df is not None and not df.empty:
                    all_chunks.append(df)

            pbar.update(1)

    if not all_chunks:
        print(f"⚠️ No data for {symbol} {tf_name} {start_year}-{end_year_exclusive}")
        return

    final_df = pd.concat(all_chunks, ignore_index=True)

    # Resample to bars if tf_name is M5/M15
    rule = tf_to_pandas_rule(tf_name)
    if rule:
        final_df = ticks_to_ohlcv(final_df, rule)

    # Save
    final_df.to_csv(out_path, index=False)
    rows = len(final_df)
    print(f"✅ Saved: {out_path}  (rows={rows})")

# =========================================
# 🏁 Main
# =========================================
if __name__ == "__main__":
    started = time.time()
    for symbol, file_list in missing_files.items():
        for f in file_list:
            # Expect patterns like "M5_2020_2021"
            try:
                tf, start, end = f.split("_")
                download_year(symbol, tf, int(start), int(end))
            except Exception as e:
                print(f"❌ Skipping {symbol}:{f} due to error: {e}")

    elapsed = time.time() - started
    print(f"All downloads completed in {elapsed/60:.1f} minutes.")
