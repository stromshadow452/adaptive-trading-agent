# src/tools/download_twelvedata.py
import os
import time
import requests
from pathlib import Path
from datetime import datetime
from typing import Optional
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

API_URL = "https://api.twelvedata.com/time_series"
API_KEY = os.environ.get("bff8e4956156428da618b11f5c654b0f")  # set this in your environment
DEFAULT_RETRY = 3
DEFAULT_SLEEP = 1.5  # seconds between calls (adjust for rate limits)

def symbol_to_filename(symbol: str) -> str:
    """Normalize symbol to filename-friendly (e.g. EUR/USD -> EURUSD)."""
    return symbol.replace("/", "").replace("-", "").replace(" ", "")

def download_symbol(symbol: str,
                    interval: str = "5min",
                    output_dir: str = "data/raw",
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None,
                    retries: int = DEFAULT_RETRY,
                    wait_between: float = DEFAULT_SLEEP) -> Optional[str]:
    """
    Downloads time series CSV for a symbol from Twelve Data and saves to output_dir.
    Returns path to saved CSV or None on failure.
    """

    if API_KEY is None:
        logger.error("TWELVEDATA_API_KEY not set in environment variables.")
        return None

    params = {
        "symbol": symbol,
        "interval": interval,
        "format": "CSV",
        "apikey": API_KEY,
        # optional: outputsize could be "5000" etc - Twelve Data supports outputsize param
    }
    if start_date:
        params["start_date"] = start_date  # e.g. "2021-01-01"
    if end_date:
        params["end_date"] = end_date

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    fname = f"{symbol_to_filename(symbol)}_{interval}.csv"
    outpath = outdir / fname

    attempt = 0
    while attempt < retries:
        attempt += 1
        try:
            logger.info("Requesting %s interval=%s (attempt %d)", symbol, interval, attempt)
            resp = requests.get(API_URL, params=params, timeout=30)
            if resp.status_code == 200:
                text = resp.text
                if "values" in text.lower() or "datetime" in text.lower() or resp.headers.get("Content-Type","").startswith("text/csv"):
                    # Save CSV
                    outpath.write_text(text)
                    logger.info("Saved %s (%d bytes)", outpath, outpath.stat().st_size)
                    return str(outpath)
                else:
                    logger.warning("Unexpected response content; saving for inspection.")
                    outpath.write_text(resp.text)
                    return str(outpath)
            else:
                logger.warning("HTTP %d: %s", resp.status_code, resp.text[:200])
                # If rate limited (429) or server error, backoff
                if resp.status_code in (429, 500, 502, 503, 504):
                    sleep_t = wait_between * attempt
                    logger.info("Sleeping %.1fs before retry (backoff)", sleep_t)
                    time.sleep(sleep_t)
                else:
                    # for other client errors, don't retry much
                    time.sleep(wait_between)
        except requests.RequestException as e:
            logger.exception("Request failed: %s", e)
            time.sleep(wait_between * attempt)

    logger.error("Failed to download %s after %d attempts", symbol, retries)
    return None

def bulk_download(symbols, interval="5min", output_dir="data/raw", start_date=None, end_date=None, pause=1.5):
    saved = []
    for s in symbols:
        path = download_symbol(s, interval=interval, output_dir=output_dir, start_date=start_date, end_date=end_date)
        saved.append((s, path))
        logger.info("Pausing %.1fs to avoid rate limit", pause)
        time.sleep(pause)
    return saved

if __name__ == "__main__":
    # Example usage:
    # export TWELVEDATA_API_KEY="your_key"
    # python src/tools/download_twelvedata.py
    symbols = ["EUR/USD", "USD/JPY"]  # change to your desired pairs
    # OPTIONAL: provide start/end dates in ISO format (YYYY-MM-DD) to limit range
    start = "2021-01-01"
    end = "2021-12-31"
    bulk_download(symbols, interval="5min", output_dir="data/raw", start_date=start, end_date=end)
