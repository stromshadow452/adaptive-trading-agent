# utils/csv_price.py
from __future__ import annotations
import os, glob
import pandas as pd
from typing import Tuple, Optional, List

CANDIDATE_COLS = ["close", "Close", "CLOSE", "bidclose", "askclose", "price", "Price"]

def _pick_close_col(df: pd.DataFrame) -> str:
    for c in CANDIDATE_COLS:
        if c in df.columns:
            return c
    # fallback: last numeric column
    for c in df.columns[::-1]:
        if pd.api.types.is_numeric_dtype(df[c]):
            return c
    raise ValueError("No close-like numeric column found.")

def _read_last_non_nan(path: str) -> Optional[float]:
    try:
        df = pd.read_csv(path)
    except Exception:
        try:
            df = pd.read_parquet(path)
        except Exception:
            return None
    try:
        col = _pick_close_col(df)
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        return float(s.iloc[-1]) if len(s) else None
    except Exception:
        return None

def find_price(symbol: str, csv_dirs: List[str]) -> Tuple[Optional[float], Optional[str]]:
    """
    Returns (price, matched_path) or (None, None). Tries common filename patterns.
    """
    patterns = [
        f"{symbol}_M15_*.csv",
        f"{symbol}_M15.csv",
        f"{symbol}.csv",
        f"{symbol}_*.csv",
        f"*{symbol}*.csv",
    ]
    for d in (csv_dirs or []):
        for pat in patterns:
            for path in sorted(glob.glob(os.path.join(d, "**", pat), recursive=True)):
                px = _read_last_non_nan(path)
                if px is not None:
                    return px, path
    return None, None
