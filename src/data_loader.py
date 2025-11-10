from __future__ import annotations
import io
from typing import Optional, Dict
import pandas as pd

_DT_CANDIDATES = ["Datetime", "Timestamp", "Date", "Time"]

def _strip_angle_and_space(name: str) -> str:
    return name.strip().strip("<>").strip()

def _coalesce_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    if not df.columns.duplicated().any():
        return df
    out = {}
    for name in pd.unique(df.columns):
        block = df.loc[:, df.columns == name]
        out[name] = block.bfill(axis=1).iloc[:, 0]
    return pd.DataFrame(out, index=df.index)

def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: _strip_angle_and_space(c) for c in df.columns})
    remap = {}
    for c in list(df.columns):
        lc = c.lower().replace(" ", "")
        if lc == "date":
            remap[c] = "Date"
        elif lc == "time":
            remap[c] = "Time"
        elif lc == "open":
            remap[c] = "Open"
        elif lc == "high":
            remap[c] = "High"
        elif lc == "low":
            remap[c] = "Low"
        elif lc == "close":
            remap[c] = "Close"
        elif lc in ("tickvol", "tickvolume", "ticks", "vol", "volume"):
            remap[c] = "Volume"
        elif lc == "spread":
            remap[c] = "Spread"
    if remap:
        df = df.rename(columns=remap)
    df = _coalesce_duplicate_columns(df)
    if "Date" in df.columns and "Time" in df.columns:
        dt = (df["Date"].astype(str).str.strip() + " " + df["Time"].astype(str).str.strip())
        df.insert(0, "Datetime", dt)
    elif "Datetime" not in df.columns and "Date" in df.columns:
        df.insert(0, "Datetime", df["Date"].astype(str).str.strip())
    return df

def _ensure_dt_index(df: pd.DataFrame, tz_aware: bool = True) -> pd.DataFrame:
    if "Datetime" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce", utc=tz_aware)
        df = df.set_index("Datetime").sort_index()
        return df
    for col in _DT_CANDIDATES:
        if col in df.columns:
            s = pd.to_datetime(df[col], errors="coerce", utc=tz_aware)
            if s.notna().any():
                df = df.assign(**{col: s}).set_index(col).sort_index()
                return df
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="raise", utc=tz_aware)
        df = df.sort_index()
    return df

def read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep=r"[\t,;]+",
        engine="python",
        header=0,
        dtype=str,
        encoding="utf-8",
        encoding_errors="ignore",
    )
    if df.shape[1] == 1 and isinstance(df.columns[0], str) and any(d in df.columns[0] for d in ("\t", ",", ";")):
        content = open(path, "r", encoding="utf-8", errors="ignore").read()
        df = pd.read_csv(io.StringIO(content), sep=r"[\t,;]+", engine="python", header=0, dtype=str)
    if df.empty:
        raise ValueError(f"CSV is empty: {path}")
    df = _standardize_columns(df)
    ordered = [c for c in ["Datetime", "Open", "High", "Low", "Close", "Volume", "Spread"] if c in df.columns]
    df = df[ordered + [c for c in df.columns if c not in ordered]]
    for col in ["Open", "High", "Low", "Close", "Volume", "Spread"]:
        if col in df.columns:
            s = df[col]
            if isinstance(s, pd.DataFrame):
                s = s.bfill(axis=1).iloc[:, 0]
            if getattr(s, "dtype", None) == object:
                s = s.str.replace(",", "", regex=False).str.replace(" ", "", regex=False)
            df[col] = pd.to_numeric(s, errors="coerce")
    df = _ensure_dt_index(df, tz_aware=True)
    return df

def resample_df(df: pd.DataFrame, rule: str, how: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("resample_df: DataFrame must have a DatetimeIndex (use read_csv first).")
    if how is None:
        how = {}
        for col in df.columns:
            cl = col.lower()
            if cl == "open":
                how[col] = "first"
            elif cl == "high":
                how[col] = "max"
            elif cl == "low":
                how[col] = "min"
            elif cl == "close":
                how[col] = "last"
            elif cl in ("volume", "vol", "tickvol"):
                how[col] = "sum"
            else:
                how[col] = "last"
    return df.resample(rule).agg(how).dropna(how="all")
