import os, pandas as pd

UNIFIED = r"data/unified"

def load_window(symbols, timeframe, years, asset="both"):
    """
    asset: "fx" | "crypto" | "both"
    symbols: list like ["EURUSD","GBPUSD","BTC-USDT","ETH-USDT"]
    timeframe: "5m","15m","1h","4h","1d"
    """
    end = pd.Timestamp.utcnow().tz_localize("UTC")
    start = end - pd.DateOffset(years=years)
    dfs = []
    for sym in symbols:
        dirp = os.path.join(UNIFIED, sym, timeframe.lower())
        if not os.path.isdir(dirp): 
            continue
        for y in range(start.year, end.year+1):
            fp = os.path.join(dirp, f"{y}.parquet")
            if os.path.exists(fp):
                d = pd.read_parquet(fp)
                d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
                dfs.append(d)
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]
    # if 'asset' missing (old forex splits), set fx
    if "asset" not in df.columns:
        df["asset"] = "fx"
    if asset in ("fx","crypto"):
        df = df[df["asset"] == asset]
    df = df.sort_values(["symbol","timestamp"]).drop_duplicates(subset=["symbol","timeframe","timestamp"])
    return df.reset_index(drop=True)
