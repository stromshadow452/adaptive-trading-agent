import numpy as np
import pandas as pd

def _ensure_lower_feature(df: pd.DataFrame, name: str):
    if name not in df.columns:
        alt = name[0].upper() + name[1:]
        if alt in df.columns:
            df[name] = df[alt]

def simple_backtest(df: pd.DataFrame, config: dict):
    if "Close" not in df.columns:
        if "close" in df.columns:
            df["Close"] = df["close"]
        else:
            return {"total_return": 0.0, "sharpe": 0.0, "winrate": 0.0, "bars": 0}, pd.Series(dtype=float), pd.Series(dtype=float)
    for feat in ["ema8", "ema21"]:
        _ensure_lower_feature(df, feat)
        if feat not in df.columns:
            return {"total_return": 0.0, "sharpe": 0.0, "winrate": 0.0, "bars": 0}, pd.Series(dtype=float), pd.Series(dtype=float)
    core = df[["Close", "ema8", "ema21"]].dropna()
    if len(core) < 2:
        return {"total_return": 0.0, "sharpe": 0.0, "winrate": 0.0, "bars": int(len(core))}, pd.Series(dtype=float), pd.Series(dtype=float)
    sig = (core["ema8"] > core["ema21"]).astype(int) * 2 - 1
    sig = sig.shift(1).fillna(0)
    ret = core["Close"].pct_change().fillna(0.0)
    strat = sig * ret
    equity = (1.0 + strat).cumprod()
    n = len(strat)
    if n > 1:
        delta_seconds = (core.index.to_series().diff().median().total_seconds() or 3600)
        bars_per_day = 86400.0 / max(1.0, delta_seconds)
        ann_factor = bars_per_day * 252
        avg = strat.mean()
        std = strat.std(ddof=0)
        sharpe = float((avg / (std + 1e-12)) * np.sqrt(ann_factor))
    else:
        sharpe = 0.0
    total_return = float(equity.iloc[-1] - 1.0)
    wins = int((strat > 0).sum())
    losses = int((strat < 0).sum())
    winrate = float(wins / max(wins + losses, 1))
    metrics = {
        "total_return": total_return,
        "sharpe": sharpe,
        "winrate": winrate,
        "bars": int(n),
    }
    return metrics, sig, equity
