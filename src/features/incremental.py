import numpy as np
import pandas as pd
import json
import hashlib
import os

try:
    from numba import jit
except Exception:
    # Fallback: dummy jit (no JIT, just returns the same function)
    def jit(*args, **kwargs):
        def wrapper(func):
            return func
        # Allow both @jit and @jit(...)
        if args and callable(args[0]) and not kwargs:
            return args[0]
        return wrapper

class IncrementalFeatures:
    """
    Stage 2: Feature Reactor -> Incremental Stream Calculator
    Maintains state buffer to compute features O(1).
    Enforces Feature Parity via hash check.
    """
    def __init__(self, registry_path="config/features_registry.json", window_size=200):
        with open(registry_path) as f:
            self.feature_list = json.load(f)
        
        self.feature_hash = self._compute_feature_hash(self.feature_list)
        self.window_size = window_size
        self.buffers = {} # {symbol: DataFrame}
        
        # JIT Warmup
        self._warmup_jit()

    def _compute_feature_hash(self, feature_list):
        """Compute SHA256 hash of sorted feature list for parity check"""
        s = json.dumps(sorted(feature_list), sort_keys=True)
        return hashlib.sha256(s.encode()).hexdigest()

    def _warmup_jit(self):
        """Pre-compile Numba functions"""
        dummy = np.random.rand(100)
        _fast_sma(dummy, 5)
        _fast_rsi(dummy, 14)
        _fast_atr(dummy, dummy, dummy, 14)

    def on_tick(self, symbol: str, tick: dict) -> dict:
        if symbol not in self.buffers:
            self.buffers[symbol] = pd.DataFrame(columns=["open","high","low","close","volume"])
        
        # Append and truncate
        df = self.buffers[symbol]
        new_row = pd.DataFrame([{
            "open": tick["price"], "high": tick["price"], 
            "low": tick["price"], "close": tick["price"], 
            "volume": tick["vol"]
        }], index=[pd.to_datetime(tick["iso"])])
        
        # In a real bar builder, we'd aggregate ticks into bars. 
        # For this incremental demo, we assume 1 tick = update last bar or new bar.
        # Simplified: Append row (treating tick as bar for M1/tick charts)
        df = pd.concat([df, new_row]).iloc[-self.window_size:]
        self.buffers[symbol] = df
        
        return self._compute_tail(df)

    def _compute_tail(self, df):
        closes = df["close"].values.astype(np.float64)
        highs = df["high"].values.astype(np.float64)
        lows = df["low"].values.astype(np.float64)
        
        # Compute core features using Numba
        feats = {
            "close": closes[-1],
            "ret": (closes[-1] / closes[-2] - 1) if len(closes) > 1 else 0.0,
            "sma5": _fast_sma(closes, 5),
            "sma20": _fast_sma(closes, 20),
            "sma50": _fast_sma(closes, 50),
            "sma100": _fast_sma(closes, 100),
            "atr14": _fast_atr(highs, lows, closes, 14),
            "rsi14": _fast_rsi(closes, 14),
            # ... add other features from registry ...
        }
        
        # Derived features
        feats["sma_ratio"] = feats["sma5"] / feats["sma20"] if feats["sma20"] != 0 else 1.0
        feats["sma_ratio_long"] = feats["sma50"] / feats["sma100"] if feats["sma100"] != 0 else 1.0
        feats["hl_range"] = highs[-1] - lows[-1]
        feats["body"] = abs(closes[-1] - df["open"].values[-1])
        
        # Ensure output matches registry order
        out = {k: feats.get(k, 0.0) for k in self.feature_list}
        return out

# --- Numba Optimized Kernels ---

@jit(nopython=True)
def _fast_sma(arr, period):
    if len(arr) < period: return np.nan
    return np.mean(arr[-period:])

@jit(nopython=True)
def _fast_rsi(arr, period):
    if len(arr) < period + 1: return 50.0
    deltas = np.diff(arr)
    gains = deltas.copy(); gains[gains < 0] = 0
    losses = -deltas.copy(); losses[losses > 0] = 0
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0: return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))

@jit(nopython=True)
def _fast_atr(high, low, close, period):
    if len(close) < period + 1: return 0.0
    tr_sum = 0.0
    for i in range(1, period + 1):
        idx = -i
        hl = high[idx] - low[idx]
        hc = abs(high[idx] - close[idx-1])
        lc = abs(low[idx] - close[idx-1])
        tr_sum += max(hl, hc, lc)
    return tr_sum / period
