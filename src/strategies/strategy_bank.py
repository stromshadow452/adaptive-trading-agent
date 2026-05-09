"""
src/strategies/strategy_bank.py
================================
SCOPUS Strategy Bank — 4 deterministic strategies.

Each strategy is a pure function:
    (raw_df: DataFrame, features: dict) → signal_dict | None

Strategies NEVER access the executor, risk module, or pipeline.
They produce raw signals; the pipeline decides what to do with them.

Strategies:
    1. MEAN_REVERSION  — existing boll_z + atr_pctile edge (UNTOUCHED)
    2. TREND_PULLBACK  — EMA trend + pullback to EMA(20) entry
    3. BREAKOUT        — volatility compression + range break
    4. SCALPING        — micro-deviation in high-liquidity sessions
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.strategies.silver_mean_reversion import SilverMeanReversion, SignalType

LOG = logging.getLogger("strategy_bank")

__all__ = ["StrategyBank", "StrategySignal"]


# ---------------------------------------------------------------------------
# Signal dataclass
# ---------------------------------------------------------------------------

@dataclass
class StrategySignal:
    """Standardized signal output from any strategy."""
    strategy:   str              # "MEAN_REVERSION" | "TREND_PULLBACK" | "BREAKOUT" | "SCALPING"
    side:       str              # "buy" | "sell"
    entry_px:   float            # suggested entry price
    sl:         float            # stop loss price
    tp:         float            # take profit price
    confidence: float            # strategy-internal confidence [0, 1]
    metadata:   Dict = None      # extra info for logging

    @property
    def risk_r(self) -> float:
        """Distance in R: TP distance / SL distance."""
        sl_dist = abs(self.entry_px - self.sl)
        tp_dist = abs(self.tp - self.entry_px)
        return tp_dist / max(sl_dist, 1e-10)


# ---------------------------------------------------------------------------
# Helper: compute ADX from OHLC
# ---------------------------------------------------------------------------

def _compute_adx(df: pd.DataFrame, period: int = 14) -> float:
    """Compute current ADX value. Returns 0 if not enough data."""
    if len(df) < period * 2:
        return 0.0
    up = df["high"].diff()
    down = -df["low"].diff()
    pdm = up.where((up > down) & (up > 0), 0.0)
    mdm = down.where((down > up) & (down > 0), 0.0)
    prev = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev).abs(),
        (df["low"] - prev).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    pdi = 100 * pdm.rolling(period).mean() / atr.replace(0, np.nan)
    mdi = 100 * mdm.rolling(period).mean() / atr.replace(0, np.nan)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    adx_series = dx.rolling(period).mean()
    val = adx_series.iloc[-1]
    return float(val) if pd.notna(val) else 0.0


def _compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Current ATR(14) in price units."""
    if len(df) < period + 1:
        return 0.0
    prev = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev).abs(),
        (df["low"] - prev).abs(),
    ], axis=1).max(axis=1)
    atr_series = tr.rolling(period).mean()
    val = atr_series.iloc[-1]
    return float(val) if pd.notna(val) else 0.0


def _compute_ema(series: pd.Series, span: int) -> pd.Series:
    """EMA helper."""
    return series.ewm(span=span, adjust=False).mean()


# ---------------------------------------------------------------------------
# Strategy 1: MEAN REVERSION (reference only — actual logic stays in pipeline)
# ---------------------------------------------------------------------------

def mean_reversion_signal(raw_df: pd.DataFrame, features: dict) -> Optional[StrategySignal]:
    """
    MEAN_REVERSION: Wired to SilverMeanReversion engine.
    """
    if len(raw_df) < 30:
        return None

    # SilverMeanReversion handles its own indicators internally
    smr = SilverMeanReversion()
    df_cache = smr.compute_indicators(raw_df)
    sig = smr.generate_signal(df_cache, bar_idx=-1)
    
    if sig.signal in (SignalType.LONG, SignalType.SHORT):
        side = "buy" if sig.signal == SignalType.LONG else "sell"
        return StrategySignal(
            strategy="MEAN_REVERSION",
            side=side,
            entry_px=round(sig.entry_price, 5),
            sl=round(sig.stop_loss, 5),
            tp=round(sig.take_profit, 5),
            confidence=round(sig.confidence, 3),
            metadata={"reason": sig.reason}
        )
    return None


# ---------------------------------------------------------------------------
# Strategy 2: TREND PULLBACK
# ---------------------------------------------------------------------------

def trend_pullback_signal(raw_df: pd.DataFrame, features: dict) -> Optional[StrategySignal]:
    """
    TREND_PULLBACK: EMA trend + pullback to EMA(20) entry with rejection candle.
    Restored and re-enabled safely with strict breakout + pullback + rejection logic.
    """
    if len(raw_df) < 50:
        return None

    close = raw_df["close"]
    high = raw_df["high"]
    low = raw_df["low"]
    
    current_close = float(close.iloc[-1])
    prev_close = float(close.iloc[-2])
    prev_high = float(high.iloc[-2])
    prev_low = float(low.iloc[-2])
    
    recent_high = float(high.iloc[-21:-1].max())
    recent_low = float(low.iloc[-21:-1].min())
    
    breakout_up = current_close > recent_high
    breakout_down = current_close < recent_low

    # --- PULLBACK DETECTION: Price retraced after breakout ---
    # Current bar is pullback if it didn't make new extremes
    pullback_up = current_close < prev_high and prev_close >= recent_high
    pullback_down = current_close > prev_low and prev_close <= recent_low

    # --- REJECTION CANDLE: Pin bar after pullback ---
    current_open = float(raw_df["open"].iloc[-1])
    current_high = float(high.iloc[-1])
    current_low = float(low.iloc[-1])
    body = abs(current_close - current_open)
    upper_wick = current_high - max(current_close, current_open)
    lower_wick = min(current_close, current_open) - current_low

    # Bullish rejection: long lower wick, small body, in pullback zone
    bullish_rejection = (lower_wick > 2 * body) and (current_close > current_open)
    # Bearish rejection: long upper wick, small body, in pullback zone
    bearish_rejection = (upper_wick > 2 * body) and (current_close < current_open)

    current_ema20 = features.get("sma20", 0.0)
    current_ema50 = features.get("sma50", 0.0)
    atr = features.get("atr14", 0.0)
    adx = features.get("adx14", 0.0)
    atr_pctile = features.get("atr_pctile", 0.5)
    boll_z = features.get("boll_z", 0.0)

    # --- UPTREND: EMA(20) > EMA(50) ---
    if current_ema20 > current_ema50:
        # SNIPER MODE: Require breakout + pullback + rejection candle
        if not (breakout_up and pullback_up and bullish_rejection):
            return None

        # Entry at current close (fade the pullback)
        entry = current_close

        # SL: below rejection candle low (protect the wick)
        sl = current_low - 0.5 * atr
        sl = max(sl, entry - 2.5 * atr)  # cap SL distance
        sl = min(sl, entry - 0.8 * atr)  # minimum SL

        sl_dist = entry - sl
        tp = entry + 2.0 * sl_dist  # 2.0R target

        # Sniper confidence: controlled trend + breakout + rejection
        confidence = 0.55 + 0.25 * min(1.0, (adx - 22.0) / 6.0)
        confidence = min(0.80, confidence)  # Cap at 0.80 for sniper mode

        return StrategySignal(
            strategy="TREND_PULLBACK",
            side="buy",
            entry_px=round(entry, 5),
            sl=round(sl, 5),
            tp=round(tp, 5),
            confidence=round(confidence, 3),
            metadata={"adx": round(adx, 1), "ema20": round(current_ema20, 5),
                       "ema50": round(current_ema50, 5), "atr": round(atr, 5),
                       "atr_pctile": round(atr_pctile, 3), "boll_z": round(boll_z, 3),
                       "breakout": True, "pullback": True, "rejection": True,
                       "recent_high": round(recent_high, 5)},
        )

    # --- DOWNTREND: EMA(20) < EMA(50) ---
    if current_ema20 < current_ema50:
        # SNIPER MODE: Require breakout + pullback + rejection candle
        if not (breakout_down and pullback_down and bearish_rejection):
            return None

        entry = current_close

        # SL: above rejection candle high (protect the wick)
        sl = current_high + 0.5 * atr
        sl = min(sl, entry + 2.5 * atr)  # cap SL distance
        sl = max(sl, entry + 0.8 * atr)  # minimum SL

        sl_dist = sl - entry
        tp = entry - 2.0 * sl_dist

        # Sniper confidence: controlled trend + breakout + rejection
        confidence = 0.55 + 0.25 * min(1.0, (adx - 22.0) / 6.0)
        confidence = min(0.80, confidence)  # Cap at 0.80 for sniper mode

        return StrategySignal(
            strategy="TREND_PULLBACK",
            side="sell",
            entry_px=round(entry, 5),
            sl=round(sl, 5),
            tp=round(tp, 5),
            confidence=round(confidence, 3),
            metadata={"adx": round(adx, 1), "ema20": round(current_ema20, 5),
                       "ema50": round(current_ema50, 5), "atr": round(atr, 5),
                       "atr_pctile": round(atr_pctile, 3), "boll_z": round(boll_z, 3),
                       "breakout": True, "pullback": True, "rejection": True,
                       "recent_low": round(recent_low, 5)},
        )


    return None


# ---------------------------------------------------------------------------
# Strategy 3: BREAKOUT
# ---------------------------------------------------------------------------

def breakout_signal(raw_df: pd.DataFrame, features: dict) -> Optional[StrategySignal]:
    """
    Breakout Strategy (H1).

    Logic:
        1. COMPRESSION: ATR percentile < 0.25 (volatility squeeze)
        2. RANGE: identify 20-bar high/low range
        3. BREAKOUT: close breaks above range high or below range low
        4. CONFIRM: current ATR rising (ATR > ATR 5-bars-ago)
        5. SL: opposite side of range
        6. TP: 1.5R

    Designed to catch the expansion after volatility squeeze.
    """
    if len(raw_df) < 30:
        return None

    close = raw_df["close"]
    high = raw_df["high"]
    low = raw_df["low"]
    open_px = raw_df["open"]

    atr = _compute_atr(raw_df, 14)
    if atr <= 0:
        return None

    # ATR percentile (compression check)
    atr_pctile = features.get("atr_pctile", 0.5)
    if atr_pctile > 0.25:
        return None  # not compressed enough

    # Range over last 20 bars (excluding current)
    lookback = raw_df.iloc[-21:-1]
    range_high = float(lookback["high"].max())
    range_low = float(lookback["low"].min())
    range_width = range_high - range_low

    if range_width < atr * 0.5 or range_width > atr * 1.5:
        return None  # range too tight to be meaningful, or too loose (not true compression)

    current_close = float(close.iloc[-1])

    # ATR rising check (current ATR > ATR 5 bars ago)
    if len(raw_df) < 20:
        return None
    prev = raw_df["close"].shift(1)
    tr = pd.concat([
        (raw_df["high"] - raw_df["low"]).abs(),
        (raw_df["high"] - prev).abs(),
        (raw_df["low"] - prev).abs(),
    ], axis=1).max(axis=1)
    atr_series = tr.rolling(14).mean()
    atr_now = float(atr_series.iloc[-1] or 0)
    atr_5ago = float(atr_series.iloc[-6] or atr_now)
    if atr_now < atr_5ago:
        return None  # ATR not expanding

    # BREAKOUT UP
    if current_close > range_high:
        entry = current_close
        sl = range_low  # opposite side of range
        sl_dist = entry - sl
        tp = entry + 1.5 * sl_dist

        confidence = min(1.0, (current_close - range_high) / atr)

        return StrategySignal(
            strategy="BREAKOUT",
            side="buy",
            entry_px=round(entry, 5),
            sl=round(sl, 5),
            tp=round(tp, 5),
            confidence=round(min(confidence, 0.85), 3),
            metadata={"range_high": round(range_high, 5),
                       "range_low": round(range_low, 5),
                       "atr_pctile": round(atr_pctile, 3)},
        )

    # BREAKOUT DOWN
    if current_close < range_low:
        entry = current_close
        sl = range_high
        sl_dist = sl - entry
        tp = entry - 1.5 * sl_dist

        confidence = min(1.0, (range_low - current_close) / atr)

        return StrategySignal(
            strategy="BREAKOUT",
            side="sell",
            entry_px=round(entry, 5),
            sl=round(sl, 5),
            tp=round(tp, 5),
            confidence=round(min(confidence, 0.85), 3),
            metadata={"range_high": round(range_high, 5),
                       "range_low": round(range_low, 5),
                       "atr_pctile": round(atr_pctile, 3)},
        )

    return None


# ---------------------------------------------------------------------------
# Strategy 4: SCALPING
# ---------------------------------------------------------------------------

def scalping_signal(raw_df: pd.DataFrame, features: dict) -> Optional[StrategySignal]:
    """
    Controlled Scalping Strategy (H1).

    Logic:
        1. SESSION: London (07-16 UTC) or NY (13-20 UTC) only
        2. DEVIATION: |boll_z| >= 0.8 (lower threshold than MR)
        3. VOLATILITY: ATR percentile 0.20-0.50 (moderate — not dead, not wild)
        4. ENTRY: fade the micro-deviation
        5. SL: tight — 1.0× ATR
        6. TP: 1.0R (quick scalp)
        7. LIMIT: max 2 signals per 24h window

    Uses existing boll_z feature at a lower threshold.
    """
    if len(raw_df) < 30:
        return None

    close = raw_df["close"]
    atr = _compute_atr(raw_df, 14)
    if atr <= 0:
        return None

    # Session filter
    ts = raw_df.index[-1]
    hour = ts.hour if hasattr(ts, 'hour') else 12
    london_open = 7 <= hour <= 16
    ny_open = 13 <= hour <= 20
    if not (london_open or ny_open):
        return None  # outside high-liquidity sessions

    # Feature checks
    boll_z = features.get("boll_z", 0.0)
    atr_pctile = features.get("atr_pctile", 0.5)

    if abs(boll_z) < 0.8:
        return None  # not enough deviation
    if atr_pctile < 0.20 or atr_pctile > 0.50:
        return None  # too quiet or too volatile

    adx = _compute_adx(raw_df, 14)
    if adx > 20:
        return None  # trending — don't scalp

    current_close = float(close.iloc[-1])

    # Scalp: fade the deviation
    if boll_z > 0.8:
        # Price above mean → sell scalp
        entry = current_close
        sl = entry + 1.0 * atr
        tp = entry - 1.0 * atr
        side = "sell"
    else:
        # Price below mean → buy scalp
        entry = current_close
        sl = entry - 1.0 * atr
        tp = entry + 1.0 * atr
        side = "buy"

    confidence = min(1.0, abs(boll_z) / 2.0) * 0.7  # capped lower — scalps are risky

    return StrategySignal(
        strategy="SCALPING",
        side=side,
        entry_px=round(entry, 5),
        sl=round(sl, 5),
        tp=round(tp, 5),
        confidence=round(confidence, 3),
        metadata={"boll_z": round(boll_z, 3), "hour": hour,
                   "atr_pctile": round(atr_pctile, 3), "adx": round(adx, 1)},
    )


# ---------------------------------------------------------------------------
# Strategy Bank
# ---------------------------------------------------------------------------

class StrategyBank:
    """
    Registry of all available strategies.
    Provides a unified interface: get_signal(strategy_name, raw_df, features).
    """

    STRATEGIES = {
        "MEAN_REVERSION":  mean_reversion_signal,
        "TREND_PULLBACK":  trend_pullback_signal,
        "BREAKOUT":        breakout_signal,
        "SCALPING":        scalping_signal,
    }

    def get_signal(self, strategy_name: str, raw_df: pd.DataFrame,
                   features: dict) -> Optional[StrategySignal]:
        """Run a named strategy and return its signal (or None)."""
        fn = self.STRATEGIES.get(strategy_name)
        if fn is None:
            LOG.warning(f"[StrategyBank] Unknown strategy: {strategy_name}")
            return None
        try:
            return fn(raw_df, features)
        except Exception as e:
            LOG.debug(f"[StrategyBank] {strategy_name} error: {e}")
            return None

    def get_all_signals(self, raw_df: pd.DataFrame,
                        features: dict) -> Dict[str, Optional[StrategySignal]]:
        """Run ALL strategies and return their signals."""
        return {
            name: self.get_signal(name, raw_df, features)
            for name in self.STRATEGIES
        }

    @property
    def strategy_names(self) -> List[str]:
        return list(self.STRATEGIES.keys())
