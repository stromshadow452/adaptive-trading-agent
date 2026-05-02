"""
src/features/cross_asset.py
=============================
SCOPUS Cross-Asset Regime Signal Features — Week 7 (Category H).

4 cross-asset features that improve regime detection:
    ca_dxy_momentum       — DXY momentum (via EURUSD inverse, 5/20-day ratio)
    ca_gold_trend         — XAUUSD 20-day trend (safe-haven flow)
    ca_silver_gold_ratio  — XAGUSD/XAUUSD ratio (risk-on/off barometer)
    ca_vol_regime_proxy   — Equity volatility proxy (US500 ATR pctile)

Uses the existing DataSourceABC interface — no new infrastructure needed.
All symbols must be available in the data source (CSV or MT5).

Fallback: returns 0.0 for any symbol that is missing or errors.
Promotion status: SHADOW_ONLY (Week 11 per plan)
"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional

logger = logging.getLogger(__name__)

CROSS_ASSET_FEATURE_LIST = [
    "ca_dxy_momentum",
    "ca_gold_trend",
    "ca_silver_gold_ratio",
    "ca_vol_regime_proxy",
]

_DEFAULTS: Dict[str, float] = {k: 0.0 for k in CROSS_ASSET_FEATURE_LIST}


class CrossAssetSignals:
    """
    Reads auxiliary symbols via DataSourceABC and returns 4 regime-awareness
    features that can be merged into the main feature vector.

    Usage:
        data_source = HistoricalCSVSource("data/")
        ca = CrossAssetSignals(data_source)
        signals = ca.compute()
        # → {"ca_dxy_momentum": -0.002, "ca_gold_trend": 0.015, ...}
    """

    AUX_SYMBOLS = ["EURUSD", "XAUUSD", "XAGUSD", "US500"]

    def __init__(self, data_source):
        """
        Args:
            data_source: Any DataSourceABC implementation
                         (HistoricalCSVSource or MT5LiveSource).
        """
        self.data_source = data_source

    def compute(self) -> Dict[str, float]:
        """
        Compute all 4 cross-asset features for the current bar.
        Returns dict of feature_name → float value.
        Missing data is silently replaced with 0.0.
        """
        signals = dict(_DEFAULTS)

        # 1. DXY Momentum — EURUSD 20-day SMA crossover (inverse of DXY)
        #    Positive = EURUSD rising = DXY weakening
        signals["ca_dxy_momentum"] = self._dxy_momentum()

        # 2. Gold trend — XAUUSD 20-day return (safe-haven flow)
        #    Positive = gold up = risk-off / uncertainty
        signals["ca_gold_trend"] = self._gold_trend()

        # 3. Silver/Gold ratio — risk-on barometer
        #    Silver outperforms gold in risk-on environments
        signals["ca_silver_gold_ratio"] = self._silver_gold_ratio()

        # 4. Equity vol regime proxy — US500 ATR percentile
        #    High = equity volatility elevated = tighten FX risk
        signals["ca_vol_regime_proxy"] = self._vol_regime_proxy()

        return signals

    def compute_as_series(self, index: pd.Index) -> pd.DataFrame:
        """
        Return a single-row DataFrame aligned to `index` (latest bar).
        For combining with bar-level feature DataFrames.
        """
        vals = self.compute()
        return pd.DataFrame([vals], index=[index[-1]])[CROSS_ASSET_FEATURE_LIST]

    # ------------------------------------------------------------------ #
    # Private per-signal computations                                     #
    # ------------------------------------------------------------------ #

    def _get_daily_close(self, symbol: str, n_days: int) -> Optional[pd.Series]:
        """Load n_days of daily bars and return close series."""
        try:
            from src.market_data.types import Timeframe
            df = self.data_source.get_history(symbol, Timeframe.D1, n_bars=n_days + 5)
            if df is None or df.empty:
                return None
            close_col = next((c for c in df.columns if c.lower() == "close"), None)
            if close_col is None:
                return None
            return df[close_col].dropna().tail(n_days)
        except Exception as e:
            logger.debug(f"[CrossAsset] {symbol} D1 history error: {e}")
            return None

    def _dxy_momentum(self) -> float:
        """
        EURUSD 5/20-day SMA ratio (inverse proxy for DXY).
        >0 = EURUSD rising = DXY weakening = bullish FX majors ex-USD
        <0 = EURUSD falling = DXY strengthening
        """
        try:
            close = self._get_daily_close("EURUSD", 22)
            if close is None or len(close) < 10:
                return 0.0
            sma5  = close.iloc[-5:].mean()
            sma20 = close.mean()
            return float(sma5 / sma20 - 1.0) if sma20 != 0 else 0.0
        except Exception as e:
            logger.debug(f"[CrossAsset] dxy_momentum error: {e}")
            return 0.0

    def _gold_trend(self) -> float:
        """
        XAUUSD 20-day return.
        Positive = gold up = risk-off / safe-haven demand / USD caution
        """
        try:
            close = self._get_daily_close("XAUUSD", 22)
            if close is None or len(close) < 5:
                return 0.0
            return float(close.iloc[-1] / close.iloc[0] - 1.0)
        except Exception as e:
            logger.debug(f"[CrossAsset] gold_trend error: {e}")
            return 0.0

    def _silver_gold_ratio(self) -> float:
        """
        Silver/Gold ratio (XAGUSD / XAUUSD, current bar).
        Rising = risk-on (silver outperforms gold)
        Falling = risk-off (gold outperforms silver)
        Returns the 10-day change in ratio (not absolute ratio).
        """
        try:
            xag = self._get_daily_close("XAGUSD", 12)
            xau = self._get_daily_close("XAUUSD", 12)
            if xag is None or xau is None:
                return 0.0
            if len(xag) < 2 or len(xau) < 2:
                return 0.0
            ratio_now  = float(xag.iloc[-1]) / max(float(xau.iloc[-1]), 1e-8)
            ratio_old  = float(xag.iloc[0])  / max(float(xau.iloc[0]),  1e-8)
            return float(ratio_now / ratio_old - 1.0) if ratio_old != 0 else 0.0
        except Exception as e:
            logger.debug(f"[CrossAsset] silver_gold_ratio error: {e}")
            return 0.0

    def _vol_regime_proxy(self) -> float:
        """
        US500 (S&P 500) ATR percentile rank over 100 days.
        High (>0.8) = elevated equity vol = risk-off → reduce FX position sizes.
        Low (<0.2)  = low equity vol = risk-on → normal sizing.
        Returns: 0.0–1.0 percentile rank.
        """
        try:
            from src.market_data.types import Timeframe
            df = self.data_source.get_history("US500", Timeframe.D1, n_bars=105)
            if df is None or len(df) < 20:
                return 0.5   # neutral default
            h = df.get("high", df.iloc[:, 1] if df.shape[1] > 1 else df.iloc[:, 0])
            l = df.get("low",  df.iloc[:, 2] if df.shape[1] > 2 else df.iloc[:, 0])
            c = df.get("close", df.iloc[:, -1])
            tr1 = (h - l).abs()
            tr2 = (h - c.shift(1)).abs()
            tr3 = (l - c.shift(1)).abs()
            tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(14, min_periods=7).mean()
            pctile = float(
                np.sum(atr.iloc[:-1] <= atr.iloc[-1]) / max(len(atr) - 1, 1)
            )
            return round(pctile, 4)
        except Exception as e:
            logger.debug(f"[CrossAsset] vol_regime_proxy error: {e}")
            return 0.5


def get_cross_asset_defaults() -> Dict[str, float]:
    """Return neutral default values when data source is unavailable."""
    return dict(_DEFAULTS)
