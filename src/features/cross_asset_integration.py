"""
src/features/cross_asset_integration.py
=========================================
SCOPUS Cross-Asset Signal Integration — Week 11.

Activates the 4 cross-asset regime signals from cross_asset.py inside
the PipelineV2 feature stack.

Key design:
  • Works with BOTH CSV and MT5 data sources via DataSourceABC
  • Cross-asset bars fetched once per day (not every bar) — cached in memory
  • Safe fallback: 0.0 for any unavailable symbol
  • Signals integrated as regular features (ca_*) — downstream model-agnostic

Daily cache avoids hammering MT5 / CSV filesystem:
    _CACHE_TTL_SECONDS = 3600 * 6   (6 hour cache)
"""
from __future__ import annotations

import logging
import time
from typing import Dict, Optional

import numpy as np
import pandas as pd

LOG = logging.getLogger("cross_asset_integration")

_CACHE_TTL_SECONDS = 6 * 3600   # 6 hours
_cache: Dict[str, dict] = {}    # symbol → {"ts": float, "signals": dict}


def get_cross_asset_signals(data_source) -> Dict[str, float]:
    """
    Return the 4 cross-asset feature values, using a 6-hour cache.

    Args:
        data_source: Any DataSourceABC (HistoricalCSVSource or MT5LiveSource).

    Returns:
        Dict {"ca_dxy_momentum": float, "ca_gold_trend": float,
               "ca_silver_gold_ratio": float, "ca_vol_regime_proxy": float}
    """
    from src.features.cross_asset import CrossAssetSignals, get_cross_asset_defaults

    cache_key = type(data_source).__name__
    now       = time.time()

    if cache_key in _cache:
        age = now - _cache[cache_key]["ts"]
        if age < _CACHE_TTL_SECONDS:
            return _cache[cache_key]["signals"]

    try:
        ca       = CrossAssetSignals(data_source)
        signals  = ca.compute()
    except Exception as e:
        LOG.debug(f"[cross_asset_integration] compute error: {e}")
        signals = get_cross_asset_defaults()

    _cache[cache_key] = {"ts": now, "signals": signals}
    return signals


def append_cross_asset_row(
    full_df: pd.DataFrame,
    data_source,
) -> pd.DataFrame:
    """
    Append cross-asset feature columns as constant values across all rows
    of full_df (since they are macro signals, same value for the whole batch).

    Returns:
        full_df with 4 new ca_* columns appended.
    """
    from src.features.cross_asset import CROSS_ASSET_FEATURE_LIST
    signals = get_cross_asset_signals(data_source)
    for col in CROSS_ASSET_FEATURE_LIST:
        full_df[col] = signals.get(col, 0.0)
    return full_df


def invalidate_cache():
    """Force cache refresh on next call (e.g. after broker reconnect)."""
    _cache.clear()


def cross_asset_summary(data_source) -> dict:
    """
    Return a human-readable dict of current cross-asset signals for logging.
    """
    signals  = get_cross_asset_signals(data_source)
    dxy_mom  = signals.get("ca_dxy_momentum", 0.0)
    gold     = signals.get("ca_gold_trend", 0.0)
    sg_ratio = signals.get("ca_silver_gold_ratio", 0.0)
    vol_prx  = signals.get("ca_vol_regime_proxy", 0.5)

    dxy_str  = "EUR_WEAK (USD STRONG)" if dxy_mom < -0.002 else (
               "EUR_STRONG (USD WEAK)" if dxy_mom > 0.002 else "NEUTRAL")
    gold_str = "RISK_OFF" if gold > 0.01 else ("RISK_ON" if gold < -0.01 else "NEUTRAL")
    vol_str  = "HIGH" if vol_prx > 0.80 else ("LOW" if vol_prx < 0.20 else "NORMAL")

    return {
        "ca_dxy_momentum":      round(dxy_mom,  5),
        "ca_dxy_regime":        dxy_str,
        "ca_gold_trend":        round(gold,      5),
        "ca_gold_regime":       gold_str,
        "ca_silver_gold_ratio": round(sg_ratio,  5),
        "ca_vol_regime_proxy":  round(vol_prx,  4),
        "ca_vol_regime":        vol_str,
    }
