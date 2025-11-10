"""
tools.utils package

Provides:
- get_threshold_for(symbol, key, cli_default=None, default=None, config_path='config/thresholds.json')
- align_features_for_lgb(...) re-exported from .feature_align

This replaces the old tools/utils.py module safely.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any, Dict

# Re-export the aligner so existing imports continue to work
try:
    from .feature_align import align_features_for_lgb  # noqa: F401
except Exception:
    # If feature_align.py isn't present, it's fine; get_threshold_for still works.
    pass

# --- symbol normalize (match executor logic) ---
def _normalize_symbol(sym: str) -> str:
    if not sym:
        return ""
    s = (
        str(sym)
        .upper()
        .replace("-", "")
        .replace("_", "")
        .replace("/", "")
        .replace("USDT", "USD")
    )
    # If it looks like BASEQUOTE keep; else just return cleaned
    return s

@lru_cache(maxsize=1)
def _load_thresholds(config_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Load thresholds JSON once and cache.
    File format example:
    {
      "EURUSD": {"primary_thresh": 0.60, "finrl_thresh": 0.80, "max_position": 1},
      "GBPUSD": {"primary_thresh": 0.60, "finrl_thresh": 0.40}
    }
    """
    path = os.path.abspath(config_path)
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            # Normalize keys to uppercase no separators
            return { _normalize_symbol(k): (v if isinstance(v, dict) else {}) for k, v in data.items() }
    except Exception:
        pass
    return {}

def get_threshold_for(symbol: str,
                      key: str,
                      *,
                      cli_default: Any = None,
                      default: Any = None,
                      config_path: str = "config/thresholds.json") -> Any:
    """
    Resolve per-symbol thresholds with CLI override semantics.

    Order of precedence (highest first):
      1) If CLI provided a value (cli_default is NOT None), return cli_default.
      2) If config has per-symbol value for `key`, return that.
      3) If `default` is provided, return default.
      4) Fallback to cli_default (even if None), for compatibility with callers.

    Args:
        symbol: e.g., "EURUSD"
        key:    e.g., "primary_thresh", "finrl_thresh", "max_position"
        cli_default: the caller's override (usually CLI value), may be None
        default: final fallback if config missing and cli_default is None
        config_path: path to thresholds json (default "config/thresholds.json")
    """
    sym = _normalize_symbol(symbol)
    conf = _load_thresholds(config_path)

    # CLI override (executor uses "Option A: CLI overrides per-symbol config")
    if cli_default is not None:
        return cli_default

    # Config value
    try:
        if sym and sym in conf and isinstance(conf[sym], dict) and key in conf[sym]:
            return conf[sym][key]
    except Exception:
        pass

    # Hard default if provided
    if default is not None:
        return default

    # Final fallback (None or whatever caller passed)
    return cli_default
