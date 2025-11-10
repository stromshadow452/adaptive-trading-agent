# tools/utils.py
import os
import json
from typing import Any, Optional

# --- add this helper ---
def _load_thresholds_config(path: str = "config/thresholds.json") -> dict:
    """
    Load thresholds config if present. Return {} if file missing or invalid.
    Expected structure (example):
    {
      "defaults": {
        "primary_thresh": 0.55,
        "finrl_thresh": 0.01,
        "max_position": 2
      },
      "symbols": {
        "EURUSD": {"primary_thresh": 0.60, "max_position": 1},
        "GBPUSD": {"finrl_thresh": 0.05}
      }
    }
    """
    try:
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf8") as f:
            return json.load(f)
    except Exception:
        return {}

def _env_override(symbol: str, name: str) -> Optional[str]:
    """
    Environment variable overrides:
      TH_<NAME>_<SYMBOL>  (highest priority)
      TH_<NAME>
    Example: TH_PRIMARY_THRESH_EURUSD or TH_PRIMARY_THRESH
    """
    env_name_sym = f"TH_{name.upper()}_{symbol.upper()}"
    env_name = f"TH_{name.upper()}"
    if env_name_sym in os.environ:
        return os.environ[env_name_sym]
    if env_name in os.environ:
        return os.environ[env_name]
    return None

def get_threshold_for(symbol: str, name: str, cli_default: Any = None,
                      cfg_path: str = "config/thresholds.json") -> Any:
    """
    Resolve threshold using priority:
      1) env override (TH_<NAME>_<SYMBOL> or TH_<NAME>)
      2) per-symbol value in config/thresholds.json
      3) global default in config/thresholds.json
      4) cli_default passed by caller
    Returns same dtype as found (caller should cast if needed).
    """
    symbol = (symbol or "").upper()
    name = name or ""

    # 1) env override
    env_val = _env_override(symbol, name)
    if env_val is not None:
        # try to parse numeric values, otherwise return string
        try:
            if "." in env_val:
                return float(env_val)
            return int(env_val)
        except Exception:
            return env_val

    # 2) config file
    cfg = _load_thresholds_config(cfg_path)
    if not cfg:
        return cli_default

    symbols_cfg = cfg.get("symbols", {})
    if symbol and symbol in symbols_cfg and name in symbols_cfg[symbol]:
        return symbols_cfg[symbol][name]

    # 3) global defaults
    defaults = cfg.get("defaults", {})
    if name in defaults:
        return defaults[name]

    # fallback to CLI default
    return cli_default
# --- end helper ---
