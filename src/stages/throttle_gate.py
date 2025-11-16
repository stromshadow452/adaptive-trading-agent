# src/stages/throttle_gate.py
"""
Throttle Gate
Prevents rapid re-entries per symbol within cooldown bars.
No persistence; uses transient state dict passed in.
"""

from typing import Dict, Any

def run_throttle_gate(ctx: Dict[str, Any], cfg: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    sym = ctx.get("symbol")
    now_bar = int(ctx.get("bar_index", 0))
    cd_bars = int(cfg.get("cooldown_bars", 3))

    last_map = state.get("last_exec_bar", {})
    last_exec_bar = last_map.get(sym)
    on_cooldown = (last_exec_bar is not None) and ((now_bar - int(last_exec_bar)) < cd_bars)

    return {
        "stage": "throttle_gate",
        "enabled": True,
        "symbol": sym,
        "cooldown_bars": cd_bars,
        "on_cooldown": on_cooldown,
        "vote": 0.0 if on_cooldown else float(ctx.get("vote", 0.0)),
        "status": "OK" if not on_cooldown else "[COOLDOWN]",
    }
