# src/stages/meta_gating_brain.py
"""
Meta-Gating Brain
Additive, deterministic stage that fuses Primary + FinRL signals.
No globals. Typed I/O only.
"""

from typing import Dict, Any

def run_meta_gating(ctx: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    primary_conf = float(ctx.get("primary_conf", 0.0))
    finrl_conf   = float(ctx.get("finrl_conf", 0.0))
    p_thresh     = float(cfg.get("primary_thresh", 0.5))
    f_thresh     = float(cfg.get("finrl_thresh", 0.7))
    grey_zone    = float(cfg.get("grey_zone", 0.08))

    allow_override = (primary_conf >= (p_thresh - grey_zone)) and (finrl_conf >= 0.95)
    vote = max(primary_conf, finrl_conf) if allow_override else primary_conf

    return {
        "stage": "meta_gating_brain",
        "enabled": True,
        "vote": round(vote, 3),
        "primary_conf": round(primary_conf, 3),
        "finrl_conf": round(finrl_conf, 3),
        "p_thresh": p_thresh,
        "f_thresh": f_thresh,
        "grey_zone": grey_zone,
        "allow_override": allow_override,
        "status": "OK",
    }
