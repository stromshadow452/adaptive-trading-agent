# src/stages/meta_gating_brain.py
"""
Meta-Gating Brain
Additive, deterministic stage that fuses Primary + FinRL signals.
Enhanced with adaptive thresholds for LEARNING vs CONFIRMATION modes.
No globals. Typed I/O only.
"""

from typing import Dict, Any, Optional

# Default thresholds by mode
THRESHOLDS = {
    "LEARNING": {
        "primary_thresh": 0.30,
        "finrl_thresh": 0.50,
        "grey_zone": 0.15,      # Wider grey zone in learning
        "finrl_override_min": 0.85,  # Lower bar for FinRL override
    },
    "CONFIRMATION": {
        "primary_thresh": 0.50,
        "finrl_thresh": 0.70,
        "grey_zone": 0.08,
        "finrl_override_min": 0.95,
    }
}


def run_meta_gating(
    ctx: Dict[str, Any], 
    cfg: Dict[str, Any],
    context_output: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Meta-gating stage with adaptive thresholds.
    
    Args:
        ctx: Context with primary_conf, finrl_conf
        cfg: Configuration overrides
        context_output: Optional Context Brain output for mode detection
        
    Returns:
        Stage result with vote and metadata
    """
    primary_conf = float(ctx.get("primary_conf", 0.0))
    finrl_conf   = float(ctx.get("finrl_conf", 0.0))
    
    # Determine operating mode from context output
    operating_mode = "CONFIRMATION"
    if context_output:
        operating_mode = context_output.get("operating_mode", "CONFIRMATION")
    
    # Get mode-specific defaults
    mode_defaults = THRESHOLDS.get(operating_mode, THRESHOLDS["CONFIRMATION"])
    
    # Allow config overrides
    p_thresh = float(cfg.get("primary_thresh", mode_defaults["primary_thresh"]))
    f_thresh = float(cfg.get("finrl_thresh", mode_defaults["finrl_thresh"]))
    grey_zone = float(cfg.get("grey_zone", mode_defaults["grey_zone"]))
    finrl_override_min = mode_defaults["finrl_override_min"]
    
    # Override logic
    allow_override = (primary_conf >= (p_thresh - grey_zone)) and (finrl_conf >= finrl_override_min)
    vote = max(primary_conf, finrl_conf) if allow_override else primary_conf
    
    # In LEARNING mode, apply a confidence floor if we have any signal
    if operating_mode == "LEARNING":
        context_conf = 0.5
        if context_output:
            context_conf = context_output.get("context_confidence", 0.5)
        
        # Blend context confidence into vote
        if vote > 0 and context_conf > 0:
            vote = 0.6 * vote + 0.4 * context_conf
            vote = max(vote, 0.35)  # Floor in LEARNING mode
    
    # Determine if this passes the gate
    passes_gate = vote >= p_thresh
    
    return {
        "stage": "meta_gating_brain",
        "enabled": True,
        "vote": round(vote, 3),
        "passes_gate": passes_gate,
        "primary_conf": round(primary_conf, 3),
        "finrl_conf": round(finrl_conf, 3),
        "p_thresh": p_thresh,
        "f_thresh": f_thresh,
        "grey_zone": grey_zone,
        "allow_override": allow_override,
        "operating_mode": operating_mode,
        "status": "OK",
    }
