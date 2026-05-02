"""
tools/portfolio_selector.py
===========================
SCOPUS Portfolio Signal Selection Layer.

Operates ABOVE the pipeline — never modifies signals, SL, TP, or sizing.
Pure selection logic: score → threshold → correlation guard → top N.

Called once per bar after ALL symbol pipelines have been run.
"""
from __future__ import annotations

from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Correlation groups  (per spec)
# ---------------------------------------------------------------------------
USD_MAJORS_GROUP = frozenset({"EURUSD", "GBPUSD"})
COMMODITY_GROUP  = frozenset({"AUDUSD", "NZDUSD", "USDCAD"})
JPY_GROUP        = frozenset({"USDJPY"})
METALS_GROUP     = frozenset({"XAUUSD", "XAGUSD"})

SCORE_THRESHOLD  = 0.50   # signals below this score are discarded
MAX_CONCURRENT   = 3      # max simultaneous portfolio positions



def get_group(symbol: str) -> Optional[str]:
    """Return correlation group name for a symbol, or None."""
    s = symbol.upper()
    if s in USD_MAJORS_GROUP: return "USD_MAJORS"
    if s in COMMODITY_GROUP:  return "COMMODITY"
    if s in JPY_GROUP:        return "JPY"
    if s in METALS_GROUP:     return "METALS"
    return None


def compute_score(boll_z: float) -> float:
    """
    score = min(abs(boll_z) / 3.0, 1.0)

    Uses ONLY the existing boll_z feature — no new indicators.
    Score of 1.0 at |boll_z| = 3 (very extended).
    Score of 0.5 threshold at |boll_z| = 1.5.
    """
    return min(abs(boll_z) / 3.0, 1.0)


def select_signals(
    candidates:     List[dict],   # [{"symbol", "signal", "score", "boll_z"}, ...]
    open_positions: dict,         # symbol -> SimulatedFill (from executor)
    max_trades:     int = MAX_CONCURRENT,
) -> List[dict]:
    """
    Apply full portfolio selection constraints.

    Step 1 — Filter:
        Remove score < SCORE_THRESHOLD
        Remove symbols with existing open positions

    Step 2 — Sort:
        Descending by score (highest |boll_z| extension first)

    Step 3 — Correlation guard:
        Allow only 1 trade per group (USD / JPY / METALS)
        If multiple candidates in same group → keep highest score

    Step 4 — Portfolio cap:
        Select at most max_trades signals

    Returns:
        List of selected candidate dicts (subset of input).
    """
    open_syms = {k.upper() for k in open_positions}
    
    # Pre-populate groups used by existing open positions
    groups_used:  set = {get_group(sym) for sym in open_syms if get_group(sym) is not None}
    
    # Calculate how many slots are actually available
    available_slots = max(0, max_trades - len(open_syms))
    if available_slots <= 0:
        return []

    # Step 1: filter
    filtered = [
        c for c in candidates
        if c["score"] >= SCORE_THRESHOLD
        and c["symbol"].upper() not in open_syms
    ]

    # Step 2: sort descending
    filtered.sort(key=lambda x: x["score"], reverse=True)

    # Step 3 & 4: correlation guard + cap
    selected:     List[dict] = []
    symbols_used: set        = set()

    for candidate in filtered:
        if len(selected) >= available_slots:
            break
        sym   = candidate["symbol"].upper()
        group = get_group(sym)

        if sym in symbols_used:
            continue
        if group is not None and group in groups_used:
            continue        # correlation violation — skip

        selected.append(candidate)
        symbols_used.add(sym)
        if group is not None:
            groups_used.add(group)

    return selected
