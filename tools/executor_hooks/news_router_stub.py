from __future__ import annotations

from typing import Any, Dict


def news_router(md: Dict[str, Any]) -> str:
    """
    Deterministic, offline stub for headline routing used by the orchestrator
    smoke path. Maps by symbol with a stable default.
    """
    sym = (md.get("symbol") or md.get("market") or "GENERIC")
    table = {
        "EURUSD": "Analysts stay cautious as euro rebounds after weak data.",
        "USDJPY": "Dollar rally pauses; risk sentiment mixed amid policy remarks.",
    }
    return table.get(sym, "Market sees steady performance with balanced commentary.")

