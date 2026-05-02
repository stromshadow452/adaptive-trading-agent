"""
Lightweight schemas for the passive AI Brain trade journal.

These records are observational only. They are not used by the trading
pipeline to alter signals, sizing, stops, or execution.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class TradeEntryRecord:
    event: str
    trade_id: str
    timestamp_entry: str
    symbol: str
    timeframe: str = "UNKNOWN"
    side: str = "unknown"
    strategy: str = "unknown"
    regime: str = "unknown"
    entry_price: float = 0.0
    size: float = 0.0
    sl: Optional[float] = None
    tp: Optional[float] = None
    confidence: float = 0.0
    ml_filter_pass: Optional[bool] = None
    edge_score: Optional[float] = None
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TradeOutcomeRecord:
    event: str
    trade_id: str
    timestamp_entry: str
    timestamp_exit: str
    symbol: str
    timeframe: str = "UNKNOWN"
    side: str = "unknown"
    strategy: str = "unknown"
    regime: str = "unknown"
    entry_price: float = 0.0
    exit_price: float = 0.0
    size: float = 0.0
    sl: Optional[float] = None
    tp: Optional[float] = None
    exit_reason: str = "unknown"
    confidence: float = 0.0
    ml_filter_pass: Optional[bool] = None
    edge_score: Optional[float] = None
    pnl: float = 0.0
    r_multiple: float = 0.0
    is_win: bool = False
    duration_minutes: float = 0.0
    mae: Optional[float] = None
    mfe: Optional[float] = None
    context: Dict[str, Any] = field(default_factory=dict)
