"""
Passive AI Brain layer.

This module is intentionally one-way: it observes fills and closed trades,
writes small journal records, and can produce reports. It never returns a
decision to the trading pipeline.
"""
from __future__ import annotations

import dataclasses
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from src.ml.report_writer import write_report
from src.ml.schemas import TradeEntryRecord, TradeOutcomeRecord
from src.ml.trade_analyzer import analyze_trades
from src.ml.trade_journal import DEFAULT_JOURNAL_PATH, TradeJournal

LOG = logging.getLogger("ai_brain")

_BRAIN = None


class AIBrain:
    def __init__(
        self,
        enabled: bool = False,
        journal_path: str = DEFAULT_JOURNAL_PATH,
        report_dir: str = "reports/ai_brain",
        max_bytes: int = 25 * 1024 * 1024,
    ):
        self.enabled = bool(enabled)
        self.journal = TradeJournal(journal_path, max_bytes=max_bytes)
        self.report_dir = report_dir

    def record_entry(
        self,
        signal: Dict[str, Any],
        fill: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.enabled:
            return
        fill_d = _as_dict(fill)
        signal = signal or {}
        meta = dict(fill_d.get("metadata") or signal.get("metadata") or {})
        if context:
            meta.update(context)

        record = TradeEntryRecord(
            event="TRADE_ENTRY",
            trade_id=_trade_id(fill_d, signal),
            timestamp_entry=str(fill_d.get("filled_at") or fill_d.get("requested_at") or _now()),
            symbol=str(fill_d.get("symbol") or signal.get("symbol") or "UNKNOWN").upper(),
            timeframe=str(signal.get("tf") or signal.get("timeframe") or meta.get("timeframe") or "UNKNOWN"),
            side=str(fill_d.get("side") or signal.get("side") or "unknown").lower(),
            strategy=str(fill_d.get("strategy") or signal.get("strategy") or "unknown"),
            regime=str(fill_d.get("regime") or signal.get("regime") or "unknown"),
            entry_price=_num(fill_d.get("fill_px"), _num(signal.get("price"))),
            size=_num(fill_d.get("size"), _num(signal.get("size"))),
            sl=_optional_num(fill_d.get("sl", signal.get("sl"))),
            tp=_optional_num(fill_d.get("tp", signal.get("tp"))),
            confidence=_num(fill_d.get("confidence"), _num(signal.get("confidence"))),
            ml_filter_pass=_optional_bool(meta.get("ml_filter_pass")),
            edge_score=_optional_num(meta.get("edge_score")),
            context=_compact_context(meta),
        )
        self.journal.append(record)

    def record_exit(self, closed_trade: Any) -> None:
        if not self.enabled:
            return
        trade = _as_dict(closed_trade)
        meta = dict(trade.get("metadata") or {})

        record = TradeOutcomeRecord(
            event="TRADE_CLOSED",
            trade_id=_trade_id(trade, trade),
            timestamp_entry=str(
                trade.get("timestamp_entry")
                or trade.get("filled_at")
                or trade.get("requested_at")
                or _now()
            ),
            timestamp_exit=str(
                trade.get("timestamp_exit")
                or trade.get("closed_at")
                or trade.get("ts")
                or _now()
            ),
            symbol=str(trade.get("symbol") or "UNKNOWN").upper(),
            timeframe=str(trade.get("tf") or trade.get("timeframe") or meta.get("timeframe") or "UNKNOWN"),
            side=str(trade.get("side") or "unknown").lower(),
            strategy=str(trade.get("strategy") or trade.get("decision_source") or "unknown"),
            regime=str(trade.get("regime") or "unknown"),
            entry_price=_num(trade.get("entry_price"), _num(trade.get("fill_px"))),
            exit_price=_num(trade.get("exit_price"), _num(trade.get("close_px"))),
            size=_num(trade.get("size")),
            sl=_optional_num(trade.get("sl") if "sl" in trade else trade.get("sl_price")),
            tp=_optional_num(trade.get("tp") if "tp" in trade else trade.get("tp_price")),
            exit_reason=str(trade.get("exit_reason") or trade.get("close_reason") or "unknown"),
            confidence=_num(
                trade.get("confidence"),
                _num(trade.get("ml_confidence"), _num(trade.get("ml_conf_calibrated"))),
            ),
            ml_filter_pass=_optional_bool(trade.get("ml_filter_pass", meta.get("ml_filter_pass"))),
            edge_score=_optional_num(trade.get("edge_score", meta.get("edge_score"))),
            pnl=_num(trade.get("pnl"), _num(trade.get("pnl_usd"))),
            r_multiple=_num(trade.get("r_multiple")),
            is_win=bool(trade.get("is_win")) if "is_win" in trade else _num(trade.get("pnl"), _num(trade.get("pnl_usd"))) > 0,
            duration_minutes=_num(trade.get("duration_minutes"), _num(trade.get("hold_minutes"))),
            mae=_optional_num(trade.get("mae")),
            mfe=_optional_num(trade.get("mfe")),
            context=_compact_context({**meta, **trade}),
        )
        self.journal.append(record)

    def analyze(self) -> Dict[str, Any]:
        return analyze_trades(self.journal.read_events())

    def generate_report(self) -> Path:
        return write_report(self.analyze(), self.report_dir)


def get_ai_brain() -> AIBrain:
    global _BRAIN
    if _BRAIN is None:
        _BRAIN = AIBrain(
            enabled=_env_enabled("AI_BRAIN_ENABLED", default=False),
            journal_path=os.getenv("AI_BRAIN_JOURNAL", DEFAULT_JOURNAL_PATH),
            report_dir=os.getenv("AI_BRAIN_REPORT_DIR", "reports/ai_brain"),
            max_bytes=int(os.getenv("AI_BRAIN_MAX_BYTES", str(25 * 1024 * 1024))),
        )
    return _BRAIN


def safe_record_entry(
    signal: Dict[str, Any],
    fill: Any,
    context: Optional[Dict[str, Any]] = None,
) -> None:
    try:
        get_ai_brain().record_entry(signal, fill, context=context)
    except Exception as exc:
        LOG.debug("[AI_BRAIN] entry logging skipped: %s", exc)


def safe_record_exit(closed_trade: Any) -> None:
    try:
        get_ai_brain().record_exit(closed_trade)
    except Exception as exc:
        LOG.debug("[AI_BRAIN] exit logging skipped: %s", exc)


def _as_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    if isinstance(obj, dict):
        return dict(obj)
    if hasattr(obj, "__dict__"):
        return dict(vars(obj))
    return {}


def _trade_id(primary: Dict[str, Any], fallback: Dict[str, Any]) -> str:
    explicit = primary.get("trade_id") or primary.get("order_id") or fallback.get("trade_id") or fallback.get("order_id")
    if explicit:
        return str(explicit)
    symbol = str(primary.get("symbol") or fallback.get("symbol") or "UNKNOWN").upper()
    ts = str(primary.get("filled_at") or primary.get("timestamp_entry") or fallback.get("filled_at") or fallback.get("timestamp_entry") or _now())
    return f"{symbol}-{ts}"


def _compact_context(source: Dict[str, Any]) -> Dict[str, Any]:
    keys = (
        "rsi", "rsi14", "atr", "atr14", "atr_value", "atr_pctile", "boll_z",
        "adx", "session", "spread", "slippage_pips", "slippage_usd",
        "probability_of_success", "signal_quality", "soft_filter_score",
    )
    return {k: source[k] for k in keys if k in source and source[k] is not None}


def _env_enabled(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _optional_num(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    return _num(value)


def _optional_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "pass", "passed"}
    return bool(value)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
