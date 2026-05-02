"""
Simple statistical analysis for the passive AI Brain.

No training, no model promotion, no behavioral feedback. This module only reads
closed trade outcomes and produces human-reviewable summaries.
"""
from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Any, Dict, Iterable, List, Tuple


MIN_GROUP_TRADES = 20


def analyze_trades(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    trades = [r for r in rows if r.get("event") == "TRADE_CLOSED"]
    if not trades:
        return {
            "summary": {
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "net_pnl": 0.0,
                "avg_r": 0.0,
                "profit_factor": 0.0,
            },
            "best_conditions": [],
            "weak_conditions": [],
            "mistakes": [],
            "suggestions": [],
        }

    summary = _summary(trades)
    groups = []
    for field in ("symbol", "strategy", "regime", "side", "timeframe", "exit_reason"):
        groups.extend(_group_stats(trades, field))
    groups.extend(_confidence_buckets(trades))

    best = sorted(
        [g for g in groups if g["trades"] >= MIN_GROUP_TRADES and g["avg_r"] > 0],
        key=lambda g: (g["avg_r"], g["win_rate"]),
        reverse=True,
    )[:5]
    weak = sorted(
        [g for g in groups if g["trades"] >= MIN_GROUP_TRADES and g["avg_r"] < 0],
        key=lambda g: (g["avg_r"], g["win_rate"]),
    )[:8]

    mistakes = _detect_mistakes(trades, groups)
    suggestions = _suggestions(weak, mistakes)

    return {
        "summary": summary,
        "best_conditions": best,
        "weak_conditions": weak,
        "mistakes": mistakes,
        "suggestions": suggestions,
    }


def _summary(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    wins = [t for t in trades if _num(t.get("pnl")) > 0]
    losses = [t for t in trades if _num(t.get("pnl")) <= 0]
    gross_profit = sum(_num(t.get("pnl")) for t in wins)
    gross_loss = abs(sum(_num(t.get("pnl")) for t in losses))
    r_values = [_num(t.get("r_multiple")) for t in trades]
    return {
        "total_trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(trades), 4),
        "net_pnl": round(sum(_num(t.get("pnl")) for t in trades), 4),
        "avg_r": round(mean(r_values), 4) if r_values else 0.0,
        "profit_factor": round(gross_profit / gross_loss, 4) if gross_loss > 0 else float("inf"),
    }


def _group_stats(trades: List[Dict[str, Any]], field: str) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for trade in trades:
        grouped[str(trade.get(field) or "unknown")].append(trade)

    result = []
    for value, items in grouped.items():
        result.append(_stats_for(items, field, value))
    return result


def _confidence_buckets(trades: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for trade in trades:
        conf = _num(trade.get("confidence"))
        if conf < 0.50:
            bucket = "confidence:<0.50"
        elif conf < 0.65:
            bucket = "confidence:0.50-0.65"
        elif conf < 0.80:
            bucket = "confidence:0.65-0.80"
        else:
            bucket = "confidence:>=0.80"
        buckets[bucket].append(trade)
    return [_stats_for(items, "bucket", name) for name, items in buckets.items()]


def _stats_for(items: List[Dict[str, Any]], field: str, value: str) -> Dict[str, Any]:
    wins = [t for t in items if _num(t.get("pnl")) > 0]
    pnl = [_num(t.get("pnl")) for t in items]
    r_values = [_num(t.get("r_multiple")) for t in items]
    return {
        "field": field,
        "value": value,
        "trades": len(items),
        "win_rate": round(len(wins) / len(items), 4) if items else 0.0,
        "avg_r": round(mean(r_values), 4) if r_values else 0.0,
        "net_pnl": round(sum(pnl), 4),
    }


def _detect_mistakes(
    trades: List[Dict[str, Any]],
    groups: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    mistakes = []

    losing_sl = [
        t for t in trades
        if _num(t.get("pnl")) < 0 and "sl" in str(t.get("exit_reason", "")).lower()
    ]
    if len(losing_sl) >= MIN_GROUP_TRADES and len(losing_sl) / len(trades) >= 0.35:
        mistakes.append({
            "type": "sl_cluster",
            "message": "Large share of trades are closing at stop loss.",
            "trades": len(losing_sl),
        })

    fast_losses = [
        t for t in trades
        if _num(t.get("pnl")) < 0 and 0 < _num(t.get("duration_minutes")) <= 30
    ]
    if len(fast_losses) >= MIN_GROUP_TRADES:
        mistakes.append({
            "type": "fast_loss",
            "message": "Repeated losses close within 30 minutes of entry.",
            "trades": len(fast_losses),
        })

    for group in groups:
        if group["trades"] < MIN_GROUP_TRADES:
            continue
        if group["field"] == "bucket" and group["avg_r"] < 0 and group["win_rate"] < 0.45:
            mistakes.append({
                "type": "confidence_bucket",
                "message": f"{group['value']} has weak expectancy.",
                "trades": group["trades"],
                "avg_r": group["avg_r"],
            })
    return mistakes[:8]


def _suggestions(
    weak_conditions: List[Dict[str, Any]],
    mistakes: List[Dict[str, Any]],
) -> List[str]:
    suggestions = []
    for group in weak_conditions[:5]:
        suggestions.append(
            "Review {field}={value}: {trades} trades, win rate {win_rate:.1%}, avg R {avg_r}.".format(
                **group
            )
        )
    for mistake in mistakes[:3]:
        suggestions.append(mistake["message"])
    return suggestions


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default
