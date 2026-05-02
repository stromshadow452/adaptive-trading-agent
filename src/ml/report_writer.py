"""
Report writer for passive AI Brain summaries.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


DEFAULT_REPORT_DIR = "reports/ai_brain"


def write_report(analysis: Dict[str, Any], report_dir: str = DEFAULT_REPORT_DIR) -> Path:
    out_dir = Path(report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"ai_brain_report_{stamp}.md"

    summary = analysis.get("summary", {})
    lines = [
        "# AI Brain Report",
        "",
        "## Summary",
        f"- Total trades: {summary.get('total_trades', 0)}",
        f"- Wins / losses: {summary.get('wins', 0)} / {summary.get('losses', 0)}",
        f"- Win rate: {_pct(summary.get('win_rate', 0.0))}",
        f"- Net PnL: {summary.get('net_pnl', 0.0)}",
        f"- Average R: {summary.get('avg_r', 0.0)}",
        f"- Profit factor: {summary.get('profit_factor', 0.0)}",
        "",
        "## Best Conditions",
    ]

    lines.extend(_condition_lines(analysis.get("best_conditions", [])))
    lines.extend(["", "## Weak Conditions"])
    lines.extend(_condition_lines(analysis.get("weak_conditions", [])))
    lines.extend(["", "## Mistake Flags"])
    mistakes = analysis.get("mistakes", [])
    if mistakes:
        for item in mistakes:
            lines.append(f"- {item.get('message', json.dumps(item, sort_keys=True))}")
    else:
        lines.append("- No statistically meaningful mistake flags yet.")

    lines.extend(["", "## Suggestions"])
    suggestions = analysis.get("suggestions", [])
    if suggestions:
        lines.extend(f"- {s}" for s in suggestions)
    else:
        lines.append("- No suggestions until more closed trades are available.")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _condition_lines(items):
    if not items:
        return ["- Not enough samples yet."]
    return [
        "- {field}={value}: {trades} trades, win rate {win_rate}, avg R {avg_r}, net PnL {net_pnl}".format(
            field=i.get("field"),
            value=i.get("value"),
            trades=i.get("trades"),
            win_rate=_pct(i.get("win_rate", 0.0)),
            avg_r=i.get("avg_r", 0.0),
            net_pnl=i.get("net_pnl", 0.0),
        )
        for i in items
    ]


def _pct(value: Any) -> str:
    try:
        return f"{float(value):.1%}"
    except (TypeError, ValueError):
        return "0.0%"
