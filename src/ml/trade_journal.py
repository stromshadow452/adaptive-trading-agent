"""
Append-only JSONL journal for the passive AI Brain.

The journal is deliberately boring: append small records, rotate by size, and
avoid any dependency on the trading decision path.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


DEFAULT_JOURNAL_PATH = "logs/ai_brain/trades.jsonl"
DEFAULT_MAX_BYTES = 25 * 1024 * 1024


class TradeJournal:
    def __init__(
        self,
        path: str = DEFAULT_JOURNAL_PATH,
        max_bytes: int = DEFAULT_MAX_BYTES,
    ):
        self.path = Path(path)
        self.max_bytes = int(max_bytes)

    def append(self, record: Any) -> None:
        payload = self._to_dict(record)
        payload.setdefault("logged_at", datetime.now(timezone.utc).isoformat())

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._rotate_if_needed()
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, default=str, sort_keys=True) + "\n")

    def read_events(self, event: Optional[str] = None) -> Iterable[Dict[str, Any]]:
        if not self.path.exists():
            return []

        rows = []
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event is None or row.get("event") == event:
                    rows.append(row)
        return rows

    def _rotate_if_needed(self) -> None:
        if self.max_bytes <= 0 or not self.path.exists():
            return
        if self.path.stat().st_size < self.max_bytes:
            return

        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        rotated = self.path.with_name(f"{self.path.stem}.{stamp}{self.path.suffix}")
        os.replace(self.path, rotated)

    @staticmethod
    def _to_dict(record: Any) -> Dict[str, Any]:
        if is_dataclass(record):
            return asdict(record)
        if isinstance(record, dict):
            return dict(record)
        raise TypeError(f"Unsupported journal record: {type(record)!r}")
