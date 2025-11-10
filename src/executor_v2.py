#!/usr/bin/env python3
"""
Executor v2 — paper/live stub

- Reads approved plans (JSON list) produced by portfolio gate
- Optional per-symbol max position cap (extra guard)
- Simulates immediate fills (paper) and appends to executions CSV
- Auto-rotates legacy CSV with mismatched headers and writes v2 header
- Maintains aggregate_summary.json (equity/peak only; PnL realized on close not implemented here)

Run:
  python -m src.executor_v2 \
    --approved reports/daily/approved.json \
    --executions reports/executions/executions.csv \
    --aggregate reports/aggregate/aggregate_summary.json \
    --max_position 1 \
    --mode paper
"""

from __future__ import annotations
import os, csv, json, argparse, time, math
from datetime import datetime, timezone
from typing import Any, Dict, List

V2_HEADERS = [
    "ts", "mode", "symbol", "tf", "side",
    "price", "size", "sl", "tp", "sl_type",
    "status", "order_id", "notes"
]

def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def safe_float(x: Any, d: float = 0.0) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return d
        return v
    except Exception:
        return d

def load_json(path: str, default: Any = None) -> Any:
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

def ensure_csv_with_header(path: str, headers: List[str]) -> None:
    """
    Ensure CSV exists with the exact provided headers.
    If a file exists with different headers, rotate it to *_legacy.csv and create a fresh one.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(headers)
        return

    # Check first line (headers)
    try:
        with open(path, "r", encoding="utf-8") as f:
            first = f.readline().strip()
        existing = [h.strip() for h in first.split(",")] if first else []
    except Exception:
        existing = []

    if existing != headers:
        base, ext = os.path.splitext(path)
        rotated = f"{base}_legacy{ext}"
        os.replace(path, rotated)
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(headers)

def append_row(path: str, row: List[Any]) -> None:
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)

def current_open_positions(executions_csv: str) -> Dict[str, int]:
    """
    Very lightweight tracker: counts currently OPEN per symbol.
    Assumes v2 headers; if legacy file exists we ignore it because we rotate it.
    """
    if not os.path.exists(executions_csv):
        return {}
    pos: Dict[str, int] = {}
    with open(executions_csv, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            sym = (row.get("symbol") or "").upper()
            status = (row.get("status") or "").upper()
            if not sym:
                continue
            if status == "OPEN":
                pos[sym] = pos.get(sym, 0) + 1
            elif status == "CLOSED":
                pos[sym] = max(0, pos.get(sym, 0) - 1)
    return pos

def simulate_fill_price(plan: Dict[str, Any]) -> float:
    """
    Paper mode price source:
      - prefer plan['price']
      - else candidate['price']
      - else fallback 100.0 to avoid zeros
    Replace with live quotes in live mode.
    """
    p = safe_float(plan.get("price"), 0.0)
    if p > 0:
        return p
    cand = plan.get("candidate") or {}
    p = safe_float(cand.get("price"), 0.0)
    return p if p > 0 else 100.0

def update_aggregate_equity(aggregate_path: str, delta: float = 0.0) -> None:
    """
    Keeps equity / peak_equity fields. No realized PnL here (opens only).
    """
    agg = load_json(aggregate_path, default={"equity": 10000.0, "peak_equity": 10000.0})
    eq = safe_float(agg.get("equity"), 10000.0) + delta
    pk = max(eq, safe_float(agg.get("peak_equity"), eq))
    agg["equity"] = round(eq, 2)
    agg["peak_equity"] = round(pk, 2)
    save_json(aggregate_path, agg)

def main(args=None) -> int:
    ap = argparse.ArgumentParser(description="Executor v2")
    ap.add_argument("--approved", required=True, help="Path to approved.json from portfolio gate")
    ap.add_argument("--executions", default="reports/executions/executions.csv", help="CSV output path")
    ap.add_argument("--aggregate", default="reports/aggregate/aggregate_summary.json", help="Aggregate summary path")
    ap.add_argument("--max_position", type=int, default=1, help="Per-symbol cap (extra guard)")
    ap.add_argument("--mode", choices=["paper", "live"], default="paper", help="Execution mode")
    parsed = ap.parse_args(args=args)

    approved = load_json(parsed.approved, default=[])
    if not approved:
        print("No approved plans found. Nothing to execute.")
        return 0

    ensure_csv_with_header(parsed.executions, V2_HEADERS)
    open_pos = current_open_positions(parsed.executions)

    executed = 0
    for plan in approved:
        if not plan.get("enter", False):
            continue

        symbol = (plan.get("symbol") or "").upper()
        if not symbol:
            continue

        # per-symbol guard
        if open_pos.get(symbol, 0) >= parsed.max_position:
            # skip placing more positions for this symbol
            continue

        side = (plan.get("side") or "hold").lower()
        if side not in ("buy", "sell"):
            continue

        price = simulate_fill_price(plan)
        size  = safe_float(plan.get("size"), 0.0)
        sl    = plan.get("sl")
        tp    = plan.get("tp")
        sl_type = plan.get("sl_type") or "none"

        order_id = f"{symbol}-{int(time.time()*1000)}"
        row = [
            utcnow_iso(), parsed.mode, symbol, plan.get("tf") or "UNK", side,
            round(price, 8), round(size, 6),
            "" if sl is None else round(float(sl), 8),
            "" if tp is None else round(float(tp), 8),
            sl_type, "OPEN", order_id, "paper_fill"
        ]
        append_row(parsed.executions, row)
        open_pos[symbol] = open_pos.get(symbol, 0) + 1
        executed += 1

    print(f"Executed (opened): {executed}")

    # No realized PnL for opens; aggregate remains unchanged by default.
    # If you want to simulate commission/slippage, call update_aggregate_equity(delta) here.

    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
