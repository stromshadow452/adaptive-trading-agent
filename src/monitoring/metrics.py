"""
src/monitoring/metrics.py
=========================
Shadow Trading Metrics — Week 5 Implementation.

Reads fills.jsonl produced by PaperExecutor and computes all 10
monitoring metrics defined in the shadow trading plan.

Usage:
    metrics = ShadowMetrics("logs/shadow/fills.jsonl")
    snapshot = metrics.compute()
    gate_result = metrics.gate_check()  # True = PASS, False = FAIL
"""
from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

import numpy as np

import logging
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gate thresholds (from shadow trading plan §5)
# ---------------------------------------------------------------------------
GATE_CRITERIA = {
    "min_trades":           200,
    "min_shadow_days":      90,
    "min_profit_factor":    1.30,
    "max_drawdown_pct":     12.0,
    "min_win_rate":         0.45,
    "min_sharpe":           1.0,
    "max_avg_slippage_pip": 1.0,   # warn threshold
    "max_latency_ms":       500.0,
    "allowed_global_trips": 0,
    "max_error_rate_pct":   0.1,
}


class ShadowMetrics:
    """
    Computes performance metrics from the shadow fills JSONL log.
    All computation is done on closed trades (event == 'CLOSE').
    """

    def __init__(self, fills_log: str = "logs/shadow/fills.jsonl"):
        self.fills_log = fills_log

    # ------------------------------------------------------------------ #
    # Data loading                                                         #
    # ------------------------------------------------------------------ #

    def _load_events(self, event_types=("CLOSE",)) -> List[dict]:
        if not os.path.exists(self.fills_log):
            return []
        events = []
        try:
            with open(self.fills_log, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        if record.get("event") in event_types:
                            events.append(record)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.error(f"[ShadowMetrics] Failed to read log: {e}")
        return events

    def _load_all_fills(self) -> List[dict]:
        return self._load_events(event_types=("FILL", "CLOSE"))

    # ------------------------------------------------------------------ #
    # Core metric computations                                             #
    # ------------------------------------------------------------------ #

    def _compute_equity_curve(self, trades: List[dict]) -> List[float]:
        equity = [10_000.0]
        for t in trades:
            pnl = float(t.get("pnl_usd") or 0.0)
            equity.append(equity[-1] + pnl)
        return equity

    def _max_drawdown(self, equity_curve: List[float]) -> float:
        if not equity_curve:
            return 0.0
        peak = equity_curve[0]
        max_dd = 0.0
        for val in equity_curve:
            peak = max(peak, val)
            dd = (peak - val) / max(peak, 1e-9) * 100.0
            max_dd = max(max_dd, dd)
        return round(max_dd, 4)

    def _sharpe(self, trades: List[dict], annualize_days: int = 252) -> Optional[float]:
        if len(trades) < 5:
            return None
        pnls = [float(t.get("pnl_usd") or 0.0) for t in trades]
        mean = np.mean(pnls)
        std  = np.std(pnls, ddof=1)
        if std == 0:
            return None
        # Daily-equivalent sample rate
        try:
            t_start = datetime.fromisoformat(trades[0]["ts"].replace("Z", "+00:00"))
            t_end   = datetime.fromisoformat(trades[-1]["ts"].replace("Z", "+00:00"))
            days    = max((t_end - t_start).days, 1)
            tpd     = len(trades) / days   # trades per day
        except Exception:
            tpd = 1.0
        annualized = (mean / std) * math.sqrt(tpd * annualize_days)
        return round(annualized, 4)

    def _profit_factor(self, trades: List[dict]) -> float:
        gross_profit = sum(float(t.get("pnl_usd") or 0) for t in trades
                          if (t.get("pnl_usd") or 0) > 0)
        gross_loss   = abs(sum(float(t.get("pnl_usd") or 0) for t in trades
                              if (t.get("pnl_usd") or 0) <= 0))
        return round(gross_profit / max(gross_loss, 0.001), 4)

    def _shadow_days(self, all_events: List[dict]) -> float:
        if len(all_events) < 2:
            return 0.0
        try:
            t0 = datetime.fromisoformat(all_events[0]["ts"].replace("Z", "+00:00"))
            t1 = datetime.fromisoformat(all_events[-1]["ts"].replace("Z", "+00:00"))
            return max((t1 - t0).total_seconds() / 86400.0, 0.0)
        except Exception:
            return 0.0

    # ------------------------------------------------------------------ #
    # Full metric snapshot                                                 #
    # ------------------------------------------------------------------ #

    def compute(self) -> Dict[str, Any]:
        trades     = self._load_events(event_types=("CLOSE",))
        all_events = self._load_events(event_types=("FILL", "CLOSE"))
        wins       = [t for t in trades if (t.get("pnl_usd") or 0) > 0]
        losses     = [t for t in trades if (t.get("pnl_usd") or 0) <= 0]
        equity_curve = self._compute_equity_curve(trades)

        slips = [float(t.get("slippage_pips") or 0) for t in all_events
                 if t.get("event") == "FILL"]
        lats  = [float(t.get("latency_ms") or 0) for t in all_events
                 if t.get("event") == "FILL"]

        return {
            "n_trades":            len(trades),
            "n_wins":              len(wins),
            "n_losses":            len(losses),
            "win_rate":            round(len(wins) / max(len(trades), 1), 4),
            "profit_factor":       self._profit_factor(trades),
            "max_drawdown_pct":    self._max_drawdown(equity_curve),
            "sharpe_annualized":   self._sharpe(trades),
            "net_pnl_usd":         round(sum(float(t.get("pnl_usd") or 0) for t in trades), 2),
            "avg_slippage_pips":   round(float(np.mean(slips)) if slips else 0.0, 3),
            "p95_latency_ms":      round(float(np.percentile(lats, 95)) if lats else 0.0, 2),
            "avg_latency_ms":      round(float(np.mean(lats)) if lats else 0.0, 2),
            "avg_r_multiple":      round(float(np.mean([t.get("r_multiple") or 0
                                                        for t in trades if t.get("r_multiple")])), 4)
                                   if trades else 0.0,
            "shadow_days":         round(self._shadow_days(all_events), 1),
            "computed_at":         datetime.now(timezone.utc).isoformat(),
        }

    def compute_rolling(self, window_days: int = 30) -> Dict[str, Any]:
        """Compute metrics over the last `window_days` only."""
        trades = self._load_events(event_types=("CLOSE",))
        cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
        trades = [t for t in trades if _parse_ts(t.get("ts", "")) >= cutoff]
        if not trades:
            return {"window_days": window_days, "n_trades": 0}
        wins   = [t for t in trades if (t.get("pnl_usd") or 0) > 0]
        slips  = [float(t.get("slippage_pips") or 0) for t in trades]
        return {
            "window_days":       window_days,
            "n_trades":          len(trades),
            "win_rate":          round(len(wins) / max(len(trades), 1), 4),
            "profit_factor":     self._profit_factor(trades),
            "net_pnl_usd":       round(sum(float(t.get("pnl_usd") or 0) for t in trades), 2),
            "avg_slippage_pips": round(float(np.mean(slips)) if slips else 0.0, 3),
            "computed_at":       datetime.now(timezone.utc).isoformat(),
        }

    # ------------------------------------------------------------------ #
    # 10-Criterion Go / No-Go Gate                                         #
    # ------------------------------------------------------------------ #

    def gate_check(self, cb_global_trips: int = 0, error_rate_pct: float = 0.0) -> Dict[str, Any]:
        """
        Run the 10-criterion shadow → live gating check.
        Returns dict with per-criterion pass/fail and an overall PASS/FAIL.
        """
        m = self.compute()

        def chk(criterion, value, threshold, op=">="):
            passed = (value >= threshold if op == ">="
                      else value <= threshold if op == "<="
                      else value == threshold)
            return {
                "criterion": criterion,
                "value":     round(value, 4) if isinstance(value, float) else value,
                "threshold": threshold,
                "op":        op,
                "passed":    passed,
            }

        sharpe = m.get("sharpe_annualized") or 0.0
        results = [
            chk("min_trades",         m["n_trades"],          GATE_CRITERIA["min_trades"],         ">="),
            chk("min_shadow_days",    m["shadow_days"],        GATE_CRITERIA["min_shadow_days"],    ">="),
            chk("profit_factor",      m["profit_factor"],      GATE_CRITERIA["min_profit_factor"],  ">="),
            chk("max_drawdown_pct",   m["max_drawdown_pct"],   GATE_CRITERIA["max_drawdown_pct"],   "<="),
            chk("win_rate",           m["win_rate"],           GATE_CRITERIA["min_win_rate"],       ">="),
            chk("sharpe_annualized",  sharpe,                  GATE_CRITERIA["min_sharpe"],         ">="),
            chk("avg_slippage_pips",  m["avg_slippage_pips"],  GATE_CRITERIA["max_avg_slippage_pip"],"<="),
            chk("p95_latency_ms",     m["p95_latency_ms"],     GATE_CRITERIA["max_latency_ms"],     "<="),
            chk("global_cb_trips",    float(cb_global_trips),  float(GATE_CRITERIA["allowed_global_trips"]),"<="),
            chk("error_rate_pct",     float(error_rate_pct),   GATE_CRITERIA["max_error_rate_pct"], "<="),
        ]

        all_pass = all(r["passed"] for r in results)
        failed   = [r["criterion"] for r in results if not r["passed"]]

        return {
            "overall":   "PASS" if all_pass else "FAIL",
            "all_pass":  all_pass,
            "failed":    failed,
            "criteria":  results,
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_ts(ts_str: str) -> datetime:
    try:
        return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except Exception:
        return datetime.min.replace(tzinfo=timezone.utc)
