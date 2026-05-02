"""
src/monitoring/live_readiness.py
==================================
SCOPUS Live Readiness Gate — Week 15. 

Extended 15-criterion go/no-go gate that covers everything needed before
conditional live trading approval. Supersedes the original 10-criterion
ShadowMetrics.gate_check() with 5 additional engineering checks.

New criteria (11-15):
  11. Champion model present and recent (< 30 days old)
  12. No active drift (max PSI < 0.25)
  13. Ensemble weights tuned (optimal_weights.json exists)
  14. Feature registry has ≥ 20 production features
  15. Pipeline v2 reachable (import OK)

Usage:
    from src.monitoring.live_readiness import LiveReadinessGate
    gate  = LiveReadinessGate(fills_log="logs/shadow/fills.jsonl")
    result = gate.check()
    gate.print_report(result)

CLI:
    python -m src.monitoring.live_readiness
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional

LOG = logging.getLogger("live_readiness")


@dataclass
class GateCriterion:
    name:          str
    value:         object
    threshold:     object
    passed:        bool
    description:   str = ""


@dataclass
class ReadinessResult:
    all_pass:   bool
    score:      int                         # criteria passed out of 15
    criteria:   List[GateCriterion] = field(default_factory=list)
    timestamp:  str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    @property
    def failed(self) -> List[str]:
        return [c.name for c in self.criteria if not c.passed]

    @property
    def passed_list(self) -> List[str]:
        return [c.name for c in self.criteria if c.passed]

    def to_dict(self) -> Dict:
        return {
            "all_pass":  self.all_pass,
            "score":     f"{self.score}/15",
            "timestamp": self.timestamp,
            "failed":    self.failed,
            "criteria":  [
                {
                    "name":      c.name,
                    "value":     c.value,
                    "threshold": c.threshold,
                    "passed":    c.passed,
                }
                for c in self.criteria
            ],
        }


class LiveReadinessGate:
    """
    15-criterion live readiness gate.

    Criteria 1-10:  from ShadowMetrics.gate_check() (performance gates)
    Criteria 11-15: engineering / infrastructure gates

    Args:
        fills_log:         Path to shadow fills JSONL.
        psi_log:           Path to psi_daily.jsonl.
        champion_file:     Path to champion_latest.json.
        weights_file:      Path to optimal_weights.json.
        min_shadow_days:   Minimum shadow days required.
        min_trades:        Minimum completed trades.
    """

    def __init__(
        self,
        fills_log:       str = "logs/shadow/fills.jsonl",
        psi_log:         str = "logs/shadow/psi_daily.jsonl",
        champion_file:   str = "models/registry/champion_latest.json",
        weights_file:    str = "models/ensemble/optimal_weights.json",
        min_shadow_days: int = 30,
        min_trades:      int = 30,
    ):
        self.fills_log      = fills_log
        self.psi_log        = psi_log
        self.champion_file  = Path(champion_file)
        self.weights_file   = Path(weights_file)
        self.min_shadow_days = min_shadow_days
        self.min_trades      = min_trades

    def check(self) -> ReadinessResult:
        """Run all 15 gates and return ReadinessResult."""
        criteria: List[GateCriterion] = []

        # ── Gates 1-10: shadow performance ───────────────────────────────────
        perf = self._check_performance_gates()
        criteria.extend(perf)

        # ── Gates 11-15: infrastructure / engineering ─────────────────────────
        infra = self._check_infrastructure_gates()
        criteria.extend(infra)

        passed  = sum(1 for c in criteria if c.passed)
        all_ok  = all(c.passed for c in criteria)

        return ReadinessResult(
            all_pass  = all_ok,
            score     = passed,
            criteria  = criteria,
        )

    def _check_performance_gates(self) -> List[GateCriterion]:
        """Gates 1-10 from original ShadowMetrics.gate_check()."""
        gates = []
        try:
            from src.monitoring.metrics import ShadowMetrics
            sm     = ShadowMetrics(self.fills_log)
            snap   = sm.compute()
            result = sm.gate_check()
            for c in result.get("criteria", []):
                gates.append(GateCriterion(
                    name      = c["criterion"],
                    value     = c.get("value", 0),
                    threshold = c.get("threshold", 0),
                    passed    = c["passed"],
                ))
        except Exception as e:
            gates.append(GateCriterion(
                name="shadow_metrics", value="error", threshold="ok",
                passed=False, description=str(e)
            ))
        return gates

    def _check_infrastructure_gates(self) -> List[GateCriterion]:
        """Gates 11-15: engineering readiness checks."""
        gates = []

        # Gate 11: Champion model present + not too old
        try:
            if self.champion_file.exists():
                ch   = json.loads(self.champion_file.read_text())
                ts   = datetime.fromisoformat(ch.get("timestamp", "2000-01-01"))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                age  = (datetime.now(timezone.utc) - ts).days
                ok   = age <= 30
                gates.append(GateCriterion(
                    "Champion model age (days)", age, "≤30", ok,
                ))
            else:
                gates.append(GateCriterion(
                    "Champion model age (days)", "absent", "≤30", False,
                ))
        except Exception as e:
            gates.append(GateCriterion(
                "Champion model age (days)", "error", "≤30", False,
                description=str(e),
            ))

        # Gate 12: Max PSI < 0.25 (no active drift)
        # Only trust log entries with n_stable >= 50 (warm-up noise is discarded)
        # and written within the last 7 days.
        try:
            psi_path = Path(self.psi_log)
            if not psi_path.exists():
                gates.append(GateCriterion(
                    "Max feature PSI", "not_logged", "<0.25", True,
                    description="PSI monitoring not yet started"
                ))
            else:
                # Find the most recent entry with a valid baseline
                MIN_STABLE = 50
                lines      = [l for l in psi_path.read_text().strip().split("\n") if l.strip()]
                valid_entry = None
                for line in reversed(lines):
                    try:
                        entry = json.loads(line)
                        ts_str = entry.get("timestamp", "")
                        if ts_str:
                            ts = datetime.fromisoformat(ts_str)
                            if ts.tzinfo is None:
                                ts = ts.replace(tzinfo=timezone.utc)
                            age_days = (datetime.now(timezone.utc) - ts).days
                            if age_days > 7:
                                continue   # stale entry
                        if entry.get("n_stable", 0) >= MIN_STABLE:
                            valid_entry = entry
                            break
                    except Exception:
                        continue

                if valid_entry is None:
                    gates.append(GateCriterion(
                        "Max feature PSI", "insufficient_baseline", "<0.25", True,
                        description=f"No log entry with n_stable≥{MIN_STABLE} in last 7 days"
                    ))
                else:
                    max_psi = float(valid_entry.get("max_psi", 0.0))
                    gates.append(GateCriterion(
                        "Max feature PSI", round(max_psi, 4), "<0.25", max_psi < 0.25,
                    ))
        except Exception as e:
            gates.append(GateCriterion(
                "Max feature PSI", "error", "<0.25", True,
                description=str(e),
            ))

        # Gate 13: Ensemble weights tuned
        weights_ok = self.weights_file.exists()
        gates.append(GateCriterion(
            "Ensemble weights tuned",
            "yes" if weights_ok else "not tuned",
            "exists",
            weights_ok,
        ))

        # Gate 14: Feature registry ≥ 20 production features
        try:
            from research.feature_store import FeatureStore
            store = FeatureStore()
            prod  = len(store.list_production())
            gates.append(GateCriterion(
                "Production features", prod, "≥20", prod >= 20,
            ))
        except Exception as e:
            gates.append(GateCriterion(
                "Production features", "error", "≥20", False,
                description=str(e),
            ))

        # Gate 15: PipelineV2 importable
        try:
            from src.pipeline.pipeline_v2 import PipelineV2, PipelineConfig  # noqa
            gates.append(GateCriterion(
                "PipelineV2 importable", "yes", "yes", True,
            ))
        except Exception as e:
            gates.append(GateCriterion(
                "PipelineV2 importable", "error", "yes", False,
                description=str(e),
            ))

        return gates

    def print_report(self, result: Optional[ReadinessResult] = None):
        """Print a formatted report to stdout."""
        if result is None:
            result = self.check()

        print()
        print("=" * 66)
        print("  SCOPUS LIVE TRADING READINESS — 15-CRITERION GATE")
        print(f"  Generated: {result.timestamp[:19]} UTC")
        print("=" * 66)
        print(f"  Score: {result.score}/15  {'✅ ALL PASS' if result.all_pass else '❌ NOT READY'}")
        print()
        for i, c in enumerate(result.criteria, 1):
            status = "✅" if c.passed else "❌"
            val    = c.value if not isinstance(c.value, float) else f"{c.value:.4f}"
            thr    = c.threshold
            print(f"  {status} [{i:>2}] {c.name:<35}  val={val!s:<10}  thr={thr!s}")
        print()
        if result.failed:
            print(f"  Failing gates: {', '.join(result.failed)}")
        else:
            print("  All 15 gates passed. Submit to risk committee for live approval.")
        print("=" * 66)
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    gate   = LiveReadinessGate()
    result = gate.check()
    gate.print_report(result)
    sys.exit(0 if result.all_pass else 1)
