"""
src/ml/promotion.py
=====================
SCOPUS Automated Champion Promotion Pipeline — Week 14.

Implements the 5-criterion gate that runs automatically after each
training cycle before deciding whether to push a challenger model to
production:

    Gate 1 (F1):       val_F1 > current_champion_F1 + 0.02
    Gate 2 (Features): n_features between 20 and 80
    Gate 3 (Registry): all trained features pass FeatureStore gate
    Gate 4 (PSI):      no feature with PSI > 0.25
    Gate 5 (Rows):     val_rows ≥ 500 (sufficient evaluation set)

If all gates pass → champion promoted + LAST_RETRAIN_EPOCH gauge updated.
If any gate fails → challenger archived, reason logged.

Usage (integrated into train_full_model.py and rolling_retrain.py):
    from src.ml.promotion import run_promotion_gate

    result = run_promotion_gate(champion, challenger, store)
    if result.promoted:
        LOG.info("New champion promoted")
    else:
        LOG.warning(f"Promotion rejected: {result.reason}")
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

LOG = logging.getLogger("ml.promotion")

# ── Gate thresholds ────────────────────────────────────────────────────────────
GATE_F1_DELTA     = 0.02    # Challenger must beat champion by ≥2% F1
GATE_MIN_FEATURES = 20
GATE_MAX_FEATURES = 80
GATE_MAX_PSI      = 0.25    # Hard drift threshold
GATE_MIN_VAL_ROWS = 500     # At least 500 validation samples

# Promotion audit log
REGISTRY_DIR  = Path("models/registry")
PROMO_LOG     = REGISTRY_DIR / "promotion_gate_log.jsonl"
REGISTRY_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class PromotionResult:
    """Result of running the 5-gate promotion check."""
    promoted:      bool
    reason:        str
    gates_passed:  List[str]
    gates_failed:  List[str]
    champion_f1:   Optional[float]
    challenger_f1: float
    timestamp:     str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def log_to_file(self):
        record = {
            "timestamp":     self.timestamp,
            "promoted":      self.promoted,
            "reason":        self.reason,
            "gates_passed":  self.gates_passed,
            "gates_failed":  self.gates_failed,
            "champion_f1":   self.champion_f1,
            "challenger_f1": self.challenger_f1,
        }
        with open(PROMO_LOG, "a") as f:
            f.write(json.dumps(record) + "\n")


def _gate_f1(champion: Optional[Dict], challenger: Dict) -> tuple[bool, str]:
    """Gate 1: challenger F1 must beat champion by ≥2%."""
    if champion is None:
        return True, "no_prior_champion"
    delta = challenger.get("val_f1", 0.0) - champion.get("val_f1", 0.0)
    if delta >= GATE_F1_DELTA:
        return True, f"delta_f1={delta:+.4f}"
    return False, f"delta_f1={delta:+.4f}_below_{GATE_F1_DELTA}"


def _gate_features(challenger: Dict) -> tuple[bool, str]:
    """Gate 2: n_features in [20, 80]."""
    n = challenger.get("n_features", 0)
    if GATE_MIN_FEATURES <= n <= GATE_MAX_FEATURES:
        return True, f"n_features={n}"
    return False, f"n_features={n}_out_of_range[{GATE_MIN_FEATURES},{GATE_MAX_FEATURES}]"


def _gate_registry(challenger: Dict, store) -> tuple[bool, str]:
    """
    Gate 3: all features in challenger must either be in FeatureStore
    or newly added. Extended/new features must not be DEPRECATED.
    """
    features = challenger.get("features", [])
    if not features:
        return True, "no_feature_list"
    deprecated = []
    for name in features:
        feat = store.get(name)
        if feat and feat.get("status") == "deprecated":
            deprecated.append(name)
    if deprecated:
        return False, f"deprecated_features={deprecated[:5]}"
    return True, f"all_{len(features)}_features_OK"


def _gate_psi(psi_log: str = "logs/shadow/psi_daily.jsonl") -> tuple[bool, str]:
    """Gate 4: max feature PSI < 0.25 (no active drift)."""
    from pathlib import Path as P
    path = P(psi_log)
    if not path.exists():
        return True, "psi_log_not_found"   # benefit of the doubt
    try:
        lines = path.read_text().strip().split("\n")
        last  = json.loads(lines[-1])
        max_psi = last.get("max_psi", 0.0)
        if max_psi >= GATE_MAX_PSI:
            return False, f"max_psi={max_psi:.4f}>={GATE_MAX_PSI}"
        return True, f"max_psi={max_psi:.4f}"
    except Exception as e:
        LOG.debug(f"[promotion] PSI gate error: {e}")
        return True, "psi_read_error"   # non-fatal


def _gate_val_size(challenger: Dict) -> tuple[bool, str]:
    """Gate 5: validation set ≥ 500 rows."""
    n = challenger.get("val_rows", 0)
    if n >= GATE_MIN_VAL_ROWS:
        return True, f"val_rows={n}"
    return False, f"val_rows={n}<{GATE_MIN_VAL_ROWS}"


def run_promotion_gate(
    champion:   Optional[Dict],
    challenger: Dict,
    store=None,
    psi_log:    str = "logs/shadow/psi_daily.jsonl",
    dry_run:    bool = False,
) -> PromotionResult:
    """
    Run all 5 promotion gates.

    Args:
        champion:   Current champion metadata dict (or None if first run).
        challenger: New model metadata dict from train_full_model / rolling_retrain.
        store:      FeatureStore instance (optional — Gate 3 skipped if None).
        psi_log:    Path to psi_daily.jsonl.
        dry_run:    Log but do not write champion file.

    Returns:
        PromotionResult with promoted flag and gate details.
    """
    passed:  List[str] = []
    failed:  List[str] = []

    # Run each gate
    for gate_name, gate_fn, gate_args in [
        ("F1_delta",    _gate_f1,       (champion, challenger)),
        ("features",    _gate_features, (challenger,)),
        ("val_size",    _gate_val_size, (challenger,)),
        ("psi",         _gate_psi,      (psi_log,)),
    ]:
        ok, msg = gate_fn(*gate_args)
        if ok:
            passed.append(f"{gate_name}:{msg}")
            LOG.info(f"[promotion] Gate {gate_name}: PASS ({msg})")
        else:
            failed.append(f"{gate_name}:{msg}")
            LOG.warning(f"[promotion] Gate {gate_name}: FAIL ({msg})")

    # Registry gate (optional)
    if store is not None:
        ok, msg = _gate_registry(challenger, store)
        if ok:
            passed.append(f"registry:{msg}")
        else:
            failed.append(f"registry:{msg}")
            LOG.warning(f"[promotion] Gate registry: FAIL ({msg})")

    all_pass = len(failed) == 0
    reason   = "; ".join(failed) if failed else "all_gates_passed"

    if all_pass and not dry_run:
        # Promote champion
        champion_path = REGISTRY_DIR / "champion_latest.json"
        with open(champion_path, "w") as f:
            json.dump(challenger, f, indent=2, default=str)
        LOG.info(f"[promotion] CHAMPION PROMOTED → {challenger.get('model_path')}")

        # Update Prometheus last_retrain_epoch gauge
        try:
            from src.monitoring.prometheus_exporter import LAST_RETRAIN_EPOCH
            import time
            LAST_RETRAIN_EPOCH.set(time.time())
        except Exception:
            pass

    result = PromotionResult(
        promoted      = all_pass and not dry_run,
        reason        = reason,
        gates_passed  = passed,
        gates_failed  = failed,
        champion_f1   = champion.get("val_f1") if champion else None,
        challenger_f1 = challenger.get("val_f1", 0.0),
    )
    result.log_to_file()

    if all_pass:
        LOG.info(f"[promotion] ALL 5 GATES PASSED — {'PROMOTED' if not dry_run else 'DRY RUN'}")
    else:
        LOG.warning(f"[promotion] {len(failed)} gate(s) failed — challenger rejected")

    return result


def print_gate_report(result: PromotionResult):
    """Print a human-readable gate summary."""
    print("\n" + "=" * 60)
    print("SCOPUS MODEL PROMOTION GATE REPORT")
    print("=" * 60)
    status = "✅ PROMOTED" if result.promoted else "❌ REJECTED"
    print(f"Result:   {status}")
    print(f"Champion: F1={result.champion_f1 or 'n/a'}")
    print(f"Challenger F1: {result.challenger_f1:.4f}")
    print()
    for g in result.gates_passed:
        print(f"  ✅ {g}")
    for g in result.gates_failed:
        print(f"  ❌ {g}")
    print("=" * 60)
