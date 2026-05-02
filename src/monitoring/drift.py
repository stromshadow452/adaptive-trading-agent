"""
src/monitoring/drift.py
========================
SCOPUS Feature Drift Detection — Week 9 Pre-build (PSI Monitor).

Population Stability Index (PSI) computed per feature for detecting
distribution shift between training baseline and live/shadow data.

PSI thresholds:
    PSI < 0.10   = stable — no action needed
    PSI 0.10–0.25 = minor shift — monitor closely
    PSI > 0.25   = significant drift — trigger retraining

Usage:
    detector = DriftDetector(baseline_df)
    psi_scores = detector.compute_psi(current_df)
    alerts = detector.check_alerts(psi_scores)
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# PSI thresholds
PSI_STABLE   = 0.10
PSI_MONITOR  = 0.25   # above this → trigger retrain


class DriftDetector:
    """
    Monitors feature distribution drift using Population Stability Index.
    One instance per model lifecycle — built from training data baseline.
    """

    def __init__(
        self,
        baseline: pd.DataFrame,
        bins: int = 10,
        alert_threshold: float = PSI_MONITOR,
    ):
        """
        Args:
            baseline       : Training data feature DataFrame (reference distribution).
            bins           : Number of histogram bins for PSI calculation.
            alert_threshold: PSI above this value triggers a drift alert.
        """
        self.bins            = bins
        self.alert_threshold = alert_threshold
        self._bin_edges: Dict[str, np.ndarray] = {}
        self._baseline_pct:  Dict[str, np.ndarray] = {}

        self._fit(baseline)

    def _fit(self, df: pd.DataFrame):
        """Compute bin edges and baseline percentages from training data."""
        for col in df.columns:
            try:
                values = df[col].dropna().values
                if len(values) < 10:
                    continue
                _, edges = np.histogram(values, bins=self.bins)
                counts, _ = np.histogram(values, bins=edges)
                pct = counts / max(counts.sum(), 1)
                pct = np.clip(pct, 1e-8, None)
                self._bin_edges[col]    = edges
                self._baseline_pct[col] = pct / pct.sum()
            except Exception as e:
                logger.debug(f"[DriftDetector] Fit error for {col}: {e}")

    def compute_psi(self, current: pd.DataFrame) -> Dict[str, float]:
        """
        Compute PSI for each feature column present in both baseline and current.

        Returns:
            dict of feature_name → PSI score. Missing features get NaN.
        """
        results: Dict[str, float] = {}
        for col in self._bin_edges:
            if col not in current.columns:
                results[col] = float("nan")
                continue
            try:
                values = current[col].dropna().values
                if len(values) < 5:
                    results[col] = float("nan")
                    continue
                edges = self._bin_edges[col]
                counts, _ = np.histogram(values, bins=edges)
                act_pct = counts / max(counts.sum(), 1)
                act_pct = np.clip(act_pct, 1e-8, None)
                act_pct = act_pct / act_pct.sum()
                exp_pct = self._baseline_pct[col]
                psi = float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))
                results[col] = round(abs(psi), 6)
            except Exception as e:
                logger.debug(f"[DriftDetector] PSI error for {col}: {e}")
                results[col] = float("nan")
        return results

    def check_alerts(self, psi_scores: Dict[str, float]) -> List[Dict]:
        """
        Check PSI scores against threshold. Returns list of alert dicts.

        Returns:
            List of {"feature": str, "psi": float, "level": str}
            where level is "warning" or "critical".
        """
        alerts = []
        for feat, psi in psi_scores.items():
            if np.isnan(psi):
                continue
            if psi > self.alert_threshold:
                level = "critical"
                logger.warning(f"[DriftDetector] DRIFT CRITICAL: {feat} PSI={psi:.4f} > {self.alert_threshold}")
            elif psi > PSI_STABLE:
                level = "warning"
                logger.info(f"[DriftDetector] Drift warning: {feat} PSI={psi:.4f}")
            else:
                continue
            alerts.append({"feature": feat, "psi": psi, "level": level})
        return alerts

    def summary(self, psi_scores: Dict[str, float]) -> Dict:
        """Return summary statistics of PSI scores."""
        valid = [v for v in psi_scores.values() if not np.isnan(v)]
        if not valid:
            return {"n_features": 0, "mean_psi": 0.0, "max_psi": 0.0,
                    "n_stable": 0, "n_warning": 0, "n_drifted": 0}
        return {
            "n_features": len(valid),
            "mean_psi":   round(float(np.mean(valid)), 6),
            "max_psi":    round(float(np.max(valid)), 6),
            "n_stable":   sum(1 for v in valid if v < PSI_STABLE),
            "n_warning":  sum(1 for v in valid if PSI_STABLE <= v < self.alert_threshold),
            "n_drifted":  sum(1 for v in valid if v >= self.alert_threshold),
        }


def compute_psi_single(expected: np.ndarray, actual: np.ndarray,
                        bins: int = 10) -> float:
    """
    Standalone PSI computation for a single feature.
    Useful for quick checks without creating a DriftDetector instance.
    """
    if len(expected) < 5 or len(actual) < 5:
        return 0.0
    _, edges   = np.histogram(expected, bins=bins)
    exp_counts, _ = np.histogram(expected, bins=edges)
    act_counts, _ = np.histogram(actual,   bins=edges)
    exp_pct = np.clip(exp_counts / max(exp_counts.sum(), 1), 1e-8, None)
    act_pct = np.clip(act_counts / max(act_counts.sum(), 1), 1e-8, None)
    exp_pct = exp_pct / exp_pct.sum()
    act_pct = act_pct / act_pct.sum()
    return float(abs(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))))
