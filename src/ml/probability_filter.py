from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
        if math.isnan(result) or math.isinf(result):
            return default
        return result
    except Exception:
        return default


@dataclass
class ProbabilityTradeRecord:
    symbol: str
    strategy: str
    regime: str
    session: str
    signal_quality: float
    edge_score: float
    ml_confidence: float
    boll_z: float
    atr_pctile: float
    adx: float
    r_multiple: float
    label: int
    timestamp: str = ""


_NUMERIC_FEATURES = [
    "signal_quality",
    "edge_score",
    "ml_confidence",
    "boll_z",
    "atr_pctile",
    "adx",
]

_NUMERIC_SCALES = {
    "signal_quality": 1.0,
    "edge_score": 1.0,
    "ml_confidence": 1.0,
    "boll_z": 3.0,
    "atr_pctile": 1.0,
    "adx": 50.0,
}


class ProbabilityBasedMLFilter:
    """
    Probability filter that learns from historical closed trades.

    This module does not generate signals. It estimates the probability that
    an already-generated trade idea will achieve at least +1R before failing
    to -1R, using labeled trade memory plus a conservative fallback blend from
    the existing quality metrics while the sample is still small.
    """

    def __init__(
        self,
        memory_path: str = "logs/shadow/ml_probability_memory.jsonl",
        min_samples: int = 20,
        k_neighbors: int = 15,
        live_min_samples: int = 200,
    ) -> None:
        self.memory_path = memory_path
        self.min_samples = min_samples
        self.k_neighbors = k_neighbors
        self.live_min_samples = live_min_samples
        self._records: List[ProbabilityTradeRecord] = []
        self._load()

    @staticmethod
    def label_trade_result(r_multiple: object) -> Optional[int]:
        r_value = _safe_float(r_multiple, 0.0)
        if r_value >= 1.0:
            return 1
        if r_value <= -1.0:
            return 0
        return None

    @classmethod
    def build_training_frame(cls, closed_trades: List[Dict]) -> pd.DataFrame:
        rows: List[Dict] = []
        for trade in closed_trades:
            label = cls.label_trade_result(trade.get("r_multiple"))
            if label is None:
                continue
            rows.append(
                {
                    "signal_quality": _safe_float(trade.get("signal_quality"), 0.0),
                    "edge_score": _safe_float(trade.get("edge_score"), 0.0),
                    "ml_confidence": _safe_float(
                        trade.get("ml_conf_calibrated", trade.get("ml_confidence", 0.0)),
                        0.0,
                    ),
                    "boll_z": _safe_float(trade.get("boll_z"), 0.0),
                    "atr_pctile": _safe_float(trade.get("atr_pctile"), 0.5),
                    "regime": str(trade.get("regime", "UNKNOWN")).upper(),
                    "strategy": str(trade.get("strategy", "UNKNOWN")).upper(),
                    "session": str(trade.get("session", "OFF")).upper(),
                    "adx": _safe_float(trade.get("adx"), 0.0),
                    "label": label,
                    "r_multiple": _safe_float(trade.get("r_multiple"), 0.0),
                }
            )
        return pd.DataFrame(rows)

    def record_closed_trade(self, trade: Dict) -> bool:
        label = self.label_trade_result(trade.get("r_multiple"))
        if label is None:
            return False

        record = ProbabilityTradeRecord(
            symbol=str(trade.get("symbol", "UNKNOWN")),
            strategy=str(trade.get("strategy", "UNKNOWN")).upper(),
            regime=str(trade.get("regime", "UNKNOWN")).upper(),
            session=str(trade.get("session", "OFF")).upper(),
            signal_quality=_safe_float(trade.get("signal_quality"), 0.0),
            edge_score=_safe_float(trade.get("edge_score"), 0.0),
            ml_confidence=_safe_float(
                trade.get("ml_conf_calibrated", trade.get("ml_confidence", 0.0)),
                0.0,
            ),
            boll_z=_safe_float(trade.get("boll_z"), 0.0),
            atr_pctile=_safe_float(trade.get("atr_pctile"), 0.5),
            adx=_safe_float(trade.get("adx"), 0.0),
            r_multiple=_safe_float(trade.get("r_multiple"), 0.0),
            label=label,
            timestamp=str(trade.get("timestamp_exit", "")),
        )

        self._records.append(record)
        if len(self._records) % 5 == 0:
            self._save()
        return True

    def predict_probability(self, features: Dict) -> Tuple[float, Dict]:
        fallback_probability = self._fallback_probability(features)
        total_samples = len(self._records)
        live_filter_active = total_samples >= self.live_min_samples

        if not self._records:
            return fallback_probability, {
                "source": "fallback_only",
                "sample_count": 0,
                "neighbor_count": 0,
                "fallback_probability": fallback_probability,
                "live_filter_active": False,
            }

        query_vector = self._numeric_vector(features)
        distances: List[Tuple[float, ProbabilityTradeRecord]] = []
        query_strategy = str(features.get("strategy", "UNKNOWN")).upper()
        query_regime = str(features.get("regime", "UNKNOWN")).upper()
        query_session = str(features.get("session", "OFF")).upper()

        for record in self._records:
            record_vector = self._numeric_vector(asdict(record))
            distance = float(np.linalg.norm(record_vector - query_vector))
            if record.strategy != query_strategy:
                distance += 0.35
            if record.regime != query_regime:
                distance += 0.20
            if record.session != query_session:
                distance += 0.10
            distances.append((distance, record))

        distances.sort(key=lambda item: item[0])
        nearest = distances[: min(self.k_neighbors, len(distances))]
        if not nearest:
            return fallback_probability, {
                "source": "fallback_only",
                "sample_count": total_samples,
                "neighbor_count": 0,
                "fallback_probability": fallback_probability,
                "live_filter_active": live_filter_active,
            }

        weights = np.array([1.0 / (dist + 0.05) for dist, _ in nearest], dtype=float)
        labels = np.array([record.label for _, record in nearest], dtype=float)

        if weights.sum() <= 0:
            knn_probability = float(labels.mean()) if len(labels) else fallback_probability
        else:
            knn_probability = float(np.dot(weights, labels) / weights.sum())

        confidence_in_memory = min(0.85, total_samples / max(float(self.min_samples), 1.0) / 2.0)
        blended_probability = (
            fallback_probability * (1.0 - confidence_in_memory)
            + knn_probability * confidence_in_memory
        )
        blended_probability = float(min(max(blended_probability, 0.0), 1.0))

        return blended_probability, {
            "source": "blended_knn",
            "sample_count": total_samples,
            "neighbor_count": len(nearest),
            "fallback_probability": fallback_probability,
            "knn_probability": round(knn_probability, 6),
            "memory_weight": round(confidence_in_memory, 6),
            "live_filter_active": live_filter_active,
        }

    def summary(self) -> Dict[str, float]:
        if not self._records:
            return {"labeled_examples": 0, "positive_rate": 0.0, "live_filter_active": False}
        labels = [record.label for record in self._records]
        return {
            "labeled_examples": len(self._records),
            "positive_rate": round(float(np.mean(labels)), 4),
            "live_filter_active": len(self._records) >= self.live_min_samples,
            "live_min_samples": self.live_min_samples,
        }

    @property
    def total_samples(self) -> int:
        return len(self._records)

    def is_live_filter_active(self) -> bool:
        return self.total_samples >= self.live_min_samples

    @classmethod
    def build_analysis_dataset(cls, closed_trades: List[Dict]) -> pd.DataFrame:
        rows: List[Dict] = []
        for trade in closed_trades:
            label = cls.label_trade_result(trade.get("r_multiple"))
            rows.append(
                {
                    "symbol": str(trade.get("symbol", "UNKNOWN")),
                    "strategy": str(trade.get("strategy", "UNKNOWN")).upper(),
                    "regime": str(trade.get("regime", "UNKNOWN")).upper(),
                    "session": str(trade.get("session", "OFF")).upper(),
                    "signal_quality": _safe_float(trade.get("signal_quality"), 0.0),
                    "edge_score": _safe_float(trade.get("edge_score"), 0.0),
                    "ml_confidence": _safe_float(
                        trade.get("ml_conf_calibrated", trade.get("ml_confidence", 0.0)),
                        0.0,
                    ),
                    "boll_z": _safe_float(trade.get("boll_z"), 0.0),
                    "atr_pctile": _safe_float(trade.get("atr_pctile"), 0.5),
                    "adx": _safe_float(trade.get("adx"), 0.0),
                    "probability_of_success": _safe_float(trade.get("probability_of_success"), 0.0),
                    "actual_outcome": label,
                    "r_multiple": _safe_float(trade.get("r_multiple"), 0.0),
                }
            )
        return pd.DataFrame(rows)

    @classmethod
    def bucket_analysis(cls, dataset: pd.DataFrame) -> List[Dict]:
        if dataset.empty:
            return []

        labeled = dataset[dataset["actual_outcome"].notna()].copy()
        if labeled.empty:
            return []

        bins = [0.50, 0.55, 0.60, 0.65, 0.70, float("inf")]
        labels = ["0.50-0.55", "0.55-0.60", "0.60-0.65", "0.65-0.70", "0.70+"]
        labeled["probability_bucket"] = pd.cut(
            labeled["probability_of_success"],
            bins=bins,
            labels=labels,
            right=False,
            include_lowest=True,
        )

        output: List[Dict] = []
        for label_name in labels:
            bucket = labeled[labeled["probability_bucket"] == label_name]
            if bucket.empty:
                output.append(
                    {
                        "bucket": label_name,
                        "samples": 0,
                        "winrate": 0.0,
                        "avg_r": 0.0,
                        "+1R_hit_rate": 0.0,
                    }
                )
                continue

            r_values = bucket["r_multiple"].astype(float)
            output.append(
                {
                    "bucket": label_name,
                    "samples": int(len(bucket)),
                    "winrate": round(float((r_values > 0).mean()), 6),
                    "avg_r": round(float(r_values.mean()), 6),
                    "+1R_hit_rate": round(float((r_values >= 1.0).mean()), 6),
                }
            )
        return output

    @classmethod
    def calibration_curve(cls, dataset: pd.DataFrame) -> List[Dict]:
        if dataset.empty:
            return []

        labeled = dataset[dataset["actual_outcome"].notna()].copy()
        if labeled.empty:
            return []

        bucket_rows = cls.bucket_analysis(labeled)
        calibration: List[Dict] = []
        for bucket in bucket_rows:
            bucket_name = bucket["bucket"]
            if bucket_name.endswith("+"):
                lower_bound = float(bucket_name.replace("+", ""))
                upper_bound = 1.0
            else:
                lower_text, upper_text = bucket_name.split("-")
                lower_bound = float(lower_text)
                upper_bound = float(upper_text)

            bucket_df = labeled[
                (labeled["probability_of_success"] >= lower_bound)
                & (
                    (labeled["probability_of_success"] < upper_bound)
                    if upper_bound < 1.0
                    else (labeled["probability_of_success"] >= lower_bound)
                )
            ]
            if bucket_df.empty:
                calibration.append(
                    {
                        "bucket": bucket_name,
                        "samples": 0,
                        "mean_predicted_probability": 0.0,
                        "observed_positive_rate": 0.0,
                    }
                )
                continue

            calibration.append(
                {
                    "bucket": bucket_name,
                    "samples": int(len(bucket_df)),
                    "mean_predicted_probability": round(
                        float(bucket_df["probability_of_success"].astype(float).mean()), 6
                    ),
                    "observed_positive_rate": round(
                        float(bucket_df["actual_outcome"].astype(float).mean()), 6
                    ),
                }
            )
        return calibration

    @classmethod
    def export_training_artifacts(cls, closed_trades: List[Dict], output_dir: str) -> Dict[str, object]:
        os.makedirs(output_dir, exist_ok=True)
        dataset = cls.build_analysis_dataset(closed_trades)
        dataset_path = os.path.join(output_dir, "ml_dataset.csv")
        dataset.to_csv(dataset_path, index=False)

        labeled = dataset[dataset["actual_outcome"].notna()].copy() if not dataset.empty else dataset
        report = {
            "total_samples": int(len(labeled)),
            "all_trades_seen": int(len(dataset)),
            "calibration_curve": cls.calibration_curve(dataset),
            "bucket_performance": cls.bucket_analysis(dataset),
        }
        report_path = os.path.join(output_dir, "ml_report.json")
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)

        return {
            "dataset_path": dataset_path,
            "report_path": report_path,
            "report": report,
        }

    def flush(self) -> None:
        self._save()

    def _load(self) -> None:
        if not os.path.exists(self.memory_path):
            return
        try:
            with open(self.memory_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    payload = json.loads(line)
                    self._records.append(ProbabilityTradeRecord(**payload))
            LOG.info("Loaded %s probability trade records", len(self._records))
        except Exception as exc:
            LOG.warning("Failed to load probability memory: %s", exc)

    def _save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.memory_path) or ".", exist_ok=True)
            with open(self.memory_path, "w", encoding="utf-8") as handle:
                for record in self._records:
                    handle.write(json.dumps(asdict(record)) + "\n")
        except Exception as exc:
            LOG.warning("Failed to save probability memory: %s", exc)

    def _numeric_vector(self, payload: Dict) -> np.ndarray:
        values = [
            _safe_float(payload.get(feature), 0.0) / _NUMERIC_SCALES[feature]
            for feature in _NUMERIC_FEATURES
        ]
        return np.array(values, dtype=float)

    def _fallback_probability(self, features: Dict) -> float:
        signal_quality = _safe_float(features.get("signal_quality"), 0.0)
        edge_score = _safe_float(features.get("edge_score"), 0.0)
        ml_confidence = _safe_float(features.get("ml_confidence"), 0.0)
        boll_z = abs(_safe_float(features.get("boll_z"), 0.0))
        atr_pctile = _safe_float(features.get("atr_pctile"), 0.5)
        adx = _safe_float(features.get("adx"), 20.0)

        boll_component = min(boll_z / 1.5, 1.0)
        atr_component = 1.0 - min(abs(atr_pctile - 0.5) / 0.5, 1.0) * 0.15
        adx_component = min(adx / 35.0, 1.0)

        fallback = (
            signal_quality * 0.30
            + edge_score * 0.30
            + ml_confidence * 0.25
            + boll_component * 0.10
            + atr_component * 0.03
            + adx_component * 0.02
        )
        return float(min(max(fallback, 0.0), 1.0))
