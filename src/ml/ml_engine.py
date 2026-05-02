"""
src/ml/ml_engine.py
====================
SCOPUS ML Decision Engine — Trade Memory + Confidence Scoring.

This is NOT a price prediction model.
This is a similarity-based confidence engine that:
    1. Stores every trade with its features and outcome
    2. Compares current features to past trades
    3. Outputs confidence per strategy

The ML layer lives at Stage 3 of the pipeline.
It CANNOT create signals — it only provides confidence scores
that the StrategyRouter can use to weight its decision.

Deterministic: uses nearest-neighbor similarity, no neural nets.
Explainable: every confidence score traces to specific past trades.
"""
from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import numpy as np

LOG = logging.getLogger("ml_engine")

__all__ = ["MLEngine", "TradeMemory"]


# ---------------------------------------------------------------------------
# Trade Memory Record
# ---------------------------------------------------------------------------

@dataclass
class TradeMemory:
    """One recorded trade for the memory system."""
    symbol:         str
    strategy:       str
    side:           str
    # Features at entry
    adx:            float
    atr_pctile:     float
    boll_z:         float
    regime_score:   float
    ret_std:        float = 0.0
    hour:           int = 12
    bar_index:      int = 0
    # Outcome
    pnl_usd:        float = 0.0
    r_multiple:     float = 0.0
    won:            bool = False
    # ML confidence at time of trade (for calibration)
    ml_confidence:  float = 0.0


# ---------------------------------------------------------------------------
# Feature vector for similarity matching
# ---------------------------------------------------------------------------

_FEATURE_KEYS = ["adx", "atr_pctile", "boll_z", "regime_score", "ret_std"]
_FEATURE_SCALES = {
    "adx":          50.0,    # normalize to ~[0, 1]
    "atr_pctile":   1.0,     # already [0, 1]
    "boll_z":       3.0,     # typical range [-3, 3]
    "regime_score": 1.0,     # already [0, 1]
    "ret_std":      0.05,    # typical range [0, 0.05]
}


def _to_vector(features: dict) -> np.ndarray:
    """Convert feature dict to normalized vector for distance calc."""
    return np.array([
        features.get(k, 0.0) / _FEATURE_SCALES[k]
        for k in _FEATURE_KEYS
    ])


# ---------------------------------------------------------------------------
# ML Engine
# ---------------------------------------------------------------------------

class MLEngine:
    """
    Similarity-based confidence scoring engine.

    Core algorithm:
        1. For a new setup, find the K nearest past trades (by feature distance)
        2. Compute win rate and average R among those neighbors
        3. Output confidence = weighted win rate × avg R factor

    This is essentially K-NN classification on trade outcomes,
    segmented by strategy. No training loop, no gradients, no black box.
    """

    def __init__(
        self,
        k_neighbors:    int = 10,
        min_memory:     int = 15,     # min trades before producing scores
        memory_path:    str = "logs/shadow/trade_memory.jsonl",
    ):
        self.k = k_neighbors
        self.min_memory = min_memory
        self.memory_path = memory_path
        self._memory: Dict[str, List[TradeMemory]] = defaultdict(list)
        self._vectors: Dict[str, np.ndarray] = {}  # strategy → (N, d) array
        self._dirty = False
        self._load()


    # ------------------------------------------------------------------ #
    # Memory management                                                    #
    # ------------------------------------------------------------------ #

    def record_trade(self, trade: TradeMemory) -> None:
        """Record a completed trade to memory."""
        self._memory[trade.strategy].append(trade)
        self._dirty = True
        # Rebuild vector cache for this strategy
        self._rebuild_vectors(trade.strategy)
        # Persist periodically (every 5 trades)
        total = sum(len(v) for v in self._memory.values())
        if total % 5 == 0:
            self._save()

    def _rebuild_vectors(self, strategy: str) -> None:
        """Rebuild the feature vector matrix for a strategy."""
        trades = self._memory.get(strategy, [])
        if not trades:
            self._vectors.pop(strategy, None)
            return
        self._vectors[strategy] = np.array([
            _to_vector({
                "adx": t.adx,
                "atr_pctile": t.atr_pctile,
                "boll_z": t.boll_z,
                "regime_score": t.regime_score,
                "ret_std": t.ret_std,
            })
            for t in trades
        ])

    def _load(self) -> None:
        """Load memory from disk."""
        if not os.path.exists(self.memory_path):
            return
        try:
            with open(self.memory_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    t = TradeMemory(**{k: d[k] for k in TradeMemory.__dataclass_fields__ if k in d})
                    self._memory[t.strategy].append(t)
            for strat in self._memory:
                self._rebuild_vectors(strat)
            total = sum(len(v) for v in self._memory.values())
            LOG.info(f"[MLEngine] Loaded {total} trades from memory")
        except Exception as e:
            LOG.warning(f"[MLEngine] Failed to load memory: {e}")

    def _save(self) -> None:
        """Persist memory to disk."""
        try:
            os.makedirs(os.path.dirname(self.memory_path) or ".", exist_ok=True)
            with open(self.memory_path, "w") as f:
                for trades in self._memory.values():
                    for t in trades:
                        f.write(json.dumps(asdict(t)) + "\n")
            self._dirty = False
        except Exception as e:
            LOG.warning(f"[MLEngine] Failed to save memory: {e}")

    # ------------------------------------------------------------------ #
    # Confidence scoring                                                   #
    # ------------------------------------------------------------------ #

    def get_confidence(self, features: dict, current_bar: int = 0) -> Dict[str, float]:
        """
        Compute confidence score for each strategy given current features.

        Returns:
            {"MEAN_REVERSION": 0.72, "TREND_PULLBACK": 0.81, ...}

        Confidence = 0.0 if fewer than min_memory trades for that strategy.
        """
        query = _to_vector(features)
        result = {}

        for strategy, trades in self._memory.items():
            if len(trades) < self.min_memory:
                result[strategy] = 0.0
                continue

            vectors = self._vectors.get(strategy)
            if vectors is None or len(vectors) == 0:
                result[strategy] = 0.0
                continue

            # Compute distances to all past trades of this strategy
            distances = np.linalg.norm(vectors - query, axis=1)

            # Get K nearest neighbors
            k = min(self.k, len(trades))
            nearest_idx = np.argsort(distances)[:k]

            # Compute win rate and avg R among neighbors
            neighbors = [trades[i] for i in nearest_idx]
            wins = sum(1 for t in neighbors if t.won)
            win_rate = wins / k

            avg_r = np.mean([t.r_multiple for t in neighbors])

            # Distance AND Time-weighted confidence
            # Closer neighbors matter more, older trades matter less
            dist_weights = 1.0 / (distances[nearest_idx] + 0.01)
            
            ages = np.array([max(0, current_bar - t.bar_index) for t in neighbors])
            decay_weights = np.exp(-ages / 2000.0)  # ~2000 hours decay (~3 months)

            weights = dist_weights * decay_weights
            if weights.sum() == 0:
                weights = np.ones_like(weights) / len(weights)
            else:
                weights = weights / weights.sum()

            weighted_wr = sum(w * (1.0 if t.won else 0.0)
                              for w, t in zip(weights, neighbors))

            # Final confidence: weighted win rate × R factor
            r_factor = max(0.0, min(1.5, 0.5 + avg_r * 0.5))  # clamp [0, 1.5]
            confidence = weighted_wr * r_factor

            result[strategy] = round(min(confidence, 1.0), 3)

        return result

    @property
    def total_trades(self) -> int:
        return sum(len(v) for v in self._memory.values())

    def summary(self) -> Dict[str, dict]:
        """Return summary stats per strategy."""
        out = {}
        for strat, trades in self._memory.items():
            if not trades:
                continue
            wins = sum(1 for t in trades if t.won)
            avg_r = np.mean([t.r_multiple for t in trades]) if trades else 0
            out[strat] = {
                "n_trades": len(trades),
                "win_rate": round(wins / len(trades), 3),
                "avg_r": round(float(avg_r), 3),
            }
        return out
