"""
src/decision/ensemble.py
=========================
SCOPUS Strategy Ensemble — Week 8.

WeightedSignalAggregator: combines signals from multiple strategies
using regime-conditional weights. Wraps around the existing meta_gating.py
without replacing it.

Architecture:
    MetaGatingBrain → regime
    strategy_A.signal() ─┐
    strategy_B.signal() ─┼─→ WeightedSignalAggregator → consolidated signal
    strategy_C.signal() ─┘     (regime-weighted, correlation-guarded)

Promotion status: SHADOW_ONLY — runs in parallel with single-strategy path.
Log to logs/shadow/ensemble.jsonl for comparison.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regime-conditional strategy weights
# Rows = regimes, Cols = [tokyo_session_mr, silver_mr, placeholder_c]
# Values represent normalized weights per regime.
# ---------------------------------------------------------------------------
_REGIME_WEIGHTS: Dict[str, List[float]] = {
    #                        tokyo_mr  silver_mr  future_str
    "RANGE":              [0.00,      0.00,       0.00],   # PF=0.42 — suspend until tuned
    "TREND":              [0.15,      0.55,       0.30],
    "TREND_BULL":         [0.15,      0.55,       0.30],
    "TREND_BEAR":         [0.15,      0.55,       0.30],
    "HIGH_VOL":           [0.05,      0.15,       0.80],   # future strategy
    "CRASH":              [0.00,      0.00,       0.00],   # no trade
    "UNCERTAIN":          [0.30,      0.30,       0.00],   # reduced weight but tradeable
}
_STRATEGY_NAMES = ["tokyo_session_mr", "silver_mr", "future_strategy"]

# Signal confidence floor — below this, strategy signal ignored
_CONFIDENCE_FLOOR  = 0.40
# Ensemble score floor — below this, no trade
_ENSEMBLE_FLOOR    = 0.25
# Directional disagreement → no trade
_ENABLE_CORR_GUARD = True


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class StrategySignal:
    """A trading signal from one strategy."""
    strategy:   str
    symbol:     str
    side:       str          # "buy" | "sell"
    confidence: float        # [0, 1]
    price:      float
    sl:         Optional[float] = None
    tp:         Optional[float] = None
    regime:     str = "UNKNOWN"
    size:       float = 0.01
    meta:       dict = field(default_factory=dict)

    def directional_score(self) -> float:
        """Signed score: +confidence for buy, -confidence for sell."""
        return self.confidence if self.side == "buy" else -self.confidence


@dataclass
class EnsembleDecision:
    """Aggregated ensemble output."""
    symbol:             str
    side:               Optional[str]    # None = no trade
    ensemble_score:     float
    weighted_confidence: float
    regime:             str
    constituent_signals: List[dict]
    correlation_guard:  bool   # True = guarded (disagreement detected)
    decided_at:         str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def should_trade(self) -> bool:
        return (
            self.side is not None
            and abs(self.ensemble_score) >= _ENSEMBLE_FLOOR
            and not self.correlation_guard
        )

    def to_signal_dict(self) -> Optional[dict]:
        """Convert to the signal dict format expected by PaperExecutor."""
        if not self.should_trade:
            return None
        # Use price/sl/tp from the highest-confidence constituent
        best = max(
            [s for s in self.constituent_signals if s.get("side") == self.side],
            key=lambda s: s.get("confidence", 0.0),
            default=None
        )
        if best is None:
            return None
        return {
            "symbol":     self.symbol,
            "side":       self.side,
            "price":      best.get("price", 0.0),
            "sl":         best.get("sl"),
            "tp":         best.get("tp"),
            "strategy":   "ensemble",
            "regime":     self.regime,
            "confidence": self.weighted_confidence,
            "ensemble_score": self.ensemble_score,
        }


# ---------------------------------------------------------------------------
# WeightedSignalAggregator
# ---------------------------------------------------------------------------

class WeightedSignalAggregator:
    """
    Aggregates multiple strategy signals into a single ensemble decision.

    Two modes:
      SHADOW: logs decisions and records comparison data — does NOT replace
              the single-strategy path yet.
      PRODUCTION: replaces single strategy for execution. (Not yet active.)

    Correlation guard:
      If the top-2 weighted strategies disagree in direction → no trade.
      Prevents whipsaw from conflicting signals.
    """

    def __init__(
        self,
        log_path:        str  = "logs/shadow/ensemble.jsonl",
        regime_weights:  Dict[str, List[float]] = None,
        shadow_mode:     bool = True,
    ):
        self.log_path      = log_path
        self.regime_weights = regime_weights or _REGIME_WEIGHTS
        self.shadow_mode   = shadow_mode
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)

    def aggregate(
        self,
        signals: List[StrategySignal],
        regime:  str,
        symbol:  str,
    ) -> EnsembleDecision:
        """
        Aggregate strategy signals using regime-conditional weights.

        Args:
            signals : List of StrategySignal from each strategy.
            regime  : Current regime from MetaGatingBrain.classify_regime().
            symbol  : Trading symbol.

        Returns:
            EnsembleDecision with should_trade, side, ensemble_score.
        """
        weights = self.regime_weights.get(regime.upper(),
                  self.regime_weights.get("UNCERTAIN", [0.0, 0.0, 0.0]))

        # Regime zero-weight → no trade immediately
        if sum(weights) == 0.0:
            return self._no_trade(symbol, regime, signals,
                                  reason="regime_blocked")

        # Pad missing signals with neutral
        padded_signals: List[Optional[StrategySignal]] = []
        for name in _STRATEGY_NAMES:
            match = next((s for s in signals if s.strategy == name), None)
            padded_signals.append(match)

        # Filter by confidence floor and compute weighted directional score
        weighted_score = 0.0
        weighted_conf  = 0.0
        weight_sum     = 0.0

        active_signals = []   # (signed_score, weight, strategy)

        for i, sig in enumerate(padded_signals):
            w = weights[i] if i < len(weights) else 0.0
            if sig is None or sig.confidence < _CONFIDENCE_FLOOR:
                continue
            d_score    = sig.directional_score()
            weighted_score += d_score * w
            weighted_conf  += sig.confidence * w
            weight_sum     += w
            active_signals.append((d_score, w, sig.strategy))

        if weight_sum == 0.0:
            return self._no_trade(symbol, regime, signals,
                                  reason="no_confident_signals")

        weighted_score /= weight_sum
        weighted_conf  /= weight_sum

        # Correlation guard: check top-2 by weight disagree in direction
        corr_guard = False
        if _ENABLE_CORR_GUARD and len(active_signals) >= 2:
            # Sort by weight descending
            top2 = sorted(active_signals, key=lambda x: x[1], reverse=True)[:2]
            if top2[0][0] * top2[1][0] < 0:   # opposite signs
                corr_guard = True
                logger.debug(
                    f"[Ensemble] {symbol} correlation guard: "
                    f"{top2[0][2]}={top2[0][0]:.2f} vs {top2[1][2]}={top2[1][0]:.2f}"
                )

        # Determine direction
        if abs(weighted_score) < _ENSEMBLE_FLOOR:
            side = None
        else:
            side = "buy" if weighted_score > 0 else "sell"

        decision = EnsembleDecision(
            symbol=symbol,
            side=side,
            ensemble_score=round(weighted_score, 4),
            weighted_confidence=round(weighted_conf, 4),
            regime=regime,
            constituent_signals=[
                {
                    "strategy":   s.strategy, "side": s.side,
                    "confidence": s.confidence,
                    "price":      s.price, "sl": s.sl, "tp": s.tp,
                }
                for s in signals
            ],
            correlation_guard=corr_guard,
        )

        self._log(decision)
        return decision

    @staticmethod
    def _no_trade(symbol, regime, signals, reason="") -> EnsembleDecision:
        return EnsembleDecision(
            symbol=symbol, side=None,
            ensemble_score=0.0, weighted_confidence=0.0,
            regime=regime,
            constituent_signals=[],
            correlation_guard=False,
        )

    def _log(self, decision: EnsembleDecision):
        """Append ensemble decision to JSONL log."""
        try:
            record = asdict(decision)
            record["event"] = "ENSEMBLE"
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception as e:
            logger.debug(f"[Ensemble] log error: {e}")

    def regime_weight_table(self) -> dict:
        """Return the regime weight table for introspection/API."""
        return {
            regime: dict(zip(_STRATEGY_NAMES, weights))
            for regime, weights in self.regime_weights.items()
        }


# ---------------------------------------------------------------------------
# Ensemble Performance Tracker
# ---------------------------------------------------------------------------

class EnsembleTracker:
    """
    Reads ensemble JSONL log and computes comparison metrics:
    ensemble vs single-strategy.
    """

    def __init__(self, fills_log:    str = "logs/shadow/fills.jsonl",
                       ensemble_log: str = "logs/shadow/ensemble.jsonl"):
        self.fills_log    = fills_log
        self.ensemble_log = ensemble_log

    def compare(self) -> dict:
        """Return ensemble vs single-strategy trade count comparison."""
        ensemble_trades  = self._count_events(self.ensemble_log, "ENSEMBLE",
                                              lambda r: r.get("side") is not None)
        ensemble_skipped = self._count_events(self.ensemble_log, "ENSEMBLE",
                                              lambda r: r.get("side") is None)
        corr_guard_trips = self._count_events(
            self.ensemble_log, "ENSEMBLE",
            lambda r: r.get("correlation_guard") is True)
        return {
            "ensemble_trade_signals":   ensemble_trades,
            "ensemble_skipped_signals": ensemble_skipped,
            "correlation_guard_trips":  corr_guard_trips,
            "ensemble_log":             self.ensemble_log,
        }

    def _count_events(self, path, event_type, predicate) -> int:
        count = 0
        if not os.path.exists(path):
            return 0
        try:
            with open(path) as f:
                for line in f:
                    try:
                        r = json.loads(line)
                        if r.get("event") == event_type and predicate(r):
                            count += 1
                    except Exception:
                        continue
        except Exception:
            pass
        return count
