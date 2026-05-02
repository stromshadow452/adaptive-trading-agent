"""
src/pipeline/strategy_router.py
================================
SCOPUS Strategy Router — Selects ONE strategy per (symbol, bar).

Uses the classified market regime as the primary driver so strategy
deployment stays aligned with system design:

    RANGE / MEAN_REVERSION regime  -> MEAN_REVERSION
    TREND regime                   -> TREND_PULLBACK
    UNCLEAR / NEUTRAL              -> SKIP

The router remains deterministic and explainable. Indicator values are
accepted only to explain the route, not to override regime alignment.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

LOG = logging.getLogger("strategy_router")

__all__ = ["StrategyRouter", "RouteDecision"]


@dataclass
class RouteDecision:
    """Output of the router."""
    strategy:   str       # "MEAN_REVERSION" | "TREND_PULLBACK" | "BREAKOUT" | "SCALPING" | "SKIP"
    reason:     str       # human-readable explanation
    confidence: float     # router confidence in this selection [0, 1]


class StrategyRouter:
    """
    Regime-aligned strategy router.

    The router preserves the classified regime instead of forcing a fallback:
        1. RANGE / MEAN_REVERSION regime -> MEAN_REVERSION
        2. TREND regime -> TREND_PULLBACK
        3. NEUTRAL / UNCLEAR / UNKNOWN -> SKIP
    """

    def __init__(
        self,
        # Trend Pullback thresholds
        adx_trend_min:      float = 25.0,
        score_trend_max:    float = 0.40,
        # Mean Reversion thresholds
        adx_mr_max:         float = 22.0,
        score_mr_min:       float = 0.45,
        # Breakout thresholds
        atr_pctile_compress: float = 0.25,
        # Scalping thresholds
        adx_scalp_max:      float = 20.0,
        atr_pctile_scalp_lo: float = 0.20,
        atr_pctile_scalp_hi: float = 0.50,
    ):
        self.adx_trend_min = adx_trend_min
        self.score_trend_max = score_trend_max
        self.adx_mr_max = adx_mr_max
        self.score_mr_min = score_mr_min
        self.atr_pctile_compress = atr_pctile_compress
        self.adx_scalp_max = adx_scalp_max
        self.atr_pctile_scalp_lo = atr_pctile_scalp_lo
        self.atr_pctile_scalp_hi = atr_pctile_scalp_hi

    def route(
        self,
        regime_score:  float,
        adx_val:       float,
        atr_pctile:    float,
        hour:          int = 12,
        ml_confidence: Optional[dict] = None,
        regime_label:  Optional[str] = None,
    ) -> RouteDecision:
        """
        Select the best strategy for current conditions.

        Args:
            regime_score: 0-1, higher = more MR-favorable
            adx_val:      raw ADX (14)
            atr_pctile:   ATR percentile rank (0-1)
            hour:         UTC hour (for session awareness)
            ml_confidence: unused for routing decisions; kept for compatibility
            regime_label: classified regime label from the regime classifier

        Returns:
            RouteDecision with selected strategy name
        """
        normalized_regime = str(regime_label or "").strip().upper()

        if normalized_regime in {"MEAN_REVERSION", "RANGE"}:
            return RouteDecision(
                strategy="MEAN_REVERSION",
                reason=f"range_regime: regime={normalized_regime or 'RANGE'} adx={adx_val:.0f} atr={atr_pctile:.2f}",
                confidence=round(max(0.0, min(1.0, regime_score)), 3),
            )

        if normalized_regime == "TREND":
            trend_conf = 1.0 - max(0.0, min(1.0, regime_score))
            return RouteDecision(
                strategy="TREND_PULLBACK",
                reason=f"trend_regime: regime=TREND adx={adx_val:.0f} atr={atr_pctile:.2f}",
                confidence=round(max(0.0, min(1.0, trend_conf)), 3),
            )

        if normalized_regime in {"NEUTRAL", "UNCLEAR"}:
            return RouteDecision(
                strategy="SKIP",
                reason=f"unclear_regime: regime={normalized_regime} adx={adx_val:.0f} atr={atr_pctile:.2f}",
                confidence=round(min(0.55, max(0.0, regime_score)), 3),
            )

        return RouteDecision(
            strategy="SKIP",
            reason=f"unknown_regime: regime={normalized_regime or 'UNKNOWN'} adx={adx_val:.0f} atr={atr_pctile:.2f}",
            confidence=0.0,
        )
