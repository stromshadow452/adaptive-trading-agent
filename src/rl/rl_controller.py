"""
src/rl/rl_controller.py
========================
SCOPUS RL Behavioral Controller — Dynamic Risk Tuning.

This is NOT a trade signal generator.
This module adjusts BEHAVIORAL parameters based on recent performance:
    - Position size multiplier
    - Strategy usage weight
    - Trade frequency control

Pipeline integration: Stage 4 (RL) → feeds into Stage 11 (Throttle/Sizing).

The RL layer CANNOT:
    - Generate trade signals
    - Override risk limits (max DD, max exposure)
    - Increase size beyond 1.25× base

The RL layer CAN:
    - Reduce size (down to 0.3× base)
    - Slow down frequency (increase bar gap between trades)
    - Weight strategies differently based on recent win/loss

Algorithm: Rule-based state machine (not deep RL).
Deterministic and explainable.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

LOG = logging.getLogger("rl_controller")

__all__ = ["RLController", "RLAction"]


# ---------------------------------------------------------------------------
# RL Action output
# ---------------------------------------------------------------------------

@dataclass
class RLAction:
    """Behavioral adjustments from the RL controller."""
    size_multiplier:     float   # [0.3, 1.25] — scales position size
    frequency_gap:       int     # minimum bars between trades (0 = no gap)
    strategy_weights:    Dict[str, float]  # {strategy: weight} for router
    state:              str     # "AGGRESSIVE" | "NORMAL" | "CAUTIOUS" | "DEFENSIVE"
    reason:             str     # human-readable explanation


# ---------------------------------------------------------------------------
# Performance tracker for RL state
# ---------------------------------------------------------------------------

@dataclass
class _PerfWindow:
    """Sliding window of recent trade results."""
    window: int = 20
    _pnls: List[float] = field(default_factory=list)
    _strategies: List[str] = field(default_factory=list)

    def record(self, pnl: float, strategy: str):
        self._pnls.append(pnl)
        self._strategies.append(strategy)
        if len(self._pnls) > self.window:
            self._pnls = self._pnls[-self.window:]
            self._strategies = self._strategies[-self.window:]

    @property
    def n(self) -> int:
        return len(self._pnls)

    @property
    def win_rate(self) -> float:
        if not self._pnls:
            return 0.5
        return sum(1 for p in self._pnls if p > 0) / len(self._pnls)

    @property
    def consecutive_losses(self) -> int:
        count = 0
        for p in reversed(self._pnls):
            if p < 0:
                count += 1
            else:
                break
        return count

    @property
    def consecutive_wins(self) -> int:
        count = 0
        for p in reversed(self._pnls):
            if p > 0:
                count += 1
            else:
                break
        return count

    @property
    def net_r(self) -> float:
        return sum(self._pnls) if self._pnls else 0.0

    def strategy_win_rate(self, strategy: str) -> float:
        """Win rate for a specific strategy."""
        pairs = [(p, s) for p, s in zip(self._pnls, self._strategies) if s == strategy]
        if not pairs:
            return 0.5
        return sum(1 for p, _ in pairs if p > 0) / len(pairs)

    def strategy_count(self, strategy: str) -> int:
        return sum(1 for s in self._strategies if s == strategy)


# ---------------------------------------------------------------------------
# RL Controller
# ---------------------------------------------------------------------------

class RLController:
    """
    Rule-based behavioral controller.

    State machine with 4 states:
        AGGRESSIVE  — winning streak → slight size increase (1.10-1.25×)
        NORMAL      — baseline performance → standard sizing (1.0×)
        CAUTIOUS    — mixed results → reduced sizing (0.7×)
        DEFENSIVE   — losing streak / high DD → minimal sizing (0.3-0.5×)

    Transitions are deterministic based on recent performance window.
    """

    def __init__(self, window: int = 20):
        self._perf = _PerfWindow(window=window)
        self._state = "NORMAL"
        self._last_trade_bar = 0  # bar count of last trade
        self._dd_pct = 0.0

    def record_trade(self, pnl: float, strategy: str):
        """Record a completed trade result."""
        self._perf.record(pnl, strategy)
        self._update_state()

    def set_drawdown(self, dd_pct: float):
        """Update current drawdown level."""
        self._dd_pct = dd_pct

    def set_last_trade_bar(self, bar: int):
        """Track when the last trade was placed."""
        self._last_trade_bar = bar

    def _update_state(self):
        """Deterministic state transition."""
        consec_l = self._perf.consecutive_losses
        consec_w = self._perf.consecutive_wins
        wr = self._perf.win_rate
        dd = self._dd_pct

        # DEFENSIVE: high DD or long losing streak
        if dd >= 5.0 or consec_l >= 4:
            self._state = "DEFENSIVE"
            return

        # CAUTIOUS: moderate concern signals
        if consec_l >= 3 or (self._perf.n >= 10 and wr < 0.35):
            self._state = "CAUTIOUS"
            return

        # FORCE NORMAL on DD > 3% (prevents leverage/blowup risk during recovery)
        if dd >= 3.0:
            self._state = "NORMAL"
            return

        # AGGRESSIVE: winning streak + good track record
        if consec_w >= 3 and wr >= 0.50 and dd < 2.0 and self._perf.n >= 8:
            self._state = "AGGRESSIVE"
            return

        # Default
        self._state = "NORMAL"

    def get_action(self, current_bar: int = 0) -> RLAction:
        """
        Get current behavioral adjustments.

        Returns RLAction with size multiplier, frequency gap,
        and strategy weights.
        """

        # Size multiplier by state (bounded; RL never blocks by returning 0.0)
        size_mult = {
            "AGGRESSIVE": 1.15,
            "NORMAL":     1.00,
            "CAUTIOUS":   0.70,
            "DEFENSIVE":  0.50,  # floor at 0.5 for production safety
        }[self._state]

        # Frequency gap (Phase 1 unblock): reduce aggressiveness.
        freq_gap = {
            "AGGRESSIVE": 0,
            "NORMAL":     0,
            "CAUTIOUS":   3,    # was 6
            "DEFENSIVE":  6,    # was 12
        }[self._state]

        # Warmup: disable frequency throttling until we have evidence.
        if self._perf.n < 8:
            freq_gap = 0

        bars_since_last = current_bar - self._last_trade_bar
        hard_throttle = (
            (self._perf.consecutive_losses >= 3 or self._dd_pct >= 3.0)
            and (freq_gap > 0)
            and (bars_since_last < freq_gap)
        )

        # RL does NOT hard-block trades by returning size_multiplier=0.0.
        # Instead it applies a conservative size throttle.
        if hard_throttle:
            size_mult = min(size_mult, 0.50)

        # Strategy weights based on individual performance
        weights = self._compute_strategy_weights()

        return RLAction(
            size_multiplier=round(size_mult, 3),
            frequency_gap=freq_gap,
            strategy_weights=weights,
            state=self._state,
            reason=(
                f"{'hard_throttle' if hard_throttle else 'state'}={self._state} "
                f"wr={self._perf.win_rate:.0%} "
                f"consec_L={self._perf.consecutive_losses} dd={self._dd_pct:.1f}% "
                f"bars_since_last={bars_since_last} gap={freq_gap}"
            ),
        )

    def _compute_strategy_weights(self) -> Dict[str, float]:
        """
        Weight each strategy based on its recent performance.

        Strategies with higher win rates get higher weights.
        Strategies with fewer than 5 trades get neutral weight (1.0).
        """
        weights = {}
        strategies = {"MEAN_REVERSION", "TREND_PULLBACK", "BREAKOUT", "SCALPING"}

        for strat in strategies:
            n = self._perf.strategy_count(strat)
            if n < 5:
                weights[strat] = 1.0  # neutral — not enough data
                continue

            wr = self._perf.strategy_win_rate(strat)

            # Weight = win_rate normalized relative to 0.45 baseline
            # WR=0.55 → weight 1.2, WR=0.35 → weight 0.8
            weights[strat] = round(max(0.5, min(1.5, wr / 0.45)), 3)

        return weights

    @property
    def state(self) -> str:
        return self._state

    def summary(self) -> dict:
        """Return current RL state summary."""
        return {
            "state": self._state,
            "total_trades": self._perf.n,
            "win_rate": round(self._perf.win_rate, 3),
            "consec_losses": self._perf.consecutive_losses,
            "consec_wins": self._perf.consecutive_wins,
            "drawdown_pct": self._dd_pct,
        }
