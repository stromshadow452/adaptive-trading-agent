"""
src/risk/adaptive_sizing.py
============================
SCOPUS Adaptive Position Sizing — Week 8.

Replaces fixed `default_size` with confidence-tiered, volatility-scaled
position sizing. Integrates with existing PortfolioBrain and PaperExecutor.

Usage:
    from src.risk.adaptive_sizing import AdaptiveSizer

    sizer = AdaptiveSizer(account_equity=10_000.0, base_risk_pct=0.01)
    size  = sizer.compute(
        confidence  = 0.72,
        atr_pctile  = 0.45,       # from vol_atr_pctile_100 feature
        entry_price = 1.1000,
        stop_price  = 1.0950,
        symbol      = "EURUSD",
    )
    # → lots: 0.07

Design:
  Step 1: confidence_multiplier()  — scales 0.5×–1.25× based on model confidence
  Step 2: vol_multiplier()         — scales 0.60×–1.10× based on ATR percentile
  Step 3: kelly_cap()              — hard Kelly fraction cap (never > 0.25 Kelly)
  Step 4: portfolio_cap()          — hard 2% per symbol, 6% total exposure cap
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Confidence tiers (from adaptive upgrade plan §4)
# ---------------------------------------------------------------------------
_CONFIDENCE_TIERS = [
    (0.80, 1.25),   # High conviction  → upsize
    (0.70, 1.15),   # Good conviction  → slight upsize
    (0.60, 1.00),   # Normal           → base size
    (0.50, 0.75),   # Weak             → reduce
    (0.40, 0.50),   # Very weak        → half size
]
_CONFIDENCE_FLOOR = 0.40    # below this → skip trade (return 0.0)

# ---------------------------------------------------------------------------
# Volatility regime multipliers
# ---------------------------------------------------------------------------
_VOL_TIERS = [
    (0.80, 0.60),   # High-vol regime  → reduce
    (0.60, 0.80),   # Elevated vol     → slightly reduce
    (0.40, 1.00),   # Normal range     → base
    (0.20, 1.05),   # Low-vol          → slight increase
    (0.00, 1.10),   # Very low-vol     → increase (but cautiously)
]

# ---------------------------------------------------------------------------
# Hard limits
# ---------------------------------------------------------------------------
MAX_SIZE_LOTS        = 1.0     # never trade > 1 standard lot (prototype)
MIN_SIZE_LOTS        = 0.01    # minimum tradeable size
MAX_RISK_PCT_SYMBOL  = 0.02    # max 2% equity risk per symbol
MAX_RISK_PCT_TOTAL   = 0.06    # max 6% total portfolio exposure
KELLY_CAP            = 0.25    # max Kelly fraction

# Pip values
_PIP = {"JPY": 0.01, "XAG": 0.01}
_DEFAULT_PIP = 0.0001
_DOLLAR_PER_PIP_PER_LOT = 10.0


def _pip_value(symbol: str) -> float:
    for suffix, val in _PIP.items():
        if suffix in symbol.upper():
            return val
    return _DEFAULT_PIP


# ---------------------------------------------------------------------------
# Standalone multiplier functions (testable in isolation)
# ---------------------------------------------------------------------------

def confidence_multiplier(confidence: float) -> float:
    """
    Map model confidence → position size multiplier.

    Args:
        confidence: Model confidence score [0, 1].

    Returns:
        Size multiplier [0, 1.25]. Returns 0.0 if below confidence floor.
    """
    if confidence < _CONFIDENCE_FLOOR:
        return 0.0
    for threshold, mult in _CONFIDENCE_TIERS:
        if confidence >= threshold:
            return mult
    return 0.50   # fallback (shouldn't reach here)


def vol_regime_multiplier(atr_pctile: float) -> float:
    """
    Map ATR percentile rank → size multiplier.

    Args:
        atr_pctile: ATR(14) percentile rank [0, 1] from vol_atr_pctile_100.

    Returns:
        Size multiplier [0.60, 1.10].
    """
    atr_pctile = float(np.clip(atr_pctile, 0.0, 1.0))
    for threshold, mult in _VOL_TIERS:
        if atr_pctile >= threshold:
            return mult
    return 1.0


def kelly_fraction(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """
    Full Kelly fraction = (win_rate / avg_loss) - (loss_rate / avg_win).
    Capped at KELLY_CAP (0.25) as a safety limit.

    Args:
        win_rate: Historical win rate [0, 1].
        avg_win : Average winning trade size (positive).
        avg_loss: Average losing trade size (positive absolute value).

    Returns:
        Capped Kelly fraction [0, KELLY_CAP].
    """
    if avg_loss <= 0 or avg_win <= 0:
        return 0.0
    loss_rate = 1.0 - win_rate
    f = (win_rate / avg_loss) - (loss_rate / avg_win)
    return float(np.clip(f, 0.0, KELLY_CAP))


# ---------------------------------------------------------------------------
# AdaptiveSizer
# ---------------------------------------------------------------------------

class AdaptiveSizer:
    """
    Computes adaptive lot size incorporating:
      1. Confidence-tiered multiplier
      2. Volatility regime scaling
      3. ATR-based risk per trade (correct formula → replaces size×1%)
      4. Kelly cap (never exceed 0.25 Kelly)
      5. Per-symbol and total portfolio hard caps
    """

    def __init__(
        self,
        account_equity:  float = 10_000.0,
        base_risk_pct:   float = 0.01,       # 1% account risk at base size
        max_risk_symbol: float = MAX_RISK_PCT_SYMBOL,
        kelly_history:   Optional[Dict] = None,   # {"win_rate": 0.5, "avg_win": 50, "avg_loss": 30}
    ):
        self.equity          = account_equity
        self.base_risk_pct   = base_risk_pct
        self.max_risk_symbol = max_risk_symbol
        self._kelly_hist     = kelly_history or {}
        self._open_risk: Dict[str, float] = {}   # symbol → current risk USD

    def compute(
        self,
        confidence:  float,
        atr_pctile:  float,
        entry_price: float,
        stop_price:  float,
        symbol:      str = "EURUSD",
        override_equity: Optional[float] = None,
    ) -> float:
        """
        Compute final position size in lots.

        Returns:
            0.0 if below confidence floor or risk caps would be breached.
            Otherwise returns lot size rounded to 2 decimal places.
        """
        equity = override_equity or self.equity

        # ── 1. Confidence gate ─────────────────────────────────────────
        conf_mult = confidence_multiplier(confidence)
        if conf_mult == 0.0:
            logger.debug(f"[AdaptiveSizer] {symbol}: confidence {confidence:.2f} below floor → skip")
            return 0.0

        # ── 2. Volatility scaling ──────────────────────────────────────
        vol_mult = vol_regime_multiplier(atr_pctile)

        # ── 3. ATR-based base risk ─────────────────────────────────────
        pip_val   = _pip_value(symbol)
        stop_pips = abs(entry_price - stop_price) / max(pip_val, 1e-10)
        if stop_pips < 1.0:
            logger.debug(f"[AdaptiveSizer] {symbol}: stop too tight ({stop_pips:.1f} pips) → skip")
            return 0.0

        # Base size: equity × risk_pct / (stop_pips × $/pip/lot)
        base_risk_usd = equity * self.base_risk_pct
        base_size     = base_risk_usd / (stop_pips * _DOLLAR_PER_PIP_PER_LOT)

        # ── 4. Apply multipliers ───────────────────────────────────────
        raw_size = base_size * conf_mult * vol_mult

        # ── 5. Kelly cap ───────────────────────────────────────────────
        if self._kelly_hist:
            k = kelly_fraction(
                self._kelly_hist.get("win_rate", 0.5),
                self._kelly_hist.get("avg_win",  50.0),
                self._kelly_hist.get("avg_loss", 30.0),
            )
            kelly_max_size = equity * k / (stop_pips * _DOLLAR_PER_PIP_PER_LOT)
            raw_size = min(raw_size, kelly_max_size)

        # ── 6. Per-symbol risk cap ─────────────────────────────────────
        max_size_by_risk = (equity * self.max_risk_symbol) / (
            stop_pips * _DOLLAR_PER_PIP_PER_LOT
        )
        raw_size = min(raw_size, max_size_by_risk)

        # ── 7. Total portfolio exposure cap ───────────────────────────
        current_total = sum(self._open_risk.values())
        max_new_risk  = max(0.0, equity * MAX_RISK_PCT_TOTAL - current_total)
        capped_new_risk  = min(raw_size * stop_pips * _DOLLAR_PER_PIP_PER_LOT, max_new_risk)
        raw_size = capped_new_risk / max(stop_pips * _DOLLAR_PER_PIP_PER_LOT, 1e-9)

        # ── 8. Absolute bounds ─────────────────────────────────────────
        final_size = float(np.clip(raw_size, MIN_SIZE_LOTS, MAX_SIZE_LOTS))
        final_size = round(final_size, 2)

        logger.debug(
            f"[AdaptiveSizer] {symbol}: conf={confidence:.2f} ({conf_mult:.2f}×) "
            f"vol_pctile={atr_pctile:.2f} ({vol_mult:.2f}×) "
            f"base={base_size:.3f} → final={final_size:.2f} lots"
        )
        return final_size

    def register_open(self, symbol: str, size: float,
                      stop_pips: float):
        """Track a newly opened position's risk for portfolio cap calculation."""
        risk_usd = size * stop_pips * _DOLLAR_PER_PIP_PER_LOT
        self._open_risk[symbol] = risk_usd

    def release_position(self, symbol: str):
        """Remove a closed position from portfolio risk tracking."""
        self._open_risk.pop(symbol, None)

    def update_equity(self, new_equity: float):
        """Update equity after PnL changes."""
        self.equity = new_equity

    def update_kelly_history(self, win_rate: float, avg_win: float, avg_loss: float):
        """Update Kelly stats — call after each closed trade."""
        self._kelly_hist = {"win_rate": win_rate, "avg_win": avg_win, "avg_loss": avg_loss}

    @property
    def portfolio_risk_pct(self) -> float:
        """Current total portfolio risk as % of equity (0.0–1.0)."""
        return sum(self._open_risk.values()) / max(self.equity, 1.0)

    def sizing_breakdown(self, confidence: float, atr_pctile: float) -> dict:
        """Return a breakdown of multipliers for transparency/logging."""
        return {
            "confidence":        round(confidence, 4),
            "conf_multiplier":   confidence_multiplier(confidence),
            "atr_pctile":        round(atr_pctile, 4),
            "vol_multiplier":    vol_regime_multiplier(atr_pctile),
            "combined_mult":     round(
                confidence_multiplier(confidence) * vol_regime_multiplier(atr_pctile), 4),
            "base_risk_pct":     self.base_risk_pct,
            "max_risk_symbol":   self.max_risk_symbol,
        }
