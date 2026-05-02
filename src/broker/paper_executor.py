"""
src/broker/paper_executor.py
============================
Shadow Mode Paper Executor — Week 5 Implementation.

Simulates order fills using a pip-slippage model. Logs every event to a
JSONL file. Sends NO real broker orders. This is the shadow trading engine.

Usage:
    from src.broker.paper_executor import PaperExecutor
    executor = PaperExecutor(log_path="logs/shadow/fills.jsonl")
    fill = executor.execute(signal)
    ...
    executor.close_position("EURUSD", exit_px=1.0850, reason="tp_hit")

Signal dict contract (same as executor.py plans):
    {
      "symbol":   str,
      "side":     "buy" | "sell",
      "price":    float,      # requested entry price
      "size":     float,      # lots
      "sl":       float,      # stop-loss price
      "tp":       float,      # take-profit price
      "strategy": str,
      "regime":   str,
      "confidence": float,
    }
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np

import logging
logger = logging.getLogger(__name__)


_SPREAD_OVERRIDE: Optional[Dict[str, float]] = None


def _load_spread_model(path: str = "config/spread_model.json") -> Optional[Dict[str, float]]:
    """
    Optional override for the spread model, in pips.

    If present, expected format:
      {"EURUSD": 0.6, "GBPUSD": 1.0, "default": 1.0, ...}
    """
    try:
        global _SPREAD_OVERRIDE
        if _SPREAD_OVERRIDE is not None:
            return _SPREAD_OVERRIDE
        if not os.path.exists(path):
            _SPREAD_OVERRIDE = None
            return None
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            return None
        out: Dict[str, float] = {}
        for k, v in raw.items():
            try:
                out[str(k).upper()] = float(v)
            except Exception:
                continue
        _SPREAD_OVERRIDE = out or None
        return _SPREAD_OVERRIDE
    except Exception:
        return None


def _parse_bar_time(meta: Dict) -> Optional[datetime]:
    """Best-effort parse of candle timestamp passed via signal metadata."""
    if not meta:
        return None
    for key in ("bar_time", "timestamp", "candle_time", "time"):
        raw = meta.get(key)
        if raw is None:
            continue
        if isinstance(raw, datetime):
            return raw if raw.tzinfo else raw.replace(tzinfo=timezone.utc)
        try:
            s = str(raw).replace("Z", "+00:00")
            dt = datetime.fromisoformat(s)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# Slippage model: realistic spread-based pip slippage per symbol
# Source: average broker spread + 30% buffer for execution delay
# ---------------------------------------------------------------------------
SLIPPAGE_MODEL: Dict[str, tuple[float, float]] = {
    # Majors: realistic per-symbol spreads (min_pips, max_pips)
    "EURUSD": (0.6, 1.2),
    "GBPUSD": (1.0, 1.8),
    "AUDUSD": (0.8, 1.5),
    "NZDUSD": (1.2, 2.0),
    "USDCAD": (1.2, 2.2),
    "USDJPY": (0.8, 1.5),
    "USDCHF": (0.8, 1.5),
    # Crosses: wider spread
    "EURGBP": (0.8, 1.5),
    "EURJPY": (0.8, 1.5),
    "GBPJPY": (1.2, 2.0),
    "AUDJPY": (1.0, 1.8),
    "NZDJPY": (1.2, 2.0),
    "EURCAD": (1.2, 2.0),
    "EURCHF": (1.0, 1.8),
    "GBPCHF": (1.5, 2.5),
    "CHFJPY": (1.3, 2.0),
    # Metals
    "XAGUSD": (2.0, 3.0),
    "XAUUSD": (1.5, 2.5),
    # Default for unknown
    "default": (1.0, 2.0),
}

# Pip value in price units (0.01 for JPY pairs, 0.0001 for all others)
def _pip_value(symbol: str) -> float:
    return 0.01 if "JPY" in symbol.upper() or "XAG" in symbol.upper() else 0.0001


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SimulatedFill:
    """Represents a simulated trade fill in shadow mode."""
    symbol:         str
    side:           str           # 'buy' | 'sell'
    strategy:       str
    regime:         str
    confidence:     float
    # Prices
    requested_px:   float         # signal price
    fill_px:        float         # with simulated slippage
    sl:             Optional[float]
    tp:             Optional[float]
    # Size
    size:           float         # lots
    # Timing
    requested_at:   str           # ISO 8601 UTC
    filled_at:      str
    latency_ms:     float
    slippage_pips:  float
    # Status
    status:         str = "open"  # 'open' | 'closed'
    # Computed at close
    close_px:       Optional[float] = None
    close_reason:   Optional[str]  = None
    pnl_usd:        Optional[float] = None
    hold_minutes:   Optional[float] = None
    r_multiple:     Optional[float] = None
    # Execution Friction
    spread_cost:    float = 0.0
    slippage_cost:  float = 0.0
    total_cost:     float = 0.0
    ideal_pnl_usd:  Optional[float] = None
    # Rich context support
    metadata:       Optional[Dict] = field(default_factory=dict)

@dataclass
class ShadowState:
    """Runtime state of the paper executor."""
    equity:             float = 10_000.0
    peak_equity:        float = 10_000.0
    open_positions:     Dict[str, SimulatedFill] = field(default_factory=dict)
    closed_trades:      List[SimulatedFill]      = field(default_factory=list)
    total_trades:       int   = 0
    winning_trades:     int   = 0


# ---------------------------------------------------------------------------
# PaperExecutor
# ---------------------------------------------------------------------------

class PaperExecutor:
    """
    Shadow mode paper trading executor.

    Records simulated fills at market price + modeled slippage.
    Logs every event to JSONL. Does NOT send any real orders.

    Thread-safe: uses file append (atomic on most OS/FS).
    """

    def __init__(
        self,
        log_path:          str   = "logs/shadow/fills.jsonl",
        starting_equity:   float = 10_000.0,
        slippage_variance: float = 0.20,    # ±20% randomness around model
    ):
        self.log_path          = log_path
        self.slippage_variance = slippage_variance
        self.state             = ShadowState(equity=starting_equity,
                                             peak_equity=starting_equity)
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        logger.info(f"[PaperExecutor] Initialized. log={log_path} equity=${starting_equity:,.0f}")

    # ------------------------------------------------------------------ #
    # Core API                                                             #
    # ------------------------------------------------------------------ #

    def execute(self, signal: dict) -> SimulatedFill:
        """
        Simulate a trade fill. Returns a SimulatedFill record.
        Raises ValueError if symbol already has an open position.
        """
        sym      = str(signal.get("symbol", "UNKNOWN")).upper()
        side     = str(signal.get("side", "buy")).lower()
        req_px   = float(signal.get("price", 0.0))
        size     = float(signal.get("size",  0.0))
        sl       = signal.get("sl")
        tp       = signal.get("tp")
        meta_in  = signal.get("metadata", {}) or {}
        # Fall back to metadata if top-level fields are missing (many callers only fill metadata).
        strategy = str(signal.get("strategy") or meta_in.get("strategy") or "unknown")
        regime   = str(signal.get("regime")   or meta_in.get("regime")   or "unknown")
        conf     = float(signal.get("confidence") or meta_in.get("confidence") or 0.0)

        if sym in self.state.open_positions:
            logger.warning(f"[PaperExecutor] {sym} already has open position — skipping")
            return self.state.open_positions[sym]

        # Use candle timestamp when available (replay correctness).
        # This prevents "instant replay" runs from collapsing all events into wall-clock time.
        bar_dt = _parse_bar_time(meta_in)
        t0 = bar_dt or datetime.now(timezone.utc)

        # Simulated latency (recorded, not slept) to support execution realism metrics.
        base_lat_ms = float(meta_in.get("latency_base_ms", 80.0))
        jitter_ms = float(meta_in.get("latency_jitter_ms", 170.0))
        # Right-skewed latency: many small, some large.
        latency_ms = max(0.0, random.triangular(0.0, jitter_ms * 3.0, jitter_ms) + base_lat_ms)

        pip_val        = _pip_value(sym)
        override = _load_spread_model()
        avg_spread_pips = (override or {}).get(sym, None)
        if avg_spread_pips is None:
            spread_range = SLIPPAGE_MODEL.get(sym, SLIPPAGE_MODEL["default"])
            avg_spread_pips = random.uniform(spread_range[0], spread_range[1])
        spread         = avg_spread_pips * pip_val
        
        bid            = req_px - spread / 2
        ask            = req_px + spread / 2
        
        atr            = meta_in.get("atr", pip_val * 10)
        atr            = float(atr) if atr is not None else (pip_val * 10)

        # Realistic slippage: bounded by min(ATR * 0.015, spread * 1.5)
        # We cap extreme slippage to prevent destroying realistic edge
        stress = float(meta_in.get("slippage_stress", float(os.getenv("EXECUTION_SLIPPAGE_STRESS", "1.0"))))
        latency_factor = 1.0 + (latency_ms / 500.0)
        
        base_slippage = min(atr * 0.015, spread * 1.5)
        max_slippage = base_slippage * latency_factor * stress

        # Triangular distribution favoring smaller slippage, capped at max_slippage
        entry_slippage = random.triangular(0.0, max_slippage, 0.0)
        
        if side == "buy":
            fill_px = ask + entry_slippage
        else:
            fill_px = bid - entry_slippage

        # Filled time uses candle time + simulated latency (replay correctness).
        t1 = t0
        if latency_ms > 0:
            try:
                from datetime import timedelta
                t1 = t0 + timedelta(milliseconds=latency_ms)
            except Exception:
                t1 = t0

        meta = dict(meta_in)
        meta["spread"] = spread
        meta["atr"] = atr
        meta["entry_slippage"] = entry_slippage
        meta["bar_time"] = (bar_dt or t0).isoformat()
        
        fill = SimulatedFill(
            symbol=sym, side=side,
            strategy=strategy, regime=regime, confidence=conf,
            requested_px=req_px, fill_px=fill_px,
            sl=float(sl) if sl is not None else None,
            tp=float(tp) if tp is not None else None,
            size=size,
            requested_at=t0.isoformat(),
            filled_at=t1.isoformat(),
            latency_ms=latency_ms,
            slippage_pips=entry_slippage / pip_val,
            metadata=meta,
        )

        self.state.open_positions[sym] = fill
        self.state.total_trades       += 1
        self._log("FILL", fill)

        logger.info(
            f"[PaperExecutor] FILL {sym} {side.upper()} {size:.3f} lots @ {fill_px:.5f} "
            f"(slip={(entry_slippage / pip_val):.2f}pip, lat={latency_ms:.1f}ms) strat={strategy}"
        )
        return fill

    def close_position(
        self,
        symbol:    str,
        exit_px:   float,
        reason:    str = "manual",
        close_time: Optional[datetime] = None,
    ) -> Optional[SimulatedFill]:
        """
        Close an open shadow position. Calculates realized PnL.
        Returns updated SimulatedFill or None if symbol not open.
        """
        sym = symbol.upper()
        if sym not in self.state.open_positions:
            logger.warning(f"[PaperExecutor] close_position: {sym} not open")
            return None

        fill      = self.state.open_positions.pop(sym)
        pip_val   = _pip_value(sym)
        
        spread = fill.metadata.get("spread")
        if spread is None:
            spread_range = SLIPPAGE_MODEL.get(sym, SLIPPAGE_MODEL["default"])
            spread = random.uniform(spread_range[0], spread_range[1]) * pip_val
            
        atr = fill.metadata.get("atr", pip_val * 10)
        
        # Exit execution friction
        bid = exit_px - spread / 2
        ask = exit_px + spread / 2
        stress = float(fill.metadata.get("slippage_stress", float(os.getenv("EXECUTION_SLIPPAGE_STRESS", "1.0"))))
        latency_ms = float(fill.latency_ms or 0.0)
        latency_factor = 1.0 + (latency_ms / 500.0)
        
        base_slippage = min(atr * 0.015, spread * 1.5)
        max_slippage = base_slippage * latency_factor * stress
        
        exit_slippage = random.triangular(0.0, max_slippage, 0.0)
        
        if fill.side == "buy":
            close_px = bid - exit_slippage
        else:
            close_px = ask + exit_slippage

        # PnL calculations
        pip_diff  = (close_px - fill.fill_px) / pip_val
        ideal_pip_diff = (exit_px - fill.requested_px) / pip_val
        if fill.side == "sell":
            pip_diff = -pip_diff
            ideal_pip_diff = -ideal_pip_diff

        dollar_pnl = pip_diff * fill.size * 10.0   # $10/pip/standard lot
        ideal_pnl  = ideal_pip_diff * fill.size * 10.0
        
        # Friction tracking
        total_spread_pips = spread / pip_val
        total_slip_pips = (fill.metadata.get("entry_slippage", 0) + exit_slippage) / pip_val
        
        spread_cost = total_spread_pips * fill.size * 10.0
        slippage_cost = total_slip_pips * fill.size * 10.0
        total_cost = spread_cost + slippage_cost

        # R-multiple: PnL relative to initial risk
        r_multiple = None
        if fill.sl is not None and fill.sl > 0 and fill.fill_px > 0:
            risk_pips = abs(fill.fill_px - fill.sl) / pip_val
            if risk_pips > 0:
                r_multiple = round(pip_diff / risk_pips, 3)

        # Hold time (use candle timestamps if provided)
        try:
            open_dt  = datetime.fromisoformat(fill.filled_at.replace("Z", "+00:00"))
            close_dt = close_time or datetime.now(timezone.utc)
            hold_min = (close_dt - open_dt).total_seconds() / 60.0
        except Exception:
            hold_min = 0.0

        fill.status       = "closed"
        fill.close_px     = close_px
        fill.close_reason = reason
        fill.pnl_usd      = round(dollar_pnl, 4)
        fill.hold_minutes = round(hold_min, 1)
        fill.r_multiple   = r_multiple
        fill.spread_cost  = round(spread_cost, 4)
        fill.slippage_cost = round(slippage_cost, 4)
        fill.total_cost   = round(total_cost, 4)
        fill.ideal_pnl_usd = round(ideal_pnl, 4)

        # Update equity
        self.state.equity      += dollar_pnl
        self.state.peak_equity  = max(self.state.peak_equity, self.state.equity)
        if dollar_pnl > 0:
            self.state.winning_trades += 1

        self.state.closed_trades.append(fill)
        self._log("CLOSE", fill)
        try:
            from src.ml.ai_brain import safe_record_exit
            safe_record_exit(fill)
        except Exception:
            pass

        logger.info(
            f"[PaperExecutor] CLOSE {sym} @ {exit_px:.5f} | "
            f"PnL=${dollar_pnl:+.2f} R={r_multiple} reason={reason}"
        )
        return fill

    def check_sl_tp(
        self,
        symbol: str,
        current_high: float,
        current_low: float,
        bar_time: Optional[datetime] = None,
    ):
        """
        Called once per bar. Auto-closes if SL or TP was touched.
        Simulates realistic fill: SL at SL price, TP at TP price.
        """
        sym = symbol.upper()
        if sym not in self.state.open_positions:
            return
        fill = self.state.open_positions[sym]

        if fill.side == "buy":
            # PESSIMISTIC RESOLUTION: Always check SL first
            if fill.sl and current_low <= fill.sl:
                self.close_position(sym, fill.sl, reason="sl_hit", close_time=bar_time)
                return
            if fill.tp and current_high >= fill.tp:
                self.close_position(sym, fill.tp, reason="tp_hit", close_time=bar_time)
        else:  # sell
            # PESSIMISTIC RESOLUTION: Always check SL first
            if fill.sl and current_high >= fill.sl:
                self.close_position(sym, fill.sl, reason="sl_hit", close_time=bar_time)
                return
            if fill.tp and current_low <= fill.tp:
                self.close_position(sym, fill.tp, reason="tp_hit", close_time=bar_time)

    # ------------------------------------------------------------------ #
    # State queries                                                        #
    # ------------------------------------------------------------------ #

    def get_open_positions(self) -> Dict[str, SimulatedFill]:
        return dict(self.state.open_positions)

    def get_equity(self) -> float:
        return self.state.equity

    def get_drawdown_pct(self) -> float:
        if self.state.peak_equity <= 0:
            return 0.0
        dd = (self.state.peak_equity - self.state.equity) / self.state.peak_equity
        return round(dd * 100.0, 4)

    def get_summary(self) -> dict:
        closed = self.state.closed_trades
        wins   = [t for t in closed if (t.pnl_usd or 0) > 0]
        losses = [t for t in closed if (t.pnl_usd or 0) <= 0]
        gross_profit = sum(t.pnl_usd or 0 for t in wins)
        gross_loss   = abs(sum(t.pnl_usd or 0 for t in losses))
        slips        = [t.slippage_pips for t in closed] or [0.0]
        lats         = [t.latency_ms     for t in closed] or [0.0]
        
        total_ideal_pnl = sum(t.ideal_pnl_usd or 0 for t in closed)
        realized_pnl = gross_profit - gross_loss
        total_spread = sum(getattr(t, "spread_cost", 0) for t in closed)
        total_slip = sum(getattr(t, "slippage_cost", 0) for t in closed)
        friction_loss_pct = ((total_ideal_pnl - realized_pnl) / abs(total_ideal_pnl) * 100.0) if total_ideal_pnl != 0 else 0.0

        # Replay correctness: measure trade frequency by candle timestamps when available.
        start_ts = None
        end_ts = None
        try:
            if closed:
                start_ts = datetime.fromisoformat(closed[0].filled_at.replace("Z", "+00:00"))
                end_ts = datetime.fromisoformat(closed[-1].filled_at.replace("Z", "+00:00"))
        except Exception:
            start_ts = None
            end_ts = None
        trades_per_day = None
        if start_ts and end_ts and end_ts >= start_ts:
            days = max(1.0, (end_ts - start_ts).total_seconds() / 86400.0)
            trades_per_day = len(closed) / days
        
        return {
            "n_trades":        len(closed),
            "n_open":          len(self.state.open_positions),
            "win_rate":        round(len(wins) / max(len(closed), 1), 4),
            "profit_factor":   round(gross_profit / max(gross_loss, 0.01), 4),
            "total_pnl_usd":   round(self.state.equity - self.state.peak_equity +
                                     gross_profit + gross_loss, 4),
            "net_pnl_usd":     round(self.state.equity - 10_000.0, 4),
            "equity":          round(self.state.equity, 2),
            "drawdown_pct":    self.get_drawdown_pct(),
            "avg_slippage_pips": round(float(np.mean(slips)), 3),
            "avg_latency_ms":    round(float(np.mean(lats)), 2),
            "friction_spread_usd": round(total_spread, 2),
            "friction_slip_usd":   round(total_slip, 2),
            "friction_loss_pct":   round(friction_loss_pct, 2),
            "trade_span_start": start_ts.isoformat() if start_ts else None,
            "trade_span_end": end_ts.isoformat() if end_ts else None,
            "trades_per_day": round(trades_per_day, 3) if trades_per_day is not None else None,
        }

    # ------------------------------------------------------------------ #
    # JSONL logging                                                        #
    # ------------------------------------------------------------------ #

    def _log(self, event_type: str, fill: SimulatedFill):
        record = {"event": event_type}
        record.update(dataclasses.asdict(fill))
        record["ts"] = datetime.now(timezone.utc).isoformat()
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception as e:
            logger.error(f"[PaperExecutor] Failed to write log: {e}")
