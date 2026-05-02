"""
src/pipeline/pipeline_v2.py
============================
SCOPUS Unified Pipeline v2 — Week 10.

Drop-in enhanced pipeline that replaces the inline helpers in shadow_runner.py.
The calling interface is identical:  BarResult = pipeline.process_bar(...)

Adds over v1 (shadow_runner.py original):
  ✅ Extended features (52 new features, shadow-only)
  ✅ Session + S/R + Volume + Cross-asset features
  ✅ AdaptiveSizer (confidence-tiered, vol-scaled, portfolio-capped)
  ✅ WeightedSignalAggregator (regime-weighted, correlation-guarded)
  ✅ PSI drift monitoring (logged per bar, Prometheus gauge)
  ✅ Ensemble comparison log alongside single-strategy fills

Architecture (per bar):
    DataSource
        ↓
    common_features   (20 features)  ← always on
    extended_features (52)           ← shadow-only flag
    session_features  (10)           ← shadow-only flag
    structure_features(8)            ← shadow-only flag
    volume_features   (8)            ← shadow-only flag
        ↓
    MetaGatingBrain → regime
        ↓
    StrategyBank → List[StrategySignal]
        ↓
    WeightedSignalAggregator → EnsembleDecision
        ↓
    AdaptiveSizer  → lot size
        ↓
    CircuitBreaker gate
        ↓
    PaperExecutor.execute() → SimulatedFill
        ↓
    DriftDetector.compute_psi() → PSI log + Prometheus gauge
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

LOG = logging.getLogger("pipeline_v2")

# ---------------------------------------------------------------------------
# Pipeline configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """All feature flags and sizing parameters for the v2 pipeline."""

    # Feature flags
    use_extended_features: bool  = True    # lag, vol regime, interaction, mic-mom
    use_session_features:  bool  = True    # cyclical time encoding
    use_structure_features: bool = True    # S/R proximity
    use_volume_features:   bool  = True    # OBV, VWAP, volume structure
    use_cross_asset:       bool  = False   # Week 11 — not active yet

    # Sizing
    account_equity:   float = 10_000.0
    base_risk_pct:    float = 0.01         # 1% per trade base
    use_adaptive_sizing: bool = True

    # Ensemble
    use_ensemble:     bool  = True         # WeightedSignalAggregator
    ensemble_log:     str   = "logs/shadow/ensemble.jsonl"

    # Drift monitoring
    monitor_psi:      bool  = True
    psi_log_path:     str   = "logs/shadow/psi_daily.jsonl"
    psi_baseline_bars: int  = 200          # bars to baseline PSI from

    # Min bars required before computing extended features
    min_bars:         int   = 150
    tier1_confidence_min: float = 0.60
    tier2_confidence_min: float = 0.55
    tier3_confidence_floor: float = 0.55

    @classmethod
    def from_cfg(cls, cfg: dict) -> "PipelineConfig":
        """Build from weapon_system.yaml config dict."""
        feat = cfg.get("extended_features", {})
        risk = cfg.get("risk", {})
        gating = cfg.get("trade_gating", {})
        return cls(
            use_extended_features  = feat.get("use_extended",   True),
            use_session_features   = feat.get("use_session",    True),
            use_structure_features = feat.get("use_structure",  True),
            use_volume_features    = feat.get("use_volume",     True),
            use_cross_asset        = feat.get("use_cross_asset", False),
            account_equity         = risk.get("account_equity",  10_000.0),
            base_risk_pct          = risk.get("base_risk_pct",   0.01),
            use_adaptive_sizing    = risk.get("use_adaptive_sizing", True),
            use_ensemble           = cfg.get("ensemble", {}).get("enabled", True),
            tier1_confidence_min   = gating.get("tier1_confidence_min", 0.60),
            tier2_confidence_min   = gating.get("tier2_confidence_min", 0.52),
            tier3_confidence_floor = gating.get("tier3_confidence_floor", 0.50),
        )


# ---------------------------------------------------------------------------
# Bar result
# ---------------------------------------------------------------------------

@dataclass
class BarResult:
    """Output from pipeline.process_bar() for one symbol and bar."""
    symbol:        str
    bar_index:     int
    feature_count: int
    regime:        str
    confidence:    float
    signal:        Optional[dict]           # None = no trade
    fill:          Optional[object]         # SimulatedFill or None
    ensemble_decision: Optional[object]     # EnsembleDecision or None
    psi_max:       float = 0.0
    errors:        List[str] = field(default_factory=list)

    @property
    def traded(self) -> bool:
        return self.fill is not None


# ---------------------------------------------------------------------------
# Lazy singletons (module level, created once per process)
# ---------------------------------------------------------------------------

_sizer:    Optional[object] = None
_ensemble: Optional[object] = None
_drift:    Optional[object] = None
_drift_baseline: Optional[pd.DataFrame] = None
_bar_count_for_psi: int = 0


def _get_sizer(cfg: PipelineConfig) -> object:
    global _sizer
    if _sizer is None and cfg.use_adaptive_sizing:
        from src.risk.adaptive_sizing import AdaptiveSizer
        _sizer = AdaptiveSizer(
            account_equity=cfg.account_equity,
            base_risk_pct=cfg.base_risk_pct,
        )
    return _sizer


def _get_ensemble(cfg: PipelineConfig) -> object:
    global _ensemble
    if _ensemble is None and cfg.use_ensemble:
        from src.decision.ensemble import WeightedSignalAggregator
        _ensemble = WeightedSignalAggregator(
            log_path=cfg.ensemble_log, shadow_mode=True
        )
    return _ensemble


# ---------------------------------------------------------------------------
# Feature computation v2
# ---------------------------------------------------------------------------

def build_full_feature_set(
    raw_df: pd.DataFrame,
    cfg: PipelineConfig,
    symbol: str = "UNKNOWN",
    data_source=None,
) -> Optional[pd.DataFrame]:
    """
    Run all feature categories and return a merged DataFrame.
    Falls back gracefully if any category fails.

    Returns:
        DataFrame with [20 + up_to_82 extended] columns, or None.
    """
    if raw_df is None or raw_df.empty or len(raw_df) < cfg.min_bars:
        return None

    try:
        from src.features.common_features import compute_features_from_ohlcv
        base_df = compute_features_from_ohlcv(raw_df)
    except Exception as e:
        LOG.warning(f"[pipeline_v2] base feature error: {e}")
        return None

    frames = [base_df]

    # Extended features (lagged, vol regime, interaction, micro-momentum)
    if cfg.use_extended_features:
        try:
            from src.features.extended_features import ExtendedFeatures
            ext_df = ExtendedFeatures.compute(raw_df, base_df)
            frames.append(ext_df)
        except Exception as e:
            LOG.debug(f"[pipeline_v2] extended_features error ({symbol}): {e}")

    # Session features
    if cfg.use_session_features:
        try:
            from src.features.session_features import compute_session_features
            sess_df = compute_session_features(raw_df)
            frames.append(sess_df)
        except Exception as e:
            LOG.debug(f"[pipeline_v2] session_features error ({symbol}): {e}")

    # S/R structure features
    if cfg.use_structure_features:
        try:
            from src.features.structure_features import compute_structure_features
            from src.features.common_features import _compute_atr
            atr14 = _compute_atr(raw_df, 14)
            sr_df  = compute_structure_features(raw_df, atr14)
            frames.append(sr_df)
        except Exception as e:
            LOG.debug(f"[pipeline_v2] structure_features error ({symbol}): {e}")

    # Volume structure features
    if cfg.use_volume_features:
        try:
            from src.features.volume_features import compute_volume_features
            from src.features.common_features import _compute_atr
            atr14  = _compute_atr(raw_df, 14)
            vol_df = compute_volume_features(raw_df, atr14)
            frames.append(vol_df)
        except Exception as e:
            LOG.debug(f"[pipeline_v2] volume_features error ({symbol}): {e}")

    full_df = pd.concat(frames, axis=1)
    full_df = full_df.ffill().bfill().fillna(0.0)

    # Cross-asset signals (Week 11 — appended last, constant across rows)
    if cfg.use_cross_asset and data_source is not None:
        try:
            from src.features.cross_asset_integration import append_cross_asset_row
            full_df = append_cross_asset_row(full_df, data_source)
        except Exception as e:
            LOG.debug(f"[pipeline_v2] cross_asset error ({symbol}): {e}")

    return full_df


# ---------------------------------------------------------------------------
# Drift monitoring
# ---------------------------------------------------------------------------

def _update_psi(
    full_df:  pd.DataFrame,
    cfg:      PipelineConfig,
    bar_idx:  int,
) -> float:
    """
    Fit or update DriftDetector with latest window, compute PSI,
    write to psi_daily.jsonl, update Prometheus gauge.

    Returns:
        max_psi across all features, or 0.0 if not measured.
    """
    global _drift, _drift_baseline, _bar_count_for_psi
    if not cfg.monitor_psi:
        return 0.0

    _bar_count_for_psi += 1

    try:
        from src.monitoring.drift import DriftDetector
        import json
        from datetime import datetime, timezone

        # On first call — build baseline from first N bars
        if _drift is None:
            if len(full_df) >= cfg.psi_baseline_bars:
                _drift_baseline = full_df.iloc[:cfg.psi_baseline_bars].copy()
                _drift = DriftDetector(_drift_baseline)
                LOG.info(f"[pipeline_v2] DriftDetector baseline fitted on {cfg.psi_baseline_bars} bars")
            return 0.0

        # Compute PSI every 50 bars (not every bar — expensive)
        if _bar_count_for_psi % 50 != 0:
            return 0.0

        current_window = full_df.iloc[-100:]  # last 100 bars as current
        psi_scores = _drift.compute_psi(current_window)
        summary    = _drift.summary(psi_scores)
        alerts     = _drift.check_alerts(psi_scores)
        max_psi    = summary.get("max_psi", 0.0)

        # Log to jsonl
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "bar_idx":   bar_idx,
            **summary,
            "n_alerts": len(alerts),
        }
        os.makedirs(os.path.dirname(os.path.abspath(cfg.psi_log_path)), exist_ok=True)
        with open(cfg.psi_log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        # Prometheus gauge update (graceful if unavailable)
        try:
            from src.monitoring.prometheus_exporter import (
                FEATURE_PSI_MAX, FEATURE_PSI_WARN, FEATURE_PSI_CRIT
            )
            FEATURE_PSI_MAX.set(max_psi)
            FEATURE_PSI_WARN.set(summary.get("n_warning", 0))
            FEATURE_PSI_CRIT.set(summary.get("n_drifted", 0))
        except Exception:
            pass

        if alerts:
            LOG.warning(f"[pipeline_v2] PSI DRIFT ALERT: {len(alerts)} features "
                        f"(max_psi={max_psi:.4f})")
        return max_psi

    except Exception as e:
        LOG.debug(f"[pipeline_v2] PSI update error: {e}")
        return 0.0


# ---------------------------------------------------------------------------
# Adaptive sizing helper
# ---------------------------------------------------------------------------

def _apply_adaptive_size(
    signal:  dict,
    regime:  dict,
    full_df: pd.DataFrame,
    cfg:     PipelineConfig,
) -> dict:
    """
    Fixed-fractional position sizing — flat risk per trade.

    AUDIT FINDING (2026-03-29): Confidence-scaled sizing was HURTING PF.
      Top-quartile lots → PF=0.891  (large lots on losers)
      Bot-quartile lots → PF=4.479  (small lots on winners)
      corr(size, R) = -0.186 → confidence is NOT a reliable size predictor.

    Fix: Use flat fixed-fractional sizing. Risk exactly base_risk_pct of
    account equity per trade, sized by SL distance in pips.
    Confidence scaling re-enabled only after 200+ trades with WR > 55%.
    """
    from src.risk.adaptive_sizing import _pip_value

    pip_val   = _pip_value(signal.get("symbol", "EURUSD"))
    price     = signal.get("price", 0.0) or 0.0
    sl        = signal.get("sl")  or price
    stop_pips = abs(price - sl) / max(pip_val, 1e-10)

    if stop_pips < 0.5:       # degenerate SL — safety floor
        signal["size"] = 0.01
        return signal

    # Risk exactly base_risk_pct of equity — no confidence scaling
    risk_usd  = cfg.account_equity * cfg.base_risk_pct
    base_size = risk_usd / max(stop_pips * 10.0, 1e-9)   # $10/pip/lot for FX majors
    signal["size"] = round(max(0.01, min(2.0, base_size)), 2)
    return signal


def _grade_signal_quality(features: dict, signal_side: str, regime_label: str) -> dict:
    """Return a letter-grade quality assessment for the selected signal."""
    default_payload = {"grade": "B", "score": None, "recommendation": "FULL"}
    if signal_side not in {"BUY", "SELL"}:
        return default_payload

    try:
        from src.backtest.quality_scorer import QualityScorer

        scorer = QualityScorer()
        result = scorer.calculate(
            features=features,
            signal=signal_side,
            regime=str(regime_label or "UNKNOWN").strip().upper(),
            confidence=0.5,
        )
        payload = result.to_dict()
        payload["score"] = result.total_score
        return payload
    except Exception:
        return default_payload


def _apply_signal_tier_gate(signal: dict, cfg: PipelineConfig, quality_grade: str, last_features: dict = None) -> tuple[Optional[dict], str, float]:
    """
    Apply the 3-tier signal gate (v2 — tightened 2026-05-01):
    - Tier-1 (strong):     conf >= 0.60 AND grade in {A, B} → full size (1.2x if conf >= 0.65)
    - Tier-2 (borderline): 0.55 <= conf < 0.60 AND grade == C → half size
    - Tier-3 (reject):     conf < 0.55 OR grade in {D, F} → skip
    - MR Exception:        0.50 <= conf < 0.55 AND |boll_z| >= 1.1 → size *= 0.6
    """
    confidence = float(signal.get("confidence", 0.0) or 0.0)
    grade = str(quality_grade or "F").strip().upper()

    if last_features is None:
        last_features = {}
    boll_z = abs(last_features.get("boll_z", 0.0))
    is_mr_exception = signal.get("strategy") == "silver_mr" and 0.50 <= confidence < 0.55 and boll_z >= 1.1

    # Tier-3: reject low-confidence OR poor quality
    if (confidence < cfg.tier3_confidence_floor and not is_mr_exception) or grade in {"D", "F"}:
        return None, "reject", 0.0

    # Exception handling for MR
    if is_mr_exception:
        gated = dict(signal)
        gated["size"] = round(max(0.01, float(signal.get("size", 0.01) or 0.01) * 0.6), 2)
        return gated, "exception_mr", 0.6

    # Tier-1: high-confidence + good quality → full size
    if confidence >= cfg.tier1_confidence_min and grade in {"A", "B"}:
        if confidence >= 0.65:
            gated = dict(signal)
            gated["size"] = round(max(0.01, float(signal.get("size", 0.01) or 0.01) * 1.2), 2)
            return gated, "strong", 1.2
        return signal, "strong", 1.0

    # Tier-2: borderline — BOTH conditions required (AND, not OR)
    if confidence >= cfg.tier2_confidence_min and confidence < cfg.tier1_confidence_min and grade == "C":
        gated = dict(signal)
        gated["size"] = round(max(0.01, float(signal.get("size", 0.01) or 0.01) * 0.5), 2)
        return gated, "borderline", 0.5

    # Everything else rejected
    return None, "reject", 0.0


# ---------------------------------------------------------------------------
# Strategy bank (multi-signal output for ensemble)
# ---------------------------------------------------------------------------

def _collect_strategy_signals(
    symbol:   str,
    features: dict,
    full_df:  pd.DataFrame,
    regime:   dict,
    cfg_yaml: dict,
) -> List[object]:
    """
    Collect StrategySignal from each enabled strategy.
    Returns list of StrategySignal (may be empty).
    """
    from src.decision.ensemble import StrategySignal
    signals = []
    enabled = cfg_yaml.get("strategies", {})
    close   = features.get("close", 0.0)
    atr     = features.get("atr14", 0.0)
    rsi     = features.get("rsi14", 50.0)

    # ── Regime-aware directional guard ───────────────────────────────────
    # BUY PF=1.364 vs SELL PF=0.865 — SELL into TREND is the identified leak.
    # In TREND regime: only take signals aligned WITH the short-term trend.
    # In UNCERTAIN: allow both sides (no clear trend to fight).
    current_regime = regime.get("regime", "UNCERTAIN")
    sma_ratio      = features.get("sma_ratio", 1.0)   # sma5/sma20: >1=uptrend, <1=downtrend
    trend_is_up    = sma_ratio > 1.0
    trend_is_down  = sma_ratio < 1.0

    # ── Entry quality gate — 2 independently validated conditions ────────
    # CONFIRMED (2026-04-01, sampling shows atr_pctile self-calibrates
    # every 100 bars, making it useless as an absolute vol filter):
    #
    # GATE 1: |boll_z| ≥ 1.0  (calibrated entry relaxation for 3-month test)
    #
    # GATE 2: ABSOLUTE ATR range  (replaces relative atr_pctile gate)
    #   Probe: D01-30 bad period has ATR=17-24pip while atr_pctile=0.29-0.50
    #   because the 100-bar window recalibrates to local high-vol conditions.
    #   EURUSD MR works when ATR 5–16 pip (H1):
    #     < 5 pip: too quiet, spread dominates, SL always hit
    #     > 16 pip: too volatile, TP never reached (SL=24pip, TP=32pip too far)
    #   Both failure zones observed in the 90-day data (D01-30 = high-vol loss)
    boll_z  = features.get("boll_z", 0.0)
    atr14   = features.get("atr14", 0.001)   # absolute ATR in price units
    atr_pctile = features.get("atr_pctile", 0.5)

    ATR_MIN_PRICE = 0.0005   # 5 pip  — floor: spread noise kills edge
    ATR_MAX_PRICE = 0.0016   # 16 pip — ceiling: TP too far, trend continuation (REVERTED)

    # === GLOBAL EDGE FILTER: effective_edge = |boll_z| × atr_pctile ===
    # Both dimensions must confirm: price displaced AND volatility is in range.
    # This is the primary quality gate — no broad bypasses.
    effective_edge = abs(boll_z) * atr_pctile

    # Hard global floor — signals below this have insufficient edge to trade
    if effective_edge < 0.50:
        return signals

    # STRICT TRANSITION CONTROL: TRANSITION regime needs stronger confirmation
    if current_regime == "TRANSITION" and effective_edge < 0.55:
        return signals

    # boll_z floor (keep 0.7 from prior fix)
    if abs(boll_z) < 0.7:
        return signals   # not extended enough

    # ATR gate: original 16 pip ceiling restored
    if atr14 < ATR_MIN_PRICE or atr14 > ATR_MAX_PRICE:
        return signals   # vol regime outside MR operating range


    def _side_allowed(side: str) -> bool:
        """Block counter-trend MR signals in TREND regime."""
        if current_regime == "TREND":
            # In uptrend: only buy-the-dip allowed, no short selling
            if side == "sell" and trend_is_up:
                return False
            # In downtrend: only sell-the-rally allowed, no buy
            if side == "buy" and trend_is_down:
                return False
        return True


    # ── Tokyo MR ────────────────────────────────────────────────────────
    if enabled.get("tokyo_session_mr", {}).get("enabled", False):
        # Momentum confirmation: RSI must be TURNING (not just at threshold)
        rsi_lag          = features.get("lag_rsi3", rsi)  # 3-bar lagged RSI
        rsi_turning_up   = rsi > rsi_lag   # recovering from oversold
        rsi_turning_down = rsi < rsi_lag   # rolling from overbought

        if rsi < 40 and rsi_turning_up and _side_allowed("buy"):
            signals.append(StrategySignal(
                strategy="tokyo_session_mr", symbol=symbol, side="buy",
                confidence=0.45 + (40 - rsi) / 100.0,
                price=close,
                sl=close - atr * 1.5,   # 1.5 ATR stop (wide enough for H1 noise)
                tp=close + atr * 2.0,   # 2.0 ATR target (asymmetric: need only 43%+ WR)
                regime=current_regime,
            ))
        elif rsi > 60 and rsi_turning_down and _side_allowed("sell"):
            signals.append(StrategySignal(
                strategy="tokyo_session_mr", symbol=symbol, side="sell",
                confidence=0.45 + (rsi - 60) / 100.0,
                price=close,
                sl=close + atr * 1.5,
                tp=close - atr * 2.0,
                regime=current_regime,
            ))

    # ── MEAN_REVERSION (MATHEMATICALLY CORRECT EDGE) ────────────────────────────────
    # Primary alpha: Mean reversion to Bollinger mean with strict edge conditions
    
    if enabled.get("silver_mr", {}).get("enabled", False):
        boll_z = features.get("boll_z", 0.0)
        atr_pctile = features.get("atr_pctile", 0.5)
        adx = features.get("adx14", 25.0)  # Default high to require explicit low ADX
        
        # === ENTRY CONDITIONS (ALL must pass) ===
        
        # 1. abs(boll_z) >= 1.5 — quality pullback from mean
        if abs(boll_z) < 1.5:
            return signals
        
        # 2. atr_pctile >= 0.55 — sufficient volatility for edge
        if atr_pctile < 0.55:
            return signals
        
        # 3. ADX < 20 — anti-trend filter (mean reversion works in ranging markets)
        if adx >= 20.0:
            return signals
        
        # 4. SYMBOL FILTER: EURUSD and NZDUSD only
        if symbol.upper() not in ["EURUSD", "NZDUSD"]:
            return signals
        
        # 5. CONFIRMATION: rejection_wick AND engulfing candle
        try:
            if len(full_df) >= 3:
                curr_open = float(full_df["open"].iloc[-1])
                curr_close = float(full_df["close"].iloc[-1])
                curr_high = float(full_df["high"].iloc[-1])
                curr_low = float(full_df["low"].iloc[-1])
                prev_close = float(full_df["close"].iloc[-2])
                prev_open = float(full_df["open"].iloc[-2])
                prev_high = float(full_df["high"].iloc[-2])
                prev_low = float(full_df["low"].iloc[-2])
            else:
                return signals
        except Exception:
            return signals
        
        # Current candle metrics
        curr_body = abs(curr_close - curr_open)
        curr_range = curr_high - curr_low
        
        # Rejection wick: long wick in direction of the reversal
        # For buy (boll_z < 0): long lower wick, close near high
        # For sell (boll_z > 0): long upper wick, close near low
        if boll_z < 0:  # Buy setup
            lower_wick = min(curr_close, curr_open) - curr_low
            rejection_wick = (lower_wick > 1.5 * curr_body) and (curr_close > curr_open)
            # Engulfing: current candle body engulfs previous candle
            engulfing = curr_body > abs(prev_close - prev_open) and curr_close > prev_close
        else:  # Sell setup
            upper_wick = curr_high - max(curr_close, curr_open)
            rejection_wick = (upper_wick > 1.5 * curr_body) and (curr_close < curr_open)
            # Engulfing: current candle body engulfs previous candle
            engulfing = curr_body > abs(prev_close - prev_open) and curr_close < prev_close
        
        if not (rejection_wick and engulfing):
            return signals
        
        # === DYNAMIC EXIT LOGIC (NOT FIXED R) ===
        # TP = Bollinger mean (middle band)
        # SL = entry ± (1.2 × ATR)
        
        bb_middle = features.get("bb_middle", close)
        if bb_middle is None or bb_middle <= 0:
            bb_middle = close  # Fallback
        
        entry = curr_close
        
        # Take Profit = distance to Bollinger mean
        if boll_z < 0:  # Buy
            tp = bb_middle
            sl = entry - (atr * 1.2)
            side = "buy"
        else:  # Sell
            tp = bb_middle
            sl = entry + (atr * 1.2)
            side = "sell"
        
        # Effective R = distance to TP / distance to SL
        tp_distance = abs(tp - entry)
        sl_distance = abs(sl - entry)
        effective_r = tp_distance / sl_distance if sl_distance > 0 else 0
        
        # Skip if effective R is too low (need at least 1:1)
        if effective_r < 1.0:
            return signals
        
        # === COST FILTER (CRITICAL FIX) ===
        # expected_move >= total_cost × 14
        
        # Get spread from slippage model
        from src.broker.paper_executor import SLIPPAGE_MODEL
        sym_upper = symbol.upper()
        pip_val = 0.01 if "JPY" in sym_upper else 0.0001
        spread_range = SLIPPAGE_MODEL.get(sym_upper, SLIPPAGE_MODEL.get("default", (1.0, 2.0)))
        avg_spread_pips = (spread_range[0] + spread_range[1]) / 2
        spread_cost = avg_spread_pips * pip_val
        
        # Expected slippage (conservative)
        expected_slippage = spread_cost
        total_cost = spread_cost + expected_slippage
        
        # Expected move = distance to TP
        expected_move = tp_distance
        
        # Strict cost filter: expected_move >= 14 × total_cost
        if expected_move < total_cost * 14:
            return signals
        
        # Calculate cost ratio for logging
        cost_ratio = expected_move / total_cost if total_cost > 0 else 0
        
        # Confidence based on edge quality
        base_conf = 0.60 + (abs(boll_z) - 1.5) * 0.10 + (atr_pctile - 0.55) * 0.20
        base_conf = min(0.85, max(0.50, base_conf))
        
        # Create signal with rich metadata
        signals.append(StrategySignal(
            strategy="silver_mr", symbol=symbol, side=side,
            confidence=round(base_conf, 3),
            price=entry,
            sl=round(sl, 5),
            tp=round(tp, 5),
            regime=current_regime,
            metadata={
                "boll_z": round(boll_z, 3),
                "atr_pctile": round(atr_pctile, 3),
                "adx": round(adx, 1),
                "expected_move": round(expected_move, 5),
                "total_cost": round(total_cost, 5),
                "cost_ratio": round(cost_ratio, 1),
                "effective_r": round(effective_r, 2),
                "bb_middle": round(bb_middle, 5),
                "rejection_wick": rejection_wick,
                "engulfing": engulfing,
            }
        ))

    return signals


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------

class PipelineV2:
    """
    Unified SCOPUS pipeline v2.

    Usage:
        pipeline = PipelineV2(cfg=PipelineConfig.from_cfg(yaml_cfg))
        for bar in data_source.stream(symbol, timeframe):
            raw_df = data_source.get_history(symbol, timeframe, n_bars=250)
            result = pipeline.process_bar(symbol, bar, raw_df, bar_idx)
            if result.fill:
                ...
    """

    def __init__(self, cfg: PipelineConfig = None, cfg_yaml: dict = None):
        self.cfg      = cfg or PipelineConfig()
        self.cfg_yaml = cfg_yaml or {}
        self._bar_idx = 0
        self._feature_names: List[str] = []

    def process_bar(
        self,
        symbol:  str,
        bar,                  # BarData namedtuple (open/high/low/close/volume)
        raw_df:  pd.DataFrame,
        executor,             # PaperExecutor
    ) -> BarResult:
        """
        Full bar processing: features → regime → ensemble → size → execute.

        Returns BarResult even on failure (errors field populated).
        """
        self._bar_idx += 1
        result = BarResult(
            symbol=symbol, bar_index=self._bar_idx,
            feature_count=0, regime="UNCERTAIN", confidence=0.0,
            signal=None, fill=None, ensemble_decision=None,
        )

        # ── 1. Check existing SL/TP ───────────────────────────────────────
        try:
            bar_time = getattr(bar, "time", None) or getattr(bar, "timestamp", None)
            executor.check_sl_tp(symbol, bar.high, bar.low, bar_time=bar_time)
        except Exception as e:
            LOG.debug(f"[pipeline_v2] sl_tp check error: {e}")

        # ── 2. Skip if position open ─────────────────────────────────────
        try:
            if symbol in executor.get_open_positions():
                return result
        except Exception:
            pass

        # ── 3. Feature pipeline ───────────────────────────────────────────
        full_df = build_full_feature_set(raw_df, self.cfg, symbol)
        if full_df is None or full_df.empty:
            result.errors.append("feature_pipeline_failed")
            return result

        result.feature_count = len(full_df.columns)
        if not self._feature_names:
            self._feature_names = list(full_df.columns)

        last_features = full_df.iloc[-1].to_dict()
        last_features["close"] = float(bar.close)  # ensure latest close

        # ── 4. Drift monitoring ───────────────────────────────────────────
        result.psi_max = _update_psi(full_df, self.cfg, self._bar_idx)

        # ── 5. Regime ─────────────────────────────────────────────────────
        try:
            from src.decision.meta_gating import MetaGatingBrain
            brain  = MetaGatingBrain()
            regime = brain.classify_regime(last_features)
            regime = regime if isinstance(regime, dict) else {"regime": str(regime), "confidence": 0.5}
        except Exception as e:
            regime = {"regime": "UNCERTAIN", "confidence": 0.0}
            LOG.debug(f"[pipeline_v2] regime error: {e}")

        result.regime     = regime.get("regime", "UNCERTAIN")
        result.confidence = float(regime.get("confidence", 0.0))

        # ── 6. Strategy bank → signals ────────────────────────────────────
        raw_signals = _collect_strategy_signals(
            symbol, last_features, full_df, regime, self.cfg_yaml
        )

        if not raw_signals:
            return result

        # ── 7. Ensemble aggregation ───────────────────────────────────────
        final_signal: Optional[dict] = None
        ens_decision = None

        if self.cfg.use_ensemble:
            ensemble = _get_ensemble(self.cfg)
            if ensemble is not None:
                ens_decision = ensemble.aggregate(raw_signals, result.regime, symbol)
                result.ensemble_decision = ens_decision
                if ens_decision.should_trade:
                    final_signal = ens_decision.to_signal_dict()
        else:
            # Fallback: single best signal (highest confidence)
            best = max(raw_signals, key=lambda s: s.confidence)
            final_signal = {
                "symbol":     symbol,
                "side":       best.side,
                "price":      best.price,
                "sl":         best.sl,
                "tp":         best.tp,
                "strategy":   best.strategy,
                "regime":     result.regime,
                "confidence": best.confidence,
            }

        if final_signal is None:
            return result

        result.signal = final_signal

        # ── 8. Adaptive sizing ────────────────────────────────────────────
        try:
            result.signal = _apply_adaptive_size(
                result.signal, regime, full_df, self.cfg
            )
            result.signal["size"] = max(result.signal.get("size", 0.01), 0.02)
        except Exception as e:
            result.signal["size"] = 0.02
            LOG.debug(f"[pipeline_v2] sizing error: {e}")

        # TREND regime: size boost for directional alpha
        _sig_regime = result.signal.get("regime", result.regime)
        if _sig_regime == "TREND":
            result.signal["size"] = round(result.signal["size"] * 1.2, 2)
        
        # Low volatility: half size rather than skip
        if last_features.get("atr_pctile", 0.5) < 0.25:
            result.signal["size"] = round(result.signal["size"] * 0.5, 2)
            result.signal["size"] = max(result.signal["size"], 0.01)

        # Weak pullback penalty (0.4 <= |boll_z| < 0.5)
        if result.signal.get("strategy") == "silver_mr":
            bz = abs(last_features.get("boll_z", 0.0))
            if 0.4 <= bz < 0.5:
                result.signal["size"] = round(result.signal["size"] * 0.7, 2)
                result.signal["size"] = max(result.signal["size"], 0.01)
        quality_payload = _grade_signal_quality(
            last_features,
            str(result.signal.get("side", "")).strip().upper(),
            _sig_regime,
        )
        quality_grade = str(quality_payload.get("grade", "B")).strip().upper()
        gated_signal, signal_tier, tier_size_multiplier = _apply_signal_tier_gate(
            result.signal,
            self.cfg,
            quality_grade,
            last_features,
        )
        if gated_signal is None:
            result.signal = None
            result.errors.append(f"tier3_reject:{quality_grade}")
            return result
        result.signal = gated_signal
        result.signal.setdefault("metadata", {})
        if isinstance(result.signal.get("metadata"), dict):
            result.signal["metadata"]["signal_tier"] = signal_tier
            result.signal["metadata"]["signal_tier_size_multiplier"] = tier_size_multiplier
            result.signal["metadata"]["quality_grade"] = quality_grade
            result.signal["metadata"]["quality_score"] = quality_payload.get("score")

        # ── 9. Risk gate ──────────────────────────────────────────────────
        try:
            from src.risk.circuit_breaker import CircuitBreaker
            cb = CircuitBreaker()
            if not cb.check_gate(symbol):
                result.signal = None
                result.errors.append("circuit_breaker_blocked")
                return result
        except RuntimeError:
            result.signal = None
            return result
        except Exception:
            pass  # Non-fatal

        # Reject zero-lot signals
        if (result.signal.get("size", 0.0) or 0.0) <= 0.0:
            LOG.debug(f"[pipeline_v2] {symbol}: sizer returned 0 lot → skip")
            result.signal = None
            return result

        # ── 9.5 Execution Cost Filter (CRITICAL: 14x for mathematical edge) ──────────────────────────
        # Skip trades where expected move < 14x total execution cost
        entry_px = float(result.signal.get("price", 0.0))
        sl_px = float(result.signal.get("sl", 0.0)) if result.signal.get("sl") else None
        tp_px = float(result.signal.get("tp", 0.0)) if result.signal.get("tp") else None
        
        if entry_px > 0 and sl_px is not None and tp_px is not None:
            # Calculate expected move (distance to TP)
            expected_move = abs(tp_px - entry_px)
            
            # Get spread from slippage model
            sym = str(result.signal.get("symbol", "UNKNOWN")).upper()
            pip_val = 0.01 if "JPY" in sym or "XAG" in sym else 0.0001
            
            from src.broker.paper_executor import SLIPPAGE_MODEL
            spread_range = SLIPPAGE_MODEL.get(sym, SLIPPAGE_MODEL.get("default", (1.0, 2.0)))
            avg_spread_pips = (spread_range[0] + spread_range[1]) / 2
            spread_cost = avg_spread_pips * pip_val
            
            # Expected slippage (conservative: 1x spread)
            expected_slippage = spread_cost
            total_cost = spread_cost + expected_slippage
            
            # MATHEMATICAL EDGE FILTER: expected move must be >= 14x total cost
            # This ensures friction costs are negligible relative to profit potential
            if expected_move < total_cost * 14:
                LOG.debug(f"[pipeline_v2] {sym}: Skipping - expected_move ({expected_move:.5f}) < 14x cost ({total_cost * 14:.5f})")
                result.signal = None
                result.errors.append(f"execution_cost_filter:expected_move={expected_move:.5f},cost_14x={total_cost*14:.5f}")
                return result
            
            # Log cost ratio for trade
            cost_ratio = expected_move / total_cost if total_cost > 0 else 0
            LOG.info(f"[pipeline_v2] {sym}: Trade passed cost filter - expected_move={expected_move:.5f}, cost_ratio={cost_ratio:.1f}x")
        
        # ── 10. Paper execute ─────────────────────────────────────────────
        try:
            # Propagate candle timestamp for replay correctness and latency metrics.
            bar_time = getattr(bar, "time", None) or getattr(bar, "timestamp", None)
            result.signal.setdefault("metadata", {})
            if isinstance(result.signal.get("metadata"), dict) and bar_time is not None:
                try:
                    result.signal["metadata"]["bar_time"] = (
                        bar_time.isoformat() if hasattr(bar_time, "isoformat") else str(bar_time)
                    )
                except Exception:
                    result.signal["metadata"]["bar_time"] = str(bar_time)

            fill = executor.execute(result.signal)
            result.fill = fill
            try:
                from src.ml.ai_brain import safe_record_entry
                safe_record_entry(
                    result.signal,
                    fill,
                    context={
                        "bar_index": self._bar_idx,
                        "feature_count": result.feature_count,
                        "psi_max": result.psi_max,
                    },
                )
            except Exception:
                pass
        except Exception as e:
            result.errors.append(f"execute_error:{e}")
            LOG.warning(f"[pipeline_v2] execute error ({symbol}): {e}")

        return result

    @property
    def feature_names(self) -> List[str]:
        return list(self._feature_names)

    @property
    def feature_count(self) -> int:
        return len(self._feature_names)
