"""
SCOPUS Trading Pipeline - 13-Stage Decision System

This module implements the complete 13-stage SCOPUS trading pipeline:
1. Market Data Ingestion
2. Feature Reactor (technical indicators)
3. Primary ML Brain (confidence calculation)
4. RL Brain (FinRL fallback in grey zone)
5. Volatility Brain (regime detection)
6. Sentiment Brain (placeholder)
7. Risk Brain (SL/TP, position sizing)
8. Meta-Gating Brain (filter conditions)
9. Portfolio Brain (correlation, exposure)
10. Risk Gates (max drawdown, position limits)
11. Throttle Gates (frequency limits)
12. Execution Reflex Engine (final decision)
13. Logging & Monitoring

Extracted from tools/executor.py for use in both backtest and live trading.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import Counter, deque
import logging
import os
import traceback
import json
import math
import numpy as np
import pandas as pd

try:
    import joblib
except ImportError:
    joblib = None

from pathlib import Path
from src.market_data import MarketDataStore, Symbol, Timeframe
from src.backtest.feature_reactor_v1 import SafeFeatureReactor
from src.backtest.ml_brain_v1 import MLBrainV1
from src.backtest.regime_engine_v1 import RegimeEngineV1
from src.backtest.risk_brain_v1 import RiskBrainV1
from src.backtest.meta_gating_v1 import MetaGatingV1

logger = logging.getLogger(__name__)


# ==================== Configuration ====================

@dataclass
class PipelineConfigV2:
    """Configuration for the trading pipeline"""
    def __post_init__(self):
        print(f"[PipelineConfigV2] Initialized with meta_gating={self.enable_meta_gating}")
    
    # Model paths
    primary_model_path: Optional[str] = None
    finrl_policies_path: Optional[str] = None
    
    # Confidence thresholds
    primary_high_thresh: float = 0.70  # Execute primary (Stage 3)
    primary_block_thresh: float = 0.40  # Block trade (Stage 3)
    finrl_min_thresh: float = 0.65     # RL activation (Stage 4)
    
    # Risk parameters (Stage 7)
    stop_atr_mult: float = 1.5
    takeprofit_rr: float = 1.5
    sl_pct: float = 0.01
    tp_pct: float = 0.02
    risk_per_trade: float = 0.01
    
    # Feature flags
    enable_meta_gating: bool = True
    enable_portfolio_brain: bool = False
    enable_rl_fallback: bool = True
    enable_weapon_system: bool = True
    enable_micro_strategies: bool = True
    enable_sentiment: bool = False
    enable_slicer: bool = False  # Added for compatibility
    
    # Logging
    verbose: bool = False
    log_features: bool = True

    # Gating & Throttling (MATHEMATICAL EDGE: 50 bars between trades, MR-only)
    min_confidence_normal: float = 0.60
    min_confidence_uncertain: float = 0.75
    max_trades_per_day_per_symbol: int = 5  # Reduced for MR-only mode
    min_bars_between_trades: int = 50  # 50 bars cooldown per symbol
    max_open_positions: int = 2  # Reduced for focused MR trades
    loss_streak_cooldown_trades: int = 3
    loss_streak_cooldown_bars: int = 50  # Consistent with main cooldown
    calibrated_confidence_min: float = 0.55  # Higher for MR-only
    confidence_deadzone_low: float = 0.55
    confidence_deadzone_high: float = 0.60
    edge_score_min: float = 0.70  # Higher for MR edge
    volatility_low_pctile: float = 0.55  # MR requires elevated vol
    volatility_high_pctile: float = 0.90
    cooldown_bars_after_loss: int = 5
    cooldown_bars_after_win: int = 0
    trade_cluster_bars: int = 50  # 50 bars
    tokyo_min_confidence: float = 0.65
    tier1_confidence_min: float = 0.65  # Higher for MR-only
    tier2_confidence_min: float = 0.60
    tier3_confidence_floor: float = 0.55


@dataclass
class Decision:
    """Trading decision output from pipeline"""
    action: str  # 'open', 'hold'
    side: Optional[str] = None  # 'buy', 'sell'
    entry_price: Optional[float] = None
    size: Optional[float] = None
    sl_price: Optional[float] = None
    tp_price: Optional[float] = None
    decision_source: str = 'UNKNOWN'  # 'PRIMARY', 'RL_FALLBACK', 'HEURISTIC'
    regime: str = 'UNKNOWN'  # 'TREND', 'RANGE', 'UNCERTAIN'
    confidence: float = 0.0
    features: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ==================== Helper Functions ====================

def _safe_float(x, default=0.0) -> float:
    """Safely convert to float"""
    try:
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return default
        return v
    except Exception:
        return default


def _compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stage 2: Feature Reactor
    Compute technical indicators from OHLCV data
    """
    df = df.copy()
    
    # Price-based features
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # Moving averages
    for period in [5, 10, 20, 50]:
        df[f'sma_{period}'] = df['close'].rolling(period).mean()
        df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema_12 = df['close'].ewm(span=12).mean()
    ema_26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + (2 * bb_std)
    df['bb_lower'] = df['bb_middle'] - (2 * bb_std)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    
    # ATR (Average True Range)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    df['atr14'] = df['atr']  # Alias
    
    # Volatility
    df['volatility'] = df['returns'].rolling(20).std()
    
    # Volume features (if available)
    if 'volume' in df.columns:
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
    
    return df


def _detect_regime(features: pd.Series) -> str:
    """
    Stage 5: Volatility Brain
    Detect market regime based on features
    """
    try:
        # Simple regime detection based on volatility and trend
        volatility = _safe_float(features.get('volatility', 0))
        atr = _safe_float(features.get('atr', 0))
        close = _safe_float(features.get('close', 0))
        sma_20 = _safe_float(features.get('sma_20', 0))
        sma_50 = _safe_float(features.get('sma_50', 0))
        
        # High volatility = uncertain
        if volatility > 0.02 or (atr / close if close > 0 else 0) > 0.015:
            return 'UNCERTAIN'
        
        # Trend detection
        if sma_20 > 0 and sma_50 > 0:
            if sma_20 > sma_50 * 1.01:  # Uptrend
                return 'TREND'
            elif sma_20 < sma_50 * 0.99:  # Downtrend
                return 'TREND'
        
        # Default to range
        return 'RANGE'
    except Exception:
        return 'UNKNOWN'


# ==================== Main Pipeline Class ====================

class TradingPipeline:
    """
    13-Stage SCOPUS Trading Pipeline with Adaptive Behavior
    """
    
    def __init__(self, config: PipelineConfigV2):
        self.config = config
        self.primary_model = None
        self.finrl_adapter = None
        
        # Initialize MarketDataStore with multiple data roots
        self.store = MarketDataStore(data_roots=[
            Path("data/raw/forex_kaggle_multiTF"),
            Path("data/raw/forex_backup_2020_2025"),  # D1/Daily data
        ])
        
        # Initialize Components
        self.feature_reactor = SafeFeatureReactor(store=self.store)
        self.regime_engine = RegimeEngineV1()
        self.risk_brain = RiskBrainV1()
        self.meta_gating = MetaGatingV1()
        
        # Initialize AdaptiveState (Core Adaptive Memory)
        from src.backtest.adaptive_state import AdaptiveState
        self.adaptive_state = AdaptiveState({
            'initial_capital': config.risk_per_trade * 100000,  # Estimate from risk
            'ewma_alpha': 0.02,
            'rolling_window': 20
        })
        
        # Initialize Context Brain and Adaptive Filters for LEARNING mode
        from src.backtest.context_brain import ContextBrain
        from src.backtest.adaptive_filters import AdaptiveFilterEngine
        self.context_brain = ContextBrain()
        self.filter_engine = AdaptiveFilterEngine()
        
        # Initialize MARK-2 Intelligence (Memory, Ego, Regime Strength)
        from src.backtest.mark2_intelligence import MARK2Intelligence
        self.mark2 = MARK2Intelligence()
        
        # Initialize MARK-3 EDGE_SCORE (Profit Acceleration)
        from src.backtest.edge_score import EdgeScoreModule, EdgeScoreSafeguards
        self.edge_score = EdgeScoreModule()
        self.edge_safeguards = EdgeScoreSafeguards()
        
        # Initialize Session Gate (trading hours control)
        from src.backtest.session_gate import SessionGate
        self.session_gate = SessionGate()

        # Initialize Weapon System Router (MARK-3.3)
        self.weapon_router = None
        self.weapon_logger = None
        self.strategy_router = None
        self.strategy_router_log_path = None
        self.decision_log_path = None
        self._router_debug_samples = 0
        self._skip_reason_counts: Counter = Counter()
        self._edge_histogram: Counter = Counter()
        self._accepted_trade_count: int = 0
        self._decision_count: int = 0
        self._decision_day_counts: Counter = Counter()
        self._accepted_trade_day_counts: Counter = Counter()
        self._hold_rejection_count: int = 0
        self._cooldown_block_count: int = 0
        self._low_volatility_block_count: int = 0
        self._signal_quality_block_count: int = 0
        self._pending_signals: Dict[str, Dict[str, Any]] = {}
        # === EXECUTION INTELLIGENCE STATE ===
        self._strategy_performance: Dict[str, List[bool]] = {}  # last N trades per strategy
        self._candles_since_last_accepted: int = 0
        self._flow_state: str = 'normal'  # 'normal' | 'relaxed' | 'tight'
        self._last_accepted_signals: Dict[str, Dict[str, Any]] = {}  # duplicate tracking
        self._current_bar_idx: int = 0
        self._calibration_params = self._load_confidence_calibration()
        self.quality_scorer = None
        try:
            from src.backtest.quality_scorer import QualityScorer
            self.quality_scorer = QualityScorer()
        except Exception as e:
            if self.config.verbose:
                print(f"[Pipeline] Quality scorer unavailable: {e}")
        if getattr(self.config, 'enable_weapon_system', False):
            try:
                from src.pipeline.strategy_router import StrategyRouter as AdaptiveStrategyRouter
                self.strategy_router = AdaptiveStrategyRouter()
                router_log_dir = Path("logs/strategy_router")
                router_log_dir.mkdir(parents=True, exist_ok=True)
                decision_log_dir = Path("logs/decision_engine")
                decision_log_dir.mkdir(parents=True, exist_ok=True)
                session_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                self.strategy_router_log_path = router_log_dir / f"router_decisions_{session_id}.jsonl"
                self.decision_log_path = decision_log_dir / f"decision_events_{session_id}.jsonl"

                from src.strategy import StrategyRouter
                from src.strategy.logging import WeaponDecisionLogger
                self.weapon_router = StrategyRouter(
                    enable_micro=getattr(self.config, 'enable_micro_strategies', False)
                )
                self.weapon_logger = WeaponDecisionLogger()
                print("[Pipeline] Weapon System Router initialized")
            except Exception as e:
                print(f"[Pipeline] Failed to initialize Weapon System: {e}")

        # Initialize Experience Reasoning Module (ERM) for adaptive learning
        self.erm = None
        if getattr(self.config, 'enable_erm', True):  # Enabled by default
            try:
                from src.reasoning import ExperienceReasoningModule
                self.erm = ExperienceReasoningModule()
                print("[Pipeline] Experience Reasoning Module initialized")
            except Exception as e:
                print(f"[Pipeline] Failed to initialize ERM: {e}")
        
        # Initialize ML Brain
        model_path = Path("models/xgb_primary_mtf.joblib")
        feature_names_path = Path("models/feature_names_mtf.txt")
        if model_path.exists() and feature_names_path.exists():
            try:
                self.ml_brain = MLBrainV1(
                    model_path=model_path,
                    feature_names_path=feature_names_path,
                    feature_reactor=self.feature_reactor
                )
                if self.config.verbose:
                    print(f"[Pipeline] Loaded ML Brain V1 from {model_path}")
            except Exception as e:
                print(f"[Pipeline] Failed to load ML Brain V1: {e}")
                self.ml_brain = None
        else:
            if self.config.verbose:
                print("WARNING: ML Brain model not found, using heuristic fallback")
            self.ml_brain = None
            
        # Load models (Legacy / Secondary)
        self._load_models()
        
        # Load gating config
        self._load_gating_config()

        # Initialize confidence history for dynamic calibration
        self._raw_conf_history = deque(maxlen=500)
    
    def _load_gating_config(self):
        """Load trade gating config from yaml"""
        config_path = "config/trade_gating.yaml"
        if os.path.exists(config_path):
            try:
                import yaml
                with open(config_path, 'r') as f:
                    gating_config = yaml.safe_load(f)
                
                if gating_config:
                    self.config.min_confidence_normal = gating_config.get('min_confidence_normal', self.config.min_confidence_normal)
                    self.config.min_confidence_uncertain = gating_config.get('min_confidence_uncertain', self.config.min_confidence_uncertain)
                    self.config.max_trades_per_day_per_symbol = 10
                    self.config.min_bars_between_trades = gating_config.get('min_bars_between_trades', self.config.min_bars_between_trades)
                    self.config.max_open_positions = gating_config.get('max_open_positions', self.config.max_open_positions)
                    self.config.loss_streak_cooldown_trades = gating_config.get('loss_streak_cooldown_trades', self.config.loss_streak_cooldown_trades)
                    self.config.loss_streak_cooldown_bars = gating_config.get('loss_streak_cooldown_bars', self.config.loss_streak_cooldown_bars)
                    self.config.calibrated_confidence_min = gating_config.get('calibrated_confidence_min', self.config.calibrated_confidence_min)
                    self.config.confidence_deadzone_low = gating_config.get('confidence_deadzone_low', self.config.confidence_deadzone_low)
                    self.config.confidence_deadzone_high = gating_config.get('confidence_deadzone_high', self.config.confidence_deadzone_high)
                    self.config.edge_score_min = gating_config.get('edge_score_min', self.config.edge_score_min)
                    self.config.tier1_confidence_min = gating_config.get('tier1_confidence_min', self.config.tier1_confidence_min)
                    self.config.tier2_confidence_min = gating_config.get('tier2_confidence_min', self.config.tier2_confidence_min)
                    self.config.tier3_confidence_floor = gating_config.get('tier3_confidence_floor', self.config.tier3_confidence_floor)
                    
                    if self.config.verbose:
                        print(f"[Pipeline] Loaded gating config: {gating_config}")
            except Exception as e:
                print(f"[Pipeline] Failed to load gating config: {e}")
    
    def _load_models(self):
        """Load primary and FinRL models"""
        # Load primary model
        if self.config.primary_model_path and os.path.exists(self.config.primary_model_path):
            try:
                if joblib:
                    self.primary_model = joblib.load(self.config.primary_model_path)
                    if self.config.verbose:
                        print(f"[Pipeline] Loaded primary model from {self.config.primary_model_path}")
            except Exception as e:
                if self.config.verbose:
                    print(f"[Pipeline] Failed to load primary model: {e}")
        
        # Load FinRL adapter (TODO: implement FinRL loading)
        if self.config.finrl_policies_path and os.path.exists(self.config.finrl_policies_path):
            if self.config.verbose:
                print(f"[Pipeline] FinRL policies path: {self.config.finrl_policies_path}")
            # TODO: Load FinRL adapter

    def _load_confidence_calibration(self) -> Dict[str, Any]:
        """Load optional Platt scaling parameters for ML confidence calibration."""
        default_params = {"method": "platt", "a": 1.0, "b": 0.0}
        calibration_path = Path("models/ml_confidence_calibration.json")
        if not calibration_path.exists():
            return default_params
        try:
            with open(calibration_path, 'r', encoding='utf-8') as f:
                payload = json.load(f)
            if payload.get("method") == "platt":
                return {
                    "method": "platt",
                    "a": float(payload.get("a", 1.0)),
                    "b": float(payload.get("b", 0.0)),
                }
        except Exception as exc:
            logger.warning("Failed to load calibration parameters: %s", exc)
        return default_params

    def _calibrate_confidence(self, raw_confidence: float) -> float:
        """Apply dynamic tanh scaling after ML prediction."""
        clipped = min(max(float(raw_confidence), 1e-6), 1.0 - 1e-6)
        
        # Update history
        self._raw_conf_history.append(clipped)
        
        # Calculate dynamic center (median of last 500)
        if len(self._raw_conf_history) > 0:
            import statistics
            center = statistics.median(self._raw_conf_history)
        else:
            center = 0.55
            
        calibrated = 0.5 + 0.5 * math.tanh((clipped - center) * 6.0)
        return float(min(max(calibrated, 0.0), 1.0))

    def _edge_bucket(self, edge_score: float) -> str:
        lower = int(max(0, min(9, edge_score * 10)))
        upper = min(10, lower + 1)
        return f"{lower / 10:.1f}-{upper / 10:.1f}"

    def _record_edge_histogram(self, edge_score: float) -> None:
        self._edge_histogram[self._edge_bucket(edge_score)] += 1

    def _decision_payload(
        self,
        candle: pd.Series,
        context: Dict,
        regime: str,
        strategy: str,
        raw_confidence: Optional[float],
        calibrated_confidence: Optional[float],
        edge_score: Optional[float],
        skip_reason: Optional[str],
        cooldown_active: bool,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        timestamp = candle.get('timestamp', None)
        payload = {
            "timestamp": timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
            "symbol": context.get('symbol', 'UNKNOWN'),
            "ml_conf_raw": None if raw_confidence is None else round(float(raw_confidence), 6),
            "ml_conf_calibrated": None if calibrated_confidence is None else round(float(calibrated_confidence), 6),
            "edge_score": None if edge_score is None else round(float(edge_score), 6),
            "regime": regime,
            "strategy": strategy,
            "skip_reason": skip_reason,
            "cooldown_active": bool(cooldown_active),
            "danger_reason": context.get('danger_reason'),
            "danger_duration_bars": context.get('danger_duration_bars'),
        }
        if extra:
            payload.update(extra)
        return payload

    def _log_decision_event(self, payload: Dict[str, Any]) -> None:
        self._decision_count += 1
        timestamp = payload.get("timestamp")
        day_key = str(timestamp)[:10] if timestamp else "UNKNOWN"
        self._decision_day_counts[day_key] += 1
        if payload.get("skip_reason"):
            self._skip_reason_counts[payload["skip_reason"]] += 1
        if payload.get("edge_score") is not None:
            self._record_edge_histogram(float(payload["edge_score"]))
        if self.decision_log_path is None:
            return
        try:
            with open(self.decision_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(payload) + "\n")
        except Exception as exc:
            if self.config.verbose:
                print(f"[DECISION] Failed to write decision log: {exc}")

    def _reject_decision(
        self,
        candle: pd.Series,
        context: Dict,
        regime: str,
        strategy: str,
        raw_confidence: Optional[float],
        calibrated_confidence: Optional[float],
        edge_score: Optional[float],
        skip_reason: str,
        cooldown_active: bool = False,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        context['rejected_reason'] = skip_reason
        self._log_decision_event(
            self._decision_payload(
                candle=candle,
                context=context,
                regime=regime,
                strategy=strategy,
                raw_confidence=raw_confidence,
                calibrated_confidence=calibrated_confidence,
                edge_score=edge_score,
                skip_reason=skip_reason,
                cooldown_active=cooldown_active,
                extra=extra,
            )
        )

    def get_filter_diagnostics(self) -> Dict[str, Any]:
        total_skips = sum(self._skip_reason_counts.values())
        block_percentages = {}
        if total_skips > 0:
            for reason, count in self._skip_reason_counts.items():
                block_percentages[reason] = round((count / total_skips) * 100.0, 2)
        session_rejections = sum(
            count for reason, count in self._skip_reason_counts.items()
            if str(reason).startswith("Session:") or str(reason) == "tokyo_low_confidence"
        )
        debug_counters = {
            "rejected_by_boll_z": int(self._skip_reason_counts.get("boll_z_filter", 0)),
            "rejected_by_confidence": int(
                self._skip_reason_counts.get("confidence_deadzone", 0)
                + self._skip_reason_counts.get("calibrated_confidence_below_min", 0)
                + self._skip_reason_counts.get("primary_confidence_below_block", 0)
            ),
            "rejected_by_session": int(session_rejections),
            "rejected_by_soft_filter": int(self._skip_reason_counts.get("soft_filter_block", 0)),
        }
        active_days = max(1, len(self._decision_day_counts))
        trades_per_day = self._accepted_trade_count / active_days
        avg_signals_per_day = self._decision_count / active_days
        idle_pct = round(
            ((self._decision_count - self._accepted_trade_count) / self._decision_count) * 100.0,
            2,
        ) if self._decision_count > 0 else 0.0
        return {
            "skip_reason_counts": dict(self._skip_reason_counts),
            "skip_reason_percentages": block_percentages,
            "edge_score_histogram": dict(sorted(self._edge_histogram.items())),
            "accepted_trades": self._accepted_trade_count,
            "trades_per_day": round(trades_per_day, 4),
            "avg_signals_per_day": round(avg_signals_per_day, 4),
            "idle_pct": idle_pct,
            "debug_counters": debug_counters,
            "hold_rejections": self._hold_rejection_count,
            "cooldown_blocks": self._cooldown_block_count,
            "low_volatility_blocks": self._low_volatility_block_count,
            "signal_quality_blocks": self._signal_quality_block_count,
        }

    def _passes_trade_timing_filters(self, context: Dict) -> Tuple[bool, bool, Optional[str]]:
        bars_since_last_exit = int(context.get('bars_since_last_exit', 9999))
        last_trade_result = str(context.get('last_trade_result', '')).upper()
        last_trade_r_multiple = float(context.get('last_trade_r_multiple', 0.0))
        bars_since_last_trade = int(context.get('bars_since_last_trade', 9999))

        cooldown_active = False
        cooldown_reason = None

        if (
            last_trade_result == 'LOSS'
            and last_trade_r_multiple < -1.0
            and bars_since_last_exit < self.config.cooldown_bars_after_loss
        ):
            cooldown_active = True
            cooldown_reason = 'cooldown_after_loss'
        elif last_trade_result == 'WIN' and self.config.cooldown_bars_after_win > 0 and bars_since_last_exit < self.config.cooldown_bars_after_win:
            cooldown_active = True
            cooldown_reason = 'cooldown_after_win'

        if bars_since_last_trade < self.config.trade_cluster_bars:
            return False, cooldown_active, 'trade_cluster_block'

        if cooldown_active:
            self._cooldown_block_count += 1
            return False, True, cooldown_reason

        return True, False, None

    def _soft_filter_score(self, confidence: float, edge_score: float, regime: str, strategy: str) -> int:
        score = 0
        if confidence > 0.60:
            score += 1
        if edge_score > 0.62:
            score += 1
        normalized_regime = str(regime).strip().upper()
        normalized_strategy = str(strategy).strip().upper()
        if (
            (normalized_regime == 'RANGE' and normalized_strategy == 'MEAN_REVERSION')
            or (normalized_regime == 'TREND' and normalized_strategy == 'TREND_PULLBACK')
            or (normalized_regime == 'NEUTRAL' and normalized_strategy == 'MEAN_REVERSION')
        ):
            score += 1
        return score

    # ==================== EXECUTION INTELLIGENCE SYSTEMS ====================

    def record_strategy_result(self, strategy: str, is_win: bool) -> None:
        """Track last N=5 trades per strategy for adaptive sizing (System 2)."""
        key = str(strategy).strip().upper()
        if key not in self._strategy_performance:
            self._strategy_performance[key] = []
        self._strategy_performance[key].append(is_win)
        # Keep only last 5
        if len(self._strategy_performance[key]) > 5:
            self._strategy_performance[key] = self._strategy_performance[key][-5:]

    def _get_performance_size_modifier(self, strategy: str) -> Tuple[float, bool]:
        """Returns (modifier, is_active) based on last 5 trades for this strategy.
        winrate < 40% -> 0.7x, winrate > 60% -> 1.2x, else 1.0x.
        """
        key = str(strategy).strip().upper()
        history = self._strategy_performance.get(key, [])
        if len(history) < 3:  # Need minimum 3 trades to activate
            return 1.0, False
        winrate = sum(1 for w in history if w) / len(history)
        if winrate < 0.4:
            return 0.7, True
        elif winrate > 0.6:
            return 1.2, True
        return 1.0, False

    def _compute_flow_state(self, trades_today: int) -> Tuple[str, float, float]:
        """Compute flow state and threshold adjustments.
        Returns (state, confidence_adj, edge_adj).
        - 50+ candles dry -> relaxed: conf -0.03, edge -0.03
        - >3 trades/day -> tight: conf +0.02, edge +0.02
        - else -> normal: 0, 0
        """
        if self._candles_since_last_accepted >= 50:
            return 'relaxed', -0.03, -0.03
        elif trades_today > 3:
            return 'tight', 0.02, 0.02
        return 'normal', 0.0, 0.0

    def _is_duplicate_signal(self, symbol: str, side: str, strategy: str) -> bool:
        """Reject if last trade was same direction + same strategy within last 5 candles."""
        last = self._last_accepted_signals.get(symbol)
        if last is None:
            return False
        bars_ago = self._current_bar_idx - last.get('bar_idx', 0)
        if bars_ago <= 5 and last.get('side') == side and last.get('strategy') == strategy:
            return True
        return False

    def _compute_signal_quality(self, boll_z: Optional[float], edge_score: float, ml_confidence: float) -> float:
        """Compute composite signal quality score (balanced, capped).
        signal_quality = min(abs(boll_z), 2.0) * 0.3 + edge_score * 0.4 + ml_confidence * 0.3
        Reject if < 0.60. Uses neutral default of 1.5 for boll_z when not available.
        """
        bz = min(abs(float(boll_z)), 2.0) if boll_z is not None else 1.5  # capped + neutral default
        return bz * 0.3 + float(edge_score) * 0.4 + float(ml_confidence) * 0.3

    def _grade_signal_quality(self, features: pd.Series, signal: str, regime: str, confidence: float) -> Dict[str, Any]:
        """Score the setup and return a stable letter-grade payload for tier gating."""
        default_payload = {
            'grade': 'B',
            'score': None,
            'recommendation': 'FULL',
        }
        if self.quality_scorer is None or signal not in {'BUY', 'SELL'}:
            return default_payload

        try:
            feature_dict = features.to_dict() if hasattr(features, 'to_dict') else dict(features)
            result = self.quality_scorer.calculate(
                features=feature_dict,
                signal=signal,
                regime=str(regime or 'UNKNOWN').strip().upper(),
                confidence=confidence,
            )
            payload = result.to_dict()
            payload['score'] = result.total_score
            return payload
        except Exception as e:
            if self.config.verbose:
                print(f"[Pipeline] Quality scoring failed: {e}")
            return default_payload

    def _apply_signal_tier_gate(
        self,
        confidence: float,
        quality_grade: str,
        size: float,
    ) -> Tuple[Optional[float], str, float]:
        """
        Three-tier signal gating (v2 — tightened 2026-05-01):
        - Tier-1 (strong):     conf >= 0.60 AND grade in {A, B} → full size (1.2x if conf >= 0.65)
        - Tier-2 (borderline): 0.55 <= conf < 0.60 AND grade == C → half size
        - Tier-3 (reject):     conf < 0.55 OR grade in {D, F} → skip
        """
        grade = str(quality_grade or 'F').strip().upper()

        # Tier-3: reject low-confidence OR poor quality
        if confidence < self.config.tier3_confidence_floor or grade in {'D', 'F'}:
            return None, 'reject', 0.0

        # Tier-1: high-confidence + good quality → full size
        if confidence >= self.config.tier1_confidence_min and grade in {'A', 'B'}:
            # High-confidence boost: 1.2x for very strong signals
            if confidence >= 0.65:
                return size * 1.2, 'strong', 1.2
            return size, 'strong', 1.0

        # Tier-2: borderline — BOTH conditions required (AND, not OR)
        if confidence >= self.config.tier2_confidence_min and confidence < self.config.tier1_confidence_min and grade == 'C':
            return size * 0.5, 'borderline', 0.5

        # Everything else rejected (e.g. conf in [0.55, 0.60) with grade != C)
        return None, 'reject', 0.0

    @staticmethod
    def _apply_delayed_entry_prices(decision: Decision, entry_price: float) -> Decision:
        """Preserve original stop/target distances after delayed confirmation."""
        original_entry = float(decision.entry_price or entry_price)
        stop_distance = abs(original_entry - float(decision.sl_price or original_entry))
        target_distance = abs(float(decision.tp_price or original_entry) - original_entry)
        side = str(decision.side or '').lower()

        decision.entry_price = float(entry_price)
        if side == 'buy':
            decision.sl_price = decision.entry_price - stop_distance
            decision.tp_price = decision.entry_price + target_distance
        elif side == 'sell':
            decision.sl_price = decision.entry_price + stop_distance
            decision.tp_price = decision.entry_price - target_distance
        return decision

    def _stage_pending_signal(self, candle: pd.Series, context: Dict, decision: Decision) -> None:
        """Queue a decision so execution can only happen on the next candle."""
        symbol = context.get('symbol', 'UNKNOWN')
        timestamp = candle.get('timestamp', None)
        signal_close = _safe_float(candle.get('close', decision.entry_price), decision.entry_price or 0.0)

        context['entry_mode'] = 'delayed_confirmation'
        decision.metadata = dict(decision.metadata or {})
        decision.metadata['entry_mode'] = 'delayed_confirmation'
        decision.metadata['signal_timestamp'] = timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp)
        decision.metadata['signal_close'] = signal_close

        self._pending_signals[symbol] = {
            'decision': decision,
            'timestamp': timestamp,
            'signal_close': signal_close,
            'direction': decision.side,
            'strategy': decision.metadata.get('strategy', 'UNKNOWN'),
        }

        self._log_decision_event(
            self._decision_payload(
                candle=candle,
                context=context,
                regime=decision.regime,
                strategy=decision.metadata.get('strategy', 'UNKNOWN'),
                raw_confidence=decision.metadata.get('ml_conf_raw'),
                calibrated_confidence=decision.confidence,
                edge_score=decision.metadata.get('edge_score'),
                skip_reason=None,
                cooldown_active=bool(decision.metadata.get('cooldown_active', False)),
                extra={
                    'decision': 'PENDING_CONFIRMATION',
                    'entry_mode': 'delayed_confirmation',
                    'signal_timestamp': decision.metadata.get('signal_timestamp'),
                },
            )
        )

    def _process_pending_signal(self, candle: pd.Series, context: Dict) -> Optional[Decision]:
        """Confirm the queued signal on the next candle or drop it permanently."""
        symbol = context.get('symbol', 'UNKNOWN')
        pending = self._pending_signals.pop(symbol, None)
        if pending is None:
            return None

        decision = pending['decision']
        current_close = _safe_float(candle.get('close', decision.entry_price), decision.entry_price or 0.0)
        signal_close = _safe_float(pending.get('signal_close', decision.entry_price), decision.entry_price or 0.0)
        side = str(decision.side or '').lower()
        confirmed = (side == 'buy' and current_close > signal_close) or (side == 'sell' and current_close < signal_close)

        if not confirmed:
            self._reject_decision(
                candle,
                context,
                decision.regime,
                decision.metadata.get('strategy', 'UNKNOWN'),
                decision.metadata.get('ml_conf_raw'),
                decision.confidence,
                decision.metadata.get('edge_score'),
                'confirmation_failed',
                extra={
                    'entry_mode': 'delayed_confirmation',
                    'signal_timestamp': decision.metadata.get('signal_timestamp'),
                    'signal_close': signal_close,
                    'confirmation_close': current_close,
                },
            )
            return None

        context['entry_mode'] = 'delayed_confirmation'
        decision.metadata = dict(decision.metadata or {})
        decision.metadata['entry_mode'] = 'delayed_confirmation'
        confirm_ts = candle.get('timestamp', None)
        decision.metadata['confirmation_timestamp'] = confirm_ts.isoformat() if hasattr(confirm_ts, 'isoformat') else str(confirm_ts)
        decision = self._apply_delayed_entry_prices(decision, current_close)

        self._log_decision_event(
            self._decision_payload(
                candle=candle,
                context=context,
                regime=decision.regime,
                strategy=decision.metadata.get('strategy', 'UNKNOWN'),
                raw_confidence=decision.metadata.get('ml_conf_raw'),
                calibrated_confidence=decision.confidence,
                edge_score=decision.metadata.get('edge_score'),
                skip_reason=None,
                cooldown_active=bool(decision.metadata.get('cooldown_active', False)),
                extra={
                    'decision': 'EXECUTE',
                    'entry_mode': 'delayed_confirmation',
                    'signal_timestamp': decision.metadata.get('signal_timestamp'),
                },
            )
        )
        # === ENTRY VALIDATION: Assert delayed confirmation (System 1) ===
        entry_mode = decision.metadata.get('entry_mode', 'UNKNOWN')
        if entry_mode != 'delayed_confirmation':
            logger.critical(f"[CRITICAL] Trade executed WITHOUT delayed confirmation! entry_mode={entry_mode}")
            print(f"[CRITICAL ERROR] Trade executed without delay! entry_mode={entry_mode}")

        self._accepted_trade_count += 1
        self._candles_since_last_accepted = 0  # Reset flow counter
        # Record for duplicate signal detection (System 4)
        accepted_strategy = decision.metadata.get('strategy', 'UNKNOWN')
        self._last_accepted_signals[symbol] = {
            'side': str(decision.side or '').lower(),
            'strategy': str(accepted_strategy).strip().upper(),
            'bar_idx': self._current_bar_idx,
        }
        day_key = confirm_ts.date().isoformat() if hasattr(confirm_ts, 'date') else str(confirm_ts)[:10]
        self._accepted_trade_day_counts[day_key] += 1
        return decision
    
    def decide(self, candle: pd.Series, context: Dict) -> Optional[Decision]:
        """
        Main decision pipeline - executes all 13 stages
        
        Args:
            candle: Current price candle (OHLCV)
            context: Trading context (symbol, timeframe, history, positions, etc.)
        
        Returns:
            Decision object or None if no trade
        """
        symbol = context.get('symbol', 'UNKNOWN')
        self._candles_since_last_accepted += 1
        self._current_bar_idx += 1
        pending_decision = self._process_pending_signal(candle, context)
        if pending_decision is not None:
            return pending_decision
        
        # Stage 1: Market Data Ingestion (already done - candle + context)
        
        # Stage 2: Feature Reactor (MTF)
        # Use 'UNKNOWN' regime for initial extraction (or last known)
        last_regime = context.get('regime', 'UNKNOWN')
        features = self.feature_reactor.extract_mtf(candle, context, last_regime)
        
        if features.empty:
            if self.config.verbose:
                print(f"[Pipeline] {symbol}: Empty features")
            return None
        
        # Stage 3: Primary ML Brain
        if self.ml_brain:
            # Pass features to avoid re-extraction
            primary_pred, primary_conf = self.ml_brain.predict(candle, context, last_regime, features=features)
        else:
            # Fallback
            primary_conf, primary_pred = self._heuristic_decision(features)

        raw_primary_conf = float(primary_conf)
        calibrated_conf = self._calibrate_confidence(raw_primary_conf)
        context['ml_conf_raw'] = raw_primary_conf
        context['ml_conf_calibrated'] = calibrated_conf
        print(f"[TRACE] {symbol}: ML pred={primary_pred}, raw_conf={raw_primary_conf:.3f}, calibrated_conf={calibrated_conf:.3f}")
        primary_conf = calibrated_conf
        normalized_side = self._normalize_trade_side(primary_pred)

        # === EXECUTION INTELLIGENCE: Flow Balancer (System 3) ===
        trades_today_count = int(context.get('trades_today', 0))
        self._flow_state, flow_conf_adj, flow_edge_adj = self._compute_flow_state(trades_today_count)
        context['flow_state'] = self._flow_state
        effective_confidence_min = self.config.calibrated_confidence_min + flow_conf_adj
        effective_edge_min = self.config.edge_score_min + flow_edge_adj
        effective_boll_z_min = 0.8 + (0.0 if self._flow_state == 'normal' else (-0.2 if self._flow_state == 'relaxed' else 0.2))
        if self._flow_state != 'normal':
            print(f"[FLOW] {symbol}: state={self._flow_state}, conf_adj={flow_conf_adj:+.3f}, edge_adj={flow_edge_adj:+.3f}, candles_dry={self._candles_since_last_accepted}")

        if str(primary_pred).strip().upper() == 'HOLD' or normalized_side is None:
            self._hold_rejection_count += 1
            context['rejected_reason'] = 'hold_rejection'
            return None

        if self.config.confidence_deadzone_low < primary_conf < self.config.confidence_deadzone_high:
            self._reject_decision(candle, context, 'UNKNOWN', 'UNROUTED', raw_primary_conf, primary_conf, None, 'confidence_deadzone')
            return None

        if primary_conf < effective_confidence_min:
            self._reject_decision(candle, context, 'UNKNOWN', 'UNROUTED', raw_primary_conf, primary_conf, None, 'calibrated_confidence_below_min',
                                  extra={'effective_min': effective_confidence_min, 'flow_state': self._flow_state})
            return None

        # Stage 4: RL Brain (grey zone fallback)
        rl_size_multiplier = 1.0
        context['rl_active'] = self.finrl_adapter is not None
        context['rl_size_multiplier'] = rl_size_multiplier
        if self.config.enable_rl_fallback:
            if self.config.primary_block_thresh < primary_conf < self.config.primary_high_thresh:
                rl_decision = self._rl_brain(features, context)
                if rl_decision:
                    self._stage_pending_signal(candle, context, rl_decision)
                    return None
        
        # Check if primary confidence is high enough
        # Get candles for mode calculation
        candles = context.get('history', [])
        n_bars = len(candles) if hasattr(candles, '__len__') else 50
        mode, sufficiency, mode_config = self.filter_engine.compute_mode(n_bars)
        
        if primary_conf < self.config.primary_block_thresh:
            # === DIAGNOSTIC: Unconditional logging ===
            can_exp = self.filter_engine.can_explore(mode_config)
            print(f"[DIAG] {symbol}: mode={mode}, pred={primary_pred}, conf={primary_conf:.3f}, can_explore={can_exp}, n_bars={n_bars}")
            
            # === EXPLORATION OVERRIDE CHECKPOINT 0 (EARLIEST) ===
            if mode == 'LEARNING' and primary_pred in ['BUY', 'SELL']:
                if can_exp:
                    print(f"[EXPLORATION] {symbol}: CHECKPOINT 0 - conf={primary_conf:.3f} < block_thresh, mode={mode}")
                    exploration_trade = self._create_exploration_trade(
                        candle, primary_pred, features, mode_config, symbol, context
                    )
                    if exploration_trade:
                        print(f"[EXPLORATION] {symbol}: Override at primary_block_thresh check!")
                        self._stage_pending_signal(candle, context, exploration_trade)
                        return None
                else:
                    print(f"[EXPLORATION] {symbol}: Cannot explore - budget exhausted")
            else:
                print(f"[DIAG] {symbol}: Exploration condition FAILED - mode={mode} (need LEARNING), pred={primary_pred} (need BUY/SELL)")
            
            self._reject_decision(
                candle,
                context,
                'UNKNOWN',
                'UNROUTED',
                raw_primary_conf,
                primary_conf,
                None,
                'primary_confidence_below_block',
            )
            if self.config.verbose:
                print(f"[Pipeline] {symbol}: Blocked - primary_conf={primary_conf:.3f} too low (mode={mode})")
            return None
        
        # === ADAPTIVE: Update AdaptiveState with candle ===
        self.adaptive_state.update_candle(candle, features)
        
        # Stage 5: Regime Engine (with AdaptiveState)
        # Map MTF features to Regime Engine expected format (M5_sma_20 -> sma_20)
        regime_features = self._map_features_for_regime(features)
        regime, regime_conf = self.regime_engine.detect(regime_features, self.adaptive_state)
        regime_context = self.regime_engine.get_regime_context()
        
        # Stage 6: Sentiment Brain (TODO)
        sentiment = None
        
        # === ADAPTIVE: Build context with adaptive info ===
        context['confidence'] = primary_conf
        context['volatility_level'] = self.adaptive_state.volatility_level
        context['drawdown'] = self.adaptive_state.rolling_drawdown
        context['loss_streak'] = self.adaptive_state.loss_streak
        context['win_streak'] = self.adaptive_state.win_streak
        context['regime'] = regime
        context['danger_reason'] = regime_context.get('danger_reason')
        context['danger_duration_bars'] = regime_context.get('danger_duration_bars', 0)

        route_decision = self._route_strategy(candle, context, features)
        context['strategy_route'] = route_decision['strategy']
        route_strategy = str(route_decision['strategy']).strip().upper()
        route_boll_z = route_decision.get('boll_z')

        if route_strategy == 'SKIP':
            self._reject_decision(
                candle,
                context,
                regime,
                route_decision['strategy'],
                raw_primary_conf,
                primary_conf,
                None,
                'router_skip',
                extra={'strategy_reason': route_decision.get('reason')},
            )
            return None

        # === EXECUTION INTELLIGENCE: Performance Memory -> Entry Strictness (System 2) ===
        perf_modifier, perf_active = self._get_performance_size_modifier(route_strategy)
        context['performance_modifier'] = perf_modifier
        context['performance_adjustment_active'] = perf_active
        if perf_active:
            if perf_modifier < 0.8:
                effective_edge_min += 0.02  # Underperforming strategy -> stricter entry
                print(f"[PERF] {symbol}: {route_strategy} underperforming (mod={perf_modifier:.2f}), tightening edge +0.02")
            elif perf_modifier > 1.1:
                effective_edge_min -= 0.02  # Outperforming strategy -> looser entry
                print(f"[PERF] {symbol}: {route_strategy} outperforming (mod={perf_modifier:.2f}), loosening edge -0.02")

        if route_strategy == 'MEAN_REVERSION' and route_boll_z is not None and abs(float(route_boll_z)) < effective_boll_z_min:
            self._reject_decision(
                candle,
                context,
                regime,
                route_decision['strategy'],
                raw_primary_conf,
                primary_conf,
                None,
                'boll_z_filter',
                extra={'boll_z': route_boll_z, 'boll_z_min': effective_boll_z_min, 'flow_state': self._flow_state},
            )
            return None

        if normalized_side is None:
            self._reject_decision(candle, context, regime, route_decision['strategy'], raw_primary_conf, primary_conf, None, f"invalid_trade_side:{primary_pred}")
            return None
        
        # Build context_output for RiskBrain V2
        context_output = {
            'operating_mode': mode,
            'manipulation_risk': 0.0,  # TODO: Get from ContextBrain
            'sr_strength': 0.5,
        }
        
        # Determine if this is an exploration trade
        is_exploration = (mode == 'LEARNING' and self.filter_engine.can_explore(mode_config))
        
        # Stage 7: Risk Brain V2 (with adaptive RR and degradative sizing)
        risk_result = self.risk_brain.calculate_sl_tp(
            entry_price=_safe_float(candle['close']),
            side=primary_pred,
            atr=_safe_float(features.get('M5_atr_14', 0)),
            regime=regime,
            context=context,
            adaptive_state=self.adaptive_state,
            exploration_trade=is_exploration,
            context_output=context_output
        )
        
        # Extract results from new dict format
        sl_price = risk_result.get('sl_price', 0)
        tp_price = risk_result.get('tp_price', 0)
        size = risk_result.get('position_size', 0)
        risk_decision = risk_result.get('risk_decision', 'BLOCK')
        risk_reason = risk_result.get('risk_reason', 'unknown')
        rr_ratio = risk_result.get('rr', 0)
        
        # === MARK-2 INTELLIGENCE ===
        # Update regime strength from features
        atr = _safe_float(features.get('M5_atr_14', 0))
        atr_avg = _safe_float(features.get('M5_atr_sma_20', atr))  # Fallback to current ATR
        self.mark2.update_regime(features.to_dict() if hasattr(features, 'to_dict') else dict(features))
        
        # Get MARK-2 modifiers
        mark2_output = self.mark2.get_modifiers(
            price=_safe_float(candle['close']),
            regime=regime,
            side=primary_pred,
            base_min_conf=mode_config.ml_buy_threshold if primary_pred == 'BUY' else mode_config.ml_sell_threshold
        )
        
        # === MARK-3 EDGE_SCORE (Profit Acceleration) ===
        from src.backtest.edge_score import calculate_volatility_alignment, calculate_structure_quality
        
        # Get structure quality from context brain (if available)
        sr_strength = context_output.get('sr_strength', 0.5)
        trend_alignment = 0.6 if regime == 'TREND' else 0.4  # Simple proxy
        structure_quality = calculate_structure_quality(sr_strength, trend_alignment)
        
        # Get volatility alignment
        vol_alignment = calculate_volatility_alignment(atr, atr_avg if atr_avg > 0 else atr)
        
        # Compute edge score
        edge_output = self.edge_score.compute(
            ml_confidence=primary_conf,
            regime_strength=mark2_output.regime_strength,
            structure_quality=structure_quality,
            volatility_alignment=vol_alignment,
            regime=regime,
            is_exploration=is_exploration
        )

        if edge_output.edge_score < effective_edge_min:
            self._reject_decision(candle, context, regime, route_decision['strategy'], raw_primary_conf, primary_conf, edge_output.edge_score, 'low_edge',
                                  extra={'effective_min': effective_edge_min, 'flow_state': self._flow_state, 'perf_active': perf_active})
            return None

        if (
            route_strategy == 'MEAN_REVERSION'
            and route_boll_z is not None
            and abs(float(route_boll_z)) < 1.0
            and edge_output.edge_score <= 0.70
        ):
            self._reject_decision(
                candle,
                context,
                regime,
                route_decision['strategy'],
                raw_primary_conf,
                primary_conf,
                edge_output.edge_score,
                'mr_quality_gate',
                extra={'boll_z': route_boll_z, 'mr_boll_z_soft_band': 1.0, 'mr_edge_override_min': 0.70},
            )
            return None
        
        # Check safeguards for boost
        boost_allowed = edge_output.boost_allowed
        if boost_allowed:
            can_boost, boost_reason = self.edge_safeguards.can_boost(candle.get('timestamp', None))
            if not can_boost:
                boost_allowed = False
                # Recompute without boost
                edge_output = self.edge_score.compute(
                    ml_confidence=primary_conf,
                    regime_strength=mark2_output.regime_strength,
                    structure_quality=structure_quality,
                    volatility_alignment=vol_alignment,
                    regime="DANGER",  # Force no boost
                    is_exploration=True
                )
        
        # === MARK-3.1: EDGE DISCOVERY (READ-ONLY OBSERVATION) ===
        # Log signal context for A-grade edge discovery
        # Does NOT change any trading behavior
        try:
            from src.backtest.edge_discovery import get_edge_discovery
            from src.backtest.session_gate import detect_session
            
            discovery = get_edge_discovery()
            timestamp = candle.get('timestamp', None)
            session = detect_session(timestamp) if timestamp else "OFF"
            
            discovery.log_signal(
                timestamp=timestamp,
                symbol=symbol,
                session=session,
                edge_score=edge_output.edge_score,
                regime_strength=mark2_output.regime_strength,
                regime_type=regime,
                rr_ratio=rr_ratio,
                atr=atr,
                atr_avg=atr_avg,
                sr_strength=sr_strength,
                decision="PENDING",  # Will be updated later
                ml_confidence=primary_conf,
                ml_side=primary_pred,
                edge_multiplier=edge_output.size_multiplier,
                mark2_modifier=mark2_output.final_size_modifier,
            )
        except Exception as e:
            # Silent fail - discovery is optional observation
            pass
        
        # === SIZING PIPELINE: base → EDGE → MARK-2 ===
        original_size = size
        
        # Step 1: Apply EDGE_SCORE multiplier (MARK-3)
        edge_adjusted = size * edge_output.size_multiplier
        
        # Step 2: Apply MARK-2 safety modifiers (always after edge)
        after_mark2 = edge_adjusted * mark2_output.final_size_modifier
        
        # Step 3: Apply performance memory modifier (System 2)
        perf_adjusted = after_mark2 * perf_modifier

        # Step 4: Apply floor (never zero)
        size = max(original_size * 0.1, perf_adjusted)
        
        # Track if boost was applied
        was_boosted = edge_output.size_multiplier > 1.0 and boost_allowed
        if was_boosted:
            self.edge_safeguards.record_trade(was_boosted=True, timestamp=candle.get('timestamp', None))
        
        # Check MARK-2 cooldown
        if not mark2_output.can_trade and risk_decision != 'BLOCK':
            risk_decision = 'BLOCK'
            risk_reason = f"MARK-2 Cooldown ({mark2_output.cooldown_remaining_min:.1f} min remaining)"
        
        # Log RiskBrain + EDGE + MARK-2 decision
        edge_info = f"EDGE:{edge_output.edge_score:.2f}({edge_output.quality_tier})"
        mark2_info = f"MARK-2:{mark2_output.final_size_modifier:.2f}"
        boost_tag = "🚀" if was_boosted else ""
        print(f"[TRACE] {symbol}: RiskBrain {risk_decision} - RR={rr_ratio:.2f}, size={size:.4f} [{edge_info}|{mark2_info}]{boost_tag}, {risk_reason}")
        
        if edge_output.edge_score > 0.7 or edge_output.size_multiplier != 1.0:
            print(f"[EDGE] {symbol}: score={edge_output.edge_score:.2f}, tier={edge_output.quality_tier}, mult={edge_output.size_multiplier:.2f}")
        if mark2_output.final_size_modifier < 0.95:
            print(f"[MARK-2] {symbol}: Memory={mark2_output.memory_mod:.2f}, Ego={mark2_output.ego_mod:.2f} (score={mark2_output.ego_score:.2f}), Regime={mark2_output.regime_mod:.2f}")

        # === SESSION GATE (Trading Hours Control) ===
        timestamp = candle.get('timestamp', None)
        if timestamp is not None:
            session_output = self.session_gate.evaluate(
                timestamp=timestamp,
                symbol=symbol,
                edge_score=edge_output.edge_score,
                regime_strength=mark2_output.regime_strength,
                confidence=primary_conf,
            )
            
            # Add session to context for downstream use
            context['session'] = session_output.session
            context['session_policy'] = session_output.policy
            
            if not session_output.allowed:
                # Session blocked - do not trade
                risk_decision = 'BLOCK'
                risk_reason = f"Session: {session_output}"
            else:
                # Apply session size multiplier
                size = size * session_output.size_multiplier
                if session_output.session == 'TOKYO' and primary_conf < self.config.tokyo_min_confidence:
                    risk_decision = 'BLOCK'
                    risk_reason = 'tokyo_low_confidence'
                if session_output.size_multiplier < 1.0:
                    print(f"[SESSION] {symbol}: {session_output.session} size×{session_output.size_multiplier:.2f}")

        # === WEAPON SYSTEM ROUTING (MARK-3.3) ===
        # Routes to appropriate strategy: RIFLE (primary) or SCALPEL (micro)
        weapon_strategy = None
        weapon_signal = None
        if self.weapon_router is not None and risk_decision != 'BLOCK':
            try:
                from src.strategy import MarketSnapshot, AgentContext
                
                # Build market snapshot
                weapon_snapshot = MarketSnapshot(
                    timestamp=timestamp,
                    symbol=symbol,
                    price=float(candle['close']),
                    atr=atr,
                    atr_avg=context.get('atr_avg', atr),
                    high_20=context.get('high_20', 0),
                    low_20=context.get('low_20', 0),
                    sma_20=context.get('sma_20', 0),
                    bb_upper=context.get('bb_upper', 0),
                    bb_lower=context.get('bb_lower', 0)
                )
                
                # Build agent context
                weapon_context = AgentContext(
                    ml_confidence=primary_conf,
                    ml_prediction=primary_pred,
                    edge_score=edge_output.edge_score,
                    edge_tier=edge_output.quality_tier,
                    regime=regime,
                    regime_strength=mark2_output.regime_strength,
                    session=context.get('session', 'OFF'),
                    mark2_can_trade=mark2_output.can_trade,
                    mark2_cooldown_min=mark2_output.cooldown_remaining_min,
                    memory_mod=mark2_output.memory_mod,
                    memory_pain_level=1.0 - mark2_output.memory_mod,
                    base_size=size,
                    rr_ratio=rr_ratio
                )
                
                # Route to strategy
                weapon_strategy, weapon_signal = self.weapon_router.route(weapon_snapshot, weapon_context)
                
                # Log decision
                if self.weapon_logger:
                    self.weapon_logger.log_decision(
                        timestamp=timestamp,
                        symbol=symbol,
                        session=weapon_context.session,
                        strategy_selected=weapon_strategy.weapon_class.value if weapon_strategy else "NONE",
                        strategy_name=weapon_strategy.name if weapon_strategy else "",
                        weapon_class=weapon_strategy.weapon_class.value if weapon_strategy else "",
                        signal=weapon_signal.signal.value if weapon_signal else "HOLD",
                        decision_reason=weapon_signal.reason if weapon_signal else "No strategy criteria met",
                        ml_confidence=primary_conf,
                        ml_prediction=primary_pred,
                        edge_score=edge_output.edge_score,
                        edge_tier=edge_output.quality_tier,
                        regime=regime,
                        regime_strength=mark2_output.regime_strength,
                        mark2_can_trade=mark2_output.can_trade,
                        mark2_modifier=mark2_output.final_size_modifier,
                        edge_multiplier=edge_output.size_multiplier,
                        strategy_size_mult=weapon_signal.size_multiplier if weapon_signal else 0.0,
                        final_size=size * (weapon_signal.size_multiplier if weapon_signal else 0.0),
                        was_blocked=weapon_strategy is None,
                        block_reason="" if weapon_strategy else "No strategy match"
                    )
                
                # Apply strategy-specific sizing (SCALPEL = 12%)
                if weapon_signal and weapon_signal.size_multiplier < 1.0:
                    size = size * weapon_signal.size_multiplier
                    print(f"[WEAPON] {symbol}: {weapon_strategy.name} size×{weapon_signal.size_multiplier:.2f}")
                    
            except Exception as e:
                # Silent fail - weapon routing is optional
                if self.config.verbose:
                    print(f"[WEAPON] Error: {e}")

        # === EXPERIENCE REASONING MODULE (ERM) ===
        # Adaptive learning from past mistakes. Does NOT block, only guides sizing.
        # MODE: ACTIVE - Apply sizing adjustments based on learned patterns
        erm_decision = None
        if self.erm is not None and risk_decision != 'BLOCK':
            try:
                # Build ERM context
                erm_context = {
                    'symbol': symbol,
                    'session': context.get('session', 'OFF'),
                    'regime': regime,
                    'regime_strength': mark2_output.regime_strength,
                    'edge_score': edge_output.edge_score,
                    'edge_tier': edge_output.quality_tier,
                    'ml_confidence': primary_conf,
                    'rr_ratio': rr_ratio,
                    'volatility_bucket': context.get('volatility_bucket', 'MID'),
                }
                
                erm_decision = self.erm.evaluate(erm_context)
                
                # Store for post-trade update
                context['erm_context'] = erm_context
                context['erm_decision'] = erm_decision
                context['erm_size_multiplier'] = 1.0
                
                # Calibration mode: keep ERM observational only (no size reduction yet).
                if self.config.verbose and erm_decision is not None:
                    print(f"[ERM] {symbol}: OBSERVE action={erm_decision.action} conf={erm_decision.confidence:.1%}")
                        
            except Exception as e:
                if self.config.verbose:
                    print(f"[ERM] Error: {e}")

        
        # Check for BLOCK decision
        if risk_decision == 'BLOCK' or size <= 0:
            self._reject_decision(candle, context, regime, route_decision['strategy'], raw_primary_conf, primary_conf, edge_output.edge_score, risk_reason)
            if self.config.verbose:
                print(f"[Pipeline] {symbol}: Blocked by RiskBrain - {risk_reason}")
            return None

        atr_pctile_for_guard = route_decision.get('atr_pctile')
        if atr_pctile_for_guard is not None:
            if atr_pctile_for_guard < self.config.volatility_low_pctile:
                if str(route_decision['strategy']).strip().upper() != 'MEAN_REVERSION':
                    self._low_volatility_block_count += 1
                    self._reject_decision(candle, context, regime, route_decision['strategy'], raw_primary_conf, primary_conf, edge_output.edge_score, 'low_volatility')
                    return None
                # System 4: Strong MR only in dead markets - require |boll_z| > 1.2
                if route_boll_z is not None and abs(float(route_boll_z)) <= 1.2:
                    self._low_volatility_block_count += 1
                    self._reject_decision(
                        candle, context, regime, route_decision['strategy'],
                        raw_primary_conf, primary_conf, edge_output.edge_score,
                        'low_vol_weak_mr',
                        extra={'atr_pctile': atr_pctile_for_guard, 'boll_z': route_boll_z, 'required_boll_z': 1.2},
                    )
                    return None
                size *= 0.5
                context['low_volatility_size_reduction'] = True
            if atr_pctile_for_guard > self.config.volatility_high_pctile:
                self._reject_decision(candle, context, regime, route_decision['strategy'], raw_primary_conf, primary_conf, edge_output.edge_score, 'extreme_volatility')
                return None

        timing_allowed, cooldown_active, timing_reason = self._passes_trade_timing_filters(context)
        context['cooldown_active'] = cooldown_active
        if not timing_allowed:
            self._reject_decision(
                candle,
                context,
                regime,
                route_decision['strategy'],
                raw_primary_conf,
                primary_conf,
                edge_output.edge_score,
                timing_reason or 'timing_filter_block',
                cooldown_active=cooldown_active,
            )
            return None

        router_regime = str(route_decision.get('classified_regime', regime)).strip().upper()
        if router_regime == 'NEUTRAL' and str(route_decision['strategy']).strip().upper() == 'MEAN_REVERSION':
            context['neutral_regime_size_reduction'] = False
        if str(regime).strip().upper() == 'DANGER':
            if str(route_decision['strategy']).strip().upper() != 'MEAN_REVERSION':
                self._reject_decision(
                    candle,
                    context,
                    regime,
                    route_decision['strategy'],
                    raw_primary_conf,
                    primary_conf,
                    edge_output.edge_score,
                    'danger_strategy_block',
                    cooldown_active=context.get('cooldown_active', False),
                    extra={
                        'danger_reason': context.get('danger_reason'),
                        'danger_duration_bars': context.get('danger_duration_bars'),
                    },
                )
                return None
            size *= 0.3
            context['danger_size_reduction'] = True
        
        # Stage 8: Meta-Gating Brain (with Adaptive Supervision)
        ml_decision = {'side': primary_pred, 'confidence': primary_conf}
        
        # === CONTEXT BRAIN: Get operating mode and adaptive thresholds ===
        n_bars = len(candles) if hasattr(candles, '__len__') else self.adaptive_state.candle_count
        mode, sufficiency, mode_config = self.filter_engine.compute_mode(n_bars)
        
        # Use adaptive threshold from filter engine (0.35 LEARNING, 0.65 CONFIRMATION)
        MIN_CONF = mode_config.ml_buy_threshold if primary_pred == 'BUY' else mode_config.ml_sell_threshold
        
        if primary_conf < MIN_CONF:
            # === EXPLORATION OVERRIDE CHECKPOINT 1 ===
            if mode == 'LEARNING' and self.filter_engine.can_explore(mode_config):
                exploration_trade = self._create_exploration_trade(
                    candle, primary_pred, features, mode_config, symbol, context
                )
                if exploration_trade:
                    print(f"[EXPLORATION] {symbol}: Override at confidence check - conf={primary_conf:.3f} < {MIN_CONF:.2f}")
                    self._stage_pending_signal(candle, context, exploration_trade)
                    return None
            
            if self.config.verbose:
                print(f"[Pipeline] {symbol}: Blocked - conf {primary_conf:.3f} < {MIN_CONF:.2f} [{mode}]")
            return None
        
        # Log mode info
        if self.config.verbose:
            print(f"[Pipeline] {symbol}: Mode={mode}, sufficiency={sufficiency:.2f}, threshold={MIN_CONF:.2f}")
        
        if self.config.enable_meta_gating:
            # === LEARNING MODE: Skip strict meta-gating, allow exploration ===
            if mode == 'LEARNING':
                if self.config.verbose:
                    print(f"[Pipeline] {symbol}: LEARNING mode - relaxed meta-gating")
            # Use adaptive supervision if AdaptiveState is ready (lowered warmup to 20)
            elif self.adaptive_state.candle_count > 20:
                allowed, reason = self.meta_gating.supervise_adaptive(
                    ml_decision, features, context, self.adaptive_state
                )
                if not allowed:
                    # === EXPLORATION OVERRIDE CHECKPOINT 2 ===
                    if mode == 'LEARNING' and self.filter_engine.can_explore(mode_config):
                        exploration_trade = self._create_exploration_trade(
                            candle, primary_pred, features, mode_config, symbol, context
                        )
                        if exploration_trade:
                            print(f"[EXPLORATION] {symbol}: Override at meta-gating - {reason}")
                            self._stage_pending_signal(candle, context, exploration_trade)
                            return None
                    
                    if self.config.verbose:
                        print(f"[Pipeline] {symbol}: Blocked by adaptive meta-gating: {reason}")
                    return None
            else:
                # Fall back to standard supervision during warmup
                if not self.meta_gating.supervise(ml_decision, None, features, context):
                    # === EXPLORATION OVERRIDE at standard meta-gating ===
                    if mode == 'LEARNING' and self.filter_engine.can_explore(mode_config):
                        print(f"[EXPLORATION] {symbol}: Override at standard meta-gating")
                        exploration_trade = self._create_exploration_trade(
                            candle, primary_pred, features, mode_config, symbol, context
                        )
                        if exploration_trade:
                            self._stage_pending_signal(candle, context, exploration_trade)
                            return None
                    if self.config.verbose:
                        print(f"[Pipeline] {symbol}: Blocked by meta-gating")
                    return None
        
        # Stage 9: Portfolio Brain (TODO)
        if self.config.enable_portfolio_brain:
            pass  # TODO: Check correlation, exposure limits
        
        # Stage 10-11: Risk + Throttle Gates
        if not self._check_gates(context):
            # === EXPLORATION OVERRIDE at risk/throttle gates ===
            if mode == 'LEARNING' and self.filter_engine.can_explore(mode_config):
                print(f"[EXPLORATION] {symbol}: Override at risk/throttle gates")
                exploration_trade = self._create_exploration_trade(
                    candle, primary_pred, features, mode_config, symbol, context
                )
                if exploration_trade:
                    self._stage_pending_signal(candle, context, exploration_trade)
                    return None
            if self.config.verbose:
                print(f"[Pipeline] {symbol}: Blocked by risk/throttle gates")
            return None

        # === EXECUTION INTELLIGENCE: Duplicate Signal Rejection (System 4) ===
        if self._is_duplicate_signal(symbol, str(normalized_side or ''), route_strategy):
            self._reject_decision(
                candle, context, regime, route_decision['strategy'],
                raw_primary_conf, primary_conf, edge_output.edge_score,
                'duplicate_signal',
                cooldown_active=context.get('cooldown_active', False),
            )
            return None

        # === EXECUTION INTELLIGENCE: Signal Quality Gate (System 5) — Adaptive Threshold ===
        signal_quality = self._compute_signal_quality(route_boll_z, edge_output.edge_score, primary_conf)
        context['signal_quality'] = signal_quality
        quality_payload = self._grade_signal_quality(features, str(primary_pred).strip().upper(), regime, primary_conf)
        quality_grade = str(quality_payload.get('grade', 'B')).strip().upper()
        context['quality_grade'] = quality_grade
        context['quality_score'] = quality_payload.get('score')

        # Adaptive threshold: base + volatility adjustment + regime adjustment
        sq_base = 0.68
        sq_atr = atr_pctile_for_guard  # from route_decision, may be None
        if sq_atr is not None:
            if sq_atr > 0.7:
                sq_threshold = sq_base - 0.03  # high vol → more opportunities
            elif sq_atr < 0.3:
                sq_threshold = sq_base + 0.03  # low vol → stricter
            else:
                sq_threshold = sq_base
        else:
            sq_threshold = sq_base

        normalized_regime_sq = str(regime).strip().upper()
        if normalized_regime_sq == 'TREND':
            sq_threshold += 0.02
        elif normalized_regime_sq == 'RANGE':
            sq_threshold -= 0.01
        # Cap combined adjustment delta to prevent over-loosening
        sq_delta = sq_threshold - sq_base
        sq_delta = max(-0.04, min(0.04, sq_delta))
        sq_threshold = sq_base + sq_delta

        # Special case: high-vol RANGE can over-relax — enforce floor
        if sq_atr is not None and sq_atr > 0.7 and normalized_regime_sq == 'RANGE':
            sq_threshold = max(sq_threshold, 0.66)

        # Clamp to safe band
        sq_threshold = max(0.60, min(0.75, sq_threshold))
        context['signal_quality_threshold'] = sq_threshold

        # Safety guards: hard floors on components
        sq_edge_floor = edge_output.edge_score >= 0.60
        sq_conf_floor = primary_conf >= 0.50

        if signal_quality < sq_threshold or not sq_edge_floor or not sq_conf_floor:
            self._signal_quality_block_count += 1
            reject_reason = 'low_signal_quality'
            if not sq_edge_floor:
                reject_reason = 'signal_quality_edge_floor'
            elif not sq_conf_floor:
                reject_reason = 'signal_quality_conf_floor'
            self._reject_decision(
                candle, context, regime, route_decision['strategy'],
                raw_primary_conf, primary_conf, edge_output.edge_score,
                reject_reason,
                cooldown_active=context.get('cooldown_active', False),
                extra={
                    'signal_quality': round(signal_quality, 4),
                    'signal_quality_threshold': round(sq_threshold, 4),
                    'atr_pctile': sq_atr,
                    'regime': normalized_regime_sq,
                    'edge_floor_pass': sq_edge_floor,
                    'conf_floor_pass': sq_conf_floor,
                    'boll_z_raw': route_boll_z,
                    'boll_z_capped': min(abs(float(route_boll_z)), 2.0) if route_boll_z is not None else 1.5,
                },
            )
            return None

        size, signal_tier, tier_size_multiplier = self._apply_signal_tier_gate(primary_conf, quality_grade, size)
        if size is None:
            reject_reason = 'tier3_reject'
            if primary_conf < self.config.tier3_confidence_floor:
                reject_reason = 'tier3_low_confidence'
            elif quality_grade in {'D', 'F'}:
                reject_reason = f'tier3_quality_{quality_grade.lower()}'
            self._reject_decision(
                candle,
                context,
                regime,
                route_decision['strategy'],
                raw_primary_conf,
                primary_conf,
                edge_output.edge_score,
                reject_reason,
                cooldown_active=context.get('cooldown_active', False),
                extra={
                    'quality_grade': quality_grade,
                    'quality_score': quality_payload.get('score'),
                    'signal_quality': round(signal_quality, 4),
                },
            )
            return None
        context['signal_tier'] = signal_tier
        context['signal_tier_size_multiplier'] = tier_size_multiplier

        normalized_regime = str(regime).strip().upper()
        normalized_strategy = str(route_decision['strategy']).strip().upper()
        soft_score = self._soft_filter_score(primary_conf, edge_output.edge_score, normalized_regime, normalized_strategy)
        high_conviction_bypass = primary_conf > 0.72 and edge_output.edge_score > 0.70
        soft_threshold = 1
        if self._flow_state == 'relaxed':
            soft_threshold = 0
        elif self._flow_state == 'tight':
            soft_threshold = 2
        context['soft_filter_score'] = soft_score
        context['soft_filter_threshold'] = soft_threshold
        context['high_conviction_bypass'] = high_conviction_bypass
        if normalized_regime == 'RANGE' and normalized_strategy != 'MEAN_REVERSION':
            self._reject_decision(candle, context, regime, route_decision['strategy'], raw_primary_conf, primary_conf, edge_output.edge_score, 'regime_alignment_violation')
            return None
        if normalized_regime == 'TREND' and normalized_strategy != 'TREND_PULLBACK':
            self._reject_decision(candle, context, regime, route_decision['strategy'], raw_primary_conf, primary_conf, edge_output.edge_score, 'regime_alignment_violation')
            return None

        # Stage 12: Execution Reflex Engine - Build final decision
        decision = Decision(
            action='open',
            side=normalized_side,
            entry_price=_safe_float(candle['close']),
            size=size,
            sl_price=sl_price,
            tp_price=tp_price,
            decision_source='PRIMARY',
            regime=regime,
            confidence=primary_conf,
            features=features.to_dict() if self.config.log_features else None,
            metadata={
                'symbol': symbol,
                'strategy': route_decision['strategy'],
                'strategy_reason': route_decision['reason'],
                'strategy_confidence': route_decision['confidence'],
                'ml_conf_raw': raw_primary_conf,
                'ml_conf_calibrated': primary_conf,
                'edge_score': edge_output.edge_score,
                'skip_reason': None,
                'cooldown_active': context.get('cooldown_active', False),
                'soft_filter_score': soft_score,
                'soft_filter_threshold': context.get('soft_filter_threshold'),
                'high_conviction_bypass': high_conviction_bypass,
                'signal_tier': context.get('signal_tier'),
                'signal_tier_size_multiplier': context.get('signal_tier_size_multiplier'),
                'quality_grade': context.get('quality_grade'),
                'quality_score': context.get('quality_score'),
                'neutral_regime_size_reduction': context.get('neutral_regime_size_reduction', False),
                'low_volatility_size_reduction': context.get('low_volatility_size_reduction', False),
                'danger_size_reduction': context.get('danger_size_reduction', False),
                'danger_reason': context.get('danger_reason'),
                'danger_duration_bars': context.get('danger_duration_bars', 0),
                'timeframe': context.get('timeframe', 'UNKNOWN'),
                'timestamp': datetime.utcnow().isoformat(),
                'regime_conf': regime_conf,
                # Adaptive metadata
                'volatility_level': self.adaptive_state.volatility_level,
                'drawdown': self.adaptive_state.rolling_drawdown,
                'loss_streak': self.adaptive_state.loss_streak,
                'risk_pct': context.get('risk_pct', 0.01),
                'rl_active': context.get('rl_active', False),
                'rl_size_multiplier': context.get('rl_size_multiplier', 1.0),
                'adjusted_threshold': self.config.primary_high_thresh,
                # === EXECUTION INTELLIGENCE LOGGING ===
                'entry_mode': 'delayed_confirmation',
                'signal_quality': context.get('signal_quality'),
                'performance_modifier': context.get('performance_modifier', 1.0),
                'performance_adjustment_active': context.get('performance_adjustment_active', False),
                'flow_state': self._flow_state,
                'flow_conf_adj': flow_conf_adj,
                'flow_edge_adj': flow_edge_adj,
                'effective_edge_min': effective_edge_min,
                'effective_boll_z_min': effective_boll_z_min,
            }
        )
        self._stage_pending_signal(candle, context, decision)

        if self.config.verbose:
            print(f"[Pipeline] {symbol}: PENDING {decision.side} @ {decision.entry_price:.5f} "
                  f"size={decision.size:.4f} conf={decision.confidence:.3f} regime={regime} "
                  f"mode=delayed_confirmation")

        return None

    @staticmethod
    def _normalize_trade_side(side: Optional[str]) -> Optional[str]:
        """Only BUY/SELL are executable; HOLD or anything else is non-tradable."""
        if side is None:
            return None
        normalized = str(side).strip().upper()
        if normalized in {"BUY", "SELL"}:
            return normalized.lower()
        return None

    def _route_strategy(self, candle: pd.Series, context: Dict, features: pd.Series) -> Dict[str, Any]:
        """Run the deterministic router every bar and return strategy metadata."""
        timestamp = candle.get('timestamp', None)
        hour = timestamp.hour if hasattr(timestamp, 'hour') else 12

        adx_val = self._require_router_feature(features, ['M5_adx_14', 'adx_14', 'adx14'], 'adx_14')
        atr_pctile = self._require_router_feature(features, ['M5_atr_pctile', 'atr_pctile', 'vol_atr_pctile_100'], 'atr_pctile')
        boll_z = self._require_router_feature(features, ['M5_boll_z', 'boll_z'], 'boll_z')
        ret_std = _safe_float(features.get('ret_std', features.get('M5_returns_std_20', 0.0)), 0.0)

        strategy = "UNROUTED"
        reason = "weapon_system_disabled"
        confidence = 0.0

        if adx_val is None or atr_pctile is None or boll_z is None:
            reason = "missing_router_features"
            route_info = {
                "timestamp": timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
                "symbol": context.get('symbol', 'UNKNOWN'),
                "strategy": "SKIP",
                "reason": reason,
                "confidence": 0.0,
                "adx": None if adx_val is None else round(float(adx_val), 4),
                "atr_pctile": None if atr_pctile is None else round(float(atr_pctile), 4),
                "boll_z": None if boll_z is None else round(float(boll_z), 4),
            }
            self._log_strategy_route(route_info)
            return route_info

        if self.strategy_router is not None:
            from src.pipeline.regime_classifier import classify_regime_scalar

            regime_info = classify_regime_scalar(adx_val, atr_pctile, ret_std)
            route = self.strategy_router.route(
                regime_score=regime_info["regime_score"],
                adx_val=adx_val,
                atr_pctile=atr_pctile,
                hour=hour,
                regime_label=regime_info["regime"],
            )
            strategy = route.strategy
            reason = route.reason
            confidence = route.confidence

        route_info = {
            "timestamp": timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
            "symbol": context.get('symbol', 'UNKNOWN'),
            "strategy": strategy,
            "reason": reason,
            "confidence": round(float(confidence), 4),
            "classified_regime": regime_info["regime"] if self.strategy_router is not None else None,
            "adx": round(float(adx_val), 4),
            "atr_pctile": round(float(atr_pctile), 4),
            "boll_z": round(float(boll_z), 4),
        }
        if self._router_debug_samples < 5:
            print(f"ADX={adx_val:.4f}, ATR={atr_pctile:.4f}, BOLL_Z={boll_z:.4f}")
            self._router_debug_samples += 1
        self._log_strategy_route(route_info)
        return route_info

    def _require_router_feature(self, features: pd.Series, keys: List[str], feature_name: str) -> Optional[float]:
        """Fetch a router-critical feature without silently falling back to misleading defaults."""
        for key in keys:
            if key in features.index:
                value = _safe_float(features.get(key), np.nan)
                if not np.isnan(value):
                    return float(value)
        LOG = globals().get("logger")
        if LOG:
            LOG.error(f"[ROUTER] Missing required feature: {feature_name} (checked {keys})")
        else:
            print(f"[ROUTER] Missing required feature: {feature_name} (checked {keys})")
        return None

    def _log_strategy_route(self, route_info: Dict[str, Any]) -> None:
        """Minimal per-bar router logging for validation."""
        if self.strategy_router_log_path is None:
            return
        try:
            with open(self.strategy_router_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(route_info) + "\n")
        except Exception:
            if self.config.verbose:
                print("[ROUTER] Failed to write router log")
    
    def _map_features_for_regime(self, features: pd.Series) -> pd.Series:
        """Map M5_ prefixed features to standard names for Regime Engine"""
        mapping = {
            'M5_sma_20': 'sma_20',
            'M5_sma_50': 'sma_50',
            'M5_bb_width': 'bb_width',
            'M5_atr_14': 'atr_14',
            'M5_volatility': 'volatility',
            'M5_close': 'close'
        }
        mapped = features.copy()
        for m5_col, std_col in mapping.items():
            if m5_col in features:
                mapped[std_col] = features[m5_col]
        return mapped

    def _extract_features(self, candle: pd.Series, context: Dict) -> Optional[pd.Series]:
        """
        Stage 2: Feature Reactor
        Extract features from current candle and historical data
        """
        try:
            # Get historical data from context
            history = context.get('history')
            if history is None or len(history) < 50:
                if self.config.verbose:
                    print(f"[Pipeline] Insufficient history: {len(history) if history is not None else 0}")
                return None
            
            # Compute technical indicators on historical data
            df_with_indicators = _compute_technical_indicators(history)
            
            # Return features from last row
            return df_with_indicators.iloc[-1]
        
        except Exception as e:
            if self.config.verbose:
                print(f"[Pipeline] Feature extraction failed: {e}")
                traceback.print_exc()
            return None
    
    def _primary_brain(self, features: pd.Series, context: Dict) -> Tuple[float, str]:
        """
        Stage 3: Primary ML Brain
        Get confidence and prediction from primary model
        """
        if self.primary_model is None:
            # Fallback to heuristic
            return self._heuristic_decision(features)
        
        try:
            # Prepare feature vector
            # TODO: Align features with model's expected features
            feature_vector = features.fillna(0).values.reshape(1, -1)
            
            # Get prediction
            if hasattr(self.primary_model, 'predict_proba'):
                proba = self.primary_model.predict_proba(feature_vector)
                confidence = float(np.max(proba[0]))
                pred_class = int(np.argmax(proba[0]))
                side = 'buy' if pred_class == 1 else 'sell'
            elif hasattr(self.primary_model, 'predict'):
                pred = self.primary_model.predict(feature_vector)
                side = 'buy' if float(pred[0]) > 0 else 'sell'
                confidence = min(abs(float(pred[0])), 1.0)
            else:
                return self._heuristic_decision(features)
            
            return confidence, side
        
        except Exception as e:
            if self.config.verbose:
                print(f"[Pipeline] Primary model error: {e}")
            return self._heuristic_decision(features)
    
    def _heuristic_decision(self, features: pd.Series) -> Tuple[float, str]:
        """Fallback heuristic decision logic"""
        try:
            # Simple momentum-based heuristic
            close = _safe_float(features.get('close', 0))
            sma_20 = _safe_float(features.get('sma_20', 0))
            rsi = _safe_float(features.get('rsi', 50))
            
            if close > sma_20 and rsi < 70:
                return 0.75, 'buy'
            elif close < sma_20 and rsi > 30:
                return 0.75, 'sell'
            else:
                return 0.30, 'hold'
        except Exception:
            return 0.0, 'hold'
    
    def _create_exploration_trade(
        self, 
        candle: pd.Series, 
        side: str, 
        features: pd.Series, 
        mode_config,
        symbol: str,
        context: Dict
    ) -> Optional[Decision]:
        """
        Create an exploration trade with safety constraints.
        
        Constraints:
        - RR >= 1.2
        - Size capped at 0.3x (exploration_size_cap)
        - Max 2 trades/day
        - No execution if manipulation_risk > 0.7
        """
        from datetime import datetime
        
        try:
            entry = _safe_float(candle['close'])
            atr = _safe_float(features.get('M5_atr_14', features.get('atr_14', 0.001)))
            normalized_side = self._normalize_trade_side(side)

            if context.get('regime') == 'DANGER' or normalized_side is None:
                return None
            
            # Safety check: Skip if ATR is too small
            if atr < 0.00001:
                print(f"[EXPLORATION] {symbol}: Skipped - ATR too small ({atr})")
                return None
            
            # Check manipulation risk from Context Brain
            manipulation_risk = context.get('manipulation_risk', 0)
            if manipulation_risk > 0.7:
                print(f"[EXPLORATION] {symbol}: Skipped - manipulation risk {manipulation_risk:.2f} > 0.7")
                return None
            
            # Calculate SL/TP with minimum RR of 1.2
            MIN_RR = 1.2
            sl_distance = atr * 1.5  # Tight SL for exploration
            tp_distance = sl_distance * MIN_RR
            
            if normalized_side == 'buy':
                sl_price = entry - sl_distance
                tp_price = entry + tp_distance
            else:  # sell
                sl_price = entry + sl_distance
                tp_price = entry - tp_distance
            
            # Capped position size (0.3x normal)
            size = mode_config.exploration_size_cap  # 0.3
            
            # Record exploration trade
            self.filter_engine.record_exploration()
            
            # Build exploration decision
            decision = Decision(
                action='open',
                side=normalized_side,
                entry_price=entry,
                size=size,
                sl_price=sl_price,
                tp_price=tp_price,
                decision_source='EXPLORATION',
                regime=context.get('regime', 'UNKNOWN'),
                confidence=0.35,  # Fixed low confidence for exploration
                features=None,  # Don't log features for exploration
                metadata={
                    'symbol': symbol,
                    'timeframe': context.get('timeframe', 'M5'),
                    'timestamp': datetime.utcnow().isoformat(),
                    'trade_type': 'EXPLORATION',
                    'strategy': context.get('strategy_route', 'EXPLORATION'),
                    'exploration_count': self.filter_engine.exploration_count,
                    'manipulation_risk': manipulation_risk,
                    'rr_ratio': MIN_RR,
                    'reason': 'LEARNING mode exploration trade'
                }
            )
            
            print(f"[EXPLORATION] {symbol}: Created {normalized_side.upper()} @ {entry:.5f} size={size:.2f} "
                  f"SL={sl_price:.5f} TP={tp_price:.5f} RR={MIN_RR} "
                  f"(trade #{self.filter_engine.exploration_count})")
            
            return decision
            
        except Exception as e:
            print(f"[EXPLORATION] {symbol}: Failed to create trade - {e}")
            return None
    
    def _rl_brain(self, features: pd.Series, context: Dict) -> Optional[Decision]:
        """
        Stage 4: RL Brain (FinRL fallback in grey zone)
        """
        if self.finrl_adapter is None:
            return None
        
        try:
            # TODO: Implement FinRL prediction
            # For now, return None
            return None
        except Exception as e:
            if self.config.verbose:
                print(f"[Pipeline] RL Brain error: {e}")
            return None
    
    def _risk_brain(self, candle: pd.Series, features: pd.Series, side: str, regime: str) -> Tuple[Optional[float], Optional[float], float]:
        """
        Stage 7: Risk Brain
        Calculate SL, TP, and position size
        """
        try:
            price = _safe_float(candle['close'])
            atr = _safe_float(features.get('atr', 0))
            
            # Calculate SL/TP
            if atr > 0:
                # ATR-based
                stop_dist = atr * self.config.stop_atr_mult
                tp_dist = stop_dist * self.config.takeprofit_rr
                
                if side == 'buy':
                    sl_price = price - stop_dist
                    tp_price = price + tp_dist
                else:
                    sl_price = price + stop_dist
                    tp_price = price - tp_dist
            else:
                # Percentage-based fallback
                if side == 'buy':
                    sl_price = price * (1 - self.config.sl_pct)
                    tp_price = price * (1 + self.config.tp_pct)
                else:
                    sl_price = price * (1 + self.config.sl_pct)
                    tp_price = price * (1 - self.config.tp_pct)
            
            # Calculate position size (risk-based)
            # TODO: Get account balance from context
            account_balance = 10000  # Default
            risk_amount = account_balance * self.config.risk_per_trade
            stop_distance = abs(price - sl_price)
            
            if stop_distance > 0:
                size = risk_amount / stop_distance
            else:
                size = 0.01  # Minimum size
            
            # Adjust size based on regime
            if regime == 'UNCERTAIN':
                size *= 0.5  # Reduce size in uncertain conditions
            
            return sl_price, tp_price, size
        
        except Exception as e:
            if self.config.verbose:
                print(f"[Pipeline] Risk Brain error: {e}")
            return None, None, 0.01
    
    def _meta_gating(self, features: pd.Series, regime: str, confidence: float, context: Dict) -> bool:
        """
        Stage 8: Meta-Gating Brain
        Filter conditions that historically produce poor trades
        """
        try:
            # 1. Confidence Filter
            min_conf = self.config.min_confidence_normal
            if regime == 'UNCERTAIN':
                min_conf = self.config.min_confidence_uncertain
            
            if confidence < min_conf:
                if self.config.verbose:
                    print(f"[Meta-Gating] Low confidence: {confidence:.3f} < {min_conf} ({regime})")
                return False
            
            # 2. Regime Filter
            if regime == 'UNCERTAIN':
                volatility = _safe_float(features.get('volatility', 0))
                if volatility > 0.03:  # Very high volatility
                    return False
            
            # 3. Extreme RSI Filter
            rsi = _safe_float(features.get('rsi', 50))
            if rsi < 10 or rsi > 90:
                return False
            
            # 4. Volume Filter
            if 'volume_ratio' in features:
                volume_ratio = _safe_float(features.get('volume_ratio', 1))
                if volume_ratio < 0.3:  # Very low volume
                    return False
            
            return True
        
        except Exception:
            return True  # Allow trade if gating fails
    
    def _check_gates(self, context: Dict) -> bool:
        """
        Stage 10-11: Risk + Throttle Gates
        Check position limits and frequency throttles
        """
        try:
            # 1. Max Open Positions
            open_positions = context.get('open_positions', [])
            if len(open_positions) >= self.config.max_open_positions:
                if self.config.verbose:
                    print(f"[Gates] Max open positions reached: {len(open_positions)}")
                return False
            
            # 2. Daily Trade Limits
            trades_today = context.get('trades_today', 0)
            if trades_today >= self.config.max_trades_per_day_per_symbol:
                if self.config.verbose:
                    print(f"[Gates] Daily trade limit reached: {trades_today}")
                return False
            
            # 3. Minimum Bars Between Trades
            bars_since_last_trade = context.get('bars_since_last_trade', 9999)
            if bars_since_last_trade < self.config.min_bars_between_trades:
                if self.config.verbose:
                    print(f"[Gates] Cooldown active: {bars_since_last_trade} < {self.config.min_bars_between_trades} bars")
                return False
            
            # 4. Loss Streak Cooldown
            consecutive_losses = context.get('consecutive_losses', 0)
            bars_since_last_loss = context.get('bars_since_last_loss', 9999)
            
            if consecutive_losses >= self.config.loss_streak_cooldown_trades:
                if bars_since_last_loss < self.config.loss_streak_cooldown_bars:
                    if self.config.verbose:
                        print(f"[Gates] Loss streak cooldown: {consecutive_losses} losses, {bars_since_last_loss} bars ago")
                    return False
            
            return True
        
        except Exception:
            return True  # Allow trade if gate check fails
