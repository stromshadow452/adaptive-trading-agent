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
from src.pipeline.pipeline_v2 import PipelineV2

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

    # Gating & Throttling (REBUILD_V1: max 1 trade, 5 bar cooldown)
    min_confidence_normal: float = 0.60
    min_confidence_uncertain: float = 0.75
    max_trades_per_day_per_symbol: int = 10
    min_bars_between_trades: int = 5  # 5 bar cooldown (REBUILD_V1)
    max_open_positions: int = 1  # REBUILD_V1: max 1 trade at a time
    loss_streak_cooldown_trades: int = 2
    loss_streak_cooldown_bars: int = 5  # Consistent with main cooldown
    calibrated_confidence_min: float = 0.50
    confidence_deadzone_low: float = 0.50
    confidence_deadzone_high: float = 0.55
    edge_score_min: float = 0.60
    volatility_low_pctile: float = 0.20
    volatility_high_pctile: float = 0.90
    cooldown_bars_after_loss: int = 2
    cooldown_bars_after_win: int = 0
    trade_cluster_bars: int = 5  # 5 bars (REBUILD_V1)
    tokyo_min_confidence: float = 0.60
    tier1_confidence_min: float = 0.60
    tier2_confidence_min: float = 0.55
    tier3_confidence_floor: float = 0.50
    enable_duplicate_signal_checks: bool = True
    duplicate_signal_window_bars: int = 5
    enable_timing_filters: bool = True
    enable_flow_balancer: bool = True
    flow_relaxed_after_bars: int = 50
    flow_tight_after_trades: int = 3
    flow_relaxed_conf_adjustment: float = -0.03
    flow_relaxed_edge_adjustment: float = -0.03
    flow_tight_conf_adjustment: float = 0.02
    flow_tight_edge_adjustment: float = 0.02
    enable_operating_mode_parameters: bool = True


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

class TradingPipeline(PipelineV2):
    """
    13-Stage SCOPUS Trading Pipeline with Adaptive Behavior
    """
    
    def __init__(self, config: PipelineConfigV2):
        PipelineV2.__init__(self, cfg=config)
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
        
        # Initialize Diagnostics & Regime V2
        from src.diagnostics.drift_monitor import DriftMonitor
        from src.diagnostics.anomaly_detector import AnomalyDetector
        from src.regime.regime_engine_v2 import RegimeEngineV2
        self.drift_monitor = DriftMonitor()
        self.anomaly_detector = AnomalyDetector()
        self.regime_engine_v2 = RegimeEngineV2()
        
        # Initialize Probability Based ML Filter
        try:
            from src.ml.probability_filter import ProbabilityBasedMLFilter
            self.probability_filter = ProbabilityBasedMLFilter()
        except Exception as e:
            self.probability_filter = None
            if self.config.verbose:
                print(f"[Pipeline] Failed to load ProbabilityBasedMLFilter: {e}")
        
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
