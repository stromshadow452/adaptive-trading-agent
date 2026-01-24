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
import os
import traceback
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
    enable_sentiment: bool = False
    enable_slicer: bool = False  # Added for compatibility
    
    # Logging
    verbose: bool = False
    log_features: bool = True

    # Gating & Throttling (loaded from config/trade_gating.yaml)
    min_confidence_normal: float = 0.60
    min_confidence_uncertain: float = 0.75
    max_trades_per_day_per_symbol: int = 5
    min_bars_between_trades: int = 12
    max_open_positions: int = 3
    loss_streak_cooldown_trades: int = 3
    loss_streak_cooldown_bars: int = 24


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
        if getattr(self.config, 'enable_weapon_system', False):
            try:
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
                    self.config.max_trades_per_day_per_symbol = gating_config.get('max_trades_per_day_per_symbol', self.config.max_trades_per_day_per_symbol)
                    self.config.min_bars_between_trades = gating_config.get('min_bars_between_trades', self.config.min_bars_between_trades)
                    self.config.max_open_positions = gating_config.get('max_open_positions', self.config.max_open_positions)
                    self.config.loss_streak_cooldown_trades = gating_config.get('loss_streak_cooldown_trades', self.config.loss_streak_cooldown_trades)
                    self.config.loss_streak_cooldown_bars = gating_config.get('loss_streak_cooldown_bars', self.config.loss_streak_cooldown_bars)
                    
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
        
        print(f"[TRACE] {symbol}: ML pred={primary_pred}, conf={primary_conf:.3f}")
        
        # Stage 4: RL Brain (grey zone fallback)
        if self.config.enable_rl_fallback:
            if self.config.primary_block_thresh < primary_conf < self.config.primary_high_thresh:
                rl_decision = self._rl_brain(features, context)
                if rl_decision:
                    return rl_decision
        
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
                        return exploration_trade
                else:
                    print(f"[EXPLORATION] {symbol}: Cannot explore - budget exhausted")
            else:
                print(f"[DIAG] {symbol}: Exploration condition FAILED - mode={mode} (need LEARNING), pred={primary_pred} (need BUY/SELL)")
            
            if self.config.verbose:
                print(f"[Pipeline] {symbol}: Blocked - primary_conf={primary_conf:.3f} too low (mode={mode})")
            return None
        
        # === ADAPTIVE: Update AdaptiveState with candle ===
        self.adaptive_state.update_candle(candle, features)
        
        # Stage 5: Regime Engine (with AdaptiveState)
        # Map MTF features to Regime Engine expected format (M5_sma_20 -> sma_20)
        regime_features = self._map_features_for_regime(features)
        regime, regime_conf = self.regime_engine.detect(regime_features, self.adaptive_state)
        
        # Stage 6: Sentiment Brain (TODO)
        sentiment = None
        
        # === ADAPTIVE: Build context with adaptive info ===
        context['confidence'] = primary_conf
        context['volatility_level'] = self.adaptive_state.volatility_level
        context['drawdown'] = self.adaptive_state.rolling_drawdown
        context['loss_streak'] = self.adaptive_state.loss_streak
        context['win_streak'] = self.adaptive_state.win_streak
        context['regime'] = regime
        
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
        
        # Step 3: Apply floor (never zero)
        size = max(original_size * 0.1, after_mark2)
        
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
                regime_strength=mark2_output.regime_strength
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
                
                # ACTIVE MODE: Apply ERM guidance (never blocks, only adjusts)
                if erm_decision.action == "IGNORE" and erm_decision.confidence > 0.7:
                    # High-confidence ignore → reduce significantly
                    size = size * 0.2
                    print(f"[ERM] {symbol}: IGNORE (conf={erm_decision.confidence:.1%}) → size×0.20 | {erm_decision.reason}")
                elif erm_decision.action == "REDUCE":
                    size = size * erm_decision.size_multiplier
                    if erm_decision.size_multiplier < 0.9:
                        print(f"[ERM] {symbol}: REDUCE → size×{erm_decision.size_multiplier:.2f} | {erm_decision.reason}")
                elif erm_decision.action == "EXECUTE" and erm_decision.confidence > 0.7:
                    # High-confidence pattern → log it
                    if self.config.verbose:
                        print(f"[ERM] {symbol}: EXECUTE (SR={erm_decision.confidence:.1%}) | {erm_decision.reason}")
                        
            except Exception as e:
                if self.config.verbose:
                    print(f"[ERM] Error: {e}")

        
        # Check for BLOCK decision
        if risk_decision == 'BLOCK' or size <= 0:
            context['rejected_reason'] = risk_reason
            if self.config.verbose:
                print(f"[Pipeline] {symbol}: Blocked by RiskBrain - {risk_reason}")
            return None
        
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
                    return exploration_trade
            
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
                            return exploration_trade
                    
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
                            return exploration_trade
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
                    return exploration_trade
            if self.config.verbose:
                print(f"[Pipeline] {symbol}: Blocked by risk/throttle gates")
            return None
        
        # Stage 12: Execution Reflex Engine - Build final decision
        decision = Decision(
            action='open',
            side=primary_pred,
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
                'timeframe': context.get('timeframe', 'UNKNOWN'),
                'timestamp': datetime.utcnow().isoformat(),
                'regime_conf': regime_conf,
                # Adaptive metadata
                'volatility_level': self.adaptive_state.volatility_level,
                'drawdown': self.adaptive_state.rolling_drawdown,
                'loss_streak': self.adaptive_state.loss_streak,
                'risk_pct': context.get('risk_pct', 0.01),
                'adjusted_threshold': self.config.primary_high_thresh
            }
        )
        
        # Stage 13: Logging & Monitoring (with Adaptive State)
        if self.config.verbose:
            print(f"[Pipeline] {symbol}: EXECUTE {decision.side} @ {decision.entry_price:.5f} "
                  f"size={decision.size:.4f} conf={decision.confidence:.3f} regime={regime} "
                  f"vol={self.adaptive_state.volatility_level} dd={self.adaptive_state.rolling_drawdown:.1%}")
        
        return decision
    
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
            
            if side == 'BUY':
                sl_price = entry - sl_distance
                tp_price = entry + tp_distance
            else:  # SELL
                sl_price = entry + sl_distance
                tp_price = entry - tp_distance
            
            # Capped position size (0.3x normal)
            size = mode_config.exploration_size_cap  # 0.3
            
            # Record exploration trade
            self.filter_engine.record_exploration()
            
            # Build exploration decision
            decision = Decision(
                action='open',
                side=side,
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
                    'exploration_count': self.filter_engine.exploration_count,
                    'manipulation_risk': manipulation_risk,
                    'rr_ratio': MIN_RR,
                    'reason': 'LEARNING mode exploration trade'
                }
            )
            
            print(f"[EXPLORATION] {symbol}: Created {side} @ {entry:.5f} size={size:.2f} "
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
