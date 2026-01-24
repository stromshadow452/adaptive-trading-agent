"""
Unified Execution Core - PHASE-3: ERM + ADAPTIVE STATE
========================================================

PHASE-3 ADDITIONS:
- ERM (Experience Reasoning Module) for learning from outcomes
- AdaptiveState for context-aware behavior (NORMAL/CAUTION/DEFENSIVE)

WHAT CHANGES:
- Trading behavior adapts to recent performance
- Slows down during loss clusters
- Full participation during good phases

WHAT STAYS UNCHANGED:
- ML thresholds (LOCKED: 0.55/0.70)
- RSI logic (confirmation only)
- Risk ladder, SL/TP, MARK-2 rules
"""

import logging
import joblib
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from collections import deque
from enum import Enum
import numpy as np

# PHASE-4: Quality Scorer Upgrade
try:
    from src.backtest.quality_scorer import QualityScorer, QualityBreakdown
    QUALITY_SCORER_AVAILABLE = True
except ImportError:
    QUALITY_SCORER_AVAILABLE = False

logger = logging.getLogger(__name__)


# ============================================================================
# EXCEPTIONS
# ============================================================================

class TradeSanityViolation(Exception):
    """Raised when trade violates correctness invariants."""
    pass


# ============================================================================
# ADAPTIVE STATE (NORMAL / CAUTION / DEFENSIVE)
# ============================================================================

class AdaptiveMode(Enum):
    NORMAL = "NORMAL"       # Full participation
    CAUTION = "CAUTION"     # Slower, more selective
    DEFENSIVE = "DEFENSIVE" # Self-preservation mode


@dataclass
class AdaptiveState:
    """
    Context-aware state that answers:
    "Is this a good time to trade aggressively or cautiously?"
    
    Inputs:
    - Recent trade outcomes (last 10 trades)
    - Loss cluster detection
    - Regime stability
    
    Output:
    - mode: NORMAL / CAUTION / DEFENSIVE
    """
    
    # Recent trade history (max 10)
    recent_trades: deque = field(default_factory=lambda: deque(maxlen=10))
    
    # Counters
    recent_wins: int = 0
    recent_losses: int = 0
    
    # Cluster detection
    last_3_outcomes: deque = field(default_factory=lambda: deque(maxlen=3))
    last_5_outcomes: deque = field(default_factory=lambda: deque(maxlen=5))
    
    # Regime tracking
    regime_changes: int = 0
    last_regime: str = "RANGE"
    
    def get_mode(self) -> AdaptiveMode:
        """
        Determine current adaptive mode based on recent performance.
        
        RULES:
        🟢 NORMAL: Default, full participation
        🟡 CAUTION: 2 losses in last 3 OR win rate < 45%
        🔴 DEFENSIVE: 3 losses in last 5 OR regime unstable
        """
        # Count losses in last 3 and last 5
        losses_in_3 = sum(1 for x in self.last_3_outcomes if not x)
        losses_in_5 = sum(1 for x in self.last_5_outcomes if not x)
        
        # Calculate recent win rate
        total = len(self.recent_trades)
        win_rate = self.recent_wins / total if total > 0 else 0.5
        
        # DEFENSIVE: 3 losses in last 5 OR regime very unstable
        if losses_in_5 >= 3:
            return AdaptiveMode.DEFENSIVE
        
        # CAUTION: 2 losses in last 3 OR win rate < 45%
        if losses_in_3 >= 2:
            return AdaptiveMode.CAUTION
        
        if total >= 5 and win_rate < 0.45:
            return AdaptiveMode.CAUTION
        
        # NORMAL: Everything looks fine
        return AdaptiveMode.NORMAL
    
    def update(self, is_win: bool, regime: str):
        """Update state after a trade closes."""
        # Track outcome
        self.recent_trades.append(is_win)
        self.last_3_outcomes.append(is_win)
        self.last_5_outcomes.append(is_win)
        
        # Update counters
        if is_win:
            self.recent_wins += 1
        else:
            self.recent_losses += 1
        
        # Handle counter overflow when deque drops old trades
        if len(self.recent_trades) == self.recent_trades.maxlen:
            # Recount from deque
            self.recent_wins = sum(1 for x in self.recent_trades if x)
            self.recent_losses = len(self.recent_trades) - self.recent_wins
        
        # Track regime changes
        if regime != self.last_regime:
            self.regime_changes += 1
            self.last_regime = regime
    
    def get_recent_winrate(self) -> float:
        """Get win rate of recent trades."""
        total = len(self.recent_trades)
        if total == 0:
            return 0.5
        return sum(1 for x in self.recent_trades if x) / total
    
    def is_loss_cluster(self) -> bool:
        """Detect if we're in a loss cluster."""
        losses_in_3 = sum(1 for x in self.last_3_outcomes if not x)
        return losses_in_3 >= 2


# ============================================================================
# TRADE RECORD
# ============================================================================

@dataclass
class TradeRecord:
    """Standardized trade record."""
    trade_id: str
    symbol: str
    
    entry_timestamp: Any
    exit_timestamp: Optional[Any] = None
    entry_candle_index: int = 0
    exit_candle_index: Optional[int] = None
    
    side: str = ""
    entry_price: float = 0.0
    exit_price: Optional[float] = None
    size: float = 0.0
    
    sl_price: float = 0.0
    tp_price: float = 0.0
    
    pnl: float = 0.0
    r_multiple: float = 0.0
    exit_reason: str = ""
    is_win: bool = False
    
    regime: str = ""
    ml_confidence: float = 0.0
    decision_source: str = ""
    adaptive_mode: str = ""  # PHASE-3: Track which mode was active
    
    def to_dict(self) -> Dict:
        return asdict(self)


# ============================================================================
# EXECUTION STATE
# ============================================================================

@dataclass
class ExecutionState:
    """Complete execution state."""
    position_open: bool = False
    position_side: str = ""
    position_entry: float = 0.0
    position_size: float = 0.0
    position_sl: float = 0.0
    position_tp: float = 0.0
    position_entry_index: int = 0
    position_entry_time: Optional[Any] = None
    position_regime: str = ""
    position_symbol: str = ""
    position_ml_confidence: float = 0.0
    position_decision_source: str = ""
    position_adaptive_mode: str = ""
    
    mark2_memory: float = 1.0
    mark2_ego: float = 1.0
    mark2_cooldown_until: int = 0
    
    current_candle_index: int = 0
    last_trade_index: int = -999
    consecutive_losses: int = 0
    
    equity: float = 10000.0
    initial_equity: float = 10000.0
    peak_equity: float = 10000.0
    max_drawdown: float = 0.0


# ============================================================================
# ML BRAIN
# ============================================================================

class MLBrain:
    """ML model wrapper."""
    
    FEATURE_NAMES = [
        "open", "high", "low", "close", "volume", "returns",
        "sma_20", "sma_50", "ema_12", "ema_26", "trend_flag",
        "rsi_14", "macd", "macd_signal", "macd_hist",
        "atr_14", "bb_middle", "bb_upper", "bb_lower", "bb_width", "volatility"
    ]
    
    def __init__(self, model_path: Optional[Path] = None):
        self.model = None
        self.model_loaded = False
        
        if model_path is None:
            model_path = Path("models/xgb_primary.joblib")
        
        if model_path.exists():
            try:
                self.model = joblib.load(model_path)
                self.model_loaded = True
                logger.info(f"[MLBrain] Loaded model from {model_path}")
            except Exception as e:
                logger.warning(f"[MLBrain] Failed to load: {e}")
    
    def predict(self, features: Dict) -> Tuple[str, float]:
        """Run ML inference."""
        if not self.model_loaded:
            return "HOLD", 0.0
        
        try:
            feature_vector = [features.get(name, 0.0) for name in self.FEATURE_NAMES]
            X = np.array([feature_vector])
            
            if hasattr(self.model, 'predict_proba'):
                probs = self.model.predict_proba(X)[0]
                if probs[1] > probs[0]:
                    return "BUY", probs[1]
                else:
                    return "SELL", probs[0]
            return "HOLD", 0.5
        except:
            return "HOLD", 0.0


# ============================================================================
# EXECUTION CORE - PHASE-3: WITH ERM + ADAPTIVE STATE
# ============================================================================

class ExecutionCore:
    """
    THE SINGLE SOURCE OF TRUTH FOR ALL TRADING DECISIONS.
    
    PHASE-3 ENHANCEMENTS:
    - AdaptiveState tracks recent performance
    - Behavior adapts based on NORMAL/CAUTION/DEFENSIVE mode
    - No threshold changes, no strategy changes
    
    LOCKED THRESHOLDS (DO NOT CHANGE):
    - ML_CONF_THRESHOLD = 0.55
    - ML_HIGH_CONF = 0.70
    - RSI_EXTREME_LOW = 25
    - RSI_EXTREME_HIGH = 75
    """
    
    # ========== LOCKED THRESHOLDS (DO NOT CHANGE) ==========
    ML_CONF_THRESHOLD = 0.55
    ML_HIGH_CONF = 0.70
    RSI_EXTREME_LOW = 25
    RSI_EXTREME_HIGH = 75
    RSI_NEUTRAL_SIZE = 0.85
    
    # ========== LOCKED CONSTANTS ==========
    MIN_BARS_BETWEEN_TRADES = 12
    ATR_SL_MULT = 1.5
    ATR_TP_RR = 1.5
    MIN_RR = 1.41
    MARK2_MEMORY_FLOOR = 0.20
    MARK2_MEMORY_DECAY = 0.95
    MARK2_MEMORY_RECOVERY = 1.02
    
    def __init__(self, initial_equity: float = 10000.0, model_path: Optional[Path] = None):
        self.state = ExecutionState()
        self.state.equity = initial_equity
        self.state.initial_equity = initial_equity
        self.state.peak_equity = initial_equity
        self.trades: List[TradeRecord] = []
        self._trade_counter = 0
        
        # ML Brain (instantiated once)
        self.ml_brain = MLBrain(model_path)
        
        # ============================================
        # PHASE-3: ADAPTIVE STATE (instantiated once)
        # ============================================
        self.adaptive_state = AdaptiveState()
        
        # ============================================
        # PHASE-4: QUALITY SCORER (Option 2 Upgrade)
        # ============================================
        self.quality_scorer = None
        if QUALITY_SCORER_AVAILABLE:
            self.quality_scorer = QualityScorer()
            logger.info(f"[ExecutionCore] Phase-4: QualityScorer enabled")
        
        logger.info(f"[ExecutionCore] ML loaded: {self.ml_brain.model_loaded}")
        logger.info(f"[ExecutionCore] Phase-3: AdaptiveState enabled")
    
    def reset(self, initial_equity: float = 10000.0):
        """Reset state for new run."""
        self.state = ExecutionState()
        self.state.equity = initial_equity
        self.state.initial_equity = initial_equity
        self.state.peak_equity = initial_equity
        self.trades = []
        self._trade_counter = 0
        self.adaptive_state = AdaptiveState()
    
    # ========================================================================
    # SINGLE ENTRY POINT
    # ========================================================================
    
    def on_candle(
        self,
        timestamp: Any,
        symbol: str,
        open_price: float,
        high_price: float,
        low_price: float,
        close_price: float,
        candle_index: int,
        features: Dict,
        regime: str,
        mode: str = "FAST",
    ) -> Optional[TradeRecord]:
        """
        SINGLE ENTRY POINT for both SHADOW and FAST modes.
        
        PHASE-3 DECISION FLOW:
        1. Check SL/TP
        2. DANGER block
        3. MARK-2 gate
        4. ML inference
        5. → ADAPTIVE STATE CHECK (NEW) ←
        6. RSI confirmation
        7. Risk calculation (with adaptive size)
        8. Open position
        """
        self.state.current_candle_index = candle_index
        
        # === STEP 1: Check SL/TP ===
        closed_trade = None
        if self.state.position_open:
            closed_trade = self._check_sl_tp_intrabar(
                high=high_price,
                low=low_price,
                timestamp=timestamp,
                candle_index=candle_index,
            )
        
        # === STEP 2: DANGER block ===
        if regime == "DANGER":
            return closed_trade
        
        # === STEP 3: MARK-2 gate ===
        if not self._apply_mark2_gate(candle_index):
            return closed_trade
        
        # === STEP 4: Check if in position ===
        if self.state.position_open:
            return closed_trade
        
        # === STEP 5: ML inference ===
        ml_signal, ml_confidence = self.ml_brain.predict(features)
        
        # ============================================
        # STEP 5.5: QUALITY SCORING (PHASE-4)
        # ============================================
        quality_size_mod = 1.0
        quality_grade = 'B'  # Default if scorer not available
        
        if self.quality_scorer is not None and ml_signal in ['BUY', 'SELL']:
            quality_result = self.quality_scorer.calculate(
                features=features,
                signal=ml_signal,
                regime=regime,
                confidence=ml_confidence,
            )
            quality_grade = quality_result.grade
            quality_size_mod = quality_result.size_multiplier
            
            # Skip if quality is F (very poor setup)
            if quality_grade == 'F' or not self.quality_scorer.should_trade(quality_result):
                logger.debug(f"[QUALITY] Skipping {ml_signal}: score={quality_result.total_score:.1f}, grade={quality_grade}")
                return closed_trade
        
        # ============================================
        # STEP 6: ADAPTIVE STATE CHECK (PHASE-3)
        # ============================================
        adaptive_mode = self.adaptive_state.get_mode()
        
        # Apply adaptive behavior BEFORE RSI check
        signal_after_adaptive, adaptive_size_mod = self._apply_adaptive_rules(
            ml_signal=ml_signal,
            ml_confidence=ml_confidence,
            adaptive_mode=adaptive_mode,
            regime=regime,
        )
        
        if signal_after_adaptive == "HOLD":
            return closed_trade
        
        # === STEP 7: RSI confirmation (unchanged logic) ===
        final_signal, decision_source, rsi_size_mod = self._evaluate_signal_with_rsi(
            ml_signal=signal_after_adaptive,
            ml_confidence=ml_confidence,
            features=features,
            regime=regime,
        )
        
        if final_signal == "HOLD":
            return closed_trade
        
        # === STEP 8: Risk calculation ===
        atr = features.get("atr_14", features.get("atr", 0.001))
        if atr < 0.0001:
            return closed_trade
        
        sl, tp, size = self._calculate_risk(
            entry=close_price,
            side=final_signal,
            atr=atr,
            regime=regime,
        )
        
        # Apply ALL size modifiers (quality + adaptive + RSI)
        size *= quality_size_mod * adaptive_size_mod * rsi_size_mod
        
        # Log quality info if available
        if self.quality_scorer is not None:
            logger.debug(f"[QUALITY] {final_signal}: grade={quality_grade}, size_mod={quality_size_mod:.2f}")
        
        # === STEP 9: RR gate ===
        sl_dist = abs(close_price - sl)
        tp_dist = abs(close_price - tp)
        rr = tp_dist / sl_dist if sl_dist > 0.00001 else 0
        if rr < self.MIN_RR:
            return closed_trade
        
        # === STEP 10: Open position ===
        self._open_position(
            side=final_signal,
            entry=close_price,
            size=size,
            sl=sl,
            tp=tp,
            candle_index=candle_index,
            timestamp=timestamp,
            regime=regime,
            symbol=symbol,
            ml_confidence=ml_confidence,
            decision_source=f"{decision_source}|{adaptive_mode.value}",
            adaptive_mode=adaptive_mode.value,
        )
        
        self._update_mark2_after_entry(candle_index)
        
        return closed_trade
    
    # ========================================================================
    # PHASE-3: ADAPTIVE RULES
    # ========================================================================
    
    def _apply_adaptive_rules(
        self,
        ml_signal: str,
        ml_confidence: float,
        adaptive_mode: AdaptiveMode,
        regime: str,
    ) -> Tuple[str, float]:
        """
        Apply adaptive behavior based on recent performance.
        
        RULES:
        🟢 NORMAL: Allow all (size_mod = 1.0)
        🟡 CAUTION: Require HIGH_CONF, reduce size ×0.8
        🔴 DEFENSIVE: Require HIGH_CONF + stable regime
        
        Returns:
            (signal, size_modifier)
        """
        # 🟢 NORMAL: Full participation
        if adaptive_mode == AdaptiveMode.NORMAL:
            if ml_confidence >= self.ML_CONF_THRESHOLD:
                return ml_signal, 1.0
            return "HOLD", 1.0
        
        # 🟡 CAUTION: Stricter requirements
        if adaptive_mode == AdaptiveMode.CAUTION:
            # Require ML_HIGH_CONF instead of ML_CONF_THRESHOLD
            if ml_confidence >= self.ML_HIGH_CONF:
                return ml_signal, 0.8  # Reduced size
            return "HOLD", 1.0
        
        # 🔴 DEFENSIVE: Most conservative
        if adaptive_mode == AdaptiveMode.DEFENSIVE:
            # Require ML_HIGH_CONF + stable regime
            if ml_confidence >= self.ML_HIGH_CONF and regime in ["RANGE", "TREND"]:
                return ml_signal, 0.6  # Significantly reduced size
            return "HOLD", 1.0
        
        return ml_signal, 1.0
    
    # ========================================================================
    # RSI CONFIRMATION (UNCHANGED)
    # ========================================================================
    
    def _evaluate_signal_with_rsi(
        self,
        ml_signal: str,
        ml_confidence: float,
        features: Dict,
        regime: str,
    ) -> Tuple[str, str, float]:
        """RSI confirmation logic (UNCHANGED from Phase-2)."""
        rsi = features.get("rsi_14", features.get("rsi", 50))
        
        # High confidence: skip RSI check
        if ml_confidence >= self.ML_HIGH_CONF:
            return ml_signal, "ML_HIGH_CONF", 1.0
        
        # RANGE regime: RSI confirmation
        if regime == "RANGE":
            if ml_signal == "BUY":
                if rsi >= self.RSI_EXTREME_HIGH:
                    return "HOLD", "RSI_BLOCK", 1.0
                elif rsi <= self.RSI_EXTREME_LOW:
                    return ml_signal, "ML+RSI_CONFIRM", 1.0
                else:
                    return ml_signal, "ML_RSI_NEUTRAL", self.RSI_NEUTRAL_SIZE
            elif ml_signal == "SELL":
                if rsi <= self.RSI_EXTREME_LOW:
                    return "HOLD", "RSI_BLOCK", 1.0
                elif rsi >= self.RSI_EXTREME_HIGH:
                    return ml_signal, "ML+RSI_CONFIRM", 1.0
                else:
                    return ml_signal, "ML_RSI_NEUTRAL", self.RSI_NEUTRAL_SIZE
        
        # TREND: minimal RSI influence
        return ml_signal, "ML_TREND", 1.0
    
    # ========================================================================
    # MARK-2 GATE (UNCHANGED)
    # ========================================================================
    
    def _apply_mark2_gate(self, candle_index: int) -> bool:
        if candle_index < self.state.mark2_cooldown_until:
            return False
        if candle_index - self.state.last_trade_index < self.MIN_BARS_BETWEEN_TRADES:
            return False
        if self.state.mark2_memory < self.MARK2_MEMORY_FLOOR:
            return False
        return True
    
    # ========================================================================
    # RISK BRAIN (UNCHANGED)
    # ========================================================================
    
    def _calculate_risk(
        self,
        entry: float,
        side: str,
        atr: float,
        regime: str,
    ) -> Tuple[float, float, float]:
        regime_mult = {"RANGE": 0.7, "TREND": 1.0}.get(regime, 0.8)
        sl_distance = atr * self.ATR_SL_MULT * regime_mult
        tp_distance = sl_distance * self.ATR_TP_RR
        
        if side == "BUY":
            sl = entry - sl_distance
            tp = entry + tp_distance
        else:
            sl = entry + sl_distance
            tp = entry - tp_distance
        
        risk_pct = 0.01 * self.state.mark2_memory
        risk_amount = self.state.equity * risk_pct
        size = risk_amount / sl_distance if sl_distance > 0 else 0
        
        return sl, tp, size
    
    # ========================================================================
    # SL/TP CHECK (UNCHANGED)
    # ========================================================================
    
    def _check_sl_tp_intrabar(
        self,
        high: float,
        low: float,
        timestamp: Any,
        candle_index: int,
    ) -> Optional[TradeRecord]:
        if not self.state.position_open:
            return None
        
        sl = self.state.position_sl
        tp = self.state.position_tp
        side = self.state.position_side
        
        exit_price = None
        exit_reason = None
        
        if side == "BUY":
            if low <= sl:
                exit_price = sl
                exit_reason = "SL_HIT"
            elif high >= tp:
                exit_price = tp
                exit_reason = "TP_HIT"
        else:
            if high >= sl:
                exit_price = sl
                exit_reason = "SL_HIT"
            elif low <= tp:
                exit_price = tp
                exit_reason = "TP_HIT"
        
        if exit_price is not None:
            return self._close_position(
                exit_price=exit_price,
                exit_reason=exit_reason,
                candle_index=candle_index,
                timestamp=timestamp,
            )
        
        return None
    
    # ========================================================================
    # POSITION MANAGEMENT
    # ========================================================================
    
    def _open_position(
        self,
        side: str,
        entry: float,
        size: float,
        sl: float,
        tp: float,
        candle_index: int,
        timestamp: Any,
        regime: str,
        symbol: str,
        ml_confidence: float,
        decision_source: str,
        adaptive_mode: str,
    ):
        self.state.position_open = True
        self.state.position_side = side
        self.state.position_entry = entry
        self.state.position_size = size
        self.state.position_sl = sl
        self.state.position_tp = tp
        self.state.position_entry_index = candle_index
        self.state.position_entry_time = timestamp
        self.state.position_regime = regime
        self.state.position_symbol = symbol
        self.state.position_ml_confidence = ml_confidence
        self.state.position_decision_source = decision_source
        self.state.position_adaptive_mode = adaptive_mode
        
        self._trade_counter += 1
    
    def _close_position(
        self,
        exit_price: float,
        exit_reason: str,
        candle_index: int,
        timestamp: Any,
    ) -> TradeRecord:
        side = self.state.position_side
        entry = self.state.position_entry
        size = self.state.position_size
        
        # Direction-aware PnL
        if side == "BUY":
            pnl = (exit_price - entry) * size
        else:
            pnl = (entry - exit_price) * size
        
        is_win = pnl > 0
        sl_dist = abs(entry - self.state.position_sl)
        r_mult = pnl / (sl_dist * size) if sl_dist > 0 and size > 0 else 0
        
        trade = TradeRecord(
            trade_id=f"T{self._trade_counter:06d}",
            symbol=self.state.position_symbol,
            entry_timestamp=self.state.position_entry_time,
            exit_timestamp=timestamp,
            entry_candle_index=self.state.position_entry_index,
            exit_candle_index=candle_index,
            side=side,
            entry_price=entry,
            exit_price=exit_price,
            size=size,
            sl_price=self.state.position_sl,
            tp_price=self.state.position_tp,
            pnl=pnl,
            r_multiple=r_mult,
            exit_reason=exit_reason,
            is_win=is_win,
            regime=self.state.position_regime,
            ml_confidence=self.state.position_ml_confidence,
            decision_source=self.state.position_decision_source,
            adaptive_mode=self.state.position_adaptive_mode,
        )
        
        # Sanity check
        self._assert_pnl_sanity(trade)
        
        self.trades.append(trade)
        
        # ============================================
        # PHASE-3: UPDATE ADAPTIVE STATE AFTER CLOSE
        # ============================================
        self.adaptive_state.update(
            is_win=is_win,
            regime=self.state.position_regime,
        )
        
        # Update equity
        self.state.equity += pnl
        
        if self.state.equity > self.state.peak_equity:
            self.state.peak_equity = self.state.equity
        else:
            dd = (self.state.peak_equity - self.state.equity) / self.state.peak_equity
            if dd > self.state.max_drawdown:
                self.state.max_drawdown = dd
        
        # Reset position
        self.state.position_open = False
        self.state.last_trade_index = candle_index
        
        # MARK-2 update
        if is_win:
            self.state.consecutive_losses = 0
            self._update_mark2_after_win()
        else:
            self.state.consecutive_losses += 1
            self._update_mark2_after_loss(candle_index)
        
        return trade
    
    # ========================================================================
    # SANITY (UNCHANGED)
    # ========================================================================
    
    def _assert_pnl_sanity(self, trade: TradeRecord):
        if trade.exit_reason == "SL_HIT" and trade.pnl >= 0:
            raise TradeSanityViolation(f"SL_HIT with positive PnL: {trade.pnl}")
        if trade.exit_reason == "TP_HIT" and trade.pnl <= 0:
            raise TradeSanityViolation(f"TP_HIT with negative PnL: {trade.pnl}")
    
    # ========================================================================
    # MARK-2 UPDATES (UNCHANGED)
    # ========================================================================
    
    def _update_mark2_after_entry(self, candle_index: int):
        self.state.mark2_cooldown_until = candle_index + self.MIN_BARS_BETWEEN_TRADES
    
    def _update_mark2_after_win(self):
        self.state.mark2_memory = min(1.0, self.state.mark2_memory * self.MARK2_MEMORY_RECOVERY)
    
    def _update_mark2_after_loss(self, candle_index: int):
        self.state.mark2_memory *= self.MARK2_MEMORY_DECAY
        if self.state.consecutive_losses >= 2:
            self.state.mark2_cooldown_until = candle_index + self.MIN_BARS_BETWEEN_TRADES * 2
    
    # ========================================================================
    # RESULTS
    # ========================================================================
    
    def get_results(self) -> Dict:
        total = len(self.trades)
        wins = sum(1 for t in self.trades if t.is_win)
        
        # Mode breakdown
        mode_counts = {"NORMAL": 0, "CAUTION": 0, "DEFENSIVE": 0}
        for t in self.trades:
            m = t.adaptive_mode
            if m in mode_counts:
                mode_counts[m] += 1
        
        return {
            "initial_equity": self.state.initial_equity,
            "final_equity": self.state.equity,
            "total_return": (self.state.equity - self.state.initial_equity) / self.state.initial_equity,
            "total_trades": total,
            "winning_trades": wins,
            "losing_trades": total - wins,
            "winrate": wins / total if total > 0 else 0,
            "max_drawdown": self.state.max_drawdown,
            "ml_model_loaded": self.ml_brain.model_loaded,
            "adaptive_mode_breakdown": mode_counts,
            "trades": [t.to_dict() for t in self.trades],
        }
