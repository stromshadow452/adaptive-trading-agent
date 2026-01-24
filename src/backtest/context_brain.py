"""
Context Brain V1 - Market Context Encoder and Pattern Detection

Sits between Features (Stage 2) and ML Brain (Stage 3) in the pipeline.
Provides probabilistic confidence instead of hard yes/no signals.

Components:
    - SRDetector: Support/Resistance detection from swing points
    - BreakoutClassifier: False breakout scoring
    - ManipulationDetector: Trap signature detection
    - ContextBrain: Main orchestrator

Author: SCOPUS Adaptive Trading System
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class SRLevel:
    """Represents a Support or Resistance level."""
    price: float
    strength: float  # 0.0 to 1.0 based on touch count
    level_type: str  # "support" or "resistance"
    touches: int
    last_touch_idx: int


@dataclass
class ContextOutput:
    """Output schema from Context Brain."""
    context_confidence: float       # 0.0 to 1.0 - overall tradability
    market_state: str               # "expansion" | "contraction" | "breakout" | "reversal"
    sr_proximity: float             # 0.0 (far) to 1.0 (at major level)
    breakout_quality: float         # -1.0 (false) to +1.0 (confirmed)
    manipulation_risk: float        # 0.0 (clean) to 1.0 (likely trap)
    data_sufficiency: float         # 0.0 (sparse) to 1.0 (rich)
    operating_mode: str             # "LEARNING" | "CONFIRMATION"
    exploration_budget: float       # 0.0 to 0.3 - allowed risk for learning
    nearest_support: Optional[float] = None
    nearest_resistance: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for pipeline compatibility."""
        return {
            "context_confidence": round(self.context_confidence, 4),
            "market_state": self.market_state,
            "sr_proximity": round(self.sr_proximity, 4),
            "breakout_quality": round(self.breakout_quality, 4),
            "manipulation_risk": round(self.manipulation_risk, 4),
            "data_sufficiency": round(self.data_sufficiency, 4),
            "operating_mode": self.operating_mode,
            "exploration_budget": round(self.exploration_budget, 4),
            "nearest_support": round(self.nearest_support, 5) if self.nearest_support else None,
            "nearest_resistance": round(self.nearest_resistance, 5) if self.nearest_resistance else None,
        }


# =============================================================================
# Support/Resistance Detector
# =============================================================================

class SRDetector:
    """
    Detects Support and Resistance levels from OHLCV data.
    
    Method:
    1. Find swing highs: high[i] > max(high[i-n:i], high[i+1:i+n+1])
    2. Find swing lows: low[i] < min(low[i-n:i], low[i+1:i+n+1])
    3. Cluster nearby levels within ATR tolerance
    4. Score by touch count
    """
    
    def __init__(self, swing_lookback: int = 3, cluster_atr_mult: float = 0.5, max_levels: int = 5):
        self.swing_lookback = swing_lookback
        self.cluster_atr_mult = cluster_atr_mult
        self.max_levels = max_levels
    
    def detect(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, 
               atr: float, lookback: int = 50) -> Tuple[List[SRLevel], List[SRLevel]]:
        """
        Detect S/R levels from price data.
        
        Args:
            highs: Array of high prices
            lows: Array of low prices
            closes: Array of close prices
            atr: Current ATR value for clustering tolerance
            lookback: Number of bars to analyze
            
        Returns:
            (support_levels, resistance_levels)
        """
        if len(highs) < lookback:
            lookback = len(highs)
        
        if lookback < self.swing_lookback * 2 + 1:
            logger.debug("Insufficient data for S/R detection")
            return [], []
        
        # Use recent data
        h = highs[-lookback:]
        l = lows[-lookback:]
        c = closes[-lookback:]
        n = self.swing_lookback
        
        swing_highs = []
        swing_lows = []
        
        # Find swing points
        for i in range(n, len(h) - n):
            # Swing high: higher than n bars before and after
            if h[i] == max(h[i-n:i+n+1]):
                swing_highs.append((h[i], i))
            
            # Swing low: lower than n bars before and after
            if l[i] == min(l[i-n:i+n+1]):
                swing_lows.append((l[i], i))
        
        # Cluster nearby levels
        tolerance = atr * self.cluster_atr_mult
        
        resistance_levels = self._cluster_levels(swing_highs, tolerance, "resistance", lookback)
        support_levels = self._cluster_levels(swing_lows, tolerance, "support", lookback)
        
        # Sort by strength and limit
        resistance_levels.sort(key=lambda x: x.strength, reverse=True)
        support_levels.sort(key=lambda x: x.strength, reverse=True)
        
        return support_levels[:self.max_levels], resistance_levels[:self.max_levels]
    
    def _cluster_levels(self, points: List[Tuple[float, int]], tolerance: float, 
                        level_type: str, total_bars: int) -> List[SRLevel]:
        """Cluster nearby price points into S/R levels."""
        if not points:
            return []
        
        # Sort by price
        points_sorted = sorted(points, key=lambda x: x[0])
        clusters = []
        current_cluster = [points_sorted[0]]
        
        for i in range(1, len(points_sorted)):
            if abs(points_sorted[i][0] - current_cluster[-1][0]) <= tolerance:
                current_cluster.append(points_sorted[i])
            else:
                clusters.append(current_cluster)
                current_cluster = [points_sorted[i]]
        clusters.append(current_cluster)
        
        # Convert clusters to SRLevel
        levels = []
        for cluster in clusters:
            prices = [p[0] for p in cluster]
            indices = [p[1] for p in cluster]
            avg_price = np.mean(prices)
            touches = len(cluster)
            # Strength: more touches = stronger, recency bonus
            recency = max(indices) / total_bars  # 0 to 1
            strength = min(1.0, (touches / 5) * 0.7 + recency * 0.3)
            
            levels.append(SRLevel(
                price=avg_price,
                strength=strength,
                level_type=level_type,
                touches=touches,
                last_touch_idx=max(indices)
            ))
        
        return levels
    
    def compute_proximity(self, current_price: float, support_levels: List[SRLevel], 
                          resistance_levels: List[SRLevel], atr: float) -> Tuple[float, Optional[float], Optional[float]]:
        """
        Compute proximity to nearest S/R level.
        
        Returns:
            (proximity_score, nearest_support, nearest_resistance)
            proximity_score: 0.0 (far) to 1.0 (at level)
        """
        nearest_support = None
        nearest_resistance = None
        min_distance = float('inf')
        
        for level in support_levels:
            if level.price < current_price:
                dist = (current_price - level.price) / atr
                if dist < min_distance:
                    min_distance = dist
                    nearest_support = level.price
        
        for level in resistance_levels:
            if level.price > current_price:
                dist = (level.price - current_price) / atr
                if dist < min_distance:
                    min_distance = dist
                    nearest_resistance = level.price
        
        # Convert distance to proximity (0 = far, 1 = at level)
        # Within 0.5 ATR = very close (0.8+), beyond 3 ATR = far (< 0.2)
        if min_distance == float('inf'):
            proximity = 0.0
        else:
            proximity = max(0.0, min(1.0, 1.0 - (min_distance / 3.0)))
        
        return proximity, nearest_support, nearest_resistance


# =============================================================================
# Breakout Classifier
# =============================================================================

class BreakoutClassifier:
    """
    Classifies breakout quality from -1.0 (false) to +1.0 (confirmed).
    
    Signals:
    - Price breaches level by > 0.5 ATR: potential breakout
    - Returns within 3 bars: FALSE breakout (-1.0)
    - Holds beyond level for 5+ bars: CONFIRMED (+1.0)
    - Immediate rejection (same bar close back): TRAP (-0.8)
    """
    
    def __init__(self, breach_threshold_atr: float = 0.5, confirm_bars: int = 5, 
                 reject_bars: int = 3):
        self.breach_threshold_atr = breach_threshold_atr
        self.confirm_bars = confirm_bars
        self.reject_bars = reject_bars
    
    def classify(self, closes: np.ndarray, highs: np.ndarray, lows: np.ndarray,
                 sr_level: float, level_type: str, atr: float) -> float:
        """
        Score breakout quality for a given S/R level.
        
        Args:
            closes: Recent close prices (newest last)
            highs: Recent high prices
            lows: Recent low prices
            sr_level: The S/R level price
            level_type: "support" or "resistance"
            atr: Current ATR
            
        Returns:
            breakout_quality: -1.0 (false) to +1.0 (confirmed)
        """
        if len(closes) < self.confirm_bars:
            return 0.0  # Insufficient data
        
        threshold = atr * self.breach_threshold_atr
        current_close = closes[-1]
        
        if level_type == "resistance":
            # Check for resistance breakout
            breach_detected = False
            breach_idx = None
            
            for i in range(len(highs) - self.confirm_bars, len(highs)):
                if highs[i] > sr_level + threshold:
                    breach_detected = True
                    breach_idx = i
                    break
            
            if not breach_detected:
                return 0.0  # No breakout attempt
            
            bars_since_breach = len(closes) - 1 - breach_idx
            
            # Check if price held above level
            closes_since = closes[breach_idx:]
            if all(c > sr_level for c in closes_since) and bars_since_breach >= self.confirm_bars:
                return 1.0  # Confirmed breakout
            elif current_close < sr_level:
                # Failed - price returned below level
                return -1.0 if bars_since_breach <= self.reject_bars else -0.5
            else:
                # Still developing
                hold_ratio = bars_since_breach / self.confirm_bars
                return min(0.8, hold_ratio * 0.6)
        
        else:  # support breakdown
            breach_detected = False
            breach_idx = None
            
            for i in range(len(lows) - self.confirm_bars, len(lows)):
                if lows[i] < sr_level - threshold:
                    breach_detected = True
                    breach_idx = i
                    break
            
            if not breach_detected:
                return 0.0
            
            bars_since_breach = len(closes) - 1 - breach_idx
            closes_since = closes[breach_idx:]
            
            if all(c < sr_level for c in closes_since) and bars_since_breach >= self.confirm_bars:
                return 1.0
            elif current_close > sr_level:
                return -1.0 if bars_since_breach <= self.reject_bars else -0.5
            else:
                hold_ratio = bars_since_breach / self.confirm_bars
                return min(0.8, hold_ratio * 0.6)


# =============================================================================
# Manipulation Detector
# =============================================================================

class ManipulationDetector:
    """
    Detects manipulation-like price patterns.
    
    Patterns:
    1. Stop Hunt: price spikes beyond range then reverses > 70%
    2. Liquidity Sweep: quick wick through S/R then close inside range
    3. Volume Anomaly: low volume on breakout (if volume available)
    4. Wick Ratio: extreme wick signals rejection
    """
    
    def __init__(self, stop_hunt_reversal: float = 0.7, wick_ratio_threshold: float = 2.0):
        self.stop_hunt_reversal = stop_hunt_reversal
        self.wick_ratio_threshold = wick_ratio_threshold
    
    def detect(self, candle: pd.Series, recent_candles: pd.DataFrame, 
               sr_levels: List[SRLevel], atr: float) -> float:
        """
        Score manipulation risk: 0.0 (clean) to 1.0 (likely trap).
        
        Args:
            candle: Current candle (Series with open, high, low, close)
            recent_candles: Recent OHLC DataFrame
            sr_levels: Combined S/R levels
            atr: Current ATR
            
        Returns:
            manipulation_risk: 0.0 to 1.0
        """
        scores = []
        
        # 1. Wick rejection score
        wick_score = self._wick_rejection_score(candle)
        scores.append(('wick', wick_score, 0.25))
        
        # 2. Stop hunt score
        if len(recent_candles) >= 10:
            stop_hunt = self._stop_hunt_score(candle, recent_candles, atr)
            scores.append(('stop_hunt', stop_hunt, 0.30))
        
        # 3. Liquidity sweep score
        if sr_levels:
            liq_sweep = self._liquidity_sweep_score(candle, sr_levels, atr)
            scores.append(('liq_sweep', liq_sweep, 0.30))
        
        # 4. Reversal candle pattern
        reversal = self._reversal_candle_score(candle, recent_candles)
        scores.append(('reversal', reversal, 0.15))
        
        # Weighted average
        total_weight = sum(s[2] for s in scores)
        if total_weight == 0:
            return 0.0
        
        risk = sum(s[1] * s[2] for s in scores) / total_weight
        return min(1.0, max(0.0, risk))
    
    def _wick_rejection_score(self, candle: pd.Series) -> float:
        """Score based on wick-to-body ratio."""
        o = candle.get('open', candle.get('Open', 0))
        h = candle.get('high', candle.get('High', 0))
        l = candle.get('low', candle.get('Low', 0))
        c = candle.get('close', candle.get('Close', 0))
        
        body = abs(c - o)
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l
        
        if body < 1e-8:
            body = 1e-8  # Avoid division by zero
        
        max_wick = max(upper_wick, lower_wick)
        wick_ratio = max_wick / body
        
        # Score: ratio > 2.0 = high rejection (score ~0.8)
        if wick_ratio > self.wick_ratio_threshold:
            return min(1.0, (wick_ratio - self.wick_ratio_threshold) / 3.0 + 0.5)
        return wick_ratio / (self.wick_ratio_threshold * 2)
    
    def _stop_hunt_score(self, candle: pd.Series, recent: pd.DataFrame, atr: float) -> float:
        """Detect stop hunt pattern."""
        h = candle.get('high', candle.get('High', 0))
        l = candle.get('low', candle.get('Low', 0))
        c = candle.get('close', candle.get('Close', 0))
        o = candle.get('open', candle.get('Open', 0))
        
        # Get recent range
        recent_high = recent['high'].max() if 'high' in recent.columns else recent['High'].max()
        recent_low = recent['low'].min() if 'low' in recent.columns else recent['Low'].min()
        
        # Check for spike beyond range with reversal
        spike_high = h > recent_high
        spike_low = l < recent_low
        
        candle_range = h - l
        if candle_range < 1e-8:
            return 0.0
        
        # Reversal ratio: how much of the spike was retraced
        if spike_high and c < o:  # Bearish close after high spike
            reversal = (h - c) / candle_range
            if reversal > self.stop_hunt_reversal:
                return min(1.0, reversal)
        
        if spike_low and c > o:  # Bullish close after low spike
            reversal = (c - l) / candle_range
            if reversal > self.stop_hunt_reversal:
                return min(1.0, reversal)
        
        return 0.0
    
    def _liquidity_sweep_score(self, candle: pd.Series, sr_levels: List[SRLevel], 
                                atr: float) -> float:
        """Detect liquidity sweep through S/R."""
        h = candle.get('high', candle.get('High', 0))
        l = candle.get('low', candle.get('Low', 0))
        c = candle.get('close', candle.get('Close', 0))
        o = candle.get('open', candle.get('Open', 0))
        
        body_high = max(o, c)
        body_low = min(o, c)
        
        for level in sr_levels:
            price = level.price
            
            # Wick pierced level but body stayed inside
            if level.level_type == "resistance":
                if h > price and body_high < price:
                    # Pierced resistance with wick, closed below
                    pierce_depth = (h - price) / atr
                    return min(1.0, 0.5 + pierce_depth * 0.5)
            else:  # support
                if l < price and body_low > price:
                    pierce_depth = (price - l) / atr
                    return min(1.0, 0.5 + pierce_depth * 0.5)
        
        return 0.0
    
    def _reversal_candle_score(self, candle: pd.Series, recent: pd.DataFrame) -> float:
        """Score reversal candle patterns (engulfing, pin bar)."""
        if len(recent) < 2:
            return 0.0
        
        c = candle.get('close', candle.get('Close', 0))
        o = candle.get('open', candle.get('Open', 0))
        h = candle.get('high', candle.get('High', 0))
        l = candle.get('low', candle.get('Low', 0))
        
        prev = recent.iloc[-1]
        prev_c = prev.get('close', prev.get('Close', 0))
        prev_o = prev.get('open', prev.get('Open', 0))
        
        current_bullish = c > o
        prev_bullish = prev_c > prev_o
        
        # Engulfing pattern
        if current_bullish and not prev_bullish:
            if o < prev_c and c > prev_o:
                return 0.7  # Bullish engulfing after bearish
        elif not current_bullish and prev_bullish:
            if o > prev_c and c < prev_o:
                return 0.7  # Bearish engulfing after bullish
        
        return 0.0


# =============================================================================
# Market State Classifier
# =============================================================================

class MarketStateClassifier:
    """Classifies market into expansion/contraction/breakout/reversal states."""
    
    def __init__(self, expansion_threshold: float = 1.2, contraction_threshold: float = 0.8):
        self.expansion_threshold = expansion_threshold
        self.contraction_threshold = contraction_threshold
    
    def classify(self, atr: float, atr_ma: float, breakout_quality: float, 
                 manipulation_risk: float) -> str:
        """
        Determine current market state.
        
        Args:
            atr: Current ATR
            atr_ma: Moving average of ATR (e.g., 20-period)
            breakout_quality: From BreakoutClassifier
            manipulation_risk: From ManipulationDetector
            
        Returns:
            "expansion" | "contraction" | "breakout" | "reversal"
        """
        atr_ratio = atr / atr_ma if atr_ma > 0 else 1.0
        
        # Breakout state
        if breakout_quality > 0.6:
            return "breakout"
        
        # Reversal state (high manipulation + expansion)
        if manipulation_risk > 0.6 and atr_ratio > self.expansion_threshold:
            return "reversal"
        
        # Expansion state
        if atr_ratio > self.expansion_threshold:
            return "expansion"
        
        # Contraction state
        if atr_ratio < self.contraction_threshold:
            return "contraction"
        
        return "expansion"  # Default


# =============================================================================
# Context Brain - Main Orchestrator
# =============================================================================

class ContextBrain:
    """
    Main Context Brain orchestrator.
    
    Sits between Features (Stage 2) and ML Brain (Stage 3).
    Outputs probabilistic confidence and market context.
    """
    
    # Data sufficiency thresholds
    MIN_BARS_LEARNING = 20      # Minimum for any operation
    BARS_LEARNING_FULL = 100    # Full LEARNING mode capacity
    BARS_CONFIRMATION = 500     # Switch to CONFIRMATION mode
    
    # Exploration budget
    MAX_EXPLORATION_BUDGET = 0.30  # 30% of normal size in LEARNING
    EXPLORATION_TRADES_PER_DAY = 2
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # Initialize components
        self.sr_detector = SRDetector(
            swing_lookback=self.config.get('sr_swing_lookback', 3),
            cluster_atr_mult=self.config.get('sr_cluster_atr', 0.5),
            max_levels=self.config.get('sr_max_levels', 5)
        )
        self.breakout_classifier = BreakoutClassifier(
            breach_threshold_atr=self.config.get('breakout_breach_atr', 0.5),
            confirm_bars=self.config.get('breakout_confirm_bars', 5)
        )
        self.manipulation_detector = ManipulationDetector(
            stop_hunt_reversal=self.config.get('stop_hunt_reversal', 0.7),
            wick_ratio_threshold=self.config.get('wick_ratio_threshold', 2.0)
        )
        self.market_classifier = MarketStateClassifier()
        
        # State tracking
        self.exploration_trades_today = 0
        self.current_date = None
        
        logger.info("Context Brain initialized")
    
    def analyze(self, candle: pd.Series, ohlcv_df: pd.DataFrame, 
                atr: float, atr_ma: Optional[float] = None) -> ContextOutput:
        """
        Analyze market context for current candle.
        
        Args:
            candle: Current OHLCV candle
            ohlcv_df: Historical OHLCV DataFrame
            atr: Current ATR value
            atr_ma: Optional ATR moving average
            
        Returns:
            ContextOutput with all context metrics
        """
        # Extract arrays
        highs = ohlcv_df['high'].values if 'high' in ohlcv_df.columns else ohlcv_df['High'].values
        lows = ohlcv_df['low'].values if 'low' in ohlcv_df.columns else ohlcv_df['Low'].values
        closes = ohlcv_df['close'].values if 'close' in ohlcv_df.columns else ohlcv_df['Close'].values
        
        n_bars = len(closes)
        current_price = closes[-1] if len(closes) > 0 else candle.get('close', candle.get('Close', 0))
        
        # 1. Determine operating mode and data sufficiency
        operating_mode, data_sufficiency = self._compute_mode(n_bars)
        
        # 2. Detect S/R levels
        support_levels, resistance_levels = self.sr_detector.detect(
            highs, lows, closes, atr, lookback=min(100, n_bars)
        )
        
        # 3. Compute S/R proximity
        sr_proximity, nearest_support, nearest_resistance = self.sr_detector.compute_proximity(
            current_price, support_levels, resistance_levels, atr
        )
        
        # 4. Classify breakout quality
        all_levels = support_levels + resistance_levels
        breakout_quality = 0.0
        if all_levels:
            # Find most relevant level (highest strength, closest)
            relevant_level = max(all_levels, key=lambda x: x.strength)
            breakout_quality = self.breakout_classifier.classify(
                closes[-20:] if len(closes) >= 20 else closes,
                highs[-20:] if len(highs) >= 20 else highs,
                lows[-20:] if len(lows) >= 20 else lows,
                relevant_level.price,
                relevant_level.level_type,
                atr
            )
        
        # 5. Detect manipulation risk
        manipulation_risk = self.manipulation_detector.detect(
            candle, ohlcv_df.iloc[-10:] if len(ohlcv_df) >= 10 else ohlcv_df,
            all_levels, atr
        )
        
        # 6. Classify market state
        atr_ma_val = atr_ma if atr_ma else atr
        market_state = self.market_classifier.classify(
            atr, atr_ma_val, breakout_quality, manipulation_risk
        )
        
        # 7. Compute exploration budget
        exploration_budget = self._compute_exploration_budget(operating_mode, data_sufficiency)
        
        # 8. Compute overall context confidence
        context_confidence = self._compute_context_confidence(
            sr_proximity, breakout_quality, manipulation_risk, 
            data_sufficiency, market_state, operating_mode
        )
        
        return ContextOutput(
            context_confidence=context_confidence,
            market_state=market_state,
            sr_proximity=sr_proximity,
            breakout_quality=breakout_quality,
            manipulation_risk=manipulation_risk,
            data_sufficiency=data_sufficiency,
            operating_mode=operating_mode,
            exploration_budget=exploration_budget,
            nearest_support=nearest_support,
            nearest_resistance=nearest_resistance
        )
    
    def _compute_mode(self, n_bars: int) -> Tuple[str, float]:
        """Determine operating mode and data sufficiency."""
        if n_bars < self.MIN_BARS_LEARNING:
            return "LEARNING", 0.0
        
        if n_bars >= self.BARS_CONFIRMATION:
            sufficiency = 1.0
            mode = "CONFIRMATION"
        else:
            # Logarithmic scaling for smooth transition
            import math
            ratio = n_bars / self.BARS_LEARNING_FULL
            sufficiency = min(1.0, math.log(ratio + 1) / math.log(6))
            mode = "LEARNING" if sufficiency < 0.8 else "CONFIRMATION"
        
        return mode, sufficiency
    
    def _compute_exploration_budget(self, mode: str, data_sufficiency: float) -> float:
        """Compute allowed exploration budget for learning trades."""
        if mode == "CONFIRMATION":
            return 0.0
        
        # More budget when data is sparse
        budget = self.MAX_EXPLORATION_BUDGET * (1.0 - data_sufficiency * 0.5)
        return max(0.05, min(self.MAX_EXPLORATION_BUDGET, budget))
    
    def _compute_context_confidence(self, sr_proximity: float, breakout_quality: float,
                                     manipulation_risk: float, data_sufficiency: float,
                                     market_state: str, operating_mode: str) -> float:
        """
        Compute overall context confidence score.
        
        Higher is better for trading.
        """
        # Base score from breakout quality and low manipulation
        base = 0.5
        
        # Breakout quality contribution (good breakout = confidence boost)
        if breakout_quality > 0:
            base += breakout_quality * 0.2
        else:
            base += breakout_quality * 0.1  # Negative quality penalizes less
        
        # Manipulation penalty
        base -= manipulation_risk * 0.25
        
        # S/R proximity: moderate bump (trading near S/R can be good or bad)
        if market_state == "breakout":
            base += sr_proximity * 0.1  # At level + breakout = good
        elif manipulation_risk > 0.5:
            base -= sr_proximity * 0.1  # At level + manipulation = bad
        
        # Market state adjustments
        state_bonus = {
            "breakout": 0.15,
            "expansion": 0.10,
            "contraction": -0.05,
            "reversal": -0.10
        }
        base += state_bonus.get(market_state, 0)
        
        # Mode adjustment
        if operating_mode == "LEARNING":
            # Boost confidence floor in learning mode
            base = max(0.35, base)
        
        return max(0.0, min(1.0, base))
    
    def reset_daily_exploration(self, date: str):
        """Reset daily exploration trade counter."""
        if date != self.current_date:
            self.current_date = date
            self.exploration_trades_today = 0
    
    def can_explore(self) -> bool:
        """Check if exploration trade is allowed today."""
        return self.exploration_trades_today < self.EXPLORATION_TRADES_PER_DAY
    
    def record_exploration_trade(self):
        """Record an exploration trade."""
        self.exploration_trades_today += 1
