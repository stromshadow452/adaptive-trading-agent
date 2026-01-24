"""
Quality Scorer - Option 2 Upgrade
=================================

Scores each trading setup from 0-100 based on multiple quality factors.
Higher score = better quality setup = higher probability of success.

Integration:
- Runs AFTER ML Brain generates signal
- BEFORE position sizing
- Does NOT decide BUY/SELL (ML Brain does that)
- Does NOT control exact position size (Risk Brain does that)
- Only provides a QUALITY SCORE that influences final decision

Factors:
1. Trend Strength (30 points) - Is the trend clear and strong?
2. RSI Position (20 points) - Is RSI in favorable zone for this signal?
3. S/R Distance (20 points) - Is price far from support/resistance?
4. ATR Ratio (15 points) - Is volatility normal (not too high/low)?
5. Momentum Alignment (15 points) - Does MACD agree with signal?

Usage:
    scorer = QualityScorer()
    score = scorer.calculate(features, signal='BUY', regime='TREND')
    # score = 0-100
    
    # Size multiplier based on score
    if score >= 80:
        size_mult = 1.0   # Full size
    elif score >= 60:
        size_mult = 0.7   # Reduced size
    elif score >= 40:
        size_mult = 0.4   # Mini size
    else:
        skip_trade = True  # Don't trade
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class QualityBreakdown:
    """Breakdown of quality score components."""
    trend_score: float
    rsi_score: float
    sr_score: float
    atr_score: float
    momentum_score: float
    total_score: float
    grade: str  # A, B, C, D, F
    size_multiplier: float
    recommendation: str  # FULL, REDUCED, MINI, SKIP
    
    def to_dict(self) -> Dict:
        return {
            "trend_score": round(self.trend_score, 1),
            "rsi_score": round(self.rsi_score, 1),
            "sr_score": round(self.sr_score, 1),
            "atr_score": round(self.atr_score, 1),
            "momentum_score": round(self.momentum_score, 1),
            "total_score": round(self.total_score, 1),
            "grade": self.grade,
            "size_multiplier": self.size_multiplier,
            "recommendation": self.recommendation,
        }


# =============================================================================
# QUALITY SCORER
# =============================================================================

class QualityScorer:
    """
    Context Quality Scorer - rates each setup 0-100.
    
    Higher score = better setup quality.
    Uses only existing features from Feature Reactor.
    Fully explainable - breaks down each component.
    """
    
    # === WEIGHTS (adjustable) ===
    TREND_WEIGHT = 30
    RSI_WEIGHT = 20
    SR_WEIGHT = 20
    ATR_WEIGHT = 15
    MOMENTUM_WEIGHT = 15
    
    # === THRESHOLDS (TUNED - Option C Hybrid) ===
    # Widened thresholds to allow more trades through, skip only worst
    GRADE_THRESHOLDS = {
        'A': 70,     # Top tier - full confidence
        'B': 55,     # Good setups
        'C': 40,     # Widened from 45 - acceptable setups
        'D': 25,     # Widened from 30 - marginal but trackable
        'F': 0,      # Truly poor setups - SKIP
    }
    
    # MILD sizing - recover returns while keeping safety
    # Goal: Average size ~0.93x instead of 0.85x
    SIZE_MULTIPLIERS = {
        'A': 1.00,   # Full size
        'B': 0.95,   # Nearly full - good setups shouldn't be penalized
        'C': 0.90,   # Minor reduction only
        'D': 0.75,   # Moderate reduction
        'F': 0.00,   # Hard skip - this is the real filter
    }
    
    RECOMMENDATIONS = {
        'A': 'FULL',
        'B': 'HIGH',    # Was REDUCED
        'C': 'REDUCED', # Was MINI
        'D': 'MINI',
        'F': 'SKIP',
    }
    
    # === REGIME-SPECIFIC AGGRESSION (NEW) ===
    # Size up in TREND where momentum edge is strongest
    # Normal in RANGE (mean-reversion, less predictable)
    # Defensive in DANGER (should be blocked earlier, but safety)
    REGIME_AGGRESSION = {
        'TREND': 1.15,   # +15% size boost - exploit momentum edge
        'RANGE': 1.00,   # Normal sizing - no bonus
        'DANGER': 0.70,  # Defensive - rarely reached
        'UNKNOWN': 1.00, # Default - no change
    }
    
    def __init__(self, config: Optional[Dict] = None):
        config = config or {}
        
        # Configurable weights
        self.trend_weight = config.get('trend_weight', self.TREND_WEIGHT)
        self.rsi_weight = config.get('rsi_weight', self.RSI_WEIGHT)
        self.sr_weight = config.get('sr_weight', self.SR_WEIGHT)
        self.atr_weight = config.get('atr_weight', self.ATR_WEIGHT)
        self.momentum_weight = config.get('momentum_weight', self.MOMENTUM_WEIGHT)
        
        # Minimum score to allow trading
        self.min_score = config.get('min_score', 35)
        
        logger.info(f"[QualityScorer] Initialized with weights: "
                   f"trend={self.trend_weight}, rsi={self.rsi_weight}, "
                   f"sr={self.sr_weight}, atr={self.atr_weight}, "
                   f"momentum={self.momentum_weight}")
    
    # =========================================================================
    # MAIN SCORING METHOD
    # =========================================================================
    
    def calculate(
        self,
        features: Dict,
        signal: str,  # 'BUY' or 'SELL'
        regime: str = 'UNKNOWN',
        confidence: float = 0.5,
    ) -> QualityBreakdown:
        """
        Calculate quality score for a trading setup.
        
        Args:
            features: Dict of technical features
            signal: 'BUY' or 'SELL'
            regime: 'TREND', 'RANGE', 'DANGER', or 'UNKNOWN'
            confidence: ML confidence (0-1)
            
        Returns:
            QualityBreakdown with scores and recommendation
        """
        # Calculate each component
        trend_score = self._score_trend_strength(features, signal)
        rsi_score = self._score_rsi_position(features, signal)
        sr_score = self._score_sr_distance(features, signal)
        atr_score = self._score_atr_ratio(features)
        momentum_score = self._score_momentum(features, signal)
        
        # Total score (0-100)
        total = trend_score + rsi_score + sr_score + atr_score + momentum_score
        
        # Regime penalty (SOFTENED)
        if regime == 'DANGER':
            total *= 0.7  # Was 0.5 - still penalize but not as harsh
        elif regime == 'RANGE' and signal == 'BUY':
            # Slightly penalize trend trades in range (softened)
            total *= 0.92  # Was 0.85
        
        # Get grade and recommendation
        grade = self._get_grade(total)
        base_size_mult = self.SIZE_MULTIPLIERS[grade]
        recommendation = self.RECOMMENDATIONS[grade]
        
        # Apply REGIME AGGRESSION (NEW)
        # Boost size in TREND, keep normal in RANGE
        regime_mult = self.REGIME_AGGRESSION.get(regime, 1.0)
        final_size_mult = base_size_mult * regime_mult
        
        # Cap at 1.15 max (A-grade in TREND = 1.0 * 1.15 = 1.15)
        final_size_mult = min(final_size_mult, 1.15)
        
        breakdown = QualityBreakdown(
            trend_score=trend_score,
            rsi_score=rsi_score,
            sr_score=sr_score,
            atr_score=atr_score,
            momentum_score=momentum_score,
            total_score=total,
            grade=grade,
            size_multiplier=final_size_mult,
            recommendation=recommendation,
        )
        
        logger.debug(f"[QualityScorer] {signal} in {regime}: score={total:.1f}, "
                    f"grade={grade}, size={final_size_mult:.2f}x")
        
        return breakdown
    
    # =========================================================================
    # SCORING COMPONENTS
    # =========================================================================
    
    def _score_trend_strength(self, features: Dict, signal: str) -> float:
        """
        Score trend strength (0 to trend_weight points).
        
        Higher score when:
        - SMA20 > SMA50 (for BUY) or SMA20 < SMA50 (for SELL)
        - EMA12 and EMA26 aligned with signal
        - Price is on right side of moving averages
        """
        score = 0.0
        max_score = self.trend_weight
        
        sma_20 = features.get('sma_20', 0)
        sma_50 = features.get('sma_50', 0)
        ema_12 = features.get('ema_12', 0)
        ema_26 = features.get('ema_26', 0)
        close = features.get('close', 0)
        
        if sma_20 == 0 or sma_50 == 0:
            return max_score * 0.5  # Neutral if no data
        
        if signal == 'BUY':
            # SMA trend alignment (12 points) - more lenient
            if sma_20 > sma_50:
                ratio = min(1.0, (sma_20 / sma_50 - 1) * 100)  # 1% diff = full (was 2%)
                score += 12 * ratio
            elif abs(sma_20 - sma_50) / sma_50 < 0.003:  # Flat SMAs = half credit
                score += 6
            
            # EMA alignment (12 points) - more important
            if ema_12 > ema_26:
                score += 12
            elif ema_12 == ema_26:
                score += 6
            
            # Price above SMA20 (6 points)
            if close > sma_20:
                score += 6
            elif close > sma_50:  # At least above SMA50
                score += 3
                
        elif signal == 'SELL':
            # SMA trend alignment (12 points) - more lenient
            if sma_20 < sma_50:
                ratio = min(1.0, (sma_50 / sma_20 - 1) * 100)  # 1% diff = full
                score += 12 * ratio
            elif abs(sma_20 - sma_50) / sma_50 < 0.003:  # Flat SMAs = half credit
                score += 6
            
            # EMA alignment (12 points)
            if ema_12 < ema_26:
                score += 12
            elif ema_12 == ema_26:
                score += 6
            
            # Price below SMA20 (6 points)
            if close < sma_20:
                score += 6
            elif close < sma_50:  # At least below SMA50
                score += 3
        
        return min(score, max_score)
    
    def _score_rsi_position(self, features: Dict, signal: str) -> float:
        """
        Score RSI position (0 to rsi_weight points).
        
        Higher score when:
        - BUY: RSI is 30-50 (not oversold, room to go up)
        - SELL: RSI is 50-70 (not overbought, room to go down)
        """
        score = 0.0
        max_score = self.rsi_weight
        
        rsi = features.get('rsi_14', features.get('rsi', 50))
        
        if signal == 'BUY':
            # Best zone: 30-55 (widened from 35-50)
            if 30 <= rsi <= 55:
                score = max_score
            # Good zone: 20-30 or 55-65 (widened)
            elif 20 <= rsi < 30:
                score = max_score * 0.8  # Was 0.7
            elif 55 < rsi <= 65:
                score = max_score * 0.7  # Was 0.6
            # Okay zone: < 20 (oversold)
            elif rsi < 20:
                score = max_score * 0.6  # Was 0.5
            # Extended zone: > 65 (not ideal)
            elif rsi <= 75:
                score = max_score * 0.5  # Was 0.2 for >60
            else:
                score = max_score * 0.3  # Very extended
                
        elif signal == 'SELL':
            # Best zone: 45-70 (widened from 50-65)
            if 45 <= rsi <= 70:
                score = max_score
            # Good zone: 70-80 or 35-45 (widened)
            elif 70 < rsi <= 80:
                score = max_score * 0.8  # Was 0.7
            elif 35 <= rsi < 45:
                score = max_score * 0.7  # Was 0.6
            # Okay zone: > 80 (overbought)
            elif rsi > 80:
                score = max_score * 0.6  # Was 0.5
            # Extended zone: < 35 (not ideal)
            elif rsi >= 25:
                score = max_score * 0.5  # Was 0.2 for <40
            else:
                score = max_score * 0.3  # Very extended
        
        return score
    
    def _score_sr_distance(self, features: Dict, signal: str) -> float:
        """
        Score distance from support/resistance (0 to sr_weight points).
        
        Higher score when:
        - BUY: Far from resistance (room to go up)
        - SELL: Far from support (room to go down)
        """
        max_score = self.sr_weight
        
        # Try to get S/R from context brain output
        sr_proximity = features.get('sr_proximity', 0.5)  # 0 = at S/R, 1 = far
        
        if sr_proximity is None:
            sr_proximity = 0.5
        
        # If we have BB bands, use them as proxy for S/R
        bb_upper = features.get('bb_upper', 0)
        bb_lower = features.get('bb_lower', 0)
        close = features.get('close', 0)
        
        if bb_upper > 0 and bb_lower > 0 and close > 0:
            bb_range = bb_upper - bb_lower
            if bb_range > 0:
                if signal == 'BUY':
                    # Distance from upper BB (resistance)
                    dist_to_resistance = (bb_upper - close) / bb_range
                    sr_proximity = dist_to_resistance
                else:
                    # Distance from lower BB (support)
                    dist_to_support = (close - bb_lower) / bb_range
                    sr_proximity = dist_to_support
        
        # Score based on proximity (far = good)
        # Normalize to 0-1 range, then scale to max_score
        score = max_score * min(1.0, max(0.0, sr_proximity))
        
        return score
    
    def _score_atr_ratio(self, features: Dict) -> float:
        """
        Score ATR ratio (0 to atr_weight points).
        
        Higher score when volatility is normal (not too high or low).
        """
        max_score = self.atr_weight
        
        # Get volatility ratio if available
        vol_ratio = features.get('volatility_ratio', 1.0)
        bb_width = features.get('bb_width', 0.02)
        
        # If volatility_ratio not available, estimate from BB width
        if vol_ratio == 1.0 and bb_width > 0:
            # Normal BB width is around 0.015-0.025
            if bb_width < 0.010:
                vol_ratio = 0.5  # Low volatility
            elif bb_width > 0.035:
                vol_ratio = 2.0  # High volatility
            else:
                vol_ratio = 1.0  # Normal
        
        # Score: best around 1.0, penalize extremes
        if 0.7 <= vol_ratio <= 1.5:
            score = max_score  # Normal range = full score
        elif 0.5 <= vol_ratio < 0.7:
            score = max_score * 0.6  # Low volatility
        elif 1.5 < vol_ratio <= 2.0:
            score = max_score * 0.7  # High but acceptable
        elif vol_ratio < 0.5:
            score = max_score * 0.3  # Very low = bad
        else:
            score = max_score * 0.4  # Very high = risky
        
        return score
    
    def _score_momentum(self, features: Dict, signal: str) -> float:
        """
        Score momentum alignment (0 to momentum_weight points).
        
        Higher score when MACD agrees with signal direction.
        """
        max_score = self.momentum_weight
        score = 0.0
        
        macd = features.get('macd', 0)
        macd_signal = features.get('macd_signal', 0)
        macd_hist = features.get('macd_hist', features.get('macd_histogram', 0))
        
        if signal == 'BUY':
            # MACD above signal line (5 points)
            if macd > macd_signal:
                score += 5
            
            # MACD histogram positive (5 points)
            if macd_hist > 0:
                score += 5
            
            # MACD above zero (5 points)
            if macd > 0:
                score += 5
                
        elif signal == 'SELL':
            # MACD below signal line (5 points)
            if macd < macd_signal:
                score += 5
            
            # MACD histogram negative (5 points)
            if macd_hist < 0:
                score += 5
            
            # MACD below zero (5 points)
            if macd < 0:
                score += 5
        
        return min(score, max_score)
    
    # =========================================================================
    # HELPERS
    # =========================================================================
    
    def _get_grade(self, score: float) -> str:
        """Convert numeric score to letter grade."""
        if score >= self.GRADE_THRESHOLDS['A']:
            return 'A'
        elif score >= self.GRADE_THRESHOLDS['B']:
            return 'B'
        elif score >= self.GRADE_THRESHOLDS['C']:
            return 'C'
        elif score >= self.GRADE_THRESHOLDS['D']:
            return 'D'
        else:
            return 'F'
    
    def should_trade(self, breakdown: QualityBreakdown) -> bool:
        """Check if quality is high enough to trade."""
        return breakdown.total_score >= self.min_score
    
    def get_size_multiplier(self, breakdown: QualityBreakdown) -> float:
        """Get position size multiplier based on quality."""
        return breakdown.size_multiplier


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_scorer_instance: Optional[QualityScorer] = None


def get_quality_scorer(config: Optional[Dict] = None) -> QualityScorer:
    """Get or create quality scorer singleton."""
    global _scorer_instance
    if _scorer_instance is None:
        _scorer_instance = QualityScorer(config)
    return _scorer_instance


def reset_quality_scorer():
    """Reset singleton (for testing)."""
    global _scorer_instance
    _scorer_instance = None


def score_setup(features: Dict, signal: str, regime: str = 'UNKNOWN') -> QualityBreakdown:
    """Convenience function to score a setup."""
    scorer = get_quality_scorer()
    return scorer.calculate(features, signal, regime)


# =============================================================================
# MAIN (Testing)
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    scorer = QualityScorer()
    
    # Test Case 1: Strong BUY setup
    features1 = {
        'close': 1.0920,
        'sma_20': 1.0905,
        'sma_50': 1.0880,
        'ema_12': 1.0912,
        'ema_26': 1.0898,
        'rsi_14': 45,
        'macd': 0.0012,
        'macd_signal': 0.0008,
        'macd_hist': 0.0004,
        'bb_upper': 1.0960,
        'bb_lower': 1.0860,
        'bb_width': 0.018,
    }
    
    result1 = scorer.calculate(features1, 'BUY', 'TREND')
    print("\n=== TEST 1: Strong BUY in TREND ===")
    print(f"Breakdown: {result1.to_dict()}")
    print(f"Should trade: {scorer.should_trade(result1)}")
    
    # Test Case 2: Weak SELL setup
    features2 = {
        'close': 1.0920,
        'sma_20': 1.0930,  # Price below SMA
        'sma_50': 1.0900,  # But SMA20 > SMA50 (uptrend)
        'ema_12': 1.0925,
        'ema_26': 1.0910,
        'rsi_14': 35,  # Already low
        'macd': 0.001,
        'macd_signal': 0.0015,
        'macd_hist': -0.0005,
        'bb_upper': 1.0960,
        'bb_lower': 1.0880,
        'bb_width': 0.015,
    }
    
    result2 = scorer.calculate(features2, 'SELL', 'TREND')
    print("\n=== TEST 2: Weak SELL against TREND ===")
    print(f"Breakdown: {result2.to_dict()}")
    print(f"Should trade: {scorer.should_trade(result2)}")
    
    # Test Case 3: Neutral setup
    features3 = {
        'close': 1.0920,
        'sma_20': 1.0920,  # At SMA
        'sma_50': 1.0920,  # Flat
        'ema_12': 1.0920,
        'ema_26': 1.0920,
        'rsi_14': 50,  # Neutral
        'macd': 0,
        'macd_signal': 0,
        'macd_hist': 0,
        'bb_upper': 1.0950,
        'bb_lower': 1.0890,
        'bb_width': 0.012,  # Low vol
    }
    
    result3 = scorer.calculate(features3, 'BUY', 'RANGE')
    print("\n=== TEST 3: Neutral BUY in RANGE ===")
    print(f"Breakdown: {result3.to_dict()}")
    print(f"Should trade: {scorer.should_trade(result3)}")
