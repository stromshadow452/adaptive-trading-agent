"""
Regime Detector

Classifies market regime as TREND, CHOPPY, TRANSITION, or DEAD.
Used to route signals to appropriate brain (Trend ML vs Choppy Engine).
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional


class MarketRegime(Enum):
    """Market regime classification."""
    TREND = 'TREND'
    CHOPPY = 'CHOPPY'
    TRANSITION = 'TRANSITION'
    DEAD = 'DEAD'


@dataclass
class RegimeResult:
    """Result of regime detection."""
    regime: MarketRegime
    confidence: float
    metrics: Dict[str, float]
    reason: str


class RegimeDetector:
    """
    Detects market regime using normalized features.
    
    Regimes:
    - TREND: Strong directional movement, use ML trend brain
    - CHOPPY: Range-bound, use mean-reversion engine
    - TRANSITION: Between regimes, no trade
    - DEAD: Extremely low volatility, no trade
    """
    
    # Threshold configuration
    THRESHOLDS = {
        # DEAD market
        'atr_ratio_dead': 0.5,
        
        # CHOPPY market
        'trend_zscore_choppy_max': 0.5,
        'atr_ratio_choppy_min': 0.6,
        'atr_ratio_choppy_max': 1.2,
        'bb_width_choppy_max': 1.0,
        'price_position_choppy_min': 0.15,
        'price_position_choppy_max': 0.85,
        
        # TREND market
        'trend_zscore_trend_min': 1.0,
    }
    
    def detect(self, features: Dict[str, float]) -> RegimeResult:
        """
        Detect market regime from features.
        
        Args:
            features: Dict with normalized features from UniversalFeatureComputer
            
        Returns:
            RegimeResult with regime, confidence, and supporting metrics
        """
        # Extract key features
        trend_zscore = abs(features.get('trend_50_zscore', 0))
        atr_ratio = features.get('atr_ratio_14_50', 1.0)
        bb_width_zscore = features.get('bb_width_zscore', 0)
        price_position = features.get('price_position_50', 0.5)
        
        metrics = {
            'trend_zscore': trend_zscore,
            'atr_ratio': atr_ratio,
            'bb_width_zscore': bb_width_zscore,
            'price_position': price_position,
        }
        
        # Check DEAD first (absolute priority)
        if self._is_dead(atr_ratio):
            return RegimeResult(
                regime=MarketRegime.DEAD,
                confidence=1.0,
                metrics=metrics,
                reason=f"ATR ratio {atr_ratio:.2f} < {self.THRESHOLDS['atr_ratio_dead']}"
            )
        
        # Check CHOPPY
        choppy_result = self._check_choppy(features, metrics)
        if choppy_result:
            return choppy_result
        
        # Check TREND
        trend_result = self._check_trend(features, metrics)
        if trend_result:
            return trend_result
        
        # Default: TRANSITION
        return RegimeResult(
            regime=MarketRegime.TRANSITION,
            confidence=0.5,
            metrics=metrics,
            reason="Between regimes, conditions not met for TREND or CHOPPY"
        )
    
    def _is_dead(self, atr_ratio: float) -> bool:
        """Check if market is dead (extremely low volatility)."""
        return atr_ratio < self.THRESHOLDS['atr_ratio_dead']
    
    def _check_choppy(
        self, 
        features: Dict[str, float], 
        metrics: Dict[str, float]
    ) -> Optional[RegimeResult]:
        """
        Check if market is in CHOPPY regime.
        
        All conditions must be met:
        1. Low trend strength (abs(trend_zscore) < 0.5)
        2. Stable volatility (0.6 <= atr_ratio <= 1.2)
        3. BB not expanding (bb_width_zscore < 1.0)
        4. Price inside range (0.15 <= position <= 0.85)
        """
        t = self.THRESHOLDS
        
        trend_zscore = metrics['trend_zscore']
        atr_ratio = metrics['atr_ratio']
        bb_width_zscore = metrics['bb_width_zscore']
        price_position = metrics['price_position']
        
        conditions = {
            'trendless': trend_zscore < t['trend_zscore_choppy_max'],
            'vol_stable': t['atr_ratio_choppy_min'] <= atr_ratio <= t['atr_ratio_choppy_max'],
            'bb_not_expanding': bb_width_zscore < t['bb_width_choppy_max'],
            'inside_range': t['price_position_choppy_min'] <= price_position <= t['price_position_choppy_max'],
        }
        
        if all(conditions.values()):
            # Calculate confidence based on how strongly conditions are met
            confidence = self._calc_choppy_confidence(
                trend_zscore, atr_ratio, bb_width_zscore, price_position
            )
            
            return RegimeResult(
                regime=MarketRegime.CHOPPY,
                confidence=confidence,
                metrics=metrics,
                reason=f"All choppy conditions met: trend_z={trend_zscore:.2f}, atr_r={atr_ratio:.2f}"
            )
        
        return None
    
    def _check_trend(
        self, 
        features: Dict[str, float], 
        metrics: Dict[str, float]
    ) -> Optional[RegimeResult]:
        """
        Check if market is in TREND regime.
        
        Condition: trend_zscore > 1.0
        """
        trend_zscore = metrics['trend_zscore']
        
        if trend_zscore > self.THRESHOLDS['trend_zscore_trend_min']:
            # Confidence increases with trend strength
            confidence = min(0.5 + (trend_zscore - 1.0) * 0.25, 1.0)
            
            # Determine direction
            raw_trend = features.get('trend_50_zscore', 0)
            direction = 'UP' if raw_trend > 0 else 'DOWN'
            
            return RegimeResult(
                regime=MarketRegime.TREND,
                confidence=confidence,
                metrics=metrics,
                reason=f"Trend strength {trend_zscore:.2f} > 1.0 ({direction})"
            )
        
        return None
    
    def _calc_choppy_confidence(
        self,
        trend_zscore: float,
        atr_ratio: float,
        bb_width_zscore: float,
        price_position: float
    ) -> float:
        """
        Calculate confidence score for choppy regime.
        Higher score = more confident in choppy classification.
        """
        score = 0.5  # Base
        
        # Lower trend = more choppy
        if trend_zscore < 0.3:
            score += 0.15
        
        # ATR ratio near 1.0 = stable
        if 0.8 <= atr_ratio <= 1.1:
            score += 0.15
        
        # BB contracting = range compression
        if bb_width_zscore < 0:
            score += 0.1
        
        # Price in middle = more range-like
        if 0.3 <= price_position <= 0.7:
            score += 0.1
        
        return min(score, 1.0)


# ============================================================================
# Convenience functions
# ============================================================================

def detect_regime(features: Dict[str, float]) -> MarketRegime:
    """Quick regime detection (returns just the regime enum)."""
    detector = RegimeDetector()
    return detector.detect(features).regime


def is_choppy(features: Dict[str, float]) -> bool:
    """Check if market is in choppy regime."""
    return detect_regime(features) == MarketRegime.CHOPPY


def is_trending(features: Dict[str, float]) -> bool:
    """Check if market is in trending regime."""
    return detect_regime(features) == MarketRegime.TREND


def is_tradeable(features: Dict[str, float]) -> bool:
    """Check if market is tradeable (not DEAD or TRANSITION)."""
    regime = detect_regime(features)
    return regime in [MarketRegime.TREND, MarketRegime.CHOPPY]
