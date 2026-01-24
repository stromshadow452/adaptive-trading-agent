"""
Exact RANGE Detection Module

Mathematical regime classification based on:
1. Structure (HH/LL count) - no directional breaks
2. ATR Compression - volatility below average
3. EMA Slope - flat momentum

This is deterministic (no ML) and optimized for 15m timeframe.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class RangeDetectionResult:
    """Result of RANGE detection analysis."""
    is_range: bool
    confidence: float
    structure_score: float
    atr_compression: float
    slope_category: str  # "FLAT", "UP", "DOWN"
    reason: str


class ExactRangeDetector:
    """
    Exact mathematical RANGE detection.
    
    RANGE = (structure_range) AND (atr_compressed) AND (slope_flat)
    """
    
    # Parameters optimized for 15m timeframe
    STRUCTURE_LOOKBACK = 20          # 5 hours of 15m candles
    STRUCTURE_THRESHOLD = 0.25       # Max 25% directional breaks for RANGE
    ATR_COMPRESSION_RATIO = 0.80     # ATR < 80% of avg = compressed
    SLOPE_THRESHOLD = 0.0001         # ±0.01% per bar = flat
    EMA_PERIOD = 20                  # Short-term trend
    ATR_LOOKBACK = 50                # ATR average lookback
    
    def __init__(self, config: dict = None):
        config = config or {}
        self.structure_lookback = config.get('structure_lookback', self.STRUCTURE_LOOKBACK)
        self.structure_threshold = config.get('structure_threshold', self.STRUCTURE_THRESHOLD)
        self.atr_compression_ratio = config.get('atr_compression_ratio', self.ATR_COMPRESSION_RATIO)
        self.slope_threshold = config.get('slope_threshold', self.SLOPE_THRESHOLD)
        self.ema_period = config.get('ema_period', self.EMA_PERIOD)
        self.atr_lookback = config.get('atr_lookback', self.ATR_LOOKBACK)
        
        # History for ATR averaging
        self.atr_history: List[float] = []
    
    def detect(self, 
               highs: List[float], 
               lows: List[float], 
               closes: List[float],
               atr_current: float) -> RangeDetectionResult:
        """
        Detect if current market is in RANGE regime.
        
        Args:
            highs: List of high prices (most recent last)
            lows: List of low prices (most recent last)
            closes: List of close prices (most recent last)
            atr_current: Current ATR value
            
        Returns:
            RangeDetectionResult with all analysis components
        """
        # Update ATR history
        self.atr_history.append(atr_current)
        if len(self.atr_history) > self.atr_lookback:
            self.atr_history = self.atr_history[-self.atr_lookback:]
        
        # 1. Structure Analysis (HH/LL count)
        structure_score = self._count_structure(highs, lows)
        structure_range = structure_score < self.structure_threshold
        
        # 2. ATR Compression
        atr_compression = self._check_atr_compression(atr_current)
        atr_compressed = atr_compression < self.atr_compression_ratio
        
        # 3. EMA Slope
        slope_category = self._compute_ema_slope(closes)
        slope_flat = slope_category == "FLAT"
        
        # Final classification
        is_range = structure_range and atr_compressed and slope_flat
        
        # Confidence calculation
        if is_range:
            # Higher confidence when all factors strongly indicate RANGE
            structure_conf = 1.0 - (structure_score / self.structure_threshold)
            atr_conf = 1.0 - (atr_compression / self.atr_compression_ratio)
            slope_conf = 0.8 if slope_flat else 0.3
            confidence = min(0.95, (structure_conf + atr_conf + slope_conf) / 3)
        else:
            confidence = 0.0
        
        # Build reason string
        reasons = []
        if structure_range:
            reasons.append(f"structure={structure_score:.2f}<{self.structure_threshold}")
        else:
            reasons.append(f"structure={structure_score:.2f}≥{self.structure_threshold}")
        if atr_compressed:
            reasons.append(f"ATR_comp={atr_compression:.2f}<{self.atr_compression_ratio}")
        else:
            reasons.append(f"ATR_comp={atr_compression:.2f}≥{self.atr_compression_ratio}")
        reasons.append(f"slope={slope_category}")
        
        reason = " | ".join(reasons)
        
        return RangeDetectionResult(
            is_range=is_range,
            confidence=confidence,
            structure_score=structure_score,
            atr_compression=atr_compression,
            slope_category=slope_category,
            reason=reason
        )
    
    def _count_structure(self, highs: List[float], lows: List[float]) -> float:
        """
        Count higher-highs (HH) and lower-lows (LL) over lookback period.
        
        Returns:
            structure_score: 0-1 normalized (lower = more range-like)
        """
        if len(highs) < self.structure_lookback + 1:
            return 0.5  # Not enough data, neutral
        
        hh_count = 0
        ll_count = 0
        
        for i in range(1, min(self.structure_lookback, len(highs))):
            # Higher-High: current high > previous high
            if highs[-i] > highs[-i-1]:
                hh_count += 1
            # Lower-Low: current low < previous low
            if lows[-i] < lows[-i-1]:
                ll_count += 1
        
        # Normalize: how many directional breaks occurred
        max_breaks = 2 * (self.structure_lookback - 1)
        structure_score = (hh_count + ll_count) / max_breaks if max_breaks > 0 else 0.5
        
        return structure_score
    
    def _check_atr_compression(self, atr_current: float) -> float:
        """
        Check if ATR is compressed relative to historical average.
        
        Returns:
            compression_ratio: current_atr / avg_atr (lower = more compressed)
        """
        if len(self.atr_history) < 10:
            return 1.0  # Not enough history, no compression
        
        atr_avg = np.mean(self.atr_history)
        if atr_avg == 0:
            return 1.0
        
        return atr_current / atr_avg
    
    def _compute_ema_slope(self, closes: List[float]) -> str:
        """
        Calculate EMA slope category.
        
        Returns:
            "FLAT", "UP", or "DOWN"
        """
        if len(closes) < self.ema_period + 5:
            return "FLAT"  # Not enough data
        
        # Calculate EMA
        ema = self._calculate_ema(closes, self.ema_period)
        
        # Slope = (EMA[now] - EMA[5 bars ago]) / EMA[5 bars ago] / 5
        if len(ema) < 6 or ema[-6] == 0:
            return "FLAT"
        
        slope = (ema[-1] - ema[-6]) / ema[-6] / 5
        
        if abs(slope) < self.slope_threshold:
            return "FLAT"
        elif slope > self.slope_threshold:
            return "UP"
        else:
            return "DOWN"
    
    def _calculate_ema(self, prices: List[float], period: int) -> List[float]:
        """Calculate Exponential Moving Average."""
        if len(prices) < period:
            return prices.copy()
        
        multiplier = 2 / (period + 1)
        ema = [np.mean(prices[:period])]  # Start with SMA
        
        for price in prices[period:]:
            ema.append((price - ema[-1]) * multiplier + ema[-1])
        
        return ema
    
    def classify_regime(self, 
                        highs: List[float], 
                        lows: List[float], 
                        closes: List[float],
                        atr_current: float) -> Tuple[str, float]:
        """
        Full regime classification: RANGE or TREND.
        
        Returns:
            (regime, confidence) tuple
        """
        result = self.detect(highs, lows, closes, atr_current)
        
        if result.is_range:
            return "RANGE", result.confidence
        else:
            # It's TREND - compute trend confidence
            # Higher structure score = stronger trend
            trend_conf = min(0.9, result.structure_score * 2)
            return "TREND", trend_conf


# Singleton for easy access
_range_detector: Optional[ExactRangeDetector] = None

def get_range_detector(config: dict = None) -> ExactRangeDetector:
    """Get or create the singleton RANGE detector."""
    global _range_detector
    if _range_detector is None:
        _range_detector = ExactRangeDetector(config)
    return _range_detector
