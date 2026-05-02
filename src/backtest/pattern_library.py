"""
SCOPUS Phase-1: Universal Pattern Library

Behavior-based patterns that work across all assets.
Patterns are detected by MARKET BEHAVIOR, not price levels.

Each pattern:
1. Has conditions based on normalized features (Z-scores, ATR ratios)
2. Specifies direction relative to context (WITH_TREND, COUNTER_EXTREME)
3. Has expected R:R ratio
4. May have size modifiers for higher-risk setups
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
from collections import defaultdict
import numpy as np
from datetime import datetime


# ============================================================================
# PATTERN DEFINITIONS
# ============================================================================

@dataclass
class PatternDefinition:
    """Definition of a universal trading pattern."""
    name: str
    description: str
    conditions: List[Dict]  # List of {feature, operator, threshold}
    direction: str  # 'WITH_TREND', 'COUNTER_EXTREME', 'WITH_BREAK', 'COUNTER_PANIC'
    expected_rr: float
    min_score: float = 0.7  # Minimum match score to trigger
    size_modifier: float = 1.0  # Size adjustment for the pattern
    priority: int = 1  # Higher = checked first


# Define the 5 core universal patterns
UNIVERSAL_PATTERNS = {
    
    'TREND_PULLBACK': PatternDefinition(
        name='TREND_PULLBACK',
        description='Strong trend with temporary pullback - buy/sell with trend',
        conditions=[
            {'feature': 'trend_50_zscore', 'op': 'abs_gt', 'threshold': 0.8},
            {'feature': 'rsi_zscore', 'op': 'sign_opposite_trend', 'threshold': 0.0},
            {'feature': 'price_position_20', 'op': 'between', 'threshold': (0.2, 0.5)},
            {'feature': 'atr_ratio_14_50', 'op': 'lt', 'threshold': 1.3},
            {'feature': 'slope_20_atr', 'op': 'sign_matches_trend', 'threshold': 0.0},
        ],
        direction='WITH_TREND',
        expected_rr=2.0,
        min_score=1.0,  # ALL 5 conditions required
        size_modifier=1.0,
        priority=2,
    ),
    
    'RANGE_REJECTION': PatternDefinition(
        name='RANGE_REJECTION',
        description='Price at range extreme with momentum reversal',
        conditions=[
            {'feature': 'trend_50_zscore', 'op': 'abs_lt', 'threshold': 0.6},
            {'feature': 'price_position_50', 'op': 'or_extreme', 'threshold': (0.15, 0.85)},
            {'feature': 'rsi_zscore', 'op': 'abs_gt', 'threshold': 1.2},
            {'feature': 'range_ratio', 'op': 'lt', 'threshold': 1.2},
            {'feature': 'bb_width_pct', 'op': 'lt', 'threshold': 0.5},
        ],
        direction='COUNTER_EXTREME',
        expected_rr=1.5,
        min_score=1.0,  # ALL 5 conditions required
        size_modifier=0.9,
        priority=1,
    ),
    
    'BREAKOUT_MOMENTUM': PatternDefinition(
        name='BREAKOUT_MOMENTUM',
        description='Range expansion with strong momentum',
        conditions=[
            {'feature': 'range_ratio', 'op': 'gt', 'threshold': 1.5},
            {'feature': 'atr_expansion', 'op': 'gt', 'threshold': 0.2},
            {'feature': 'price_position_50', 'op': 'or_extreme', 'threshold': (0.1, 0.9)},
            {'feature': 'roc_10_zscore', 'op': 'abs_gt', 'threshold': 1.5},
            {'feature': 'bb_width_zscore', 'op': 'gt', 'threshold': 0.5},
        ],
        direction='WITH_BREAK',
        expected_rr=3.0,
        min_score=1.0,  # ALL 5 conditions required
        size_modifier=0.8,
        priority=3,
    ),
    
    'PANIC_REVERSAL': PatternDefinition(
        name='PANIC_REVERSAL',
        description='Extreme volatility spike with exhaustion signals',
        conditions=[
            {'feature': 'atr_ratio_14_50', 'op': 'gt', 'threshold': 1.8},
            {'feature': 'rsi_zscore', 'op': 'abs_gt', 'threshold': 2.0},
            {'feature': 'range_ratio', 'op': 'gt', 'threshold': 2.0},
            {'feature': 'price_position_20', 'op': 'or_extreme', 'threshold': (0.05, 0.95)},
        ],
        direction='COUNTER_PANIC',
        expected_rr=2.5,
        min_score=1.0,  # All 4 conditions required (most strict)
        size_modifier=0.5,  # Reduced size for reversal trades
        priority=4,
    ),
    
    'CONTINUATION_FLAG': PatternDefinition(
        name='CONTINUATION_FLAG',
        description='Trend pause with volatility contraction (flag/pennant)',
        conditions=[
            {'feature': 'trend_50_zscore', 'op': 'abs_gt', 'threshold': 0.7},
            {'feature': 'atr_ratio_14_50', 'op': 'lt', 'threshold': 0.85},
            {'feature': 'rsi_zscore', 'op': 'abs_lt', 'threshold': 0.7},
            {'feature': 'bb_width_pct', 'op': 'lt', 'threshold': 0.3},
            {'feature': 'price_position_20', 'op': 'between', 'threshold': (0.3, 0.7)},
        ],
        direction='WITH_TREND',
        expected_rr=2.0,
        min_score=1.0,  # ALL 5 conditions required
        size_modifier=1.0,
        priority=2,
    ),
}


# ============================================================================
# PATTERN MATCHER
# ============================================================================

class PatternMatcher:
    """
    Match current market state against universal patterns.
    """
    
    def __init__(self, patterns: Dict[str, PatternDefinition] = None):
        self.patterns = patterns or UNIVERSAL_PATTERNS
    
    def evaluate_condition(
        self, 
        feature_value: float, 
        condition: Dict,
        features: Dict[str, float] = None
    ) -> bool:
        """
        Evaluate a single condition.
        
        Operators:
        - gt, lt, eq: greater than, less than, equal
        - abs_gt, abs_lt: absolute value comparison
        - between: value in range (threshold is tuple)
        - or_extreme: value below low OR above high
        - sign_matches_trend: same sign as trend_50_zscore
        - sign_opposite_trend: opposite sign to trend_50_zscore
        """
        op = condition['op']
        threshold = condition['threshold']
        
        if feature_value is None or np.isnan(feature_value):
            return False
        
        if op == 'gt':
            return feature_value > threshold
        elif op == 'lt':
            return feature_value < threshold
        elif op == 'eq':
            return abs(feature_value - threshold) < 0.01
        elif op == 'abs_gt':
            return abs(feature_value) > threshold
        elif op == 'abs_lt':
            return abs(feature_value) < threshold
        elif op == 'between':
            low, high = threshold
            return low <= feature_value <= high
        elif op == 'or_extreme':
            low, high = threshold
            return feature_value < low or feature_value > high
        elif op == 'sign_matches_trend':
            if features:
                trend = features.get('trend_50_zscore', 0)
                return (feature_value > 0 and trend > 0) or (feature_value < 0 and trend < 0)
            return False
        elif op == 'sign_opposite_trend':
            if features:
                trend = features.get('trend_50_zscore', 0)
                return (feature_value > 0 and trend < 0) or (feature_value < 0 and trend > 0)
            return False
        else:
            return False
    
    def score_pattern(
        self, 
        pattern: PatternDefinition, 
        features: Dict[str, float]
    ) -> float:
        """
        Score how well features match a pattern.
        
        Returns:
            0.0 to 1.0 match score
        """
        if not pattern.conditions:
            return 0.0
        
        matches = 0
        for condition in pattern.conditions:
            feature_name = condition['feature']
            feature_value = features.get(feature_name)
            
            if self.evaluate_condition(feature_value, condition, features):
                matches += 1
        
        return matches / len(pattern.conditions)
    
    def detect_pattern(self, features: Dict[str, float]) -> Optional[Dict]:
        """
        Detect the best matching pattern.
        
        Returns:
            Dict with pattern info or None if no match
        """
        best_match = None
        best_score = 0.0
        
        # Sort by priority (higher priority checked first)
        sorted_patterns = sorted(
            self.patterns.values(), 
            key=lambda p: p.priority, 
            reverse=True
        )
        
        for pattern in sorted_patterns:
            score = self.score_pattern(pattern, features)
            
            if score >= pattern.min_score and score > best_score:
                best_match = {
                    'name': pattern.name,
                    'score': score,
                    'direction': pattern.direction,
                    'expected_rr': pattern.expected_rr,
                    'size_modifier': pattern.size_modifier,
                }
                best_score = score
        
        return best_match
    
    def get_trade_direction(
        self, 
        pattern_direction: str, 
        features: Dict[str, float]
    ) -> str:
        """
        Convert pattern direction to BUY/SELL.
        
        Args:
            pattern_direction: WITH_TREND, COUNTER_EXTREME, etc.
            features: Current feature values
        
        Returns:
            'BUY' or 'SELL'
        """
        trend = features.get('trend_50_zscore', 0)
        price_pos = features.get('price_position_50', 0.5)
        rsi = features.get('rsi_14', 0.5)
        
        if pattern_direction == 'WITH_TREND':
            return 'BUY' if trend > 0 else 'SELL'
        
        elif pattern_direction == 'COUNTER_EXTREME':
            # Buy at low extreme, sell at high extreme
            return 'BUY' if price_pos < 0.5 else 'SELL'
        
        elif pattern_direction == 'WITH_BREAK':
            # Follow the breakout direction
            if price_pos > 0.8:
                return 'BUY'  # Breaking high
            elif price_pos < 0.2:
                return 'SELL'  # Breaking low
            else:
                return 'BUY' if trend > 0 else 'SELL'
        
        elif pattern_direction == 'COUNTER_PANIC':
            # Fade the panic
            return 'BUY' if rsi < 0.5 else 'SELL'
        
        else:
            return 'HOLD'


# ============================================================================
# PATTERN OUTCOME TRACKER
# ============================================================================

@dataclass
class PatternOutcome:
    """Record of a pattern trade outcome."""
    pattern_name: str
    asset_class: str
    regime: str
    outcome_r: float  # Outcome in R-multiples
    direction: str
    timestamp: str


class PatternOutcomeTracker:
    """
    Track pattern outcomes WITHOUT asset names.
    Groups by: pattern_type, asset_class, regime
    """
    
    def __init__(self, min_samples: int = 20):
        self.outcomes: Dict[tuple, List[float]] = defaultdict(list)
        self.min_samples = min_samples
    
    def record_trade(
        self,
        pattern_name: str,
        asset_class: str,
        regime: str,
        outcome_r: float,
        direction: str = None,
        timestamp: str = None,
    ):
        """Record a trade outcome."""
        key = (pattern_name, asset_class, regime)
        self.outcomes[key].append(outcome_r)
    
    def get_pattern_stats(
        self, 
        pattern_name: str, 
        asset_class: str = None,
        regime: str = None,
    ) -> Dict:
        """
        Get statistics for a pattern.
        
        Returns:
            Dict with win_rate, avg_r, sample_size, is_reliable
        """
        matching = []
        for key, outcomes in self.outcomes.items():
            p_name, a_class, reg = key
            if p_name == pattern_name:
                if asset_class is None or a_class == asset_class:
                    if regime is None or reg == regime:
                        matching.extend(outcomes)
        
        if not matching:
            return {
                'win_rate': 0.5,
                'avg_r': 0.0,
                'sample_size': 0,
                'is_reliable': False,
            }
        
        wins = sum(1 for r in matching if r > 0)
        return {
            'win_rate': wins / len(matching),
            'avg_r': np.mean(matching),
            'sample_size': len(matching),
            'is_reliable': len(matching) >= self.min_samples,
        }
    
    def get_pattern_adjustment(
        self,
        pattern_name: str,
        asset_class: str,
        regime: str,
    ) -> float:
        """
        Get size adjustment multiplier based on recent performance.
        
        Returns:
            0.5 to 1.2 multiplier
        """
        stats = self.get_pattern_stats(pattern_name, asset_class, regime)
        
        if not stats['is_reliable']:
            return 1.0  # No adjustment without enough data
        
        # Adjust based on win rate deviation from expected
        expected_wr = 0.55
        wr_diff = stats['win_rate'] - expected_wr
        
        # +10% win rate = +10% size, -10% = -20% size (asymmetric)
        if wr_diff >= 0:
            adjustment = 1.0 + min(wr_diff, 0.15)  # Cap at 1.15
        else:
            adjustment = 1.0 + 2 * max(wr_diff, -0.25)  # Faster reduction
        
        return max(0.5, min(1.2, adjustment))
    
    def get_summary(self) -> Dict[str, Dict]:
        """Get summary statistics for all patterns."""
        summary = {}
        pattern_names = set(k[0] for k in self.outcomes.keys())
        
        for pattern_name in pattern_names:
            summary[pattern_name] = self.get_pattern_stats(pattern_name)
        
        return summary


# ============================================================================
# COMBINED PATTERN DETECTOR
# ============================================================================

class UniversalPatternDetector:
    """
    High-level interface for pattern detection with outcome tracking.
    """
    
    def __init__(self):
        self.matcher = PatternMatcher()
        self.tracker = PatternOutcomeTracker()
    
    def detect(
        self, 
        features: Dict[str, float],
        asset_class: str = 'FX_MAJOR',
        regime: str = 'UNKNOWN',
    ) -> Optional[Dict]:
        """
        Detect pattern and return trade signal.
        
        Returns:
            Dict with:
            - pattern_name
            - direction (BUY/SELL)
            - score
            - expected_rr
            - size_modifier (combined pattern + performance)
        """
        match = self.matcher.detect_pattern(features)
        
        if match is None:
            return None
        
        # Get trade direction
        direction = self.matcher.get_trade_direction(
            match['direction'], 
            features
        )
        
        if direction == 'HOLD':
            return None
        
        # Get performance-based adjustment
        perf_adjustment = self.tracker.get_pattern_adjustment(
            match['name'],
            asset_class,
            regime,
        )
        
        return {
            'pattern_name': match['name'],
            'pattern_score': match['score'],
            'direction': direction,
            'expected_rr': match['expected_rr'],
            'size_modifier': match['size_modifier'] * perf_adjustment,
            'asset_class': asset_class,
            'regime': regime,
        }
    
    def record_outcome(
        self,
        pattern_name: str,
        asset_class: str,
        regime: str,
        outcome_r: float,
        direction: str = None,
    ):
        """Record trade outcome for pattern learning."""
        self.tracker.record_trade(
            pattern_name=pattern_name,
            asset_class=asset_class,
            regime=regime,
            outcome_r=outcome_r,
            direction=direction,
            timestamp=datetime.now().isoformat(),
        )
    
    def get_pattern_summary(self) -> Dict:
        """Get performance summary of all patterns."""
        return self.tracker.get_summary()
