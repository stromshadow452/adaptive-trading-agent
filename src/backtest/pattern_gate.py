"""
SCOPUS Phase-1: Universal Pattern Gate

Integrates universal patterns as a FILTER for ML-generated signals.
This is NOT a replacement for the ML brain - it's an additional gate.

Flow: ML Signal → Adaptive Gate → RSI Gate → Pattern Gate → Quality Gate → Trade
"""

import sys
sys.path.insert(0, '.')

from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from src.backtest.universal_features import UniversalFeatureComputer
from src.backtest.pattern_library import UniversalPatternDetector, UNIVERSAL_PATTERNS
from src.backtest.asset_class import detect_asset_class, get_class_parameters


@dataclass
class PatternGateResult:
    """Result of pattern gate check."""
    passed: bool
    pattern_name: Optional[str] = None
    pattern_score: float = 0.0
    direction_aligned: bool = False
    size_modifier: float = 1.0
    reason: str = ""


class UniversalPatternGate:
    """
    Pattern-based filter for ML signals.
    
    NOT a signal generator - validates ML signals against universal patterns.
    Allows signals that:
    1. Align with a detected pattern direction, OR
    2. Have no conflicting pattern (neutral)
    
    Blocks signals that:
    1. Contradict a strong detected pattern
    """
    
    def __init__(
        self,
        require_pattern_alignment: bool = False,  # If True, MUST have pattern
        allow_neutral: bool = True,  # If True, allow trades with no matching pattern
        conflict_threshold: float = 0.8,  # Score above this = strong conflict
    ):
        self.detector = UniversalPatternDetector()
        self.require_alignment = require_pattern_alignment
        self.allow_neutral = allow_neutral
        self.conflict_threshold = conflict_threshold
        
        # Stats tracking
        self.stats = {
            'total_checks': 0,
            'passed': 0,
            'blocked_conflict': 0,
            'blocked_no_pattern': 0,
            'pattern_aligned': 0,
            'neutral_passed': 0,
        }
    
    def check(
        self,
        ml_direction: str,  # 'BUY' or 'SELL' from ML
        features: Dict[str, float],  # Universal feature dict
        asset_class: str = 'FX_MAJOR',
        regime: str = 'UNKNOWN',
    ) -> PatternGateResult:
        """
        Check if ML signal passes the pattern gate.
        
        Args:
            ml_direction: Direction from ML prediction
            features: Universal normalized features
            asset_class: Asset class for adjustments
            regime: Current market regime
        
        Returns:
            PatternGateResult with pass/fail and modifiers
        """
        self.stats['total_checks'] += 1
        
        # Detect pattern
        signal = self.detector.detect(features, asset_class=asset_class, regime=regime)
        
        if signal is None:
            # No pattern detected
            if self.require_alignment:
                self.stats['blocked_no_pattern'] += 1
                return PatternGateResult(
                    passed=False,
                    reason='NO_PATTERN_MATCH',
                )
            else:
                # Neutral - allow with no modifier
                self.stats['neutral_passed'] += 1
                self.stats['passed'] += 1
                return PatternGateResult(
                    passed=True,
                    direction_aligned=False,
                    size_modifier=0.8,  # Slight reduction for no pattern confirmation
                    reason='NEUTRAL_NO_PATTERN',
                )
        
        # Pattern detected - check alignment
        pattern_direction = signal['direction']
        pattern_score = signal['pattern_score']
        pattern_name = signal['pattern_name']
        
        # Check if directions align
        if pattern_direction == ml_direction:
            # Perfect alignment - boost
            self.stats['pattern_aligned'] += 1
            self.stats['passed'] += 1
            return PatternGateResult(
                passed=True,
                pattern_name=pattern_name,
                pattern_score=pattern_score,
                direction_aligned=True,
                size_modifier=1.0 + (pattern_score - 0.8) * 0.5,  # Up to 1.1x boost
                reason='PATTERN_ALIGNED',
            )
        
        elif pattern_score >= self.conflict_threshold:
            # Strong conflicting pattern - block
            self.stats['blocked_conflict'] += 1
            return PatternGateResult(
                passed=False,
                pattern_name=pattern_name,
                pattern_score=pattern_score,
                direction_aligned=False,
                reason=f'CONFLICT_{pattern_name}_{pattern_direction}',
            )
        
        else:
            # Weak pattern, allow with reduction
            if self.allow_neutral:
                self.stats['neutral_passed'] += 1
                self.stats['passed'] += 1
                return PatternGateResult(
                    passed=True,
                    pattern_name=pattern_name,
                    pattern_score=pattern_score,
                    direction_aligned=False,
                    size_modifier=0.7,  # Reduce size for weak conflict
                    reason='WEAK_CONFLICT_ALLOWED',
                )
            else:
                self.stats['blocked_conflict'] += 1
                return PatternGateResult(
                    passed=False,
                    pattern_name=pattern_name,
                    pattern_score=pattern_score,
                    reason=f'CONFLICT_{pattern_name}',
                )
    
    def get_stats(self) -> Dict:
        """Get gate statistics."""
        total = self.stats['total_checks']
        if total == 0:
            return self.stats
        
        return {
            **self.stats,
            'pass_rate': self.stats['passed'] / total,
            'block_rate': 1 - (self.stats['passed'] / total),
            'alignment_rate': self.stats['pattern_aligned'] / total,
        }
    
    def reset_stats(self):
        """Reset statistics."""
        for key in self.stats:
            self.stats[key] = 0


class UniversalSLTPAdjuster:
    """
    Adjust SL/TP based on universal features and patterns.
    Uses ATR-based calculations instead of fixed pips.
    """
    
    def __init__(self):
        pass
    
    def adjust(
        self,
        entry: float,
        base_sl: float,
        base_tp: float,
        direction: str,
        features: Dict[str, float],
        pattern: Optional[str] = None,
    ) -> Tuple[float, float]:
        """
        Adjust SL/TP based on market conditions.
        
        Returns:
            (adjusted_sl, adjusted_tp)
        """
        # Get volatility expansion
        atr_ratio = features.get('atr_ratio_14_50', 1.0)
        range_ratio = features.get('range_ratio', 1.0)
        
        # Expand SL in high volatility
        sl_mult = 1.0
        if atr_ratio > 1.5:
            sl_mult = 1.2  # Widen SL in volatile conditions
        elif atr_ratio < 0.7:
            sl_mult = 0.9  # Tighten in quiet conditions
        
        # Adjust TP based on trend
        tp_mult = 1.0
        trend = features.get('trend_50_zscore', 0)
        if direction == 'BUY' and trend > 1.0:
            tp_mult = 1.2  # Extend TP in strong trend
        elif direction == 'SELL' and trend < -1.0:
            tp_mult = 1.2
        
        # Calculate new levels
        if direction == 'BUY':
            sl_dist = entry - base_sl
            tp_dist = base_tp - entry
            new_sl = entry - sl_dist * sl_mult
            new_tp = entry + tp_dist * tp_mult
        else:
            sl_dist = base_sl - entry
            tp_dist = entry - base_tp
            new_sl = entry + sl_dist * sl_mult
            new_tp = entry - tp_dist * tp_mult
        
        return new_sl, new_tp


# ============================================================================
# INTEGRATION HELPER
# ============================================================================

def create_pattern_gate(
    mode: str = 'BALANCED'
) -> UniversalPatternGate:
    """
    Create pattern gate with preset configuration.
    
    Modes:
    - STRICT: Require pattern alignment, block all conflicts
    - BALANCED: Allow neutral, block strong conflicts
    - PERMISSIVE: Allow most, only block perfect conflicts
    """
    if mode == 'STRICT':
        return UniversalPatternGate(
            require_pattern_alignment=True,
            allow_neutral=False,
            conflict_threshold=0.7,
        )
    elif mode == 'PERMISSIVE':
        return UniversalPatternGate(
            require_pattern_alignment=False,
            allow_neutral=True,
            conflict_threshold=1.0,  # Never blocks
        )
    else:  # BALANCED
        return UniversalPatternGate(
            require_pattern_alignment=False,
            allow_neutral=True,
            conflict_threshold=0.8,
        )
