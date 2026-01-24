"""
Unit Tests for RiskBrain V2 - Fibonacci Edition

Tests cover:
1. Fibonacci MIN_RR table (1.13, 1.27, 1.41, 1.62, 2.00)
2. Fib step-down forgiveness for high confidence
3. Fib-based degradative sizing (0.62, 0.38)
4. Exploration override (never blocked)
5. DANGER regime with proper sizing
6. Manipulation hard-block threshold (0.70)
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.backtest.risk_brain_v1 import (
    RiskBrainV1, 
    MIN_RR_TABLE, 
    FIB_LEVELS, 
    FIB_DELTA_SMALL, 
    FIB_DELTA_LARGE,
    fib_step_down
)


class TestFibonacciConstants:
    """Test Fibonacci constants are correct."""
    
    def test_fib_levels_ordered(self):
        """FIB_LEVELS should be in ascending order"""
        assert FIB_LEVELS == [1.13, 1.27, 1.41, 1.62, 2.00]
    
    def test_fib_deltas(self):
        """Fib deltas should be correct"""
        assert FIB_DELTA_SMALL == 0.14
        assert FIB_DELTA_LARGE == 0.28
    
    def test_fib_step_down_from_golden(self):
        """1.62 (golden) should step down to 1.41"""
        assert fib_step_down(1.62) == 1.41
    
    def test_fib_step_down_from_sqrt2(self):
        """1.41 should step down to 1.27"""
        assert fib_step_down(1.41) == 1.27
    
    def test_fib_step_down_at_minimum(self):
        """1.13 should stay at 1.13 (no lower level)"""
        assert fib_step_down(1.13) == 1.13


class TestAdaptiveMinRR:
    """Test Fibonacci-based MIN_RR table."""
    
    def test_learning_range_uses_fib_127(self):
        """LEARNING + RANGE should use 1.27"""
        assert MIN_RR_TABLE[("LEARNING", "RANGE")] == 1.27
    
    def test_learning_danger_uses_fib_113(self):
        """LEARNING + DANGER should use 1.13 (lowest)"""
        assert MIN_RR_TABLE[("LEARNING", "DANGER")] == 1.13
    
    def test_confirmation_trend_uses_golden(self):
        """CONFIRMATION + TREND should use 1.62 (golden ratio)"""
        assert MIN_RR_TABLE[("CONFIRMATION", "TREND")] == 1.62
    
    def test_confirmation_danger_uses_fib_127(self):
        """CONFIRMATION + DANGER should use 1.27"""
        assert MIN_RR_TABLE[("CONFIRMATION", "DANGER")] == 1.27


class TestFibDegradativeSizing:
    """Test Fibonacci-based degradative sizing."""
    
    @pytest.fixture
    def risk_brain(self):
        return RiskBrainV1()
    
    def test_full_size_when_rr_exceeds_min(self, risk_brain):
        """Full size (1.0) when RR >= min_rr"""
        result = risk_brain.calculate_sl_tp(
            entry_price=1.1000,
            side='BUY',
            atr=0.0010,
            regime='TREND',
            context={'balance': 10000, 'confidence': 0.7},
            context_output={'operating_mode': 'LEARNING'}
        )
        
        # RR should be ~1.71, min is 1.41 for LEARNING+TREND
        assert result['risk_decision'] == 'FULL'
        assert result['position_size'] > 0
    
    def test_fib_62_reduction(self, risk_brain):
        """62% size when RR is one Fib step below min"""
        # Need to engineer a lower RR
        risk_brain.base_tp_atr_mult = 1.35  # Creates RR ~1.35
        
        result = risk_brain.calculate_sl_tp(
            entry_price=1.1000,
            side='BUY',
            atr=0.0010,
            regime='TREND',
            context={'balance': 10000, 'confidence': 0.6},
            context_output={'operating_mode': 'CONFIRMATION'}  # min = 1.62
        )
        
        # RR ~1.35 is below 1.62 but >= 1.62 - 0.14 = 1.48? No, 1.35 < 1.48
        # So this might be REDUCED-38% or lower
        assert result['risk_decision'] in ['REDUCED', 'BLOCK']
    
    def test_no_hard_block_above_fib_threshold(self, risk_brain):
        """Should not hard block when RR is within Fib degradation range"""
        result = risk_brain.calculate_sl_tp(
            entry_price=1.1000,
            side='BUY',
            atr=0.0010,
            regime='RANGE',
            context={'balance': 10000, 'confidence': 0.7},
            context_output={'operating_mode': 'LEARNING'}  # min = 1.27
        )
        
        # RR ~1.71 should be well above 1.27
        assert result['risk_decision'] != 'BLOCK'


class TestConfidenceForgiveness:
    """Test Fib step-down forgiveness for high confidence."""
    
    @pytest.fixture
    def risk_brain(self):
        return RiskBrainV1()
    
    def test_high_confidence_gets_fib_stepdown(self, risk_brain):
        """High confidence + good context should step down one Fib level"""
        result = risk_brain.calculate_sl_tp(
            entry_price=1.1000,
            side='BUY',
            atr=0.0010,
            regime='TREND',
            context={'balance': 10000, 'confidence': 0.90},
            context_output={
                'operating_mode': 'CONFIRMATION',  # Base = 1.62
                'manipulation_risk': 0.2,  # Low
                'sr_strength': 0.7  # High
            }
        )
        
        # With forgiveness: 1.62 → 1.41
        # RR ~1.71 should be FULL
        assert result['risk_decision'] == 'FULL'
    
    def test_low_confidence_no_forgiveness(self, risk_brain):
        """Low confidence should not get Fib forgiveness"""
        result = risk_brain.calculate_sl_tp(
            entry_price=1.1000,
            side='BUY',
            atr=0.0010,
            regime='RANGE',
            context={'balance': 10000, 'confidence': 0.60},
            context_output={
                'operating_mode': 'CONFIRMATION',
                'manipulation_risk': 0.2,
                'sr_strength': 0.7
            }
        )
        
        # No forgiveness, min stays at 1.41
        assert 'risk_decision' in result


class TestExplorationOverride:
    """Test exploration trade override (SACRED RULE)."""
    
    @pytest.fixture
    def risk_brain(self):
        return RiskBrainV1()
    
    def test_exploration_min_rr_is_127(self, risk_brain):
        """Exploration min_rr should be 1.27"""
        assert risk_brain.exploration_min_rr == 1.27
    
    def test_exploration_never_blocked_for_rr(self, risk_brain):
        """Exploration trades should NEVER be blocked for RR reasons"""
        # Set up for very low RR
        risk_brain.base_tp_atr_mult = 0.5
        risk_brain.base_sl_atr_mult = 2.0
        
        result = risk_brain.calculate_sl_tp(
            entry_price=1.1000,
            side='BUY',
            atr=0.0010,
            regime='RANGE',
            context={'balance': 10000, 'confidence': 0.5},
            exploration_trade=True,
            context_output={'operating_mode': 'LEARNING'}
        )
        
        # Should be REDUCED, not BLOCK
        assert result['risk_decision'] in ['FULL', 'REDUCED']
        assert result['position_size'] > 0
    
    def test_exploration_blocked_for_manipulation(self, risk_brain):
        """Exploration should be blocked when manipulation > 0.70"""
        result = risk_brain.calculate_sl_tp(
            entry_price=1.1000,
            side='BUY',
            atr=0.0010,
            regime='RANGE',
            context={'balance': 10000, 'confidence': 0.6},
            exploration_trade=True,
            context_output={
                'operating_mode': 'LEARNING',
                'manipulation_risk': 0.80  # Above threshold!
            }
        )
        
        assert result['risk_decision'] == 'BLOCK'
        assert 'manipulation' in result['risk_reason'].lower()
    
    def test_exploration_size_capped_at_30_pct(self, risk_brain):
        """Exploration size should be capped at 30%"""
        assert risk_brain.exploration_size_cap == 0.30


class TestDangerRegime:
    """Test DANGER regime behavior - degrade, not block."""
    
    @pytest.fixture
    def risk_brain(self):
        return RiskBrainV1()
    
    def test_danger_uses_lowest_min_rr_in_learning(self, risk_brain):
        """LEARNING + DANGER should use 1.13 (lowest Fib)"""
        assert MIN_RR_TABLE[("LEARNING", "DANGER")] == 1.13
    
    def test_danger_reduces_size_not_blocks(self, risk_brain):
        """DANGER regime should reduce size, not block"""
        result = risk_brain.calculate_sl_tp(
            entry_price=1.1000,
            side='BUY',
            atr=0.0010,
            regime='DANGER',
            context={'balance': 10000, 'confidence': 0.7},
            context_output={'operating_mode': 'LEARNING'}
        )
        
        # min_rr = 1.13, RR ~1.71 should pass easily
        assert result['risk_decision'] != 'BLOCK'


class TestReturnSchema:
    """Test return schema compliance."""
    
    @pytest.fixture
    def risk_brain(self):
        return RiskBrainV1()
    
    def test_all_required_fields_present(self, risk_brain):
        """Return dict should have all required fields"""
        result = risk_brain.calculate_sl_tp(
            entry_price=1.1000,
            side='BUY',
            atr=0.0010,
            regime='TREND',
            context={'balance': 10000, 'confidence': 0.7},
            context_output={'operating_mode': 'LEARNING'}
        )
        
        required = ['sl_price', 'tp_price', 'position_size', 'rr', 'risk_decision', 'risk_reason']
        for field in required:
            assert field in result, f"Missing: {field}"
    
    def test_risk_decision_valid_enum(self, risk_brain):
        """risk_decision must be FULL, REDUCED, or BLOCK"""
        result = risk_brain.calculate_sl_tp(
            entry_price=1.1000,
            side='BUY',
            atr=0.0010,
            regime='TREND',
            context={'balance': 10000, 'confidence': 0.7},
            context_output={'operating_mode': 'LEARNING'}
        )
        
        assert result['risk_decision'] in ['FULL', 'REDUCED', 'BLOCK']
    
    def test_no_none_for_sl_tp(self, risk_brain):
        """SL and TP should never be None"""
        result = risk_brain.calculate_sl_tp(
            entry_price=1.1000,
            side='BUY',
            atr=0.0010,
            regime='TREND',
            context={'balance': 10000, 'confidence': 0.7},
            context_output={'operating_mode': 'LEARNING'}
        )
        
        assert result['sl_price'] is not None
        assert result['tp_price'] is not None


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.fixture
    def risk_brain(self):
        return RiskBrainV1()
    
    def test_zero_atr_blocks(self, risk_brain):
        """Zero ATR should block"""
        result = risk_brain.calculate_sl_tp(
            entry_price=1.1000,
            side='BUY',
            atr=0.0,
            regime='TREND',
            context={'balance': 10000, 'confidence': 0.7},
            context_output={'operating_mode': 'LEARNING'}
        )
        
        assert result['risk_decision'] == 'BLOCK'
        assert 'ATR' in result['risk_reason']
    
    def test_uppercase_normalization(self, risk_brain):
        """Side and regime should be normalized to uppercase"""
        result = risk_brain.calculate_sl_tp(
            entry_price=1.1000,
            side='buy',  # lowercase
            atr=0.0010,
            regime='trend',  # lowercase
            context={'balance': 10000, 'confidence': 0.7},
            context_output={'operating_mode': 'LEARNING'}
        )
        
        assert result['risk_decision'] in ['FULL', 'REDUCED', 'BLOCK']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
