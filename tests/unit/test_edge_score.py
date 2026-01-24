"""
Unit Tests for MARK-3 EDGE_SCORE Module

Tests:
1. Edge score computation with various inputs
2. Size multiplier mapping
3. Structure quality cap
4. DANGER regime blocking
5. Safeguards (daily limits, consecutive limits)
6. Integration with sizing pipeline
"""

import pytest
import sys
import os
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.backtest.edge_score import (
    EdgeScoreModule,
    EdgeScoreSafeguards,
    apply_edge_score_to_sizing,
    calculate_volatility_alignment,
    calculate_structure_quality,
    EDGE_BLOCK_THRESHOLD,
    EDGE_REDUCED_THRESHOLD,
    EDGE_NORMAL_THRESHOLD,
    SIZE_MULT_MINIMUM,
    SIZE_MULT_BOOST_MAX,
)


class TestEdgeScoreComputation:
    """Test EDGE_SCORE calculation."""
    
    @pytest.fixture
    def edge_module(self):
        return EdgeScoreModule()
    
    def test_perfect_inputs_high_score(self, edge_module):
        """All inputs at 1.0 should produce high edge score."""
        output = edge_module.compute(
            ml_confidence=1.0,
            regime_strength=1.0,
            structure_quality=1.0,
            volatility_alignment=1.0,
            regime="TREND"
        )
        
        assert output.edge_score >= 0.9
        assert output.quality_tier == "A"
        assert output.boost_allowed
    
    def test_poor_inputs_low_score(self, edge_module):
        """All inputs at 0.0 should produce low edge score."""
        output = edge_module.compute(
            ml_confidence=0.0,
            regime_strength=0.0,
            structure_quality=0.0,
            volatility_alignment=0.0,
            regime="RANGE"
        )
        
        assert output.edge_score <= 0.1
        assert output.quality_tier == "D"
        assert not output.boost_allowed
    
    def test_average_inputs_mid_score(self, edge_module):
        """Average inputs should produce mid-range score."""
        output = edge_module.compute(
            ml_confidence=0.5,
            regime_strength=0.5,
            structure_quality=0.5,
            volatility_alignment=0.5,
            regime="RANGE"
        )
        
        assert 0.4 <= output.edge_score <= 0.6
        assert output.quality_tier in ["B", "C"]


class TestDangerRegimeBlocking:
    """Test DANGER regime prevents boost."""
    
    @pytest.fixture
    def edge_module(self):
        return EdgeScoreModule()
    
    def test_danger_regime_no_boost(self, edge_module):
        """DANGER regime should block size boost."""
        output = edge_module.compute(
            ml_confidence=1.0,
            regime_strength=1.0,
            structure_quality=1.0,
            volatility_alignment=1.0,
            regime="DANGER"
        )
        
        assert output.edge_score >= 0.9  # High score
        assert not output.boost_allowed   # But no boost
        assert output.danger_blocked
        assert output.size_multiplier <= 1.0
    
    def test_trend_regime_allows_boost(self, edge_module):
        """TREND regime should allow boost."""
        output = edge_module.compute(
            ml_confidence=1.0,
            regime_strength=1.0,
            structure_quality=1.0,
            volatility_alignment=1.0,
            regime="TREND"
        )
        
        assert output.boost_allowed
        assert not output.danger_blocked
        assert output.size_multiplier > 1.0


class TestStructureQualityCap:
    """Test low structure quality caps edge score."""
    
    @pytest.fixture
    def edge_module(self):
        return EdgeScoreModule()
    
    def test_low_structure_caps_edge(self, edge_module):
        """Low structure quality should cap edge score."""
        output = edge_module.compute(
            ml_confidence=1.0,
            regime_strength=1.0,
            structure_quality=0.2,  # Below 0.4 threshold
            volatility_alignment=1.0,
            regime="TREND"
        )
        
        assert output.structure_capped
        assert output.edge_score <= 0.5  # Capped at 0.50
    
    def test_good_structure_no_cap(self, edge_module):
        """Good structure quality should not cap edge score."""
        output = edge_module.compute(
            ml_confidence=1.0,
            regime_strength=1.0,
            structure_quality=0.8,  # Above 0.4 threshold
            volatility_alignment=1.0,
            regime="TREND"
        )
        
        assert not output.structure_capped
        assert output.edge_score > 0.5


class TestSizeMultiplierMapping:
    """Test edge score to size multiplier mapping."""
    
    @pytest.fixture
    def edge_module(self):
        return EdgeScoreModule()
    
    def test_d_tier_minimum_size(self, edge_module):
        """D-tier (< 0.40) should get minimum size."""
        output = edge_module.compute(
            ml_confidence=0.1,
            regime_strength=0.1,
            structure_quality=0.1,
            volatility_alignment=0.1,
            regime="RANGE"
        )
        
        assert output.edge_score < EDGE_BLOCK_THRESHOLD
        assert output.size_multiplier == SIZE_MULT_MINIMUM
        assert output.quality_tier == "D"
    
    def test_a_tier_boost(self, edge_module):
        """A-tier (> 0.75) should get boost."""
        output = edge_module.compute(
            ml_confidence=1.0,
            regime_strength=1.0,
            structure_quality=1.0,
            volatility_alignment=1.0,
            regime="TREND"
        )
        
        assert output.edge_score > EDGE_NORMAL_THRESHOLD
        assert output.size_multiplier > 1.0
        assert output.size_multiplier <= SIZE_MULT_BOOST_MAX
        assert output.quality_tier == "A"


class TestExplorationTrades:
    """Test exploration trades don't get boost."""
    
    @pytest.fixture
    def edge_module(self):
        return EdgeScoreModule()
    
    def test_exploration_no_boost(self, edge_module):
        """Exploration trades should not get boost."""
        output = edge_module.compute(
            ml_confidence=1.0,
            regime_strength=1.0,
            structure_quality=1.0,
            volatility_alignment=1.0,
            regime="TREND",
            is_exploration=True
        )
        
        assert not output.boost_allowed
        assert output.size_multiplier <= 1.0


class TestSafeguards:
    """Test EdgeScoreSafeguards."""
    
    @pytest.fixture
    def safeguards(self):
        return EdgeScoreSafeguards()
    
    def test_initial_boost_allowed(self, safeguards):
        """First boost should be allowed."""
        now = datetime.utcnow()
        allowed, reason = safeguards.can_boost(now)
        assert allowed
    
    def test_daily_limit(self, safeguards):
        """Daily boost limit should be enforced."""
        now = datetime.utcnow()
        
        # Directly set the counter to test daily limit logic
        safeguards.boost_trades_today = 3  # At limit
        safeguards.current_date = now.date()
        safeguards.consecutive_boosts = 0  # Reset to avoid consecutive limit
        safeguards.last_boost_time = now - timedelta(hours=2)  # Avoid interval limit
        
        allowed, reason = safeguards.can_boost(now)
        assert not allowed
        assert "Daily" in reason
    
    def test_consecutive_limit(self, safeguards):
        """Consecutive boost limit should be enforced."""
        now = datetime.utcnow()
        
        # Record 2 consecutive boosts (max allowed)
        safeguards.record_trade(was_boosted=True, timestamp=now)
        safeguards.record_trade(was_boosted=True, timestamp=now)
        
        allowed, reason = safeguards.can_boost(now)
        assert not allowed
        assert "Consecutive" in reason
    
    def test_non_boost_resets_consecutive(self, safeguards):
        """Non-boosted trade should reset consecutive counter."""
        now = datetime.utcnow()
        later = now + timedelta(hours=2)  # Ensure interval is met
        
        safeguards.record_trade(was_boosted=True, timestamp=now)
        safeguards.record_trade(was_boosted=False, timestamp=now)  # Resets consecutive
        
        allowed, reason = safeguards.can_boost(later)  # Check with enough time passed
        assert allowed


class TestSizingPipelineIntegration:
    """Test integration with sizing pipeline."""
    
    @pytest.fixture
    def edge_module(self):
        return EdgeScoreModule()
    
    def test_boost_reduced_by_memory(self, edge_module):
        """Boost should be reduced by memory pain."""
        output = edge_module.compute(
            ml_confidence=1.0,
            regime_strength=1.0,
            structure_quality=1.0,
            volatility_alignment=1.0,
            regime="TREND"
        )
        
        base_size = 100.0
        
        final_size, info = apply_edge_score_to_sizing(
            base_size=base_size,
            edge_output=output,
            memory_mod=0.5,  # Pain zone reducing size
            ego_mod=1.0,
            regime_mod=1.0
        )
        
        # Boost applied but then reduced by memory
        assert info['edge_adjusted'] > base_size  # Boost applied
        assert final_size < info['edge_adjusted']  # Memory reduced it
        assert info['safety_reduced']
    
    def test_floor_prevents_zero_size(self, edge_module):
        """Size should never go to zero."""
        output = edge_module.compute(
            ml_confidence=0.1,
            regime_strength=0.1,
            structure_quality=0.1,
            volatility_alignment=0.1,
            regime="DANGER"
        )
        
        base_size = 100.0
        
        final_size, info = apply_edge_score_to_sizing(
            base_size=base_size,
            edge_output=output,
            memory_mod=0.1,  # Heavy pain
            ego_mod=0.5,     # High ego
            regime_mod=0.5   # Weak regime
        )
        
        assert final_size >= base_size * 0.1  # Floor applied


class TestVolatilityAlignment:
    """Test volatility alignment calculation."""
    
    def test_optimal_volatility(self):
        """Slightly elevated ATR should give high alignment."""
        alignment = calculate_volatility_alignment(
            current_atr=1.1,
            avg_atr=1.0
        )
        assert alignment > 0.7
    
    def test_too_volatile(self):
        """Very high ATR should give low alignment."""
        alignment = calculate_volatility_alignment(
            current_atr=2.0,
            avg_atr=1.0
        )
        assert alignment < 0.5
    
    def test_too_quiet(self):
        """Very low ATR should give low alignment."""
        alignment = calculate_volatility_alignment(
            current_atr=0.5,
            avg_atr=1.0
        )
        assert alignment < 0.5


class TestStructureQuality:
    """Test structure quality calculation."""
    
    def test_perfect_structure(self):
        """All inputs at 1.0 should give high quality."""
        quality = calculate_structure_quality(
            sr_strength=1.0,
            trend_alignment=1.0,
            pattern_quality=1.0
        )
        assert quality >= 0.9
    
    def test_no_structure(self):
        """All inputs at 0.0 should give low quality."""
        quality = calculate_structure_quality(
            sr_strength=0.0,
            trend_alignment=0.0,
            pattern_quality=0.0
        )
        assert quality <= 0.1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
