"""
Unit Tests for MARK-2 Intelligence Modules

Tests:
1. Memory Module - Pain zones, regime pain, loss clusters
2. Ego Control - Overconfidence detection and modifiers
3. Regime Strength - Continuous measurement and transition
4. Integration - Combined modifiers
"""

import pytest
import sys
import os
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.backtest.memory_module import MemoryModule, PainZone, LossCluster
from src.backtest.ego_control import EgoControl
from src.backtest.regime_strength import RegimeStrength
from src.backtest.mark2_intelligence import MARK2Intelligence


# ============================================================================
# MEMORY MODULE TESTS
# ============================================================================

class TestMemoryModule:
    """Test Memory Module functionality."""
    
    @pytest.fixture
    def memory(self):
        return MemoryModule()
    
    def test_initial_state_no_pain(self, memory):
        """Fresh memory should return 1.0 modifier."""
        mod = memory.get_memory_modifier(1.1000, "RANGE", "BUY")
        assert mod == 1.0
    
    def test_loss_creates_pain_zone(self, memory):
        """Single loss should create pain zone."""
        memory.record_trade_result(
            entry_price=1.1000,
            atr=0.001,
            regime="RANGE",
            side="BUY",
            is_loss=True,
            r_multiple=-1.0
        )
        
        assert len(memory.pain_zones) == 1
        assert memory.pain_zones[0].price_center == 1.1000
    
    def test_pain_zone_reduces_size(self, memory):
        """Trading in pain zone should reduce size."""
        # Create pain zone
        memory.record_trade_result(
            entry_price=1.1000,
            atr=0.001,
            regime="RANGE",
            side="BUY",
            is_loss=True,
            r_multiple=-1.0
        )
        
        # Check modifier at same price
        mod = memory.get_memory_modifier(1.1000, "RANGE", "BUY")
        assert mod < 1.0
        assert mod >= 0.1  # Never below floor
    
    def test_loss_cluster_detection(self, memory):
        """Multiple losses should trigger cluster."""
        for i in range(4):
            memory.record_trade_result(
                entry_price=1.1000 + i * 0.001,
                atr=0.001,
                regime="RANGE",
                side="BUY",
                is_loss=True,
                r_multiple=-1.0
            )
        
        assert memory.loss_cluster.cluster_active
        assert memory.loss_cluster.count_losses() >= 4
    
    def test_win_heals_pain_zone(self, memory):
        """Win at pain zone should reduce its strength."""
        # Create pain zone
        memory.record_trade_result(
            entry_price=1.1000, atr=0.001, regime="RANGE", side="BUY",
            is_loss=True, r_multiple=-1.0
        )
        
        initial_decay = memory.pain_zones[0].decay_factor
        
        # Win at same price
        memory.record_trade_result(
            entry_price=1.1000, atr=0.001, regime="RANGE", side="BUY",
            is_loss=False, r_multiple=1.0
        )
        
        assert memory.pain_zones[0].decay_factor < initial_decay


class TestLossCluster:
    """Test Loss Cluster functionality."""
    
    def test_cluster_not_active_initially(self):
        cluster = LossCluster(buffer_size=5)
        assert not cluster.cluster_active
    
    def test_cluster_activates_at_3_losses(self):
        cluster = LossCluster(buffer_size=5)
        cluster.add_result(True)   # Loss
        cluster.add_result(False)  # Win
        cluster.add_result(True)   # Loss
        cluster.add_result(True)   # Loss
        
        assert cluster.cluster_active  # 3 losses in 4 trades


# ============================================================================
# EGO CONTROL TESTS
# ============================================================================

class TestEgoControl:
    """Test Ego Control functionality."""
    
    @pytest.fixture
    def ego(self):
        return EgoControl()
    
    def test_initial_ego_is_zero(self, ego):
        """Fresh ego should be 0."""
        assert ego.ego_score == 0.0
    
    def test_win_streak_increases_ego(self, ego):
        """Multiple wins should increase ego."""
        for i in range(5):
            ego.update_on_trade(
                is_win=True,
                confidence=0.7,
                regime="TREND",
                side="BUY"
            )
        
        assert ego.ego_score > 0.2
        assert ego.win_streak == 5
    
    def test_loss_reduces_ego(self, ego):
        """Loss should reduce ego."""
        # Build ego
        for i in range(5):
            ego.update_on_trade(is_win=True, confidence=0.7, regime="TREND", side="BUY")
        
        high_ego = ego.ego_score
        
        # Take a loss
        ego.update_on_trade(is_win=False, confidence=0.7, regime="TREND", side="BUY", loss_streak=1)
        
        assert ego.ego_score < high_ego
        assert ego.win_streak == 0
    
    def test_high_ego_reduces_size(self, ego):
        """High ego should reduce position size."""
        # Build high ego
        for i in range(8):
            ego.update_on_trade(is_win=True, confidence=0.8, regime="TREND", side="BUY")
        
        size_mod, _, _ = ego.get_ego_modifiers()
        
        assert size_mod < 1.0
        assert size_mod >= 0.5
    
    def test_high_ego_increases_cooldown(self, ego):
        """High ego should increase cooldown."""
        # Build high ego
        for i in range(8):
            ego.update_on_trade(is_win=True, confidence=0.8, regime="TREND", side="BUY")
        
        _, cooldown_mult, _ = ego.get_ego_modifiers()
        
        if ego.ego_score > 0.5:
            assert cooldown_mult > 1.0
    
    def test_same_setup_increases_tunnel_ego(self, ego):
        """Repeated same setup should increase ego."""
        for i in range(5):
            ego.update_on_trade(
                is_win=True,
                confidence=0.7,
                regime="TREND",  # Same regime
                side="BUY"       # Same side
            )
        
        assert ego.same_setup_count >= 4


# ============================================================================
# REGIME STRENGTH TESTS
# ============================================================================

class TestRegimeStrength:
    """Test Regime Strength functionality."""
    
    @pytest.fixture
    def regime(self):
        return RegimeStrength()
    
    def test_initial_regime_is_range(self, regime):
        """Default regime should be RANGE."""
        assert regime.current_regime == "RANGE"
    
    def test_high_adx_increases_trend_strength(self, regime):
        """High ADX should increase trend strength."""
        features = {
            'adx_14': 35.0,
            'atr_14': 0.001,
            'close': 1.1000,
            'sma_20': 1.0950,
        }
        
        output = regime.update(features)
        
        assert regime.smoothed_trend_strength > 0.5
    
    def test_low_adx_increases_range_strength(self, regime):
        """Low ADX should increase range strength."""
        features = {
            'adx_14': 15.0,
            'atr_14': 0.001,
            'close': 1.1000,
            'sma_20': 1.1000,
        }
        
        output = regime.update(features)
        
        assert regime.smoothed_range_strength > 0.5
    
    def test_weak_regime_reduces_size(self, regime):
        """Weak regime should reduce size modifier."""
        # Force low regime strength
        regime.regime_strength = 0.3
        
        size_mod, _, _ = regime.get_modifiers()
        
        assert size_mod < 1.0
        assert size_mod >= 0.5
    
    def test_strong_regime_allows_lower_rr(self, regime):
        """Strong regime should reduce RR requirement."""
        regime.regime_strength = 0.8
        
        _, rr_adj, _ = regime.get_modifiers()
        
        assert rr_adj < 0  # Negative = lower RR allowed
    
    def test_hysteresis_prevents_flip_flop(self, regime):
        """Regime shouldn't flip on small changes."""
        regime.current_regime = "TREND"
        regime.smoothed_trend_strength = 0.4  # Below ENTRY but above EXIT
        
        # Should stay in TREND
        regime._maybe_transition()
        
        assert regime.current_regime == "TREND"


# ============================================================================
# MARK-2 INTEGRATION TESTS
# ============================================================================

class TestMARK2Integration:
    """Test MARK-2 unified intelligence."""
    
    @pytest.fixture
    def mark2(self):
        return MARK2Intelligence()
    
    def test_initial_modifiers_are_neutral(self, mark2):
        """Fresh MARK-2 should have neutral modifiers."""
        output = mark2.get_modifiers(1.1000, "RANGE", "BUY")
        
        assert output.final_size_modifier == 1.0
        assert output.memory_mod == 1.0
        assert output.ego_mod == 1.0
    
    def test_combined_modifiers_are_multiplicative(self, mark2):
        """Modifiers should combine multiplicatively."""
        # Create some pain
        for i in range(3):
            mark2.record_trade_result(
                entry_price=1.1000,
                atr=0.001,
                regime="RANGE",
                side="BUY",
                is_win=False,
                confidence=0.7,
                r_multiple=-1.0
            )
        
        output = mark2.get_modifiers(1.1000, "RANGE", "BUY")
        
        # Combined should be product of individuals
        expected = output.memory_mod * output.ego_mod * output.regime_mod
        expected = max(0.1, expected)  # Floor
        
        assert abs(output.final_size_modifier - expected) < 0.01
    
    def test_size_never_goes_to_zero(self, mark2):
        """Size modifier should never be zero."""
        # Create maximum pain
        for i in range(10):
            mark2.record_trade_result(
                entry_price=1.1000,
                atr=0.001,
                regime="DANGER",
                side="BUY",
                is_win=False,
                confidence=0.7,
                r_multiple=-2.0,
                loss_streak=i+1
            )
        
        output = mark2.get_modifiers(1.1000, "DANGER", "BUY")
        
        assert output.final_size_modifier >= 0.1
    
    def test_reset_clears_all_modules(self, mark2):
        """Reset should clear all MARK-2 state."""
        # Create some state
        mark2.record_trade_result(
            entry_price=1.1000, atr=0.001, regime="RANGE", side="BUY",
            is_win=False, confidence=0.7, r_multiple=-1.0
        )
        
        mark2.reset()
        
        output = mark2.get_modifiers(1.1000, "RANGE", "BUY")
        assert output.final_size_modifier == 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
