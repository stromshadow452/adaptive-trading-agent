"""
Unit Tests for RL Brain Fallback (Stage 5)

Tests the grey-zone decision logic:
- primary_conf >= 0.70 → EXECUTE_PRIMARY
- primary_conf <= 0.40 → SKIPPED_LOW_CONF
- 0.40 < primary_conf < 0.70 → Try RL fallback
"""
import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
import sys
import os

# Add repo root to path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.agents.finrl_adapter import FinRLAdapter, load_finrl_adapter


class TestFinRLAdapter:
    """Test FinRLAdapter class functionality"""
    
    def test_adapter_unavailable_if_no_model_path(self):
        """RL adapter should be unavailable if model path is None"""
        mock_registry = Mock()
        adapter = FinRLAdapter(
            model_path=None,
            registry=mock_registry,
            primary_meta={}
        )
        
        assert not adapter.is_available()
        assert adapter.model is None
    
    def test_adapter_unavailable_if_model_path_not_exists(self):
        """RL adapter should be unavailable if model file doesn't exist"""
        mock_registry = Mock()
        adapter = FinRLAdapter(
            model_path="/nonexistent/model.joblib",
            registry=mock_registry,
            primary_meta={}
        )
        
        assert not adapter.is_available()
        assert adapter.model is None
    
    def test_adapter_feature_parity_check(self, tmp_path):
        """RL adapter should check feature parity with primary model"""
        # Create dummy model file
        model_file = tmp_path / "rl_model.joblib"
        model_file.touch()
        sig_file = tmp_path / "rl_model.joblib.sig"
        sig_file.write_text('{"hmac": "dummy"}')
        
        mock_registry = Mock()
        mock_model = Mock()
        mock_model.predict = Mock(return_value=(np.array([1]), None))
        
        # Simulate feature hash mismatch
        mock_registry.load_model_from_path = Mock(return_value=(
            mock_model,
            {"feature_hash": "rl_hash_123"}
        ))
        
        primary_meta = {"feature_hash": "primary_hash_456"}
        
        adapter = FinRLAdapter(
            model_path=str(model_file),
            registry=mock_registry,
            primary_meta=primary_meta
        )
        
        # Should be unavailable due to feature mismatch
        assert not adapter.is_available()
    
    def test_adapter_predict_proba_with_ppo_model(self, tmp_path):
        """RL adapter should handle PPO model predictions"""
        model_file = tmp_path / "rl_model.joblib"
        model_file.touch()
        sig_file = tmp_path / "rl_model.joblib.sig"
        sig_file.write_text('{"hmac": "dummy"}')
        
        mock_registry = Mock()
        mock_model = Mock()
        # PPO returns (action, states)
        mock_model.predict = Mock(return_value=(np.array([1]), None))
        
        feature_hash = "same_hash_123"
        mock_registry.load_model_from_path = Mock(return_value=(
            mock_model,
            {"feature_hash": feature_hash}
        ))
        
        adapter = FinRLAdapter(
            model_path=str(model_file),
            registry=mock_registry,
            primary_meta={"feature_hash": feature_hash}
        )
        
        assert adapter.is_available()
        
        # Test prediction
        features = np.array([0.1, 0.2, 0.3])
        conf, action = adapter.predict_proba(features)
        
        assert 0.0 <= conf <= 1.0
        assert action in [-1, 0, 1]
        assert action == 1  # Buy action


class TestRLFallbackDecisionLogic:
    """Test grey-zone decision logic in decide_and_execute"""
    
    @pytest.fixture
    def mock_plan(self):
        return {
            "symbol": "EURUSD",
            "tf": "M15",
            "side": "buy",
            "price": 1.1000,
            "size": 0.01
        }
    
    @pytest.fixture
    def mock_finrl_adapter(self):
        """Mock FinRL adapter that returns high confidence"""
        adapter = Mock()
        adapter.is_available = Mock(return_value=True)
        adapter.predict_proba = Mock(return_value=(0.75, 1))  # High conf, buy action
        return adapter
    
    def test_primary_high_no_rl_called(self, mock_plan):
        """
        Test: primary_conf >= 0.70 → EXECUTE_PRIMARY
        RL adapter should NOT be called
        """
        from tools.executor import decide_and_execute
        
        mock_finrl_adapter = Mock()
        mock_finrl_adapter.is_available = Mock(return_value=True)
        
        # Mock get_confidences to return high primary confidence
        with patch('tools.executor.get_confidences', return_value=(0.85, 0.0)):
            decision, size_factor, reason = decide_and_execute(
                plan=mock_plan,
                primary_model=None,
                finrl_model=None,
                primary_features=[],
                csv_price_dirs=[],
                primary_thresh=None,
                finrl_thresh=None,
                grey_zone=0.05,
                primary_high=0.70,
                primary_block=0.40,
                rl_size_reduction=0.4,
                finrl_adapter=mock_finrl_adapter,
                verbose=False
            )
        
        assert decision == "EXECUTE_PRIMARY"
        assert size_factor == 1.0
        assert "0.85" in reason
        # RL adapter should NOT have been called
        mock_finrl_adapter.predict_proba.assert_not_called()
    
    def test_primary_low_blocked(self, mock_plan):
        """
        Test: primary_conf <= 0.40 → SKIPPED_LOW_CONF
        Trade should be blocked, RL ignored
        """
        from tools.executor import decide_and_execute
        
        mock_finrl_adapter = Mock()
        mock_finrl_adapter.is_available = Mock(return_value=True)
        
        # Mock get_confidences to return low primary confidence
        with patch('tools.executor.get_confidences', return_value=(0.30, 0.0)):
            decision, size_factor, reason = decide_and_execute(
                plan=mock_plan,
                primary_model=None,
                finrl_model=None,
                primary_features=[],
                csv_price_dirs=[],
                primary_thresh=None,
                finrl_thresh=None,
                grey_zone=0.05,
                primary_high=0.70,
                primary_block=0.40,
                rl_size_reduction=0.4,
                finrl_adapter=mock_finrl_adapter,
                verbose=False
            )
        
        assert decision == "SKIPPED_LOW_CONF"
        assert size_factor is None
        assert "0.30" in reason
        # RL adapter should NOT have been called
        mock_finrl_adapter.predict_proba.assert_not_called()
    
    def test_grey_zone_rl_fallback_success(self, mock_plan, mock_finrl_adapter):
        """
        Test: 0.40 < primary_conf < 0.70 AND RL available
        → EXECUTE_FINRL_FALLBACK with size reduction
        """
        from tools.executor import decide_and_execute
        
        # Mock get_confidences to return grey-zone primary confidence
        with patch('tools.executor.get_confidences', return_value=(0.55, 0.0)):
            with patch('tools.executor.build_feature_vector_from_csv', return_value=(np.array([0.1, 0.2]), "2025-11-20T10:00:00Z")):
                decision, size_factor, reason = decide_and_execute(
                    plan=mock_plan,
                    primary_model=None,
                    finrl_model=None,
                    primary_features=["close", "rsi14"],
                    csv_price_dirs=["temp_prices"],
                    primary_thresh=None,
                    finrl_thresh=0.65,
                    grey_zone=0.05,
                    primary_high=0.70,
                    primary_block=0.40,
                    rl_size_reduction=0.4,
                    finrl_adapter=mock_finrl_adapter,
                    verbose=False
                )
        
        assert decision == "EXECUTE_FINRL_FALLBACK"
        assert size_factor == 0.4  # Size reduction applied
        assert "0.55" in reason  # Primary conf in grey zone
        assert "0.75" in reason  # RL conf
        # RL adapter SHOULD have been called
        mock_finrl_adapter.predict_proba.assert_called_once()
    
    def test_grey_zone_rl_unavailable(self, mock_plan):
        """
        Test: Grey zone but RL adapter unavailable
        → SKIPPED_LOW_CONF
        """
        from tools.executor import decide_and_execute
        
        # Mock get_confidences to return grey-zone primary confidence
        with patch('tools.executor.get_confidences', return_value=(0.55, 0.0)):
            decision, size_factor, reason = decide_and_execute(
                plan=mock_plan,
                primary_model=None,
                finrl_model=None,
                primary_features=[],
                csv_price_dirs=[],
                primary_thresh=None,
                finrl_thresh=None,
                grey_zone=0.05,
                primary_high=0.70,
                primary_block=0.40,
                rl_size_reduction=0.4,
                finrl_adapter=None,  # RL unavailable
                verbose=False
            )
        
        assert decision == "SKIPPED_LOW_CONF"
        assert size_factor is None
        assert "grey zone" in reason.lower()
        assert "unavailable" in reason.lower()
    
    def test_grey_zone_rl_low_confidence(self, mock_plan):
        """
        Test: Grey zone but RL confidence too low
        → SKIPPED_LOW_CONF
        """
        from tools.executor import decide_and_execute
        
        mock_finrl_adapter = Mock()
        mock_finrl_adapter.is_available = Mock(return_value=True)
        mock_finrl_adapter.predict_proba = Mock(return_value=(0.50, 1))  # Low RL conf
        
        # Mock get_confidences to return grey-zone primary confidence
        with patch('tools.executor.get_confidences', return_value=(0.55, 0.0)):
            with patch('tools.executor.build_feature_vector_from_csv', return_value=(np.array([0.1, 0.2]), "2025-11-20T10:00:00Z")):
                decision, size_factor, reason = decide_and_execute(
                    plan=mock_plan,
                    primary_model=None,
                    finrl_model=None,
                    primary_features=["close", "rsi14"],
                    csv_price_dirs=["temp_prices"],
                    primary_thresh=None,
                    finrl_thresh=0.65,  # RL threshold
                    grey_zone=0.05,
                    primary_high=0.70,
                    primary_block=0.40,
                    rl_size_reduction=0.4,
                    finrl_adapter=mock_finrl_adapter,
                    verbose=False
                )
        
        # RL conf (0.50) < threshold (0.65) → fallback to skip
        assert decision == "SKIPPED_LOW_CONF"
        assert size_factor is None


class TestJARVISGuardsWithRL:
    """Test that JARVIS guards still work with RL fallback"""
    
    def test_circuit_breaker_blocks_rl_fallback(self):
        """Circuit breaker should block RL fallback trades"""
        from src.risk.circuit_breaker import CircuitBreaker
        
        cb = CircuitBreaker()
        cb.trip("EURUSD", reason="Test circuit breaker")
        
        # Circuit breaker should raise error
        with pytest.raises(RuntimeError, match="CIRCUIT BREAKER TRIPPED"):
            cb.check_gate("EURUSD")
    
    def test_feature_parity_prevents_rl_loading(self, tmp_path):
        """Feature hash mismatch should prevent RL model loading"""
        model_file = tmp_path / "rl_model.joblib"
        model_file.touch()
        sig_file = tmp_path / "rl_model.joblib.sig"
        sig_file.write_text('{"hmac": "dummy"}')
        
        mock_registry = Mock()
        mock_registry.load_model_from_path = Mock(return_value=(
            Mock(),
            {"feature_hash": "rl_different_hash"}
        ))
        
        adapter = FinRLAdapter(
            model_path=str(model_file),
            registry=mock_registry,
            primary_meta={"feature_hash": "primary_hash"}
        )
        
        # Should be unavailable due to feature mismatch
        assert not adapter.is_available()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
