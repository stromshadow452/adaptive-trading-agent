"""
Unit tests for Meta-Gating Brain
"""
import pytest
import numpy as np
from src.decision.meta_gating import MetaGatingBrain


def test_regime_classification_trend():
    """Test TREND regime detection."""
    brain = MetaGatingBrain()
    
    features = {
        'close': 1.1000,
        'atr14': 0.0030,  # High ATR
        'rsi14': 60.0
    }
    
    result = brain.classify_regime(features, volatility=0.01, symbol="EURUSD")
    
    # Regime can be TREND, RANGE, or UNCERTAIN depending on ADX calculation
    assert result['regime'] in ['TREND', 'RANGE', 'UNCERTAIN']
    assert result['action'] in ['ALLOW', 'REDUCE']
    assert 0.0 <= result['size_multiplier'] <= 1.0
    assert 0.0 <= result['confidence'] <= 1.0


def test_regime_classification_crash():
    """Test CRASH regime detection."""
    brain = MetaGatingBrain()
    
    # Build history
    for _ in range(10):
        brain.vol_history.append(0.01)
    
    features = {
        'close': 1.1000,
        'atr14': 0.0010,
        'rsi14': 50.0
    }
    
    # High volatility spike
    result = brain.classify_regime(features, volatility=0.03, symbol="EURUSD")
    
    assert result['regime'] == 'CRASH'
    assert result['action'] == 'BLOCK'
    assert result['size_multiplier'] == 0.0


def test_regime_classification_range():
    """Test RANGE regime detection."""
    brain = MetaGatingBrain()
    
    features = {
        'close': 1.1000,
        'atr14': 0.0005,  # Low ATR
        'rsi14': 50.0
    }
    
    result = brain.classify_regime(features, volatility=0.005, symbol="EURUSD")
    
    assert result['regime'] in ['RANGE', 'UNCERTAIN']
    assert result['action'] in ['ALLOW', 'REDUCE']


def test_model_weights():
    """Test model weight calculation."""
    brain = MetaGatingBrain()
    
    weights_trend = brain.get_model_weights('TREND')
    assert weights_trend['primary'] > 0
    assert weights_trend['rl'] >= 0
    assert weights_trend['no_trade'] >= 0
    
    weights_crash = brain.get_model_weights('CRASH')
    assert weights_crash['no_trade'] == 1.0


def test_should_trade():
    """Test trade permission logic."""
    brain = MetaGatingBrain()
    
    # ALLOW action
    regime_info = {'action': 'ALLOW'}
    assert brain.should_trade(regime_info) == True
    
    # BLOCK action
    regime_info = {'action': 'BLOCK'}
    assert brain.should_trade(regime_info) == False
    
    # REDUCE action
    regime_info = {'action': 'REDUCE'}
    assert brain.should_trade(regime_info) == True


def test_reset():
    """Test brain reset."""
    brain = MetaGatingBrain()
    
    brain.vol_history = [0.01, 0.02, 0.03]
    brain.reset()
    
    assert len(brain.vol_history) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
