"""
Unit tests for Portfolio Brain
"""
import pytest
import pandas as pd
import numpy as np
from src.risk.portfolio import PortfolioBrain


def test_correlation_matrix_update():
    """Test correlation matrix calculation."""
    brain = PortfolioBrain()
    
    # Create sample price data
    dates = pd.date_range('2024-01-01', periods=100, freq='15min')
    price_data = {
        'EURUSD': pd.DataFrame({
            'timestamp': dates,
            'close': np.random.randn(100).cumsum() + 1.1
        }),
        'GBPUSD': pd.DataFrame({
            'timestamp': dates,
            'close': np.random.randn(100).cumsum() + 1.3
        })
    }
    
    brain.update_correlation_matrix(price_data)
    
    assert brain.correlation_matrix is not None
    assert brain.correlation_matrix.shape == (2, 2)


def test_get_correlation():
    """Test correlation retrieval."""
    brain = PortfolioBrain()
    
    # No matrix yet
    corr = brain.get_correlation('EURUSD', 'GBPUSD')
    assert corr == 0.0


def test_adjust_size_no_positions():
    """Test size adjustment with no open positions."""
    brain = PortfolioBrain()
    
    adjusted = brain.adjust_size(
        symbol='EURUSD',
        base_size=0.1,
        open_positions={},
        csv_data=None
    )
    
    assert adjusted == 0.1  # No adjustment


def test_adjust_size_high_portfolio_risk():
    """Test size reduction for high portfolio risk."""
    brain = PortfolioBrain(max_portfolio_risk=0.01)
    
    # Add positions to exceed risk limit
    brain.open_positions = {
        'EURUSD': {'size': 0.1, 'price': 1.1, 'side': 'buy', 'risk': 0.015}
    }
    
    adjusted = brain.adjust_size(
        symbol='GBPUSD',
        base_size=0.1,
        open_positions=brain.open_positions,
        csv_data=None
    )
    
    assert adjusted < 0.1  # Size reduced


def test_position_tracking():
    """Test position add/remove."""
    brain = PortfolioBrain()
    
    brain.add_position('EURUSD', 0.1, 1.1000, 'buy')
    assert 'EURUSD' in brain.open_positions
    assert brain.open_positions['EURUSD']['size'] == 0.1
    
    brain.remove_position('EURUSD')
    assert 'EURUSD' not in brain.open_positions


def test_portfolio_summary():
    """Test portfolio summary generation."""
    brain = PortfolioBrain()
    
    brain.add_position('EURUSD', 0.1, 1.1000, 'buy')
    brain.add_position('GBPUSD', 0.05, 1.3000, 'sell')
    
    summary = brain.get_portfolio_summary()
    
    assert summary['num_positions'] == 2
    assert summary['total_risk'] > 0
    assert 'EURUSD' in summary['symbols']
    assert 'GBPUSD' in summary['symbols']


def test_reset():
    """Test portfolio reset."""
    brain = PortfolioBrain()
    
    brain.add_position('EURUSD', 0.1, 1.1000, 'buy')
    brain.reset()
    
    assert len(brain.open_positions) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
