"""
Unit tests for Execution Slicer
"""
import pytest
import pandas as pd
import numpy as np
from src.execution.slicer import ExecutionSlicer, ExecutionResult


def test_create_twap_plan():
    """Test TWAP plan generation."""
    slicer = ExecutionSlicer()
    
    plan = slicer.create_twap_plan(
        symbol='EURUSD',
        side='buy',
        size=0.5,
        num_slices=5,
        csv_data=None
    )
    
    assert len(plan) == 5
    assert all(p['size'] == 0.1 for p in plan)


def test_create_vwap_plan():
    """Test VWAP plan generation."""
    # Use smaller min_slice_size to avoid filtering
    slicer = ExecutionSlicer(min_slice_size=0.001)
    
    # Create sample data with volume (deterministic)
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=10, freq='15min')
    csv_data = pd.DataFrame({
        'close': np.random.randn(10) + 1.1,
        'volume': np.random.randint(100, 1000, 10)
    }, index=dates)
    
    plan = slicer.create_vwap_plan(
        symbol='EURUSD',
        side='buy',
        size=0.5,
        csv_data=csv_data
    )
    
    assert len(plan) > 0
    total_size = sum(p['size'] for p in plan)
    # With smaller min_slice_size, total should be close to 0.5
    assert abs(total_size - 0.5) < 0.05  # Allow 5% tolerance


def test_simulate_slippage():
    """Test slippage simulation."""
    slicer = ExecutionSlicer(default_spread_bps=1.0, slippage_factor=0.5)
    
    # Buy order should have positive slippage
    actual_buy = slicer.simulate_slippage(
        target_price=1.1000,
        size=0.1,
        side='buy',
        spread_bps=1.0
    )
    assert actual_buy > 1.1000
    
    # Sell order should have negative slippage
    actual_sell = slicer.simulate_slippage(
        target_price=1.1000,
        size=0.1,
        side='sell',
        spread_bps=1.0
    )
    assert actual_sell < 1.1000


def test_execute_twap():
    """Test TWAP execution."""
    slicer = ExecutionSlicer()
    
    result = slicer.execute_twap(
        symbol='EURUSD',
        side='buy',
        size=0.5,
        csv_data=None,
        num_slices=5,
        current_price=1.1000
    )
    
    assert isinstance(result, ExecutionResult)
    assert result.total_size == 0.5
    assert result.num_slices == 5
    assert result.avg_price > 0
    assert result.total_slippage >= 0


def test_execute_vwap():
    """Test VWAP execution."""
    slicer = ExecutionSlicer(min_slice_size=0.001)
    
    # Create sample data (deterministic)
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=10, freq='15min')
    csv_data = pd.DataFrame({
        'close': [1.1000 + i*0.0001 for i in range(10)],
        'volume': np.random.randint(100, 1000, 10)
    }, index=dates)
    
    result = slicer.execute_vwap(
        symbol='EURUSD',
        side='buy',
        size=0.5,
        csv_data=csv_data
    )
    
    assert isinstance(result, ExecutionResult)
    assert result.total_size <= 0.5
    assert result.total_size > 0.4  # Should be close to 0.5
    assert result.num_slices > 0
    assert result.avg_price > 0


def test_min_slice_size():
    """Test minimum slice size enforcement."""
    slicer = ExecutionSlicer(min_slice_size=0.05)
    
    # Try to create plan with too many slices
    plan = slicer.create_twap_plan(
        symbol='EURUSD',
        side='buy',
        size=0.1,
        num_slices=10,  # Would create 0.01 slices
        csv_data=None
    )
    
    # Should reduce num_slices to meet min_slice_size
    assert len(plan) <= 2  # 0.1 / 0.05 = 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
