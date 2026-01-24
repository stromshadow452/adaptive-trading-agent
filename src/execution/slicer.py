"""
Execution Slicer - Stage 11 Extension

TWAP/VWAP simulation with slippage modeling for large orders.
Splits orders into child slices and simulates execution over multiple CSV candles.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ExecutionSlice:
    """Single execution slice."""
    slice_id: int
    size: float
    target_price: float
    actual_price: float
    slippage: float
    timestamp: Optional[pd.Timestamp] = None


@dataclass
class ExecutionResult:
    """Complete execution result."""
    symbol: str
    side: str
    total_size: float
    avg_price: float
    total_slippage: float
    num_slices: int
    slices: List[ExecutionSlice]
    execution_time_ms: float


class ExecutionSlicer:
    """
    TWAP/VWAP order slicer with slippage simulation.
    
    Splits large orders into smaller child orders and simulates
    execution over multiple CSV candles with realistic slippage.
    """
    
    def __init__(self,
                 default_spread_bps: float = 1.0,
                 slippage_factor: float = 0.5,
                 min_slice_size: float = 0.01):
        """
        Initialize Execution Slicer.
        
        Args:
            default_spread_bps: Default bid-ask spread in basis points
            slippage_factor: Slippage multiplier (0.5 = 50% of spread)
            min_slice_size: Minimum size per slice
        """
        self.default_spread_bps = default_spread_bps
        self.slippage_factor = slippage_factor
        self.min_slice_size = min_slice_size
    
    def create_twap_plan(self,
                        symbol: str,
                        side: str,
                        size: float,
                        num_slices: int,
                        csv_data: Optional[pd.DataFrame] = None) -> List[Dict]:
        """
        Create TWAP (Time-Weighted Average Price) execution plan.
        
        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            size: Total order size
            num_slices: Number of slices to split into
            csv_data: Future price data for execution simulation
            
        Returns:
            List of slice plans with target prices
        """
        if size <= 0 or num_slices <= 0:
            return []
        
        # Calculate slice size
        slice_size = size / num_slices
        if slice_size < self.min_slice_size:
            logger.warning(f"[SLICER] Slice size {slice_size:.4f} below minimum, "
                          f"reducing num_slices")
            num_slices = max(1, int(size / self.min_slice_size))
            slice_size = size / num_slices
        
        # Create slice plan
        slices = []
        for i in range(num_slices):
            # Use future prices if available
            if csv_data is not None and len(csv_data) > i:
                target_price = csv_data.iloc[i]['close']
            else:
                target_price = None  # Will use current price
            
            slices.append({
                'slice_id': i,
                'size': slice_size,
                'target_price': target_price,
                'timestamp': csv_data.index[i] if csv_data is not None and len(csv_data) > i else None
            })
        
        logger.info(f"[SLICER] Created TWAP plan for {symbol}: {num_slices} slices of {slice_size:.4f}")
        
        return slices
    
    def create_vwap_plan(self,
                        symbol: str,
                        side: str,
                        size: float,
                        csv_data: pd.DataFrame) -> List[Dict]:
        """
        Create VWAP (Volume-Weighted Average Price) execution plan.
        
        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            size: Total order size
            csv_data: Future price data with volume
            
        Returns:
            List of slice plans weighted by volume
        """
        if size <= 0 or csv_data is None or len(csv_data) == 0:
            return []
        
        # Calculate volume weights
        if 'volume' in csv_data.columns:
            volumes = csv_data['volume'].values
            total_volume = volumes.sum()
            
            if total_volume > 0:
                volume_weights = volumes / total_volume
            else:
                # Fallback to equal weights
                volume_weights = np.ones(len(csv_data)) / len(csv_data)
        else:
            # No volume data, use equal weights (TWAP)
            volume_weights = np.ones(len(csv_data)) / len(csv_data)
        
        # Create slices
        slices = []
        for i, weight in enumerate(volume_weights):
            slice_size = size * weight
            
            if slice_size < self.min_slice_size:
                continue
            
            slices.append({
                'slice_id': i,
                'size': slice_size,
                'target_price': csv_data.iloc[i]['close'],
                'timestamp': csv_data.index[i]
            })
        
        logger.info(f"[SLICER] Created VWAP plan for {symbol}: {len(slices)} slices")
        
        return slices
    
    def simulate_slippage(self,
                         target_price: float,
                         size: float,
                         side: str,
                         spread_bps: Optional[float] = None) -> float:
        """
        Simulate execution slippage.
        
        Args:
            target_price: Target execution price
            size: Order size
            side: 'buy' or 'sell'
            spread_bps: Bid-ask spread in basis points
            
        Returns:
            Actual execution price (with slippage)
        """
        if spread_bps is None:
            spread_bps = self.default_spread_bps
        
        # Calculate spread in price units
        spread = target_price * (spread_bps / 10000)
        
        # Slippage = fraction of spread based on size
        # Larger orders get more slippage
        size_impact = min(size * 10, 1.0)  # Cap at 100%
        slippage = spread * self.slippage_factor * size_impact
        
        # Apply slippage direction
        if side == 'buy':
            actual_price = target_price + slippage  # Pay more
        else:
            actual_price = target_price - slippage  # Receive less
        
        return actual_price
    
    def execute_twap(self,
                    symbol: str,
                    side: str,
                    size: float,
                    csv_data: Optional[pd.DataFrame] = None,
                    num_slices: int = 5,
                    current_price: float = 1.0) -> ExecutionResult:
        """
        Execute TWAP order with slippage simulation.
        
        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            size: Total order size
            csv_data: Future price data
            num_slices: Number of slices
            current_price: Current market price (fallback)
            
        Returns:
            ExecutionResult with all slice details
        """
        import time
        start_time = time.time()
        
        # Create TWAP plan
        slice_plan = self.create_twap_plan(symbol, side, size, num_slices, csv_data)
        
        # Execute each slice
        executed_slices = []
        total_cost = 0.0
        total_slippage = 0.0
        
        for plan in slice_plan:
            target_price = plan['target_price'] if plan['target_price'] is not None else current_price
            
            # Simulate execution with slippage
            actual_price = self.simulate_slippage(
                target_price=target_price,
                size=plan['size'],
                side=side
            )
            
            slippage = abs(actual_price - target_price)
            
            executed_slices.append(ExecutionSlice(
                slice_id=plan['slice_id'],
                size=plan['size'],
                target_price=target_price,
                actual_price=actual_price,
                slippage=slippage,
                timestamp=plan['timestamp']
            ))
            
            total_cost += actual_price * plan['size']
            total_slippage += slippage * plan['size']
        
        # Calculate average price
        avg_price = total_cost / size if size > 0 else 0.0
        
        execution_time = (time.time() - start_time) * 1000  # ms
        
        result = ExecutionResult(
            symbol=symbol,
            side=side,
            total_size=size,
            avg_price=avg_price,
            total_slippage=total_slippage,
            num_slices=len(executed_slices),
            slices=executed_slices,
            execution_time_ms=execution_time
        )
        
        logger.info(f"[SLICER] Executed TWAP: {symbol} {side} {size:.4f} @ {avg_price:.5f}, "
                   f"slippage={total_slippage:.6f}, slices={len(executed_slices)}")
        
        return result
    
    def execute_vwap(self,
                    symbol: str,
                    side: str,
                    size: float,
                    csv_data: pd.DataFrame) -> ExecutionResult:
        """
        Execute VWAP order with slippage simulation.
        
        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            size: Total order size
            csv_data: Future price data with volume
            
        Returns:
            ExecutionResult with all slice details
        """
        import time
        start_time = time.time()
        
        # Create VWAP plan
        slice_plan = self.create_vwap_plan(symbol, side, size, csv_data)
        
        # Execute each slice
        executed_slices = []
        total_cost = 0.0
        total_slippage = 0.0
        
        for plan in slice_plan:
            # Simulate execution with slippage
            actual_price = self.simulate_slippage(
                target_price=plan['target_price'],
                size=plan['size'],
                side=side
            )
            
            slippage = abs(actual_price - plan['target_price'])
            
            executed_slices.append(ExecutionSlice(
                slice_id=plan['slice_id'],
                size=plan['size'],
                target_price=plan['target_price'],
                actual_price=actual_price,
                slippage=slippage,
                timestamp=plan['timestamp']
            ))
            
            total_cost += actual_price * plan['size']
            total_slippage += slippage * plan['size']
        
        # Calculate average price
        avg_price = total_cost / size if size > 0 else 0.0
        
        execution_time = (time.time() - start_time) * 1000  # ms
        
        result = ExecutionResult(
            symbol=symbol,
            side=side,
            total_size=size,
            avg_price=avg_price,
            total_slippage=total_slippage,
            num_slices=len(executed_slices),
            slices=executed_slices,
            execution_time_ms=execution_time
        )
        
        logger.info(f"[SLICER] Executed VWAP: {symbol} {side} {size:.4f} @ {avg_price:.5f}, "
                   f"slippage={total_slippage:.6f}, slices={len(executed_slices)}")
        
        return result


# Factory function
def create_execution_slicer(**kwargs) -> ExecutionSlicer:
    """Create and return ExecutionSlicer instance."""
    return ExecutionSlicer(**kwargs)
