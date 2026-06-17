"""
Shared Execution Service

Unified execution for Adaptive Trading OS.

Handles:
- Order routing
- Paper/Live abstraction
- Slippage modeling
- Spread modeling
- Retry logic
- Execution adapters
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import asyncio

from ..interfaces import (
    Decision, ExecutionResult, MarketData,
    ExecutionServiceInterface, SharedService,
    ExecutionError
)

logger = logging.getLogger(__name__)


@dataclass
class ExecutionConfig:
    """Execution configuration."""
    mode: str = "paper"  # replay, paper, live
    
    # Slippage
    slippage_model: str = "fixed"  # fixed, variable, adaptive
    slippage_bps: float = 1.0  # 1 basis point
    slippage_std: float = 0.5  # Standard deviation
    
    # Spread
    spread_model: str = "fixed"  # fixed, variable
    spread_bps: float = 2.0  # 2 basis points
    
    # Costs
    commission_per_lot: float = 0.0
    financing_rate: float = 0.0
    
    # Retry
    max_retries: int = 3
    retry_delay_ms: int = 100
    
    # Session
    allow_pre_market: bool = False
    allow_after_hours: bool = False
    
    # Liquidity
    min_volume: float = 0  # Minimum volume to execute
    max_order_size: float = 100  # Max lots


@dataclass
class Order:
    """Order structure."""
    order_id: str
    timestamp: datetime
    symbol: str
    side: str  # buy, sell
    size: float
    order_type: str  # market, limit, stop
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    
    # Status
    status: str = "pending"  # pending, open, filled, partial, rejected
    filled_price: Optional[float] = None
    filled_size: float = 0.0
    slippage: float = 0.0
    commission: float = 0.0
    
    # Metadata
    source: str = ""  # Which alpha pod
    decision: Optional[Decision] = None
    metadata: Dict = field(default_factory=dict)


class ExecutionAdapter(ABC):
    """Abstract execution adapter."""
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to broker."""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Disconnect from broker."""
        pass
    
    @abstractmethod
    async def send_order(self, order: Order) -> Order:
        """Send order."""
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order."""
        pass
    
    @abstractmethod
    async def get_positions(self) -> List[Dict]:
        """Get current positions."""
        pass
    
    @abstractmethod
    async def get_account(self) -> Dict:
        """Get account info."""
        pass


class PaperTradingAdapter(ExecutionAdapter):
    """Paper trading adapter."""
    
    def __init__(self, config: Dict):
        self.config = config
        self._positions: Dict[str, Dict] = {}
        self._orders: Dict[str, Order] = {}
        self._connected = False
        
    async def connect(self) -> bool:
        self._connected = True
        logger.info("Paper trading adapter connected")
        return True
    
    async def disconnect(self):
        self._connected = False
    
    async def send_order(self, order: Order) -> Order:
        """Simulate order fill."""
        if not self._connected:
            raise ExecutionError("Not connected")
        
        # Simulate fill at current market price
        # (would use data service in real implementation)
        order.filled_price = self._get_current_price(order.symbol)
        order.filled_size = order.size
        order.status = "filled"
        
        # Update positions
        self._update_positions(order)
        
        return order
    
    def _get_current_price(self, symbol: str) -> float:
        """Get current price (simplified)."""
        # Would use data service
        return 1.0
    
    def _update_positions(self, order: Order):
        """Update positions."""
        symbol = order.symbol
        
        if symbol not in self._positions:
            self._positions[symbol] = {
                'size': 0,
                'avg_price': 0
            }
        
        pos = self._positions[symbol]
        
        if order.side == 'buy':
            pos['size'] += order.filled_size
        else:
            pos['size'] -= order.filled_size
    
    async def cancel_order(self, order_id: str) -> bool:
        if order_id in self._orders:
            self._orders[order_id].status = "cancelled"
            return True
        return False
    
    async def get_positions(self) -> List[Dict]:
        return [
            {'symbol': s, **p}
            for s, p in self._positions.items()
        ]
    
    async def get_account(self) -> Dict:
        return {
            'balance': 10000.0,
            'equity': 10000.0,
            'margin': 0.0
        }


class SharedExecutionService(ExecutionServiceInterface):
    """
    Shared execution service for all alpha pods.
    
    Centralized order routing with:
    - Adapter abstraction
    - Slippage modeling
    - Retry logic
    - Execution tracking
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize execution service.
        
        Args:
            config: Configuration
        """
        self.config = ExecutionConfig(**config.get('execution', {}))
        
        # Adapter
        self.adapter: Optional[ExecutionAdapter] = None
        self._adapter_type = self.config.mode
        
        # Order tracking
        self._orders: Dict[str, Order] = {}
        self._order_count = 0
        
        # Execution stats
        self._stats = {
            'total_orders': 0,
            'filled_orders': 0,
            'rejected_orders': 0,
            'avg_slippage_bps': 0.0,
            'total_commission': 0.0
        }
        
        logger.info(f"SharedExecutionService initialized")
        logger.info(f"  Mode: {self.config.mode}")
        logger.info(f"  Slippage: {self.config.slippage_bps} bps")
    
    async def initialize(self) -> bool:
        """Initialize service."""
        try:
            logger.info("Initializing execution service...")
            
            # Create adapter
            self.adapter = self._create_adapter()
            
            # Connect
            if not await self.adapter.connect():
                raise ExecutionError("Failed to connect adapter")
            
            logger.info("Execution service initialized")
            return True
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False
    
    async def health_check(self) -> bool:
        """Health check."""
        if self.adapter is None:
            return False
        try:
            account = await self.adapter.get_account()
            return True
        except:
            return False
    
    async def shutdown(self):
        """Shutdown service."""
        if self.adapter:
            await self.adapter.disconnect()
        logger.info("Execution service shutdown")
    
    def _create_adapter(self) -> ExecutionAdapter:
        """Create execution adapter."""
        if self._adapter_type == "paper":
            return PaperTradingAdapter(self.config.__dict__)
        elif self._adapter_type == "replay":
            return ReplayAdapter(self.config.__dict__)
        elif self._adapter_type == "live":
            return LiveAdapter(self.config.__dict__)
        else:
            raise ValueError(f"Unknown adapter type: {self._adapter_type}")
    
    # ========================================================================
    # CORE EXECUTION
    # ========================================================================
    
    async def execute(self, decision: Decision) -> ExecutionResult:
        """
        Execute decision.
        
        Args:
            decision: Trading decision
            
        Returns:
            ExecutionResult
        """
        try:
            # 1. Pre-execution checks
            if not self._pre_execution_checks(decision):
                return self._create_rejected_result(
                    decision, "PRE_EXECUTION_FAILED"
                )
            
            # 2. Create order
            order = self._create_order(decision)
            
            # 3. Send with retry
            filled_order = await self._send_with_retry(order)
            
            # 4. Post-execution
            result = self._create_result(decision, filled_order)
            
            # 5. Update stats
            self._update_stats(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Execution error: {e}")
            return self._create_rejected_result(decision, str(e))
    
    def _pre_execution_checks(self, decision: Decision) -> bool:
        """Pre-execution validation."""
        # Check order size
        if decision.size > self.config.max_order_size:
            logger.warning(f"Order size {decision.size} exceeds max")
            return False
        
        # Check liquidity (simplified)
        # Would check volume here
        
        return True
    
    def _create_order(self, decision: Decision) -> Order:
        """Create order from decision."""
        self._order_count += 1
        
        return Order(
            order_id=f"ORD_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self._order_count:04d}",
            timestamp=datetime.now(),
            symbol=decision.symbol,
            side=decision.side,
            size=decision.size,
            order_type="market",
            source=decision.sources[0] if decision.sources else "unknown",
            decision=decision
        )
    
    async def _send_with_retry(self, order: Order) -> Order:
        """Send order with retry logic."""
        for attempt in range(self.config.max_retries + 1):
            try:
                filled = await self.adapter.send_order(order)
                return filled
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                
                if attempt < self.config.max_retries:
                    await asyncio.sleep(
                        self.config.retry_delay_ms / 1000
                    )
                else:
                    raise
        
        return order
    
    def _create_result(self, 
                      decision: Decision, 
                      order: Order) -> ExecutionResult:
        """Create execution result."""
        
        # Calculate slippage
        requested = decision.entry_price
        filled = order.filled_price or requested
        
        if requested > 0:
            slippage = abs(filled - requested) / requested
        else:
            slippage = 0
        
        # Calculate commission
        commission = self._calculate_commission(order)
        
        # Status
        if order.status == "filled":
            status = "filled"
        elif order.status == "partial":
            status = "partial"
        else:
            status = "rejected"
        
        return ExecutionResult(
            decision=decision,
            status=status,
            filled_price=filled,
            filled_size=order.filled_size,
            slippage=slippage,
            commission=commission,
            timestamp=datetime.now(),
            order_id=order.order_id
        )
    
    def _create_rejected_result(self,
                                decision: Decision,
                                error: str) -> ExecutionResult:
        """Create rejected result."""
        return ExecutionResult(
            decision=decision,
            status="rejected",
            filled_price=0,
            filled_size=0,
            slippage=0,
            commission=0,
            timestamp=datetime.now(),
            order_id="",
            error=error
        )
    
    def _calculate_commission(self, order: Order) -> float:
        """Calculate commission."""
        return order.filled_size * self.config.commission_per_lot
    
    def _update_stats(self, result: ExecutionResult):
        """Update execution statistics."""
        self._stats['total_orders'] += 1
        
        if result.status == "filled":
            self._stats['filled_orders'] += 1
        else:
            self._stats['rejected_orders'] += 1
        
        # Update average slippage
        n = self._stats['total_orders']
        self._stats['avg_slippage_bps'] = (
            (self._stats['avg_slippage_bps'] * (n - 1) + 
             result.slippage * 10000) / n
        )
        
        self._stats['total_commission'] += result.commission
    
    # ========================================================================
    # POSITION MANAGEMENT
    # ========================================================================
    
    async def get_positions(self) -> List[Dict]:
        """Get current positions."""
        if self.adapter is None:
            return []
        return await self.adapter.get_positions()
    
    async def close_position(self, position_id: str):
        """Close position."""
        # Implementation depends on adapter
        pass
    
    async def close_all_positions(self):
        """Close all positions."""
        positions = await self.get_positions()
        for pos in positions:
            await self.close_position(pos.get('id', ''))
    
    # ========================================================================
    # SLIPPAGE MODELING
    # ========================================================================
    
    def model_slippage(self,
                      symbol: str,
                      size: float,
                      market_volatility: float) -> float:
        """
        Model execution slippage.
        
        Args:
            symbol: Symbol
            size: Order size
            market_volatility: Current volatility
            
        Returns:
            Slippage in basis points
        """
        if self.config.slippage_model == "fixed":
            return self.config.slippage_bps
        
        elif self.config.slippage_model == "adaptive":
            # Scale with size and volatility
            base = self.config.slippage_bps
            size_factor = np.log1p(size) * 0.1
            vol_factor = market_volatility * 100
            
            slippage = base * (1 + size_factor + vol_factor)
            return slippage
        
        else:
            return self.config.slippage_bps
    
    # ========================================================================
    # STATUS
    # ========================================================================
    
    def get_status(self) -> Dict[str, Any]:
        """Get execution service status."""
        return {
            'mode': self.config.mode,
            'adapter': self._adapter_type,
            'total_orders': self._stats['total_orders'],
            'filled_orders': self._stats['filled_orders'],
            'rejected_orders': self._stats['rejected_orders'],
            'avg_slippage_bps': self._stats['avg_slippage_bps'],
            'total_commission': self._stats['total_commission'],
            'open_orders': len(self._orders)
        }
