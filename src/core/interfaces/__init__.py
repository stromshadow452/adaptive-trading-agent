"""
Core Interfaces for Adaptive Trading OS

Defines abstract base classes for all system components.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
import pandas as pd
import numpy as np


# =============================================================================
# ENUMS
# =============================================================================

class SignalDirection(Enum):
    """Trading signal directions."""
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"
    PASS = "pass"


class MarketRegime(Enum):
    """Market regime classifications."""
    TRENDING = "trending"
    RANGING = "ranging"
    VOLATILE = "volatile"
    DANGER = "danger"
    UNKNOWN = "unknown"


class DecisionAction(Enum):
    """Final decision actions."""
    BUY = "buy"
    SELL = "sell"
    PASS = "pass"
    CLOSE = "close"
    SCALE_IN = "scale_in"
    SCALE_OUT = "scale_out"


class PipelineStage(Enum):
    """13 pipeline stages."""
    DATA_LOAD = 1
    FEATURES = 2
    ML_BRAIN = 3
    RL_BRAIN = 4
    REGIME = 5
    SENTIMENT = 6
    RISK = 7
    META_GATING = 8
    PORTFOLIO = 9
    RISK_GATES = 10
    THROTTLE_GATES = 11
    DECISION = 12
    LOG = 13


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class MarketData:
    """Standardized market data container."""
    symbol: str
    timestamp: datetime
    timeframe: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    # Optional fields
    bid: Optional[float] = None
    ask: Optional[float] = None
    spread: Optional[float] = None
    
    # Additional data
    features: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame."""
        return pd.DataFrame([{
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume
        }], index=[self.timestamp])


@dataclass
class AlphaSignal:
    """Standardized alpha signal from any pod."""
    # Required
    source: str                          # Alpha pod name
    timestamp: datetime
    symbol: str
    direction: SignalDirection
    
    # Confidence & quality
    confidence: float = 0.5              # 0.0 - 1.0
    expected_return: float = 0.0        # Annualized
    volatility: float = 0.0               # Expected volatility
    sharpe: float = 0.0                   # Expected Sharpe
    
    # Sizing
    recommended_size: float = 0.0         # Position size (0-1)
    max_position: float = 0.5             # Max allowed position
    
    # Risk
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    time_horizon: int = 5                 # Bars to hold
    
    # Metadata
    features: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate signal."""
        assert 0.0 <= self.confidence <= 1.0, "Confidence must be 0-1"
        assert self.volatility >= 0, "Volatility must be >= 0"


@dataclass
class RiskDecision:
    """Risk-adjusted decision."""
    signal: AlphaSignal
    position_size: float
    stop_loss: float
    take_profit: float
    risk_amount: float
    risk_percent: float
    rr_ratio: float
    approved: bool
    reason: str = ""


@dataclass
class PortfolioAllocation:
    """Portfolio-level allocation."""
    symbol: str
    target_weight: float
    current_weight: float
    delta: float
    risk_budget: float
    correlation_adjusted: bool = False


@dataclass
class Decision:
    """Final trading decision."""
    action: DecisionAction
    symbol: str
    side: str
    size: float
    entry_price: float
    stop_loss: float
    take_profit: float
    
    # Source tracking
    sources: List[str] = field(default_factory=list)
    signals: List[AlphaSignal] = field(default_factory=list)
    
    # Metadata
    regime: str = "unknown"
    confidence: float = 0.0
    timestamp: Optional[datetime] = None
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class ExecutionResult:
    """Execution result."""
    decision: Decision
    status: str  # "filled", "rejected", "partial"
    filled_price: float
    filled_size: float
    slippage: float
    commission: float
    timestamp: datetime
    order_id: str
    error: Optional[str] = None


# =============================================================================
# ABSTRACT BASE CLASSES
# =============================================================================

class AlphaPod(ABC):
    """
    Abstract base class for all alpha pods.
    
    Alpha pods are modular signal generators that plug into the unified pipeline.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._data_service: Optional[Any] = None
        self._risk_service: Optional[Any] = None
        self._is_active: bool = True
        self._performance: Dict[str, float] = {}
        
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique alpha pod name."""
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """Pod version."""
        pass
    
    @property
    @abstractmethod
    def universe(self) -> List[str]:
        """Symbols this pod covers."""
        pass
    
    @property
    @abstractmethod
    def timeframe(self) -> str:
        """Primary timeframe (e.g., 'D1', 'H1', 'M15')."""
        pass
    
    # -------------------------------------------------------------------------
    # Core Methods
    # -------------------------------------------------------------------------
    
    @abstractmethod
    def generate_signal(self, data: MarketData) -> Optional[AlphaSignal]:
        """
        Generate trading signal.
        
        Args:
            data: Current market data
            
        Returns:
            AlphaSignal or None if no signal
        """
        pass
    
    @abstractmethod
    def get_features(self, data: MarketData) -> Dict[str, float]:
        """
        Return features used by this pod.
        
        Args:
            data: Market data
            
        Returns:
            Dict of feature names to values
        """
        pass
    
    # -------------------------------------------------------------------------
    # Lifecycle Methods
    # -------------------------------------------------------------------------
    
    def initialize(self) -> bool:
        """
        Initialize pod.
        
        Returns:
            bool: Success
        """
        return True
    
    def shutdown(self):
        """Shutdown pod."""
        self._is_active = False
    
    def on_trade_executed(self, trade: ExecutionResult):
        """
        Callback when trade from this pod executes.
        
        Args:
            trade: Execution result
        """
        pass
    
    def on_trade_closed(self, pnl: float, metadata: Dict):
        """
        Callback when trade closes.
        
        Args:
            pnl: P&L
            metadata: Trade metadata
        """
        pass
    
    # -------------------------------------------------------------------------
    # Service Injection
    # -------------------------------------------------------------------------
    
    def set_data_service(self, service: Any):
        """Inject data service."""
        self._data_service = service
    
    def set_risk_service(self, service: Any):
        """Inject risk service."""
        self._risk_service = service
    
    # -------------------------------------------------------------------------
    # Status
    # -------------------------------------------------------------------------
    
    @property
    def is_active(self) -> bool:
        """Check if pod is active."""
        return self._is_active
    
    def deactivate(self, reason: str = ""):
        """Deactivate pod."""
        self._is_active = False
        
    def get_status(self) -> Dict[str, Any]:
        """Get pod status."""
        return {
            'name': self.name,
            'version': self.version,
            'active': self._is_active,
            'universe': self.universe,
            'timeframe': self.timeframe
        }


class PipelineStageInterface(ABC):
    """
    Abstract base class for pipeline stages.
    """
    
    @property
    @abstractmethod
    def stage_number(self) -> int:
        """Pipeline stage number (1-13)."""
        pass
    
    @property
    @abstractmethod
    def stage_name(self) -> str:
        """Stage name."""
        pass
    
    @abstractmethod
    def process(self, 
                input_data: Any,
                context: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """
        Process stage.
        
        Args:
            input_data: Input from previous stage
            context: Shared context
            
        Returns:
            (output_data, updated_context)
        """
        pass
    
    @abstractmethod
    def validate_input(self, input_data: Any) -> bool:
        """Validate stage input."""
        pass
    
    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """Get input/output schema."""
        pass


class SharedService(ABC):
    """Abstract base class for shared services."""
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize service."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Health check."""
        pass
    
    @abstractmethod
    async def shutdown(self):
        """Shutdown service."""
        pass


# =============================================================================
# SHARED SERVICE INTERFACES
# =============================================================================

class DataServiceInterface(SharedService):
    """Data service interface."""
    
    @abstractmethod
    def get_prices(self, 
                   symbol: str, 
                   timeframe: str,
                   start: Optional[datetime] = None,
                   end: Optional[datetime] = None) -> pd.DataFrame:
        """Get historical prices."""
        pass
    
    @abstractmethod
    def get_latest(self, symbol: str) -> MarketData:
        """Get latest price."""
        pass
    
    @abstractmethod
    def get_features(self, symbol: str) -> Dict[str, float]:
        """Get computed features."""
        pass


class RiskServiceInterface(SharedService):
    """Risk service interface."""
    
    @abstractmethod
    def check_limits(self, decision: Decision) -> Tuple[bool, str]:
        """Check if decision passes risk limits."""
        pass
    
    @abstractmethod
    def calculate_position_size(self,
                                 signal: AlphaSignal,
                                 portfolio_value: float) -> float:
        """Calculate position size."""
        pass
    
    @abstractmethod
    def calculate_stops(self,
                       entry: float,
                       signal: AlphaSignal) -> Tuple[float, float]:
        """Calculate stop loss and take profit."""
        pass


class ExecutionServiceInterface(SharedService):
    """Execution service interface."""
    
    @abstractmethod
    async def execute(self, decision: Decision) -> ExecutionResult:
        """Execute decision."""
        pass
    
    @abstractmethod
    async def get_positions(self) -> List[Dict]:
        """Get current positions."""
        pass
    
    @abstractmethod
    async def close_position(self, position_id: str):
        """Close position."""
        pass


class MonitoringServiceInterface(SharedService):
    """Monitoring service interface."""
    
    @abstractmethod
    def log_decision(self, 
                     decision: Decision,
                     signals: List[AlphaSignal],
                     result: ExecutionResult):
        """Log decision."""
        pass
    
    @abstractmethod
    def alert_critical(self, message: str):
        """Send critical alert."""
        pass
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, float]:
        """Get current metrics."""
        pass


# =============================================================================
# EXCEPTIONS
# =============================================================================

class AlphaPodError(Exception):
    """Alpha pod error."""
    pass


class PipelineError(Exception):
    """Pipeline error."""
    pass


class OrchestratorError(Exception):
    """Orchestrator error."""
    pass


class RiskLimitError(Exception):
    """Risk limit exceeded."""
    pass


class ExecutionError(Exception):
    """Execution error."""
    pass


# =============================================================================
# UTILITIES
# =============================================================================

def validate_signal(signal: AlphaSignal) -> bool:
    """Validate alpha signal."""
    if not signal.source:
        return False
    if not signal.symbol:
        return False
    if signal.confidence < 0 or signal.confidence > 1:
        return False
    return True


def combine_signals(signals: List[AlphaSignal],
                   method: str = "weighted") -> AlphaSignal:
    """
    Combine multiple signals.
    
    Args:
        signals: List of signals
        method: "weighted", "majority", "best"
        
    Returns:
        Combined signal
    """
    if not signals:
        return None
    
    if len(signals) == 1:
        return signals[0]
    
    if method == "weighted":
        # Weight by confidence
        total_conf = sum(s.confidence for s in signals)
        if total_conf == 0:
            return signals[0]
        
        weights = [s.confidence / total_conf for s in signals]
        
        # Weighted average of returns
        exp_return = sum(s.expected_return * w 
                        for s, w in zip(signals, weights))
        
        # Max volatility
        volatility = max(s.volatility for s in signals)
        
        # Majority direction
        longs = sum(1 for s in signals if s.direction == SignalDirection.LONG)
        shorts = sum(1 for s in signals if s.direction == SignalDirection.SHORT)
        
        direction = SignalDirection.LONG if longs > shorts else SignalDirection.SHORT
        
        return AlphaSignal(
            source="ensemble",
            timestamp=datetime.now(),
            symbol=signals[0].symbol,
            direction=direction,
            confidence=total_conf / len(signals),
            expected_return=exp_return,
            volatility=volatility,
            metadata={'sources': [s.source for s in signals]}
        )
    
    elif method == "best":
        # Return best Sharpe
        return max(signals, key=lambda s: s.sharpe)
    
    else:
        return signals[0]
