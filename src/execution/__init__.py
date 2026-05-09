"""
MT5 Live Execution Module
=========================
Complete live trading infrastructure for MetaTrader 5.
"""

# Try to import MT5 components (may not be available without MT5 installed)
try:
    from src.execution.mt5_connector import (
        MT5Connector,
        MT5Config,
        TickData,
        OrderResult,
        ExecutionRecord,
        PositionSnapshot,
        ReconciliationError
    )
    from src.execution.live_agent import LiveTradingAgent
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

from src.execution.trade_logger import (
    TradeLogger,
    TradeEntry,
    TradeExit,
    CompleteTrade
)

from src.execution.metrics import (
    MetricsCalculator,
    MetricsDashboard,
    ExecutionMetrics
)

from src.execution.risk_engine import (
    CorrelationRiskEngine,
    RiskThrottler,
    Position,
    RiskStatus,
    CurrencyExposure
)

from src.execution.scoring_engine import (
    ScoringEngine,
    ProductionScoringAgent,
    FeatureCalculator,
    SignalFeatures,
    ScoredSignal
)

from src.execution.risk_controller import (
    HardRiskController,
    RiskAction,
    KillSwitchReason,
    RiskState,
    RiskEvent
)

__version__ = "2.0.0"
__author__ = "Trading Systems"

__all__ = [
    'ScoringEngine',
    'ProductionScoringAgent',
    'FeatureCalculator',
    'SignalFeatures',
    'ScoredSignal',
    'HardRiskController',
    'RiskAction',
    'KillSwitchReason',
    'RiskState',
    'RiskEvent',
]
