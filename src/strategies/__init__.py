"""
SCOPUS v3.0 — Strategies Package

Dual-strategy system:
- Strategy A: Choppy Engine (FX Defense)
- Strategy B: Silver Mean Reversion (Offense)
"""

from src.strategies.choppy_engine import ChoppyEngine, ChoppySignal, ChoppyConfig
from src.strategies.silver_mean_reversion import (
    SilverMeanReversion, SilverConfig, SignalType, SilverTrader
)
from src.strategies.portfolio_orchestrator import (
    PortfolioOrchestrator, PortfolioConfig, PortfolioSignal, PortfolioAction
)

__all__ = [
    # Strategy A
    'ChoppyEngine',
    'ChoppySignal',
    'ChoppyConfig',
    
    # Strategy B
    'SilverMeanReversion',
    'SilverConfig',
    'SignalType',
    'SilverTrader',
    
    # Portfolio
    'PortfolioOrchestrator',
    'PortfolioConfig',
    'PortfolioSignal',
    'PortfolioAction',
]
