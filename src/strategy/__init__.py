"""
WEAPON SYSTEM: Strategy Module
===============================

Agent-controlled strategy framework.

Components:
- base.py: Strategy interface, enums, data classes
- scalpel.py: Micro strategies (10-15% size)
- rifle.py: Standard strategies (100% size)  
- router.py: Central decision point

Usage:
    from src.strategy import StrategyRouter, MarketSnapshot, AgentContext
    
    router = StrategyRouter(enable_micro=True)
    strategy, signal = router.route(snapshot, context)
"""

from src.strategy.base import (
    Strategy,
    WeaponClass,
    Signal,
    MarketSnapshot,
    AgentContext,
    StrategySignal,
    StrategyStats,
    ShieldStrategy,
)

from src.strategy.scalpel import (
    ScalpelStrategy,
    RangeMicroMR,
    LiquiditySweepFade,
)

from src.strategy.rsi_range_reversion import (
    RSIRangeReversion,
)

from src.strategy.rifle import (
    RifleStrategy,
    MLPrimaryStrategy,
)

from src.strategy.router import (
    StrategyRouter,
    StrategyManager,
)

from src.strategy.logging import (
    WeaponDecisionLog,
    WeaponDecisionLogger,
)

__all__ = [
    # Base
    'Strategy',
    'WeaponClass',
    'Signal',
    'MarketSnapshot',
    'AgentContext',
    'StrategySignal',
    'StrategyStats',
    'ShieldStrategy',
    # Scalpel
    'ScalpelStrategy',
    'RangeMicroMR',
    'LiquiditySweepFade',
    'RSIRangeReversion',
    # Rifle
    'RifleStrategy',
    'MLPrimaryStrategy',
    # Router
    'StrategyRouter',
    'StrategyManager',
    # Logging
    'WeaponDecisionLog',
    'WeaponDecisionLogger',
]
