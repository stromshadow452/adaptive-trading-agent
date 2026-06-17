"""
Shared Risk Service

Unified risk management for Adaptive Trading OS.

Centralized:
- Volatility targeting
- Position sizing
- Portfolio heat tracking
- Drawdown throttling
- Kill switch
- Correlation controls
- SL/TP framework
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

from ..interfaces import (
    AlphaSignal, Decision, RiskDecision, MarketData,
    RiskServiceInterface, SharedService,
    RiskLimitError
)

logger = logging.getLogger(__name__)


@dataclass
class RiskLimits:
    """Risk limits configuration."""
    # Position limits
    max_position_size: float = 0.10  # 10% per trade
    max_pair_exposure: float = 0.40  # 40% per pair
    max_total_exposure: float = 2.0  # 200% gross
    max_net_exposure: float = 1.0  # 100% net
    
    # Volatility targeting
    target_volatility: float = 0.10  # 10% annual
    vol_lookback: int = 20  # days
    vol_scaling: bool = True
    
    # Drawdown
    max_drawdown: float = 0.20  # 20%
    daily_loss_limit: float = 0.05  # 5%
    weekly_loss_limit: float = 0.10  # 10%
    
    # Correlation
    max_correlation: float = 0.70  # 70%
    correlation_lookback: int = 60  # days
    
    # Emergency
    circuit_breaker_drawdown: float = 0.15  # 15%
    kill_switch_drawdown: float = 0.25  # 25%
    
    # Risk-reward
    min_rr_ratio: float = 1.5  # minimum R:R
    max_rr_ratio: float = 5.0  # maximum R:R


@dataclass
class RiskState:
    """Current risk state."""
    timestamp: datetime
    
    # Portfolio state
    portfolio_value: float = 1.0
    peak_portfolio_value: float = 1.0
    current_drawdown: float = 0.0
    
    # Daily tracking
    daily_pnl: float = 0.0
    daily_trades: int = 0
    daily_loss: float = 0.0
    
    # Exposure
    gross_exposure: float = 0.0
    net_exposure: float = 0.0
    pair_exposures: Dict[str, float] = field(default_factory=dict)
    
    # Volatility
    current_volatility: float = 0.10
    vol_scalar: float = 1.0
    
    # Status
    circuit_breaker_tripped: bool = False
    throttle_level: float = 1.0  # 1.0 = full speed
    
    # Heat
    portfolio_heat: float = 0.0  # 0-1 heat level
    

@dataclass
class PositionRisk:
    """Risk metrics for a position."""
    symbol: str
    direction: str
    size: float
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_amount: float
    risk_percent: float
    rr_ratio: float
    position_heat: float


class SharedRiskService(RiskServiceInterface):
    """
    Shared risk service for all alpha pods.
    
    Centralized risk management including:
    - Position sizing with volatility targeting
    - Portfolio heat tracking
    - Drawdown throttling
    - Correlation guards
    - Kill switch management
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize risk service.
        
        Args:
            config: Configuration dict
        """
        self.config = config
        self.limits = RiskLimits(**config.get('limits', {}))
        
        # State
        self.state = RiskState(timestamp=datetime.now())
        self.position_history: List[PositionRisk] = []
        self.drawdown_history: List[float] = []
        
        # Daily tracking
        self._current_day: Optional[datetime] = None
        self._day_start_value: float = 1.0
        
        # Correlation matrix
        self._correlation_matrix: Optional[pd.DataFrame] = None
        self._correlation_last_update: Optional[datetime] = None
        
        logger.info("SharedRiskService initialized")
        logger.info(f"  Max position: {self.limits.max_position_size:.1%}")
        logger.info(f"  Target vol: {self.limits.target_volatility:.1%}")
        logger.info(f"  Max DD: {self.limits.max_drawdown:.1%}")
    
    async def initialize(self) -> bool:
        """Initialize service."""
        try:
            logger.info("Initializing risk service...")
            
            # Load historical data for correlation
            self._update_correlation_matrix()
            
            logger.info("Risk service initialized")
            return True
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False
    
    async def health_check(self) -> bool:
        """Health check."""
        return True
    
    async def shutdown(self):
        """Shutdown service."""
        logger.info("Risk service shutdown")
    
    # ========================================================================
    # CORE RISK METHODS
    # ========================================================================
    
    def check_limits(self, decision: Decision) -> Tuple[bool, str]:
        """
        Check if decision passes all risk limits.
        
        Args:
            decision: Trading decision
            
        Returns:
            (allowed, reason)
        """
        # 1. Check kill switch
        if self._is_kill_switch_triggered():
            return False, "KILL_SWITCH"
        
        # 2. Check circuit breaker
        if self.state.circuit_breaker_tripped:
            return False, "CIRCUIT_BREAKER"
        
        # 3. Check daily loss
        if self.state.daily_loss >= self.limits.daily_loss_limit:
            return False, "DAILY_LOSS_LIMIT"
        
        # 4. Check position size
        if decision.size > self.limits.max_position_size:
            return False, "POSITION_SIZE_LIMIT"
        
        # 5. Check pair exposure
        current_exposure = self.state.pair_exposures.get(decision.symbol, 0)
        new_exposure = current_exposure + decision.size
        if new_exposure > self.limits.max_pair_exposure:
            return False, "PAIR_EXPOSURE_LIMIT"
        
        # 6. Check total exposure
        new_gross = self.state.gross_exposure + decision.size
        if new_gross > self.limits.max_total_exposure:
            return False, "TOTAL_EXPOSURE_LIMIT"
        
        # 7. Check correlation
        if self._exceeds_correlation_limit(decision):
            return False, "CORRELATION_LIMIT"
        
        return True, "OK"
    
    def calculate_position_size(self,
                              signal: AlphaSignal,
                              portfolio_value: float) -> float:
        """
        Calculate position size with volatility targeting.
        
        Formula:
        position_size = (target_risk / current_vol) * confidence
        
        Args:
            signal: Alpha signal
            portfolio_value: Current portfolio value
            
        Returns:
            Position size (0-1)
        """
        # Base size from signal
        base_size = signal.recommended_size
        
        # Volatility scaling
        if self.limits.vol_scaling:
            vol_scalar = self._calculate_vol_scalar()
            size = base_size * vol_scalar * signal.confidence
        else:
            size = base_size * signal.confidence
        
        # Apply limits
        size = min(size, self.limits.max_position_size)
        size = min(size, signal.max_position)
        
        # Apply throttling
        size *= self.state.throttle_level
        
        return size
    
    def calculate_stops(self,
                       entry: float,
                       signal: AlphaSignal) -> Tuple[float, float]:
        """
        Calculate stop loss and take profit.
        
        Uses ATR-based or fixed percentage based on signal.
        
        Args:
            entry: Entry price
            signal: Alpha signal
            
        Returns:
            (stop_loss, take_profit)
        """
        # Use signal stops if provided
        if signal.stop_loss and signal.take_profit:
            return signal.stop_loss, signal.take_profit
        
        # Calculate based on volatility
        vol = max(signal.volatility, 0.05)  # Min 5%
        
        if signal.direction == SignalDirection.LONG:
            # Long: SL below, TP above
            stop = entry * (1 - vol * 1.5)  # 1.5x vol for SL
            target = entry * (1 + vol * 3.0)  # 3x vol for TP (2:1 RR)
        else:
            # Short: SL above, TP below
            stop = entry * (1 + vol * 1.5)
            target = entry * (1 - vol * 3.0)
        
        return stop, target
    
    # ========================================================================
    # POSITION RISK
    # ========================================================================
    
    def calculate_position_risk(self,
                               symbol: str,
                               direction: str,
                               size: float,
                               entry: float,
                               stop: float,
                               target: float) -> PositionRisk:
        """
        Calculate risk metrics for a position.
        
        Args:
            symbol: Symbol
            direction: 'long' or 'short'
            size: Position size
            entry: Entry price
            stop: Stop loss
            target: Take profit
            
        Returns:
            PositionRisk
        """
        # Risk amount
        if direction == 'long':
            risk_per_unit = entry - stop
        else:
            risk_per_unit = stop - entry
        
        risk_amount = abs(risk_per_unit) * size
        risk_percent = risk_amount / self.state.portfolio_value
        
        # R:R ratio
        if direction == 'long':
            reward = target - entry
        else:
            reward = entry - target
        
        rr_ratio = abs(reward / risk_per_unit) if risk_per_unit != 0 else 0
        
        # Position heat
        position_heat = self._calculate_position_heat(symbol, size)
        
        return PositionRisk(
            symbol=symbol,
            direction=direction,
            size=size,
            entry_price=entry,
            stop_loss=stop,
            take_profit=target,
            risk_amount=risk_amount,
            risk_percent=risk_percent,
            rr_ratio=rr_ratio,
            position_heat=position_heat
        )
    
    def _calculate_position_heat(self, symbol: str, size: float) -> float:
        """
        Calculate position heat level.
        
        Heat = position size / max_position_size
        
        Args:
            symbol: Symbol
            size: Position size
            
        Returns:
            Heat level (0-1)
        """
        return size / self.limits.max_position_size
    
    # ========================================================================
    # VOLATILITY TARGETING
    # ========================================================================
    
    def _calculate_vol_scalar(self) -> float:
        """
        Calculate volatility scaling factor.
        
        scalar = target_vol / current_vol
        
        Returns:
            Scaling factor
        """
        current_vol = self.state.current_volatility
        if current_vol == 0:
            return 1.0
        
        scalar = self.limits.target_volatility / current_vol
        
        # Limit scalar to reasonable range
        scalar = np.clip(scalar, 0.5, 2.0)
        
        return scalar
    
    def update_volatility(self, returns: pd.Series):
        """
        Update volatility estimate.
        
        Args:
            returns: Series of returns
        """
        if len(returns) < self.limits.vol_lookback:
            return
        
        # Annualized volatility
        vol = returns.rolling(self.limits.vol_lookback).std().iloc[-1]
        self.state.current_volatility = vol * np.sqrt(252)  # Annualize
        
        # Update scalar
        self.state.vol_scalar = self._calculate_vol_scalar()
    
    # ========================================================================
    # DRAWDOWN MANAGEMENT
    # ========================================================================
    
    def update_portfolio_value(self, value: float):
        """
        Update portfolio value and check drawdown.
        
        Args:
            value: Current portfolio value
        """
        self.state.portfolio_value = value
        
        # Update peak
        if value > self.state.peak_portfolio_value:
            self.state.peak_portfolio_value = value
        
        # Calculate drawdown
        self.state.current_drawdown = (
            self.state.peak_portfolio_value - value
        ) / self.state.peak_portfolio_value
        
        self.drawdown_history.append(self.state.current_drawdown)
        
        # Check circuit breaker
        if self.state.current_drawdown > self.limits.circuit_breaker_drawdown:
            if not self.state.circuit_breaker_tripped:
                self._trip_circuit_breaker()
        
        # Check kill switch
        if self.state.current_drawdown > self.limits.kill_switch_drawdown:
            self._trigger_kill_switch()
        
        # Update throttle
        self._update_throttle()
    
    def _trip_circuit_breaker(self):
        """Trip circuit breaker."""
        self.state.circuit_breaker_tripped = True
        self.state.throttle_level = 0.5  # Reduce to 50%
        logger.critical(f"CIRCUIT BREAKER TRIPPED: DD {self.state.current_drawdown:.1%}")
    
    def _reset_circuit_breaker(self):
        """Reset circuit breaker."""
        self.state.circuit_breaker_tripped = False
        self.state.throttle_level = 1.0
        logger.info("Circuit breaker reset")
    
    def _is_kill_switch_triggered(self) -> bool:
        """Check if kill switch is triggered."""
        return self.state.current_drawdown > self.limits.kill_switch_drawdown
    
    def _trigger_kill_switch(self):
        """Trigger kill switch."""
        logger.critical(f"KILL SWITCH: DD {self.state.current_drawdown:.1%}")
        # This is handled by orchestrator
    
    def _update_throttle(self):
        """Update throttle level based on drawdown."""
        dd = self.state.current_drawdown
        
        if dd < 0.05:
            self.state.throttle_level = 1.0
        elif dd < 0.10:
            self.state.throttle_level = 0.8
        elif dd < 0.15:
            self.state.throttle_level = 0.5
        else:
            self.state.throttle_level = 0.25
    
    # ========================================================================
    # CORRELATION GUARD
    # ========================================================================
    
    def _update_correlation_matrix(self):
        """Update correlation matrix."""
        # This would load from data service
        # For now, use identity
        pass
    
    def _exceeds_correlation_limit(self, decision: Decision) -> bool:
        """
        Check if decision would exceed correlation limit.
        
        Args:
            decision: Trading decision
            
        Returns:
            bool: Would exceed limit
        """
        if self._correlation_matrix is None:
            return False
        
        symbol = decision.symbol
        
        # Check correlation with existing positions
        for pos_symbol in self.state.pair_exposures.keys():
            if symbol in self._correlation_matrix.columns and \
               pos_symbol in self._correlation_matrix.columns:
                corr = self._correlation_matrix.loc[symbol, pos_symbol]
                if abs(corr) > self.limits.max_correlation:
                    return True
        
        return False
    
    # ========================================================================
    # PORTFOLIO HEAT
    # ========================================================================
    
    def update_exposure(self,
                       symbol: str,
                       size: float,
                       is_entry: bool = True):
        """
        Update exposure tracking.
        
        Args:
            symbol: Symbol
            size: Position size (positive for long, negative for short)
            is_entry: True for entry, False for exit
        """
        current = self.state.pair_exposures.get(symbol, 0)
        
        if is_entry:
            new_size = current + size
        else:
            new_size = current - size
        
        self.state.pair_exposures[symbol] = new_size
        
        # Recalculate totals
        self.state.gross_exposure = sum(
            abs(s) for s in self.state.pair_exposures.values()
        )
        self.state.net_exposure = sum(
            self.state.pair_exposures.values()
        )
        
        # Update heat
        self._update_portfolio_heat()
    
    def _update_portfolio_heat(self):
        """Update portfolio heat level."""
        # Heat = gross exposure / max exposure
        heat = self.state.gross_exposure / self.limits.max_total_exposure
        self.state.portfolio_heat = min(heat, 1.0)
    
    # ========================================================================
    # DAILY TRACKING
    # ========================================================================
    
    def new_day(self, portfolio_value: float):
        """
        Reset daily tracking.
        
        Args:
            portfolio_value: Starting portfolio value
        """
        self._current_day = datetime.now().date()
        self._day_start_value = portfolio_value
        self.state.daily_pnl = 0
        self.state.daily_trades = 0
        self.state.daily_loss = 0
        
        logger.info(f"New day: {self._current_day}")
    
    def record_trade_pnl(self, pnl: float):
        """
        Record trade P&L.
        
        Args:
            pnl: Trade P&L
        """
        self.state.daily_pnl += pnl
        self.state.daily_trades += 1
        
        if pnl < 0:
            self.state.daily_loss += abs(pnl)
    
    # ========================================================================
    # STATUS
    # ========================================================================
    
    def get_status(self) -> Dict[str, Any]:
        """Get risk service status."""
        return {
            'portfolio_value': self.state.portfolio_value,
            'peak_value': self.state.peak_portfolio_value,
            'drawdown': self.state.current_drawdown,
            'gross_exposure': self.state.gross_exposure,
            'net_exposure': self.state.net_exposure,
            'portfolio_heat': self.state.portfolio_heat,
            'throttle': self.state.throttle_level,
            'volatility': self.state.current_volatility,
            'circuit_breaker': self.state.circuit_breaker_tripped,
            'kill_switch': self._is_kill_switch_triggered(),
            'daily_pnl': self.state.daily_pnl,
            'daily_loss': self.state.daily_loss
        }
    
    def get_risk_report(self) -> str:
        """Get formatted risk report."""
        status = self.get_status()
        
        report = f"""
Risk Report ({self.state.timestamp})
{'='*50}
Portfolio Value: ${status['portfolio_value']:,.2f}
Peak Value: ${status['peak_value']:,.2f}
Drawdown: {status['drawdown']:.1%}

Exposure:
  Gross: {status['gross_exposure']:.1%}
  Net: {status['net_exposure']:.1%}
  Heat: {status['portfolio_heat']:.1%}

Controls:
  Throttle: {status['throttle']:.0%}
  Volatility: {status['volatility']:.1%}
  Circuit Breaker: {'TRIPPED' if status['circuit_breaker'] else 'OK'}

Daily:
  P&L: ${status['daily_pnl']:,.2f}
  Loss: ${status['daily_loss']:,.2f}
{'='*50}
"""
        return report
