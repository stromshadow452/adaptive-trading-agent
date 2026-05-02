"""
Capital Protection Module

Executes protective actions based on danger level.
Implements the SAFE → CAUTION → DANGER → CRITICAL action hierarchy.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional
import logging

from .danger_detector import DangerLevel, DangerAssessment
from .portfolio_state import PortfolioState, Position


@dataclass
class ProtectionAction:
    """Record of a protection action taken."""
    timestamp: datetime
    action_type: str
    details: str
    danger_level: DangerLevel


class CapitalProtector:
    """
    Executes capital protection actions based on danger level.
    
    Action Matrix:
    - SAFE: Normal trading, full size
    - CAUTION: Trading allowed, 0.5x size
    - DANGER: Halt new trades, close losing positions
    - CRITICAL: Close all positions, halt 24h
    """
    
    def __init__(
        self,
        caution_size_mult: float = 0.5,
        halt_duration_hours: int = 8,
    ):
        self.caution_size_mult = caution_size_mult
        self.halt_duration_hours = halt_duration_hours
        
        # Action history
        self.actions: List[ProtectionAction] = []
        self.caution_count = 0
        self.danger_count = 0
        self.critical_count = 0
        self.positions_force_closed = 0
    
    def apply_protection(
        self,
        assessment: DangerAssessment,
        portfolio: PortfolioState,
        current_prices: Dict[str, float],
    ) -> List[ProtectionAction]:
        """
        Apply protection actions based on danger assessment.
        
        Returns list of actions taken.
        """
        actions = []
        current_time = assessment.timestamp
        
        # Update portfolio danger level
        portfolio.danger_level = assessment.level.value
        
        if assessment.level == DangerLevel.CRITICAL:
            actions.extend(self._action_critical(portfolio, current_prices, current_time))
        elif assessment.level == DangerLevel.DANGER:
            actions.extend(self._action_danger(portfolio, current_prices, current_time))
        elif assessment.level == DangerLevel.CAUTION:
            actions.extend(self._action_caution(portfolio, current_time))
        else:
            # SAFE - resume normal if was halted
            if portfolio.trading_halted and portfolio.halt_until:
                if current_time >= portfolio.halt_until:
                    portfolio.trading_halted = False
                    portfolio.halt_until = None
                    actions.append(ProtectionAction(
                        timestamp=current_time,
                        action_type="RESUME_TRADING",
                        details="Halt period ended, resuming normal operation",
                        danger_level=DangerLevel.SAFE,
                    ))
        
        self.actions.extend(actions)
        return actions
    
    def _action_critical(
        self,
        portfolio: PortfolioState,
        current_prices: Dict[str, float],
        current_time: datetime,
    ) -> List[ProtectionAction]:
        """CRITICAL: Close all positions and halt trading."""
        actions = []
        self.critical_count += 1
        
        # Close all positions
        if portfolio.open_positions:
            num_positions = len(portfolio.open_positions)
            portfolio.close_all_positions(current_prices, current_time, "CRITICAL_EXIT")
            self.positions_force_closed += num_positions
            
            actions.append(ProtectionAction(
                timestamp=current_time,
                action_type="CLOSE_ALL",
                details=f"Closed {num_positions} positions (CRITICAL)",
                danger_level=DangerLevel.CRITICAL,
            ))
        
        # Halt trading for extended period
        portfolio.trading_halted = True
        portfolio.halt_until = current_time + timedelta(hours=self.halt_duration_hours * 3)
        
        actions.append(ProtectionAction(
            timestamp=current_time,
            action_type="HALT_TRADING",
            details=f"Trading halted for {self.halt_duration_hours * 3}h (CRITICAL)",
            danger_level=DangerLevel.CRITICAL,
        ))
        
        return actions
    
    def _action_danger(
        self,
        portfolio: PortfolioState,
        current_prices: Dict[str, float],
        current_time: datetime,
    ) -> List[ProtectionAction]:
        """DANGER: Close losing positions and halt new trades."""
        actions = []
        self.danger_count += 1
        
        # Close losing positions only
        losing_positions = [
            symbol for symbol, pos in portfolio.open_positions.items()
            if pos.unrealized_pnl < 0
        ]
        
        for symbol in losing_positions:
            if symbol in current_prices:
                portfolio.close_position(
                    symbol, 
                    current_prices[symbol], 
                    current_time, 
                    "DANGER_EXIT"
                )
                self.positions_force_closed += 1
        
        if losing_positions:
            actions.append(ProtectionAction(
                timestamp=current_time,
                action_type="CLOSE_LOSING",
                details=f"Closed {len(losing_positions)} losing positions",
                danger_level=DangerLevel.DANGER,
            ))
        
        # Halt new trades
        if not portfolio.trading_halted:
            portfolio.trading_halted = True
            portfolio.halt_until = current_time + timedelta(hours=self.halt_duration_hours)
            
            actions.append(ProtectionAction(
                timestamp=current_time,
                action_type="HALT_TRADING",
                details=f"Trading halted for {self.halt_duration_hours}h",
                danger_level=DangerLevel.DANGER,
            ))
        
        return actions
    
    def _action_caution(
        self,
        portfolio: PortfolioState,
        current_time: datetime,
    ) -> List[ProtectionAction]:
        """CAUTION: Log warning, trading continues at reduced size."""
        self.caution_count += 1
        
        return [ProtectionAction(
            timestamp=current_time,
            action_type="CAUTION_MODE",
            details=f"Trading at {self.caution_size_mult*100:.0f}% size",
            danger_level=DangerLevel.CAUTION,
        )]
    
    def get_size_multiplier(self, danger_level: DangerLevel) -> float:
        """Get position size multiplier for current danger level."""
        if danger_level == DangerLevel.CRITICAL:
            return 0.0
        elif danger_level == DangerLevel.DANGER:
            return 0.0
        elif danger_level == DangerLevel.CAUTION:
            return self.caution_size_mult
        else:
            return 1.0
    
    def get_summary(self) -> dict:
        """Get protection activity summary."""
        return {
            'caution_events': self.caution_count,
            'danger_events': self.danger_count,
            'critical_events': self.critical_count,
            'positions_force_closed': self.positions_force_closed,
            'total_actions': len(self.actions),
        }
    
    def reset(self):
        """Reset for new simulation."""
        self.actions.clear()
        self.caution_count = 0
        self.danger_count = 0
        self.critical_count = 0
        self.positions_force_closed = 0
