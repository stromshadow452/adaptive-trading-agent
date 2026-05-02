"""
Correlation Risk Guard

Prevents excessive exposure to correlated assets.
Blocks trades that would create dangerous currency concentration.
"""

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

from .portfolio_state import PortfolioState, Position, PositionSide


# Currency groupings for correlation detection
CURRENCY_GROUPS = {
    'JPY': ['EURJPY', 'USDJPY', 'GBPJPY', 'AUDJPY', 'NZDJPY', 'CADJPY', 'CHFJPY'],
    'USD': ['EURUSD', 'GBPUSD', 'AUDUSD', 'NZDUSD', 'USDCAD', 'USDCHF', 'USDJPY'],
    'EUR': ['EURUSD', 'EURJPY', 'EURGBP', 'EURAUD', 'EURNZD', 'EURCHF', 'EURCAD'],
    'GBP': ['GBPUSD', 'GBPJPY', 'EURGBP', 'GBPAUD', 'GBPNZD', 'GBPCHF', 'GBPCAD'],
}


@dataclass
class CorrelationRisk:
    """Correlation risk assessment result."""
    is_risky: bool
    currency: str
    direction: str
    exposure_count: int
    details: str


class CorrelationGuard:
    """
    Guards against excessive correlated positions.
    
    Rules:
    1. Max 2 positions exposed to same currency direction
    2. Block new trades that would exceed limit
    3. Consider both long/short exposure per currency
    """
    
    def __init__(self, max_currency_exposure: int = 2):
        self.max_currency_exposure = max_currency_exposure
    
    def check_correlation(
        self,
        new_symbol: str,
        new_side: PositionSide,
        portfolio: PortfolioState,
    ) -> CorrelationRisk:
        """
        Check if new trade would create dangerous correlation.
        
        Args:
            new_symbol: Symbol for proposed trade
            new_side: LONG or SHORT
            portfolio: Current portfolio state
            
        Returns:
            CorrelationRisk assessment
        """
        # Extract currencies from new symbol
        base_ccy = new_symbol[:3]
        quote_ccy = new_symbol[3:]
        
        # Check each currency group
        for currency in [base_ccy, quote_ccy]:
            # Get current exposure to this currency
            current_exposure = self._count_currency_exposure(
                currency, 
                portfolio.open_positions
            )
            
            # Determine if new trade adds to same-direction exposure
            new_direction = self._get_currency_direction(
                new_symbol, new_side, currency
            )
            
            same_direction_count = current_exposure.get(new_direction, 0)
            
            if same_direction_count >= self.max_currency_exposure:
                return CorrelationRisk(
                    is_risky=True,
                    currency=currency,
                    direction=new_direction,
                    exposure_count=same_direction_count,
                    details=f"Already {same_direction_count} {new_direction} {currency} positions"
                )
        
        return CorrelationRisk(
            is_risky=False,
            currency="",
            direction="",
            exposure_count=0,
            details=""
        )
    
    def _count_currency_exposure(
        self, 
        currency: str, 
        positions: Dict[str, Position]
    ) -> Dict[str, int]:
        """Count long/short exposure to a currency."""
        exposure = {'LONG': 0, 'SHORT': 0}
        
        for symbol, pos in positions.items():
            if currency not in symbol:
                continue
            
            direction = self._get_currency_direction(
                symbol, pos.side, currency
            )
            exposure[direction] += 1
        
        return exposure
    
    def _get_currency_direction(
        self, 
        symbol: str, 
        side: PositionSide, 
        currency: str
    ) -> str:
        """
        Determine effective direction for a currency.
        
        Examples:
        - EURJPY LONG = EUR Long, JPY Short
        - USDJPY SHORT = USD Short, JPY Long
        """
        base_ccy = symbol[:3]
        quote_ccy = symbol[3:]
        
        if currency == base_ccy:
            return 'LONG' if side == PositionSide.LONG else 'SHORT'
        elif currency == quote_ccy:
            return 'SHORT' if side == PositionSide.LONG else 'LONG'
        else:
            return 'NONE'
    
    def get_portfolio_correlation_map(
        self, 
        portfolio: PortfolioState
    ) -> Dict[str, Dict[str, int]]:
        """
        Get full currency exposure map.
        
        Returns:
            Dict[currency, Dict[direction, count]]
        """
        result = {}
        
        all_currencies = set()
        for symbol in portfolio.open_positions:
            all_currencies.add(symbol[:3])
            all_currencies.add(symbol[3:])
        
        for currency in all_currencies:
            result[currency] = self._count_currency_exposure(
                currency, 
                portfolio.open_positions
            )
        
        return result
