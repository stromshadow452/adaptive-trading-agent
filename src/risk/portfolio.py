"""
Portfolio Risk Brain - Stage 7 Extension

Multi-symbol correlation-aware position sizing and portfolio-level risk management.
Uses CSV data to calculate rolling correlations and adjust position sizes.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class PortfolioBrain:
    """
    Portfolio-level risk management with correlation awareness.
    
    Tracks simulated positions and adjusts sizing based on:
    - Symbol-level risk (volatility)
    - Portfolio correlation (reduce size for correlated positions)
    - Total portfolio exposure
    """
    
    def __init__(self,
                 max_portfolio_risk: float = 0.02,
                 correlation_lookback: int = 60,
                 max_correlated_exposure: float = 0.03):
        """
        Initialize Portfolio Brain.
        
        Args:
            max_portfolio_risk: Max portfolio risk as fraction of capital
            correlation_lookback: Rolling window for correlation calculation
            max_correlated_exposure: Max exposure to correlated positions
        """
        self.max_portfolio_risk = max_portfolio_risk
        self.correlation_lookback = correlation_lookback
        self.max_correlated_exposure = max_correlated_exposure
        
        # Simulated position tracking
        self.open_positions: Dict[str, Dict] = {}
        
        # Correlation matrix cache
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.last_corr_update = None
    
    def update_correlation_matrix(self, price_data: Dict[str, pd.DataFrame]):
        """
        Calculate rolling correlation matrix from CSV price data.
        
        Args:
            price_data: {symbol: DataFrame with 'close' and 'timestamp'}
        """
        if not price_data or len(price_data) < 2:
            logger.warning("[PORTFOLIO] Not enough symbols for correlation matrix")
            self.correlation_matrix = None
            return
        
        # Extract returns for each symbol
        returns_dict = {}
        for symbol, df in price_data.items():
            if 'close' in df.columns and len(df) > 1:
                returns = df['close'].pct_change().fillna(0)
                returns_dict[symbol] = returns.tail(self.correlation_lookback)
        
        if len(returns_dict) < 2:
            self.correlation_matrix = None
            return
        
        # Create returns DataFrame
        returns_df = pd.DataFrame(returns_dict)
        
        # Calculate correlation matrix
        self.correlation_matrix = returns_df.corr()
        self.last_corr_update = pd.Timestamp.now()
        
        logger.info(f"[PORTFOLIO] Updated correlation matrix for {len(returns_dict)} symbols")
    
    def get_correlation(self, symbol1: str, symbol2: str) -> float:
        """
        Get correlation between two symbols.
        
        Args:
            symbol1: First symbol
            symbol2: Second symbol
            
        Returns:
            Correlation coefficient [-1, 1], or 0 if not available
        """
        if self.correlation_matrix is None:
            return 0.0
        
        try:
            return self.correlation_matrix.loc[symbol1, symbol2]
        except (KeyError, IndexError):
            return 0.0
    
    def calculate_portfolio_risk(self) -> float:
        """
        Calculate current portfolio risk.
        
        Returns:
            Portfolio risk as fraction of capital
        """
        if not self.open_positions:
            return 0.0
        
        # Simple sum of position risks
        total_risk = sum(pos.get('risk', 0.0) for pos in self.open_positions.values())
        
        return total_risk
    
    def adjust_size(self,
                   symbol: str,
                   base_size: float,
                   open_positions: Optional[Dict[str, Dict]] = None,
                   csv_data: Optional[pd.DataFrame] = None) -> float:
        """
        Adjust position size based on portfolio risk and correlations.
        
        Args:
            symbol: Trading symbol
            base_size: Base position size (before adjustment)
            open_positions: Current open positions (for correlation check)
            csv_data: Price data for volatility calculation
            
        Returns:
            Adjusted position size
        """
        if base_size <= 0:
            return base_size
        
        # Use provided positions or internal tracking
        positions = open_positions if open_positions is not None else self.open_positions
        
        # Start with base size
        adjusted_size = base_size
        
        # 1. Portfolio risk check
        current_portfolio_risk = self.calculate_portfolio_risk()
        if current_portfolio_risk >= self.max_portfolio_risk:
            logger.warning(f"[PORTFOLIO] Max portfolio risk reached ({current_portfolio_risk:.3f}), "
                          f"reducing size by 50%")
            adjusted_size *= 0.5
        
        # 2. Correlation-based adjustment
        if positions and self.correlation_matrix is not None:
            max_corr = 0.0
            correlated_exposure = 0.0
            
            for pos_symbol, pos_info in positions.items():
                if pos_symbol == symbol:
                    continue
                
                corr = self.get_correlation(symbol, pos_symbol)
                if abs(corr) > 0.7:  # High correlation threshold
                    max_corr = max(max_corr, abs(corr))
                    correlated_exposure += pos_info.get('size', 0.0)
            
            if max_corr > 0.7:
                # Reduce size for highly correlated positions
                corr_factor = 1.0 - (max_corr - 0.7) / 0.3  # Scale from 1.0 to 0.0
                adjusted_size *= corr_factor
                logger.info(f"[PORTFOLIO] High correlation detected ({max_corr:.2f}), "
                           f"reducing size by {(1-corr_factor)*100:.0f}%")
            
            if correlated_exposure > self.max_correlated_exposure:
                logger.warning(f"[PORTFOLIO] Max correlated exposure reached, blocking trade")
                return 0.0
        
        # 3. Volatility-based adjustment (if CSV data provided)
        if csv_data is not None and 'close' in csv_data.columns:
            returns = csv_data['close'].pct_change().fillna(0)
            volatility = returns.std()
            
            # Reduce size for high volatility symbols
            if volatility > 0.02:  # 2% daily volatility threshold
                vol_factor = 0.02 / volatility
                adjusted_size *= vol_factor
                logger.info(f"[PORTFOLIO] High volatility ({volatility:.4f}), "
                           f"reducing size by {(1-vol_factor)*100:.0f}%")
        
        # Ensure size is positive
        adjusted_size = max(0.0, adjusted_size)
        
        logger.info(f"[PORTFOLIO] {symbol}: base_size={base_size:.4f}, "
                   f"adjusted_size={adjusted_size:.4f}, "
                   f"reduction={(1-adjusted_size/base_size)*100:.1f}%")
        
        return adjusted_size
    
    def add_position(self, symbol: str, size: float, price: float, side: str):
        """
        Track a new position (for simulation).
        
        Args:
            symbol: Trading symbol
            size: Position size
            price: Entry price
            side: 'buy' or 'sell'
        """
        self.open_positions[symbol] = {
            'size': size,
            'price': price,
            'side': side,
            'risk': abs(size) * 0.01  # Assume 1% risk per position
        }
        logger.info(f"[PORTFOLIO] Added position: {symbol} {side} {size:.4f} @ {price:.5f}")
    
    def remove_position(self, symbol: str):
        """
        Remove a position (for simulation).
        
        Args:
            symbol: Trading symbol
        """
        if symbol in self.open_positions:
            del self.open_positions[symbol]
            logger.info(f"[PORTFOLIO] Removed position: {symbol}")
    
    def get_portfolio_summary(self) -> Dict:
        """
        Get portfolio summary statistics.
        
        Returns:
            {
                'num_positions': int,
                'total_risk': float,
                'symbols': List[str],
                'avg_correlation': float
            }
        """
        num_positions = len(self.open_positions)
        total_risk = self.calculate_portfolio_risk()
        symbols = list(self.open_positions.keys())
        
        # Calculate average correlation between open positions
        avg_corr = 0.0
        if num_positions > 1 and self.correlation_matrix is not None:
            corr_sum = 0.0
            corr_count = 0
            for i, sym1 in enumerate(symbols):
                for sym2 in symbols[i+1:]:
                    corr_sum += abs(self.get_correlation(sym1, sym2))
                    corr_count += 1
            avg_corr = corr_sum / corr_count if corr_count > 0 else 0.0
        
        return {
            'num_positions': num_positions,
            'total_risk': total_risk,
            'symbols': symbols,
            'avg_correlation': avg_corr
        }
    
    def reset(self):
        """Reset portfolio state (for new backtest session)."""
        self.open_positions = {}
        self.correlation_matrix = None


# Factory function
def create_portfolio_brain(**kwargs) -> PortfolioBrain:
    """Create and return PortfolioBrain instance."""
    return PortfolioBrain(**kwargs)
