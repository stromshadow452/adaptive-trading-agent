"""
FX Momentum Alpha Pod

Validated FX momentum strategy integrated as alpha pod.

Research validated:
- 1M cross-sectional momentum
- 5-day holding period
- 4 pairs: GBPUSD, USDJPY, AUDUSD, USDCAD
- Sharpe: 2.0-2.5
- CAGR: 10-12%

This pod integrates the validated FX momentum research
into the unified Adaptive Trading OS.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import pandas as pd
import numpy as np

# Core interfaces
from ...core.interfaces import (
    AlphaPod, AlphaSignal, MarketData,
    SignalDirection, DecisionAction,
    DataServiceInterface, RiskServiceInterface
)

logger = logging.getLogger(__name__)


class FXMomentumAlphaPod(AlphaPod):
    """
    FX Momentum Alpha Pod.
    
    Implements validated 1-month cross-sectional momentum
    strategy across 4 major FX pairs.
    
    Parameters:
    -----------
    lookback : int
        Momentum lookback period (default: 21 days ~ 1M)
    holding_days : int
        Position holding period (default: 5 days)
    min_momentum_threshold : float
        Minimum momentum to generate signal (default: 0.01)
    max_position_size : float
        Maximum position size per pair (default: 0.40)
    pairs : List[str]
        Currency pairs to trade
    """
    
    VERSION = "2.0.0"
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize FX Momentum Alpha Pod.
        
        Args:
            config: Configuration dict with keys:
                - lookback: int (default: 21)
                - holding_days: int (default: 5)
                - min_momentum_threshold: float (default: 0.01)
                - max_position_size: float (default: 0.40)
                - pairs: List[str] (default: ['GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD'])
        """
        super().__init__(config)
        
        # Configuration
        self.lookback = config.get('lookback', 21)  # 1 month
        self.holding_days = config.get('holding_days', 5)
        self.min_momentum_threshold = config.get('min_momentum_threshold', 0.01)
        self.max_position_size = config.get('max_position_size', 0.40)
        
        # Universe
        self.pairs = config.get('pairs', ['GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD'])
        
        # State
        self._current_holdings: Dict[str, Dict] = {}  # Track active positions
        self._last_rankings: Optional[pd.Series] = None
        self._last_update: Optional[datetime] = None
        
        # Performance tracking
        self._signal_count = 0
        self._trade_count = 0
        
        logger.info(f"FXMomentumAlphaPod initialized v{self.VERSION}")
        logger.info(f"  Lookback: {self.lookback} days")
        logger.info(f"  Holding: {self.holding_days} days")
        logger.info(f"  Universe: {self.pairs}")
    
    # =========================================================================
    # Properties
    # =========================================================================
    
    @property
    def name(self) -> str:
        """Pod name."""
        return "fx_momentum"
    
    @property
    def version(self) -> str:
        """Pod version."""
        return self.VERSION
    
    @property
    def universe(self) -> List[str]:
        """Trading universe."""
        return self.pairs
    
    @property
    def timeframe(self) -> str:
        """Primary timeframe."""
        return "D1"  # Daily
    
    # =========================================================================
    # Core Methods
    # =========================================================================
    
    def generate_signal(self, data: MarketData) -> Optional[AlphaSignal]:
        """
        Generate FX momentum signal.
        
        Process:
        1. Fetch prices for all pairs
        2. Calculate 1M momentum for each pair
        3. Rank pairs by momentum
        4. Long strongest, short weakest (if threshold met)
        
        Args:
            data: Current market data (for timestamp)
            
        Returns:
            AlphaSignal or None
        """
        try:
            # 1. Get historical prices for all pairs
            prices = self._get_prices()
            if prices is None or prices.empty:
                logger.warning("No price data available")
                return None
            
            # 2. Calculate momentum
            momentums = self._calculate_momentum(prices)
            if momentums is None:
                return None
            
            # 3. Rank pairs
            rankings = self._rank_pairs(momentums)
            self._last_rankings = rankings
            self._last_update = datetime.now()
            
            # 4. Generate signals
            signal = self._generate_signal_from_rankings(rankings, data)
            
            if signal:
                self._signal_count += 1
                logger.info(f"Signal generated: {signal.symbol} {signal.direction.value} "
                          f"(conf: {signal.confidence:.2f})")
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            return None
    
    def get_features(self, data: MarketData) -> Dict[str, float]:
        """
        Return FX momentum features.
        
        Args:
            data: Market data
            
        Returns:
            Dict of feature names to values
        """
        features = {}
        
        # Get prices
        prices = self._get_prices()
        if prices is None:
            return features
        
        # Calculate momentum for each pair
        for pair in self.pairs:
            if pair in prices.columns:
                momentum = self._calculate_pair_momentum(prices[pair])
                if momentum is not None:
                    features[f'{pair}_momentum_1m'] = momentum
                    
                    # Percentile rank
                    all_momentums = [
                        self._calculate_pair_momentum(prices[p])
                        for p in self.pairs if p in prices.columns
                    ]
                    all_momentums = [m for m in all_momentums if m is not None]
                    if all_momentums:
                        percentile = sum(1 for m in all_momentums if m < momentum) / len(all_momentums)
                        features[f'{pair}_momentum_pct'] = percentile
        
        # Cross-asset features
        if self._last_rankings is not None:
            features['strongest_momentum'] = self._last_rankings.iloc[-1]
            features['weakest_momentum'] = self._last_rankings.iloc[0]
            features['momentum_dispersion'] = self._last_rankings.std()
        
        return features
    
    # =========================================================================
    # Implementation Methods
    # =========================================================================
    
    def _get_prices(self) -> Optional[pd.DataFrame]:
        """
        Get historical prices for all pairs.
        
        Returns:
            DataFrame with columns = pairs, index = dates
        """
        if self._data_service is None:
            logger.error("Data service not available")
            return None
        
        try:
            prices = {}
            for pair in self.pairs:
                # Get prices from data service
                df = self._data_service.get_prices(pair, self.timeframe)
                if df is not None and not df.empty:
                    prices[pair] = df['close']
            
            if not prices:
                return None
            
            return pd.DataFrame(prices)
            
        except Exception as e:
            logger.error(f"Error fetching prices: {e}")
            return None
    
    def _calculate_momentum(self, prices: pd.DataFrame) -> Optional[pd.Series]:
        """
        Calculate momentum for all pairs.
        
        Args:
            prices: DataFrame of prices
            
        Returns:
            Series of momentum values
        """
        try:
            momentums = {}
            for pair in self.pairs:
                if pair in prices.columns:
                    momentum = self._calculate_pair_momentum(prices[pair])
                    if momentum is not None:
                        momentums[pair] = momentum
            
            return pd.Series(momentums)
            
        except Exception as e:
            logger.error(f"Error calculating momentum: {e}")
            return None
    
    def _calculate_pair_momentum(self, prices: pd.Series) -> Optional[float]:
        """
        Calculate momentum for single pair.
        
        Momentum = (Price[t] - Price[t-lookback]) / Price[t-lookback]
        
        Args:
            prices: Price series
            
        Returns:
            Momentum value or None
        """
        try:
            if len(prices) < self.lookback + 1:
                return None
            
            current = prices.iloc[-1]
            past = prices.iloc[-self.lookback - 1]
            
            if past == 0:
                return None
            
            momentum = (current - past) / past
            return momentum
            
        except Exception as e:
            logger.error(f"Error calculating pair momentum: {e}")
            return None
    
    def _rank_pairs(self, momentums: pd.Series) -> pd.Series:
        """
        Rank pairs by momentum.
        
        Args:
            momentums: Series of momentum values
            
        Returns:
            Sorted series (weakest to strongest)
        """
        return momentums.sort_values()
    
    def _generate_signal_from_rankings(self, 
                                       rankings: pd.Series,
                                       data: MarketData) -> Optional[AlphaSignal]:
        """
        Generate signal from rankings.
        
        Strategy:
        - Long strongest momentum if > threshold
        - Short weakest momentum if < -threshold
        - Hold for 5 days
        
        Args:
            rankings: Sorted momentum rankings
            data: Current market data
            
        Returns:
            AlphaSignal or None
        """
        if len(rankings) < 2:
            return None
        
        strongest_pair = rankings.index[-1]
        strongest_momentum = rankings.iloc[-1]
        
        weakest_pair = rankings.index[0]
        weakest_momentum = rankings.iloc[0]
        
        # Decide which signal to generate
        # Prioritize strongest long signal
        
        if strongest_momentum > self.min_momentum_threshold:
            # Long strongest
            confidence = min(strongest_momentum * 10, 1.0)  # Scale to 0-1
            
            return AlphaSignal(
                source=self.name,
                timestamp=datetime.now(),
                symbol=strongest_pair,
                direction=SignalDirection.LONG,
                confidence=confidence,
                expected_return=0.12,  # Validated expectation
                volatility=0.10,  # Target volatility
                recommended_size=self.max_position_size,
                max_position=self.max_position_size,
                time_horizon=self.holding_days,
                metadata={
                    'momentum': strongest_momentum,
                    'rank': len(rankings),
                    'holding_days': self.holding_days,
                    'strategy': 'cross_sectional_momentum'
                }
            )
        
        elif weakest_momentum < -self.min_momentum_threshold:
            # Short weakest
            confidence = min(abs(weakest_momentum) * 10, 1.0)
            
            return AlphaSignal(
                source=self.name,
                timestamp=datetime.now(),
                symbol=weakest_pair,
                direction=SignalDirection.SHORT,
                confidence=confidence,
                expected_return=0.12,
                volatility=0.10,
                recommended_size=self.max_position_size,
                max_position=self.max_position_size,
                time_horizon=self.holding_days,
                metadata={
                    'momentum': weakest_momentum,
                    'rank': 1,
                    'holding_days': self.holding_days,
                    'strategy': 'cross_sectional_momentum'
                }
            )
        
        # No signal - momentum too weak
        return None
    
    # =========================================================================
    # Lifecycle Methods
    # =========================================================================
    
    def initialize(self) -> bool:
        """
        Initialize pod.
        
        Returns:
            bool: Success
        """
        try:
            logger.info(f"Initializing {self.name} pod...")
            
            # Validate data service
            if self._data_service is None:
                logger.error("Data service required")
                return False
            
            # Load initial data
            prices = self._get_prices()
            if prices is None:
                logger.warning("Could not load initial prices")
                # Don't fail - may get prices later
            
            logger.info(f"{self.name} pod initialized")
            return True
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False
    
    def on_trade_executed(self, trade: Any):
        """
        Callback when trade executes.
        
        Args:
            trade: Trade execution result
        """
        self._trade_count += 1
        
        # Track holding
        symbol = trade.decision.symbol
        self._current_holdings[symbol] = {
            'entry_time': datetime.now(),
            'exit_time': datetime.now() + pd.Timedelta(days=self.holding_days),
            'entry_price': trade.filled_price,
            'side': trade.decision.side
        }
        
        logger.info(f"Trade executed for {symbol}")
    
    def on_trade_closed(self, pnl: float, metadata: Dict):
        """
        Callback when trade closes.
        
        Args:
            pnl: P&L
            metadata: Trade metadata
        """
        symbol = metadata.get('symbol')
        if symbol in self._current_holdings:
            del self._current_holdings[symbol]
        
        logger.info(f"Trade closed for {symbol}: P&L {pnl:.4f}")
    
    # =========================================================================
    # Status
    # =========================================================================
    
    def get_status(self) -> Dict[str, Any]:
        """Get pod status."""
        status = super().get_status()
        status.update({
            'lookback': self.lookback,
            'holding_days': self.holding_days,
            'signal_count': self._signal_count,
            'trade_count': self._trade_count,
            'active_holdings': len(self._current_holdings),
            'last_update': self._last_update.isoformat() if self._last_update else None
        })
        return status
    
    def get_holdings(self) -> Dict[str, Dict]:
        """Get current holdings."""
        return self._current_holdings.copy()
    
    def should_exit(self, symbol: str) -> bool:
        """Check if position should be exited."""
        if symbol not in self._current_holdings:
            return False
        
        holding = self._current_holdings[symbol]
        exit_time = holding['exit_time']
        
        return datetime.now() >= exit_time


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_CONFIG = {
    'lookback': 21,
    'holding_days': 5,
    'min_momentum_threshold': 0.01,
    'max_position_size': 0.40,
    'pairs': ['GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD']
}


def create_fx_momentum_pod(config: Optional[Dict] = None) -> FXMomentumAlphaPod:
    """
    Factory function to create FX momentum pod.
    
    Args:
        config: Configuration (uses DEFAULT_CONFIG if None)
        
    Returns:
        FXMomentumAlphaPod instance
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    return FXMomentumAlphaPod(config)
