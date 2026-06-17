"""
Shared Data Service

Unified data access layer.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

from ..interfaces import MarketData, DataServiceInterface

logger = logging.getLogger(__name__)


class SharedDataService(DataServiceInterface):
    """
    Shared data service.
    
    Provides unified access to:
    - Historical prices
    - Real-time prices
    - Cached features
    - Multi-asset data
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize data service.
        
        Args:
            config: Configuration with keys:
                - data_root: Path to data directory
                - cache_enabled: bool
                - symbols: List of symbols
                - timeframes: List of timeframes
        """
        self.config = config
        self.data_root = Path(config.get('data_root', 'data/canonical'))
        self.cache_enabled = config.get('cache_enabled', True)
        
        self.symbols = config.get('symbols', [])
        self.timeframes = config.get('timeframes', ['M15', 'H1', 'D1'])
        
        # Cache
        self._cache: Dict[str, pd.DataFrame] = {}
        self._feature_cache: Dict[str, Dict] = {}
        
        # Connection
        self._connected = False
        
        logger.info("SharedDataService initialized")
    
    def initialize(self) -> bool:
        """Initialize service."""
        try:
            logger.info("Initializing data service...")
            
            # Validate paths
            if not self.data_root.exists():
                logger.warning(f"Data root not found: {self.data_root}")
            
            # Preload cache if enabled
            if self.cache_enabled:
                self._preload_cache()
            
            logger.info("Data service initialized")
            return True
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False
    
    def health_check(self) -> bool:
        """Health check."""
        return self._connected or not self.cache_enabled
    
    def shutdown(self):
        """Shutdown service."""
        self._connected = False
        self._cache.clear()
        logger.info("Data service shutdown")
    
    def _get_cache_key(self, symbol: str, timeframe: str) -> str:
        """Get cache key."""
        return f"{symbol}_{timeframe}"
    
    def _preload_cache(self):
        """Preload data into cache."""
        logger.info("Preloading cache...")
        
        for symbol in self.symbols:
            for timeframe in self.timeframes:
                try:
                    df = self._load_from_disk(symbol, timeframe)
                    if df is not None:
                        key = self._get_cache_key(symbol, timeframe)
                        self._cache[key] = df
                        logger.debug(f"Cached: {key}")
                except Exception as e:
                    logger.warning(f"Failed to cache {symbol}_{timeframe}: {e}")
        
        logger.info(f"Cache preloaded: {len(self._cache)} datasets")
    
    def _load_from_disk(self, 
                       symbol: str, 
                       timeframe: str) -> Optional[pd.DataFrame]:
        """Load data from disk."""
        try:
            # Construct path
            filename = f"{symbol}_{timeframe}_MASTER.parquet"
            filepath = self.data_root / filename
            
            if not filepath.exists():
                logger.warning(f"File not found: {filepath}")
                return None
            
            # Load
            df = pd.read_parquet(filepath)
            
            # Ensure datetime index
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            
            logger.debug(f"Loaded {len(df)} rows from {filepath}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading {symbol}_{timeframe}: {e}")
            return None
    
    def get_prices(self,
                   symbol: str,
                   timeframe: str,
                   start: Optional[datetime] = None,
                   end: Optional[datetime] = None) -> pd.DataFrame:
        """
        Get historical prices.
        
        Args:
            symbol: Currency pair (e.g., 'EURUSD')
            timeframe: Timeframe (e.g., 'M15', 'H1', 'D1')
            start: Start date
            end: End date
            
        Returns:
            DataFrame with OHLCV data
        """
        key = self._get_cache_key(symbol, timeframe)
        
        # Check cache
        if self.cache_enabled and key in self._cache:
            df = self._cache[key].copy()
        else:
            # Load from disk
            df = self._load_from_disk(symbol, timeframe)
        
        if df is None:
            return pd.DataFrame()
        
        # Filter by date
        if start:
            df = df[df.index >= start]
        if end:
            df = df[df.index <= end]
        
        return df
    
    def get_latest(self, symbol: str, timeframe: str = 'M15') -> MarketData:
        """
        Get latest price.
        
        Args:
            symbol: Currency pair
            timeframe: Timeframe
            
        Returns:
            MarketData
        """
        df = self.get_prices(symbol, timeframe)
        
        if df.empty:
            raise ValueError(f"No data for {symbol}_{timeframe}")
        
        # Get last row
        last = df.iloc[-1]
        timestamp = df.index[-1]
        
        return MarketData(
            symbol=symbol,
            timestamp=timestamp,
            timeframe=timeframe,
            open=last.get('open', 0),
            high=last.get('high', 0),
            low=last.get('low', 0),
            close=last.get('close', 0),
            volume=last.get('volume', 0)
        )
    
    def get_features(self, symbol: str) -> Dict[str, float]:
        """
        Get computed features.
        
        Args:
            symbol: Currency pair
            
        Returns:
            Dict of features
        """
        # Check cache
        if symbol in self._feature_cache:
            return self._feature_cache[symbol].copy()
        
        # Compute features
        features = {}
        
        try:
            # Get prices
            df = self.get_prices(symbol, 'D1')
            if df.empty:
                return features
            
            # Calculate features
            closes = df['close']
            
            # Returns
            features['return_1d'] = closes.pct_change().iloc[-1]
            features['return_1w'] = closes.pct_change(5).iloc[-1]
            features['return_1m'] = closes.pct_change(21).iloc[-1]
            
            # Volatility
            features['volatility_20d'] = closes.pct_change().rolling(20).std().iloc[-1]
            
            # Trend
            features['trend_20d'] = 1 if closes.iloc[-1] > closes.iloc[-20] else -1
            
            # RSI (simple)
            delta = closes.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            features['rsi_14'] = rsi.iloc[-1]
            
            # Cache
            if self.cache_enabled:
                self._feature_cache[symbol] = features.copy()
            
        except Exception as e:
            logger.error(f"Error computing features for {symbol}: {e}")
        
        return features
    
    def clear_cache(self):
        """Clear all caches."""
        self._cache.clear()
        self._feature_cache.clear()
        logger.info("Cache cleared")
