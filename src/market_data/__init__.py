"""
SCOPUS Data Layer - Canonical Market Data Architecture

Provides type-safe, unified access to multi-source, multi-timeframe OHLCV data.

Usage:
    >>> from src.market_data import MarketDataStore, Symbol, Timeframe, build_mtf_context
    >>> from pathlib import Path
    >>> 
    >>> # Initialize store
    >>> store = MarketDataStore(data_roots=[
    ...     Path("data/raw/forex_kaggle_multiTF"),
    ...     Path("data/raw/forex_backup_2020_2025"),
    ... ])
    >>> 
    >>> # Load single timeframe
    >>> ohlcv = store.load_ohlcv(
    ...     symbol=Symbol.EURUSD,
    ...     timeframe=Timeframe.M15,
    ...     start=datetime(2024, 1, 1),
    ...     end=datetime(2024, 12, 31),
    ... )
    >>> 
    >>> # Build multi-timeframe context
    >>> mtf = build_mtf_context(
    ...     store=store,
    ...     symbol=Symbol.EURUSD,
    ...     base_timeframe=Timeframe.M5,
    ...     higher_timeframes=[Timeframe.M15, Timeframe.H1, Timeframe.D1],
    ...     start=datetime(2024, 1, 1),
    ...     end=datetime(2024, 12, 31),
    ... )
"""

from .types import DataSource, Timeframe, Symbol, OHLCVBar, OHLCVFrame, MTFFrame
from .store import MarketDataStore, FileRegistry, FileRecord
from .mtf import build_mtf_context, add_mtf_features
from .features import (
    CANONICAL_FEATURES,
    FEATURE_GROUPS,
    get_features_for_timeframe,
    get_feature_columns_for_mtf,
    validate_features,
)

__all__ = [
    # Types
    "DataSource",
    "Timeframe", 
    "Symbol",
    "OHLCVBar",
    "OHLCVFrame",
    "MTFFrame",
    
    # Store
    "MarketDataStore",
    "FileRegistry",
    "FileRecord",
    
    # MTF
    "build_mtf_context",
    "add_mtf_features",
    
    # Features
    "CANONICAL_FEATURES",
    "FEATURE_GROUPS",
    "get_features_for_timeframe",
    "get_feature_columns_for_mtf",
    "validate_features",
]
