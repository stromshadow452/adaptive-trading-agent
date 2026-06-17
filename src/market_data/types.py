"""
SCOPUS Data Layer - Core Types & Enums

Type-safe definitions for symbols, timeframes, and data sources.
Prevents string-based bugs and enables autocomplete.

ADAPTER CONTRACT:
    Both historical and live data sources implement DataSourceABC,
    so the pipeline never touches raw files or WebSocket frames directly.
    Switching history → live = swap the adapter only.
"""

from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from typing import TypeAlias, Iterator, List, Optional
import pandas as pd



class DataSource(str, Enum):
    """Data source identifier"""
    KAGGLE = "kaggle"
    MT5 = "mt5"
    SYNTHETIC = "synthetic"


class Timeframe(str, Enum):
    """Standard forex timeframes"""
    M1  = "M1"   # 1 minute
    M5  = "M5"   # 5 minutes
    M15 = "M15"  # 15 minutes
    M30 = "M30"  # 30 minutes
    H1  = "H1"   # 1 hour
    H4  = "H4"   # 4 hours
    D1  = "D1"   # Daily
    W1  = "W1"   # Weekly
    MN1 = "MN1"  # Monthly
    
    @property
    def minutes(self) -> int:
        """Convert timeframe to minutes"""
        mapping = {
            "M1": 1, "M5": 5, "M15": 15, "M30": 30,
            "H1": 60, "H4": 240, "D1": 1440,
            "W1": 10080, "MN1": 43200
        }
        return mapping[self.value]


class Symbol(str, Enum):
    """Supported trading symbols"""
    # Major pairs
    EURUSD = "EURUSD"
    GBPUSD = "GBPUSD"
    USDJPY = "USDJPY"
    USDCAD = "USDCAD"
    USDCHF = "USDCHF"
    
    # Cross pairs
    EURGBP = "EURGBP"
    EURJPY = "EURJPY"
    GBPJPY = "GBPJPY"
    AUDUSD = "AUDUSD"
    AUDJPY = "AUDJPY"
    NZDUSD = "NZDUSD"
    NZDJPY = "NZDJPY"
    
    # Exotic pairs
    EURCAD = "EURCAD"
    EURCHF = "EURCHF"
    GBPCHF = "GBPCHF"
    CHFJPY = "CHFJPY"
    
    # Metals
    XAGUSD = "XAGUSD"  # Silver


@dataclass
class OHLCVBar:
    """
    Single OHLCV bar - canonical schema contract.
    
    This is the schema contract for all market data.
    For performance, we use DataFrames, but this defines the structure.
    """
    timestamp: datetime  # tz-aware UTC
    symbol: Symbol
    timeframe: Timeframe
    open: float
    high: float
    low: float
    close: float
    volume: float
    source: DataSource


# Type aliases for clarity
OHLCVFrame: TypeAlias = pd.DataFrame
"""
DataFrame with OHLCV data for a single (symbol, timeframe) pair.

Index: DatetimeIndex (tz-aware UTC, sorted ascending)
Columns: ["open", "high", "low", "close", "volume", "source"]

Invariants:
- No duplicate timestamps
- Index is sorted
- All timestamps are tz-aware UTC
"""

MTFFrame: TypeAlias = pd.DataFrame
"""
Multi-timeframe feature frame.

Index: DatetimeIndex (base timeframe, tz-aware UTC)
Columns: Prefixed by timeframe (e.g., "M5_open", "H1_rsi_14", "D1_trend_flag")

Example columns:
- M5_open, M5_high, M5_low, M5_close, M5_volume
- M5_rsi_14, M5_atr_14, M5_trend_flag
- H1_open, H1_close, H1_rsi_14, H1_trend_flag
- D1_open, D1_close, D1_trend_flag, D1_sr_zone_id
"""


# ===========================================================================
# DATA SOURCE ADAPTER CONTRACT
# ===========================================================================
#
# Both historical CSV readers and live WebSocket / MT5 adapters
# must implement this interface. The pipeline calls only these methods
# and therefore NEVER needs to know about files, sockets, or brokers.
#
# To switch from backtest → live: swap the adapter, not the pipeline.
# ===========================================================================

class DataSourceABC(ABC):
    """
    Abstract base for all market data sources.

    Implement this for:
      - HistoricalCSVSource  (current — reads from CSV files)
      - MT5LiveSource         (future — streams from MT5 terminal)
      - WebSocketSource       (future — direct exchange WebSocket)
    """

    @abstractmethod
    def stream(self, symbol: str, timeframe: Timeframe) -> Iterator[OHLCVBar]:
        """
        Yield OHLCVBar objects one at a time (bar-by-bar).
        For historical: iterate over CSV rows.
        For live: block until next bar closes, then yield.
        """
        ...

    @abstractmethod
    def get_history(self, symbol: str, timeframe: Timeframe, n_bars: int) -> OHLCVFrame:
        """
        Return the last n_bars as an OHLCVFrame (OHLCV DataFrame).
        Must return bars sorted ascending by timestamp.
        """
        ...

    @abstractmethod
    def get_latest_bar(self, symbol: str, timeframe: Timeframe) -> Optional[OHLCVBar]:
        """Return the most recent completed bar, or None if not available."""
        ...

    @property
    @abstractmethod
    def source_type(self) -> DataSource:
        """Identify the data source type (KAGGLE / MT5 / SYNTHETIC)."""
        ...


class HistoricalCSVSource(DataSourceABC):
    """
    Historical data adapter — reads from local CSV files.
    Current production implementation for backtest mode.

    Directory structure expected:
        data/<symbol>_<timeframe>.csv
        e.g. data/EURUSD_H1.csv

    TODO: Connect to src/market_data/unified_loader.py
    """

    def __init__(self, data_dir: str = "data/raw/forex_kaggle_multiTF"):
        self.data_dir = data_dir
        self._cache: dict = {}

    @property
    def source_type(self) -> DataSource:
        return DataSource.KAGGLE

    def get_history(self, symbol: str, timeframe: Timeframe, n_bars: int) -> OHLCVFrame:
        """Load from CSV cache. Returns last n_bars rows."""
        # Use UnifiedDataLoader so timestamp semantics match backtest.
        # (Backtest relies on candle timestamps, not wall-clock time.)
        key = f"{symbol}_{timeframe.value}"
        if key not in self._cache:
            from src.market_data.unified_loader import UnifiedDataLoader

            loader = UnifiedDataLoader(source1_path=self.data_dir)
            df, _audit = loader.load(symbol, timeframe.value)
            if df is None or len(df) == 0:
                raise FileNotFoundError(f"No data loaded for {key} in {self.data_dir}")

            df = df.copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
            df = df.set_index("timestamp")
            self._cache[key] = df

        df = self._cache[key]
        return df.tail(n_bars)

    def get_latest_bar(self, symbol: str, timeframe: Timeframe) -> Optional[OHLCVBar]:
        """Return the last row of the CSV as an OHLCVBar."""
        try:
            df = self.get_history(symbol, timeframe, 1)
            if df.empty:
                return None
            row = df.iloc[-1]
            ts = df.index[-1]
            return OHLCVBar(
                timestamp=ts.to_pydatetime(),
                symbol=symbol,    # type: ignore
                timeframe=timeframe,
                open=float(row.get("open", 0)),
                high=float(row.get("high", 0)),
                low=float(row.get("low", 0)),
                close=float(row.get("close", 0)),
                volume=float(row.get("volume", 0)),
                source=DataSource.KAGGLE,
            )
        except Exception:
            return None

    def stream(self, symbol: str, timeframe: Timeframe) -> Iterator[OHLCVBar]:
        """Iterate over all bars in CSV (backtest mode: instant replay)."""
        df = self.get_history(symbol, timeframe, n_bars=99999)
        for ts, row in df.iterrows():
            yield OHLCVBar(
                timestamp=pd.Timestamp(ts).to_pydatetime(),
                symbol=symbol,    # type: ignore
                timeframe=timeframe,
                open=float(row.get("open", 0)),
                high=float(row.get("high", 0)),
                low=float(row.get("low", 0)),
                close=float(row.get("close", 0)),
                volume=float(row.get("volume", 0)),
                source=DataSource.KAGGLE,
            )

