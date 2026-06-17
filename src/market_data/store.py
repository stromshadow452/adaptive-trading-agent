"""
SCOPUS Data Layer - Market Data Store

Unified gateway to all market data sources.
Handles CSV/Parquet loading, deduplication, and caching.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
import pandas as pd
import logging

from .types import Symbol, Timeframe, DataSource, OHLCVFrame

logger = logging.getLogger(__name__)


FileKey = Tuple[Symbol, Timeframe, int, DataSource]


@dataclass
class FileRecord:
    """Metadata for a single data file"""
    path: Path
    symbol: Symbol
    timeframe: Timeframe
    year: int
    source: DataSource
    start: datetime
    end: datetime
    row_count: int


class FileRegistry:
    """
    Index of all available data files.
    
    Maps (symbol, timeframe, year, source) -> FileRecord
    Enables fast lookup of which files contain data for a given time range.
    """
    
    def __init__(self):
        self.files: Dict[FileKey, FileRecord] = {}
    
    def add_file(self, rec: FileRecord):
        """Register a data file"""
        key = (rec.symbol, rec.timeframe, rec.year, rec.source)
        self.files[key] = rec
        logger.debug(f"Registered: {rec.symbol.value}_{rec.timeframe.value}_{rec.year} ({rec.source.value})")
    
    def find_files(
        self,
        symbol: Symbol,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
    ) -> List[FileRecord]:
        """Find all files that intersect the given time range"""
        import pytz
        
        # Ensure start/end are timezone-aware (UTC)
        if start.tzinfo is None:
            start = pytz.UTC.localize(start)
        if end.tzinfo is None:
            end = pytz.UTC.localize(end)
        
        matching = []
        
        for key, rec in self.files.items():
            if rec.symbol != symbol or rec.timeframe != timeframe:
                continue
            
            # Check if file's time range intersects [start, end]
            if rec.end < start or rec.start > end:
                continue
            
            matching.append(rec)
        
        # Sort by start time
        matching.sort(key=lambda r: r.start)
        return matching
    
    def get_available_symbols(self) -> List[Symbol]:
        """Get list of all available symbols"""
        return sorted(list(set(rec.symbol for rec in self.files.values())))
    
    def get_available_timeframes(self, symbol: Symbol) -> List[Timeframe]:
        """Get available timeframes for a symbol"""
        return sorted(list(set(
            rec.timeframe for rec in self.files.values()
            if rec.symbol == symbol
        )), key=lambda tf: tf.minutes)


class MarketDataStore:
    """
    Unified market data access layer.
    
    Provides clean, cached access to OHLCV data from multiple sources.
    Handles deduplication, normalization, and alignment.
    """
    
    def __init__(self, data_roots: List[Path], registry: Optional[FileRegistry] = None):
        """
        Initialize market data store.
        
        Args:
            data_roots: List of root directories containing data
            registry: Pre-built file registry (or None to build on init)
        """
        self.data_roots = data_roots
        self.registry = registry or FileRegistry()
        
        if registry is None:
            self._build_registry()
    
    def _build_registry(self):
        """Scan data roots and build file registry"""
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = lambda x, **kwargs: x

        files_found = []
        for root in self.data_roots:
            if not root.exists():
                continue
            
            # Recursively find all CSV and Parquet files
            files = list(root.rglob("*.csv")) + list(root.rglob("*.parquet"))
            files_found.extend(files)
            
        logger.info(f"Scanning {len(files_found)} files...")
        
        for file_path in tqdm(files_found, desc="Building Registry"):
            try:
                self._register_file(file_path)
            except Exception as e:
                logger.warning(f"Failed to register {file_path}: {e}")

    def _register_file(self, path: Path):
        """Register a single file"""
        rec = self._parse_file_metadata(path)
        if rec:
            self.registry.add_file(rec)

    def _parse_file_metadata(self, path: Path) -> Optional[FileRecord]:
        """
        Extract metadata from filename and file contents.
        Optimized to avoid full file reads.
        """
        filename = path.stem
        parts = filename.split('_')
        
        if len(parts) < 2:
            return None
        
        # Timeframe alias mapping for different naming conventions
        TIMEFRAME_ALIASES = {
            'Daily': 'D1', 'daily': 'D1', '1D': 'D1',
            'Weekly': 'W1', 'weekly': 'W1',
            'Monthly': 'MN1', 'monthly': 'MN1',
            '1H': 'H1', '4H': 'H4',
            '5M': 'M5', '15M': 'M15', '30M': 'M30',
        }
        
        # Parse symbol and timeframe
        try:
            symbol = Symbol(parts[0])
            tf_raw = parts[1]
            tf_normalized = TIMEFRAME_ALIASES.get(tf_raw, tf_raw)
            timeframe = Timeframe(tf_normalized)
        except (ValueError, KeyError):
            return None
        
        # Detect source from path
        if "kaggle" in str(path).lower():
            source = DataSource.KAGGLE
        elif "mt5" in str(path).lower() or "backup" in str(path).lower():
            source = DataSource.MT5
        else:
            source = DataSource.KAGGLE  # Default
        
        try:
            # 1. Read header and first row (auto-detect delimiter)
            df_first = self._read_csv_auto(path, nrows=1)
            timestamp_col = self._detect_timestamp_column(df_first)
            start = pd.to_datetime(df_first[timestamp_col].iloc[0].replace('.', '-') if isinstance(df_first[timestamp_col].iloc[0], str) else df_first[timestamp_col].iloc[0])
            
            # 2. Read last row efficiently (seek to end)
            with open(path, 'rb') as f:
                f.seek(0, 2) # Seek to end
                file_size = f.tell()
                
                # Seek back to find last newline
                offset = min(1024, file_size)
                f.seek(-offset, 2)
                last_chunk = f.read()
                last_lines = last_chunk.decode('utf-8', errors='ignore').splitlines()
                
                # Parse last line
                if last_lines:
                    last_line = last_lines[-1]
                    # Split by comma (assuming CSV)
                    vals = last_line.split(',')
                    # Find timestamp index from header
                    header = list(df_first.columns)
                    ts_idx = header.index(timestamp_col)
                    
                    if len(vals) > ts_idx:
                        end = pd.to_datetime(vals[ts_idx])
                    else:
                        # Fallback if parsing fails
                        end = start
                else:
                    end = start

            # 3. Estimate row count (file_size / avg_row_size)
            # This is much faster than counting lines
            avg_row_size = 50 # bytes
            row_count = int(file_size / avg_row_size)
            
            year = start.year
            
            # Make timestamps timezone-aware (UTC) for comparison
            if start.tzinfo is None:
                start = start.tz_localize('UTC')
            if end.tzinfo is None:
                end = end.tz_localize('UTC')
            
            return FileRecord(
                path=path,
                symbol=symbol,
                timeframe=timeframe,
                year=year,
                source=source,
                start=start,
                end=end,
                row_count=row_count
            )
        except Exception as e:
            logger.debug(f"Could not parse metadata for {path}: {e}")
            return None

    def _read_csv_auto(self, path: Path, **kwargs) -> pd.DataFrame:
        """
        Read CSV with auto-detection of delimiter (comma vs tab).
        Handles MT5 export format with tabs.
        """
        # First, try to detect delimiter by reading first line
        try:
            with open(path, 'r', encoding='utf-8') as f:
                first_line = f.readline()
            
            # If tabs present, use tab delimiter
            if '\t' in first_line:
                df = pd.read_csv(path, sep='\t', **kwargs)
            else:
                df = pd.read_csv(path, **kwargs)
            
            return df
        except Exception as e:
            logger.debug(f"_read_csv_auto failed for {path}: {e}")
            return pd.DataFrame()

    def _detect_timestamp_column(self, df: pd.DataFrame) -> str:
        """Auto-detect timestamp column name"""
        # Include MT5 format with angle brackets and tabs
        for col in ['timestamp', 'Date', 'date', 'time', 'datetime', 'Datetime', 'Time', '<DATE>', '<date>']:
            if col in df.columns:
                return col
        raise ValueError(f"No timestamp column found in {df.columns.tolist()}")
    
    @lru_cache(maxsize=256)
    def load_ohlcv(
        self,
        symbol: Symbol,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
    ) -> OHLCVFrame:
        """
        Load OHLCV data for a symbol and timeframe.
        
        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            start: Start datetime (UTC)
            end: End datetime (UTC)
        
        Returns:
            OHLCVFrame with canonical schema
        """
        files = self.registry.find_files(symbol, timeframe, start, end)
        
        if not files:
            logger.warning(f"No data found for {symbol.value} {timeframe.value} [{start} to {end}]")
            return self._empty_ohlcv_frame()
        
        # === DEBUG D1 ===
        if timeframe.value == 'D1':
            logger.warning(f"[load_ohlcv D1] Found {len(files)} files to load")
            for f in files:
                logger.warning(f"[load_ohlcv D1] File: {f.path}")
        
        logger.debug(f"Loading {len(files)} files for {symbol.value} {timeframe.value}")
        
        dfs = []
        for rec in files:
            df = self._read_file(rec.path, rec.source)
            if timeframe.value == 'D1':
                logger.warning(f"[load_ohlcv D1] Read {len(df)} rows from {rec.path}")
            dfs.append(df)
        
        if not dfs:
            return self._empty_ohlcv_frame()
        
        # Concatenate and sort
        raw = pd.concat(dfs).sort_index()
        
        if timeframe.value == 'D1':
            logger.debug(f"[load_ohlcv D1] After concat: {len(raw)} rows")
            if len(raw) > 0:
                logger.debug(f"[load_ohlcv D1] Date range in data: {raw.index.min()} to {raw.index.max()}")
        
        # Deduplicate: prefer MT5 over Kaggle
        cleaned = self._dedupe_prefer_mt5(raw)
        
        # === FINAL FIX FOR D1: Use normalized DatetimeIndex (not .date) ===
        if timeframe.value == 'D1':
            logger.debug(f"[D1] After dedupe: {len(cleaned)} rows")
            
            # Force index normalization to midnight UTC timestamps
            # Do NOT use .index.date - it silently fails on non-DatetimeIndex
            cleaned.index = pd.to_datetime(cleaned.index, utc=True).normalize()
            
            # Normalize start/end to midnight UTC for comparison
            start_d = pd.to_datetime(start, utc=True).normalize()
            end_d = pd.to_datetime(end, utc=True).normalize()
            
            logger.debug(f"[D1] Filter: {start_d} to {end_d}")
            
            # Filter using DatetimeIndex comparison (reliable)
            cleaned = cleaned.loc[(cleaned.index >= start_d) & (cleaned.index <= end_d)]
            
            logger.debug(f"[D1] After filter: {len(cleaned)} rows")
            
            if len(cleaned) == 0:
                logger.warning(f"[D1] Zero rows returned for {symbol.value} - data may not cover range.")
            
            # EARLY RETURN - do NOT fall through to intraday logic
            logger.debug(f"Loaded {len(cleaned)} bars for {symbol.value} D1")
            return cleaned
        
        # Intraday timeframes: use timestamp comparison with timezone handling
        import pytz
        if start.tzinfo is None:
            start = pytz.UTC.localize(start)
        if end.tzinfo is None:
            end = pytz.UTC.localize(end)
        
        # Ensure index is timezone-aware for comparison
        if cleaned.index.tz is None:
            cleaned.index = cleaned.index.tz_localize("UTC")
        
        cleaned = cleaned[(cleaned.index >= start) & (cleaned.index <= end)]
        
        logger.info(f"Loaded {len(cleaned)} bars for {symbol.value} {timeframe.value}")
        
        return cleaned
    
    def _read_file(self, path: Path, source: DataSource) -> OHLCVFrame:
        """Read and normalize a single data file"""
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            try:
                df = self._read_csv_auto(path)
            except Exception as e:
                # Fallback to python engine for corrupted files
                logger.warning(f"CSV parse error, trying python engine: {path}")
                try:
                    df = pd.read_csv(path, engine='python', on_bad_lines='skip')
                except Exception as e2:
                    logger.error(f"Failed to read CSV: {path}: {e2}")
                    return self._empty_ohlcv_frame()
            df = self._normalize_csv(df, source)
        
        # Ensure canonical schema
        df = df.set_index("timestamp").sort_index()
        
        # Ensure UTC timezone
        if df.index.tzinfo is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        
        return df[["open", "high", "low", "close", "volume", "source"]]
    
    def _normalize_csv(self, df: pd.DataFrame, source: DataSource) -> pd.DataFrame:
        """
        Normalize CSV to canonical schema.
        
        Handles different column naming conventions:
        - Kaggle: Date, Open, High, Low, Close, Volume
        - MT5: timestamp, open, high, low, close, volume
        - MT5 Export: <DATE>, <OPEN>, <HIGH>, <LOW>, <CLOSE>, <TICKVOL>
        """
        # Detect and rename timestamp column
        timestamp_col = self._detect_timestamp_column(df)
        df = df.rename(columns={timestamp_col: "timestamp"})
        
        # Normalize OHLCV column names to lowercase (handle multiple formats)
        column_mapping = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
            # MT5 export format with angle brackets
            "<OPEN>": "open",
            "<HIGH>": "high",
            "<LOW>": "low",
            "<CLOSE>": "close",
            "<TICKVOL>": "volume",
            "<VOL>": "volume",
        }
        df = df.rename(columns=column_mapping)
        
        # Parse timestamp (handle 2022.01.03 format)
        df["timestamp"] = df["timestamp"].apply(
            lambda x: x.replace('.', '-') if isinstance(x, str) and '.' in x[:10] else x
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Add source column
        df["source"] = source.value
        
        # Ensure required columns exist
        for col in ['open', 'high', 'low', 'close']:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        if 'volume' not in df.columns:
            df['volume'] = 0
        
        return df
    
    def _dedupe_prefer_mt5(self, df: OHLCVFrame) -> OHLCVFrame:
        """
        Remove duplicate timestamps, preferring MT5 over Kaggle.
        """
        df = df.reset_index()
        
        # Define source priority
        source_priority = {"mt5": 1, "kaggle": 0, "synthetic": -1}
        df["source_priority"] = df["source"].map(source_priority)
        
        # Sort by timestamp and priority (descending)
        df = df.sort_values(["timestamp", "source_priority"], ascending=[True, False])
        
        # Keep first (highest priority) for each timestamp
        df = df.drop_duplicates(subset=["timestamp"], keep="first")
        
        # Clean up
        df = df.drop(columns=["source_priority"])
        
        return df.set_index("timestamp").sort_index()
    
    def _empty_ohlcv_frame(self) -> OHLCVFrame:
        """Return empty OHLCVFrame with correct schema"""
        return pd.DataFrame(
            columns=["open", "high", "low", "close", "volume", "source"]
        ).rename_axis("timestamp")
