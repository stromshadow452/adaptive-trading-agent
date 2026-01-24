"""
Simplified training data preparation - uses existing CSV directly

This version skips MarketDataStore and loads CSV directly for faster testing.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators"""
    df = df.copy()
    
    # Price-based
    df['returns'] = df['close'].pct_change()
    
    # Trend
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['ema_12'] = df['close'].ewm(span=12).mean()
    df['ema_26'] = df['close'].ewm(span=26).mean()
    df['trend_flag'] = (df['sma_20'] > df['sma_50']).astype(int)
    
    # Momentum
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Volatility
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(14).mean()
    
    df['bb_middle'] = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + (2 * bb_std)
    df['bb_lower'] = df['bb_middle'] - (2 * bb_std)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    
    df['volatility'] = df['returns'].rolling(20).std()
    
    return df


def create_target(df: pd.DataFrame, forward_bars: int = 10) -> pd.DataFrame:
    """Create target variable (future return)"""
    df = df.copy()
    
    # Future return
    df['future_return'] = df['close'].shift(-forward_bars) / df['close'] - 1
    
    # Binary target: 1 if future return > 0, else 0
    df['target'] = (df['future_return'] > 0).astype(int)
    
    return df


def main():
    logger.info("Preparing training data (simplified version)...")
    
    # Load directly from CSV
    csv_file = Path("data/raw/forex_kaggle_multiTF/EURUSD_M15.csv")
    
    if not csv_file.exists():
        logger.error(f"CSV file not found: {csv_file}")
        logger.error("Please check your data directory")
        return
    
    logger.info(f"Loading {csv_file}...")
    df = pd.read_csv(csv_file)
    
    # Detect timestamp column
    timestamp_col = None
    for col in df.columns:
        if 'time' in col.lower() or 'date' in col.lower():
            timestamp_col = col
            break
    
    if timestamp_col:
        df['timestamp'] = pd.to_datetime(df[timestamp_col])
    
    # Normalize column names
    df.columns = df.columns.str.lower()
    
    logger.info(f"Loaded {len(df)} candles")
    
    # Filter to 2020-2023 for faster processing
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df[(df['timestamp'] >= '2020-01-01') & (df['timestamp'] <= '2023-12-31')]
        logger.info(f"Filtered to 2020-2023: {len(df)} candles")
    
    # Compute features
    logger.info("Computing features...")
    df = compute_features(df)
    
    # Create target
    logger.info("Creating target variable...")
    df = create_target(df, forward_bars=10)
    
    # Drop NaN rows
    df = df.dropna()
    
    logger.info(f"Final dataset: {len(df)} rows")
    
    # Save
    output_dir = Path("data/training")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "eurusd_m15_features.csv"
    df.to_csv(output_file, index=False)
    
    logger.info(f"Saved to {output_file}")
    logger.info(f"Target distribution: {df['target'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
