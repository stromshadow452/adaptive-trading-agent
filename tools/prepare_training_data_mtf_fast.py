"""
Fast MTF training data preparation - Direct CSV loading

Skips registry building for faster execution.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from tqdm import tqdm
import numpy as np
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_csv_direct(csv_path, timeframe_name):
    """Load CSV and add timeframe prefix to columns"""
    logger.info(f"Loading {csv_path.name}...")
    
    df = pd.read_csv(csv_path)
    
    # Detect timestamp column - try common names first
    timestamp_col = None
    for col in ['timestamp', 'Datetime', 'Date', 'date', 'time', 'datetime', 'Time', 'DATE', 'Timestamp']:
        if col in df.columns:
            timestamp_col = col
            break
    
    # If still not found, use first column
    if not timestamp_col and len(df.columns) > 0:
        timestamp_col = df.columns[0]
        logger.info(f"Using first column as timestamp: {timestamp_col}")
    
    if not timestamp_col:
        logger.error(f"Columns in file: {df.columns.tolist()}")
        raise ValueError("No timestamp column found")
    
    # Normalize
    df['timestamp'] = pd.to_datetime(df[timestamp_col])
    
    # Remove timezone if present (for consistent merging)
    if df['timestamp'].dt.tz is not None:
        df['timestamp'] = df['timestamp'].dt.tz_localize(None)
    
    df = df.set_index('timestamp').sort_index()
    
    # Normalize column names
    df.columns = df.columns.str.lower()
    
    # Keep only OHLCV
    keep_cols = ['open', 'high', 'low', 'close', 'volume']
    df = df[[col for col in keep_cols if col in df.columns]]
    
    # Add timeframe prefix
    df = df.rename(columns={col: f"{timeframe_name}_{col}" for col in df.columns})
    
    return df


def compute_features(df, tf_prefix):
    """Compute features for a timeframe"""
    close_col = f"{tf_prefix}_close"
    
    if close_col not in df.columns:
        return df
    
    # RSI
    delta = df[close_col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df[f"{tf_prefix}_rsi_14"] = 100 - (100 / (1 + rs))
    
    # ATR
    if f"{tf_prefix}_high" in df.columns and f"{tf_prefix}_low" in df.columns:
        high_low = df[f"{tf_prefix}_high"] - df[f"{tf_prefix}_low"]
        high_close = np.abs(df[f"{tf_prefix}_high"] - df[close_col].shift())
        low_close = np.abs(df[f"{tf_prefix}_low"] - df[close_col].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df[f"{tf_prefix}_atr_14"] = tr.rolling(14).mean()
    
    # Trend flag
    sma_20 = df[close_col].rolling(20).mean()
    sma_50 = df[close_col].rolling(50).mean()
    df[f"{tf_prefix}_trend_flag"] = 0
    df.loc[sma_20 > sma_50, f"{tf_prefix}_trend_flag"] = 1
    df.loc[sma_20 < sma_50, f"{tf_prefix}_trend_flag"] = -1
    
    return df


def main():
    logger.info("Fast MTF training data preparation...")
    logger.info("=" * 60)
    
    # Define CSV files (only available timeframes)
    data_dir = Path("data/raw/forex_kaggle_multiTF")
    
    csv_files = {
        'M15': data_dir / "EURUSD_M15.csv",
        'H1': data_dir / "EURUSD_H1.csv",
        'H4': data_dir / "EURUSD_H4.csv",
        'D1': data_dir / "EURUSD_D1.csv",
    }
    
    # Check files exist
    print("\n📁 Checking files...")
    for tf, path in tqdm(csv_files.items(), desc="Checking", ncols=80):
        if not path.exists():
            logger.error(f"File not found: {path}")
            return
    print("✅ All files found!")
    
    # 1. Load base timeframe (M15)
    print("\n📊 Step 1/5: Loading base timeframe (M15)...")
    base_df = load_csv_direct(csv_files['M15'], 'M15')
    
    # Filter to 2020-2023
    base_df = base_df[(base_df.index >= '2020-01-01') & (base_df.index <= '2023-12-31')]
    print(f"✅ Loaded {len(base_df):,} bars")
    
    # 2. Load and merge higher timeframes
    print("\n🔗 Step 2/5: Merging higher timeframes...")
    
    for tf in tqdm(['H1', 'H4', 'D1'], desc="Merging", ncols=80):
        tf_df = load_csv_direct(csv_files[tf], tf)
        
        # Filter to same range (with padding)
        tf_df = tf_df[(tf_df.index >= '2019-01-01') & (tf_df.index <= '2023-12-31')]
        
        # Merge asof (backward)
        base_df = pd.merge_asof(
            left=base_df.reset_index().sort_values('timestamp'),
            right=tf_df.reset_index().sort_values('timestamp'),
            on='timestamp',
            direction='backward'
        ).set_index('timestamp').sort_index()
    
    print(f"✅ Merged: {len(base_df):,} bars, {len(base_df.columns)} columns")
    
    # 3. Compute features
    print("\n⚙️  Step 3/5: Computing features...")
    
    for tf in tqdm(['M15', 'H1', 'H4', 'D1'], desc="Features", ncols=80):
        base_df = compute_features(base_df, tf)
    
    print(f"✅ Total columns: {len(base_df.columns)}")
    
    # 4. Create target
    print("\n🎯 Step 4/5: Creating target variable...")
    
    base_df['future_return'] = base_df['M15_close'].shift(-10) / base_df['M15_close'] - 1
    base_df['target'] = (base_df['future_return'] > 0).astype(int)
    
    # 5. Drop NaN
    base_df = base_df.dropna()
    
    print(f"✅ Final dataset: {len(base_df):,} rows")
    print(f"   Target distribution: {base_df['target'].value_counts().to_dict()}")
    
    # 6. Save
    print("\n💾 Step 5/5: Saving...")
    output_dir = Path("data/training")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "eurusd_m15_mtf_features.csv"
    base_df.reset_index().to_csv(output_file, index=False)
    
    print(f"\n✅ SUCCESS! Saved to {output_file}")
    print(f"   Total features: {len(base_df.columns)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
