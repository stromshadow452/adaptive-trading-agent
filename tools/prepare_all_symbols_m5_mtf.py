"""
MTF Training Data - All Symbols with M5 Base

Creates MTF features for all available symbols using M5 as base timeframe.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from tqdm import tqdm
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_csv_direct(csv_path, timeframe_name):
    """Load CSV and add timeframe prefix to columns"""
    df = pd.read_csv(csv_path)
    
    # Detect timestamp column
    timestamp_col = None
    for col in ['timestamp', 'Datetime', 'Date', 'date', 'time', 'datetime', 'Time']:
        if col in df.columns:
            timestamp_col = col
            break
    
    if not timestamp_col and len(df.columns) > 0:
        timestamp_col = df.columns[0]
    
    # Normalize
    df['timestamp'] = pd.to_datetime(df[timestamp_col])
    
    # Remove timezone if present
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


def process_symbol(symbol, data_dir):
    """Process one symbol"""
    print(f"\n{'='*60}")
    print(f"Processing {symbol}...")
    print(f"{'='*60}")
    
    # Define files
    csv_files = {
        'M5': data_dir / f"{symbol}_M5.csv",
        'M15': data_dir / f"{symbol}_M15.csv",
        'H1': data_dir / f"{symbol}_H1.csv",
        'H4': data_dir / f"{symbol}_H4.csv",
        'D1': data_dir / f"{symbol}_D1.csv",
    }
    
    # Check which files exist
    available_tfs = []
    for tf, path in csv_files.items():
        if path.exists():
            available_tfs.append(tf)
    
    if 'M5' not in available_tfs:
        print(f"⚠️  Skipping {symbol}: M5 file not found")
        return None
    
    print(f"✅ Available timeframes: {available_tfs}")
    
    # Load base (M5)
    print("📊 Loading M5 base...")
    base_df = load_csv_direct(csv_files['M5'], 'M5')
    
    # Filter to 2020-2023
    base_df = base_df[(base_df.index >= '2020-01-01') & (base_df.index <= '2023-12-31')]
    print(f"   {len(base_df):,} bars")
    
    # Merge higher timeframes
    higher_tfs = [tf for tf in ['M15', 'M30', 'H1', 'H4', 'D1'] if tf in available_tfs]
    
    if higher_tfs:
        print(f"🔗 Merging {len(higher_tfs)} higher timeframes...")
        for tf in tqdm(higher_tfs, desc="Merging", ncols=60):
            tf_df = load_csv_direct(csv_files[tf], tf)
            tf_df = tf_df[(tf_df.index >= '2019-01-01') & (tf_df.index <= '2023-12-31')]
            
            base_df = pd.merge_asof(
                left=base_df.reset_index().sort_values('timestamp'),
                right=tf_df.reset_index().sort_values('timestamp'),
                on='timestamp',
                direction='backward'
            ).set_index('timestamp').sort_index()
    
    # Compute features
    all_tfs = ['M5'] + higher_tfs
    print(f"⚙️  Computing features for {len(all_tfs)} timeframes...")
    for tf in tqdm(all_tfs, desc="Features", ncols=60):
        base_df = compute_features(base_df, tf)
    
    # Create target
    base_df['future_return'] = base_df['M5_close'].shift(-10) / base_df['M5_close'] - 1
    base_df['target'] = (base_df['future_return'] > 0).astype(int)
    
    # Drop NaN
    base_df = base_df.dropna()
    
    print(f"✅ Final: {len(base_df):,} rows, {len(base_df.columns)} features")
    print(f"   Target: {base_df['target'].value_counts().to_dict()}")
    
    return base_df


def main():
    print("="*60)
    print("MTF Training Data - All Symbols (M5 Base)")
    print("="*60)
    
    data_dir = Path("data/raw/forex_kaggle_multiTF")
    
    # Find all M5 files
    m5_files = list(data_dir.glob("*_M5.csv"))
    symbols = [f.stem.replace('_M5', '') for f in m5_files]
    
    print(f"\n📁 Found {len(symbols)} symbols with M5 data:")
    print(f"   {', '.join(symbols)}")
    
    # Process each symbol
    results = {}
    for symbol in symbols:
        try:
            df = process_symbol(symbol, data_dir)
            if df is not None:
                results[symbol] = df
        except Exception as e:
            print(f"❌ Error processing {symbol}: {e}")
    
    # Save results
    output_dir = Path("data/training")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("💾 Saving results...")
    print(f"{'='*60}")
    
    for symbol, df in results.items():
        output_file = output_dir / f"{symbol.lower()}_m5_mtf_features.csv"
        df.reset_index().to_csv(output_file, index=False)
        print(f"✅ {symbol}: {output_file} ({len(df):,} rows)")
    
    print(f"\n{'='*60}")
    print(f"✅ SUCCESS! Processed {len(results)} symbols")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
