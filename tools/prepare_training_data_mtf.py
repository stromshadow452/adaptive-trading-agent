"""
Prepare MTF training data for MVP v1

Creates multi-timeframe features for model training.
Uses build_mtf_context() for proper alignment.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.market_data import (
    MarketDataStore,
    Symbol,
    Timeframe,
    build_mtf_context,
    add_mtf_features
)
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_target(df, forward_bars=10):
    """Create target variable (future return)"""
    # Use base timeframe close
    close_col = 'M15_close'  # Assuming M15 base
    
    if close_col not in df.columns:
        logger.error(f"Column {close_col} not found")
        return df
    
    # Future return
    df['future_return'] = df[close_col].shift(-forward_bars) / df[close_col] - 1
    
    # Binary target: 1 if future return > 0, else 0
    df['target'] = (df['future_return'] > 0).astype(int)
    
    return df


def main():
    logger.info("Preparing MTF training data...")
    
    # 1. Initialize store
    logger.info("Initializing MarketDataStore...")
    store = MarketDataStore(data_roots=[
        Path("data/raw/forex_kaggle_multiTF"),
    ])
    
    # 2. Build MTF context
    logger.info("Building MTF context (M15 base with M30, H1, H4, D1)...")
    mtf_df = build_mtf_context(
        store=store,
        symbol=Symbol.EURUSD,
        base_timeframe=Timeframe.M15,
        higher_timeframes=[Timeframe.M30, Timeframe.H1, Timeframe.H4, Timeframe.D1],
        start=datetime(2020, 1, 1),
        end=datetime(2023, 12, 31),
    )
    
    if mtf_df.empty:
        logger.error("No MTF data loaded!")
        return
    
    logger.info(f"MTF context: {len(mtf_df)} rows, {len(mtf_df.columns)} columns")
    
    # 3. Add features
    logger.info("Adding MTF features...")
    mtf_df = add_mtf_features(
        df=mtf_df,
        base_timeframe=Timeframe.M15,
        higher_timeframes=[Timeframe.M30, Timeframe.H1, Timeframe.H4, Timeframe.D1],
    )
    
    logger.info(f"After features: {len(mtf_df.columns)} columns")
    
    # 4. Create target
    logger.info("Creating target variable...")
    mtf_df = create_target(mtf_df, forward_bars=10)
    
    # 5. Drop NaN rows
    mtf_df = mtf_df.dropna()
    
    logger.info(f"Final dataset: {len(mtf_df)} rows")
    
    # 6. Save
    output_dir = Path("data/training")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "eurusd_m15_mtf_features.csv"
    
    # Reset index to save timestamp
    mtf_df_save = mtf_df.reset_index()
    mtf_df_save.to_csv(output_file, index=False)
    
    logger.info(f"Saved to {output_file}")
    logger.info(f"Target distribution: {mtf_df['target'].value_counts().to_dict()}")
    logger.info(f"Columns: {mtf_df.columns.tolist()}")


if __name__ == "__main__":
    main()
