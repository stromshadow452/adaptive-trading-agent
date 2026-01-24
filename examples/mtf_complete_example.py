"""
MTF Data Architecture - Complete Usage Example

Demonstrates the full MTF system with dual sources, normalization, and context building.
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


def main():
    logger.info("=" * 60)
    logger.info("MTF Data Architecture - Complete Example")
    logger.info("=" * 60)
    
    # 1. Initialize store with dual sources
    logger.info("\n1. Initializing MarketDataStore with dual sources...")
    store = MarketDataStore(data_roots=[
        Path("data/raw/forex_kaggle_multiTF"),      # Kaggle (2001-2023)
        Path("data/raw/forex_backup_2020_2025"),    # MT5 (2020-2025)
    ])
    
    logger.info(f"   Registry: {len(store.registry.files)} files indexed")
    logger.info(f"   Symbols: {[s.value for s in store.registry.get_available_symbols()]}")
    
    # 2. Load single timeframe (basic usage)
    logger.info("\n2. Loading single timeframe (EURUSD M15)...")
    df_m15 = store.load_ohlcv(
        Symbol.EURUSD,
        Timeframe.M15,
        start=datetime(2024, 1, 1),
        end=datetime(2024, 1, 31),
    )
    
    logger.info(f"   Loaded: {len(df_m15)} bars")
    logger.info(f"   Columns: {df_m15.columns.tolist()}")
    logger.info(f"   Sources: {df_m15['source'].value_counts().to_dict()}")
    
    # 3. Build MTF context
    logger.info("\n3. Building MTF context (M5 base with M15, H1, H4, D1)...")
    mtf_df = build_mtf_context(
        store=store,
        symbol=Symbol.EURUSD,
        base_timeframe=Timeframe.M5,
        higher_timeframes=[Timeframe.M15, Timeframe.H1, Timeframe.H4, Timeframe.D1],
        start=datetime(2024, 1, 1),
        end=datetime(2024, 1, 7),  # 1 week for demo
    )
    
    logger.info(f"   MTF rows: {len(mtf_df)}")
    logger.info(f"   MTF columns: {len(mtf_df.columns)}")
    logger.info(f"   Sample columns: {mtf_df.columns.tolist()[:10]}...")
    
    # 4. Add features
    logger.info("\n4. Adding MTF features (RSI, ATR, Trend)...")
    mtf_df = add_mtf_features(
        df=mtf_df,
        base_timeframe=Timeframe.M5,
        higher_timeframes=[Timeframe.M15, Timeframe.H1, Timeframe.H4, Timeframe.D1],
    )
    
    logger.info(f"   Total columns after features: {len(mtf_df.columns)}")
    
    # Show feature columns
    feature_cols = [col for col in mtf_df.columns if any(x in col for x in ['rsi', 'atr', 'trend'])]
    logger.info(f"   Feature columns: {feature_cols}")
    
    # 5. Example decision logic
    logger.info("\n5. Example decision logic (top-down analysis)...")
    
    # Get last row
    last_row = mtf_df.iloc[-1]
    
    d1_trend = last_row.get('D1_trend_flag', 0)
    h4_trend = last_row.get('H4_trend_flag', 0)
    h1_rsi = last_row.get('H1_rsi_14', 50)
    m5_rsi = last_row.get('M5_rsi_14', 50)
    
    logger.info(f"   D1 trend: {d1_trend} (1=up, -1=down, 0=neutral)")
    logger.info(f"   H4 trend: {h4_trend}")
    logger.info(f"   H1 RSI: {h1_rsi:.1f}")
    logger.info(f"   M5 RSI: {m5_rsi:.1f}")
    
    # Decision
    if d1_trend == 1 and h4_trend == 1:
        bias = "LONG_ONLY"
    elif d1_trend == -1 and h4_trend == -1:
        bias = "SHORT_ONLY"
    else:
        bias = "NEUTRAL"
    
    logger.info(f"   → Bias: {bias}")
    
    if bias == "LONG_ONLY" and m5_rsi < 30:
        signal = "BUY (oversold in uptrend)"
    elif bias == "SHORT_ONLY" and m5_rsi > 70:
        signal = "SELL (overbought in downtrend)"
    else:
        signal = "WAIT"
    
    logger.info(f"   → Signal: {signal}")
    
    # 6. Save sample output
    logger.info("\n6. Saving sample output...")
    output_file = Path("data/mtf_example_output.csv")
    mtf_df.head(100).to_csv(output_file)
    logger.info(f"   Saved to: {output_file}")
    
    logger.info("\n" + "=" * 60)
    logger.info("MTF Architecture Demo Complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
