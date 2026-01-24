"""
ML Brain Integration Test

Verifies the complete flow:
Data -> Store -> MTF Context -> Features -> Model -> Signal
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import logging
from datetime import datetime

from src.market_data import MarketDataStore, Symbol, Timeframe
from src.backtest.feature_reactor_v1 import SafeFeatureReactor
from src.backtest.ml_brain_v1 import MLBrainV1

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("ML Brain Integration Test")
    logger.info("=" * 60)
    
    # 1. Initialize Store
    logger.info("\n1. Initializing MarketDataStore...")
    store = MarketDataStore(data_roots=[
        Path("data/raw/forex_kaggle_multiTF"),
    ])
    
    # 2. Initialize Feature Reactor
    logger.info("\n2. Initializing SafeFeatureReactor...")
    reactor = SafeFeatureReactor(store=store)
    
    # 3. Initialize ML Brain
    logger.info("\n3. Initializing MLBrainV1...")
    model_path = Path("models/xgb_primary_mtf.joblib")
    feature_names_path = Path("models/feature_names_mtf.txt")
    
    if not model_path.exists() or not feature_names_path.exists():
        logger.error("Model or feature names not found! Run training first.")
        return
        
    brain = MLBrainV1(
        model_path=model_path,
        feature_names_path=feature_names_path,
        feature_reactor=reactor,
        config={'buy_threshold': 0.6, 'sell_threshold': 0.6}
    )
    
    # 4. Load Test Data (M5)
    logger.info("\n4. Loading Test Data (M5)...")
    # Load a small chunk of M5 data to simulate live feed
    m5_data = store.load_ohlcv(
        Symbol.EURUSD,
        Timeframe.M5,
        start=datetime(2023, 1, 1),
        end=datetime(2023, 1, 2)
    )
    
    if m5_data.empty:
        logger.error("No test data found!")
        return
        
    logger.info(f"Loaded {len(m5_data)} candles")
    
    # 5. Run Prediction Loop
    logger.info("\n5. Running Prediction Loop (First 5 candles)...")
    
    context = {'symbol': Symbol.EURUSD}
    regime = 'TREND' # Mock regime
    
    for i in range(5):
        candle = m5_data.iloc[i]
        timestamp = candle.name
        
        logger.info(f"\nCandle: {timestamp} | Close: {candle['close']}")
        
        # Predict
        signal, confidence = brain.predict(candle, context, regime)
        
        logger.info(f"Signal: {signal} | Confidence: {confidence:.4f}")
        
        if signal != 'HOLD':
            logger.info(f"🔥 TRADE SIGNAL: {signal}!")
            
    logger.info("\n" + "=" * 60)
    logger.info("Integration Test Complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
