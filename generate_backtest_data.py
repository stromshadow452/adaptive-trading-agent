"""
Generate sample CSV data for backtesting
Creates OHLCV data for multiple symbols
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# Create temp_prices directory
os.makedirs("temp_prices", exist_ok=True)

# Date range for data
start_date = datetime(2024, 9, 1)
end_date = datetime(2025, 8, 31)

# Generate 15-minute intervals
date_range = pd.date_range(start=start_date, end=end_date, freq='15min')

symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD']
base_prices = {
    'EURUSD': 1.1000,
    'GBPUSD': 1.2500,
    'USDJPY': 110.00,
    'AUDUSD': 0.7500,
    'USDCAD': 1.3500
}

for symbol in symbols:
    print(f"Generating {symbol}...")
    
    base_price = base_prices[symbol]
    n = len(date_range)
    
    # Generate realistic price movement
    np.random.seed(42)
    returns = np.random.normal(0, 0.0005, n)  # Small random returns
    trend = np.linspace(0, 0.05, n)  # Slight upward trend
    
    # Generate close prices
    close = base_price * (1 + np.cumsum(returns) + trend)
    
    # Generate OHLC from close
    high = close * (1 + np.abs(np.random.normal(0, 0.0002, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.0002, n)))
    open_price = close + np.random.normal(0, 0.0001, n)
    volume = np.random.randint(100, 10000, n)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': date_range,
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    
    # Save to CSV
    filename = f"temp_prices/{symbol}_M15.csv"
    df.to_csv(filename, index=False)
    print(f"  ✅ Created {filename} ({len(df)} rows)")

print(f"\n✅ Generated CSV files for {len(symbols)} symbols")
print(f"📅 Date range: {start_date.date()} to {end_date.date()}")
print(f"📊 Total bars per symbol: {len(date_range)}")
print(f"\n🚀 Now run the backtesting dashboard!")
