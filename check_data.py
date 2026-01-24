import pandas as pd

# Check EURUSD data
df = pd.read_csv('temp_prices/EURUSD_M15.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"Total rows: {len(df)}")
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nLast 5 rows:")
print(df.tail())
