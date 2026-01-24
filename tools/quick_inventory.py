"""Quick data inventory using UnifiedDataLoader."""
import sys
sys.path.insert(0, '.')
from src.market_data.unified_loader import UnifiedDataLoader

loader = UnifiedDataLoader()
symbols = loader.get_available_symbols()

print("=" * 70)
print(" COMPLETE DATA INVENTORY")
print("=" * 70)
print()

total_candles = 0
for sym in symbols:
    df, audit = loader.load(sym, "M15")
    if audit.total_candles > 0:
        start_year = audit.earliest_date.strftime("%Y")
        end_year = audit.latest_date.strftime("%Y")
        print(f"{sym}: {start_year} - {end_year} ({audit.total_years:.1f}y) | {audit.total_candles:>10,} candles")
        total_candles += audit.total_candles

print()
print("-" * 70)
print(f"TOTAL: {len(symbols)} symbols | {total_candles:,} M15 candles")
print("=" * 70)
