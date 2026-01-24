"""
Historical Data Coverage Audit
===============================
Scans all CSV files to determine total years of market data available.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def audit_data_coverage():
    """Audit all CSV files for date coverage."""
    
    data_dirs = [
        Path("data/raw/forex_kaggle_multiTF"),
        Path("data/raw/forex_backup_2020_2025"),
    ]
    
    results = {}
    total_candles = 0
    all_dates = []
    timeframe_candles = {}
    symbol_coverage = {}
    
    for data_dir in data_dirs:
        if not data_dir.exists():
            print(f"Directory not found: {data_dir}")
            continue
        
        csv_files = list(data_dir.glob("*.csv"))
        print(f"Scanning {data_dir.name}: {len(csv_files)} files")
        
        for f in csv_files:
            try:
                # Read sample to get column names
                sample = pd.read_csv(f, nrows=2)
                date_col = sample.columns[0]
                
                # Read full file for date range
                df = pd.read_csv(f, usecols=[date_col])
                row_count = len(df)
                total_candles += row_count
                
                # Parse dates
                df[date_col] = pd.to_datetime(df[date_col])
                first_date = df[date_col].min()
                last_date = df[date_col].max()
                
                all_dates.extend([first_date, last_date])
                
                # Track by timeframe
                name = f.stem
                for tf in ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "Daily"]:
                    if tf in name:
                        tf_key = "D1" if tf == "Daily" else tf
                        timeframe_candles[tf_key] = timeframe_candles.get(tf_key, 0) + row_count
                        break
                
                # Track by symbol
                symbol = name.split("_")[0]
                if symbol not in symbol_coverage:
                    symbol_coverage[symbol] = {"first": first_date, "last": last_date, "candles": 0}
                else:
                    if first_date < symbol_coverage[symbol]["first"]:
                        symbol_coverage[symbol]["first"] = first_date
                    if last_date > symbol_coverage[symbol]["last"]:
                        symbol_coverage[symbol]["last"] = last_date
                symbol_coverage[symbol]["candles"] += row_count
                
                results[f.name] = {
                    "first": first_date,
                    "last": last_date,
                    "rows": row_count
                }
            except Exception as e:
                print(f"  Error reading {f.name}: {e}")
    
    # Calculate summary
    if not all_dates:
        print("No data found!")
        return
    
    earliest = min(all_dates)
    latest = max(all_dates)
    days = (latest - earliest).days
    years = days / 365.25
    
    # Print report
    print()
    print("=" * 70)
    print(" HISTORICAL DATA COVERAGE AUDIT")
    print("=" * 70)
    print()
    print("  SUMMARY")
    print("-" * 70)
    fmt_e = earliest.strftime("%Y-%m-%d %H:%M")
    fmt_l = latest.strftime("%Y-%m-%d %H:%M")
    print(f"  Earliest date:    {fmt_e}")
    print(f"  Latest date:      {fmt_l}")
    print(f"  Total span:       {days:,} days (~{years:.2f} years)")
    print(f"  Total candles:    {total_candles:,}")
    print(f"  Total files:      {len(results)}")
    print()
    
    print("  TIMEFRAME BREAKDOWN")
    print("-" * 70)
    for tf in ["M5", "M15", "M30", "H1", "H4", "D1"]:
        count = timeframe_candles.get(tf, 0)
        if count > 0:
            print(f"  {tf:6}: {count:>12,} candles")
    print()
    
    print("  SYMBOL COVERAGE")
    print("-" * 70)
    for symbol, info in sorted(symbol_coverage.items()):
        span_days = (info["last"] - info["first"]).days
        span_years = span_days / 365.25
        first_str = info["first"].strftime("%Y-%m-%d")
        last_str = info["last"].strftime("%Y-%m-%d")
        print(f"  {symbol:8}: {first_str} to {last_str} ({span_years:.1f}y, {info['candles']:,} candles)")
    print()
    
    print("=" * 70)
    print(" VERDICT")
    print("=" * 70)
    if years >= 5:
        print(f"  ✅ EXCELLENT: {years:.1f} years of data is sufficient for mature risk evolution")
    elif years >= 3:
        print(f"  ✅ GOOD: {years:.1f} years of data is adequate for risk evolution")
    elif years >= 1:
        print(f"  ⚠️ LIMITED: {years:.1f} years may not be enough for full risk evolution")
    else:
        print(f"  ❌ INSUFFICIENT: {years:.1f} years is too short for reliable testing")
    print("=" * 70)


if __name__ == "__main__":
    audit_data_coverage()
