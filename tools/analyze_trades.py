"""
Trade Analysis Tool for SCOPUS Backtest Results

Analyzes trades.csv to identify patterns, issues, and opportunities for improvement.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

def analyze_trades(trades_csv='backtest_results/trades.csv'):
    """Comprehensive trade analysis"""
    
    try:
        df = pd.read_csv(trades_csv)
    except FileNotFoundError:
        print(f"Error: {trades_csv} not found!")
        return None
    
    print("=" * 80)
    print("SCOPUS TRADE ANALYSIS REPORT")
    print("=" * 80)
    
    # Basic stats
    wins = df[df['pnl'] > 0]
    losses = df[df['pnl'] < 0]
    
    print(f"\n📊 OVERVIEW")
    print(f"{'Total Trades:':<30} {len(df)}")
    print(f"{'Wins:':<30} {len(wins)} ({len(wins)/len(df)*100:.1f}%)")
    print(f"{'Losses:':<30} {len(losses)} ({len(losses)/len(df)*100:.1f}%)")
    print(f"{'Total P&L:':<30} ${df['pnl'].sum():.2f}")
    print(f"{'Avg P&L per trade:':<30} ${df['pnl'].mean():.2f}")
    
    # Side breakdown
    print(f"\n📈 TRADE SIDE DISTRIBUTION")
    for side in df['side'].unique():
        side_df = df[df['side'] == side]
        side_wins = side_df[side_df['pnl'] > 0]
        print(f"  {side:<10} {len(side_df):>3} trades ({len(side_df)/len(df)*100:>5.1f}%) | "
              f"Win rate: {len(side_wins)/len(side_df)*100:>5.1f}% | "
              f"Avg P&L: ${side_df['pnl'].mean():>7.2f}")
    
    # Exit reason breakdown
    print(f"\n🚪 EXIT REASONS")
    for reason in df['exit_reason'].unique():
        reason_df = df[df['exit_reason'] == reason]
        reason_wins = reason_df[reason_df['pnl'] > 0]
        print(f"  {reason:<15} {len(reason_df):>3} trades ({len(reason_df)/len(df)*100:>5.1f}%) | "
              f"Win rate: {len(reason_wins)/len(reason_df)*100:>5.1f}% | "
              f"Avg P&L: ${reason_df['pnl'].mean():>7.2f}")
    
    # R-Multiple analysis
    print(f"\n💰 R-MULTIPLE ANALYSIS")
    print(f"  {'Avg R-Multiple (all):':<30} {df['r_multiple'].mean():>7.2f}")
    print(f"  {'Avg R-Multiple (wins):':<30} {wins['r_multiple'].mean():>7.2f}")
    print(f"  {'Avg R-Multiple (losses):':<30} {losses['r_multiple'].mean():>7.2f}")
    print(f"  {'Expectancy:':<30} {df['r_multiple'].mean():>7.2f}")
    
    # MAE/MFE analysis
    print(f"\n📉 MAE/MFE ANALYSIS (Max Adverse/Favorable Excursion)")
    print(f"  {'Avg MAE (all):':<30} {df['mae'].mean():.5f} ({df['mae'].mean()/df['entry_price'].mean()*10000:.1f} pips)")
    print(f"  {'Avg MAE (losses):':<30} {losses['mae'].mean():.5f} ({losses['mae'].mean()/losses['entry_price'].mean()*10000:.1f} pips)")
    print(f"  {'Avg MFE (all):':<30} {df['mfe'].mean():.5f} ({df['mfe'].mean()/df['entry_price'].mean()*10000:.1f} pips)")
    print(f"  {'Avg MFE (wins):':<30} {wins['mfe'].mean():.5f} ({wins['mfe'].mean()/wins['entry_price'].mean()*10000:.1f} pips)")
    
    # Duration analysis
    print(f"\n⏱️  DURATION ANALYSIS")
    print(f"  {'Avg Duration (all):':<30} {df['duration_minutes'].mean():>7.1f} minutes")
    print(f"  {'Avg Duration (wins):':<30} {wins['duration_minutes'].mean():>7.1f} minutes")
    print(f"  {'Avg Duration (losses):':<30} {losses['duration_minutes'].mean():>7.1f} minutes")
    print(f"  {'Min Duration:':<30} {df['duration_minutes'].min():>7.0f} minutes")
    print(f"  {'Max Duration:':<30} {df['duration_minutes'].max():>7.0f} minutes")
    
    # Regime breakdown
    print(f"\n🌍 REGIME BREAKDOWN")
    for regime in df['regime'].unique():
        regime_df = df[df['regime'] == regime]
        regime_wins = regime_df[regime_df['pnl'] > 0]
        print(f"  {regime:<15} {len(regime_df):>3} trades ({len(regime_df)/len(df)*100:>5.1f}%) | "
              f"Win rate: {len(regime_wins)/len(regime_df)*100:>5.1f}% | "
              f"Avg P&L: ${regime_df['pnl'].mean():>7.2f}")
    
    # Decision source
    print(f"\n🧠 DECISION SOURCE")
    for source in df['decision_source'].unique():
        source_df = df[df['decision_source'] == source]
        source_wins = source_df[source_df['pnl'] > 0]
        print(f"  {source:<15} {len(source_df):>3} trades ({len(source_df)/len(df)*100:>5.1f}%) | "
              f"Win rate: {len(source_wins)/len(source_df)*100:>5.1f}% | "
              f"Avg P&L: ${source_df['pnl'].mean():>7.2f}")
    
    # Key insights
    print(f"\n💡 KEY INSIGHTS")
    
    # Check if SL is too tight
    sl_hit_pct = len(df[df['exit_reason'] == 'SL_HIT']) / len(df) * 100
    if sl_hit_pct > 60:
        print(f"  ⚠️  HIGH SL HIT RATE: {sl_hit_pct:.1f}% of trades hit SL")
        print(f"     → Consider widening stop loss or improving entry timing")
    
    # Check if all one regime
    if len(df['regime'].unique()) == 1:
        print(f"  ⚠️  ALL TRADES IN {df['regime'].iloc[0]} REGIME")
        print(f"     → Check regime detection logic or test in different market conditions")
    
    # Check if model is biased
    if len(df['side'].unique()) > 1:
        side_counts = df['side'].value_counts()
        max_side = side_counts.idxmax()
        max_pct = side_counts.max() / len(df) * 100
        if max_pct > 70:
            print(f"  ⚠️  DIRECTIONAL BIAS: {max_pct:.1f}% of trades are {max_side}")
            print(f"     → Model may be biased, check training data balance")
    
    # Check win rate
    win_rate = len(wins) / len(df) * 100
    if win_rate < 40:
        print(f"  ⚠️  LOW WIN RATE: {win_rate:.1f}%")
        print(f"     → Model needs retraining or better features")
    elif win_rate > 55:
        print(f"  ✅ GOOD WIN RATE: {win_rate:.1f}%")
    
    # Check expectancy
    expectancy = df['r_multiple'].mean()
    if expectancy < 0:
        print(f"  ⚠️  NEGATIVE EXPECTANCY: {expectancy:.2f}R")
        print(f"     → System is losing money on average per trade")
    elif expectancy > 0.5:
        print(f"  ✅ POSITIVE EXPECTANCY: {expectancy:.2f}R")
    
    print("\n" + "=" * 80)
    
    return df


if __name__ == "__main__":
    if len(sys.argv) > 1:
        trades_csv = sys.argv[1]
    else:
        trades_csv = 'backtest_results/trades.csv'
    
    analyze_trades(trades_csv)
