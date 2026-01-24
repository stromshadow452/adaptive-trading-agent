"""
SCOPUS Self-Learning Trainer - Performance Analyzer

This module analyzes completed trades and labels them for machine learning.
It acts like a professional trader reviewing their journal, identifying
which trades were "good" (to repeat) and which were "mistakes" (to avoid).

Author: SCOPUS Team
Date: 2025-11-24
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """
    Analyzes trade performance and labels trades for self-learning.
    
    Responsibilities:
    1. Load trades from CSV
    2. Enrich with market context/features
    3. Label good vs bad trades
    4. Detect error types (fight_trend, overtrading, etc.)
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize PerformanceAnalyzer.
        
        Args:
            config: Configuration dict with thresholds:
                - good_trade_min_r: Minimum R-multiple for good trade (default: 1.0)
                - bad_trade_max_r: Maximum R-multiple for bad trade (default: 0.5)
                - overtrading_threshold: Max trades per 24h (default: 5)
                - late_entry_pct: % of move for late entry (default: 0.8)
        """
        self.config = config or {}
        
        # Thresholds
        self.good_trade_min_r = self.config.get('good_trade_min_r', 1.0)
        self.bad_trade_max_r = self.config.get('bad_trade_max_r', 0.5)
        self.overtrading_threshold = self.config.get('overtrading_threshold', 5)
        self.late_entry_pct = self.config.get('late_entry_pct', 0.8)
        
        logger.info(f"PerformanceAnalyzer initialized with config: {self.config}")
    
    def load_trades(self, trades_path: str) -> pd.DataFrame:
        """
        Load trades from CSV file.
        
        Args:
            trades_path: Path to trades.csv
            
        Returns:
            DataFrame with trade data
            
        Raises:
            FileNotFoundError: If trades file doesn't exist
            ValueError: If required columns are missing
        """
        trades_path = Path(trades_path)
        
        if not trades_path.exists():
            raise FileNotFoundError(f"Trades file not found: {trades_path}")
        
        logger.info(f"Loading trades from {trades_path}")
        df = pd.read_csv(trades_path)
        
        # Validate required columns
        required_cols = [
            'timestamp_entry', 'timestamp_exit', 'symbol', 'side',
            'entry_price', 'exit_price', 'sl_price', 'tp_price',
            'pnl', 'r_multiple', 'exit_reason'
        ]
        
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Parse timestamps
        df['timestamp_entry'] = pd.to_datetime(df['timestamp_entry'])
        df['timestamp_exit'] = pd.to_datetime(df['timestamp_exit'])
        
        # Sort by entry time
        df = df.sort_values('timestamp_entry').reset_index(drop=True)
        
        logger.info(f"Loaded {len(df)} trades from {df['timestamp_entry'].min()} to {df['timestamp_exit'].max()}")
        
        return df
    
    def enrich_with_context(
        self, 
        trades_df: pd.DataFrame,
        features_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Enrich trades with market context and features.
        
        Args:
            trades_df: DataFrame with trades
            features_df: Optional DataFrame with market features indexed by timestamp
                        If None, will compute basic context from trades
            
        Returns:
            Enhanced DataFrame with context columns
        """
        logger.info("Enriching trades with context...")
        
        df = trades_df.copy()
        
        # Basic time-based features
        df['hour_of_day'] = df['timestamp_entry'].dt.hour
        df['day_of_week'] = df['timestamp_entry'].dt.dayofweek  # 0=Monday
        df['trade_duration_hours'] = (
            df['timestamp_exit'] - df['timestamp_entry']
        ).dt.total_seconds() / 3600
        
        # Recent performance context (rolling window)
        df['recent_winrate_10'] = (
            (df['pnl'] > 0).rolling(window=10, min_periods=1).mean()
        )
        df['consecutive_losses'] = self._compute_consecutive_losses(df)
        df['bars_since_last_trade'] = self._compute_bars_since_last(df)
        
        # If features_df provided, merge market features at entry time
        if features_df is not None:
            df = self._merge_market_features(df, features_df)
        else:
            logger.warning("No features_df provided, using basic context only")
            # Add placeholder columns
            for col in ['regime', 'volatility', 'rsi', 'bb_width', 'atr_pct']:
                if col not in df.columns:
                    df[col] = np.nan
        
        logger.info(f"Enrichment complete. Added context columns.")
        
        return df
    
    def label_good_vs_bad(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """
        Label each trade as good (1) or bad (0).
        
        Criteria for GOOD trade:
        - pnl > 0 AND r_multiple >= good_trade_min_r
        - OR: pnl > 0 AND exit_reason == "TP_HIT"
        
        Criteria for BAD trade:
        - pnl < 0 OR r_multiple < bad_trade_max_r
        
        Args:
            trades_df: DataFrame with trades
            
        Returns:
            DataFrame with 'good_trade' column (1/0)
        """
        logger.info("Labeling good vs bad trades...")
        
        df = trades_df.copy()
        
        # Initialize
        df['good_trade'] = 0
        
        # Good trade conditions
        condition_good = (
            (df['pnl'] > 0) & (df['r_multiple'] >= self.good_trade_min_r)
        ) | (
            (df['pnl'] > 0) & (df['exit_reason'] == 'TP_HIT')
        )
        
        df.loc[condition_good, 'good_trade'] = 1
        
        # Log statistics
        n_good = (df['good_trade'] == 1).sum()
        n_bad = (df['good_trade'] == 0).sum()
        pct_good = 100 * n_good / len(df) if len(df) > 0 else 0
        
        logger.info(f"Labeled {n_good} good trades ({pct_good:.1f}%) and {n_bad} bad trades")
        
        return df
    
    def detect_error_types(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect and categorize trade errors/mistakes.
        
        Error types:
        - fight_trend: Entry against major trend
        - overtrading: Too many trades in short period
        - late_entry: Entry after most of move is done
        - chop_breakout: Breakout in low volatility
        - bad_rr: Risk/reward ratio too low
        - none: No obvious error
        
        Args:
            trades_df: DataFrame with trades (must have context columns)
            
        Returns:
            DataFrame with 'error_type' column
        """
        logger.info("Detecting error types...")
        
        df = trades_df.copy()
        
        # Initialize
        df['error_type'] = 'none'
        
        # 1. Fight Trend (requires trend indicator)
        if 'trend_direction' in df.columns:
            fight_trend = (
                ((df['side'] == 'BUY') & (df['trend_direction'] == 'DOWN')) |
                ((df['side'] == 'SELL') & (df['trend_direction'] == 'UP'))
            ) & (df['good_trade'] == 0)
            df.loc[fight_trend, 'error_type'] = 'fight_trend'
        
        # 2. Overtrading (count trades in last 24 hours for each trade)
        df['trades_last_24h'] = 0
        for idx in df.index:
            current_time = df.loc[idx, 'timestamp_entry']
            symbol = df.loc[idx, 'symbol']
            # Count trades in same symbol within 24 hours before this trade
            mask = (
                (df['symbol'] == symbol) &
                (df['timestamp_entry'] < current_time) &
                (df['timestamp_entry'] >= current_time - pd.Timedelta(hours=24))
            )
            df.loc[idx, 'trades_last_24h'] = mask.sum() + 1  # +1 for current trade
        
        overtrading = (
            (df['trades_last_24h'] > self.overtrading_threshold) &
            (df['good_trade'] == 0)
        )
        df.loc[overtrading, 'error_type'] = 'overtrading'
        
        # 3. Bad R:R
        bad_rr = (
            (df['r_multiple'] < 1.0) &
            (df['good_trade'] == 0)
        )
        df.loc[bad_rr, 'error_type'] = 'bad_rr'
        
        # 4. Chop Breakout (low volatility breakout)
        if 'atr_pct' in df.columns:
            chop_breakout = (
                (df['atr_pct'] < 0.005) &  # Very low volatility
                (df['good_trade'] == 0)
            )
            df.loc[chop_breakout, 'error_type'] = 'chop_breakout'
        
        # Log error distribution
        error_counts = df['error_type'].value_counts()
        logger.info(f"Error type distribution:\n{error_counts}")
        
        return df
    
    def analyze_trades(
        self,
        trades_path: str,
        features_df: Optional[pd.DataFrame] = None,
        output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Complete analysis pipeline: load, enrich, label, detect errors.
        
        Args:
            trades_path: Path to trades.csv
            features_df: Optional market features DataFrame
            output_path: Optional path to save labeled dataset
            
        Returns:
            Fully labeled and enriched DataFrame
        """
        logger.info("=" * 80)
        logger.info("Starting Performance Analysis")
        logger.info("=" * 80)
        
        # Load trades
        df = self.load_trades(trades_path)
        
        # Enrich with context
        df = self.enrich_with_context(df, features_df)
        
        # Label good vs bad
        df = self.label_good_vs_bad(df)
        
        # Detect error types
        df = self.detect_error_types(df)
        
        # Save if requested
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            logger.info(f"Saved labeled dataset to {output_path}")
        
        # Summary statistics
        self._print_summary(df)
        
        logger.info("=" * 80)
        logger.info("Performance Analysis Complete")
        logger.info("=" * 80)
        
        return df
    
    # ==================== Helper Methods ====================
    
    def _compute_consecutive_losses(self, df: pd.DataFrame) -> pd.Series:
        """Compute consecutive losses at each trade."""
        losses = (df['pnl'] < 0).astype(int)
        consecutive = losses * (losses.groupby((losses != losses.shift()).cumsum()).cumcount() + 1)
        return consecutive
    
    def _compute_bars_since_last(self, df: pd.DataFrame) -> pd.Series:
        """Compute bars (trades) since last trade."""
        return df.groupby('symbol').cumcount()
    
    def _merge_market_features(
        self,
        trades_df: pd.DataFrame,
        features_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge market features at entry timestamp.
        
        Uses asof merge to get features closest to entry time.
        """
        # Ensure features_df has timestamp index
        if 'timestamp' in features_df.columns:
            features_df = features_df.set_index('timestamp')
        
        # Merge on entry timestamp
        merged = pd.merge_asof(
            trades_df.sort_values('timestamp_entry'),
            features_df.sort_index(),
            left_on='timestamp_entry',
            right_index=True,
            direction='backward',
            suffixes=('', '_market')
        )
        
        return merged
    
    def _print_summary(self, df: pd.DataFrame):
        """Print analysis summary."""
        total = len(df)
        good = (df['good_trade'] == 1).sum()
        bad = (df['good_trade'] == 0).sum()
        
        avg_pnl_good = df[df['good_trade'] == 1]['pnl'].mean()
        avg_pnl_bad = df[df['good_trade'] == 0]['pnl'].mean()
        
        avg_r_good = df[df['good_trade'] == 1]['r_multiple'].mean()
        avg_r_bad = df[df['good_trade'] == 0]['r_multiple'].mean()
        
        logger.info("\n" + "=" * 60)
        logger.info("ANALYSIS SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total Trades:        {total}")
        logger.info(f"Good Trades:         {good} ({100*good/total:.1f}%)")
        logger.info(f"Bad Trades:          {bad} ({100*bad/total:.1f}%)")
        logger.info(f"")
        logger.info(f"Avg P&L (Good):      ${avg_pnl_good:.2f}")
        logger.info(f"Avg P&L (Bad):       ${avg_pnl_bad:.2f}")
        logger.info(f"Avg R-Multiple (Good): {avg_r_good:.2f}R")
        logger.info(f"Avg R-Multiple (Bad):  {avg_r_bad:.2f}R")
        logger.info("=" * 60)


# ==================== Standalone Usage ====================

if __name__ == "__main__":
    import argparse
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description="Analyze trade performance")
    parser.add_argument('--trades', required=True, help="Path to trades.csv")
    parser.add_argument('--features', help="Optional path to market features CSV")
    parser.add_argument('--output', help="Output path for labeled dataset")
    parser.add_argument('--good-r', type=float, default=1.0, help="Min R for good trade")
    parser.add_argument('--bad-r', type=float, default=0.5, help="Max R for bad trade")
    
    args = parser.parse_args()
    
    # Load features if provided
    features_df = None
    if args.features:
        features_df = pd.read_csv(args.features, parse_dates=['timestamp'])
    
    # Create analyzer
    analyzer = PerformanceAnalyzer(config={
        'good_trade_min_r': args.good_r,
        'bad_trade_max_r': args.bad_r
    })
    
    # Run analysis
    labeled_df = analyzer.analyze_trades(
        trades_path=args.trades,
        features_df=features_df,
        output_path=args.output
    )
    
    print(f"\n✓ Analysis complete. Labeled {len(labeled_df)} trades.")
    if args.output:
        print(f"✓ Saved to {args.output}")
