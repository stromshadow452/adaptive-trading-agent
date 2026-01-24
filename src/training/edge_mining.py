"""
SCOPUS Self-Learning Trainer - Edge and Mistake Mining

This module discovers patterns in labeled trades to identify:
1. Edges: High-probability, high-profit setups to favor
2. Mistakes: Common error patterns to avoid

It acts like a professional trader analyzing their journal to find
what works and what doesn't.

Author: SCOPUS Team
Date: 2025-11-24
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class EdgeAndMistakeMiner:
    """
    Mines patterns from labeled trades to discover edges and mistakes.
    
    Responsibilities:
    1. Find high-profit patterns (edges)
    2. Find high-loss patterns (mistakes)
    3. Generate executable rules
    4. Export to JSON libraries
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize EdgeAndMistakeMiner.
        
        Args:
            config: Configuration dict:
                - min_support: Minimum trades for pattern (default: 5)
                - min_edge_winrate: Minimum winrate for edge (default: 0.65)
                - min_edge_avg_r: Minimum avg R for edge (default: 1.5)
                - max_mistake_winrate: Maximum winrate for mistake (default: 0.35)
        """
        self.config = config or {}
        
        self.min_support = self.config.get('min_support', 5)
        self.min_edge_winrate = self.config.get('min_edge_winrate', 0.65)
        self.min_edge_avg_r = self.config.get('min_edge_avg_r', 1.5)
        self.max_mistake_winrate = self.config.get('max_mistake_winrate', 0.35)
        
        logger.info(f"EdgeAndMistakeMiner initialized with config: {self.config}")
    
    def mine_edges(self, labeled_trades: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Mine high-profit edge patterns.
        
        An edge is a pattern where:
        - Winrate >= min_edge_winrate
        - Avg R-multiple >= min_edge_avg_r
        - Support >= min_support
        
        Args:
            labeled_trades: DataFrame with labeled trades
            
        Returns:
            List of edge patterns with stats
        """
        logger.info("Mining edge patterns...")
        
        edges = []
        
        # Define pattern dimensions to explore
        pattern_dims = self._get_pattern_dimensions(labeled_trades)
        
        # Explore each dimension
        for dim_name, dim_values in pattern_dims.items():
            for value in dim_values:
                # Filter trades matching this pattern
                mask = labeled_trades[dim_name] == value
                pattern_trades = labeled_trades[mask]
                
                if len(pattern_trades) < self.min_support:
                    continue
                
                # Compute stats
                stats = self._compute_pattern_stats(pattern_trades)
                
                # Check if this is an edge
                if (stats['winrate'] >= self.min_edge_winrate and
                    stats['avg_r_multiple'] >= self.min_edge_avg_r):
                    
                    edge = {
                        'pattern_type': 'single_dim',
                        'dimension': dim_name,
                        'value': value,
                        'condition': f"{dim_name} == '{value}'",
                        'stats': stats,
                        'description': self._generate_edge_description(dim_name, value, stats)
                    }
                    edges.append(edge)
        
        # Mine multi-dimensional patterns (combinations)
        multi_edges = self._mine_multi_dim_edges(labeled_trades)
        edges.extend(multi_edges)
        
        # Sort by quality score
        edges = sorted(edges, key=lambda x: self._edge_quality_score(x), reverse=True)
        
        logger.info(f"Found {len(edges)} edge patterns")
        
        return edges
    
    def mine_mistakes(self, labeled_trades: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Mine common mistake patterns.
        
        A mistake is a pattern where:
        - Winrate <= max_mistake_winrate
        - Avg P&L < 0
        - Support >= min_support
        
        Args:
            labeled_trades: DataFrame with labeled trades
            
        Returns:
            List of mistake patterns with stats
        """
        logger.info("Mining mistake patterns...")
        
        mistakes = []
        
        # Focus on trades with error_type != 'none'
        error_trades = labeled_trades[labeled_trades['error_type'] != 'none']
        
        # Group by error_type
        for error_type in error_trades['error_type'].unique():
            pattern_trades = error_trades[error_trades['error_type'] == error_type]
            
            if len(pattern_trades) < self.min_support:
                continue
            
            stats = self._compute_pattern_stats(pattern_trades)
            
            if stats['winrate'] <= self.max_mistake_winrate:
                mistake = {
                    'pattern_type': 'error_type',
                    'error_type': error_type,
                    'condition': f"error_type == '{error_type}'",
                    'stats': stats,
                    'description': self._generate_mistake_description(error_type, stats)
                }
                mistakes.append(mistake)
        
        # Mine additional multi-dimensional mistake patterns
        multi_mistakes = self._mine_multi_dim_mistakes(labeled_trades)
        mistakes.extend(multi_mistakes)
        
        # Sort by severity (avg loss)
        mistakes = sorted(mistakes, key=lambda x: x['stats']['avg_pnl'])
        
        logger.info(f"Found {len(mistakes)} mistake patterns")
        
        return mistakes
    
    def generate_rules(self, patterns: List[Dict]) -> List[Dict[str, Any]]:
        """
        Convert patterns to executable rules.
        
        Args:
            patterns: List of edge or mistake patterns
            
        Returns:
            List of rules with executable conditions
        """
        rules = []
        
        for pattern in patterns:
            rule = {
                'rule_id': f"rule_{len(rules) + 1}",
                'condition': pattern['condition'],
                'action': 'FAVOR' if 'edge' in str(pattern.get('pattern_type', '')) else 'AVOID',
                'stats': pattern['stats'],
                'description': pattern['description']
            }
            rules.append(rule)
        
        return rules
    
    def export_libraries(
        self,
        edges: List[Dict],
        mistakes: List[Dict],
        output_dir: str
    ):
        """
        Export edge and mistake libraries to JSON.
        
        Args:
            edges: List of edge patterns
            mistakes: List of mistake patterns
            output_dir: Directory to save JSON files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export edges
        edge_path = output_dir / 'edge_library.json'
        with open(edge_path, 'w') as f:
            json.dump({
                'generated_at': pd.Timestamp.now().isoformat(),
                'total_edges': len(edges),
                'edges': edges
            }, f, indent=2, default=str)
        logger.info(f"Exported {len(edges)} edges to {edge_path}")
        
        # Export mistakes
        mistake_path = output_dir / 'mistake_library.json'
        with open(mistake_path, 'w') as f:
            json.dump({
                'generated_at': pd.Timestamp.now().isoformat(),
                'total_mistakes': len(mistakes),
                'mistakes': mistakes
            }, f, indent=2, default=str)
        logger.info(f"Exported {len(mistakes)} mistakes to {mistake_path}")
    
    def analyze_and_export(
        self,
        labeled_trades_path: str,
        output_dir: str
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Complete mining pipeline: load, mine, export.
        
        Args:
            labeled_trades_path: Path to labeled trades CSV
            output_dir: Directory to save libraries
            
        Returns:
            Tuple of (edges, mistakes)
        """
        logger.info("=" * 80)
        logger.info("Starting Edge and Mistake Mining")
        logger.info("=" * 80)
        
        # Load labeled trades
        df = pd.read_csv(labeled_trades_path, parse_dates=['timestamp_entry', 'timestamp_exit'])
        logger.info(f"Loaded {len(df)} labeled trades")
        
        # Mine edges
        edges = self.mine_edges(df)
        
        # Mine mistakes
        mistakes = self.mine_mistakes(df)
        
        # Export
        self.export_libraries(edges, mistakes, output_dir)
        
        # Print summary
        self._print_summary(edges, mistakes)
        
        logger.info("=" * 80)
        logger.info("Mining Complete")
        logger.info("=" * 80)
        
        return edges, mistakes
    
    # ==================== Helper Methods ====================
    
    def _get_pattern_dimensions(self, df: pd.DataFrame) -> Dict[str, List]:
        """Get available pattern dimensions and their values."""
        dimensions = {}
        
        # Categorical dimensions
        categorical_cols = ['regime', 'side', 'exit_reason', 'error_type']
        for col in categorical_cols:
            if col in df.columns:
                dimensions[col] = df[col].unique().tolist()
        
        # Binned numerical dimensions
        if 'hour_of_day' in df.columns:
            dimensions['hour_of_day'] = df['hour_of_day'].unique().tolist()
        
        if 'day_of_week' in df.columns:
            dimensions['day_of_week'] = df['day_of_week'].unique().tolist()
        
        # Volatility bins
        if 'volatility' in df.columns:
            try:
                # Try to create 3 bins
                df['volatility_bin'] = pd.qcut(df['volatility'], q=3, labels=['low', 'medium', 'high'], duplicates='drop')
                dimensions['volatility_bin'] = df['volatility_bin'].unique().tolist()
            except ValueError:
                # If 3 bins fail, try 2 bins
                try:
                    df['volatility_bin'] = pd.qcut(df['volatility'], q=2, labels=['low', 'high'], duplicates='drop')
                    dimensions['volatility_bin'] = df['volatility_bin'].unique().tolist()
                except ValueError:
                    # If still fails, skip volatility binning
                    logger.warning("Insufficient unique volatility values for binning, skipping volatility_bin dimension")
        
        # RSI bins
        if 'rsi' in df.columns:
            try:
                df['rsi_bin'] = pd.cut(df['rsi'], bins=[0, 30, 70, 100], labels=['oversold', 'neutral', 'overbought'])
                dimensions['rsi_bin'] = df['rsi_bin'].unique().tolist()
            except ValueError as e:
                logger.warning(f"Failed to bin RSI values: {e}, skipping rsi_bin dimension")
        
        return dimensions
    
    def _compute_pattern_stats(self, pattern_trades: pd.DataFrame) -> Dict[str, float]:
        """Compute statistics for a pattern."""
        total = len(pattern_trades)
        wins = (pattern_trades['pnl'] > 0).sum()
        
        return {
            'support': total,
            'winrate': wins / total if total > 0 else 0,
            'avg_pnl': pattern_trades['pnl'].mean(),
            'total_pnl': pattern_trades['pnl'].sum(),
            'avg_r_multiple': pattern_trades['r_multiple'].mean(),
            'max_r_multiple': pattern_trades['r_multiple'].max(),
            'min_r_multiple': pattern_trades['r_multiple'].min(),
            'std_pnl': pattern_trades['pnl'].std()
        }
    
    def _mine_multi_dim_edges(self, df: pd.DataFrame) -> List[Dict]:
        """Mine multi-dimensional edge patterns (e.g., regime + time_of_day)."""
        edges = []
        
        # Example: regime + hour_of_day
        if 'regime' in df.columns and 'hour_of_day' in df.columns:
            for regime in df['regime'].unique():
                for hour in df['hour_of_day'].unique():
                    mask = (df['regime'] == regime) & (df['hour_of_day'] == hour)
                    pattern_trades = df[mask]
                    
                    if len(pattern_trades) < self.min_support:
                        continue
                    
                    stats = self._compute_pattern_stats(pattern_trades)
                    
                    if (stats['winrate'] >= self.min_edge_winrate and
                        stats['avg_r_multiple'] >= self.min_edge_avg_r):
                        
                        edge = {
                            'pattern_type': 'multi_dim',
                            'dimensions': {'regime': regime, 'hour_of_day': hour},
                            'condition': f"regime == '{regime}' AND hour_of_day == {hour}",
                            'stats': stats,
                            'description': f"{regime} regime at hour {hour}: {stats['winrate']:.1%} winrate, {stats['avg_r_multiple']:.2f}R avg"
                        }
                        edges.append(edge)
        
        return edges
    
    def _mine_multi_dim_mistakes(self, df: pd.DataFrame) -> List[Dict]:
        """Mine multi-dimensional mistake patterns."""
        mistakes = []
        
        # Example: error_type + regime
        if 'error_type' in df.columns and 'regime' in df.columns:
            for error in df['error_type'].unique():
                if error == 'none':
                    continue
                    
                for regime in df['regime'].unique():
                    mask = (df['error_type'] == error) & (df['regime'] == regime)
                    pattern_trades = df[mask]
                    
                    if len(pattern_trades) < self.min_support:
                        continue
                    
                    stats = self._compute_pattern_stats(pattern_trades)
                    
                    if stats['winrate'] <= self.max_mistake_winrate:
                        mistake = {
                            'pattern_type': 'multi_dim',
                            'dimensions': {'error_type': error, 'regime': regime},
                            'condition': f"error_type == '{error}' AND regime == '{regime}'",
                            'stats': stats,
                            'description': f"{error} in {regime} regime: {stats['winrate']:.1%} winrate, ${stats['avg_pnl']:.2f} avg loss"
                        }
                        mistakes.append(mistake)
        
        return mistakes
    
    def _edge_quality_score(self, edge: Dict) -> float:
        """Compute quality score for ranking edges."""
        stats = edge['stats']
        # Weighted combination of winrate, avg_r, and support
        return (
            stats['winrate'] * 0.4 +
            min(stats['avg_r_multiple'] / 3.0, 1.0) * 0.4 +
            min(stats['support'] / 50.0, 1.0) * 0.2
        )
    
    def _generate_edge_description(self, dim: str, value: Any, stats: Dict) -> str:
        """Generate human-readable edge description."""
        return (
            f"Edge: {dim}={value} | "
            f"Winrate: {stats['winrate']:.1%} | "
            f"Avg R: {stats['avg_r_multiple']:.2f} | "
            f"Support: {stats['support']} trades"
        )
    
    def _generate_mistake_description(self, error_type: str, stats: Dict) -> str:
        """Generate human-readable mistake description."""
        return (
            f"Mistake: {error_type} | "
            f"Winrate: {stats['winrate']:.1%} | "
            f"Avg Loss: ${stats['avg_pnl']:.2f} | "
            f"Support: {stats['support']} trades"
        )
    
    def _print_summary(self, edges: List[Dict], mistakes: List[Dict]):
        """Print mining summary."""
        logger.info("\n" + "=" * 60)
        logger.info("MINING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total Edges Found:    {len(edges)}")
        logger.info(f"Total Mistakes Found: {len(mistakes)}")
        logger.info("")
        
        if edges:
            logger.info("Top 3 Edges:")
            for i, edge in enumerate(edges[:3], 1):
                logger.info(f"  {i}. {edge['description']}")
        
        logger.info("")
        
        if mistakes:
            logger.info("Top 3 Mistakes:")
            for i, mistake in enumerate(mistakes[:3], 1):
                logger.info(f"  {i}. {mistake['description']}")
        
        logger.info("=" * 60)


# ==================== Standalone Usage ====================

if __name__ == "__main__":
    import argparse
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description="Mine edges and mistakes from labeled trades")
    parser.add_argument('--labeled-trades', required=True, help="Path to labeled trades CSV")
    parser.add_argument('--output-dir', required=True, help="Output directory for libraries")
    parser.add_argument('--min-support', type=int, default=5, help="Minimum trades for pattern")
    parser.add_argument('--min-edge-winrate', type=float, default=0.65, help="Min winrate for edge")
    
    args = parser.parse_args()
    
    # Create miner
    miner = EdgeAndMistakeMiner(config={
        'min_support': args.min_support,
        'min_edge_winrate': args.min_edge_winrate
    })
    
    # Run mining
    edges, mistakes = miner.analyze_and_export(
        labeled_trades_path=args.labeled_trades,
        output_dir=args.output_dir
    )
    
    print(f"\n✓ Mining complete.")
    print(f"✓ Found {len(edges)} edges and {len(mistakes)} mistakes")
    print(f"✓ Libraries saved to {args.output_dir}")
