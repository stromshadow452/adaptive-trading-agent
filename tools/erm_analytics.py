"""
ERM Analytics & Insight Layer
==============================

Offline analytics for debugging and optimization:
1. Pattern heatmaps (session × edge_tier → avg R)
2. Failure cluster detection
3. Size-multiplier effectiveness curves

Usage:
    python tools/erm_analytics.py
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from typing import List, Dict, Any
from datetime import datetime

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("Note: matplotlib/seaborn not available, skipping plot generation")

from src.reasoning.erm import (
    ExperienceReasoningModule, get_erm, 
    REGIME_MAP, SESSION_MAP, VOL_MAP, EDGE_TIER_MAP
)


# Inverse maps for display
REGIME_NAMES = {v: k for k, v in REGIME_MAP.items()}
SESSION_NAMES = {v: k for k, v in SESSION_MAP.items()}
VOL_NAMES = {v: k for k, v in VOL_MAP.items()}
EDGE_TIER_NAMES = {v: k for k, v in EDGE_TIER_MAP.items()}


def generate_pattern_heatmap(erm: ExperienceReasoningModule, 
                              regime: str = "RANGE",
                              output_dir: Path = None) -> pd.DataFrame:
    """
    Generate heatmap: X=session, Y=edge_tier, Color=avg_R
    
    Shows which session × edge combinations are most profitable.
    """
    output_dir = output_dir or Path("analytics")
    output_dir.mkdir(exist_ok=True)
    
    data = []
    regime_id = REGIME_MAP.get(regime, 0)
    
    for pattern, record in erm.memory.records.items():
        if pattern.regime == regime_id:
            best = record.best_solution
            if best and best.total_trades >= 3:
                data.append({
                    'session': SESSION_NAMES.get(pattern.session, 'UNK'),
                    'edge_tier': EDGE_TIER_NAMES.get(pattern.edge_tier, 'C'),
                    'avg_r': best.avg_r,
                    'success_rate': best.success_rate,
                    'trades': best.total_trades
                })
    
    if not data:
        print(f"[Analytics] No data for {regime} regime")
        return pd.DataFrame()
    
    df = pd.DataFrame(data)
    
    # Create pivot table
    pivot = df.pivot_table(
        values='avg_r', 
        index='edge_tier', 
        columns='session', 
        aggfunc='mean'
    )
    
    # Reorder for display
    session_order = ['OFF', 'SYDNEY', 'TOKYO', 'LONDON', 'NEW_YORK']
    tier_order = ['D', 'C', 'B', 'A']
    
    pivot = pivot.reindex(columns=[s for s in session_order if s in pivot.columns])
    pivot = pivot.reindex(index=[t for t in tier_order if t in pivot.index])
    
    if HAS_PLOTTING:
        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot, annot=True, cmap='RdYlGn', center=0,
                    fmt='.2f', linewidths=0.5, cbar_kws={'label': 'Avg R-Multiple'})
        plt.title(f'{regime} Regime — Avg R by Session × Edge Tier\n(Patterns with ≥3 trades)')
        plt.ylabel('Edge Tier')
        plt.xlabel('Trading Session')
        plt.tight_layout()
        
        outpath = output_dir / f'erm_heatmap_{regime.lower()}.png'
        plt.savefig(outpath, dpi=150)
        plt.close()
        print(f"[Analytics] Saved heatmap to {outpath}")
    
    # Also generate text version
    print(f"\n{'='*60}")
    print(f" {regime} REGIME HEATMAP (Avg R)")
    print(f"{'='*60}")
    print(pivot.to_string())
    print(f"{'='*60}\n")
    
    return pivot


def find_failure_clusters(erm: ExperienceReasoningModule, 
                          min_failures: int = 5) -> List[Dict[str, Any]]:
    """
    Identify patterns with repeated losses (failure clusters).
    
    These patterns should potentially be blocked or heavily reduced.
    """
    failures = []
    
    for pattern, record in erm.memory.records.items():
        for solution in record.solutions.values():
            if solution.failure_count >= min_failures:
                if solution.success_rate < 0.40:
                    failures.append({
                        'pattern': pattern.to_string(),
                        'regime': REGIME_NAMES.get(pattern.regime, 'UNK'),
                        'session': SESSION_NAMES.get(pattern.session, 'UNK'),
                        'edge_tier': EDGE_TIER_NAMES.get(pattern.edge_tier, 'C'),
                        'solution': solution.solution_type,
                        'failures': solution.failure_count,
                        'successes': solution.success_count,
                        'success_rate': solution.success_rate,
                        'avg_r': solution.avg_r,
                        'recommendation': 'BLOCK' if solution.success_rate < 0.30 else 'HEAVY_REDUCE'
                    })
    
    failures = sorted(failures, key=lambda x: (-x['failures'], x['success_rate']))
    
    print(f"\n{'='*60}")
    print(f" FAILURE CLUSTERS (≥{min_failures} failures, SR<40%)")
    print(f"{'='*60}")
    
    if not failures:
        print("No failure clusters found.")
    else:
        df = pd.DataFrame(failures)
        print(df.to_string(index=False))
    
    print(f"{'='*60}\n")
    
    return failures


def analyze_size_multiplier_effectiveness(erm: ExperienceReasoningModule,
                                           output_dir: Path = None) -> pd.DataFrame:
    """
    Analyze relationship between size multiplier and outcome.
    
    Shows if ERM sizing adjustments improve results.
    """
    output_dir = output_dir or Path("analytics")
    output_dir.mkdir(exist_ok=True)
    
    data = []
    
    for pattern, record in erm.memory.records.items():
        for solution in record.solutions.values():
            if solution.total_trades >= 3:
                data.append({
                    'size_multiplier': solution.size_multiplier,
                    'avg_r': solution.avg_r,
                    'success_rate': solution.success_rate,
                    'action': solution.solution_type,
                    'regime': REGIME_NAMES.get(pattern.regime, 'UNK'),
                    'trades': solution.total_trades
                })
    
    if not data:
        print("[Analytics] No data for size analysis")
        return pd.DataFrame()
    
    df = pd.DataFrame(data)
    
    if HAS_PLOTTING:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Size vs Avg R
        ax1 = axes[0]
        for action in df['action'].unique():
            subset = df[df['action'] == action]
            ax1.scatter(subset['size_multiplier'], subset['avg_r'], 
                        label=action, alpha=0.6, s=subset['trades']*3)
        ax1.set_xlabel('Size Multiplier')
        ax1.set_ylabel('Average R-Multiple')
        ax1.set_title('Size Multiplier vs Avg R')
        ax1.legend()
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax1.axvline(x=0.5, color='gray', linestyle=':', alpha=0.3)
        
        # Plot 2: Size vs Success Rate
        ax2 = axes[1]
        for action in df['action'].unique():
            subset = df[df['action'] == action]
            ax2.scatter(subset['size_multiplier'], subset['success_rate'], 
                        label=action, alpha=0.6, s=subset['trades']*3)
        ax2.set_xlabel('Size Multiplier')
        ax2.set_ylabel('Success Rate')
        ax2.set_title('Size Multiplier vs Win Rate')
        ax2.legend()
        ax2.axhline(y=0.5, color='gray', linestyle=':', alpha=0.3)
        
        plt.tight_layout()
        outpath = output_dir / 'erm_size_effectiveness.png'
        plt.savefig(outpath, dpi=150)
        plt.close()
        print(f"[Analytics] Saved size analysis to {outpath}")
    
    # Summary stats
    print(f"\n{'='*60}")
    print(" SIZE MULTIPLIER EFFECTIVENESS")
    print(f"{'='*60}")
    
    for action in sorted(df['action'].unique()):
        subset = df[df['action'] == action]
        print(f"\n{action}:")
        print(f"  Patterns: {len(subset)}")
        print(f"  Avg Size Mult: {subset['size_multiplier'].mean():.2f}")
        print(f"  Avg R: {subset['avg_r'].mean():.3f}")
        print(f"  Avg Win Rate: {subset['success_rate'].mean():.1%}")
    
    print(f"\n{'='*60}\n")
    
    return df


def generate_regime_comparison(erm: ExperienceReasoningModule) -> pd.DataFrame:
    """
    Compare performance across regimes.
    
    Critical for validating regime separation.
    """
    data = []
    
    for pattern, record in erm.memory.records.items():
        best = record.best_solution
        if best and best.total_trades >= 3:
            data.append({
                'regime': REGIME_NAMES.get(pattern.regime, 'UNK'),
                'avg_r': best.avg_r,
                'success_rate': best.success_rate,
                'trades': best.total_trades
            })
    
    if not data:
        print("[Analytics] No data for regime comparison")
        return pd.DataFrame()
    
    df = pd.DataFrame(data)
    
    print(f"\n{'='*60}")
    print(" REGIME COMPARISON")
    print(f"{'='*60}")
    
    for regime in ['RANGE', 'TREND', 'DANGER']:
        subset = df[df['regime'] == regime]
        if len(subset) > 0:
            print(f"\n{regime}:")
            print(f"  Patterns: {len(subset)}")
            print(f"  Total Trades: {subset['trades'].sum()}")
            print(f"  Avg R: {(subset['avg_r'] * subset['trades']).sum() / subset['trades'].sum():.3f}")
            print(f"  Avg Win Rate: {(subset['success_rate'] * subset['trades']).sum() / subset['trades'].sum():.1%}")
    
    print(f"\n{'='*60}\n")
    
    return df


def generate_full_report(erm: ExperienceReasoningModule = None,
                         output_dir: Path = None):
    """Generate complete analytics report."""
    if erm is None:
        erm = get_erm()
    
    output_dir = output_dir or Path("analytics")
    output_dir.mkdir(exist_ok=True)
    
    print("\n" + "=" * 70)
    print(" ERM ANALYTICS REPORT")
    print(f" Generated: {datetime.now().isoformat()}")
    print("=" * 70)
    
    # Basic stats
    stats = erm.stats()
    print(f"\nTotal Patterns: {stats['memory_stats']['total_patterns']}")
    print(f"Total Evaluations: {stats['total_evaluations']}")
    print(f"Total Updates: {stats['total_updates']}")
    
    # Generate all analyses
    print("\n[1/5] Generating RANGE heatmap...")
    generate_pattern_heatmap(erm, "RANGE", output_dir)
    
    print("[2/5] Generating TREND heatmap...")
    generate_pattern_heatmap(erm, "TREND", output_dir)
    
    print("[3/5] Finding failure clusters...")
    find_failure_clusters(erm)
    
    print("[4/5] Analyzing size effectiveness...")
    analyze_size_multiplier_effectiveness(erm, output_dir)
    
    print("[5/5] Comparing regimes...")
    generate_regime_comparison(erm)
    
    print("\n" + "=" * 70)
    print(" REPORT COMPLETE")
    print(f" Output directory: {output_dir.absolute()}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    # Run full analytics
    generate_full_report()
