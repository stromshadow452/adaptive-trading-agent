"""
Metrics & Telemetry System

Comprehensive session metrics for backtest analysis.
Calculates Sharpe, Sortino, drawdown, winrate, and tracks blocking reasons.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import json
import os
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collect and calculate trading session metrics."""
    
    def __init__(self):
        self.trades: List[Dict] = []
        self.blocks: Dict[str, int] = {
            'JARVIS_GUARD': 0,
            'CIRCUIT_BREAKER': 0,
            'REGIME_CRASH': 0,
            'REGIME_MISMATCH': 0,
            'LOW_CONFIDENCE': 0,
            'PORTFOLIO_RISK': 0
        }
        self.brain_usage: Dict[str, int] = {
            'PRIMARY': 0,
            'FINRL_FALLBACK': 0,
            'HEURISTIC': 0
        }
        self.regime_counts: Dict[str, int] = {}
    
    def add_trade(self, trade: Dict):
        """Add a trade to metrics."""
        self.trades.append(trade)
        
        # Track brain usage
        decision_source = trade.get('decision_source', 'UNKNOWN')
        if decision_source in self.brain_usage:
            self.brain_usage[decision_source] += 1
    
    def add_block(self, reason: str):
        """Track a blocked trade."""
        if reason in self.blocks:
            self.blocks[reason] += 1
        else:
            self.blocks['OTHER'] = self.blocks.get('OTHER', 0) + 1
    
    def add_regime(self, regime: str):
        """Track regime occurrence."""
        self.regime_counts[regime] = self.regime_counts.get(regime, 0) + 1
    
    def calculate_metrics(self, initial_capital: float = 10000.0) -> Dict:
        """
        Calculate comprehensive trading metrics.
        
        Returns:
            Dict with all metrics
        """
        if not self.trades:
            return self._empty_metrics()
        
        # Convert trades to DataFrame
        df = pd.DataFrame(self.trades)
        
        # Calculate returns
        if 'pnl' in df.columns:
            returns = df['pnl'].values
        else:
            returns = np.zeros(len(df))
        
        # Basic metrics
        total_trades = len(self.trades)
        winning_trades = sum(1 for r in returns if r > 0)
        losing_trades = sum(1 for r in returns if r < 0)
        winrate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        total_pnl = returns.sum()
        avg_pnl = returns.mean()
        
        # Equity curve
        equity = initial_capital + np.cumsum(returns)
        
        # Drawdown
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max
        max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0.0
        
        # Sharpe ratio (annualized, assuming daily returns)
        if len(returns) > 1 and returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            sharpe = 0.0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 1 and downside_returns.std() > 0:
            sortino = (returns.mean() / downside_returns.std()) * np.sqrt(252)
        else:
            sortino = 0.0
        
        # Profit factor
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
        
        # Return metrics
        total_return = (equity[-1] - initial_capital) / initial_capital
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'winrate': winrate,
            'total_pnl': total_pnl,
            'avg_pnl_per_trade': avg_pnl,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'profit_factor': profit_factor,
            'final_equity': equity[-1],
            'brain_usage': self.brain_usage.copy(),
            'blocks': self.blocks.copy(),
            'regime_counts': self.regime_counts.copy()
        }
    
    def _empty_metrics(self) -> Dict:
        """Return empty metrics dict."""
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'winrate': 0.0,
            'total_pnl': 0.0,
            'avg_pnl_per_trade': 0.0,
            'total_return': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'profit_factor': 0.0,
            'final_equity': 10000.0,
            'brain_usage': self.brain_usage.copy(),
            'blocks': self.blocks.copy(),
            'regime_counts': self.regime_counts.copy()
        }
    
    def save_session_summary(self, output_dir: str = "logs/summary"):
        """
        Save session metrics to JSON file.
        
        Args:
            output_dir: Directory to save summary
        """
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"session_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        metrics = self.calculate_metrics()
        metrics['timestamp'] = timestamp
        metrics['num_trades'] = len(self.trades)
        
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"[METRICS] Saved session summary to {filepath}")
        
        return filepath
    
    def print_summary(self):
        """Print metrics summary to console."""
        metrics = self.calculate_metrics()
        
        print("\n" + "=" * 60)
        print("SESSION METRICS SUMMARY")
        print("=" * 60)
        print(f"Total Trades:     {metrics['total_trades']}")
        print(f"Winrate:          {metrics['winrate']*100:.1f}%")
        print(f"Total Return:     {metrics['total_return']*100:.2f}%")
        print(f"Max Drawdown:     {metrics['max_drawdown']*100:.2f}%")
        print(f"Sharpe Ratio:     {metrics['sharpe_ratio']:.2f}")
        print(f"Sortino Ratio:    {metrics['sortino_ratio']:.2f}")
        print(f"Profit Factor:    {metrics['profit_factor']:.2f}")
        print(f"\nBrain Usage:")
        for brain, count in metrics['brain_usage'].items():
            print(f"  {brain}: {count}")
        print(f"\nBlocked Trades:")
        for reason, count in metrics['blocks'].items():
            if count > 0:
                print(f"  {reason}: {count}")
        print("=" * 60 + "\n")


# Factory function
def create_metrics_collector() -> MetricsCollector:
    """Create and return MetricsCollector instance."""
    return MetricsCollector()
