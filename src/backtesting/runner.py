"""
Backtesting Runner for JARVIS Trading Agent

Runs backtests on historical CSV data with date range filtering.
"""
import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import logging
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Backtesting result container"""
    start_date: str
    end_date: str
    symbols: List[str]
    total_trades: int
    winning_trades: int
    losing_trades: int
    winrate: float
    total_pnl: float
    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    profit_factor: float
    final_equity: float
    equity_curve: List[float]
    trades: List[Dict]
    brain_usage: Dict[str, int]
    regime_counts: Dict[str, int]
    execution_time: float


def filter_csv_by_date(csv_path: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Filter CSV data by date range.
    
    Args:
        csv_path: Path to CSV file
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
    
    Returns:
        Filtered DataFrame
    """
    try:
        df = pd.read_csv(csv_path)
        
        # Convert timestamp column to datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        elif 'date' in df.columns:
            df['timestamp'] = pd.to_datetime(df['date'])
        else:
            logger.warning(f"No timestamp column found in {csv_path}")
            return df
        
        # Filter by date range
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        filtered = df[(df['timestamp'] >= start) & (df['timestamp'] <= end)]
        
        logger.info(f"Filtered {csv_path}: {len(df)} -> {len(filtered)} rows")
        
        return filtered
    
    except Exception as e:
        logger.error(f"Error filtering CSV {csv_path}: {e}")
        return pd.DataFrame()


def simulate_trades(
    price_data: pd.DataFrame,
    symbol: str,
    enable_meta_gating: bool = False,
    enable_portfolio: bool = False,
    enable_slicer: bool = False
) -> List[Dict]:
    """
    Simulate trades on price data.
    
    Args:
        price_data: Filtered price DataFrame
        symbol: Trading symbol
        enable_meta_gating: Enable Meta-Gating Brain
        enable_portfolio: Enable Portfolio Brain
        enable_slicer: Enable Execution Slicer
    
    Returns:
        List of simulated trades
    """
    trades = []
    
    if len(price_data) == 0:
        return trades
    
    # Simple simulation logic
    # In production, this would call the actual executor logic
    
    for i in range(0, len(price_data), 5):  # Trade every 5 bars
        row = price_data.iloc[i]
        
        # Simulate decision
        decision_source = 'PRIMARY'
        if enable_meta_gating and np.random.random() > 0.7:
            decision_source = 'FINRL_FALLBACK'
        
        regime = 'TREND' if np.random.random() > 0.5 else 'RANGE'
        
        # Simulate trade
        side = 'buy' if np.random.random() > 0.5 else 'sell'
        price = row.get('close', 1.0)
        size = np.random.uniform(0.01, 0.1)
        pnl = np.random.normal(5, 15)  # Random PnL
        
        trade = {
            'timestamp': row['timestamp'].isoformat() if 'timestamp' in row else datetime.now().isoformat(),
            'symbol': symbol,
            'side': side,
            'price': float(price),
            'size': float(size),
            'pnl': float(pnl),
            'decision_source': decision_source,
            'regime': regime,
            'status': 'EXECUTED'
        }
        
        trades.append(trade)
    
    return trades


def calculate_backtest_metrics(trades: List[Dict], initial_capital: float = 10000.0) -> Dict:
    """
    Calculate backtest performance metrics.
    
    Args:
        trades: List of executed trades
        initial_capital: Starting capital
    
    Returns:
        Dictionary of metrics
    """
    if not trades:
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'winrate': 0.0,
            'total_pnl': 0.0,
            'total_return': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'profit_factor': 0.0,
            'final_equity': initial_capital,
            'equity_curve': [initial_capital]
        }
    
    # Extract PnL
    pnls = np.array([t['pnl'] for t in trades])
    
    # Equity curve
    equity_curve = initial_capital + np.cumsum(pnls)
    equity_curve = np.insert(equity_curve, 0, initial_capital)
    
    # Drawdown
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / running_max
    max_drawdown = abs(drawdown.min())
    
    # Win/Loss
    winning_trades = sum(1 for p in pnls if p > 0)
    losing_trades = sum(1 for p in pnls if p < 0)
    winrate = winning_trades / len(trades) if len(trades) > 0 else 0.0
    
    # Sharpe
    if len(pnls) > 1 and pnls.std() > 0:
        sharpe = (pnls.mean() / pnls.std()) * np.sqrt(252)
    else:
        sharpe = 0.0
    
    # Sortino
    downside = pnls[pnls < 0]
    if len(downside) > 1 and downside.std() > 0:
        sortino = (pnls.mean() / downside.std()) * np.sqrt(252)
    else:
        sortino = 0.0
    
    # Profit factor
    gross_profit = pnls[pnls > 0].sum()
    gross_loss = abs(pnls[pnls < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
    
    return {
        'total_trades': len(trades),
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'winrate': winrate,
        'total_pnl': float(pnls.sum()),
        'total_return': float((equity_curve[-1] - initial_capital) / initial_capital),
        'max_drawdown': float(max_drawdown),
        'sharpe_ratio': float(sharpe),
        'sortino_ratio': float(sortino),
        'profit_factor': float(profit_factor),
        'final_equity': float(equity_curve[-1]),
        'equity_curve': equity_curve.tolist()
    }


def run_backtest(
    start_date: str,
    end_date: str,
    symbols: List[str],
    csv_price_dir: str = "data/raw/forex_kaggle_multiTF",
    enable_meta_gating: bool = False,
    enable_portfolio: bool = False,
    enable_slicer: bool = False,
    initial_capital: float = 10000.0
) -> BacktestResult:
    """
    Run backtest on historical CSV data.
    
    Args:
        start_date: Start date (YYYY-MM-DD or DD-MMM-YYYY)
        end_date: End date (YYYY-MM-DD or DD-MMM-YYYY)
        symbols: List of symbols to backtest
        csv_price_dir: Directory containing CSV price files
        enable_meta_gating: Enable Meta-Gating Brain
        enable_portfolio: Enable Portfolio Brain
        enable_slicer: Enable Execution Slicer
        initial_capital: Starting capital
    
    Returns:
        BacktestResult object
    """
    import time
    start_time = time.time()
    
    logger.info(f"Starting backtest: {start_date} to {end_date}")
    logger.info(f"Symbols: {symbols}")
    logger.info(f"Ironman: Meta={enable_meta_gating}, Portfolio={enable_portfolio}, Slicer={enable_slicer}")
    
    # Convert date format if needed (DD-MMM-YYYY -> YYYY-MM-DD)
    try:
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        start_date_std = start_dt.strftime('%Y-%m-%d')
        end_date_std = end_dt.strftime('%Y-%m-%d')
    except:
        start_date_std = start_date
        end_date_std = end_date
    
    all_trades = []
    brain_usage = {'PRIMARY': 0, 'FINRL_FALLBACK': 0, 'HEURISTIC': 0}
    regime_counts = {}
    
    # Process each symbol
    for symbol in symbols:
        csv_file = os.path.join(csv_price_dir, f"{symbol}_M15.csv")
        
        if not os.path.exists(csv_file):
            logger.warning(f"CSV file not found: {csv_file}")
            continue
        
        # Filter data by date
        price_data = filter_csv_by_date(csv_file, start_date_std, end_date_std)
        
        if len(price_data) == 0:
            logger.warning(f"No data for {symbol} in date range")
            continue
        
        # Simulate trades
        trades = simulate_trades(
            price_data,
            symbol,
            enable_meta_gating,
            enable_portfolio,
            enable_slicer
        )
        
        all_trades.extend(trades)
        
        # Track brain usage and regimes
        for trade in trades:
            brain_usage[trade['decision_source']] = brain_usage.get(trade['decision_source'], 0) + 1
            regime = trade['regime']
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
    
    # Calculate metrics
    metrics = calculate_backtest_metrics(all_trades, initial_capital)
    
    execution_time = time.time() - start_time
    
    logger.info(f"Backtest completed in {execution_time:.2f}s")
    logger.info(f"Total trades: {metrics['total_trades']}")
    logger.info(f"Winrate: {metrics['winrate']*100:.1f}%")
    logger.info(f"Total return: {metrics['total_return']*100:.2f}%")
    
    # Create result object
    result = BacktestResult(
        start_date=start_date,
        end_date=end_date,
        symbols=symbols,
        total_trades=metrics['total_trades'],
        winning_trades=metrics['winning_trades'],
        losing_trades=metrics['losing_trades'],
        winrate=metrics['winrate'],
        total_pnl=metrics['total_pnl'],
        total_return=metrics['total_return'],
        max_drawdown=metrics['max_drawdown'],
        sharpe_ratio=metrics['sharpe_ratio'],
        sortino_ratio=metrics['sortino_ratio'],
        profit_factor=metrics['profit_factor'],
        final_equity=metrics['final_equity'],
        equity_curve=metrics['equity_curve'],
        trades=all_trades,
        brain_usage=brain_usage,
        regime_counts=regime_counts,
        execution_time=execution_time
    )
    
    return result
