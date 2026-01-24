"""
SCOPUS Backtest Runner
Direct in-process backtest execution - NO subprocesses, NO timeouts
"""
import yaml
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import pandas as pd

# Import existing backtest engine
from src.backtest.engine import run_backtest as engine_run_backtest
from src.market_data.types import Symbol


def run_backtest(
    config_path: str,
    symbols: List[str],
    start_date: str,
    end_date: str,
    initial_capital: float = 10000.0
) -> Dict[str, Any]:
    """
    Run backtest directly in-process (NO subprocess)
    
    Args:
        config_path: Path to config YAML (e.g., "config/mvp_v1.yaml")
        symbols: List of symbol strings (e.g., ["EURUSD", "GBPUSD"])
        start_date: ISO date string (e.g., "2023-01-01")
        end_date: ISO date string (e.g., "2023-01-07")
        initial_capital: Starting capital in USD
    
    Returns:
        Dict with:
            - trades: int
            - winrate: float (0-1)
            - pnl: float
            - sharpe: float
            - max_dd: float (0-1)
            - regime_stats: dict
            - file_paths: dict
            - execution_time: float
    """
    import time
    start_time = time.time()
    
    # Load config
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Convert symbol strings to Symbol enum
    symbol_enums = [Symbol[s] for s in symbols]
    
    # Parse dates
    start_dt = pd.Timestamp(start_date, tz='UTC')
    end_dt = pd.Timestamp(end_date, tz='UTC')
    
    # Run existing backtest engine
    result = engine_run_backtest(
        symbols=symbol_enums,
        start_date=start_dt,
        end_date=end_dt,
        initial_capital=initial_capital,
        csv_price_dir="data/raw/forex_kaggle_multiTF",
        output_dir="backtest_results",
        use_real_pipeline=True
    )
    
    # Calculate execution time
    execution_time = time.time() - start_time
    
    # Extract summary statistics from BacktestResult
    summary = {
        'trades': result.total_trades,
        'winrate': result.winrate,
        'pnl': result.final_equity - initial_capital,
        'sharpe': result.sharpe_ratio,
        'max_dd': result.max_drawdown,
        'profit_factor': result.profit_factor,
        'regime_stats': result.regime_breakdown,
        'decision_stats': result.decision_source_breakdown,
        'file_paths': {
            'summary': 'backtest_results/summary.json',
            'trades': 'backtest_results/trades.csv',
            'equity': 'backtest_results/equity.csv'
        },
        'execution_time': execution_time,
        'config': {
            'symbols': symbols,
            'start_date': start_date,
            'end_date': end_date,
            'initial_capital': initial_capital
        }
    }
    
    return summary


def validate_inputs(
    config_path: str,
    symbols: List[str],
    start_date: str,
    end_date: str
) -> tuple[bool, str]:
    """
    Validate backtest inputs
    
    Returns:
        (is_valid, error_message)
    """
    # Check config exists
    if not Path(config_path).exists():
        return False, f"Config file not found: {config_path}"
    
    # Check symbols are valid
    valid_symbols = [s.name for s in Symbol]
    for sym in symbols:
        if sym not in valid_symbols:
            return False, f"Invalid symbol: {sym}. Valid: {valid_symbols}"
    
    # Check dates
    try:
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        if start >= end:
            return False, "Start date must be before end date"
    except Exception as e:
        return False, f"Invalid date format: {e}"
    
    return True, ""
