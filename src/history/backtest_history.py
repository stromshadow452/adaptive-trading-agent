"""
SCOPUS Backtest History Module

Persistent storage and retrieval of backtest run history using JSONLines format.

Author: SCOPUS Team
Date: 2025-11-25
"""

import json
import os
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Storage path
HISTORY_DIR = Path("data/history")
BACKTEST_RUNS_FILE = HISTORY_DIR / "backtest_runs.jsonl"


@dataclass
class BacktestRun:
    """Schema for a single backtest run."""
    run_id: str = ""
    created_at: str = ""  # ISO datetime UTC
    symbol: str = ""
    start_date: str = ""  # YYYY-MM-DD
    end_date: str = ""    # YYYY-MM-DD
    config_flags: Optional[Dict[str, bool]] = None  # use_meta_judge, use_finrl, use_strategy_bank
    performance: Optional[Dict[str, float]] = None  # trades, winrate, pnl, max_drawdown, sharpe, expectancy
    meta_stats: Optional[Dict[str, float]] = None  # meta_block_rate, meta_score_mean, meta_score_std
    source_files: Optional[Dict[str, str]] = None  # summary_path, trades_path
    metrics: Optional[Dict[str, Any]] = None  # Legacy field for backwards compatibility
    extra: Optional[Dict[str, Any]] = None  # Catch-all for unknown fields


def _ensure_history_dir():
    """Ensure history directory exists."""
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def _generate_run_id(symbol: str) -> str:
    """Generate unique run ID."""
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"bt_{symbol}_{timestamp}"


def log_backtest_run(run: BacktestRun) -> None:
    """
    Log a backtest run to persistent storage.
    
    Args:
        run: BacktestRun instance
    """
    _ensure_history_dir()
    
    # Auto-generate run_id if not provided
    if not run.run_id:
        run.run_id = _generate_run_id(run.symbol)
    
    # Auto-set created_at if not provided
    if not run.created_at:
        run.created_at = datetime.utcnow().isoformat() + 'Z'
    
    # Append to JSONLines file
    with open(BACKTEST_RUNS_FILE, 'a', encoding='utf-8') as f:
        f.write(json.dumps(asdict(run)) + '\n')
    
    logger.info(f"Logged backtest run: {run.run_id}")


def list_backtest_runs(
    symbol: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    limit: int = 100
) -> List[BacktestRun]:
    """
    List backtest runs with optional filters.
    
    Args:
        symbol: Filter by symbol (e.g., 'EURUSD')
        date_from: Filter runs created after this date (ISO format)
        date_to: Filter runs created before this date (ISO format)
        limit: Maximum number of runs to return
        
    Returns:
        List of BacktestRun instances, sorted by created_at (newest first)
    """
    if not BACKTEST_RUNS_FILE.exists():
        return []
    
    runs = []
    
    with open(BACKTEST_RUNS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            try:
                data = json.loads(line)
                # Filter to only known fields to handle legacy data with extra fields
                known_fields = {'run_id', 'created_at', 'symbol', 'start_date', 'end_date', 
                               'config_flags', 'performance', 'meta_stats', 'source_files', 
                               'metrics', 'extra'}
                filtered_data = {k: v for k, v in data.items() if k in known_fields}
                run = BacktestRun(**filtered_data)
                
                # Apply filters
                if symbol and run.symbol != symbol:
                    continue
                
                if date_from and run.created_at < date_from:
                    continue
                
                if date_to and run.created_at > date_to:
                    continue
                
                runs.append(run)
                
            except Exception as e:
                logger.error(f"Error parsing backtest run: {e}")
                continue
    
    # Sort by created_at (newest first)
    runs.sort(key=lambda r: r.created_at, reverse=True)
    
    # Apply limit
    return runs[:limit]


def get_backtest_run(run_id: str) -> Optional[BacktestRun]:
    """
    Get a single backtest run by ID.
    
    Args:
        run_id: Run ID to retrieve
        
    Returns:
        BacktestRun instance or None if not found
    """
    if not BACKTEST_RUNS_FILE.exists():
        return None
    
    with open(BACKTEST_RUNS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            try:
                data = json.loads(line)
                if data.get('run_id') == run_id:
                    # Filter to only known fields
                    known_fields = {'run_id', 'created_at', 'symbol', 'start_date', 'end_date', 
                                   'config_flags', 'performance', 'meta_stats', 'source_files', 
                                   'metrics', 'extra'}
                    filtered_data = {k: v for k, v in data.items() if k in known_fields}
                    return BacktestRun(**filtered_data)
            except Exception as e:
                logger.error(f"Error parsing backtest run: {e}")
                continue
    
    return None


def get_latest_run(symbol: Optional[str] = None) -> Optional[BacktestRun]:
    """
    Get the most recent backtest run.
    
    Args:
        symbol: Optional symbol filter
        
    Returns:
        Latest BacktestRun or None
    """
    runs = list_backtest_runs(symbol=symbol, limit=1)
    return runs[0] if runs else None


# ==================== Utility Functions ====================

def get_run_statistics(symbol: Optional[str] = None) -> Dict[str, Any]:
    """
    Get aggregate statistics across all runs.
    
    Args:
        symbol: Optional symbol filter
        
    Returns:
        Dict with statistics
    """
    runs = list_backtest_runs(symbol=symbol, limit=1000)
    
    if not runs:
        return {
            'total_runs': 0,
            'avg_winrate': 0,
            'avg_pnl': 0,
            'total_trades': 0
        }
    
    total_trades = sum(r.performance.get('trades', 0) for r in runs)
    avg_winrate = sum(r.performance.get('winrate', 0) for r in runs) / len(runs)
    avg_pnl = sum(r.performance.get('pnl', 0) for r in runs) / len(runs)
    
    return {
        'total_runs': len(runs),
        'avg_winrate': avg_winrate,
        'avg_pnl': avg_pnl,
        'total_trades': total_trades,
        'symbols': list(set(r.symbol for r in runs))
    }


# ==================== Standalone Usage ====================

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create sample run
    sample_run = BacktestRun(
        run_id="",  # Will be auto-generated
        created_at="",  # Will be auto-generated
        symbol="EURUSD",
        start_date="2023-01-01",
        end_date="2023-01-31",
        config_flags={
            "use_meta_judge": True,
            "use_finrl": False,
            "use_strategy_bank": False
        },
        performance={
            "trades": 50,
            "winrate": 0.60,
            "pnl": 125.50,
            "max_drawdown": 0.05,
            "sharpe": 1.8,
            "expectancy": 2.51
        },
        meta_stats={
            "meta_block_rate": 0.30,
            "meta_score_mean": 0.65,
            "meta_score_std": 0.15
        },
        source_files={
            "summary_path": "backtest_results/summary.json",
            "trades_path": "backtest_results/trades.csv"
        }
    )
    
    # Log it
    log_backtest_run(sample_run)
    
    # List runs
    runs = list_backtest_runs(symbol="EURUSD", limit=10)
    print(f"Found {len(runs)} runs for EURUSD")
    
    # Get statistics
    stats = get_run_statistics()
    print(f"Statistics: {stats}")
