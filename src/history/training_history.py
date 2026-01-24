"""
SCOPUS Training History Module

Persistent storage and retrieval of quarterly training run history using JSONLines format.

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
TRAINING_RUNS_FILE = HISTORY_DIR / "training_runs.jsonl"


@dataclass
class TrainingRun:
    """Schema for a single training run."""
    training_id: str
    created_at: str  # ISO datetime UTC
    symbol: str
    period_from: str  # YYYY-MM-DD
    period_to: str    # YYYY-MM-DD
    total_trades: int
    good_trades: int
    bad_trades: int
    edges_found: int
    mistakes_found: int
    meta_stats: Dict[str, float]  # meta_auc, meta_accuracy, meta_precision, meta_recall
    paths: Dict[str, str]  # report_path, meta_model_path, edge_library_path, mistake_library_path


def _ensure_history_dir():
    """Ensure history directory exists."""
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def _generate_training_id(symbol: str) -> str:
    """Generate unique training ID."""
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"train_{symbol}_{timestamp}"


def log_training_run(run: TrainingRun) -> None:
    """
    Log a training run to persistent storage.
    
    Args:
        run: TrainingRun instance
    """
    _ensure_history_dir()
    
    # Auto-generate training_id if not provided
    if not run.training_id:
        run.training_id = _generate_training_id(run.symbol)
    
    # Auto-set created_at if not provided
    if not run.created_at:
        run.created_at = datetime.utcnow().isoformat() + 'Z'
    
    # Append to JSONLines file
    with open(TRAINING_RUNS_FILE, 'a', encoding='utf-8') as f:
        f.write(json.dumps(asdict(run)) + '\n')
    
    logger.info(f"Logged training run: {run.training_id}")


def list_training_runs(
    symbol: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    limit: int = 50
) -> List[TrainingRun]:
    """
    List training runs with optional filters.
    
    Args:
        symbol: Filter by symbol (e.g., 'EURUSD')
        date_from: Filter runs created after this date (ISO format)
        date_to: Filter runs created before this date (ISO format)
        limit: Maximum number of runs to return
        
    Returns:
        List of TrainingRun instances, sorted by created_at (newest first)
    """
    if not TRAINING_RUNS_FILE.exists():
        return []
    
    runs = []
    
    with open(TRAINING_RUNS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            try:
                data = json.loads(line)
                run = TrainingRun(**data)
                
                # Apply filters
                if symbol and run.symbol != symbol:
                    continue
                
                if date_from and run.created_at < date_from:
                    continue
                
                if date_to and run.created_at > date_to:
                    continue
                
                runs.append(run)
                
            except Exception as e:
                logger.error(f"Error parsing training run: {e}")
                continue
    
    # Sort by created_at (newest first)
    runs.sort(key=lambda r: r.created_at, reverse=True)
    
    # Apply limit
    return runs[:limit]


def get_training_run(training_id: str) -> Optional[TrainingRun]:
    """
    Get a single training run by ID.
    
    Args:
        training_id: Training ID to retrieve
        
    Returns:
        TrainingRun instance or None if not found
    """
    if not TRAINING_RUNS_FILE.exists():
        return None
    
    with open(TRAINING_RUNS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            try:
                data = json.loads(line)
                if data.get('training_id') == training_id:
                    return TrainingRun(**data)
            except Exception as e:
                logger.error(f"Error parsing training run: {e}")
                continue
    
    return None


def get_latest_training(symbol: Optional[str] = None) -> Optional[TrainingRun]:
    """
    Get the most recent training run.
    
    Args:
        symbol: Optional symbol filter
        
    Returns:
        Latest TrainingRun or None
    """
    runs = list_training_runs(symbol=symbol, limit=1)
    return runs[0] if runs else None


# ==================== Utility Functions ====================

def get_training_statistics(symbol: Optional[str] = None) -> Dict[str, Any]:
    """
    Get aggregate statistics across all training runs.
    
    Args:
        symbol: Optional symbol filter
        
    Returns:
        Dict with statistics
    """
    runs = list_training_runs(symbol=symbol, limit=1000)
    
    if not runs:
        return {
            'total_trainings': 0,
            'avg_edges_found': 0,
            'avg_mistakes_found': 0,
            'avg_meta_auc': 0
        }
    
    avg_edges = sum(r.edges_found for r in runs) / len(runs)
    avg_mistakes = sum(r.mistakes_found for r in runs) / len(runs)
    avg_auc = sum(r.meta_stats.get('meta_auc', 0) for r in runs) / len(runs)
    
    return {
        'total_trainings': len(runs),
        'avg_edges_found': avg_edges,
        'avg_mistakes_found': avg_mistakes,
        'avg_meta_auc': avg_auc,
        'symbols': list(set(r.symbol for r in runs))
    }


def load_edges_from_training(training_id: str) -> Optional[List[Dict]]:
    """
    Load edge library from a training run.
    
    Args:
        training_id: Training ID
        
    Returns:
        List of edges or None
    """
    run = get_training_run(training_id)
    if not run:
        return None
    
    edge_path = run.paths.get('edge_library_path')
    if not edge_path or not Path(edge_path).exists():
        return None
    
    try:
        with open(edge_path, 'r') as f:
            data = json.load(f)
            return data.get('edges', [])
    except Exception as e:
        logger.error(f"Error loading edges: {e}")
        return None


def load_mistakes_from_training(training_id: str) -> Optional[List[Dict]]:
    """
    Load mistake library from a training run.
    
    Args:
        training_id: Training ID
        
    Returns:
        List of mistakes or None
    """
    run = get_training_run(training_id)
    if not run:
        return None
    
    mistake_path = run.paths.get('mistake_library_path')
    if not mistake_path or not Path(mistake_path).exists():
        return None
    
    try:
        with open(mistake_path, 'r') as f:
            data = json.load(f)
            return data.get('mistakes', [])
    except Exception as e:
        logger.error(f"Error loading mistakes: {e}")
        return None


# ==================== Standalone Usage ====================

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create sample training run
    sample_run = TrainingRun(
        training_id="",  # Will be auto-generated
        created_at="",  # Will be auto-generated
        symbol="EURUSD",
        period_from="2023-01-01",
        period_to="2023-03-31",
        total_trades=206,
        good_trades=61,
        bad_trades=145,
        edges_found=2,
        mistakes_found=2,
        meta_stats={
            "meta_auc": 1.0,
            "meta_accuracy": 0.7143,
            "meta_precision": 0.0,
            "meta_recall": 0.0
        },
        paths={
            "report_path": "test_output/quarterly_test/training_report_20231125_142302.json",
            "meta_model_path": "test_output/quarterly_test/meta_judge_latest.joblib",
            "edge_library_path": "test_output/quarterly_test/edge_library.json",
            "mistake_library_path": "test_output/quarterly_test/mistake_library.json"
        }
    )
    
    # Log it
    log_training_run(sample_run)
    
    # List runs
    runs = list_training_runs(symbol="EURUSD", limit=10)
    print(f"Found {len(runs)} training runs for EURUSD")
    
    # Get statistics
    stats = get_training_statistics()
    print(f"Statistics: {stats}")
