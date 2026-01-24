"""
SCOPUS Job Runner
Watches jobs/pending/ directory for new backtest/training jobs and executes them via terminal commands.
"""
import os
import sys
import json
import time
import subprocess
import logging
import traceback
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/job_runner.log')
    ]
)
logger = logging.getLogger(__name__)

JOBS_DIR = Path("jobs")
PENDING_DIR = JOBS_DIR / "pending"
COMPLETED_DIR = JOBS_DIR / "completed"
FAILED_DIR = JOBS_DIR / "failed"
HISTORY_FILE = Path("data/history/backtest_runs.jsonl")
TRAINING_HISTORY_FILE = Path("data/history/training_runs.jsonl")

def setup_dirs():
    JOBS_DIR.mkdir(exist_ok=True)
    PENDING_DIR.mkdir(exist_ok=True)
    COMPLETED_DIR.mkdir(exist_ok=True)
    FAILED_DIR.mkdir(exist_ok=True)
    Path("data/history").mkdir(parents=True, exist_ok=True)

def process_backtest_job(job_file: Path, job_data: dict):
    """Execute backtest job"""
    job_id = job_data.get("job_id")
    symbol = job_data.get("symbol")
    start = job_data.get("start")
    end = job_data.get("end")
    config = job_data.get("config", "config/mvp_v1.yaml")
    
    logger.info(f"Processing backtest job {job_id} for {symbol}")
    
    # Create output directory
    output_dir = Path(f"data/history/{job_id}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Construct command
    cmd = [
        sys.executable,
        "-m", "src.backtest.engine",
        "--config", config,
        "--symbols", symbol,
        "--start", start,
        "--end", end,
        "--output", str(output_dir)
    ]
    
    logger.info(f"Running command: {' '.join(cmd)}")
    
    # Execute - don't capture output so tqdm shows in real-time
    start_time = time.time()
    result = subprocess.run(
        cmd,
        cwd=os.getcwd()
    )
    duration = time.time() - start_time
    
    if result.returncode != 0:
        raise Exception(f"Backtest failed with return code {result.returncode}")
        
    logger.info(f"Backtest completed in {duration:.2f}s")
    
    # Load results to log to history
    summary_file = output_dir / "summary.json"
    if not summary_file.exists():
        raise Exception("Summary file not found - backtest may have failed silently")
        
    with open(summary_file, 'r') as f:
        summary = json.load(f)
        
    # Append to history - calculate P&L from equity
    initial_capital = summary.get("initial_capital", 10000)
    final_equity = summary.get("final_equity", initial_capital)
    total_pnl = final_equity - initial_capital
    
    history_entry = {
        "run_id": job_id,
        "created_at": datetime.now().isoformat(),
        "symbol": symbol,
        "start_date": start,
        "end_date": end,
        "performance": {
            "total_trades": summary.get("total_trades", 0),
            "winrate": summary.get("winrate", 0),
            "total_pnl": total_pnl,
            "sharpe_ratio": summary.get("sharpe_ratio", 0),
            "max_drawdown": summary.get("max_drawdown", 0),
            "profit_factor": summary.get("profit_factor", 0),
            "final_equity": final_equity
        },
        "config_flags": summary.get("config", {}),
        "status": "success"
    }
    
    with open(HISTORY_FILE, 'a') as f:
        f.write(json.dumps(history_entry) + '\n')

def process_training_job(job_file: Path, job_data: dict):
    """Execute training job"""
    job_id = job_data.get("job_id")
    symbol = job_data.get("symbol")
    run_id = job_data.get("run_id") # Source backtest run
    start_date = job_data.get("start", "2023-01-01")
    end_date = job_data.get("end", "2023-03-31")
    trades_file = job_data.get("trades_file")
    
    if not trades_file:
        # Fallback if not provided in JSON
        trades_file = f"data/history/{run_id}/trades.csv"
    
    logger.info(f"Processing training job {job_id} for {symbol}")
    
    # Create output directory
    output_dir = Path(f"data/history/{job_id}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Construct command
    cmd = [
        sys.executable,
        "-m", "tools.run_quarterly_training",
        "--symbol", symbol,
        "--from", start_date,
        "--to", end_date,
        "--trades", str(trades_file),
        "--output-dir", str(output_dir)
    ]
        
    logger.info(f"Running command: {' '.join(cmd)}")
    
    # Execute
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=os.getcwd()
    )
    
    if result.returncode != 0:
        raise Exception(f"Training failed: {result.stderr}")
        
    logger.info("Training completed successfully")
    
    # Log to training history
    history_entry = {
        "training_id": job_id,
        "run_id": run_id,
        "created_at": datetime.now().isoformat(),
        "symbol": symbol,
        "status": "success",
        "output_dir": str(output_dir)
    }
    
    with open(TRAINING_HISTORY_FILE, 'a') as f:
        f.write(json.dumps(history_entry) + '\n')

def main():
    setup_dirs()
    logger.info("Job Runner started. Watching jobs/pending/ directory...")
    
    while True:
        try:
            # Look for JSON files in pending directory
            job_files = list(PENDING_DIR.glob("*.json"))
            
            for job_file in job_files:
                try:
                    with open(job_file, 'r') as f:
                        job_data = json.load(f)
                    
                    job_type = job_data.get("type", "backtest")
                    
                    if job_type == "backtest":
                        process_backtest_job(job_file, job_data)
                    elif job_type == "training":
                        process_training_job(job_file, job_data)
                    else:
                        logger.warning(f"Unknown job type: {job_type}")
                    
                    # Move to completed
                    job_file.rename(COMPLETED_DIR / job_file.name)
                    
                except Exception as e:
                    logger.error(f"Error processing {job_file}: {e}")
                    traceback.print_exc()
                    # Move to failed
                    job_file.rename(FAILED_DIR / job_file.name)
            
            time.sleep(1)
            
        except KeyboardInterrupt:
            logger.info("Stopping Job Runner...")
            break
        except Exception as e:
            logger.error(f"Main loop error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()
