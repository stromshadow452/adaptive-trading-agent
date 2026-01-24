"""
SCOPUS Async Backtest Runner

Runs backtests in background asyncio tasks while streaming logs and progress
to the SSE endpoint. Captures stdout/stderr and emits structured events.

Key Features:
1. Non-blocking execution via asyncio.create_task
2. Real-time log capture and streaming
3. Progress callbacks for tqdm-style updates
4. Automatic error handling and cleanup

Author: SCOPUS Team
Date: 2025-12-29
"""

import asyncio
import json
import sys
import io
import time
import traceback
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass

# Import streaming module for event emission
from src.api.streaming import (
    job_registry, emit_log, emit_progress, emit_result, emit_error,
    LogEvent, ProgressEvent
)

logger = logging.getLogger(__name__)


@dataclass
class BacktestJob:
    """Backtest job specification"""
    job_id: str
    symbol: str
    start_date: str
    end_date: str
    config_path: str = "config/mvp_v1.yaml"
    initial_capital: float = 10000.0
    output_dir: Optional[str] = None
    
    def __post_init__(self):
        if self.output_dir is None:
            self.output_dir = f"data/history/{self.job_id}"


class StreamingLogHandler(logging.Handler):
    """
    Custom logging handler that emits logs to SSE stream.
    Captures all Python logging and forwards to job queue.
    """
    
    def __init__(self, job_id: str):
        super().__init__()
        self.job_id = job_id
        self._loop = None
    
    def emit(self, record):
        try:
            msg = self.format(record)
            level = record.levelname
            
            # Get or create event loop
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
            
            # Schedule coroutine to run
            asyncio.run_coroutine_threadsafe(
                self._emit_async(msg, level),
                loop
            )
        except Exception:
            pass  # Don't crash on logging errors
    
    async def _emit_async(self, message: str, level: str):
        await emit_log(self.job_id, message, level)


class ProgressCallback:
    """
    Callback class for progress updates.
    Can be passed to backtest engine for real-time progress streaming.
    """
    
    def __init__(self, job_id: str, total: int, symbol: str = ""):
        self.job_id = job_id
        self.total = total
        self.symbol = symbol
        self.current = 0
        self.start_time = time.time()
        self._loop = None
    
    def update(self, n: int = 1):
        """Update progress by n steps"""
        self.current += n
        self._emit_progress()
    
    def set_progress(self, current: int, total: int = None):
        """Set absolute progress"""
        self.current = current
        if total is not None:
            self.total = total
        self._emit_progress()
    
    def _emit_progress(self):
        """Emit progress event"""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop - skip emission
            return
        
        elapsed = time.time() - self.start_time
        if self.current > 0:
            eta = (elapsed / self.current) * (self.total - self.current)
        else:
            eta = 0
        
        asyncio.create_task(
            emit_progress(
                self.job_id,
                self.current,
                self.total,
                self.symbol,
                eta,
                f"Processing {self.symbol}: {self.current}/{self.total}"
            )
        )


async def run_backtest_async(job: BacktestJob) -> Dict[str, Any]:
    """
    Run a backtest asynchronously with event streaming.
    
    This is the main entry point for async backtest execution.
    Captures all output and streams to the SSE endpoint.
    
    Args:
        job: BacktestJob specification
    
    Returns:
        Dict with backtest results/metrics
    """
    # Register job in registry
    await job_registry.register(job.job_id)
    
    # Emit start log
    await emit_log(job.job_id, f"Starting backtest for {job.symbol}", "INFO")
    await emit_log(job.job_id, f"Date range: {job.start_date} to {job.end_date}", "INFO")
    
    try:
        # Import backtest engine dynamically to avoid circular imports
        from src.backtest.engine import run_backtest
        from src.market_data import MarketDataStore
        from pathlib import Path
        import pandas as pd
        
        # Create output directory
        output_path = Path(job.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        await emit_log(job.job_id, "Loading market data...", "INFO")
        
        # Parse dates
        start_dt = pd.to_datetime(job.start_date).tz_localize("UTC")
        end_dt = pd.to_datetime(job.end_date).tz_localize("UTC")
        
        # Run in executor to not block event loop
        def _run_blocking():
            return run_backtest(
                symbols=[job.symbol],
                start_date=start_dt,
                end_date=end_dt,
                initial_capital=job.initial_capital,
                output_dir=str(output_path),
                use_real_pipeline=True,
                csv_price_dir="data/raw/forex_kaggle_multiTF",
                # Pass progress callback if supported
            )
        
        # Emit progress start
        await emit_progress(job.job_id, 0, 100, job.symbol, 0, "Initializing...")
        
        # Run blocking backtest in thread executor
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _run_blocking)
        
        # Emit completion
        await emit_progress(job.job_id, 100, 100, job.symbol, 0, "Completed!")
        
        # Extract metrics
        metrics = {
            "total_trades": result.total_trades,
            "winning_trades": result.winning_trades,
            "losing_trades": result.losing_trades,
            "winrate": result.winrate,
            "total_pnl": result.final_equity - job.initial_capital,
            "final_equity": result.final_equity,
            "max_drawdown": result.max_drawdown,
            "sharpe_ratio": result.sharpe_ratio,
            "profit_factor": result.profit_factor,
            "total_candles": result.total_candles,
            "execution_time": result.execution_time,
        }
        
        await emit_log(job.job_id, f"Backtest completed: {result.total_trades} trades", "INFO")
        await emit_log(job.job_id, f"Winrate: {result.winrate*100:.1f}%, P&L: ${metrics['total_pnl']:.2f}", "INFO")
        
        # Emit final result
        await emit_result(job.job_id, metrics, "success")
        
        return metrics
        
    except Exception as e:
        error_msg = str(e)
        tb = traceback.format_exc()
        logger.error(f"Backtest error for {job.job_id}: {error_msg}")
        
        await emit_log(job.job_id, f"ERROR: {error_msg}", "ERROR")
        await emit_error(job.job_id, error_msg, tb)
        
        return {"error": error_msg, "traceback": tb}


async def submit_backtest_job(job: BacktestJob) -> str:
    """
    Submit a backtest job for async execution.
    Returns immediately with job_id.
    
    Args:
        job: BacktestJob specification
    
    Returns:
        job_id string
    """
    # Register job queue
    await job_registry.register(job.job_id)
    
    # Start backtest in background task
    asyncio.create_task(run_backtest_async(job))
    
    logger.info(f"Submitted backtest job: {job.job_id}")
    return job.job_id


# ============================================================================
# Subprocess-based Runner (Alternative for complete isolation)
# ============================================================================

async def run_backtest_subprocess(job: BacktestJob) -> None:
    """
    Run backtest as subprocess with comprehensive error handling.
    
    Key fixes for Windows paths with spaces:
    1. Uses shell=True with quoted paths
    2. Sets PYTHONPATH explicitly
    3. Uses UTF-8 encoding with error replacement
    4. Captures and logs all stderr output
    5. Logs full command for debugging
    """
    import subprocess
    import threading
    import queue
    import os
    
    await job_registry.register(job.job_id)
    
    # Resolve all paths
    python_exe = sys.executable
    project_root = Path(__file__).parent.parent.parent.resolve()
    config_path = project_root / job.config_path
    output_path = Path(job.output_dir).resolve()
    
    # Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Build command with proper quoting for paths with spaces
    # Use shell execution to handle Windows path escaping
    cmd = (
        f'"{python_exe}" -m src.backtest.engine '
        f'--config "{config_path}" '
        f'--symbols {job.symbol} '
        f'--start {job.start_date} '
        f'--end {job.end_date} '
        f'--output "{output_path}"'
    )
    
    # Set up environment with PYTHONPATH
    env = os.environ.copy()
    env['PYTHONPATH'] = str(project_root)
    env['PYTHONIOENCODING'] = 'utf-8'
    
    # Log command for debugging
    print(f"\n{'='*60}", flush=True)
    print(f"[{job.job_id}] Starting subprocess", flush=True)
    print(f"  Command: {cmd}", flush=True)
    print(f"  CWD: {project_root}", flush=True)
    print(f"  PYTHONPATH: {project_root}", flush=True)
    print(f"{'='*60}\n", flush=True)
    
    await emit_log(job.job_id, f"Starting backtest for {job.symbol}...", "INFO")
    
    # Use queue for thread-safe message passing
    msg_queue = queue.Queue()
    
    def run_in_thread():
        """Run subprocess in separate thread to avoid blocking event loop"""
        try:
            # Use shell=True for proper path handling on Windows
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Merge stderr into stdout
                cwd=str(project_root),
                env=env,
                shell=True,  # Required for proper path escaping on Windows
                text=True,
                encoding='utf-8',
                errors='replace',
                bufsize=1,
            )
            
            print(f"[{job.job_id}] Process started with PID: {process.pid}", flush=True)
            msg_queue.put(('log', f"[PID: {process.pid}] Process started"))
            
            # Read lines one by one
            for line in iter(process.stdout.readline, ''):
                line = line.rstrip()
                if line:
                    # Log to terminal for debugging
                    print(f"[{job.job_id}] {line}", flush=True)
                    msg_queue.put(('log', line))
                    
                    # Parse FINAL_METRICS for instant result streaming
                    if line.startswith("FINAL_METRICS:"):
                        try:
                            metrics_json = line.split("FINAL_METRICS:")[1]
                            metrics = json.loads(metrics_json)
                            print(f"[{job.job_id}] Parsed FINAL_METRICS instantly!", flush=True)
                            msg_queue.put(('metrics', metrics))
                        except Exception as e:
                            print(f"[{job.job_id}] Failed to parse FINAL_METRICS: {e}", flush=True)
                    
                    # Parse progress
                    elif "Backtesting" in line and "%" in line:
                        try:
                            parts = line.split("|")[0]
                            pct = float(parts.split("%")[0].split()[-1])
                            nums = line.split("|")[-1].strip().split("/")
                            current = int(nums[0].split()[0])
                            total = int(nums[1].split()[0])
                            msg_queue.put(('progress', (current, total, pct)))
                        except:
                            pass
            
            process.stdout.close()
            returncode = process.wait()
            
            print(f"[{job.job_id}] Process exited with code: {returncode}", flush=True)
            msg_queue.put(('done', returncode))
            
        except Exception as e:
            import traceback
            error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            print(f"[{job.job_id}] ERROR: {error_msg}", flush=True)
            msg_queue.put(('error', error_msg))
    
    # Start thread
    thread = threading.Thread(target=run_in_thread, daemon=True)
    thread.start()
    
    # Process messages from queue
    output_lines = []
    returncode = None
    startup_timeout = 30  # seconds to wait for first log
    startup_received = False
    streamed_metrics = None  # Metrics from FINAL_METRICS line
    
    # Progress throttling - batch updates to reduce SSE spam
    last_progress_time = 0
    last_progress_candle = 0
    
    while True:
        try:
            # Use shorter timeout initially to detect startup failures
            timeout = 5 if not startup_received else 15
            msg_type, data = msg_queue.get(timeout=timeout)
            startup_received = True
            
            if msg_type == 'log':
                output_lines.append(data)
                await emit_log(job.job_id, data, "INFO")
                
            elif msg_type == 'progress':
                current, total, pct = data
                now = time.time()
                # Throttle: emit only every 50 candles or 500ms (prevents SSE spam)
                if (current - last_progress_candle >= 50 or 
                    now - last_progress_time >= 0.5 or
                    pct >= 99):  # Always emit final progress
                    await emit_progress(job.job_id, current, total, job.symbol, 0, f"{pct:.0f}%")
                    last_progress_time = now
                    last_progress_candle = current
                
            elif msg_type == 'metrics':
                # Instant metrics from FINAL_METRICS line - no disk I/O needed!
                streamed_metrics = data
                print(f"[{job.job_id}] Captured instant metrics!", flush=True)
                
            elif msg_type == 'done':
                returncode = data
                break
                
            elif msg_type == 'error':
                await emit_error(job.job_id, data, "")
                return
                
        except queue.Empty:
            if not startup_received:
                # No output received - process likely failed silently
                await emit_log(job.job_id, "Waiting for subprocess startup...", "WARNING")
                startup_timeout -= 5
                if startup_timeout <= 0:
                    await emit_error(job.job_id, 
                        "Subprocess failed to start (no output received)",
                        f"Command: {cmd}\n\nCheck API server terminal for actual error.")
                    return
            else:
                # Normal heartbeat
                await emit_log(job.job_id, "Still running...", "INFO")
    
    # Wait for thread cleanup
    thread.join(timeout=5)
    
    await emit_log(job.job_id, f"Process completed with code: {returncode}", "INFO")
    
    if returncode == 0:
        # OPTIMIZATION: Use streamed metrics if available (instant, no disk I/O)
        if streamed_metrics:
            print(f"[{job.job_id}] Using instant streamed metrics (no disk read)!", flush=True)
            await emit_result(job.job_id, streamed_metrics, "success")
        else:
            # Fallback: Read from summary.json (slower, requires disk I/O)
            print(f"[{job.job_id}] Falling back to disk read for metrics", flush=True)
            summary_file = output_path / "summary.json"
            if summary_file.exists():
                with open(summary_file) as f:
                    summary = json.load(f)
                
                metrics = {
                    "total_trades": summary.get("total_trades", 0),
                    "winrate": summary.get("winrate", 0),
                    "total_pnl": summary.get("final_equity", 10000) - summary.get("initial_capital", 10000),
                    "final_equity": summary.get("final_equity", 10000),
                    "sharpe_ratio": summary.get("sharpe_ratio", 0),
                    "max_drawdown": summary.get("max_drawdown", 0),
                    "profit_factor": summary.get("profit_factor", 0),
                }
                await emit_result(job.job_id, metrics, "success")
            else:
                await emit_error(job.job_id, "Summary file not found", str(summary_file))
    else:
        error_context = "\n".join(output_lines[-30:]) if output_lines else "No output captured"
        await emit_error(job.job_id, f"Process exited with code {returncode}", error_context)


async def submit_backtest_subprocess(job: BacktestJob) -> str:
    """Submit backtest as subprocess for execution"""
    await job_registry.register(job.job_id)
    asyncio.create_task(run_backtest_subprocess(job))
    return job.job_id

