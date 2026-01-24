"""
SCOPUS FastAPI Job Server
Lightweight server that:
1. Receives job requests (Backtest/Training)
2. Writes job files to jobs/ directory
3. Serves OHLCV data
4. Serves History data
"""
import sys
import os
import json
import uuid
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.market_data.store import MarketDataStore
from src.market_data.types import Symbol, Timeframe
from src.history.backtest_history import list_backtest_runs, get_backtest_run
from src.history.training_history import list_training_runs, get_training_run

# Import streaming module
from src.api.streaming import router as streaming_router
from src.backtest.async_runner import BacktestJob, submit_backtest_subprocess

# Initialize data store with ALL data sources (including D1 backup)
data_store = MarketDataStore([
    project_root / "data" / "raw" / "forex_kaggle_multiTF",
    project_root / "data" / "raw" / "forex_backup_2020_2025",  # D1/Daily data
])

app = FastAPI(title="SCOPUS Job Server", version="4.0.0")

# Mount streaming router
app.include_router(streaming_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class BacktestRequest(BaseModel):
    config: str = "config/mvp_v1.yaml"
    symbols: List[str]
    start: str
    end: str
    initial_capital: float = 10000.0

class TrainingRequest(BaseModel):
    run_id: str
    symbol: str
    start: Optional[str] = None
    end: Optional[str] = None

# Endpoints

@app.post("/jobs/submit/backtest")
async def submit_backtest_job(request: BacktestRequest):
    """Submit a backtest job (legacy - uses job runner)"""
    job_id = f"bt_{uuid.uuid4().hex[:8]}"
    
    job_data = {
        "type": "backtest",
        "job_id": job_id,
        "created_at": datetime.now().isoformat(),
        "config": request.config,
        "symbol": request.symbols[0], # MVP supports single symbol
        "start": request.start,
        "end": request.end,
        "initial_capital": request.initial_capital
    }
    
    # Write to jobs/pending directory
    pending_dir = project_root / "jobs" / "pending"
    pending_dir.mkdir(parents=True, exist_ok=True)
    job_file = pending_dir / f"{job_id}.json"
    with open(job_file, "w") as f:
        json.dump(job_data, f, indent=2)
        
    return {"job_id": job_id, "status": "queued"}


@app.post("/jobs/submit/backtest/stream")
async def submit_streaming_backtest(request: BacktestRequest):
    """
    Submit a backtest job with SSE streaming support.
    
    Returns job_id immediately. Connect to /stream/backtest/{job_id}
    to receive real-time logs, progress, and results.
    
    This endpoint runs the backtest as a subprocess and streams
    stdout/stderr in real-time. NEVER times out.
    """
    job_id = f"bt_{uuid.uuid4().hex[:8]}"
    
    # Create output directory
    output_dir = project_root / "data" / "history" / job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create job spec
    job = BacktestJob(
        job_id=job_id,
        symbol=request.symbols[0],
        start_date=request.start,
        end_date=request.end,
        config_path=request.config,
        initial_capital=request.initial_capital,
        output_dir=str(output_dir)
    )
    
    # Submit async task - returns immediately
    await submit_backtest_subprocess(job)
    
    return {
        "job_id": job_id,
        "status": "running",
        "stream_url": f"/stream/backtest/{job_id}"
    }

@app.post("/jobs/submit/training")
async def submit_training_job(request: TrainingRequest):
    """Submit a training job"""
    job_id = f"train_{uuid.uuid4().hex[:8]}"
    
    job_data = {
        "type": "training",
        "job_id": job_id,
        "created_at": datetime.now().isoformat(),
        "run_id": request.run_id,
        "symbol": request.symbol,
        "start": request.start,
        "end": request.end
    }
    
    # Write to jobs/pending directory
    pending_dir = project_root / "jobs" / "pending"
    pending_dir.mkdir(parents=True, exist_ok=True)
    job_file = pending_dir / f"{job_id}.json"
    with open(job_file, "w") as f:
        json.dump(job_data, f, indent=2)
        
    return {"job_id": job_id, "status": "queued"}

@app.get("/api/ohlcv")
async def get_ohlcv_data(
    symbol: str,
    from_date: str = None,
    to_date: str = None,
    timeframe: str = "M15"
):
    """Fetch OHLCV data for charts"""
    try:
        # Support both 'from_date/to_date' and 'from/to' query params
        start_dt = pd.to_datetime(from_date).tz_localize("UTC")
        end_dt = pd.to_datetime(to_date).tz_localize("UTC")
        
        sym = Symbol(symbol)
        tf = Timeframe(timeframe)
        
        df = data_store.load_ohlcv(sym, tf, start_dt, end_dt)
        
        # Limit to 5000 candles
        if len(df) > 5000:
            step = len(df) // 5000 + 1
            df = df.iloc[::step]
            
        result = []
        for idx, row in df.iterrows():
            # LightweightCharts expects Unix timestamp in seconds
            result.append({
                "time": int(idx.timestamp()),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"])
            })
            
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/backtests/history")
async def get_backtest_history(limit: int = 100):
    """Get backtest history"""
    try:
        runs = list_backtest_runs(limit=limit)
        runs_data = []
        for r in runs:
            run_dict = r.__dict__.copy()
            
            # Try to load actual summary.json for accurate P&L
            summary_file = project_root / "data" / "history" / r.run_id / "summary.json"
            if summary_file.exists():
                try:
                    with open(summary_file, 'r') as f:
                        summary = json.load(f)
                    initial_cap = summary.get("initial_capital", 10000)
                    final_eq = summary.get("final_equity", initial_cap)
                    total_pnl = final_eq - initial_cap
                    
                    run_dict['performance'] = {
                        'total_trades': summary.get('total_trades', 0),
                        'winrate': summary.get('winrate', 0),
                        'total_pnl': total_pnl,
                        'sharpe_ratio': summary.get('sharpe_ratio', 0),
                        'max_drawdown': summary.get('max_drawdown', 0),
                        'profit_factor': summary.get('profit_factor', 0)
                    }
                except Exception:
                    pass
            
            # Fallback: ensure 'performance' exists from 'metrics' if needed
            if 'metrics' in run_dict and run_dict.get('performance') is None:
                run_dict['performance'] = run_dict.pop('metrics')
            
            runs_data.append(run_dict)
        return {"runs": runs_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/backtests/run/{run_id}")
async def get_run_details(run_id: str):
    """Get run details"""
    run = get_backtest_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run.__dict__

@app.get("/backtests/run/{run_id}/trades")
async def get_run_trades(run_id: str):
    """Get trades for a run"""
    trades_file = project_root / "data" / "history" / run_id / "trades.csv"
    if not trades_file.exists():
        # Try legacy path
        trades_file = project_root / "backtest_results" / f"trades_{run_id}.csv"
        
    if trades_file.exists():
        df = pd.read_csv(trades_file)
        return df.to_dict(orient="records")
    return []

@app.get("/backtests/run/{run_id}/equity")
async def get_run_equity(run_id: str):
    """Get equity curve for a run"""
    equity_file = project_root / "data" / "history" / run_id / "equity.json"
    if equity_file.exists():
        with open(equity_file, 'r') as f:
            return json.load(f)
    return []

@app.get("/agent/mistakes")
async def get_agent_mistakes():
    """Get mistake library"""
    mistakes_file = project_root / "models" / "mistake_library.json"
    if mistakes_file.exists():
        with open(mistakes_file, 'r') as f:
            return json.load(f)
    return []

@app.get("/agent/edges")
async def get_agent_edges():
    """Get edge library"""
    edges_file = project_root / "models" / "edge_library.json"
    if edges_file.exists():
        with open(edges_file, 'r') as f:
            return json.load(f)
    return []

@app.get("/training/{training_id}/status")
async def get_training_status(training_id: str):
    """Get training job status"""
    # Check if in history (completed)
    run = get_training_run(training_id)
    if run:
        return {"status": "completed", "run": run.__dict__}
    
    # Check if pending
    pending_file = project_root / "jobs" / "pending" / f"{training_id}.json"
    if pending_file.exists():
        return {"status": "pending"}
        
    return {"status": "unknown"}

@app.get("/training/history")
async def get_training_history(limit: int = 50):
    """Get training history"""
    try:
        runs = list_training_runs(limit=limit)
        return {"runs": [r.__dict__ for r in runs]}
    except Exception as e:
        # If training history module fails or file doesn't exist, return empty
        print(f"Error fetching training history: {e}")
        return {"runs": []}

@app.get("/backtests/run/{run_id}/regimes")
async def get_run_regimes(run_id: str):
    """Get regime timeline for a run"""
    regime_file = project_root / "data" / "history" / run_id / "regimes.json"
    if regime_file.exists():
        with open(regime_file, 'r') as f:
            return json.load(f)
    return []

@app.get("/backtests/run/{run_id}/metajudge")
async def get_run_metajudge(run_id: str):
    """Get MetaJudge timeline for a run"""
    meta_file = project_root / "data" / "history" / run_id / "metajudge.json"
    if meta_file.exists():
        with open(meta_file, 'r') as f:
            return json.load(f)
    return []

@app.get("/backtests/run/{run_id}/download")
async def download_run_data(run_id: str):
    """Download full run data"""
    from fastapi.responses import FileResponse
    
    # For now, just return the run_data.json
    # In future, could zip the whole folder
    run_file = project_root / "data" / "history" / run_id / "run_data.json"
    if run_file.exists():
        return FileResponse(
            path=run_file, 
            filename=f"run_{run_id}.json",
            media_type='application/json'
        )
    raise HTTPException(status_code=404, detail="Run data not found")

@app.get("/training/run/{training_id}")
async def get_training_run_details(training_id: str):
    """Get training run details"""
    run = get_training_run(training_id)
    if not run:
        raise HTTPException(status_code=404, detail="Training run not found")
    return run.__dict__

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
