"""
SCOPUS Real-time Streaming Module

SSE (Server-Sent Events) endpoint for streaming backtest logs, progress, and results.
Eliminates HTTP timeout issues for long-running backtests (hours/days).

Architecture:
1. Client submits job via POST → gets job_id immediately
2. Client opens SSE connection to /stream/backtest/{job_id}
3. Server streams: LOG, PROGRESS, HEARTBEAT, RESULT, ERROR events
4. Heartbeat every 10s prevents connection timeout
5. Connection stays open until job completes or client disconnects

Author: SCOPUS Team
Date: 2025-12-29
"""

import asyncio
import json
import logging
import traceback
from datetime import datetime
from typing import Dict, Optional, Any, AsyncGenerator
from dataclasses import dataclass, asdict
from enum import Enum

from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)

# ============================================================================
# Event Types & Schemas
# ============================================================================

class EventType(str, Enum):
    LOG = "log"
    PROGRESS = "progress"
    HEARTBEAT = "heartbeat"
    RESULT = "result"
    ERROR = "error"


@dataclass
class LogEvent:
    """Log message event"""
    type: str = "log"
    timestamp: str = ""
    level: str = "INFO"
    message: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat() + "Z"


@dataclass
class ProgressEvent:
    """Progress update event"""
    type: str = "progress"
    current: int = 0
    total: int = 0
    percent: float = 0.0
    symbol: str = ""
    eta_seconds: float = 0.0
    message: str = ""


@dataclass 
class HeartbeatEvent:
    """Keep-alive heartbeat"""
    type: str = "heartbeat"
    timestamp: str = ""
    job_id: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat() + "Z"


@dataclass
class ResultEvent:
    """Final result event"""
    type: str = "result"
    job_id: str = ""
    status: str = "success"  # success | failed
    metrics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}


@dataclass
class ErrorEvent:
    """Error event"""
    type: str = "error"
    message: str = ""
    traceback: str = ""


# ============================================================================
# Job Registry - Tracks active jobs and their queues
# ============================================================================

class JobRegistry:
    """
    Thread-safe registry of active backtest jobs.
    Each job has its own asyncio.Queue for message passing.
    """
    
    def __init__(self):
        self._jobs: Dict[str, asyncio.Queue] = {}
        self._heartbeat_tasks: Dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()
    
    async def register(self, job_id: str) -> asyncio.Queue:
        """Register a new job and return its message queue"""
        async with self._lock:
            if job_id in self._jobs:
                return self._jobs[job_id]
            
            queue = asyncio.Queue()
            self._jobs[job_id] = queue
            logger.info(f"Registered job: {job_id}")
            return queue
    
    async def get_queue(self, job_id: str) -> Optional[asyncio.Queue]:
        """Get the message queue for a job"""
        async with self._lock:
            return self._jobs.get(job_id)
    
    async def unregister(self, job_id: str):
        """Remove job from registry and cleanup"""
        async with self._lock:
            if job_id in self._jobs:
                del self._jobs[job_id]
                logger.info(f"Unregistered job: {job_id}")
            
            if job_id in self._heartbeat_tasks:
                self._heartbeat_tasks[job_id].cancel()
                del self._heartbeat_tasks[job_id]
    
    async def emit(self, job_id: str, event: Any):
        """Emit an event to a job's queue"""
        queue = await self.get_queue(job_id)
        if queue:
            await queue.put(event)
    
    def start_heartbeat(self, job_id: str):
        """Start independent heartbeat task for a job"""
        if job_id not in self._heartbeat_tasks:
            task = asyncio.create_task(self._heartbeat_loop(job_id))
            self._heartbeat_tasks[job_id] = task
    
    async def _heartbeat_loop(self, job_id: str):
        """Send heartbeat every 10 seconds"""
        try:
            while True:
                await asyncio.sleep(10)
                queue = await self.get_queue(job_id)
                if queue:
                    event = HeartbeatEvent(job_id=job_id)
                    await queue.put(event)
                else:
                    break  # Job was unregistered
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Heartbeat error for {job_id}: {e}")


# Global registry instance
job_registry = JobRegistry()


# ============================================================================
# SSE Streaming Endpoint
# ============================================================================

router = APIRouter(tags=["streaming"])


def format_sse(event_type: str, data: dict) -> str:
    """Format data as SSE message"""
    json_data = json.dumps(data)
    return f"event: {event_type}\ndata: {json_data}\n\n"


async def event_generator(job_id: str) -> AsyncGenerator[str, None]:
    """
    Async generator that yields SSE events for a job.
    Runs indefinitely until job completes or client disconnects.
    """
    queue = await job_registry.get_queue(job_id)
    
    if not queue:
        # Job not found - might need to wait for registration
        await asyncio.sleep(1)
        queue = await job_registry.get_queue(job_id)
        
    if not queue:
        yield format_sse("error", {"message": f"Job {job_id} not found"})
        return
    
    # Start heartbeat for this job
    job_registry.start_heartbeat(job_id)
    
    # Send initial connection event
    yield format_sse("connected", {
        "job_id": job_id,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    })
    
    try:
        while True:
            try:
                # Wait for next event with timeout (allows graceful shutdown)
                event = await asyncio.wait_for(queue.get(), timeout=30)
                
                if isinstance(event, (LogEvent, ProgressEvent, HeartbeatEvent, 
                                      ResultEvent, ErrorEvent)):
                    event_dict = asdict(event)
                    event_type = event_dict.pop("type")
                    yield format_sse(event_type, event_dict)
                    
                    # If result or error, end the stream
                    if isinstance(event, (ResultEvent, ErrorEvent)):
                        break
                else:
                    # Handle raw dict events
                    event_type = event.get("type", "log")
                    yield format_sse(event_type, event)
                    
            except asyncio.TimeoutError:
                # No events for 30s - send heartbeat to keep connection alive
                yield format_sse("heartbeat", {
                    "job_id": job_id,
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                })
                
    except asyncio.CancelledError:
        logger.info(f"Client disconnected from job {job_id}")
    except Exception as e:
        logger.error(f"Stream error for {job_id}: {e}")
        yield format_sse("error", {"message": str(e)})
    finally:
        # Don't unregister - job might still be running
        # Client can reconnect
        pass


@router.get("/stream/backtest/{job_id}")
async def stream_backtest(job_id: str):
    """
    SSE endpoint for streaming backtest events.
    
    Events:
    - connected: Initial connection confirmation
    - log: Log messages from backtest
    - progress: Progress updates (current/total candles)
    - heartbeat: Keep-alive signal (every 10s)
    - result: Final backtest results
    - error: Error messages
    
    Usage:
    ```javascript
    const es = new EventSource('/stream/backtest/bt_123456');
    es.addEventListener('log', (e) => console.log(JSON.parse(e.data)));
    es.addEventListener('progress', (e) => updateProgress(JSON.parse(e.data)));
    es.addEventListener('result', (e) => showResults(JSON.parse(e.data)));
    ```
    """
    return StreamingResponse(
        event_generator(job_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )


# ============================================================================
# Helper Functions for Emitting Events
# ============================================================================

async def emit_log(job_id: str, message: str, level: str = "INFO"):
    """Emit a log event"""
    event = LogEvent(level=level, message=message)
    await job_registry.emit(job_id, event)


async def emit_progress(job_id: str, current: int, total: int, symbol: str = "", 
                        eta_seconds: float = 0.0, message: str = ""):
    """Emit a progress event"""
    percent = (current / total * 100) if total > 0 else 0
    event = ProgressEvent(
        current=current,
        total=total,
        percent=percent,
        symbol=symbol,
        eta_seconds=eta_seconds,
        message=message
    )
    await job_registry.emit(job_id, event)


async def emit_result(job_id: str, metrics: Dict[str, Any], status: str = "success"):
    """Emit the final result event"""
    event = ResultEvent(job_id=job_id, status=status, metrics=metrics)
    await job_registry.emit(job_id, event)
    # Cleanup after result
    await asyncio.sleep(1)  # Give time for client to receive
    await job_registry.unregister(job_id)


async def emit_error(job_id: str, message: str, tb: str = ""):
    """Emit an error event"""
    event = ErrorEvent(message=message, traceback=tb)
    await job_registry.emit(job_id, event)
    # Cleanup after error
    await asyncio.sleep(1)
    await job_registry.unregister(job_id)


# ============================================================================
# Test Endpoint
# ============================================================================

@router.get("/stream/test/{job_id}")
async def test_stream(job_id: str):
    """Test SSE endpoint - simulates a backtest with fake events"""
    
    async def test_generator():
        yield format_sse("connected", {"job_id": job_id})
        
        for i in range(10):
            await asyncio.sleep(1)
            yield format_sse("log", {
                "level": "INFO",
                "message": f"Processing step {i+1}/10",
                "timestamp": datetime.utcnow().isoformat() + "Z"
            })
            yield format_sse("progress", {
                "current": i + 1,
                "total": 10,
                "percent": (i + 1) * 10,
                "symbol": "EURUSD"
            })
        
        yield format_sse("result", {
            "job_id": job_id,
            "status": "success",
            "metrics": {
                "total_trades": 25,
                "winrate": 0.4,
                "total_pnl": 150.0
            }
        })
    
    return StreamingResponse(
        test_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )
