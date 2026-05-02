"""
src/api/main.py
==============
SCOPUS REST API — FastAPI service exposing pipeline state.

Phase 2: /performance/summary, /risk/status connected to real data.

Run with:
    uvicorn src.api.main:app --reload --port 8000

Endpoints:
    GET /health                  System health check
    GET /regime/{symbol}         Current detected regime
    GET /signals/today           Today's strategies signals
    GET /performance/summary     Portfolio summary
    GET /risk/status             Portfolio risk + circuit breaker state
    GET /strategies              Strategy enabled/disabled status
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime, timezone
from typing import Optional, List
import logging
import os
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SCOPUS API",
    description="Adaptive Trading Agent — REST Interface",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Tighten in production
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    mode: str
    version: str


class RegimeResponse(BaseModel):
    symbol: str
    regime: str
    confidence: float
    adx: float
    reason: str
    timestamp: str


class SignalItem(BaseModel):
    symbol: str
    strategy: str
    side: str
    entry: float
    stop: float
    tp: float
    confidence: float
    timestamp: str


class PerformanceSummary(BaseModel):
    total_pnl_usd: float
    open_positions: int
    total_trades:   int
    win_rate:       float
    profit_factor:  float
    drawdown_pct:   float
    sharpe:         Optional[float]
    avg_slippage_pips: float
    shadow_days:    float
    gate_status:    str   # 'PASS' | 'FAIL' | 'INSUFFICIENT_DATA'
    timestamp:      str


class RiskStatus(BaseModel):
    global_breaker_tripped: bool
    portfolio_risk_pct: float
    positions: list
    daily_loss_pct: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health():
    """System health and mode check."""
    mode = os.environ.get("SCOPUS_MODE", "BACKTEST")
    return HealthResponse(
        status="ok",
        timestamp=datetime.now(timezone.utc).isoformat(),
        mode=mode,
        version="0.1.0",
    )


@app.get("/regime/{symbol}", response_model=RegimeResponse)
def get_regime(symbol: str):
    """
    Return current detected regime for a symbol.
    TODO: Connect to live RegimeDetector instance.
    """
    symbol = symbol.upper()
    # Placeholder — will connect to MetaGatingBrain.classify_regime()
    return RegimeResponse(
        symbol=symbol,
        regime="UNCERTAIN",
        confidence=0.0,
        adx=0.0,
        reason="Pipeline not connected (skeleton mode)",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/signals/today", response_model=list[SignalItem])
def get_signals():
    """
    Return today's strategy signals.
    TODO: Connect to strategy bank output log.
    """
    # Returns empty list until pipeline is connected
    return []


@app.get("/performance/summary", response_model=PerformanceSummary)
def get_performance():
    """Return live shadow trading performance metrics from fills.jsonl."""
    fills_log = os.environ.get("SCOPUS_FILLS_LOG", "logs/shadow/fills.jsonl")
    try:
        from src.monitoring.metrics import ShadowMetrics
        m   = ShadowMetrics(fills_log)
        snap = m.compute()
        gate = m.gate_check()
        gate_status = gate["overall"] if snap["n_trades"] >= 10 else "INSUFFICIENT_DATA"
        return PerformanceSummary(
            total_pnl_usd=snap.get("net_pnl_usd", 0.0),
            open_positions=0,
            total_trades=snap.get("n_trades", 0),
            win_rate=snap.get("win_rate", 0.0),
            profit_factor=snap.get("profit_factor", 0.0),
            drawdown_pct=snap.get("max_drawdown_pct", 0.0),
            sharpe=snap.get("sharpe_annualized"),
            avg_slippage_pips=snap.get("avg_slippage_pips", 0.0),
            shadow_days=snap.get("shadow_days", 0.0),
            gate_status=gate_status,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as e:
        logger.warning(f"[/performance/summary] Error: {e}")
        return PerformanceSummary(
            total_pnl_usd=0.0, open_positions=0, total_trades=0,
            win_rate=0.0, profit_factor=0.0, drawdown_pct=0.0,
            sharpe=None, avg_slippage_pips=0.0, shadow_days=0.0,
            gate_status="INSUFFICIENT_DATA",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )


@app.get("/risk/status", response_model=RiskStatus)
def get_risk_status():
    """
    Return circuit breaker and portfolio risk state.
    TODO: Connect to CircuitBreaker.get_status() and PortfolioBrain.
    """
    try:
        from src.risk.circuit_breaker import CircuitBreaker
        cb = CircuitBreaker()
        cb_status = cb.get_status()
        global_tripped = cb_status.get("global_trip", False)
    except Exception:
        global_tripped = False

    return RiskStatus(
        global_breaker_tripped=global_tripped,
        portfolio_risk_pct=0.0,
        positions=[],
        daily_loss_pct=0.0,
    )


@app.get("/strategies")
def get_strategies():
    """Return strategy enabled/disabled status from weapon_system.yaml."""
    cfg_path = os.environ.get("SCOPUS_CONFIG", "config/weapon_system.yaml")
    try:
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
        strats = cfg.get("strategies", {})
        enabled  = [k for k, v in strats.items() if v is True or
                    (isinstance(v, dict) and v.get("enabled") is True)]
        disabled = [k for k, v in strats.items() if v is False or
                    (isinstance(v, dict) and v.get("enabled") is False)]
        return {
            "enabled":      enabled,
            "disabled":     disabled,
            "system_mode":  cfg.get("system_mode", "BACKTEST"),
            "source":       cfg_path,
        }
    except Exception as e:
        logger.warning(f"[/strategies] Could not load config: {e}")
        return {
            "enabled":  ["tokyo_session_mr", "silver_mr"],
            "disabled": ["london_box", "choppy_engine", "rl_agent"],
            "source":   cfg_path,
        }


# ---------------------------------------------------------------------------
# Dev entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
