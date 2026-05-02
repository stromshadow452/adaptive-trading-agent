import os
import pytest
from datetime import datetime, timedelta, timezone
from src.risk.risk_engine import RealTimeRiskEngine

@pytest.fixture
def risk_engine():
    # Use a temporary state file
    engine = RealTimeRiskEngine(state_file="logs/test_risk_state.json")
    # Clear state for testing
    engine.state["last_50_trades"] = []
    engine.state["total_pnl"] = 0.0
    engine.state["loss_streak"] = 0
    engine.state["sym_loss_streak"] = {}
    engine.state["sym_cooldowns"] = {}
    engine.state["regime_cooldowns"] = {}
    engine.state["global_cooldown_until"] = None
    engine.state["slippage_ema"] = {}
    engine.state["slippage_strikes"] = {}
    engine.state["trend_hold_times"] = []
    engine.state["consecutive_trend_exceeds"] = 0
    
    yield engine
    if os.path.exists("logs/test_risk_state.json"):
        os.remove("logs/test_risk_state.json")

def test_global_circuit_breaker(risk_engine):
    ts = datetime.now(timezone.utc)
    # Simulate a massive loss to breach -3.5%
    # Initial equity = 10000, 3.5% = 350
    risk_engine.update_post_trade({"symbol": "EURUSD", "pnl_usd": -400.0, "ts": ts.isoformat()})
    
    allow, size, reason = risk_engine.evaluate_pre_trade({"symbol": "GBPUSD"}, {}, ts + timedelta(minutes=5))
    assert allow == False
    assert "circuit_breaker" in reason

def test_loss_streak_protection(risk_engine):
    ts = datetime.now(timezone.utc)
    # Generate 10 small losses to drop PF < 0.8
    for i in range(10):
        risk_engine.update_post_trade({"symbol": "EURUSD", "pnl_usd": -10.0, "ts": ts.isoformat()})
    
    # 10 losses means loss_streak >= 5 and PF < 0.8.
    allow, size, reason = risk_engine.evaluate_pre_trade({"symbol": "GBPUSD"}, {}, ts + timedelta(minutes=5))
    assert allow == True
    assert size == 0.5

def test_symbol_cooldown(risk_engine):
    ts = datetime.now(timezone.utc)
    for i in range(3):
        risk_engine.update_post_trade({"symbol": "AUDUSD", "pnl_usd": -10.0, "ts": ts.isoformat()})
    
    # Should block AUDUSD
    allow, size, reason = risk_engine.evaluate_pre_trade({"symbol": "AUDUSD"}, {}, ts + timedelta(minutes=5))
    assert allow == False
    assert "symbol_cooldown" in reason

    # Should allow EURUSD
    allow_eur, _, _ = risk_engine.evaluate_pre_trade({"symbol": "EURUSD"}, {}, ts + timedelta(minutes=5))
    assert allow_eur == True

def test_correlation_control(risk_engine):
    ts = datetime.now(timezone.utc)
    open_positions = {
        "EURUSD": {},
        "GBPUSD": {}
    }
    # Trying to add AUDUSD (correlates with USD). 
    # Open positions already have 2 USD correlated trades.
    # The new trade will be the 3rd. So size should be 0.5x.
    allow, size, reason = risk_engine.evaluate_pre_trade({"symbol": "AUDUSD"}, open_positions, ts)
    assert allow == True
    assert size == 0.7

    # If 3 are already open, it just sets size to 0.5 and allows it
    open_positions["NZDUSD"] = {}
    allow, size, reason = risk_engine.evaluate_pre_trade({"symbol": "USDCAD"}, open_positions, ts)
    assert allow == True
    assert size == 0.5

def test_rolling_pf_monitor(risk_engine):
    ts = datetime.now(timezone.utc)
    # Add 30 trades with PF ~ 0.5
    for i in range(25):
        risk_engine.update_post_trade({"symbol": "EURUSD", "pnl_usd": -10.0, "ts": ts.isoformat()})
    for i in range(5):
        risk_engine.update_post_trade({"symbol": "EURUSD", "pnl_usd": 5.0, "ts": ts.isoformat()})
        
    allow, size, reason = risk_engine.evaluate_pre_trade({"symbol": "GBPUSD"}, {}, ts)
    # PF(15) is very low, PF(30) is very low. Should reduce size by 50%.
    assert allow == True
    assert size == 0.5

def test_spread_does_not_block(risk_engine):
    ts = datetime.now(timezone.utc)
    # Spread filter was removed — spread is handled in the PaperExecutor.
    # Verify that a signal with high spread metadata is NOT blocked.
    signal = {
        "symbol": "EURUSD",
        "metadata": {
            "atr": 0.0020,
            "spread": 0.0005
        }
    }
    allow, size, reason = risk_engine.evaluate_pre_trade(signal, {}, ts)
    assert allow == True

def test_regime_failure_detection(risk_engine):
    ts = datetime.now(timezone.utc)
    # Need 10 trades to establish strong median hold time. Let's make median hold time = 30 minutes.
    for i in range(10):
        risk_engine.update_post_trade({
            "symbol": "EURUSD", "pnl_usd": 10.0, "ts": ts.isoformat(),
            "regime": "TREND", "hold_minutes": 30.0
        })
    
    # 3 consecutive trades > 90 minutes.
    for i in range(5):
        risk_engine.update_post_trade({
            "symbol": "EURUSD", "pnl_usd": -5.0, "ts": ts.isoformat(),
            "regime": "TREND", "hold_minutes": 100.0
        })
        
    allow, size, reason = risk_engine.evaluate_pre_trade({"symbol": "GBPUSD", "regime": "TREND"}, {}, ts)
    assert allow == False
    assert "trend_regime_failure_trigger" in reason
