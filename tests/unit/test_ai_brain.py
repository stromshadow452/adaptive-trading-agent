from pathlib import Path

from src.ml.ai_brain import AIBrain
from src.ml.trade_analyzer import analyze_trades


def test_ai_brain_records_entry_and_exit(tmp_path):
    journal = tmp_path / "trades.jsonl"
    brain = AIBrain(enabled=True, journal_path=str(journal))

    signal = {
        "symbol": "EURUSD",
        "side": "buy",
        "strategy": "tokyo_session_mr",
        "regime": "RANGE",
        "price": 1.1000,
        "size": 0.10,
        "sl": 1.0980,
        "tp": 1.1040,
        "confidence": 0.66,
        "metadata": {"atr": 0.0012, "boll_z": -1.4},
    }
    fill = {
        "symbol": "EURUSD",
        "side": "buy",
        "strategy": "tokyo_session_mr",
        "regime": "RANGE",
        "fill_px": 1.1001,
        "size": 0.10,
        "sl": 1.0980,
        "tp": 1.1040,
        "confidence": 0.66,
        "filled_at": "2026-04-25T10:00:00+00:00",
        "metadata": {"atr": 0.0012, "boll_z": -1.4},
    }

    brain.record_entry(signal, fill)
    brain.record_exit({
        **fill,
        "close_px": 1.1040,
        "close_reason": "tp_hit",
        "pnl_usd": 39.0,
        "r_multiple": 1.9,
        "hold_minutes": 75,
    })

    rows = list(brain.journal.read_events())
    assert [r["event"] for r in rows] == ["TRADE_ENTRY", "TRADE_CLOSED"]
    assert rows[1]["is_win"] is True
    assert rows[1]["context"]["boll_z"] == -1.4


def test_analyzer_flags_weak_confidence_bucket():
    rows = []
    for i in range(20):
        rows.append({
            "event": "TRADE_CLOSED",
            "symbol": "EURUSD",
            "strategy": "tokyo_session_mr",
            "regime": "RANGE",
            "side": "buy",
            "confidence": 0.42,
            "pnl": -10.0,
            "r_multiple": -1.0,
            "exit_reason": "sl_hit",
            "duration_minutes": 20,
        })

    analysis = analyze_trades(rows)

    assert analysis["summary"]["total_trades"] == 20
    assert analysis["summary"]["win_rate"] == 0.0
    assert any(m["type"] == "confidence_bucket" for m in analysis["mistakes"])


def test_report_generation(tmp_path):
    brain = AIBrain(
        enabled=True,
        journal_path=str(tmp_path / "trades.jsonl"),
        report_dir=str(tmp_path / "reports"),
    )
    path = brain.generate_report()

    assert isinstance(path, Path)
    assert path.exists()
    assert "AI Brain Report" in path.read_text(encoding="utf-8")
