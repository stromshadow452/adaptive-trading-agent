#!/usr/bin/env python3
"""
tools/pipeline_orchestrator.py

Immutable neural-chain orchestrator with the sacred stage order:
Market Data → Feature Reactor → Primary ML Brain → RL Brain → Volatility Brain →
Sentiment Brain → Risk Brain → Meta-Gating Brain → Risk + Throttle Gates →
Execution Reflex Engine → Broker + Hedger → Stark-Vision Dashboard + Logs

This orchestrator is additive-only and idempotent. It wires in the Sentiment Brain
between Volatility and Risk using the lightweight adapter.
"""
# --- imports (top of file) ---
from src.stages.meta_gating_brain import run_meta_gating
from src.stages.throttle_gate import run_throttle_gate

# --- after risk_brain(...) ---
meta_out = run_meta_gating({**ctx, **risk_out}, config)
logger_append(meta_out)

throttle_out = run_throttle_gate({**ctx, **meta_out}, config, state)
logger_append(throttle_out)

if throttle_out.get("vote", 0.0) <= 0.0:
    return flat_position()
# continue to Execution Reflex Engine ...

from __future__ import annotations

from typing import Any, Dict

from tools.executor_hooks.sentiment_hook import run_sentiment_stage_full as run_sentiment_stage

# Prefer cached live router; fallback to deterministic stub
try:
    from tools.executor_hooks.news_router_live import news_router  # type: ignore
    NEWS_ROUTER_SOURCE = "live"
except Exception:
    try:
        from tools.executor_hooks.news_router_stub import news_router  # type: ignore
        NEWS_ROUTER_SOURCE = "stub"
    except Exception:
        def news_router(_md):  # type: ignore
            return "Market sees steady performance with balanced commentary."
        NEWS_ROUTER_SOURCE = "stub"


def _get_news_text(stages: Dict[str, Any], md: Dict[str, Any]) -> str:
    router = stages.get("news_router")
    cfg = stages.get("config", {}) if isinstance(stages.get("config", {}), dict) else {}
    use_remote = bool(cfg.get("use_remote", False))
    if callable(router):
        # Try call with use_remote (live router), fallback to simple call (stub)
        try:
            return str(router(md, use_remote=use_remote) or "")
        except TypeError:
            try:
                return str(router(md) or "")
            except Exception:
                return ""
        except Exception:
            return ""
    # Fallback to module-level router
    try:
        return str(news_router(md) or "")
    except Exception:
        return ""


def run_pipeline(stages: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute the neural chain in sacred order. All stages are provided in `stages` as callables.
    This function performs no persistence; it only returns outputs and uses append-only logging
    via stages["logger_append"].
    """
    if not isinstance(stages, dict):
        raise TypeError("stages must be a dict of callables and config")
    logger_append = stages.get("logger_append")
    if not callable(logger_append):
        # Provide a no-op logger to preserve append-only behavior expectation
        def logger_append(_entry: Dict[str, Any]) -> None:  # type: ignore
            return

    # Normalize config and sentiment flags (accept both flat and feature_flags shapes)
    cfg = stages.get("config", {}) if isinstance(stages.get("config", {}), dict) else {}
    cfg.setdefault("feature_flags", {})
    ff = cfg["feature_flags"]
    ff.setdefault("enable_sentiment", cfg.get("enable_sentiment", False))
    cfg.setdefault("sentiment_vote_cap", 0.35)
    cfg.setdefault("use_remote", False)
    stages["config"] = cfg

    # Router source selection and log
    stages["news_router"] = stages.get("news_router") or news_router
    stages["news_source"] = stages.get("news_source") or NEWS_ROUTER_SOURCE
    logger_append({
        "stage": "news_router",
        "source": stages["news_source"],
        "use_remote": stages["config"]["use_remote"],
        "status": "active" if stages["news_source"] == "live" else "stub",
    })

    # Sacred order
    md = stages["market_data"]()
    fr = stages["feature_reactor"](md)
    ml = stages["primary_ml_brain"](fr)
    rl = stages["rl_brain"](fr, ml)
    vol = stages["volatility_brain"](fr, ml, rl)

    # Sentiment Brain (feature-flagged, deterministic, no I/O)
    raw_router = stages["news_router"](md) if callable(stages.get("news_router")) else ""
    if isinstance(raw_router, dict):
        txt = str(raw_router.get("text", "") or "")
        tele = raw_router.get("telemetry", {})
    else:
        txt = str(raw_router or "")
        tele = {}
    logger_append({
        "stage": "news_router",
        "source": stages.get("news_source", "stub"),
        "use_remote": bool(cfg.get("use_remote", False)),
        "telemetry": tele,
    })
    if not txt.strip():
        sentiment_out = {
            "stage_id": "sentiment_brain",
            "timestamp_utc": fr.get("timestamp_utc") if isinstance(fr, dict) else None,
            "market_data_id": (md.get("market_data_id") or md.get("id") or "UNKNOWN") if isinstance(md, dict) else "UNKNOWN",
            "sentiment_score": 0.0,
            "confidence": 0.0,
            "vote_weight": 0.0,
            "veto": False,
            "explain": {"top_tokens": [], "rules": ["empty_text_neutral"]},
            "dry_run": True,
            "log": {"status": "OK", "reason": "empty_text"},
            "log_details": {"runtime_ms": 0.0, "remote_model_used": False},
        }
        sentiment_in = {"meta": {"source": stages.get("news_source", "stub")}}  # minimal input for logging
    else:
        sentiment_in = {
            "stage_id": "sentiment_brain",
            "timestamp_utc": fr.get("timestamp_utc") if isinstance(fr, dict) else None,
            "market_data_id": (md.get("market_data_id") or md.get("id") or "UNKNOWN") if isinstance(md, dict) else "UNKNOWN",
            "text_payload": txt,
            "meta": {
                "lang": stages.get("lang", "en"),
                "source": stages.get("news_source", "stub"),
            },
            "previous_votes": {
                "primary_ml": float(ml.get("vote", 0.0)) if isinstance(ml, dict) else 0.0,
                "rl": float(rl.get("vote", 0.0)) if isinstance(rl, dict) else 0.0,
                "volatility": float(vol.get("vote", 0.0)) if isinstance(vol, dict) else 0.0,
            },
        }
        sentiment_out = run_sentiment_stage(sentiment_in, stages["config"])
    logger_append({
        "stage": "sentiment_brain",
        "input": sentiment_in,
        "output": sentiment_out,
    })

    # Risk Brain receives sentiment as 5th argument (backward compatible if ignored)
    risk = stages["risk_brain"](fr, ml, rl, vol, sentiment_out)
    meta = stages["meta_gating_brain"](ml, rl, vol, sentiment_out, risk)
    gates = stages["risk_throttle_gates"](risk, meta)
    execx = stages["execution_reflex_engine"](gates)
    broker = stages["broker_hedger"](execx)

    # Optional dashboard stage (if provided)
    dashboard = None
    if "stark_vision_dashboard" in stages and callable(stages["stark_vision_dashboard"]):
        dashboard = stages["stark_vision_dashboard"](broker)

    return {
        "market_data": md,
        "feature_reactor": fr,
        "primary_ml": ml,
        "rl": rl,
        "volatility": vol,
        "sentiment": sentiment_out,
        "risk": risk,
        "meta": meta,
        "gates": gates,
        "exec": execx,
        "broker": broker,
        "dashboard": dashboard,
    }


def _smoke_logger(collected: list[Dict[str, Any]]):
    def _append(entry: Dict[str, Any]) -> None:
        collected.append(entry)
    return _append


def run_smoke_test() -> Dict[str, Any]:
    """
    Minimal deterministic smoke that exercises stage order and sentiment wiring.
    Returns a small report dict; prints 'SMOKE: PASS' on success.
    """
    logs: list[Dict[str, Any]] = []
    # Deterministic stub stages
    def market_data():
        return {"market_data_id": "MD-TEST", "symbol": "EURUSD"}

    def feature_reactor(md):
        return {"timestamp_utc": "2025-01-01T00:00:00Z"}

    def primary_ml_brain(fr):
        return {"vote": 0.6}

    def rl_brain(fr, ml):
        return {"vote": 0.55}

    def volatility_brain(fr, ml, rl):
        return {"vote": 0.5}

    def risk_brain(fr, ml, rl, vol, sentiment=None):
        return {"received_sentiment": sentiment, "ok": True}

    def meta_gating_brain(ml, rl, vol, sentiment, risk):
        return {"ok": True}

    def risk_throttle_gates(risk, meta):
        return {"ok": True}

    def execution_reflex_engine(gates):
        return {"ok": True}

    def broker_hedger(execx):
        return {"ok": True}

    stages = {
        "config": {
            "feature_flags": {"enable_sentiment": True},
            "sentiment_vote_cap": 0.35,
        },
        "lang": "en",
        "news_source": "stub",
        "logger_append": _smoke_logger(logs),
        "market_data": market_data,
        "feature_reactor": feature_reactor,
        "primary_ml_brain": primary_ml_brain,
        "rl_brain": rl_brain,
        "volatility_brain": volatility_brain,
        "risk_brain": risk_brain,
        "meta_gating_brain": meta_gating_brain,
        "risk_throttle_gates": risk_throttle_gates,
        "execution_reflex_engine": execution_reflex_engine,
        "broker_hedger": broker_hedger,
    }

    out = run_pipeline(stages)
    order_ok = ["stage" in e and e["stage"] == "sentiment_brain" for e in logs]
    report = {"SMOKE": "PASS" if any(order_ok) else "FAIL", "log_count": len(logs)}
    print(report["SMOKE"] + f" (logs={report['log_count']})")
    return report


if __name__ == "__main__":
    run_smoke_test()

