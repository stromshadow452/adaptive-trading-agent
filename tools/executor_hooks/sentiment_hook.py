# FILE: tools/executor_hooks/sentiment_hook.py
from __future__ import annotations

from typing import Any, Dict, TypedDict, cast

# Full-stage implementation lives in sentiment_brain.py
from sentiment_brain import process as sentiment_process


class SentimentStageIn(TypedDict, total=False):
    stage_id: str
    timestamp_utc: str
    market_data_id: str
    text_payload: str
    meta: Dict[str, str]
    previous_votes: Dict[str, float]


class SentimentStageOut(TypedDict):
    stage_id: str
    timestamp_utc: str
    market_data_id: str
    sentiment_score: float
    confidence: float
    vote_weight: float
    veto: bool
    explain: Dict[str, Any]
    dry_run: bool
    log: Dict[str, str]
    log_details: Dict[str, Any]


def run_sentiment_stage_full(payload: SentimentStageIn, config: Dict[str, Any]) -> SentimentStageOut:
    """
    Full adapter that calls `sentiment_brain.process` and then applies:
      - strict cap on vote_weight
      - deterministic soft smoothing (confidence-aware, score-compressing)
      - empty-text hard-neutral guard (idempotent; bypasses previous_votes & smoothing)

    Contract:
      - Pure function (no globals, no I/O)
      - Deterministic + idempotent (same inputs -> same outputs)
      - Fields preserved as SentimentOutput shape
    """
    # ---- Empty-text hard-neutral (FIRST guard; deterministic) ----
    text = str((payload.get("text_payload") or ""))  # type: ignore[arg-type]
    if not text.strip():
        cap_val = float(config.get("sentiment_vote_cap", 0.35))
        out: Dict[str, Any] = {
            "stage_id": "sentiment_brain",
            "timestamp_utc": payload.get("timestamp_utc", "1970-01-01T00:00:00Z"),
            "market_data_id": payload.get("market_data_id", "UNKNOWN"),
            "sentiment_score": 0.0,
            "confidence": 0.0,
            "vote_weight": 0.0,  # hard zero
            "veto": False,
            "explain": {"top_tokens": [], "rules": ["empty_text_neutral"]},
            "dry_run": bool(config.get("dry_run", True)),
            "log": {"status": "OK", "reason": "empty_text"},
            "log_details": {
                "masked_input": {
                    "stage_id": "sentiment_brain",
                    "timestamp_utc": payload.get("timestamp_utc", "1970-01-01T00:00:00Z"),
                    "market_data_id": payload.get("market_data_id", "UNKNOWN"),
                    "text_payload": "",
                    "meta": {
                        "lang": (payload.get("meta", {}) or {}).get("lang", "en"),
                        "source": (payload.get("meta", {}) or {}).get("source", "stub"),
                    },
                },
                "model_version": str(config.get("model_version", "sentiment-lexical-stub-v1")),
                "remote_model_used": False,
                "runtime_ms": 0.0,
                "seed": int(config.get("seed", 2025)),
                "vote_weight_calc": "neutral_on_empty_text",
                "vote_weight_post_smooth": 0.0,
                "cap": cap_val,
            },
        }
        return cast(SentimentStageOut, out)

    # ---- Normal path: call core brain deterministically ----
    feature_flags = cast(Dict[str, Any], config.get("feature_flags", {}))
    enable_flag = bool(feature_flags.get("enable_sentiment", False))

    cfg: Dict[str, Any] = {
        "feature_flags": {"enable_sentiment": enable_flag},
        "seed": int(config.get("seed", 2025)),
        "model_version": str(config.get("model_version", "sentiment-lexical-stub-v1")),
        "dry_run": bool(config.get("dry_run", False)) or (not enable_flag),
        "use_remote_model": bool(config.get("use_remote_model", False)),
    }

    # Call the pure processor with a defensive copy
    out = cast(Dict[str, Any], sentiment_process(dict(payload), cfg))

    # ---- Strict cap (pre-smoothing) ----
    cap = float(config.get("sentiment_vote_cap", 0.35))
    vw_raw = float(out.get("vote_weight", 0.0))
    out["vote_weight"] = max(0.0, min(cap, vw_raw))

    # Add telemetry fields (additive)
    ld = out.setdefault("log_details", {})
    ld["remote_model_used"] = bool(cfg["use_remote_model"])
    ld["remote_model_note"] = "offline fallback to classifier_stub" if cfg["use_remote_model"] else "local stub"

    # ---- Deterministic soft smoothing + strict cap (no external state) ----
    conf = float(out.get("confidence", 0.0))
    score = float(out.get("sentiment_score", 0.0))
    vw = float(out.get("vote_weight", 0.0))

    # Confidence-aware scaling (0.5..1.0) and soft compression by |score|
    soft = vw * (0.5 + 0.5 * conf)
    soft = soft / (1.0 + 0.5 * abs(score))

    vw_final = max(0.0, min(cap, round(soft, 6)))
    out["vote_weight"] = vw_final
    ld["vote_weight_post_smooth"] = vw_final
    ld["cap"] = cap

    return cast(SentimentStageOut, out)


def run_sentiment_stage(inp: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Lightweight deterministic adapter (vote-only form).
      - No I/O, no globals.
      - Inputs are optional: news_score, tw_score, quant_sent_score in [-1, 1].
      - Returns: {'stage':'sentiment_brain','enabled':bool,'vote':float,'meta':{...}}
    """
    ff = cast(Dict[str, Any], cfg.get("feature_flags", {})) if isinstance(cfg, dict) else {}
    enabled = bool(ff.get("enable_sentiment", cfg.get("enable_sentiment", False)))
    if not enabled:
        return {"stage": "sentiment_brain", "enabled": False, "vote": 0.0, "meta": {"mode": "disabled"}}

    # Expect typed input; do not mutate.
    news_score = float(inp.get("news_score", 0.0))           # [-1, 1]
    tw_score = float(inp.get("tw_score", 0.0))               # [-1, 1]
    qscore = float(inp.get("quant_sent_score", 0.0))         # [-1, 1]

    cap = float(cfg.get("sentiment_vote_cap", 0.35))
    raw = 0.5 * news_score + 0.3 * tw_score + 0.2 * qscore
    vote = max(min(raw, cap), -cap)  # symmetric cap; caller may ignore negatives

    return {
        "stage": "sentiment_brain",
        "enabled": True,
        "vote": float(vote),
        "meta": {"cap": cap, "inputs": {"news": news_score, "tw": tw_score, "q": qscore}},
    }
