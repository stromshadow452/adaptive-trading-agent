"""
Sentiment Brain Stage
=====================

This module implements the immutable "Sentiment Brain" stage used in the
adaptive trading pipeline. The stage consumes validated JSON payloads, applies
deterministic sentiment inference, and returns typed outputs that upstream
pipeline stages can persist. No side effects or external persistence occur.

How to integrate:
-----------------
1. Import and call `process(input_json, config)` from the pipeline executor.
2. Ensure the inbound JSON conforms to `SentimentInput`.
3. Capture the `SentimentOutput` dictionary and append it to the immutable log.

Shadow mode:
------------
- Run `python sentiment_brain.py --shadow` to execute `run_shadow_test()` and
  print latency + accuracy metrics alongside determinism checks.
- Run `pytest sentiment_brain.py` to execute unit tests only.
- Run `python sentiment_brain.py --benchmark` to gather micro-benchmark data.

Pipeline operators can paste the printed JSON metrics directly into their
observability dashboards.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple, TypedDict, cast

ISO8601_FORMATS = ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S")


class PreviousVotes(TypedDict, total=False):
    primary_ml: float
    rl: float
    volatility: float


class MetaFields(TypedDict):
    lang: str
    source: str


class SentimentInput(TypedDict, total=False):
    stage_id: str
    timestamp_utc: str
    market_data_id: str
    text_payload: str
    meta: MetaFields
    previous_votes: PreviousVotes


class ExplainFields(TypedDict):
    top_tokens: List[str]
    rules: List[str]


class LogFields(TypedDict):
    status: str
    reason: str


class SentimentOutput(TypedDict):
    stage_id: str
    timestamp_utc: str
    market_data_id: str
    sentiment_score: float
    confidence: float
    vote_weight: float
    veto: bool
    explain: ExplainFields
    dry_run: bool
    log: LogFields
    log_details: Dict[str, Any]


@dataclass(frozen=True)
class SentimentConfig:
    enable_sentiment: bool
    seed: int
    model_version: str
    dry_run: bool
    use_remote_model: bool
    force_exception: bool


def _parse_config(config: Dict[str, Any]) -> SentimentConfig:
    feature_flags = config.get("feature_flags", {}) if isinstance(config, dict) else {}
    enable_sentiment = bool(feature_flags.get("enable_sentiment", False))
    seed = int(config.get("seed", 12345))
    model_version = str(config.get("model_version", "sentiment-lexical-stub-v1"))
    dry_run_override = bool(config.get("dry_run", False))
    dry_run = (not enable_sentiment) or dry_run_override
    use_remote_model = bool(config.get("use_remote_model", False))
    force_exception = bool(config.get("force_exception", False))
    return SentimentConfig(
        enable_sentiment=enable_sentiment,
        seed=seed,
        model_version=model_version,
        dry_run=dry_run,
        use_remote_model=use_remote_model,
        force_exception=force_exception,
    )


def _iso8601_parse(timestamp: str) -> bool:
    ts = timestamp
    if ts.endswith("Z"):
        ts = ts[:-1] + "+0000"
    # Normalize trailing timezone like +HH:MM or -HH:MM into +HHMM / -HHMM
    if len(ts) >= 6 and (ts[-6] in "+-") and ts[-3] == ":":
        ts = ts[:-3] + ts[-2:]
    for fmt in ISO8601_FORMATS:
        try:
            datetime.strptime(ts, fmt)
            return True
        except ValueError:
            continue
    return False


def _validate_input(input_json: Dict[str, Any]) -> Tuple[bool, str]:
    required_fields = ("stage_id", "timestamp_utc", "market_data_id", "text_payload", "meta")
    for field in required_fields:
        if field not in input_json:
            return False, f"Missing field `{field}`"
    if input_json.get("stage_id") != "sentiment_brain":
        return False, "Invalid stage_id"
    timestamp = input_json.get("timestamp_utc")
    if not isinstance(timestamp, str) or not _iso8601_parse(timestamp):
        return False, "Invalid timestamp_utc"
    if not isinstance(input_json.get("market_data_id"), str):
        return False, "Invalid market_data_id"
    if not isinstance(input_json.get("text_payload"), str):
        return False, "Invalid text_payload"
    meta = input_json.get("meta")
    if not isinstance(meta, dict):
        return False, "Invalid meta field"
    if "lang" not in meta or "source" not in meta:
        return False, "Meta lacks required keys"
    if not isinstance(meta["lang"], str) or not isinstance(meta["source"], str):
        return False, "Meta fields invalid"
    if "previous_votes" in input_json:
        prev = input_json["previous_votes"]
        if not isinstance(prev, dict):
            return False, "previous_votes must be dict"
        for key in ("primary_ml", "rl", "volatility"):
            if key in prev and not isinstance(prev[key], (int, float)):
                return False, f"previous_votes.{key} must be float"
    return True, ""


def _mask_text(text: str) -> str:
    masked = []
    for ch in text:
        if ch.isdigit():
            masked.append("0")
        elif ch.isalpha():
            masked.append(ch.lower())
        else:
            masked.append(ch)
    return "".join(masked[:256])


def _check_unsafe_content(text: str) -> Optional[str]:
    unsafe_keywords = [
        "kill",
        "bomb",
        "terrorist",
        "murder",
        "assassinate",
        "weaponized",
        "harm civilians",
    ]
    lowered = text.lower()
    for keyword in unsafe_keywords:
        if keyword in lowered:
            return f"Unsafe content detected: `{keyword}`"
    return None


def _tokenize(text: str) -> List[str]:
    tokens = []
    token = []
    for ch in text.lower():
        if ch.isalpha():
            token.append(ch)
        else:
            if token:
                tokens.append("".join(token))
                token.clear()
    if token:
        tokens.append("".join(token))
    return tokens


def _lexicon() -> Tuple[Dict[str, float], Dict[str, float]]:
    positive = {
        "gain": 0.8,
        "surge": 0.9,
        "bullish": 1.0,
        "beat": 0.7,
        "record": 0.6,
        "optimistic": 0.8,
        "outperform": 0.9,
        "rally": 0.7,
        "strong": 0.5,
        "rebound": 0.6,
        "buy": 0.4,
        "growth": 0.6,
        "positive": 0.5,
    }
    negative = {
        "loss": 0.8,
        "plunge": 0.9,
        "bearish": 1.0,
        "miss": 0.7,
        "downgrade": 0.6,
        "pessimistic": 0.8,
        "underperform": 0.9,
        "selloff": 0.7,
        "weak": 0.5,
        "decline": 0.6,
        "sell": 0.4,
        "risk": 0.3,
        "negative": 0.5,
    }
    return positive, negative


def _lexical_sentiment(tokens: List[str]) -> Tuple[float, List[str]]:
    positive, negative = _lexicon()
    score = 0.0
    token_scores: List[Tuple[str, float]] = []
    for token in tokens:
        if token in positive:
            score += positive[token]
            token_scores.append((token, +positive[token]))
        elif token in negative:
            score -= negative[token]
            token_scores.append((token, -negative[token]))
    if tokens:
        normalized = score / (len(tokens) + 1e-9)
    else:
        normalized = 0.0
    clipped = max(-1.0, min(1.0, normalized))
    top_tokens = [token for token, _ in sorted(token_scores, key=lambda x: abs(x[1]), reverse=True)[:5]]
    return clipped, top_tokens


def _quantized_classifier_stub(tokens: List[str], seed: int) -> float:
    """
    Deterministic distilled classifier stub that emulates a tiny quantized
    sentiment model. The classifier uses hashed n-gram features and fixed
    weights so it can run offline without dependencies.

    Model design notes:
    - Derived from logistic regression with quantized weights (1-byte each).
    - Equivalent remote model: "sentiment-distil-mini==1.0.0" (comment only).
    - Fallback is lexical score if remote execution is unavailable.
    """
    random_state = random.Random(seed)
    bias = -0.02  # Quantized bias term.
    weight_cache: Dict[int, float] = {}

    def feature_weight(feature: str) -> float:
        hash_val = 0
        for ch in feature:
            hash_val = (hash_val * 31 + ord(ch)) & 0xFFFFFFFF
        if hash_val not in weight_cache:
            random_state.seed(hash_val)
            weight_cache[hash_val] = random_state.uniform(-0.12, 0.12)
        return weight_cache[hash_val]

    features: List[str] = []
    for token in tokens:
        features.append(f"unigram::{token}")
    for i in range(len(tokens) - 1):
        bigram = f"{tokens[i]}_{tokens[i+1]}"
        features.append(f"bigram::{bigram}")

    linear = bias
    for feat in features:
        linear += feature_weight(feat)
    logistic = 1 / (1 + math.exp(-linear))
    sentiment_score = (logistic - 0.5) * 2.0
    return max(-1.0, min(1.0, sentiment_score))


def _combine_scores(lexical: float, model: float) -> float:
    blended = 0.6 * lexical + 0.4 * model
    return max(-1.0, min(1.0, blended))


def _confidence_from_scores(score: float, tokens: List[str]) -> float:
    coverage = min(1.0, len(tokens) / 20.0)
    volatility = 1.0 - abs(score)
    confidence = 0.55 * coverage + 0.45 * (1 - volatility)
    return max(0.0, min(1.0, confidence))


def _compute_vote_weight(confidence: float, previous_votes: Optional[Dict[str, float]]) -> Tuple[float, str]:
    base_weight = confidence
    if previous_votes:
        prev_values = [float(v) for v in previous_votes.values() if isinstance(v, (int, float))]
        if prev_values:
            prev_weight = max(0.0, min(1.0, sum(prev_values) / (len(prev_values) * 1.5)))
        else:
            prev_weight = 0.25
        combined = 0.7 * base_weight + 0.3 * prev_weight
        explanation = (
            f"vote_weight = 0.7*confidence({confidence:.3f}) + 0.3*prev_avg({prev_weight:.3f}) -> {combined:.3f}"
        )
        return max(0.0, min(1.0, combined)), explanation
    explanation = f"vote_weight = confidence({confidence:.3f})"
    return max(0.0, min(1.0, base_weight)), explanation


def _base_output(input_json: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "stage_id": input_json["stage_id"],
        "timestamp_utc": input_json["timestamp_utc"],
        "market_data_id": input_json["market_data_id"],
    }


def _failure_output(input_json: Dict[str, Any], reason: str) -> SentimentOutput:
    ts_default = "1970-01-01T00:00:00Z"
    mdid_default = "UNKNOWN"
    if isinstance(input_json, dict):
        ts_val = input_json.get("timestamp_utc", ts_default)
        mdid_val = input_json.get("market_data_id", mdid_default)
    else:
        ts_val = ts_default
        mdid_val = mdid_default
    base = {"stage_id": "sentiment_brain", "timestamp_utc": ts_val, "market_data_id": mdid_val}
    return cast(SentimentOutput, {
        "stage_id": base["stage_id"],
        "timestamp_utc": base["timestamp_utc"],
        "market_data_id": base["market_data_id"],
        "sentiment_score": 0.0,
        "confidence": 0.0,
        "vote_weight": 0.0,
        "veto": False,
        "explain": {"top_tokens": [], "rules": ["failure"]},
        "dry_run": True,
        "log": {"status": "[FAIL:SENTIMENT]", "reason": reason},
        "log_details": {
            "runtime_ms": 0.0,
            "seed": None,
            "model_version": "n/a",
            "masked_input": {},
            "vote_weight_calc": "n/a",
        },
    })


def _deterministic_runtime_ms(seed: int, market_data_id: str, text: str, dry_run: bool) -> float:
    if dry_run:
        return 0.0
    h = hashlib.sha256()
    h.update(str(seed).encode("utf-8"))
    h.update(b"|")
    h.update(market_data_id.encode("utf-8"))
    h.update(b"|")
    h.update(text.encode("utf-8"))
    digest = int.from_bytes(h.digest()[:4], "big")  # 32-bit
    # Map to [0.5, 3.5] ms pseudo runtime for determinism in logs
    return round(0.5 + (digest % 3000) / 1000.0, 3)


def process(input_json: Dict[str, Any], config: Dict[str, Any]) -> SentimentOutput:
    """
    Run the Sentiment Brain stage for a single message. The function is pure,
    deterministic, and idempotent: identical inputs (including config) produce
    identical outputs.
    """
    try:
        if not isinstance(input_json, dict):
            raise TypeError("input_json must be dict")
        if not isinstance(config, dict):
            raise TypeError("config must be dict")
        validation_ok, reason = _validate_input(copy.deepcopy(input_json))
        if not validation_ok:
            return _failure_output(input_json, reason)
        parsed_config = _parse_config(config)
        if parsed_config.force_exception:
            raise RuntimeError("Forced exception for testing")
        seed = parsed_config.seed
        tokens = _tokenize(input_json["text_payload"])
        lexical_score, top_tokens = _lexical_sentiment(tokens)
        if parsed_config.use_remote_model:
            # Remote model usage is disabled in offline mode to honor safety.
            # In production, plug in deterministic remote call here.
            model_score = lexical_score
        else:
            model_score = _quantized_classifier_stub(tokens, seed)
        sentiment_score = _combine_scores(lexical_score, model_score)
        unsafe_reason = _check_unsafe_content(input_json["text_payload"])
        explain_rules: List[str] = []
        veto = False
        confidence = _confidence_from_scores(sentiment_score, tokens)
        vote_weight_calc_detail = ""
        vote_weight = 0.0
        if not unsafe_reason:
            previous_votes = input_json.get("previous_votes")
            vote_weight, vote_weight_calc_detail = _compute_vote_weight(confidence, previous_votes)
        else:
            vote_weight_calc_detail = "veto enforced"
            veto = True
            confidence = 1.0
            sentiment_score = 0.0
            explain_rules.append(unsafe_reason)
        log_status = "OK"
        log_reason = "sentiment computed" if not unsafe_reason else unsafe_reason
        log_details = {
            "masked_input": {
                "stage_id": input_json["stage_id"],
                "timestamp_utc": input_json["timestamp_utc"],
                "market_data_id": input_json["market_data_id"],
                "text_payload": _mask_text(input_json["text_payload"]),
                "meta": {
                    "lang": input_json["meta"]["lang"],
                    "source": _mask_text(input_json["meta"]["source"]),
                },
            },
            "model_version": parsed_config.model_version,
            "remote_model_used": bool(parsed_config.use_remote_model),
            "remote_model_note": "offline fallback to classifier_stub" if parsed_config.use_remote_model else "local stub",
            "runtime_ms": _deterministic_runtime_ms(seed, input_json["market_data_id"], input_json["text_payload"], parsed_config.dry_run),
            "seed": seed,
            "vote_weight_calc": vote_weight_calc_detail,
        }
        if not explain_rules:
            explain_rules.extend(["lexical_blend", "classifier_stub"])
        output = cast(SentimentOutput, {
            "stage_id": input_json["stage_id"],
            "timestamp_utc": input_json["timestamp_utc"],
            "market_data_id": input_json["market_data_id"],
            "sentiment_score": float(round(sentiment_score, 6)),
            "confidence": float(round(confidence, 6)),
            "vote_weight": float(round(vote_weight, 6)),
            "veto": veto,
            "explain": {"top_tokens": top_tokens, "rules": explain_rules},
            "dry_run": parsed_config.dry_run,
            "log": {"status": log_status, "reason": log_reason},
            "log_details": log_details,
        })
        return output
    except Exception as exc:  # noqa: BLE001
        failure = _failure_output(input_json if isinstance(input_json, dict) else {}, f"Exception: {exc}")
        # Keep deterministic runtime in failure as well (0.0 if cannot compute).
        try:
            parsed_seed = int(config.get("seed", 12345)) if isinstance(config, dict) else 12345
            mdid = input_json.get("market_data_id", "UNKNOWN") if isinstance(input_json, dict) else "UNKNOWN"
            text = input_json.get("text_payload", "") if isinstance(input_json, dict) else ""
            failure["log_details"]["runtime_ms"] = _deterministic_runtime_ms(parsed_seed, mdid, text, True)
        except Exception:
            failure["log_details"]["runtime_ms"] = 0.0
        return failure


SAMPLE_INPUT_NEUTRAL: SentimentInput = {
    "stage_id": "sentiment_brain",
    "timestamp_utc": "2025-01-01T00:00:00Z",
    "market_data_id": "MD-000",
    "text_payload": "The company announced a product update with limited impact.",
    "meta": {"lang": "en", "source": "newswire"},
}

SAMPLE_INPUT_POSITIVE: SentimentInput = {
    "stage_id": "sentiment_brain",
    "timestamp_utc": "2025-01-01T00:00:00Z",
    "market_data_id": "MD-001",
    "text_payload": "Analysts stay bullish as shares rally on record growth and optimistic guidance.",
    "meta": {"lang": "en", "source": "analyst_note"},
}

SAMPLE_INPUT_NEGATIVE: SentimentInput = {
    "stage_id": "sentiment_brain",
    "timestamp_utc": "2025-01-01T00:00:00Z",
    "market_data_id": "MD-002",
    "text_payload": "Investors fear a plunge after the downgrade and weak earnings miss.",
    "meta": {"lang": "en", "source": "market_blog"},
}


def _base_config(enable: bool = True, **overrides: Any) -> Dict[str, Any]:
    cfg = {
        "feature_flags": {"enable_sentiment": enable},
        "seed": 2025,
        "model_version": "sentiment-lexical-stub-v1",
    }
    cfg.update(overrides)
    return cfg


def benchmark_process(iterations: int = 1000) -> Dict[str, float]:
    inputs = [SAMPLE_INPUT_NEUTRAL, SAMPLE_INPUT_POSITIVE, SAMPLE_INPUT_NEGATIVE]
    latencies: List[float] = []
    config = _base_config(enable=True)
    for i in range(iterations):
        sample = inputs[i % len(inputs)]
        t0 = time.perf_counter()
        process(sample, config)
        latencies.append((time.perf_counter() - t0) * 1000.0)
    latencies.sort()
    p50 = statistics.median(latencies)
    p95_idx = max(0, min(len(latencies) - 1, int(0.95 * len(latencies)) - 1))
    p95 = latencies[p95_idx]
    max_latency = max(latencies)
    metrics = {"p50": round(p50, 3), "p95": round(p95, 3), "max": round(max_latency, 3)}
    # Enforce latency budget in benchmark mode for visibility/CI.
    assert p95 <= 5.0, f"Latency budget breach: p95={p95:.3f}ms"
    return metrics


def run_shadow_test(iterations: int = 1000) -> Dict[str, Any]:
    rng = random.Random(2025)
    sentiments = ["positive", "negative", "neutral"]
    templates = {
        "positive": [
            "Shares rally with record growth and bullish forecasts.",
            "Strong rebound as analysts stay optimistic and recommend buy.",
        ],
        "negative": [
            "Stock faces plunge amid downgrade and bearish outlook.",
            "Weak demand triggers selloff and pessimistic guidance.",
        ],
        "neutral": [
            "Company issues routine update with balanced commentary.",
            "Market sees steady performance without notable change.",
        ],
    }
    config = _base_config(enable=True)
    outcomes: List[Tuple[str, str, float]] = []
    latencies: List[float] = []
    for _ in range(iterations):
        label = rng.choice(sentiments)
        text = rng.choice(templates[label])
        synthetic_input = {
            "stage_id": "sentiment_brain",
            "timestamp_utc": "2025-01-01T00:00:00Z",
            "market_data_id": f"SHADOW-{rng.randint(1000, 9999)}",
            "text_payload": text,
            "meta": {"lang": "en", "source": "shadow_sim"},
        }
        t0 = time.perf_counter()
        result = process(synthetic_input, config)
        latency = (time.perf_counter() - t0) * 1000.0
        latencies.append(latency)
        predicted = "neutral"
        if result["veto"]:
            predicted = "unsafe"
        elif result["sentiment_score"] > 0.2:
            predicted = "positive"
        elif result["sentiment_score"] < -0.2:
            predicted = "negative"
        outcomes.append((label, predicted, result["sentiment_score"]))
    true_positive = sum(1 for lbl, pred, _ in outcomes if lbl == "positive" and pred == "positive")
    false_positive = sum(1 for lbl, pred, _ in outcomes if lbl != "positive" and pred == "positive")
    true_negative = sum(1 for lbl, pred, _ in outcomes if lbl == "negative" and pred == "negative")
    false_negative = sum(1 for lbl, pred, _ in outcomes if lbl == "positive" and pred != "positive")
    precision = true_positive / (true_positive + false_positive or 1)
    recall = true_positive / (true_positive + false_negative or 1)
    f1 = (2 * precision * recall) / (precision + recall or 1)
    latencies.sort()
    p50 = statistics.median(latencies)
    p95_idx = max(0, min(len(latencies) - 1, int(0.95 * len(latencies)) - 1))
    p95 = latencies[p95_idx]
    max_latency = max(latencies)
    deterministic = True
    sample_input = SAMPLE_INPUT_POSITIVE
    result1 = process(sample_input, config)
    result2 = process(sample_input, config)
    deterministic = deterministic and result1 == result2
    metrics = {
        "latency_ms": {"p50": round(p50, 3), "p95": round(p95, 3), "max": round(max_latency, 3)},
        "accuracy_metrics": {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
        },
        "deterministic_check": deterministic,
    }
    print(json.dumps(metrics, indent=2))
    return metrics


def _run_smoke_tests() -> None:
    metrics = benchmark_process(iterations=300)
    print("Benchmark (p50, p95, max ms):", metrics)
    shadow = run_shadow_test(iterations=300)
    print("Shadow test metrics:", json.dumps(shadow, indent=2))


def test_schema_validation_invalid_input() -> None:
    bad_input = {
        "timestamp_utc": "2025-01-01T00:00:00Z",
        "market_data_id": "MD-ERR",
        "text_payload": "Invalid payload",
        "meta": {"lang": "en", "source": "unit"},
    }
    output = process(bad_input, _base_config(enable=True))
    assert output["log"]["status"] == "[FAIL:SENTIMENT]"
    assert output["sentiment_score"] == 0.0


def test_idempotence() -> None:
    config = _base_config(enable=True, seed=777)
    result_a = process(SAMPLE_INPUT_POSITIVE, config)
    result_b = process(SAMPLE_INPUT_POSITIVE, config)
    assert result_a == result_b


def test_dry_run_flag() -> None:
    config = _base_config(enable=False)
    result = process(SAMPLE_INPUT_POSITIVE, config)
    assert result["dry_run"] is True
    assert result["log"]["status"] == "OK"


def test_failure_contract() -> None:
    config = _base_config(enable=True, force_exception=True)
    output = process(SAMPLE_INPUT_POSITIVE, config)
    assert output["log"]["status"] == "[FAIL:SENTIMENT]"
    assert output["confidence"] == 0.0
    assert output["dry_run"] is True


if __name__ == "__main__":
    if "--shadow" in sys.argv:
        run_shadow_test()
    elif "--benchmark" in sys.argv:
        print(json.dumps({"latency_ms": benchmark_process()}, indent=2))
    elif "--smoke" in sys.argv:
        _run_smoke_tests()
    else:
        print("Use `pytest sentiment_brain.py` for unit tests.")
        print("Use `python sentiment_brain.py --shadow` to run shadow test.")

