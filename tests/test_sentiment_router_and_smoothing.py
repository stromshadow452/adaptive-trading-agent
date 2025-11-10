from __future__ import annotations

from tools.executor_hooks.sentiment_hook import run_sentiment_stage_full as run_full
from sentiment_brain import SAMPLE_INPUT_POSITIVE, _base_config


def test_neutral_on_empty_text():
    cfg = _base_config(enable=True, seed=2025, dry_run=True)
    inp = dict(SAMPLE_INPUT_POSITIVE)
    inp["text_payload"] = ""
    out = run_full(inp, cfg)
    assert out["vote_weight"] == 0.0 and out["veto"] is False


def test_smoothing_cap_and_determinism():
    cfg = _base_config(enable=True, seed=2025, dry_run=True)
    a = run_full(SAMPLE_INPUT_POSITIVE, cfg)
    b = run_full(SAMPLE_INPUT_POSITIVE, cfg)
    assert a == b
    assert 0.0 <= a["vote_weight"] <= 0.35
    assert "vote_weight_post_smooth" in a["log_details"]

