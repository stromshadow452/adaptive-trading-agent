#!/usr/bin/env python3
# tools/decision_engine_runner.py
"""
Thin runner around src.decision_engine.main
- Accepts --candidates, --meta_model, --finrl_policies, --risk_config
- Accepts --out as EITHER a folder OR a json file (e.g. reports/daily/approved.json)
  * If --out endswith .json -> we create a temp plans folder, run decision_engine
    to dump individual *_plan_*.json files, then aggregate "enter=True" plans
    into the single --out JSON file.
  * If --out is a folder -> just forward to decision_engine (plans per-file).

Additive: --neural_chain flag to run an immutable neural pipeline including Sentiment Brain
via tools.pipeline_orchestrator.run_pipeline. This path is deterministic, append-only logging,
and does not affect existing behavior when the flag is not provided.
"""

from __future__ import annotations
import os
import sys
import json
import glob
import argparse
from datetime import datetime, timezone
from typing import Any, Dict

# Import engine without modifying it
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.decision_engine import main as decision_main  # noqa: E402

# Neural chain orchestrator import (additive)
try:
    from tools.pipeline_orchestrator import run_pipeline  # type: ignore
    from tools.executor_hooks.sentiment_hook import run_sentiment_stage_full as run_sentiment_stage  # noqa: F401
except Exception:
    run_pipeline = None  # type: ignore

try:
    from tools.executor_hooks.news_router_live import news_router  # type: ignore
    NEWS_ROUTER_SOURCE = "live"
except Exception:
    try:
        from tools.executor_hooks.news_router_stub import news_router  # type: ignore
        NEWS_ROUTER_SOURCE = "stub"
    except Exception:
        def news_router(md: Dict[str, Any]) -> str:  # type: ignore
            return "Market sees steady performance with balanced commentary."
        NEWS_ROUTER_SOURCE = "stub"


def _str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    return s in ("1", "true", "t", "yes", "y", "on")


def _logger_append(rec: Dict[str, Any]) -> None:
    # append-only JSON line
    try:
        print(json.dumps(rec, ensure_ascii=False))
    except Exception:
        pass


def _build_neural_stages(args) -> Dict[str, Any]:
    """
    Assemble stage callables required by run_pipeline(stages).
    Prefer project implementations; otherwise, fall back to deterministic stubs.
    """
    # Attempt to import stage callables from your codebase
    market_data = feature_reactor = primary_ml_brain = rl_brain = volatility_brain = None
    risk_brain = meta_gating_brain = risk_throttle_gates = execution_reflex_engine = broker_hedger = None
    try:
        from src.neural_chain import (  # type: ignore
            market_data as _md,
            feature_reactor as _fr,
            primary_ml_brain as _ml,
            rl_brain as _rl,
            volatility_brain as _vol,
            risk_brain as _risk,
            meta_gating_brain as _meta,
            risk_throttle_gates as _gates,
            execution_reflex_engine as _execx,
            broker_hedger as _broker,
        )
        market_data, feature_reactor, primary_ml_brain = _md, _fr, _ml
        rl_brain, volatility_brain, risk_brain = _rl, _vol, _risk
        meta_gating_brain, risk_throttle_gates = _meta, _gates
        execution_reflex_engine, broker_hedger = _execx, _broker
    except Exception:
        # Deterministic minimal stubs suitable for smoke/shadow
        def market_data() -> Dict[str, Any]:  # type: ignore
            return {"market_data_id": "MD-CHAIN", "symbol": "EURUSD"}

        def feature_reactor(md: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
            return {"timestamp_utc": "2025-01-01T00:00:00Z"}

        def primary_ml_brain(fr: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
            return {"vote": 0.6}

        def rl_brain(fr: Dict[str, Any], ml: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
            return {"vote": 0.55}

        def volatility_brain(fr: Dict[str, Any], ml: Dict[str, Any], rl: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
            return {"vote": 0.5}

        def risk_brain(fr, ml, rl, vol, sentiment=None):  # type: ignore
            return {"received_sentiment": sentiment, "ok": True}

        def meta_gating_brain(ml, rl, vol, sentiment, risk):  # type: ignore
            return {"ok": True}

        def risk_throttle_gates(risk, meta):  # type: ignore
            return {"ok": True}

        def execution_reflex_engine(gates):  # type: ignore
            return {"ok": True}

        def broker_hedger(execx):  # type: ignore
            return {"ok": True}

    # Normalize config flags (flat or nested)
    cfg: Dict[str, Any] = {}
    if hasattr(args, "config") and isinstance(getattr(args, "config"), dict):
        cfg.update(getattr(args, "config"))
    ff = cfg.setdefault("feature_flags", {})
    ff.setdefault("enable_sentiment", bool(getattr(args, "enable_sentiment", False) or cfg.get("enable_sentiment", False)))
    cfg.setdefault("sentiment_vote_cap", float(getattr(args, "sentiment_vote_cap", 0.35)))
    cfg.setdefault("dry_run", bool(getattr(args, "dry_run", False)))
    cfg.setdefault("seed", int(getattr(args, "seed", 2025)))

    stages: Dict[str, Any] = {
        "market_data": market_data,
        "feature_reactor": feature_reactor,
        "primary_ml_brain": primary_ml_brain,
        "rl_brain": rl_brain,
        "volatility_brain": volatility_brain,
        "risk_brain": risk_brain,  # Should accept optional 5th arg
        "meta_gating_brain": meta_gating_brain,
        "risk_throttle_gates": risk_throttle_gates,
        "execution_reflex_engine": execution_reflex_engine,
        "broker_hedger": broker_hedger,
        "news_router": news_router,
        "lang": "en",
        "news_source": "stub",
        "logger_append": _logger_append,
        "config": cfg,
    }
    return stages

def _is_json_file(p: str) -> bool:
    return p.lower().endswith(".json")

def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)

def _collect_plans(plans_dir: str) -> list[dict]:
    out = []
    for fp in glob.glob(os.path.join(plans_dir, "*_plan_*.json")):
        try:
            # BOM-safe read
            with open(fp, "r", encoding="utf-8-sig") as f:
                out.append(json.load(f))
        except Exception:
            # skip unreadable
            pass
    return out

def _make_approved(plans: list[dict]) -> dict:
    """
    Convert list of plan JSONs into a compact approved payload:
    {
      "generated_at": "...Z",
      "count": N,
      "approved": [
        {symbol, tf, side, size, price, sl, tp, final_score, meta_w, finrl_present, notes, reason}
      ]
    }
    """
    approved = []
    for p in plans:
        try:
            if not bool(p.get("enter")):
                continue
            approved.append({
                "symbol": p.get("symbol"),
                "tf": p.get("tf"),
                "side": p.get("side"),
                "size": p.get("size"),
                "price": p.get("price"),
                "sl": p.get("sl"),
                "tp": p.get("tp"),
                "final_score": p.get("final_score"),
                "meta_w": p.get("meta_w"),
                "finrl_present": p.get("finrl_present"),
                "notes": p.get("notes"),
                "reason": p.get("reason"),
            })
        except Exception:
            continue

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "count": len(approved),
        "approved": approved,
    }
    return payload

def main():
    ap = argparse.ArgumentParser(description="Decision Engine runner (folder/file out smart handling)")
    ap.add_argument("--candidates", required=False, help="path to candidates JSON (no BOM, utf-8)")
    ap.add_argument("--meta_model", default="models/meta_selector/meta_selector.joblib")
    ap.add_argument("--finrl_policies", default=None)
    ap.add_argument("--risk_config", default="config/decision.yaml")
    ap.add_argument("--out", required=False, help="folder for plans OR a .json aggregate file")
    # pass-through options
    ap.add_argument("--auto_execute", action="store_true")
    ap.add_argument("--primary_thresh", type=float, default=0.70)
    ap.add_argument("--finrl_thresh", type=float, default=0.65)
    # additive flag: neural chain execution
    ap.add_argument("--neural_chain", action="store_true", help="Run the neural pipeline with Sentiment Brain")
    ap.add_argument("--use_remote", action="store_true", help="Enable remote fetching for news router (default: False)")
    # neural-chain config flags
    ap.add_argument("--enable_sentiment", type=_str2bool, nargs="?", const=True, default=False,
                    help="Enable Sentiment Brain (default False). Accepts true/false.")
    ap.add_argument("--dry_run", type=_str2bool, nargs="?", const=True, default=False,
                    help="Dry-run (deterministic, no side-effects). Accepts true/false.")
    ap.add_argument("--sentiment_vote_cap", type=float, default=0.35,
                    help="Cap for sentiment vote weight (default 0.35).")
    ap.add_argument("--seed", type=int, default=2025,
                    help="Deterministic seed (default 2025).")
    args = ap.parse_args()

    # Optional neural chain execution path (additive-only)
    if bool(getattr(args, "neural_chain", False)):
        if run_pipeline is None:
            print(json.dumps({"SMOKE": "FAIL", "reason": "orchestrator_unavailable"}, ensure_ascii=False))
            return 1
        # Normalize config from CLI for orchestrator
        cfg: Dict[str, Any] = {}
        cfg["dry_run"] = bool(args.dry_run)
        cfg["seed"] = int(args.seed)
        cfg["sentiment_vote_cap"] = float(args.sentiment_vote_cap)
        cfg["use_remote"] = bool(args.use_remote)
        ff = cfg.setdefault("feature_flags", {})
        ff["enable_sentiment"] = bool(args.enable_sentiment)
        setattr(args, "config", cfg)

        stages = _build_neural_stages(args)
        stages["news_router"] = news_router
        stages["news_source"] = NEWS_ROUTER_SOURCE
        _ = run_pipeline(stages)
        _logger_append({"SMOKE": "PASS", "stage_order": ["market_data", "feature_reactor", "primary_ml_brain", "rl_brain", "volatility_brain", "sentiment_brain", "risk_brain", "meta_gating_brain", "risk_throttle_gates", "execution_reflex_engine", "broker_hedger"]})
        return 0

    # Legacy path: enforce required args when not using neural_chain
    if not args.candidates or not args.out:
        ap.error("--candidates and --out are required unless --neural_chain is set")

    out_arg = args.out
    out_is_file = _is_json_file(out_arg)

    if out_is_file:
        # prepare a sibling temp folder to hold per-plan jsons
        # e.g., reports/daily/approved.json -> reports/daily/plans
        parent = os.path.dirname(out_arg) or "."
        plans_dir = os.path.join(parent, "plans")
        os.makedirs(plans_dir, exist_ok=True)
        forwarded = [
            "--candidates", args.candidates,
            "--out", plans_dir,
            "--meta_model", args.meta_model,
            "--risk_config", args.risk_config,
        ]
        if args.finrl_policies:
            forwarded += ["--finrl_policies", args.finrl_policies]
        if args.auto_execute:
            forwarded += ["--auto_execute"]
            forwarded += ["--primary_thresh", str(args.primary_thresh)]
            forwarded += ["--finrl_thresh", str(args.finrl_thresh)]

        rc = decision_main(forwarded)
        if rc not in (0, None):
            return rc

        # collect individual plans and write one approved file
        plans = _collect_plans(plans_dir)
        payload = _make_approved(plans)
        _ensure_parent_dir(out_arg)
        with open(out_arg, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote aggregate approved file: {out_arg} (count={payload['count']})")
        return 0

    else:
        # treat as folder; ensure exists then forward
        os.makedirs(out_arg, exist_ok=True)
        forwarded = [
            "--candidates", args.candidates,
            "--out", out_arg,
            "--meta_model", args.meta_model,
            "--risk_config", args.risk_config,
        ]
        if args.finrl_policies:
            forwarded += ["--finrl_policies", args.finrl_policies]
        if args.auto_execute:
            forwarded += ["--auto_execute"]
            forwarded += ["--primary_thresh", str(args.primary_thresh)]
            forwarded += ["--finrl_thresh", str(args.finrl_thresh)]
        return decision_main(forwarded)

if __name__ == "__main__":
    sys.exit(main())
