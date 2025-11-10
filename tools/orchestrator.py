#!/usr/bin/env python3
"""
tools/orchestrator.py

Master pipeline runner (screener -> paper_deploy -> summary -> meta -> decision -> finrl_adapter)
Usage example:
  python tools/orchestrator.py --config config/hybrid_pipeline.yaml --mode exec --autoretrain --finrl_policies models/finrl --finrl_algo PPO
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
import logging
import yaml

LOG = logging.getLogger("orchestrator")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

DEFAULT_CONFIG = "config/hybrid_pipeline.yaml"
RUNS_DIR = Path("reports") / "orchestrator_runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)


def now_rfc3339():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def ts_safe():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_yaml(path):
    p = Path(path)
    if not p.exists():
        LOG.warning("Config file not found: %s -> using empty config", path)
        return {}
    with p.open() as f:
        return yaml.safe_load(f) or {}


def run_cmd(cmd, env=None, dry_run=False, cwd=None):
    """
    Run a command via subprocess.run. Returns completed process or raises CalledProcessError.
    Logs command first.
    """
    LOG.info("CMD: %s", " ".join(cmd))
    if dry_run:
        LOG.info("DRY RUN: skipping command execution")
        return None
    cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, cwd=cwd)
    if cp.returncode != 0:
        LOG.error("Command failed (rc=%s): %s", cp.returncode, cp.stderr.strip() or cp.stdout.strip())
        # raise for caller to handle
        raise subprocess.CalledProcessError(cp.returncode, cmd, output=cp.stdout, stderr=cp.stderr)
    return cp


def step_screener(cfg, dry_run=False):
    screener_cfg = cfg.get("screener", {})
    out = screener_cfg.get("out", "reports/screener")
    Path(out).mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "tools/multi_pair_screener.py", "--config", screener_cfg.get("cfg_path", "config/screener.yaml"), "--out", out]
    return run_cmd(cmd, dry_run=dry_run)


def step_paper_deploy(cfg, dry_run=False):
    pd_cfg = cfg.get("paper_deploy", {})
    candidates = pd_cfg.get("candidates", "reports/screener/20251021_intraday.json")
    equity = str(pd_cfg.get("equity", 10000))
    top = str(pd_cfg.get("top", 3))
    out = pd_cfg.get("out", "reports/daily_logs")
    Path(out).mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "tools/paper_deploy.py", "--candidates", candidates, "--top", top, "--equity", equity, "--out", out]
    return run_cmd(cmd, dry_run=dry_run)


def step_generate_summary(cfg, dry_run=False):
    gen_cfg = cfg.get("generate_summary", {})
    reports = gen_cfg.get("reports", "reports/daily_logs")
    out = gen_cfg.get("out", "reports/aggregate")
    Path(out).mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "tools/generate_run_summary.py", "--reports", reports, "--out", out, "--verbose"]
    return run_cmd(cmd, dry_run=dry_run)


def step_train_meta(cfg, dry_run=False):
    tr_cfg = cfg.get("train_meta", {})
    logs = tr_cfg.get("logs", "reports/daily_logs")
    out = tr_cfg.get("out", "models/meta_selector/meta_selector.joblib")
    Path(Path(out).parent).mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "tools/train_meta_selector.py", "--logs", logs, "--out", out]
    return run_cmd(cmd, dry_run=dry_run)


def step_decision_engine(cfg, finrl_policies=None, dry_run=False):
    dec_cfg = cfg.get("decision_engine", {})
    out = dec_cfg.get("out", "reports/daily")
    Path(out).mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "tools/decision_engine_runner.py", "--out", out]
    if finrl_policies:
        cmd += ["--finrl_policies", finrl_policies]
    return run_cmd(cmd, dry_run=dry_run)


def step_finrl_adapter(cfg, finrl_policies=None, algo=None, dry_run=False):
    fa_cfg = cfg.get("finrl_adapter", {})
    policies = finrl_policies or fa_cfg.get("policies", "models/finrl")
    algo = algo or fa_cfg.get("algo", "PPO")
    cmd = [sys.executable, "tools/finrl_adapter.py", "--policies", policies, "--algo", algo]
    return run_cmd(cmd, dry_run=dry_run)


def safe_backup_config(decision_cfg_path="config/decision.yaml"):
    p = Path(decision_cfg_path)
    if p.exists():
        bk = p.parent / f"{p.stem}.backup_{ts_safe()}{p.suffix}"
        shutil.copy(p, bk)
        LOG.info("Backed up %s -> %s", p.as_posix(), bk.as_posix())
        return bk.as_posix()
    else:
        LOG.info("No decision config to backup at %s", decision_cfg_path)
        return None


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", "-c", default=DEFAULT_CONFIG, help="pipeline config yaml")
    p.add_argument("--mode", choices=["dry_run", "exec"], default="dry_run", help="dry_run or exec")
    p.add_argument("--autoretrain", action="store_true", help="run meta retrain step")
    p.add_argument("--finrl_policies", help="path to finrl policies directory")
    p.add_argument("--finrl_algo", help="finrl algo name (PPO, A2C, etc.)")
    args = p.parse_args(argv)

    cfg = load_yaml(args.config)
    risk_cfg = cfg.get("risk", {})
    kill_on_errors = bool(risk_cfg.get("kill_on_errors", True))

    dry_run = args.mode == "dry_run"
    run_id = f"run_{ts_safe()}"
    run_meta = {
        "id": run_id,
        "start": now_rfc3339(),
        "mode": args.mode,
        "autoretrain": args.autoretrain,
        "steps": [],
        "dry_run": dry_run,
    }

    steps = [
        ("screener", step_screener),
        ("paper_deploy", step_paper_deploy),
        ("generate_summary", step_generate_summary),
    ]
    if args.autoretrain:
        steps.append(("train_meta", step_train_meta))
    steps.append(("decision_engine", lambda c, d: step_decision_engine(c, finrl_policies=args.finrl_policies, dry_run=d)))
    steps.append(("finrl_adapter", lambda c, d: step_finrl_adapter(c, finrl_policies=args.finrl_policies, algo=args.finrl_algo, dry_run=d)))

    # run steps in order
    for name, fn in steps:
        step_meta = {"name": name, "start": now_rfc3339(), "status": "running"}
        try:
            LOG.info("Running step: %s", name)
            # Call step; pass full cfg and dry_run where expected
            # Many step functions accept (cfg, dry_run)
            if callable(fn):
                # some step functions were wrapped with lambdas (above) that already accept (cfg, dry_run)
                res = fn(cfg, dry_run)
            else:
                res = fn(cfg, dry_run=dry_run)
            step_meta["status"] = "ok"
            LOG.info("Step %s completed", name)
        except subprocess.CalledProcessError as e:
            step_meta["status"] = "failed"
            step_meta["rc"] = getattr(e, "returncode", None)
            step_meta["stderr"] = getattr(e, "stderr", "") or getattr(e, "output", "")
            LOG.error("Step %s failed: rc=%s", name, step_meta.get("rc"))
            if kill_on_errors:
                run_meta["steps"].append(step_meta)
                run_meta["end"] = now_rfc3339()
                out_file = RUNS_DIR / f"{run_id}.json"
                with out_file.open("w") as fh:
                    json.dump(run_meta, fh, indent=2)
                LOG.error("Orchestrator aborted due to error; run metadata saved to %s", out_file)
                sys.exit(1)
        except Exception as e:
            step_meta["status"] = "error"
            step_meta["error"] = str(e)
            LOG.exception("Uncaught exception in step %s", name)
            if kill_on_errors:
                run_meta["steps"].append(step_meta)
                run_meta["end"] = now_rfc3339()
                out_file = RUNS_DIR / f"{run_id}.json"
                with out_file.open("w") as fh:
                    json.dump(run_meta, fh, indent=2)
                LOG.error("Orchestrator aborted due to exception; run metadata saved to %s", out_file)
                sys.exit(1)
        finally:
            step_meta["end"] = now_rfc3339()
            run_meta["steps"].append(step_meta)

    run_meta["end"] = now_rfc3339()
    out_file = RUNS_DIR / f"{run_id}.json"
    with out_file.open("w") as fh:
        json.dump(run_meta, fh, indent=2)
    LOG.info("Orchestrator finished successfully: %s", out_file.as_posix())


if __name__ == "__main__":
    main()
