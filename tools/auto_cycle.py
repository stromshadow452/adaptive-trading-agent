#!/usr/bin/env python3
"""
Adaptive Cycle Orchestrator
Train → Predeploy Check → Register → Promote → Aggregate → PDF → Log

Examples:
  python tools/auto_cycle.py --symbol EURUSD --tf_base M15 --start 2022-01-01 --end 2025-08-31
  python tools/auto_cycle.py --symbol USDJPY --tf_fast M5 --tf_base M15 --tf_slow H1 --data_root "E:/fxdata"
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, date
from pathlib import Path


def run(cmd: str, check: bool = True):
    print(f"\n🚀 Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    if check and result.returncode != 0:
        raise RuntimeError(f"❌ Command failed: {cmd}")
    return result


def today_str():
    return date.today().strftime("%Y-%m-%d")


def timestamp_str():
    return datetime.now().strftime("%Y%m%d_%H%M")


def find_latest(root: Path, *patterns):
    """Return latest file that matches any of the glob patterns under root/**"""
    candidates = []
    for pat in patterns:
        candidates += list(root.glob(f"**/{pat}"))
    candidates = [p for p in candidates if p.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main():
    ap = argparse.ArgumentParser(description="Run full adaptive cycle")
    ap.add_argument("--symbol", required=True)

    # Timeframes (you may pass none; we auto-detect/fallback)
    ap.add_argument("--tf_fast", default=None)
    ap.add_argument("--tf_base", default="M15")
    ap.add_argument("--tf_slow", default=None)

    # Data
    ap.add_argument("--data_root", default="data/raw/forex_kaggle_multiTF",
                    help="Folder that contains files like <SYMBOL>_<TF>.csv")

    # Dates
    ap.add_argument("--start", default=None, help="YYYY-MM-DD (optional)")
    ap.add_argument("--end", default=today_str(), help="YYYY-MM-DD (default: today)")

    # Training knobs (passed through to your train_finrl.py)
    ap.add_argument("--algo", default="PPO", choices=["PPO", "A2C"])
    ap.add_argument("--steps", type=int, default=200000)
    ap.add_argument("--cost_bps", type=float, default=1.0)
    ap.add_argument("--eval_freq", type=int, default=10000)
    ap.add_argument("--max_episode_steps", type=int, default=None)

    args = ap.parse_args()

    t0 = time.time()
    symbol = args.symbol.upper()

    # ---------- AUTO-DETECT DATA ROOT & AVAILABLE TFS ----------
    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f"⚠️ data_root does not exist: {data_root}", file=sys.stderr)

    def tf_exists(tf: str):
        return (data_root / f"{symbol}_{tf.upper()}.csv").exists()

    # Preferred order if we need to fallback
    tfs_try = ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1"]

    def pick_closest(target: str):
        # If target exists, keep it; otherwise pick the first available TF in tfs_try
        if target and tf_exists(target):
            return target.upper()
        for tf in tfs_try:
            if tf_exists(tf):
                return tf
        return None

    # BASE TF
    tf_base = (args.tf_base or "M15").upper()
    if not tf_exists(tf_base):
        tf_base = pick_closest(tf_base)

    # FAST TF
    tf_fast = (args.tf_fast.upper() if args.tf_fast else None)
    if not tf_fast or not tf_exists(tf_fast):
        # prefer same as base if exists, else any available
        tf_fast = tf_base if (tf_base and tf_exists(tf_base)) else pick_closest(tf_base)

    # SLOW TF
    tf_slow = (args.tf_slow.upper() if args.tf_slow else None)
    if not tf_slow or not tf_exists(tf_slow):
        # prefer one coarser step if available, else base
        coarse_order = {"M1": "M5", "M5": "M15", "M15": "H1", "M30": "H1",
                        "H1": "H4", "H4": "D1", "D1": "W1", "W1": "W1"}
        pref = coarse_order.get(tf_base, "H1") if tf_base else "H1"
        tf_slow = pref if tf_exists(pref) else tf_base

    # Final sanity
    if not tf_base:
        raise RuntimeError("❌ Could not determine a valid base TF from data_root.")
    if not tf_fast:
        tf_fast = tf_base
    if not tf_slow:
        tf_slow = tf_base

    print(f"✅ Using data_root: {data_root}")
    print(f"✅ Timeframes -> fast:{tf_fast} base:{tf_base} slow:{tf_slow}")

    # ---------- TRAIN ----------
    out_dir = Path("models/finrl") / f"{args.algo.lower()}_{symbol}_{tf_base}_{timestamp_str()}"
    train_cmd = [
        "python", "tools/train_finrl.py",
        "--symbol", symbol,
        "--data_root", str(data_root),
        "--tf_fast", tf_fast,
        "--tf_base", tf_base,
        "--tf_slow", tf_slow,
        "--algo", args.algo,
        "--steps", str(args.steps),
        "--cost_bps", str(args.cost_bps),
        "--out", str(out_dir),
        "--eval_freq", str(args.eval_freq),
    ]
    if args.start:
        train_cmd += ["--start", args.start]
    if args.end:
        train_cmd += ["--end", args.end]
    if args.max_episode_steps:
        train_cmd += ["--max_episode_steps", str(args.max_episode_steps)]

    run(" ".join(train_cmd), check=True)

    # ---------- PREDEPLOY CHECKS ----------
    reports = Path("reports")
    reports.mkdir(exist_ok=True)
    equity = find_latest(reports, "*equity*.csv")
    trades = find_latest(reports, "*trades*.csv")

    if not equity and not trades:
        raise RuntimeError("❌ No equity/trades CSV found under ./reports after training.")

    predeploy_cmd = ["python", "tools/predeploy_checks.py"]
    if equity:
        predeploy_cmd += ["--equity_csv", str(equity)]
    if trades:
        predeploy_cmd += ["--trades_csv", str(trades)]
    predeploy_cmd += ["--min_sharpe", "1.2", "--max_dd", "-0.2"]
    run(" ".join(predeploy_cmd), check=True)

    # ---------- REGISTER ----------
    model_name = f"{args.algo.lower()}_{symbol}_{tf_base}_{timestamp_str()}"
    reg_cmd = ["python", "tools/register_model_from_predeploy.py", "--name", model_name]
    if equity:
        reg_cmd += ["--equity_csv", str(equity)]
    if out_dir.exists():
        reg_cmd += ["--copy_from", str(out_dir)]
    run(" ".join(reg_cmd), check=True)

    # ---------- PROMOTE ----------
    run("python tools/promote_model.py --require_predeploy_pass", check=True)

    # ---------- AGGREGATE RETURNS ----------
    if equity:
        run(f"python tools/aggregate_returns.py --equity_csv \"{equity}\"", check=True)
    else:
        run("python tools/aggregate_returns.py --equity_csv reports/equity.csv", check=False)

    # ---------- PDF REPORT ----------
    run("python tools/generate_returns_report.py", check=False)

    # ---------- LOG ----------
    cycle_log = reports / "cycle_log.json"
    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "symbol": symbol,
        "tf_fast": tf_fast,
        "tf_base": tf_base,
        "tf_slow": tf_slow,
        "start": args.start,
        "end": args.end,
        "algo": args.algo,
        "data_root": str(data_root),
        "train_out": str(out_dir),
        "duration_min": round((time.time() - t0) / 60, 2),
    }
    try:
        log = json.loads(cycle_log.read_text()) if cycle_log.exists() else []
    except Exception:
        log = []
    log.append(entry)
    cycle_log.write_text(json.dumps(log, indent=2), encoding="utf-8")

    print(f"\n✅ Cycle complete. Log: {cycle_log}")
    print("   Live model: models/live")
    print("   Report PDF: reports/returns_report.pdf")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Auto-cycle failed: {e}", file=sys.stderr)
        sys.exit(1)
