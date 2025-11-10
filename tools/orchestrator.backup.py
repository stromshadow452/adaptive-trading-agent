# orchestrator.py
"""
Simple orchestrator: screener -> paper_deploy -> score with meta-model -> create plans -> (optional) execute paper.
Usage:
python orchestrator.py --mode dry_run
"""

import subprocess, argparse, json, glob, time
from pathlib import Path
from src.decision_engine import score_candidates

SCR_DIR = "reports/screener"
PAPER_OUT = "reports/daily_logs"
PLANS_OUT = "reports/daily"

def run_screener():
    print("Running screener...")
    cmd = ["python", "tools/multi_pair_screener.py", "--config", "config/screener.yaml", "--out", SCR_DIR]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print("⚠️ tools/multi_pair_screener.py not found. Skipping screener step.")
        return []
    time.sleep(1)
    files = sorted(glob.glob(f"{SCR_DIR}/*_intraday.json")) + sorted(glob.glob(f"{SCR_DIR}/*_swing.json"))
    print(f"Screener files: {len(files)}")
    return files

def run_paper_deploy(candidates_file: str):
    print("Running paper_deploy for", candidates_file)
    cmd = ["python", "tools/paper_deploy.py", "--candidates", candidates_file, "--top", "3", "--equity", "10000", "--out", PAPER_OUT]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print("⚠️ tools/paper_deploy.py not found. Continuing without preanalytics.")

def load_candidates_from_screener(json_path: str):
    js = json.load(open(json_path, "r", encoding="utf-8"))
    cands = []
    for x in js:
        cands.append({
            "pair": x.get("pair"),
            "tf": x.get("tf"),
            "tech_score": x.get("score", 0.0),
            "fund_score": x.get("fund_score", 0.0),
            "impact_score": x.get("impact", x.get("impact_score", 0.0)),
            "surprise_norm": x.get("surprise_norm", 0.0),
            "adx": x.get("adx", 0.0),
            "atr": x.get("atr", 0.0),
            "vol_ratio": x.get("vol_ratio", 0.0),
            "tech_sharpe": x.get("tech_sharpe", 0.0),
            "tech_trades": x.get("tech_trades", x.get("trades", 0.0)),
            "close": x.get("close", x.get("price", 0.0)),
            "within_1h_event": x.get("within_1h_event", False),
        })
    return cands

def main(dry_run=True):
    Path(PLANS_OUT).mkdir(parents=True, exist_ok=True)
    screener_files = run_screener()
    created_plans = []
    for scr in screener_files:
        run_paper_deploy(scr)
        candidates = load_candidates_from_screener(scr)
        plans = score_candidates(candidates, out_dir=PLANS_OUT)
        for plan, path in plans:
            print("Saved plan:", path, plan.get("pair"), plan.get("action"), f"size={plan['size_pct']:.4f}")
            created_plans.append(path)

    if not dry_run:
        for p in created_plans:
            print("Executing plan (paper simulate):", p)
            cmd = ["python", "tools/paper_executor.py", "--plan", p, "--equity", "10000", "--out", PAPER_OUT]
            try:
                subprocess.run(cmd, check=True)
            except FileNotFoundError:
                print("⚠️ tools/paper_executor.py not found. Skipping execution for", p)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="dry_run", choices=["dry_run","exec"])
    args = ap.parse_args()
    main(dry_run=(args.mode=="dry_run"))
