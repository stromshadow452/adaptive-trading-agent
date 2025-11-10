#!/usr/bin/env python3
"""
Promote best paper-trade model to 'live' slot if predeploy checks passed.

New flags:
  --prefer_newer          : Tiebreak by most-recent folder mtime
  --force <model_name>    : Force-promote a specific paper_pool/<model_name>
"""

import argparse, json, os, sys, shutil, time
from pathlib import Path

def load_metrics(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def find_candidates(pool_dir: Path):
    for d in sorted(pool_dir.glob("*")):
        if d.is_dir() and (d / "metrics.json").exists():
            yield d

def score_tuple(m: dict):
    # Higher is better for all 3 below
    return (
        float(m.get("sharpe", 0.0)),
        float(m.get("max_drawdown", -1.0)),  # less negative = higher
        float(m.get("total_pnl", 0.0)),
    )

def choose_best(cands, prefer_newer=False):
    best = None
    for d in cands:
        m = load_metrics(d / "metrics.json")
        s = score_tuple(m)
        tiebreak = d.stat().st_mtime if prefer_newer else 0.0
        item = (s, tiebreak, d, m)
        if best is None or item > best:
            best = item
    return best  # (score, tiebreak, dir, metrics)

def update_symlink_or_copy(link_path: Path, target_dir: Path):
    if link_path.exists() or link_path.is_symlink():
        try:
            if link_path.is_symlink() or link_path.is_file():
                link_path.unlink()
            else:
                shutil.rmtree(link_path)
        except Exception:
            pass
    try:
        link_path.symlink_to(target_dir.resolve())
    except OSError:
        shutil.copytree(target_dir, link_path)

def main():
    ap = argparse.ArgumentParser(description="Promote best model from paper pool")
    ap.add_argument("--pool_dir", type=str, default="models/paper_pool")
    ap.add_argument("--live_link", type=str, default="models/live")
    ap.add_argument("--require_predeploy_pass", action="store_true")
    ap.add_argument("--predeploy_report", type=str, default="reports/predeploy_report.json")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--prefer_newer", action="store_true", help="Prefer newer model on score ties")
    ap.add_argument("--force", type=str, default=None, help="Force promote this model name in paper_pool")
    args = ap.parse_args()

    pool = Path(args.pool_dir)
    if not pool.exists():
        print(f" No paper pool found at: {pool}", file=sys.stderr)
        sys.exit(5)

    # Optional PASS gate
    if args.require_predeploy_pass:
        rep_path = Path(args.predeploy_report)
        if not rep_path.exists():
            print(" Missing predeploy_report.json while --require_predeploy_pass is set", file=sys.stderr)
            sys.exit(3)
        rep = json.loads(rep_path.read_text(encoding="utf-8"))
        if rep.get("result") != "PASS":
            print(" Predeploy checks did NOT PASS. Aborting promotion.", file=sys.stderr)
            sys.exit(4)

    # Force path?
    if args.force:
        target = pool / args.force
        if not target.exists():
            print(f" Force target not found: {target}", file=sys.stderr)
            sys.exit(6)
        best_dir = target
        best_m = load_metrics(best_dir / "metrics.json")
    else:
        cands = list(find_candidates(pool))
        if not cands:
            print(f" No candidates with metrics.json under: {pool}", file=sys.stderr)
            sys.exit(6)
        picked = choose_best(cands, prefer_newer=args.prefer_newer)
        _, _, best_dir, best_m = picked

    if args.dry_run:
        print(f"[DRY-RUN] Would promote: {best_dir}")
        print(json.dumps(best_m, indent=2))
        sys.exit(0)

    live_link = Path(args.live_link)
    update_symlink_or_copy(live_link, best_dir)

    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "promoted_dir": str(best_dir),
        "metrics": best_m
    }
    reg_path = Path("models/promotion_registry.json")
    try:
        reg = json.loads(reg_path.read_text(encoding="utf-8")) if reg_path.exists() else []
    except Exception:
        reg = []
    reg.append(entry)
    reg_path.write_text(json.dumps(reg, indent=2), encoding="utf-8")

    print(f"Promoted: {best_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
