# tools/run_executor_per_plan.py
#!/usr/bin/env python3
import json, subprocess, sys
from pathlib import Path

BASE = Path.cwd()
approved = BASE / "reports/daily/approved_paths_actionable_filtered.json"
if not approved.exists():
    print("approved list not found:", approved); sys.exit(1)

paths = json.loads(approved.read_text(encoding="utf-8-sig"))
executor = BASE / "tools" / "executor.py"
if not executor.exists():
    print("executor.py not found at tools/executor.py"); sys.exit(1)

logs_dir = BASE / "logs" 
logs_dir.mkdir(parents=True, exist_ok=True)

for p in paths:
    ppath = str(p)
    logpath = logs_dir / (Path(ppath).stem + ".executor.log")
    cmd = [
        sys.executable, str(executor),
        "--approved", ppath,
        "--executions", "reports/executions/executions.csv",
        "--aggregate", "reports/aggregate/aggregate_summary.json",
        "--max_position", "2",
        "--mode", "paper"
    ]
    print("RUN:", " ".join(cmd))
    with logpath.open("w", encoding="utf-8") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
    print(" -> log saved:", logpath)
