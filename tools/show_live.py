#!/usr/bin/env python3
"""
show_live.py
Print a single-line display of the currently deployed (live) strategy.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "models" / "live"
METRICS = LIVE / "metrics.json"

def main():
    if not LIVE.exists():
        print("LIVE: <none>")
        return
    name = LIVE.resolve().name
    if METRICS.exists():
        try:
            m = json.loads(METRICS.read_text(encoding="utf-8"))
            sharpe = m.get("sharpe", "?")
            dd = m.get("max_drawdown", "?")
            print(f"LIVE: {name} | Sharpe={sharpe} | MaxDD={dd}")
            return
        except Exception:
            pass
    print(f"LIVE: {name}")

if __name__ == "__main__":
    main()
