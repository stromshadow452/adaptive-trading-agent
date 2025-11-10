#!/usr/bin/env python3
import csv
from pathlib import Path

INP = Path("reports/executions/executions.csv")
OUT = Path("reports/executions/executions_dedup.csv")

def main():
    if not INP.exists():
        raise SystemExit(f"Input not found: {INP}")
    with INP.open(newline="", encoding="utf-8") as fh:
        rdr = csv.DictReader(fh)
        rows = list(rdr)
        if not rows:
            raise SystemExit("Empty executions.csv")
        headers = rdr.fieldnames

    seen = set()
    out = []
    for r in rows:
        key = (
            (r.get("symbol") or "").upper(),
            (r.get("side") or "").lower(),
            r.get("price") or "",
            r.get("timestamp") or "",
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(r)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=headers)
        w.writeheader()
        for r in out:
            w.writerow(r)

    print(f"[OK] wrote {len(out)} rows -> {OUT.as_posix()}")

if __name__ == "__main__":
    main()
