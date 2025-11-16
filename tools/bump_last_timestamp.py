# bump_last_timestamp.py
import csv, shutil, sys
from datetime import datetime, timezone
from pathlib import Path

def bump(path):
    p = Path(path)
    if not p.exists(): 
        print("missing:", p); return
    bak = p.with_suffix(p.suffix + ".timestamp.bak")
    shutil.copy2(p, bak)
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    if not rows or len(rows) < 2:
        print("no data:", p); return
    hdr = rows[0]
    # find timestamp-like column
    ts_idx = None
    for i,h in enumerate(hdr):
        low = h.strip().lower()
        if "timestamp" in low or "time"==low or "date" in low:
            ts_idx = i
            break
    if ts_idx is None:
        print("no timestamp column found in", p); return
    nowz = datetime.now(timezone.utc).isoformat().replace("+00:00","Z")
    # replace only the last non-empty data row's ts
    for i in range(len(rows)-1, 0, -1):
        row = rows[i]
        if any(cell.strip() for cell in row):
            row[ts_idx] = nowz
            rows[i] = row
            break
    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print("bumped", p, "-> backup:", bak.name)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python bump_last_timestamp.py EURUSD_M15_2024_to_2025.csv [other.csv ...]")
    for arg in sys.argv[1:]:
        bump(arg)
