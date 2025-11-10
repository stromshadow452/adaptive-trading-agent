# tools/check_csvs.py
import os, json, sys

dirs = ["data/raw/M15_only", "data/raw/forex_backup_2020_2025"]
expected = []

# Try to read reports/daily/approved.json for expected symbols
try:
    with open("reports/daily/approved.json", "r", encoding="utf8") as f:
        approved = json.load(f)
    for r in approved:
        s = r.get("symbol") or r.get("pair") or r.get("ticker")
        tf = r.get("timeframe", "M15")
        if s:
            # Keep possible filename patterns used by your executor
            expected.append(f"{s}_{tf}.csv")
            expected.append(f"{s}_{tf}_2023_to_2024.csv")   # common style
            expected.append(f"{s}_{tf}_2021_to_2022.csv")
except FileNotFoundError:
    print("reports/daily/approved.json not found — will only list CSVs in folders.")
except Exception as e:
    print("Could not parse approved.json:", e)

# scan folders and collect actual filenames
found = set()
for d in dirs:
    if not os.path.isdir(d):
        print("Warning: dir not found:", d)
        continue
    for root,_,files in os.walk(d):
        for f in files:
            if f.lower().endswith(".csv"):
                found.add(f)

print("\n--- Summary ---")
print("Folders scanned:", [d for d in dirs if os.path.isdir(d)])
print("Total CSV files found:", len(found))

if expected:
    missing = [f for f in expected if f not in found]
    print("Expected examples (first 20):", expected[:20])
    print("Missing count:", len(missing))
    print("Missing sample:", missing[:20])
else:
    print("No expected list available from approved.json — cannot compute missing list.")

# optional: print a small sample of found files
print("\nSample found CSVs (first 40):")
for i,fn in enumerate(sorted(found)):
    if i>=40: break
    print(" ", fn)
