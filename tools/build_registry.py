# tools/build_registry.py
import os, re, yaml, hashlib, json

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_PATH = os.path.join(ROOT, "strategy_bank", "external_raw")
OUT_PATH = os.path.join(ROOT, "strategy_bank", "normalized_yaml")

# Auto-create folders if missing
os.makedirs(RAW_PATH, exist_ok=True)
os.makedirs(OUT_PATH, exist_ok=True)

def guess_family(txt: str) -> str:
    t = (txt or "").lower()
    if re.search(r"(ema|macd|adx|trend|supertrend)", t): return "trend"
    if re.search(r"(rsi|bollinger|mean|zscore|vwap)", t): return "mean_reversion"
    if re.search(r"(breakout|donchian|atr|range)", t): return "breakout"
    if re.search(r"(pair|stat|arbitrage|cointegrat)", t): return "stat_arb"
    if re.search(r"(sentiment|news|nlp|event)", t): return "event_driven"
    if re.search(r"(ppo|a2c|sac|dqn|reinforcement)", t): return "ml_rl"
    return "unknown"

def main():
    registry = []
    if not os.path.isdir(RAW_PATH):
        print("RAW_PATH missing:", RAW_PATH)
        return

    vendors = [v for v in os.listdir(RAW_PATH) if os.path.isdir(os.path.join(RAW_PATH, v))]
    for vendor in vendors:
        vendor_dir = os.path.join(RAW_PATH, vendor)
        print(f"📦 Scanning: {vendor}")
        for root, _, files in os.walk(vendor_dir):
            for f in files:
                if not f.endswith(".py"): 
                    continue
                path = os.path.join(root, f)
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                        src = fh.read()
                except Exception:
                    src = ""
                fam = guess_family(src)
                sha = hashlib.sha1((vendor + "|" + f + "|" + path).encode()).hexdigest()[:12]
                entry = {
                    "id": f"{vendor}_{sha}",
                    "source_vendor": vendor,
                    "file": os.path.relpath(path, ROOT).replace("\\", "/"),
                    "name": os.path.splitext(f)[0],
                    "family": fam,
                    "timeframe": "auto",
                    "assets": ["AUTO"],
                    "params": {},
                    "entry_rule": "TBD",
                    "exit_rule": "TBD",
                }
                registry.append(entry)

                # one YAML per strategy
                out_yaml = os.path.join(OUT_PATH, f"{entry['id']}.yaml")
                with open(out_yaml, "w", encoding="utf-8") as yh:
                    yaml.safe_dump(entry, yh, sort_keys=False)

    print(f"\n✅ Total strategies found: {len(registry)}")
    with open(os.path.join(OUT_PATH, "_registry.json"), "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)

if __name__ == "__main__":
    main()
