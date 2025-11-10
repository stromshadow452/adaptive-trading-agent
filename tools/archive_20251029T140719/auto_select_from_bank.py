# tools/auto_select_from_bank.py
import os, json, argparse, glob

def load_metric_from_json(path, metric):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # flexible paths: direct, nested, or in "stats"
        for keypath in [
            [metric],
            ["metrics", metric],
            ["stats", metric],
            ["summary", metric],
        ]:
            d = data
            ok = True
            for k in keypath:
                if isinstance(d, dict) and k in d:
                    d = d[k]
                else:
                    ok = False
                    break
            if ok and isinstance(d, (int, float)):
                return float(d)
    except Exception:
        pass
    return None

def best_from_bank(bank_dir, metric):
    """Return (strategy_id, score, meta_path)"""
    cand = []
    for root,dirs,files in os.walk(bank_dir):
        for name in files:
            if name.lower() in ("metrics.json","report.json","summary.json"):
                mp = os.path.join(root, name)
                score = load_metric_from_json(mp, metric)
                if score is not None:
                    # strategy id = parent folder name
                    strategy_id = os.path.basename(os.path.dirname(mp))
                    cand.append((strategy_id, score, mp))
    if not cand:
        return None, None, None
    # higher is better for sharpe etc.
    cand.sort(key=lambda x: x[1], reverse=True)
    return cand[0]

def main():
    ap = argparse.ArgumentParser(description="Select best strategy from bank by metric and annotate candidates.")
    ap.add_argument("--candidates", required=True, help="screener JSON (list or {candidates:[]})")
    ap.add_argument("--bank", required=True, help="folder containing many strategy subfolders with metrics.json/report.json")
    ap.add_argument("--metric", default="sharpe", help="metric key to rank (default: sharpe)")
    ap.add_argument("--out", required=True, help="output candidates JSON with selected strategy attached")
    args = ap.parse_args()

    # 1) choose best strategy from bank
    sid, score, meta_path = best_from_bank(args.bank, args.metric)
    if sid is None:
        raise SystemExit(f"No strategy metrics found in: {args.bank}")

    # 2) load candidates
    with open(args.candidates, "r", encoding="utf-8") as f:
        raw = json.load(f)
    cands = raw["candidates"] if isinstance(raw, dict) and "candidates" in raw else raw
    if not isinstance(cands, list):
        raise SystemExit("Bad candidates JSON")

    # 3) inject selection into each candidate
    for c in cands:
        c["selected_strategy"] = {
            "id": sid,
            "metric": args.metric,
            "score": score,
            "meta_path": meta_path
        }

    # 4) save
    out_obj = {"candidates": cands}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, indent=2)
    print(f"[OK] selected={sid} ({args.metric}={score}) -> {args.out}")

if __name__ == "__main__":
    main()
