#!/usr/bin/env python3
# tools/mark2close_pnl.py — robust mark-to-close with diagnostics + seek-move (v5.1)
import argparse, csv, fnmatch, logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

LOG = logging.getLogger("mark2close")
logging.basicConfig(level=logging.INFO, format="%(message)s")

EXEC_DIR = Path("reports") / "executions"
DEFAULT_IN  = EXEC_DIR / "executions.csv"
DEFAULT_OUT = EXEC_DIR / "executions_mark2close.csv"
DEBUG_OUT   = EXEC_DIR / "mark2close_debug.csv"

SEARCH_ROOTS = [Path("data") / "raw", Path("data") / "ohlcv", Path("data")]
CLOSE_CANDS  = ["close","Close","adj_close","Adj Close","bidclose","askclose"]
OPEN_CANDS   = ["open","Open","bidopen","askopen"]

def to_float(x, d=0.0):
    try:
        v = float(x)
        return d if v!=v or v in (float("inf"), float("-inf")) else v
    except Exception:
        return d

def truthy(x):
    return str(x or "").strip().lower() in ("1","true","yes","y")

def parse_symbol_tf_from_plan(plan_file: str) -> Tuple[str, Optional[str]]:
    name = Path(plan_file).name.replace(".json","")
    parts = name.split("_")
    sym = (parts[0] if parts else name).upper()
    tf = None
    for p in parts[1:]:
        u = p.upper()
        if u in ("M1","M5","M15","M30","H1","H2","H4","H6","H8","H12","D1","W1","1D","4H","15M","5M","1H"):
            tf = u.replace("1D","D1").replace("4H","H4").replace("15M","M15").replace("5M","M5").replace("1H","H1")
            break
    return sym, tf

def iter_patterns(symbol: str, tf: Optional[str]) -> List[str]:
    pats = []
    if tf:
        pats += [f"{symbol}_{tf}.csv", f"{symbol}-{tf}.csv"]
    pats += [f"{symbol}_M1.csv", f"{symbol}.csv", f"{symbol}_*.csv", f"*{symbol}*.csv"]
    return pats

def glob_first(root: Path, pattern: str) -> Optional[Path]:
    if not root.exists(): return None
    for p in root.rglob("*.csv"):
        if fnmatch.fnmatch(p.name, pattern):
            return p
    return None

def read_all_rows(csv_path: Path) -> Tuple[List[Dict[str,Any]], List[str]]:
    try:
        with csv_path.open(newline="", encoding="utf-8") as fh:
            rdr = csv.DictReader(fh); rows = list(rdr)
        if not rows: return [], []
        header = list(rows[-1].keys())
        return rows, header
    except Exception:
        return [], []

def pick_col(header: List[str], prefer: str="close") -> Optional[str]:
    cands = CLOSE_CANDS if prefer.lower()=="close" else OPEN_CANDS
    for k in cands:
        if k in header: return k
    lower = {h.lower(): h for h in header}
    for k in cands:
        if k.lower() in lower: return lower[k.lower()]
    return None

def resolve_csv(symbol: str, tf: Optional[str], debug: bool=False) -> Tuple[Optional[Path], List[str]]:
    tried = []
    for root in SEARCH_ROOTS:
        if not root.exists():
            tried.append(f"{root} (missing)"); continue
        for pat in iter_patterns(symbol, tf):
            fp = glob_first(root, pat)
            if not fp:
                tried.append(f"{root}//{pat} (no match)"); continue
            rows, header = read_all_rows(fp)
            if rows:
                tried.append(f"{fp} (rows={len(rows)})")
                if debug:
                    print(f"[DEBUG] csv({symbol}) -> {fp.name} rows={len(rows)}")
                return fp, tried
            else:
                tried.append(f"{fp} (empty)")
    return None, tried

def compute_m2c(entry: float, exit_px: float, side: str, qty: float) -> Tuple[float,float]:
    side = (side or "").lower()
    if entry<=0 or exit_px<=0 or qty<=0 or side not in ("buy","sell"):
        return 0.0, 0.0
    if side == "buy":
        ret = (exit_px/entry) - 1.0; pnl = qty*(exit_px - entry)
    else:
        ret = (entry/exit_px) - 1.0; pnl = qty*(entry - exit_px)
    return ret, pnl

def detect_fields(row: Dict[str,Any], assume_executed: bool) -> Tuple[bool,str,float,float,str,str,str]:
    side     = (row.get("side") or "").lower()
    qty      = to_float(row.get("qty"), None)
    if qty is None: qty = to_float(row.get("size"), 0.0)
    price    = to_float(row.get("price"), 0.0)
    symbol   = (row.get("symbol") or "").upper()
    plan_file= row.get("plan_file") or ""
    mode     = row.get("mode") or row.get("status") or ""

    executed = truthy(row.get("executed"))
    if not executed:
        status = (row.get("status") or "").lower()
        if status in ("filled","filled_paper"): executed = True
        if assume_executed and side in ("buy","sell") and qty>0 and price>0:
            executed = True
        elif not assume_executed and side in ("buy","sell") and qty>0 and price>0:
            executed = True
    return executed, side, qty, price, symbol, plan_file, mode

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=str(DEFAULT_IN))
    ap.add_argument("--out", dest="out", default=str(DEFAULT_OUT))
    ap.add_argument("--horizon", type=int, default=1, help="exit = entry + horizon bars")
    ap.add_argument("--entry-shift", type=int, default=1, help="entry = last_bar - entry_shift (default 1 = penultimate)")
    ap.add_argument("--seek-move", type=int, default=0, help="scan up to N bars ahead for the first price move vs entry")
    ap.add_argument("--exit-field", choices=["close","open"], default="close", help="price column to use")
    ap.add_argument("--eps", type=float, default=1e-9, help="minimum price change to count as a move")
    ap.add_argument("--price-debug", action="store_true")
    ap.add_argument("--assume-executed", action="store_true", help="Treat valid side/qty/price rows as executed")
    ap.add_argument("--why-debug", action="store_true", help="Write per-row reasons to mark2close_debug.csv")
    args = ap.parse_args()

    inp = Path(args.inp); outp = Path(args.out)
    if not inp.exists(): raise SystemExit(f"Input not found: {inp}")

    with inp.open(newline="", encoding="utf-8") as fh:
        rdr = csv.DictReader(fh); rows = list(rdr)
    if not rows:
        print("[INFO] No executions to process."); outp.write_text(""); return

    out_headers = ["timestamp","mode","symbol","side","qty","price","executed","plan_file",
                   "exit_time","exit_price","horizon_bars","ret","pnl"]
    out_data: List[Dict[str,Any]] = []
    debug_rows: List[Dict[str,Any]] = []

    processed = 0; realized = 0

    for r in rows:
        processed += 1
        executed, side, qty, price, symbol, plan_file, mode = detect_fields(r, args.assume_executed)
        timestamp = r.get("timestamp") or r.get("time") or ""
        exit_time=""; exit_price=""; ret=""; pnl=""; reason=""

        if not executed:
            reason = "not_executed_flag"
        elif side not in ("buy","sell"):
            reason = f"side={side}"
        elif not symbol:
            reason = "missing_symbol"
        elif qty<=0 or price<=0:
            reason = f"qty_or_price_nonpos qty={qty} price={price}"
        else:
            sym_from_plan, tf = parse_symbol_tf_from_plan(plan_file) if plan_file else (symbol, None)
            sym = symbol or sym_from_plan

            csv_path, tried = resolve_csv(sym, tf, debug=args.price_debug)
            if not csv_path:
                reason = "csv_not_found"
                if args.price_debug:
                    print(f"[DEBUG] {sym} CSV not found. Tried:")
                    for t in tried: print("   -", t)
            else:
                bars, header = read_all_rows(csv_path)
                if not bars or not header:
                    reason = "csv_no_rows"
                else:
                    price_col = pick_col(header, prefer=args.exit_field) or pick_col(header, prefer="close")
                    if not price_col:
                        reason = "no_price_col"
                    else:
                        n = len(bars)
                        entry_idx = max(0, n-1 - max(0, int(args.entry_shift)))
                        exit_idx  = min(entry_idx + max(1, int(args.horizon)), n-1)

                        entry_px_csv = to_float(bars[entry_idx].get(price_col), 0.0)

                        # Optionally seek the first bar with a real price move
                        if args.seek_move > 0 and entry_px_csv > 0:
                            target = None
                            for k in range(1, int(args.seek_move)+1):
                                i = entry_idx + k
                                if i >= n: break
                                px = to_float(bars[i].get(price_col), 0.0)
                                if abs(px - entry_px_csv) > args.eps:
                                    target = i; break
                            if target is not None:
                                exit_idx = target
                            else:
                                # if nothing moved, ensure exit > entry when possible
                                if exit_idx <= entry_idx and n >= 2:
                                    exit_idx = n-1
                                    entry_idx = max(0, exit_idx-1)

                        exit_px_csv  = to_float(bars[exit_idx].get(price_col), 0.0)
                        if price <= 0.0: price = entry_px_csv

                        r_ret, r_pnl = compute_m2c(price, exit_px_csv, side, qty)
                        ret, pnl = r_ret, r_pnl

                        # exit timestamp
                        last_row = bars[exit_idx]
                        for tcol in ("timestamp","time","datetime","Date","date"):
                            if tcol in last_row:
                                exit_time = last_row[tcol]; break
                        exit_price = exit_px_csv
                        realized += 1
                        reason = ""  # realized OK

        out_row = {
            "timestamp": timestamp,
            "mode": mode,
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "price": price,
            "executed": str(bool(executed)).lower(),
            "plan_file": plan_file,
            "exit_time": exit_time,
            "exit_price": exit_price,
            "horizon_bars": max(1,int(args.horizon)),
            "ret": ret,
            "pnl": pnl
        }
        out_data.append(out_row)

        if args.why_debug:
            dbg = dict(out_row); dbg["skip_reason"] = reason
            debug_rows.append(dbg)

    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=out_headers)
        w.writeheader()
        for r in out_data:
            w.writerow({k:r.get(k,"") for k in out_headers})

    if args.why_debug:
        with DEBUG_OUT.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=out_headers+["skip_reason"])
            w.writeheader()
            for r in debug_rows:
                w.writerow({k:r.get(k,"") for k in out_headers+["skip_reason"]})

    print(f"[OK] Processed rows: {processed} | realized trades: {realized}")
    print(f"[INFO] Wrote -> {outp.as_posix()}")
    if args.why_debug:
        print(f"[INFO] Debug -> {DEBUG_OUT.as_posix()}")

if __name__ == "__main__":
    main()
