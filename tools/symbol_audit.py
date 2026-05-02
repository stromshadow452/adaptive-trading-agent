"""
SCOPUS Symbol-Wise Validation Audit
=====================================
Runs the existing run_multi_symbol_shadow() for each symbol in isolation.
This reuses the IDENTICAL pipeline, edge filters, and 2024-2025 date alignment
as the portfolio run — but with no inter-symbol competition.

Usage:
    python tools/symbol_audit.py
    python tools/symbol_audit.py --days 90
    python tools/symbol_audit.py --symbols EURUSD GBPUSD USDJPY
"""
from __future__ import annotations

import argparse
import logging
import sys
from typing import List

sys.path.insert(0, ".")

# ── Silence ALL output noise ──────────────────────────────────────────────────
# 1. Python warnings (FutureWarning from pandas/volume_features.py etc.)
import warnings
warnings.filterwarnings("ignore")

# 2. Logging (INFO/WARNING from pipeline, broker, drift detector, etc.)
import logging
for _n in [
    "fast_shadow", "pipeline_v2", "src.decision", "src.decision.meta_gating",
    "src.risk", "src.risk.circuit_breaker", "src.features", "src.broker",
    "src.broker.paper_executor", "src.monitoring", "src.monitoring.drift",
    "src.strategies", "src.regime", "portfolio_selector",
]:
    logging.getLogger(_n).setLevel(logging.CRITICAL)
logging.getLogger().setLevel(logging.CRITICAL)   # root fallback

SYMBOLS = [
    "EURUSD", "GBPUSD", "USDJPY",
    "USDCAD", "AUDUSD", "NZDUSD", "XAGUSD",
]

# ── Classification thresholds ────────────────────────────────────────────────
def _label(pf: float, trades: int) -> str:
    if trades == 0:
        return "NO SIGNAL"
    if pf >= 1.30:
        return "STRONG ✓"
    if pf >= 1.00:
        return "WEAK   ~"
    return "LOSER  ✗"


def run_audit(symbols: List[str], days: int = 90) -> List[dict]:
    from tools.fast_shadow import run_multi_symbol_shadow

    results = []
    total = len(symbols)

    for i, sym in enumerate(symbols, 1):
        print(f"  [{i}/{total}] Running {sym:8s} ...", end="", flush=True)
        try:
            summary = run_multi_symbol_shadow(
                symbols  = [sym],
                tf       = "H1",
                days     = days,
                cfg_path = "config/weapon_system.yaml",
                dry_run  = False,
                verbose  = False,          # tqdm + fill logs suppressed
            )
        except ValueError as e:
            err_str = str(e)
            if "No symbol data found" in err_str:
                print(f"  no 2024-2025 data → skipping")
                results.append({
                    "symbol":   sym,
                    "trades":   0,
                    "win_rate": 0.0,
                    "pf":       0.0,
                    "pnl":      0.0,
                    "dd":       0.0,
                    "label":    "NO DATA",
                })
            else:
                print(f"  ERROR: {e}")
                results.append({
                    "symbol":   sym,
                    "trades":   0,
                    "win_rate": 0.0,
                    "pf":       0.0,
                    "pnl":      0.0,
                    "dd":       0.0,
                    "label":    f"ERROR",
                })
            continue

        trades   = summary.get("trades", 0)
        pf       = summary.get("profit_factor", 0.0) or 0.0
        wr       = summary.get("win_rate", 0.0) or 0.0
        pnl      = summary.get("net_pnl_usd", 0.0) or 0.0
        dd       = summary.get("drawdown_pct", 0.0) or 0.0
        label    = _label(pf, trades)

        results.append({
            "symbol":   sym,
            "trades":   trades,
            "win_rate": wr,
            "pf":       pf,
            "pnl":      pnl,
            "dd":       dd,
            "label":    label,
        })
        print(f"  trades={trades:3d}  PF={pf:.3f}  [{label}]")

    return results


def print_table(results: List[dict], days: int) -> None:
    # Sort by PF descending
    ranked = sorted(results, key=lambda r: r["pf"], reverse=True)

    SEP  = "=" * 80
    DASH = "-" * 80
    HDR  = (
        f"  {'Symbol':<8}  {'Trades':>6}  {'WinRate':>7}  "
        f"{'PF':>6}  {'PnL':>9}  {'MaxDD':>6}  Label"
    )

    print(f"\n{SEP}")
    print(f"  SCOPUS SYMBOL-WISE AUDIT — {days}-day simulation  (2024–2025 data)")
    print(SEP)
    print(HDR)
    print(DASH)
    for r in ranked:
        print(
            f"  {r['symbol']:<8}  "
            f"{r['trades']:>6}  "
            f"{r['win_rate']:>6.1%}  "
            f"{r['pf']:>6.3f}  "
            f"${r['pnl']:>+8,.2f}  "
            f"{r['dd']:>5.1f}%  "
            f"{r['label']}"
        )
    print(DASH)

    strong = [r["symbol"] for r in ranked if r["pf"] >= 1.30 and r["trades"] > 0]
    weak   = [r["symbol"] for r in ranked if 1.00 <= r["pf"] < 1.30]
    loser  = [r["symbol"] for r in ranked if r["pf"] < 1.00 and r["trades"] > 0]
    nosig  = [r["symbol"] for r in ranked if r["trades"] == 0]

    print(f"  STRONG  (PF≥1.30): {strong or ['none']}")
    print(f"  WEAK    (1.0-1.3): {weak   or ['none']}")
    print(f"  LOSER   (PF<1.00): {loser  or ['none']}")
    print(f"  NO SIG  (0 trades): {nosig  or ['none']}")
    print(SEP)

    # Recommended for portfolio
    candidates = [r for r in ranked if r["pf"] >= 1.30 and r["trades"] >= 10]
    if candidates:
        print(f"\n  ✓ Portfolio candidates (PF≥1.30, trades≥10):")
        for r in candidates:
            print(f"      {r['symbol']}  PF={r['pf']:.3f}  trades={r['trades']}")
    else:
        print(f"\n  ✗ No symbol meets PF≥1.30 with ≥10 trades on this period.")
        print(f"    Consider relaxing filters or using a different evaluation window.")
    print()


def main():
    p = argparse.ArgumentParser(description="SCOPUS per-symbol edge audit")
    p.add_argument("--symbols", nargs="*", default=SYMBOLS,
                   help="Space-separated list of symbols (default: all 7)")
    p.add_argument("--days", type=int, default=90,
                   help="Simulation window in days (default: 90)")
    args = p.parse_args()

    syms = [s.upper() for s in args.symbols]
    print(f"\nSCOPUS Symbol Audit — {len(syms)} symbols × {args.days} days")
    print(f"Filters: |boll_z|≥1.3, atr_pctile≥0.30, score≥0.40\n")

    results = run_audit(syms, days=args.days)
    print_table(results, days=args.days)


if __name__ == "__main__":
    main()
