"""
tools/shadow_report.py
========================
SCOPUS Shadow Trading Performance Report — Week 15.

Generates a comprehensive terminal summary of the shadow trading run:
  1.  10 core shadow metrics (ShadowMetrics.compute())
  2.  10-criterion go/no-go gate result
  3.  Feature registry status (production / shadow / deprecated)
  4.  Champion model info
  5.  Drift (PSI) status
  6.  Pipeline v2 feature count
  7.  Ensemble weight file status
  8.  Live readiness pre-check

Usage:
    python tools/shadow_report.py
    python tools/shadow_report.py --fills-log logs/shadow/fills.jsonl
    python tools/shadow_report.py --json   # machine-readable output
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, ".")


# ── ANSI colours (stripped automatically if not a TTY) ───────────────────────
def _is_tty():
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

def _c(code: str, text: str) -> str:
    if not _is_tty():
        return text
    colours = {"green": "32", "red": "31", "yellow": "33",
               "cyan": "36", "bold": "1", "reset": "0"}
    return f"\033[{colours.get(code,'0')}m{text}\033[0m"

OK  = _c("green",  "✅")
FAIL = _c("red",   "❌")
WARN = _c("yellow","⚠️ ")
BOX  = _c("cyan",  "──")


# ---------------------------------------------------------------------------
# Section helpers
# ---------------------------------------------------------------------------

def _section(title: str):
    width = 62
    print()
    print(_c("cyan", "─" * width))
    print(_c("bold", f"  {title}"))
    print(_c("cyan", "─" * width))


def _row(label: str, value, ok: bool = True, width: int = 30):
    icon  = OK if ok else FAIL
    label = f"{label:<{width}}"
    print(f"  {icon}  {label}  {value}")


def _warn_row(label: str, value, width: int = 30):
    label = f"{label:<{width}}"
    print(f"  {WARN}  {label}  {value}")


# ---------------------------------------------------------------------------
# 1. Shadow Metrics
# ---------------------------------------------------------------------------

def report_shadow_metrics(fills_log: str) -> dict:
    _section("Shadow Trading Metrics")
    try:
        from src.monitoring.metrics import ShadowMetrics
        sm   = ShadowMetrics(fills_log)
        snap = sm.compute()
    except Exception as e:
        print(f"  [error] {e}")
        return {}

    pairs = [
        ("Net PnL",           f"${snap.get('net_pnl_usd', 0):+,.2f}",          snap.get("net_pnl_usd", 0) >= 0),
        ("Profit Factor",     f"{snap.get('profit_factor', 0):.3f}",             snap.get("profit_factor", 0) >= 1.1),
        ("Win Rate",          f"{snap.get('win_rate', 0):.1%}",                  snap.get("win_rate", 0) >= 0.50),
        ("Sharpe Ratio",      f"{snap.get('sharpe_annualized') or 0:.2f}",       (snap.get("sharpe_annualized") or 0) >= 0.5),
        ("Max Drawdown",      f"{snap.get('max_drawdown_pct', 0):.2f}%",         snap.get("max_drawdown_pct", 0) <= 10),
        ("Avg Slippage",      f"{snap.get('avg_slippage_pips', 0):.2f} pips",    snap.get("avg_slippage_pips", 0) <= 2.0),
        ("P95 Latency",       f"{snap.get('p95_latency_ms', 0):.0f} ms",         snap.get("p95_latency_ms", 0) <= 500),
        ("Total Trades",      snap.get("n_trades", 0),                           snap.get("n_trades", 0) >= 30),
        ("Shadow Days",       f"{snap.get('shadow_days', 0):.0f}",               snap.get("shadow_days", 0) >= 30),
    ]
    for label, val, ok in pairs:
        _row(label, val, ok)
    return snap


# ---------------------------------------------------------------------------
# 2. Go / No-Go Gate (10 criteria)
# ---------------------------------------------------------------------------

def report_gate(fills_log: str) -> bool:
    _section("Shadow → Live Go/No-Go Gate")
    try:
        from src.monitoring.metrics import ShadowMetrics
        sm     = ShadowMetrics(fills_log)
        result = sm.gate_check()
    except Exception as e:
        print(f"  [error] {e}")
        return False

    for c in result.get("criteria", []):
        ok  = c["passed"]
        val = c.get("value", "?")
        thr = c.get("threshold", "?")
        _row(c["criterion"], f"{val!s:>10}  (≥{thr!s})", ok)

    passed_all = result.get("all_pass", False)
    print()
    if passed_all:
        print(_c("green", "  ✅  OVERALL: PASS — Eligible for conditional live trading"))
    else:
        failed_list = ", ".join(result.get("failed", []))
        print(_c("red", f"  ❌  OVERALL: FAIL — Failed: {failed_list}"))
    return passed_all


# ---------------------------------------------------------------------------
# 3. Feature Registry
# ---------------------------------------------------------------------------

def report_feature_registry() -> dict:
    _section("Feature Registry Status")
    try:
        from research.feature_store import FeatureStore
        store   = FeatureStore()
        summary = store.summary()
        total   = sum(summary.values())

        for status in ("production", "shadow", "research", "deprecated"):
            n = summary.get(status, 0)
            ok = (status != "deprecated") or (n == 0)
            icon = OK if ok else WARN
            print(f"  {icon}  {status:<15}: {n:>4} features")

        print()
        gate_ready = store.list_gate_ready()
        print(f"  {OK if gate_ready else WARN}  Gate-ready (shadow→production): {len(gate_ready)}")
        return summary
    except Exception as e:
        print(f"  [error] {e}")
        return {}


# ---------------------------------------------------------------------------
# 4. Champion Model
# ---------------------------------------------------------------------------

def report_champion() -> dict:
    _section("Champion Model")
    champion_file = Path("models/registry/champion_latest.json")
    if not champion_file.exists():
        print(f"  {FAIL}  No champion model file found")
        return {}
    try:
        with open(champion_file) as f:
            ch = json.load(f)
        ts = ch.get("timestamp", "?")
        _row("Model path",     ch.get("model_path", "?"))
        _row("Val F1",         f"{ch.get('val_f1', 0):.4f}",          ch.get("val_f1", 0) >= 0.35)
        _row("Val accuracy",   f"{ch.get('val_accuracy', 0):.4f}")
        _row("Features kept",  ch.get("n_features", "?"),             True)
        _row("Best iteration", ch.get("best_iter", "?"),              True)
        _row("Trained",        ts[:19] if ts != "?" else "unknown",   True)
        _row("Train rows",     f"{ch.get('train_rows', 0):,}",        True)
        return ch
    except Exception as e:
        print(f"  [error] {e}")
        return {}


# ---------------------------------------------------------------------------
# 5. PSI Drift Status
# ---------------------------------------------------------------------------

def report_psi() -> float:
    _section("Drift Detection (PSI)")
    psi_log = Path("logs/shadow/psi_daily.jsonl")
    if not psi_log.exists():
        _warn_row("PSI log", "not found — DriftDetector not yet run")
        return 0.0
    try:
        lines  = psi_log.read_text().strip().split("\n")
        last   = json.loads(lines[-1])
        max_psi  = last.get("max_psi", 0.0)
        n_warn   = last.get("n_warning", 0)
        n_drift  = last.get("n_drifted", 0)
        _row("Max PSI",           f"{max_psi:.4f}",     max_psi < 0.15)
        _row("Warning features",  f"{n_warn}",          n_warn < 5)
        _row("Critical (drift)",  f"{n_drift}",         n_drift == 0)
        _row("Last check",        last.get("timestamp", "?")[:19], True)
        return max_psi
    except Exception as e:
        print(f"  [error] {e}")
        return 0.0


# ---------------------------------------------------------------------------
# 6. System Status
# ---------------------------------------------------------------------------

def report_system_status() -> dict:
    _section("System Component Status")
    status = {}

    # Fills log
    fills_log = Path("logs/shadow/fills.jsonl")
    status["fills_log"] = fills_log.exists()
    _row("Fills log present",  fills_log if fills_log.exists() else "missing",
         status["fills_log"])

    # Ensemble weights
    weights_file = Path("models/ensemble/optimal_weights.json")
    status["ensemble_weights"] = weights_file.exists()
    _row("Ensemble weights",   weights_file if weights_file.exists() else "not yet tuned",
         status["ensemble_weights"])

    # Prometheus
    try:
        import prometheus_client   # noqa: F401
        status["prometheus"] = True
        _row("prometheus_client",  "installed", True)
    except ImportError:
        status["prometheus"] = False
        _warn_row("prometheus_client", "not installed (optional)")

    # LightGBM
    try:
        import lightgbm   # noqa: F401
        status["lightgbm"] = True
        _row("LightGBM",           "installed", True)
    except ImportError:
        status["lightgbm"] = False
        _row("LightGBM",           "NOT installed — training disabled", False)

    # shadow_runner.py
    sr = Path("tools/shadow_runner.py")
    status["shadow_runner"] = sr.exists()
    _row("shadow_runner.py",   "present" if sr.exists() else "missing",
         status["shadow_runner"])

    return status


# ---------------------------------------------------------------------------
# 7. Ensemble weights file summary
# ---------------------------------------------------------------------------

def report_ensemble_weights():
    _section("Ensemble Weight Status")
    f = Path("models/ensemble/optimal_weights.json")
    if not f.exists():
        _warn_row("optimal_weights.json", "not yet generated — run tune_ensemble_weights.py")
        return
    try:
        data = json.load(open(f))
        ts   = data.get("optimised_at", "?")[:19]
        fills = data.get("fills_used", "?")
        print(f"  {OK}  Optimised at: {ts}   Fills used: {fills}")
        print()
        for regime, info in data.get("regimes", {}).items():
            w = info.get("weights", [])
            pf = info.get("pf_score", 0.0)
            w_str = "  ".join(f"{x:.2f}" for x in w)
            ok_flag = pf > 1.0 or all(x == 0.0 for x in w)
            _row(f"{regime}", f"[{w_str}]  PF={pf:.3f}", ok_flag)
    except Exception as e:
        print(f"  [error] {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description="SCOPUS Shadow Performance Report")
    p.add_argument("--fills-log", default="logs/shadow/fills.jsonl", dest="fills_log")
    p.add_argument("--json",      action="store_true",
                   help="Emit JSON output (machine-readable)")
    args = p.parse_args(argv)

    if args.json:
        # Machine-readable summary
        output = {}
        try:
            from src.monitoring.metrics import ShadowMetrics
            sm = ShadowMetrics(args.fills_log)
            output["metrics"] = sm.compute()
            output["gate"]    = sm.gate_check()
        except Exception as e:
            output["error"] = str(e)
        print(json.dumps(output, indent=2, default=str))
        return

    print()
    print(_c("bold", "=" * 62))
    print(_c("bold", "   SCOPUS SHADOW TRADING PERFORMANCE REPORT"))
    print(_c("bold", f"   Generated: {datetime.now(timezone.utc).isoformat()[:19]} UTC"))
    print(_c("bold", "=" * 62))

    snap       = report_shadow_metrics(args.fills_log)
    gate_pass  = report_gate(args.fills_log)
    reg        = report_feature_registry()
    champion   = report_champion()
    max_psi    = report_psi()
    sys_status = report_system_status()
    report_ensemble_weights()

    _section("SUMMARY")
    total_feats  = sum(reg.values()) if reg else 0
    prod_feats   = reg.get("production", 0)
    shadow_feats = reg.get("shadow", 0)
    lgb_ok       = sys_status.get("lightgbm", False)
    champion_ok  = bool(champion)

    print(f"  Features:     {prod_feats} production  +  {shadow_feats} in shadow")
    print(f"  Champion:     {'present' if champion_ok else 'not yet trained'}")
    print(f"  Drift:        max PSI = {max_psi:.4f} {'(SAFE)' if max_psi < 0.15 else '(DRIFT DETECTED)'}")
    print(f"  Gate:         {'PASS' if gate_pass else 'FAIL'}")
    print()

    if gate_pass and champion_ok and lgb_ok and max_psi < 0.25:
        print(_c("green", "  ✅  System is ELIGIBLE for conditional live trading."))
        print(_c("green", "      Final step: obtain risk committee approval."))
    else:
        print(_c("yellow", "  ⚠️   System is NOT YET ready for live trading."))
        missing = []
        if not gate_pass:    missing.append("shadow gate failing")
        if not champion_ok:  missing.append("no champion model")
        if not lgb_ok:       missing.append("LightGBM not installed")
        if max_psi >= 0.25:  missing.append(f"drift PSI={max_psi:.3f}")
        print(_c("yellow", f"      Blockers: {', '.join(missing)}"))

    print()
    print(_c("bold", "=" * 62))
    print()


if __name__ == "__main__":
    main()
