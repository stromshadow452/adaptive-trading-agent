#!/usr/bin/env python
"""
tools/brain_report.py
=====================
CLI tool to run the offline AI Brain analysis.

Usage:
    python tools/brain_report.py                         # All trades
    python tools/brain_report.py --days 7                # Last 7 days
    python tools/brain_report.py --days 90 --strategy MR # MR trades only
    python tools/brain_report.py --journal path/to/file  # Custom journal path

The report is saved to logs/brain/reports/ as a Markdown file.
"""
from __future__ import annotations

import argparse
import logging
import sys

# Ensure project root is importable
sys.path.insert(0, ".")

from src.brain.journal import load_trades
from src.brain.analyzer import analyze
from src.brain.edge_analyzer import analyze_edge
from src.brain.reporter import generate_report

LOG = logging.getLogger("brain_report")


def main(argv=None):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    p = argparse.ArgumentParser(
        description="AI Brain — Offline Trade Analysis Report Generator"
    )
    p.add_argument(
        "--journal", default=None,
        help="Path to JSONL journal file. Default: auto-detect."
    )
    p.add_argument(
        "--days", default=None, type=int,
        help="Only analyze trades from the last N days."
    )
    p.add_argument(
        "--strategy", default=None,
        help="Filter to a specific strategy (e.g. MEAN_REVERSION)."
    )
    p.add_argument(
        "--symbol", default=None,
        help="Filter to a specific symbol (e.g. GBPUSD)."
    )
    p.add_argument(
        "--report-dir", default="logs/brain/reports", dest="report_dir",
        help="Directory to save the report."
    )
    args = p.parse_args(argv)

    # Load trades
    trades = load_trades(
        path=args.journal,
        days=args.days,
        strategy=args.strategy,
        symbol=args.symbol,
    )

    if not trades:
        LOG.warning("No trades found. Run a simulation first to populate the journal.")
        LOG.warning("  Example: python tools/fast_shadow.py --symbols ALL --days 90")
        return

    LOG.info(f"Analyzing {len(trades)} trades...")

    # Run analysis
    analysis = analyze(trades)
    edge = analyze_edge(trades)

    # Generate report
    report_path = generate_report(analysis, edge, report_dir=args.report_dir)

    # Print summary to console
    overall = analysis.get("overall", {})
    print()
    print("=" * 60)
    print("AI BRAIN — ANALYSIS COMPLETE")
    print(f"  Trades:      {overall.get('trades', 0)}")
    print(f"  Win Rate:    {overall.get('win_rate', 0):.1%}")
    print(f"  PF:          {overall.get('pf', 0):.3f}")
    print(f"  Expectancy:  ${overall.get('expectancy', 0):+.2f}/trade")
    print(f"  Net PnL:     ${overall.get('net_pnl', 0):+.2f}")
    print(f"  Strong zones: {len(analysis.get('strong_zones', []))}")
    print(f"  Weak zones:   {len(analysis.get('weak_zones', []))}")
    print(f"  Report:      {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
