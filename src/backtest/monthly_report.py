"""
Monthly Report Generator

Generates formatted monthly performance reports for the Portfolio Shadow Engine.
"""

from datetime import datetime
from typing import Dict, List, Optional
import json


def generate_monthly_report(
    portfolio_stats: dict,
    protection_stats: dict,
    dampener_stats: dict,
    asset_breakdown: Dict[str, dict],
    month_name: str = "MONTH",
) -> str:
    """
    Generate formatted monthly report.
    
    Args:
        portfolio_stats: From PortfolioState.get_monthly_stats()
        protection_stats: From CapitalProtector.get_summary()
        dampener_stats: From LossStreakDampener.get_summary()
        asset_breakdown: Per-asset performance dict
        month_name: Display name for the month
        
    Returns:
        Formatted report string
    """
    # Header
    lines = [
        "╔" + "═" * 62 + "╗",
        "║" + "SCOPUS PORTFOLIO - MONTHLY REPORT".center(62) + "║",
        "║" + month_name.center(62) + "║",
        "╠" + "═" * 62 + "╣",
        "║" + " " * 62 + "║",
    ]
    
    # Performance Summary
    lines.append("║  PERFORMANCE SUMMARY" + " " * 41 + "║")
    lines.append("║  " + "─" * 57 + "   ║")
    
    starting = portfolio_stats.get('starting_capital', 10000)
    ending = portfolio_stats.get('ending_capital', 10000)
    net_return = portfolio_stats.get('net_return_pct', 0)
    gross_profit = portfolio_stats.get('gross_profit', 0)
    gross_loss = portfolio_stats.get('gross_loss', 0)
    pf = portfolio_stats.get('profit_factor', 0)
    
    lines.append(f"║  Starting Capital:        ${starting:,.2f}".ljust(63) + "║")
    lines.append(f"║  Ending Capital:          ${ending:,.2f}".ljust(63) + "║")
    
    sign = "+" if net_return >= 0 else ""
    lines.append(f"║  Net Monthly Return:      {sign}{net_return:.2f}%".ljust(63) + "║")
    lines.append("║" + " " * 62 + "║")
    lines.append(f"║  Gross Profit:            ${gross_profit:,.2f}".ljust(63) + "║")
    lines.append(f"║  Gross Loss:              (${gross_loss:,.2f})".ljust(63) + "║")
    lines.append(f"║  Profit Factor:           {pf:.2f}".ljust(63) + "║")
    lines.append("║" + " " * 62 + "║")
    
    # Risk Metrics
    lines.append("║  RISK METRICS" + " " * 48 + "║")
    lines.append("║  " + "─" * 57 + "   ║")
    
    max_dd = portfolio_stats.get('max_drawdown_pct', 0)
    max_dd_date = portfolio_stats.get('max_drawdown_date')
    worst_day = portfolio_stats.get('worst_day', (None, 0))
    best_day = portfolio_stats.get('best_day', (None, 0))
    win_rate = portfolio_stats.get('win_rate', 0)
    
    lines.append(f"║  Max Drawdown:            {max_dd:.2f}%".ljust(63) + "║")
    if max_dd_date:
        lines.append(f"║  Max DD Date:             {max_dd_date}".ljust(63) + "║")
    
    if worst_day[0]:
        lines.append(f"║  Worst Day:               {worst_day[1]*100:.2f}% ({worst_day[0]})".ljust(63) + "║")
    if best_day[0]:
        lines.append(f"║  Best Day:                +{best_day[1]*100:.2f}% ({best_day[0]})".ljust(63) + "║")
    
    lines.append(f"║  Win Rate:                {win_rate:.1f}%".ljust(63) + "║")
    lines.append("║" + " " * 62 + "║")
    
    # Trade Statistics
    lines.append("║  TRADE STATISTICS" + " " * 44 + "║")
    lines.append("║  " + "─" * 57 + "   ║")
    
    total_trades = portfolio_stats.get('total_trades', 0)
    wins = portfolio_stats.get('wins', 0)
    losses = portfolio_stats.get('losses', 0)
    
    lines.append(f"║  Total Trades:            {total_trades}".ljust(63) + "║")
    lines.append(f"║  Wins:                    {wins}".ljust(63) + "║")
    lines.append(f"║  Losses:                  {losses}".ljust(63) + "║")
    lines.append("║" + " " * 62 + "║")
    
    # Asset Breakdown
    if asset_breakdown:
        lines.append("║  ASSET BREAKDOWN" + " " * 45 + "║")
        lines.append("║  " + "─" * 57 + "   ║")
        
        for symbol, stats in asset_breakdown.items():
            brain = stats.get('brain', 'Trend')
            ret = stats.get('return_pct', 0)
            trades = stats.get('trades', 0)
            asset_pf = stats.get('pf', 0)
            
            sign = "+" if ret >= 0 else ""
            line = f"║  {symbol} ({brain}):".ljust(22) + f"{sign}{ret:.2f}%  │ {trades} trades │ PF {asset_pf:.2f}"
            lines.append(line.ljust(63) + "║")
        
        lines.append("║" + " " * 62 + "║")
    
    # Capital Protection Events
    lines.append("║  CAPITAL PROTECTION EVENTS" + " " * 35 + "║")
    lines.append("║  " + "─" * 57 + "   ║")
    
    caution = protection_stats.get('caution_events', 0)
    danger = protection_stats.get('danger_events', 0)
    critical = protection_stats.get('critical_events', 0)
    force_closed = protection_stats.get('positions_force_closed', 0)
    
    lines.append(f"║  CAUTION triggered:       {caution} times".ljust(63) + "║")
    lines.append(f"║  DANGER triggered:        {danger} times".ljust(63) + "║")
    lines.append(f"║  CRITICAL triggered:      {critical} times".ljust(63) + "║")
    lines.append(f"║  Positions force-closed:  {force_closed}".ljust(63) + "║")
    lines.append("║" + " " * 62 + "║")
    
    # Dampener Activity
    lines.append("║  DAMPENER ACTIVITY" + " " * 43 + "║")
    lines.append("║  " + "─" * 57 + "   ║")
    
    dampened_pct = dampener_stats.get('dampened_pct', 0)
    pauses = dampener_stats.get('total_pauses', 0)
    
    lines.append(f"║  Trades at full size:     {100-dampened_pct:.1f}%".ljust(63) + "║")
    lines.append(f"║  Trades dampened:         {dampened_pct:.1f}%".ljust(63) + "║")
    lines.append(f"║  Pause events:            {pauses}".ljust(63) + "║")
    lines.append("║" + " " * 62 + "║")
    
    # Verdict
    lines.append("║  " + "─" * 58 + "  ║")
    
    if net_return >= 0 and max_dd < 6:
        verdict = "VERDICT: PROFITABLE MONTH"
        status = "System performed within risk parameters."
    elif net_return >= 0:
        verdict = "VERDICT: PROFITABLE BUT HIGH RISK"
        status = f"Drawdown {max_dd:.1f}% exceeded target. Review risk settings."
    else:
        verdict = "VERDICT: LOSING MONTH"
        status = "System took controlled losses. Edge may be temporary."
    
    lines.append("║  " + verdict.ljust(60) + "║")
    lines.append("║  " + status.ljust(60) + "║")
    
    # Footer
    lines.append("╚" + "═" * 62 + "╝")
    
    return "\n".join(lines)


def save_monthly_report(
    report: str,
    filepath: str,
    portfolio_stats: dict,
    protection_stats: dict,
) -> None:
    """Save report to file along with JSON data."""
    # Save text report
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(report)
    
    # Save JSON data
    json_path = filepath.replace('.txt', '.json')
    data = {
        'portfolio': portfolio_stats,
        'protection': protection_stats,
        'generated_at': datetime.now().isoformat(),
    }
    
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
