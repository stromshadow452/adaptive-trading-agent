"""
SCOPUS Phase-1: Shadow Mode Monitor

Daily monitoring script for shadow mode validation.
Run at end of each trading day (00:00 UTC).

Monitors:
- Signal count per asset
- Pattern distribution
- Consecutive losses
- Shadow PnL
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json
import csv

# ============================================================================
# CONFIGURATION
# ============================================================================

SHADOW_ASSETS = ['EURUSD', 'GBPUSD', 'AUDUSD']  # USDJPY blocked
BLOCKED_ASSETS = ['USDJPY']

# STOP thresholds
STOP_THRESHOLDS = {
    'max_dd_asset': 0.08,       # 8% DD on any asset
    'max_dd_portfolio': 0.10,   # 10% portfolio DD
    'max_consec_losses': 8,     # 8 consecutive losses
    'max_daily_loss': 0.02,     # 2% daily loss
    'min_daily_signals': 0,     # 0 = check after 2 days
    'max_daily_signals': 10,    # Flag if > 10
}

# Alert thresholds (warning, not STOP)
ALERT_THRESHOLDS = {
    'pattern_dominance': 0.70,  # One pattern > 70%
    'consec_losses_warn': 5,    # Warn at 5
    'low_win_rate': 0.30,       # Pattern WR < 30%
}

# Paths
SHADOW_LOG = Path('shadow_log.csv')
SHADOW_STATE = Path('shadow_state.json')


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class AssetState:
    """Track state for one asset."""
    symbol: str
    signals: int = 0
    wins: int = 0
    losses: int = 0
    consec_losses: int = 0
    pnl: float = 0.0
    peak_equity: float = 1.0
    current_equity: float = 1.0
    max_dd: float = 0.0
    patterns: Dict[str, int] = field(default_factory=dict)
    pattern_wins: Dict[str, int] = field(default_factory=dict)
    last_signal_date: str = ""
    status: str = "ACTIVE"  # ACTIVE, PAUSED, STOPPED


@dataclass 
class ShadowState:
    """Overall shadow mode state."""
    start_date: str = ""
    current_week: int = 1
    assets: Dict[str, AssetState] = field(default_factory=dict)
    daily_logs: List[Dict] = field(default_factory=list)
    alerts: List[str] = field(default_factory=list)
    stops_triggered: List[str] = field(default_factory=list)
    
    def to_dict(self):
        return {
            'start_date': self.start_date,
            'current_week': self.current_week,
            'assets': {k: v.__dict__ for k, v in self.assets.items()},
            'alerts': self.alerts[-20:],  # Keep last 20
            'stops_triggered': self.stops_triggered,
        }


# ============================================================================
# SHADOW MONITOR
# ============================================================================

class ShadowMonitor:
    """
    Shadow mode monitoring and validation.
    """
    
    def __init__(self):
        self.state = self._load_state()
    
    def _load_state(self) -> ShadowState:
        """Load or initialize state."""
        if SHADOW_STATE.exists():
            try:
                with open(SHADOW_STATE) as f:
                    data = json.load(f)
                state = ShadowState(
                    start_date=data.get('start_date', ''),
                    current_week=data.get('current_week', 1),
                    alerts=data.get('alerts', []),
                    stops_triggered=data.get('stops_triggered', []),
                )
                for symbol, asset_data in data.get('assets', {}).items():
                    state.assets[symbol] = AssetState(**asset_data)
                return state
            except:
                pass
        
        # Initialize new state
        state = ShadowState(
            start_date=datetime.now().strftime('%Y-%m-%d'),
        )
        for symbol in SHADOW_ASSETS:
            state.assets[symbol] = AssetState(symbol=symbol)
        return state
    
    def _save_state(self):
        """Save state to disk."""
        with open(SHADOW_STATE, 'w') as f:
            json.dump(self.state.to_dict(), f, indent=2)
    
    def record_signal(
        self,
        symbol: str,
        pattern: str,
        direction: str,
        entry: float,
        sl: float,
        tp: float,
        ml_conf: float,
    ):
        """Record a new signal."""
        if symbol not in self.state.assets:
            return
        
        asset = self.state.assets[symbol]
        asset.signals += 1
        asset.last_signal_date = datetime.now().strftime('%Y-%m-%d')
        
        # Track pattern
        asset.patterns[pattern] = asset.patterns.get(pattern, 0) + 1
        
        # Log
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'pattern': pattern,
            'direction': direction,
            'entry': entry,
            'sl': sl,
            'tp': tp,
            'ml_conf': ml_conf,
            'status': 'OPEN',
        }
        self.state.daily_logs.append(log_entry)
        
        self._save_state()
    
    def record_outcome(
        self,
        symbol: str,
        pattern: str,
        result: str,  # 'WIN' or 'LOSS'
        pnl_pct: float,
    ):
        """Record trade outcome."""
        if symbol not in self.state.assets:
            return
        
        asset = self.state.assets[symbol]
        
        if result == 'WIN':
            asset.wins += 1
            asset.consec_losses = 0
            asset.pattern_wins[pattern] = asset.pattern_wins.get(pattern, 0) + 1
        else:
            asset.losses += 1
            asset.consec_losses += 1
        
        # Update equity
        asset.pnl += pnl_pct
        asset.current_equity *= (1 + pnl_pct)
        
        if asset.current_equity > asset.peak_equity:
            asset.peak_equity = asset.current_equity
        
        dd = 1 - asset.current_equity / asset.peak_equity
        if dd > asset.max_dd:
            asset.max_dd = dd
        
        # Check STOP conditions
        self._check_stop_conditions(symbol)
        
        self._save_state()
    
    def _check_stop_conditions(self, symbol: str):
        """Check if STOP conditions are met."""
        asset = self.state.assets[symbol]
        
        # Max DD
        if asset.max_dd > STOP_THRESHOLDS['max_dd_asset']:
            self._trigger_stop(symbol, f'DD_EXCEEDED_{asset.max_dd*100:.1f}%')
            return
        
        # Consecutive losses
        if asset.consec_losses >= STOP_THRESHOLDS['max_consec_losses']:
            self._trigger_stop(symbol, f'CONSEC_LOSSES_{asset.consec_losses}')
            return
    
    def _trigger_stop(self, symbol: str, reason: str):
        """Trigger STOP for an asset."""
        if symbol in self.state.assets:
            self.state.assets[symbol].status = 'STOPPED'
        
        stop_msg = f"[STOP] {symbol}: {reason} at {datetime.now().isoformat()}"
        self.state.stops_triggered.append(stop_msg)
        self.state.alerts.append(stop_msg)
        print(f"🛑 {stop_msg}")
        
        self._save_state()
    
    def daily_check(self) -> Dict:
        """Run daily validation checks."""
        print("=" * 60)
        print(f" SHADOW DAILY CHECK - {datetime.now().strftime('%Y-%m-%d')}")
        print("=" * 60)
        
        report = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'assets': {},
            'alerts': [],
            'stops': [],
            'overall_status': 'OK',
        }
        
        for symbol, asset in self.state.assets.items():
            asset_report = self._check_asset(symbol, asset)
            report['assets'][symbol] = asset_report
            
            if asset_report.get('alerts'):
                report['alerts'].extend(asset_report['alerts'])
            
            if asset.status == 'STOPPED':
                report['stops'].append(symbol)
        
        # Overall status
        if report['stops']:
            report['overall_status'] = 'STOPPED'
        elif report['alerts']:
            report['overall_status'] = 'ALERT'
        
        # Print summary
        self._print_daily_summary(report)
        
        return report
    
    def _check_asset(self, symbol: str, asset: AssetState) -> Dict:
        """Check one asset."""
        alerts = []
        
        # Pattern dominance
        total_patterns = sum(asset.patterns.values())
        if total_patterns > 0:
            for pattern, count in asset.patterns.items():
                if count / total_patterns > ALERT_THRESHOLDS['pattern_dominance']:
                    alerts.append(f"{symbol}: {pattern} dominates ({count/total_patterns*100:.0f}%)")
        
        # Consecutive losses warning
        if asset.consec_losses >= ALERT_THRESHOLDS['consec_losses_warn']:
            alerts.append(f"{symbol}: {asset.consec_losses} consecutive losses")
        
        # Pattern win rates
        for pattern, count in asset.patterns.items():
            wins = asset.pattern_wins.get(pattern, 0)
            if count >= 5:  # Minimum sample
                wr = wins / count
                if wr < ALERT_THRESHOLDS['low_win_rate']:
                    alerts.append(f"{symbol}: {pattern} WR={wr*100:.0f}%")
        
        # Store alerts
        for alert in alerts:
            if alert not in self.state.alerts[-10:]:
                self.state.alerts.append(f"[ALERT] {alert}")
        
        total_trades = asset.wins + asset.losses
        win_rate = asset.wins / total_trades if total_trades > 0 else 0
        
        return {
            'status': asset.status,
            'signals': asset.signals,
            'trades': total_trades,
            'win_rate': win_rate,
            'pnl': asset.pnl,
            'max_dd': asset.max_dd,
            'consec_losses': asset.consec_losses,
            'alerts': alerts,
        }
    
    def _print_daily_summary(self, report: Dict):
        """Print daily summary."""
        print(f"\n Overall Status: {report['overall_status']}")
        print("\n Asset Summary:")
        print(f" {'Asset':<10} {'Status':<10} {'Trades':<8} {'WR':<8} {'DD':<8}")
        print("-" * 50)
        
        for symbol, data in report['assets'].items():
            print(f" {symbol:<10} {data['status']:<10} {data['trades']:<8} "
                  f"{data['win_rate']*100:.0f}%{'':<4} {data['max_dd']*100:.1f}%")
        
        if report['alerts']:
            print(f"\n Alerts ({len(report['alerts'])}):")
            for alert in report['alerts'][:5]:
                print(f"   ⚠️ {alert}")
        
        if report['stops']:
            print(f"\n STOPPED: {', '.join(report['stops'])}")
        
        print()
    
    def weekly_report(self) -> Dict:
        """Generate weekly report."""
        print("=" * 60)
        print(f" SHADOW WEEKLY REPORT - Week {self.state.current_week}")
        print("=" * 60)
        
        report = {
            'week': self.state.current_week,
            'assets': {},
            'pattern_health': {},
            'freeze_checklist': {},
        }
        
        # Asset metrics
        for symbol, asset in self.state.assets.items():
            total = asset.wins + asset.losses
            report['assets'][symbol] = {
                'status': asset.status,
                'trades': total,
                'win_rate': asset.wins / total if total > 0 else 0,
                'pf': self._calc_pf(asset),
                'max_dd': asset.max_dd,
            }
        
        # Pattern health
        all_patterns = {}
        for asset in self.state.assets.values():
            for pattern, count in asset.patterns.items():
                wins = asset.pattern_wins.get(pattern, 0)
                if pattern not in all_patterns:
                    all_patterns[pattern] = {'count': 0, 'wins': 0}
                all_patterns[pattern]['count'] += count
                all_patterns[pattern]['wins'] += wins
        
        for pattern, data in all_patterns.items():
            report['pattern_health'][pattern] = {
                'signals': data['count'],
                'win_rate': data['wins'] / data['count'] if data['count'] > 0 else 0,
            }
        
        # Print
        self._print_weekly(report)
        
        # Increment week
        self.state.current_week += 1
        self._save_state()
        
        return report
    
    def _calc_pf(self, asset: AssetState) -> float:
        """Calculate profit factor."""
        if asset.losses == 0:
            return 999 if asset.wins > 0 else 0
        # Simplified - assumes 1R wins and losses
        return asset.wins / asset.losses if asset.losses > 0 else 0
    
    def _print_weekly(self, report: Dict):
        """Print weekly report."""
        print("\n Asset Performance:")
        print(f" {'Asset':<10} {'Status':<10} {'Trades':<8} {'PF':<8} {'DD':<8} {'WR':<8}")
        print("-" * 60)
        
        for symbol, data in report['assets'].items():
            print(f" {symbol:<10} {data['status']:<10} {data['trades']:<8} "
                  f"{data['pf']:.2f}{'':<4} {data['max_dd']*100:.1f}%{'':<3} {data['win_rate']*100:.0f}%")
        
        print("\n Pattern Health:")
        for pattern, data in report['pattern_health'].items():
            wr = data['win_rate'] * 100
            status = '✅' if wr >= 30 else '⚠️'
            print(f"   {status} {pattern}: {data['signals']} signals, {wr:.0f}% WR")
        
        print()
    
    def check_freeze_ready(self) -> Dict:
        """Check if ready for Phase-1 freeze."""
        checklist = {
            '4_weeks_complete': self.state.current_week >= 4,
            'no_stops': len(self.state.stops_triggered) == 0,
            'eurusd_profitable': False,
            'sister_profitable': False,
            'max_dd_ok': True,
            'all_patterns_ok': True,
        }
        
        # EURUSD check
        if 'EURUSD' in self.state.assets:
            eurusd = self.state.assets['EURUSD']
            pf = self._calc_pf(eurusd)
            checklist['eurusd_profitable'] = pf >= 1.0
        
        # Sister asset check
        for symbol in ['GBPUSD', 'AUDUSD']:
            if symbol in self.state.assets:
                asset = self.state.assets[symbol]
                if self._calc_pf(asset) >= 0.95:
                    checklist['sister_profitable'] = True
                    break
        
        # DD check
        for asset in self.state.assets.values():
            if asset.max_dd > 0.12:
                checklist['max_dd_ok'] = False
        
        # Pattern check
        for asset in self.state.assets.values():
            for pattern, count in asset.patterns.items():
                if count >= 5:
                    wins = asset.pattern_wins.get(pattern, 0)
                    if wins / count < 0.30:
                        checklist['all_patterns_ok'] = False
        
        # Overall
        checklist['ready'] = all(checklist.values())
        
        return checklist


# ============================================================================
# CLI
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Shadow Mode Monitor')
    parser.add_argument('command', choices=['daily', 'weekly', 'freeze', 'status'])
    
    args = parser.parse_args()
    
    monitor = ShadowMonitor()
    
    if args.command == 'daily':
        monitor.daily_check()
    elif args.command == 'weekly':
        monitor.weekly_report()
    elif args.command == 'freeze':
        checklist = monitor.check_freeze_ready()
        print("\n FREEZE CHECKLIST:")
        for item, status in checklist.items():
            emoji = '✅' if status else '❌'
            print(f"   {emoji} {item}: {status}")
    elif args.command == 'status':
        print("\n SHADOW STATUS:")
        print(f" Started: {monitor.state.start_date}")
        print(f" Week: {monitor.state.current_week}")
        print(f" Stops: {len(monitor.state.stops_triggered)}")
        print(f" Alerts: {len(monitor.state.alerts)}")


if __name__ == '__main__':
    main()
