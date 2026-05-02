"""
SCOPUS Phase-1: Hard Guardrails

Non-negotiable safety rules that CANNOT be overridden.
These protect the account from catastrophic losses.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from datetime import datetime, timedelta
import numpy as np


# ============================================================================
# GUARDRAIL CONSTANTS
# ============================================================================

@dataclass(frozen=True)
class GuardrailLimits:
    """Non-negotiable safety limits."""
    
    # Portfolio-level limits
    MAX_PORTFOLIO_DD: float = 0.10       # 10% max portfolio drawdown
    MAX_DAILY_DD: float = 0.03           # 3% max daily drawdown
    MAX_WEEKLY_DD: float = 0.06          # 6% max weekly drawdown
    
    # Per-trade limits
    MAX_SINGLE_TRADE_RISK: float = 0.01  # 1% max risk per trade
    MAX_CORRELATED_EXPOSURE: float = 0.03  # 3% max in correlated positions
    
    # Per-asset limits
    PF_FLOOR: float = 0.8                # Profit factor floor
    MIN_WIN_RATE: float = 0.35           # Minimum acceptable win rate
    MAX_CONSEC_LOSSES: int = 8           # Max consecutive losses
    MAX_ASSET_DD: float = 0.15           # Max DD per asset
    
    # Size limits
    MAX_POSITION_SIZE_PCT: float = 2.0   # Max % of equity per position
    MIN_POSITION_SIZE_PCT: float = 0.1   # Min % (skip if below)
    MAX_OPEN_POSITIONS: int = 5          # Max simultaneous positions
    MAX_DAILY_TRADES: int = 10           # Max trades per day


LIMITS = GuardrailLimits()


# ============================================================================
# PORTFOLIO STATE TRACKER
# ============================================================================

@dataclass
class PortfolioState:
    """Current portfolio state for guardrail checks."""
    equity: float
    peak_equity: float
    daily_start_equity: float
    weekly_start_equity: float
    open_positions: List[Dict]
    daily_trades: int
    daily_pnl: float
    current_dd: float = 0.0
    daily_dd: float = 0.0
    weekly_dd: float = 0.0
    
    def update_drawdowns(self):
        """Update drawdown calculations."""
        if self.peak_equity > 0:
            self.current_dd = 1 - (self.equity / self.peak_equity)
        if self.daily_start_equity > 0:
            self.daily_dd = max(0, 1 - (self.equity / self.daily_start_equity))
        if self.weekly_start_equity > 0:
            self.weekly_dd = max(0, 1 - (self.equity / self.weekly_start_equity))


# ============================================================================
# HARD GUARDRAILS
# ============================================================================

class HardGuardrails:
    """
    Non-negotiable safety rules.
    These CANNOT be bypassed by any trading logic.
    """
    
    def __init__(self, limits: GuardrailLimits = None):
        self.limits = limits or LIMITS
    
    def check_trade(
        self, 
        trade: Dict, 
        portfolio_state: PortfolioState
    ) -> Tuple[bool, str]:
        """
        Check if a trade is allowed.
        
        Args:
            trade: Trade parameters (risk_pct, asset, etc.)
            portfolio_state: Current portfolio state
        
        Returns:
            (allowed, reason) tuple
        """
        # Update drawdowns
        portfolio_state.update_drawdowns()
        
        # 1. Portfolio DD check
        if portfolio_state.current_dd > self.limits.MAX_PORTFOLIO_DD:
            return False, f'PORTFOLIO_DD_EXCEEDED ({portfolio_state.current_dd:.1%} > {self.limits.MAX_PORTFOLIO_DD:.1%})'
        
        # 2. Daily DD check
        if portfolio_state.daily_dd > self.limits.MAX_DAILY_DD:
            return False, f'DAILY_DD_EXCEEDED ({portfolio_state.daily_dd:.1%} > {self.limits.MAX_DAILY_DD:.1%})'
        
        # 3. Weekly DD check
        if portfolio_state.weekly_dd > self.limits.MAX_WEEKLY_DD:
            return False, f'WEEKLY_DD_EXCEEDED ({portfolio_state.weekly_dd:.1%} > {self.limits.MAX_WEEKLY_DD:.1%})'
        
        # 4. Single trade risk check
        trade_risk = trade.get('risk_pct', 0.01)
        if trade_risk > self.limits.MAX_SINGLE_TRADE_RISK:
            return False, f'TRADE_RISK_EXCEEDED ({trade_risk:.1%} > {self.limits.MAX_SINGLE_TRADE_RISK:.1%})'
        
        # 5. Max open positions check
        if len(portfolio_state.open_positions) >= self.limits.MAX_OPEN_POSITIONS:
            return False, f'MAX_POSITIONS_REACHED ({len(portfolio_state.open_positions)} >= {self.limits.MAX_OPEN_POSITIONS})'
        
        # 6. Daily trades check
        if portfolio_state.daily_trades >= self.limits.MAX_DAILY_TRADES:
            return False, f'MAX_DAILY_TRADES_REACHED ({portfolio_state.daily_trades} >= {self.limits.MAX_DAILY_TRADES})'
        
        # 7. Correlation check
        if portfolio_state.open_positions:
            correlated_exposure = self._calculate_correlated_exposure(
                trade.get('asset', ''),
                trade_risk,
                portfolio_state.open_positions
            )
            if correlated_exposure > self.limits.MAX_CORRELATED_EXPOSURE:
                return False, f'CORRELATION_LIMIT ({correlated_exposure:.1%} > {self.limits.MAX_CORRELATED_EXPOSURE:.1%})'
        
        # 8. Position size check
        size_pct = trade.get('size_pct', 1.0)
        if size_pct > self.limits.MAX_POSITION_SIZE_PCT:
            return False, f'SIZE_EXCEEDED ({size_pct:.1%} > {self.limits.MAX_POSITION_SIZE_PCT:.1%})'
        if size_pct < self.limits.MIN_POSITION_SIZE_PCT:
            return False, f'SIZE_TOO_SMALL ({size_pct:.1%} < {self.limits.MIN_POSITION_SIZE_PCT:.1%})'
        
        return True, 'ALLOWED'
    
    def _calculate_correlated_exposure(
        self,
        asset: str,
        new_risk: float,
        open_positions: List[Dict]
    ) -> float:
        """Calculate total exposure in correlated positions."""
        
        # Simple correlation groups
        CORRELATION_GROUPS = {
            'USD_LONG': ['EURUSD_SELL', 'GBPUSD_SELL', 'AUDUSD_SELL', 'USDJPY_BUY', 'USDCHF_BUY', 'USDCAD_BUY'],
            'USD_SHORT': ['EURUSD_BUY', 'GBPUSD_BUY', 'AUDUSD_BUY', 'USDJPY_SELL', 'USDCHF_SELL', 'USDCAD_SELL'],
            'JPY_LONG': ['USDJPY_SELL', 'EURJPY_SELL', 'GBPJPY_SELL'],
            'JPY_SHORT': ['USDJPY_BUY', 'EURJPY_BUY', 'GBPJPY_BUY'],
            'RISK_ON': ['SPX500_BUY', 'NAS100_BUY', 'AUDUSD_BUY', 'BTCUSD_BUY'],
            'RISK_OFF': ['XAUUSD_BUY', 'USDJPY_SELL', 'USDCHF_BUY'],
        }
        
        # Find which groups the new trade belongs to
        asset_key = asset.upper()
        new_groups = set()
        for group, members in CORRELATION_GROUPS.items():
            if any(asset_key in m for m in members):
                new_groups.add(group)
        
        # Sum exposure in same groups
        total_exposure = new_risk
        for pos in open_positions:
            pos_asset = pos.get('asset', '').upper()
            pos_risk = pos.get('risk_pct', 0.01)
            
            pos_groups = set()
            for group, members in CORRELATION_GROUPS.items():
                if any(pos_asset in m for m in members):
                    pos_groups.add(group)
            
            # If any overlap in groups, add exposure
            if new_groups & pos_groups:
                total_exposure += pos_risk
        
        return total_exposure
    
    def enforce_limits(self, trade: Dict) -> Dict:
        """
        Enforce limits on trade parameters without blocking.
        
        Args:
            trade: Trade parameters
        
        Returns:
            Trade with enforced limits
        """
        enforced = trade.copy()
        
        # Cap risk
        if 'risk_pct' in enforced:
            enforced['risk_pct'] = min(
                enforced['risk_pct'], 
                self.limits.MAX_SINGLE_TRADE_RISK
            )
        
        # Cap size
        if 'size_pct' in enforced:
            enforced['size_pct'] = min(
                enforced['size_pct'],
                self.limits.MAX_POSITION_SIZE_PCT
            )
            # Zero out if below minimum
            if enforced['size_pct'] < self.limits.MIN_POSITION_SIZE_PCT:
                enforced['size_pct'] = 0
                enforced['_skip_reason'] = 'SIZE_BELOW_MINIMUM'
        
        return enforced


# ============================================================================
# ASSET MONITOR
# ============================================================================

class AssetMonitor:
    """
    Monitor per-asset performance and enforce limits.
    Tracks drawdown, profit factor, and consecutive losses.
    """
    
    # Status levels
    STATUS_ACTIVE = 'ACTIVE'
    STATUS_SHADOW = 'SHADOW'      # Paper trade only
    STATUS_BLOCKED = 'BLOCKED'    # No trading allowed
    
    def __init__(self, limits: GuardrailLimits = None):
        self.limits = limits or LIMITS
        self.asset_stats: Dict[str, Dict] = defaultdict(lambda: {
            'trades': [],
            'equity_curve': [1.0],
            'status': self.STATUS_ACTIVE,
            'block_reason': None,
            'last_updated': None,
        })
    
    def update(self, asset: str, trade_result_pct: float):
        """
        Update asset statistics with a trade result.
        
        Args:
            asset: Asset symbol
            trade_result_pct: Trade result as decimal (e.g., 0.02 for 2%)
        """
        stats = self.asset_stats[asset]
        stats['trades'].append(trade_result_pct)
        
        # Update equity curve
        current = stats['equity_curve'][-1]
        stats['equity_curve'].append(current * (1 + trade_result_pct))
        stats['last_updated'] = datetime.now().isoformat()
        
        # Evaluate status
        self._evaluate_status(asset)
    
    def _evaluate_status(self, asset: str):
        """Evaluate and update asset status."""
        stats = self.asset_stats[asset]
        trades = stats['trades']
        equity = stats['equity_curve']
        
        if len(trades) < 5:
            # Not enough data
            return
        
        # Calculate metrics
        peak = max(equity)
        current = equity[-1]
        dd = 1 - (current / peak) if peak > 0 else 0
        
        wins = [t for t in trades if t > 0]
        losses = [t for t in trades if t < 0]
        
        win_rate = len(wins) / len(trades) if trades else 0.5
        
        total_wins = sum(wins) if wins else 0
        total_losses = abs(sum(losses)) if losses else 1
        pf = total_wins / total_losses if total_losses > 0 else 999
        
        consec_losses = self._count_consecutive_losses(trades)
        
        # Evaluate against limits
        if (dd > self.limits.MAX_ASSET_DD or 
            pf < 0.5 or 
            consec_losses > self.limits.MAX_CONSEC_LOSSES + 2):
            stats['status'] = self.STATUS_BLOCKED
            stats['block_reason'] = f'DD={dd:.1%}, PF={pf:.2f}, ConsecL={consec_losses}'
        
        elif (dd > self.limits.MAX_ASSET_DD * 0.7 or 
              pf < self.limits.PF_FLOOR or 
              win_rate < self.limits.MIN_WIN_RATE or
              consec_losses > self.limits.MAX_CONSEC_LOSSES - 1):
            stats['status'] = self.STATUS_SHADOW
            stats['block_reason'] = f'DD={dd:.1%}, PF={pf:.2f}, WR={win_rate:.1%}'
        
        else:
            stats['status'] = self.STATUS_ACTIVE
            stats['block_reason'] = None
    
    def _count_consecutive_losses(self, trades: List[float]) -> int:
        """Count current consecutive losing streak."""
        if not trades:
            return 0
        
        count = 0
        for t in reversed(trades):
            if t < 0:
                count += 1
            else:
                break
        return count
    
    def can_trade(self, asset: str) -> Tuple[bool, str]:
        """
        Check if trading is allowed for an asset.
        
        Returns:
            (allowed, status) tuple
        """
        status = self.asset_stats[asset]['status']
        
        if status == self.STATUS_BLOCKED:
            reason = self.asset_stats[asset].get('block_reason', 'BLOCKED')
            return False, f'ASSET_BLOCKED: {reason}'
        
        elif status == self.STATUS_SHADOW:
            return True, 'SHADOW_MODE'
        
        return True, 'ACTIVE'
    
    def get_status(self, asset: str) -> str:
        """Get current status for an asset."""
        return self.asset_stats[asset]['status']
    
    def get_all_statuses(self) -> Dict[str, str]:
        """Get statuses for all tracked assets."""
        return {
            asset: stats['status'] 
            for asset, stats in self.asset_stats.items()
        }
    
    def get_asset_metrics(self, asset: str) -> Dict:
        """Get detailed metrics for an asset."""
        stats = self.asset_stats[asset]
        trades = stats['trades']
        equity = stats['equity_curve']
        
        if not trades:
            return {'status': stats['status'], 'trades': 0}
        
        peak = max(equity)
        current = equity[-1]
        dd = 1 - (current / peak) if peak > 0 else 0
        
        wins = [t for t in trades if t > 0]
        losses = [t for t in trades if t < 0]
        
        return {
            'status': stats['status'],
            'block_reason': stats['block_reason'],
            'trades': len(trades),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': len(wins) / len(trades) if trades else 0,
            'total_return': (current - 1) * 100,
            'max_drawdown': dd * 100,
            'profit_factor': sum(wins) / abs(sum(losses)) if losses else 999,
            'consec_losses': self._count_consecutive_losses(trades),
        }
    
    def reset_asset(self, asset: str):
        """Reset asset to ACTIVE status."""
        if asset in self.asset_stats:
            self.asset_stats[asset]['status'] = self.STATUS_ACTIVE
            self.asset_stats[asset]['block_reason'] = None


# ============================================================================
# COMBINED GUARDRAIL SYSTEM
# ============================================================================

class GuardrailSystem:
    """
    Combined guardrail system with all safety checks.
    """
    
    def __init__(self):
        self.hard_guardrails = HardGuardrails()
        self.asset_monitor = AssetMonitor()
    
    def check_trade(
        self,
        trade: Dict,
        portfolio_state: PortfolioState,
    ) -> Tuple[bool, str, str]:
        """
        Full guardrail check.
        
        Returns:
            (allowed, reason, mode) tuple
            mode is 'ACTIVE', 'SHADOW', or 'BLOCKED'
        """
        asset = trade.get('asset', '')
        
        # Check hard guardrails first
        allowed, reason = self.hard_guardrails.check_trade(trade, portfolio_state)
        if not allowed:
            return False, reason, 'BLOCKED'
        
        # Check asset-specific status
        can_trade, asset_status = self.asset_monitor.can_trade(asset)
        if not can_trade:
            return False, asset_status, 'BLOCKED'
        
        if 'SHADOW' in asset_status:
            return True, 'SHADOW_MODE', 'SHADOW'
        
        return True, 'ALLOWED', 'ACTIVE'
    
    def record_trade(self, asset: str, result_pct: float):
        """Record a trade result for monitoring."""
        self.asset_monitor.update(asset, result_pct)
    
    def get_system_status(self) -> Dict:
        """Get overall system status."""
        statuses = self.asset_monitor.get_all_statuses()
        return {
            'total_assets': len(statuses),
            'active': sum(1 for s in statuses.values() if s == 'ACTIVE'),
            'shadow': sum(1 for s in statuses.values() if s == 'SHADOW'),
            'blocked': sum(1 for s in statuses.values() if s == 'BLOCKED'),
            'assets': statuses,
        }
