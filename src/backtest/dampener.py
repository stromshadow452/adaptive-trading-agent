"""
Loss-Streak Dampener

Deterministic risk throttling module that reduces position sizing
during consecutive loss clusters.

State Machine:
    NORMAL → CAUTION → COOLDOWN → PAUSED → RECOVERY → NORMAL

Key Features:
- Per-asset state management (isolated failures)
- Gradual size reduction (not binary)
- Time-based cooldown after PAUSED
- Gradual recovery (not instant full-size)
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Optional, Tuple


class DampenerState(Enum):
    """Dampener state machine states."""
    NORMAL = 'NORMAL'       # Full size (1.0x)
    CAUTION = 'CAUTION'     # Reduced (0.5x)
    COOLDOWN = 'COOLDOWN'   # Minimal (0.25x)
    PAUSED = 'PAUSED'       # No trading (0x)
    RECOVERY = 'RECOVERY'   # Rebuilding (0.5x)


# ============================================================================
# DEFAULT CONFIGURATION
# ============================================================================

DAMPENER_CONFIG = {
    # Consecutive loss thresholds
    'CAUTION_THRESHOLD': 2,      # REVERTED: Keep at 2 to avoid over-dampening
    'COOLDOWN_THRESHOLD': 4,     # Enter COOLDOWN after 4 losses
    'PAUSED_THRESHOLD': 6,       # Enter PAUSED after 6 losses
    
    # Size multipliers
    'NORMAL_SIZE': 1.0,
    'CAUTION_SIZE': 0.4,         # TUNED: Was 0.5 - reduce size more in caution
    'COOLDOWN_SIZE': 0.25,
    'PAUSED_SIZE': 0.0,
    'RECOVERY_SIZE': 0.5,
    
    # Time-based cooldown (hours)
    'PAUSED_DURATION_HOURS': 8,
    
    # Recovery conditions
    'RECOVERY_WINS_TO_NORMAL': 2,
    'RECOVERY_LOSSES_TO_COOLDOWN': 2,
    
    # Wins needed to upgrade state
    'WINS_TO_UPGRADE': 2,
}


# ============================================================================
# PER-ASSET STATE
# ============================================================================

@dataclass
class AssetDampenerState:
    """Per-asset dampener state tracking."""
    state: DampenerState = DampenerState.NORMAL
    consecutive_losses: int = 0
    consecutive_wins: int = 0
    paused_since: Optional[datetime] = None
    recovery_wins: int = 0
    recovery_losses: int = 0
    
    # Statistics
    total_trades: int = 0
    total_dampened_trades: int = 0
    pause_count: int = 0
    
    def reset(self):
        """Reset to initial state."""
        self.state = DampenerState.NORMAL
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.paused_since = None
        self.recovery_wins = 0
        self.recovery_losses = 0


# ============================================================================
# LOSS-STREAK DAMPENER
# ============================================================================

class LossStreakDampener:
    """
    Loss-streak dampener with per-asset state management.
    
    Reduces position size during consecutive loss clusters.
    Deterministic, regime-aware, reversible.
    
    Usage:
        dampener = LossStreakDampener()
        
        # Before trade
        can_trade, reason = dampener.can_trade(symbol, current_time)
        if can_trade:
            size_mult = dampener.get_size_multiplier(symbol)
            # Apply size_mult to base position size
        
        # After trade closes
        dampener.record_trade_result(symbol, is_win, exit_time)
    """
    
    def __init__(self, config: dict = None):
        """
        Initialize dampener.
        
        Args:
            config: Optional configuration override
        """
        self.config = {**DAMPENER_CONFIG, **(config or {})}
        self.asset_states: Dict[str, AssetDampenerState] = {}
        
        # Global statistics
        self.total_pauses = 0
        self.total_dampened = 0
    
    def get_state(self, symbol: str) -> AssetDampenerState:
        """Get or create asset state."""
        if symbol not in self.asset_states:
            self.asset_states[symbol] = AssetDampenerState()
        return self.asset_states[symbol]
    
    def get_size_multiplier(self, symbol: str) -> float:
        """
        Get current size multiplier for asset.
        
        Returns:
            float: Size multiplier (0.0 to 1.0)
        """
        state = self.get_state(symbol)
        
        size_map = {
            DampenerState.NORMAL: self.config['NORMAL_SIZE'],
            DampenerState.CAUTION: self.config['CAUTION_SIZE'],
            DampenerState.COOLDOWN: self.config['COOLDOWN_SIZE'],
            DampenerState.PAUSED: self.config['PAUSED_SIZE'],
            DampenerState.RECOVERY: self.config['RECOVERY_SIZE'],
        }
        
        return size_map.get(state.state, 0.0)
    
    def can_trade(
        self, 
        symbol: str, 
        current_time: datetime = None
    ) -> Tuple[bool, str]:
        """
        Check if trading is allowed for this asset.
        
        Args:
            symbol: Asset symbol
            current_time: Current timestamp
            
        Returns:
            (can_trade, reason)
        """
        state = self.get_state(symbol)
        current_time = current_time or datetime.now()
        
        # Check if still paused
        if state.state == DampenerState.PAUSED:
            if state.paused_since:
                hours_paused = (current_time - state.paused_since).total_seconds() / 3600
                if hours_paused < self.config['PAUSED_DURATION_HOURS']:
                    remaining = self.config['PAUSED_DURATION_HOURS'] - hours_paused
                    return False, f"PAUSED ({remaining:.1f}h remaining)"
                else:
                    # Transition to RECOVERY
                    self._transition_to_recovery(symbol)
        
        # After potential transition, check again
        if state.state == DampenerState.PAUSED:
            return False, "PAUSED"
        
        return True, state.state.value
    
    def record_trade_result(
        self,
        symbol: str,
        is_win: bool,
        current_time: datetime = None
    ) -> DampenerState:
        """
        Record trade result and update state machine.
        
        Args:
            symbol: Asset symbol
            is_win: True if trade was profitable
            current_time: Trade close timestamp
            
        Returns:
            New state after update
        """
        state = self.get_state(symbol)
        current_time = current_time or datetime.now()
        
        # Track statistics
        state.total_trades += 1
        if state.state != DampenerState.NORMAL:
            state.total_dampened_trades += 1
            self.total_dampened += 1
        
        if is_win:
            self._handle_win(symbol, current_time)
        else:
            self._handle_loss(symbol, current_time)
        
        return self.get_state(symbol).state
    
    def _handle_win(self, symbol: str, current_time: datetime):
        """Handle winning trade."""
        state = self.get_state(symbol)
        
        state.consecutive_wins += 1
        state.consecutive_losses = 0
        
        if state.state == DampenerState.RECOVERY:
            state.recovery_wins += 1
            if state.recovery_wins >= self.config['RECOVERY_WINS_TO_NORMAL']:
                self._transition_to_normal(symbol)
        
        elif state.state == DampenerState.COOLDOWN:
            # From COOLDOWN: 2 wins → CAUTION
            if state.consecutive_wins >= self.config['WINS_TO_UPGRADE']:
                self._transition_to_caution(symbol)
        
        elif state.state == DampenerState.CAUTION:
            # From CAUTION: 2 wins → NORMAL
            if state.consecutive_wins >= self.config['WINS_TO_UPGRADE']:
                self._transition_to_normal(symbol)
    
    def _handle_loss(self, symbol: str, current_time: datetime):
        """Handle losing trade."""
        state = self.get_state(symbol)
        
        state.consecutive_losses += 1
        state.consecutive_wins = 0
        
        if state.state == DampenerState.RECOVERY:
            state.recovery_losses += 1
            if state.recovery_losses >= self.config['RECOVERY_LOSSES_TO_COOLDOWN']:
                self._transition_to_cooldown(symbol, current_time)
        
        else:
            # Check thresholds for state transitions (in order of severity)
            if state.consecutive_losses >= self.config['PAUSED_THRESHOLD']:
                self._transition_to_paused(symbol, current_time)
            elif state.consecutive_losses >= self.config['COOLDOWN_THRESHOLD']:
                self._transition_to_cooldown(symbol, current_time)
            elif state.consecutive_losses >= self.config['CAUTION_THRESHOLD']:
                self._transition_to_caution(symbol)
    
    def _transition_to_normal(self, symbol: str):
        """Reset to normal state."""
        state = self.get_state(symbol)
        state.state = DampenerState.NORMAL
        state.consecutive_losses = 0
        state.consecutive_wins = 0
        state.paused_since = None
        state.recovery_wins = 0
        state.recovery_losses = 0
    
    def _transition_to_caution(self, symbol: str):
        """Enter caution state."""
        state = self.get_state(symbol)
        state.state = DampenerState.CAUTION
        # Reset win counter for next upgrade
        state.consecutive_wins = 0
    
    def _transition_to_cooldown(self, symbol: str, current_time: datetime):
        """Enter cooldown state."""
        state = self.get_state(symbol)
        state.state = DampenerState.COOLDOWN
        state.consecutive_wins = 0
    
    def _transition_to_paused(self, symbol: str, current_time: datetime):
        """Enter paused state."""
        state = self.get_state(symbol)
        state.state = DampenerState.PAUSED
        state.paused_since = current_time
        state.pause_count += 1
        self.total_pauses += 1
    
    def _transition_to_recovery(self, symbol: str):
        """Enter recovery state after pause ends."""
        state = self.get_state(symbol)
        state.state = DampenerState.RECOVERY
        state.consecutive_losses = 0
        state.consecutive_wins = 0
        state.recovery_wins = 0
        state.recovery_losses = 0
        state.paused_since = None
    
    def get_summary(self) -> dict:
        """
        Get dampener summary statistics.
        
        Returns:
            Summary dict with key metrics
        """
        total_trades = sum(s.total_trades for s in self.asset_states.values())
        
        state_counts = {}
        for symbol, asset_state in self.asset_states.items():
            state_name = asset_state.state.value
            state_counts[state_name] = state_counts.get(state_name, 0) + 1
        
        return {
            'total_pauses': self.total_pauses,
            'total_dampened_trades': self.total_dampened,
            'dampened_pct': self.total_dampened / max(total_trades, 1) * 100,
            'assets_by_state': state_counts,
            'per_asset': {
                symbol: {
                    'state': s.state.value,
                    'consecutive_losses': s.consecutive_losses,
                    'pause_count': s.pause_count,
                    'trades': s.total_trades,
                    'dampened': s.total_dampened_trades,
                }
                for symbol, s in self.asset_states.items()
            }
        }
    
    def reset_all(self):
        """Reset all asset states."""
        for state in self.asset_states.values():
            state.reset()
        self.total_pauses = 0
        self.total_dampened = 0


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def calculate_dampened_size(
    base_risk: float,
    asset_tier_mult: float,
    dampener_mult: float,
    circuit_breaker_active: bool = False
) -> float:
    """
    Calculate final position size with all modifiers.
    
    Order of application:
    1. Base risk %
    2. Asset tier multiplier
    3. Dampener multiplier
    4. Circuit breaker override
    
    Args:
        base_risk: Base risk percentage (e.g., 0.005 for 0.5%)
        asset_tier_mult: Asset tier size multiplier
        dampener_mult: Dampener size multiplier
        circuit_breaker_active: If True, returns 0
        
    Returns:
        Final risk percentage
    """
    if circuit_breaker_active:
        return 0.0
    
    return base_risk * asset_tier_mult * dampener_mult


def estimate_dd_reduction(
    consecutive_losses: int,
    base_risk_pct: float = 0.5
) -> Tuple[float, float]:
    """
    Estimate DD reduction with dampener.
    
    Args:
        consecutive_losses: Number of losses in streak
        base_risk_pct: Risk per trade in percent
        
    Returns:
        (dd_without_dampener, dd_with_dampener)
    """
    # Without dampener: all at full size
    dd_without = consecutive_losses * base_risk_pct
    
    # With dampener
    dd_with = 0.0
    risk = base_risk_pct
    
    for i in range(consecutive_losses):
        if i < 2:
            mult = 1.0  # NORMAL
        elif i < 4:
            mult = 0.5  # CAUTION
        elif i < 6:
            mult = 0.25  # COOLDOWN
        else:
            mult = 0.0  # PAUSED
        
        dd_with += risk * mult
    
    return dd_without, dd_with
