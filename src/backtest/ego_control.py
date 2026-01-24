"""
MARK-2 Ego Control Module

Prevents overconfidence by detecting behavioral signals:
- Win streaks
- High trade frequency
- Repeated same-setup trades
- Elevated confidence levels

EFFECTS:
- Reduces position size
- Increases trade cooldown
- Raises minimum confidence threshold

"The suit that throttles itself before the pilot does something stupid."
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, List
from datetime import datetime, timedelta
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """Record of a single trade for ego tracking."""
    is_win: bool
    confidence: float
    regime: str
    side: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


class EgoControl:
    """
    MARK-2 Ego Control System
    
    Detects overconfidence from behavioral signals:
    - Win streaks
    - Trade frequency
    - Same-setup repetition
    - Confidence inflation
    
    Effects:
    - Size reduction (up to 49%)
    - Cooldown increase (up to 3x)
    - Confidence boost (up to +0.15)
    
    Never blocks. Only slows down.
    """
    
    # Rolling window sizes
    RECENT_TRADES_WINDOW = 10
    FREQUENCY_WINDOW_HOURS = 4
    
    # Base cooldown
    BASE_COOLDOWN_MINUTES = 15
    
    # Decay rates
    TIME_DECAY_RATE = 0.1  # per hour
    LOSS_DECAY_MULTIPLIER = 0.7  # per loss
    
    def __init__(self):
        self.win_streak: int = 0
        self.recent_trades: deque = deque(maxlen=self.RECENT_TRADES_WINDOW)
        self.last_trade_time: Optional[datetime] = None
        self.same_setup_count: int = 0
        self.last_setup: Optional[Tuple[str, str]] = None  # (regime, side)
        self.ego_score: float = 0.0
        self._last_decay_time: datetime = datetime.utcnow()
        
        logger.info("EgoControl initialized - Overconfidence prevention active")
    
    def update_on_trade(
        self,
        is_win: bool,
        confidence: float,
        regime: str,
        side: str,
        loss_streak: int = 0
    ):
        """
        Called after each trade closes.
        Updates all ego signals and recalculates ego score.
        """
        now = datetime.utcnow()
        
        # Update win streak
        if is_win:
            self.win_streak += 1
        else:
            # Loss-based ego decay
            decay_power = max(1, loss_streak)
            self.ego_score *= (self.LOSS_DECAY_MULTIPLIER ** decay_power)
            self.win_streak = 0
            logger.debug(f"EGO: Reality check - ego dropped to {self.ego_score:.2f}")
        
        # Track setup diversity
        current_setup = (regime, side)
        if current_setup == self.last_setup:
            self.same_setup_count += 1
        else:
            self.same_setup_count = 0
        self.last_setup = current_setup
        
        # Add to recent trades
        self.recent_trades.append(TradeRecord(
            is_win=is_win,
            confidence=confidence,
            regime=regime,
            side=side,
            timestamp=now
        ))
        
        self.last_trade_time = now
        
        # Recalculate ego score
        self._recalculate_ego_score()
    
    def _recalculate_ego_score(self):
        """Compute ego score from behavioral signals."""
        
        # Apply time decay first
        self._apply_time_decay()
        
        # === COMPONENT 1: Streak ego (0.0 - 0.3) ===
        streak_ego = min(0.3, self.win_streak * 0.06)
        
        # === COMPONENT 2: Winrate ego (0.0 - 0.25) ===
        if len(self.recent_trades) > 0:
            wins = sum(1 for t in self.recent_trades if t.is_win)
            winrate = wins / len(self.recent_trades)
            winrate_ego = max(0.0, (winrate - 0.5) * 0.5)
        else:
            winrate_ego = 0.0
        
        # === COMPONENT 3: Confidence inflation (0.0 - 0.2) ===
        if len(self.recent_trades) > 0:
            avg_conf = sum(t.confidence for t in self.recent_trades) / len(self.recent_trades)
            conf_ego = max(0.0, (avg_conf - 0.7) * 0.67)
        else:
            conf_ego = 0.0
        
        # === COMPONENT 4: Frequency ego (0.0 - 0.15) ===
        freq = self._calculate_trade_frequency()
        freq_ego = min(0.15, max(0.0, (freq - 2.0) * 0.05))
        
        # === COMPONENT 5: Tunnel vision (0.0 - 0.1) ===
        tunnel_ego = min(0.1, self.same_setup_count * 0.025)
        
        # === FINAL EGO SCORE ===
        self.ego_score = streak_ego + winrate_ego + conf_ego + freq_ego + tunnel_ego
        self.ego_score = min(1.0, self.ego_score)
        
        if self.ego_score > 0.3:
            logger.info(
                f"EGO: Score={self.ego_score:.2f} "
                f"(streak={streak_ego:.2f}, wr={winrate_ego:.2f}, "
                f"conf={conf_ego:.2f}, freq={freq_ego:.2f}, tunnel={tunnel_ego:.2f})"
            )
    
    def _calculate_trade_frequency(self) -> float:
        """Calculate trades per hour in recent window."""
        now = datetime.utcnow()
        cutoff = now - timedelta(hours=self.FREQUENCY_WINDOW_HOURS)
        
        recent_count = sum(
            1 for t in self.recent_trades
            if t.timestamp > cutoff
        )
        
        return recent_count / self.FREQUENCY_WINDOW_HOURS
    
    def _apply_time_decay(self):
        """Apply time-based ego decay."""
        now = datetime.utcnow()
        hours_elapsed = (now - self._last_decay_time).total_seconds() / 3600
        
        if hours_elapsed > 0.1:  # At least 6 minutes
            self.ego_score *= (1.0 - self.TIME_DECAY_RATE) ** hours_elapsed
            self._last_decay_time = now
    
    def get_ego_modifiers(self) -> Tuple[float, float, float]:
        """
        Returns (size_modifier, cooldown_multiplier, confidence_boost).
        
        size_modifier: 0.51 - 1.0
        cooldown_multiplier: 1.0 - 3.0
        confidence_boost: 0.0 - 0.15
        """
        # Apply time decay
        self._apply_time_decay()
        
        # === SIZE MODIFIER ===
        if self.ego_score > 0.3:
            size_mod = 1.0 - (self.ego_score - 0.3) * 0.7
            # At ego 0.3 → 1.0
            # At ego 0.7 → 0.72
            # At ego 1.0 → 0.51
        else:
            size_mod = 1.0
        
        # === COOLDOWN MULTIPLIER ===
        if self.ego_score > 0.5:
            cooldown_mult = 1.0 + (self.ego_score - 0.5) * 4.0
            # At ego 0.5 → 1.0x
            # At ego 0.7 → 1.8x
            # At ego 1.0 → 3.0x
        else:
            cooldown_mult = 1.0
        
        # === CONFIDENCE BOOST ===
        if self.ego_score > 0.4:
            conf_boost = (self.ego_score - 0.4) * 0.25
            # At ego 0.4 → 0.0
            # At ego 0.7 → 0.075
            # At ego 1.0 → 0.15
        else:
            conf_boost = 0.0
        
        return (size_mod, cooldown_mult, conf_boost)
    
    def can_trade_now(self) -> bool:
        """
        Check if cooldown has elapsed.
        Returns True if agent can trade.
        
        Cooldown only applies when ego_score > 0.3 (overconfidence detected).
        """
        # No cooldown if ego is low (not overconfident)
        if self.ego_score <= 0.3:
            return True
        
        if self.last_trade_time is None:
            return True
        
        _, cooldown_mult, _ = self.get_ego_modifiers()
        effective_cooldown = self.BASE_COOLDOWN_MINUTES * cooldown_mult
        
        minutes_since = (datetime.utcnow() - self.last_trade_time).total_seconds() / 60
        
        if minutes_since < effective_cooldown:
            logger.debug(
                f"EGO: Cooldown active (ego={self.ego_score:.2f}) - {effective_cooldown - minutes_since:.1f} min remaining"
            )
            return False
        
        return True
    
    def get_effective_min_confidence(self, base_conf: float) -> float:
        """
        Get minimum confidence with ego boost applied.
        Caps at 0.9.
        """
        _, _, conf_boost = self.get_ego_modifiers()
        return min(0.9, base_conf + conf_boost)
    
    def get_state(self) -> dict:
        """Get current ego state for logging."""
        size_mod, cooldown_mult, conf_boost = self.get_ego_modifiers()
        
        return {
            'ego_score': self.ego_score,
            'win_streak': self.win_streak,
            'same_setup_count': self.same_setup_count,
            'size_modifier': size_mod,
            'cooldown_multiplier': cooldown_mult,
            'confidence_boost': conf_boost,
        }
    
    def reset(self):
        """Reset ego state (for testing or new session)."""
        self.win_streak = 0
        self.recent_trades.clear()
        self.last_trade_time = None
        self.same_setup_count = 0
        self.last_setup = None
        self.ego_score = 0.0
        self._last_decay_time = datetime.utcnow()
        logger.info("EGO: Reset complete")
