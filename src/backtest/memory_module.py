"""
MARK-2 Memory Module

Stores and manages agent's pain memories:
- Pain Zones: Price levels with loss clusters
- Regime Pain: Regime+side failure tracking
- Loss Clusters: Time-based loss streaks

CORE RULE: Memory only DEGRADES size, never BLOCKS.

"The suit learns from near-deaths, not victories."
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import deque
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class PainZone:
    """Price zone where losses clustered."""
    price_center: float
    price_range: float  # ± range from center
    loss_count: int = 1
    total_loss_r: float = 0.0
    last_touched: datetime = field(default_factory=datetime.utcnow)
    decay_factor: float = 0.5  # 0.0 to 1.0


@dataclass
class RegimePainRecord:
    """Tracks failures in specific regime+side combinations."""
    regime: str
    side: str
    loss_streak: int = 0
    total_losses: int = 0
    total_wins: int = 0
    pain_score: float = 0.0  # 0.0 to 1.0
    last_updated: datetime = field(default_factory=datetime.utcnow)


class LossCluster:
    """Detects when losses cluster in time."""
    
    def __init__(self, buffer_size: int = 5):
        self.results: deque = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
    
    def add_result(self, is_loss: bool):
        """Add trade result (True = loss, False = win)."""
        self.results.append(is_loss)
    
    def count_losses(self) -> int:
        """Count losses in buffer."""
        return sum(1 for r in self.results if r)
    
    @property
    def cluster_active(self) -> bool:
        """True if 3+ losses in buffer."""
        return self.count_losses() >= 3
    
    @property
    def cluster_severity(self) -> float:
        """0.0 to 1.0 based on loss density."""
        if len(self.results) == 0:
            return 0.0
        return self.count_losses() / len(self.results)


# ============================================================================
# MEMORY MODULE
# ============================================================================

class MemoryModule:
    """
    MARK-2 Pain Memory System
    
    Remembers:
    - Price zones where losses cluster
    - Regimes where mistakes repeat
    - Recent loss clusters
    
    NEVER blocks. Only reduces size.
    """
    
    # Limits
    MAX_PAIN_ZONES = 20
    
    # Decay half-lives (hours)
    PAIN_ZONE_HALF_LIFE = 48.0
    REGIME_PAIN_HALF_LIFE = 24.0
    
    # Zone parameters
    ATR_ZONE_MULTIPLIER = 2.0
    
    def __init__(self):
        self.pain_zones: List[PainZone] = []
        self.regime_pain: Dict[Tuple[str, str], RegimePainRecord] = {}
        self.loss_cluster = LossCluster(buffer_size=5)
        
        logger.info("MemoryModule initialized - Pain memory active")
    
    def record_trade_result(
        self,
        entry_price: float,
        atr: float,
        regime: str,
        side: str,
        is_loss: bool,
        r_multiple: float = 0.0
    ):
        """
        Record trade outcome in memory.
        Called after each trade closes.
        """
        # Update loss cluster
        self.loss_cluster.add_result(is_loss)
        
        if is_loss:
            # Record pain zone
            self._add_or_update_pain_zone(entry_price, atr, abs(r_multiple))
            
            # Record regime pain
            self._update_regime_pain(regime, side, is_loss=True)
            
            logger.debug(f"MEMORY: Recorded loss at {entry_price:.5f} in {regime}/{side}")
        else:
            # Heal pain zone if exists
            self._heal_pain_zone(entry_price)
            
            # Reduce regime pain
            self._update_regime_pain(regime, side, is_loss=False)
    
    def get_memory_modifier(
        self,
        price: float,
        regime: str,
        side: str
    ) -> float:
        """
        Get combined size modifier from all memory types.
        Returns 0.1 to 1.0 (never zero).
        """
        # Apply time decay first
        self._apply_decay()
        
        # Calculate individual modifiers
        zone_mod = self._calculate_pain_zone_modifier(price)
        regime_mod = self._calculate_regime_pain_modifier(regime, side)
        cluster_mod = self._calculate_cluster_modifier()
        
        # Combine multiplicatively
        combined = zone_mod * regime_mod * cluster_mod
        
        # Floor at 10%
        combined = max(0.1, combined)
        
        if combined < 1.0:
            logger.info(
                f"MEMORY: Modifier={combined:.2f} "
                f"(zone={zone_mod:.2f}, regime={regime_mod:.2f}, cluster={cluster_mod:.2f})"
            )
        
        return combined
    
    def _add_or_update_pain_zone(self, price: float, atr: float, loss_r: float):
        """Add new pain zone or strengthen existing one."""
        zone_width = atr * self.ATR_ZONE_MULTIPLIER
        
        # Check if price is in existing zone
        for zone in self.pain_zones:
            if abs(zone.price_center - price) < zone.price_range:
                # Strengthen existing zone
                zone.loss_count += 1
                zone.total_loss_r += loss_r
                zone.decay_factor = min(1.0, zone.decay_factor + 0.3)
                zone.last_touched = datetime.utcnow()
                return
        
        # Create new zone
        if len(self.pain_zones) >= self.MAX_PAIN_ZONES:
            self._remove_weakest_zone()
        
        self.pain_zones.append(PainZone(
            price_center=price,
            price_range=zone_width,
            loss_count=1,
            total_loss_r=loss_r,
            decay_factor=0.5
        ))
    
    def _heal_pain_zone(self, price: float):
        """Reduce pain in zone on win."""
        for zone in self.pain_zones:
            if abs(zone.price_center - price) < zone.price_range:
                zone.decay_factor *= 0.7  # Accelerate decay
                zone.loss_count = max(0, zone.loss_count - 1)
                return
    
    def _update_regime_pain(self, regime: str, side: str, is_loss: bool):
        """Update regime-specific pain score."""
        key = (regime, side)
        
        if key not in self.regime_pain:
            self.regime_pain[key] = RegimePainRecord(regime=regime, side=side)
        
        record = self.regime_pain[key]
        
        if is_loss:
            record.loss_streak += 1
            record.total_losses += 1
            record.pain_score = min(1.0, record.loss_streak * 0.2)
        else:
            record.loss_streak = 0
            record.total_wins += 1
            record.pain_score *= 0.8
        
        record.last_updated = datetime.utcnow()
    
    def _calculate_pain_zone_modifier(self, price: float) -> float:
        """Check if price is in any pain zone."""
        for zone in self.pain_zones:
            if abs(zone.price_center - price) < zone.price_range:
                # In pain zone - reduce by up to 70%
                reduction = zone.decay_factor * 0.7
                return 1.0 - reduction
        return 1.0
    
    def _calculate_regime_pain_modifier(self, regime: str, side: str) -> float:
        """Calculate modifier based on regime pain."""
        key = (regime, side)
        
        if key not in self.regime_pain:
            return 1.0
        
        record = self.regime_pain[key]
        
        if record.pain_score > 0.5:
            # Reduce by up to 60%
            return 1.0 - (record.pain_score * 0.6)
        
        return 1.0
    
    def _calculate_cluster_modifier(self) -> float:
        """Calculate modifier based on loss clustering."""
        losses = self.loss_cluster.count_losses()
        
        if losses >= 4:
            return 0.2  # Severe cluster
        elif losses >= 3:
            return 0.4  # Moderate cluster
        elif losses >= 2:
            return 0.7  # Mild cluster
        
        return 1.0
    
    def _apply_decay(self):
        """Apply time-based decay to all memories."""
        now = datetime.utcnow()
        
        # Decay pain zones
        zones_to_remove = []
        for zone in self.pain_zones:
            hours = (now - zone.last_touched).total_seconds() / 3600
            zone.decay_factor *= (0.5 ** (hours / self.PAIN_ZONE_HALF_LIFE))
            
            if zone.decay_factor < 0.1:
                zones_to_remove.append(zone)
        
        for zone in zones_to_remove:
            self.pain_zones.remove(zone)
        
        # Decay regime pain
        keys_to_remove = []
        for key, record in self.regime_pain.items():
            hours = (now - record.last_updated).total_seconds() / 3600
            record.pain_score *= (0.5 ** (hours / self.REGIME_PAIN_HALF_LIFE))
            
            if record.pain_score < 0.05:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.regime_pain[key]
    
    def _remove_weakest_zone(self):
        """Remove zone with lowest decay factor."""
        if not self.pain_zones:
            return
        
        weakest = min(self.pain_zones, key=lambda z: z.decay_factor)
        self.pain_zones.remove(weakest)
    
    def get_state(self) -> dict:
        """Get current memory state for logging."""
        return {
            'pain_zones': len(self.pain_zones),
            'regime_pain_active': len([r for r in self.regime_pain.values() if r.pain_score > 0.1]),
            'cluster_active': self.loss_cluster.cluster_active,
            'cluster_severity': self.loss_cluster.cluster_severity,
        }
    
    def reset(self):
        """Reset all memories (for testing or new session)."""
        self.pain_zones = []
        self.regime_pain = {}
        self.loss_cluster = LossCluster(buffer_size=5)
        logger.info("MEMORY: Reset complete")
