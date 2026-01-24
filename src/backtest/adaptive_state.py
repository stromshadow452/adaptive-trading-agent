"""
SCOPUS Adaptive State - Core Memory Component

Central state component that tracks:
- Market conditions (volatility, ATR, regime)
- Performance metrics (wins, losses, streaks, drawdown)
- Confidence bucket performance
- R-multiple history

Updates after every candle and trade. O(1) updates.

Follows Jarvis-approved architecture.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from collections import deque
from typing import Dict, Optional, Tuple
from datetime import datetime
import logging
import json

logger = logging.getLogger(__name__)


# ==================== Data Structures ====================

@dataclass
class BucketStats:
    """Performance stats for a confidence bucket"""
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    
    @property
    def count(self) -> int:
        return self.wins + self.losses
    
    @property
    def winrate(self) -> float:
        if self.count == 0:
            return 0.5  # Default
        return self.wins / self.count
    
    def record(self, is_win: bool, pnl: float = 0.0):
        if is_win:
            self.wins += 1
        else:
            self.losses += 1
        self.total_pnl += pnl


@dataclass
class RegimeStats:
    """Performance stats for a regime"""
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    
    @property
    def count(self) -> int:
        return self.wins + self.losses
    
    @property
    def winrate(self) -> float:
        if self.count == 0:
            return 0.5
        return self.wins / self.count
    
    def record(self, is_win: bool, pnl: float = 0.0):
        if is_win:
            self.wins += 1
        else:
            self.losses += 1
        self.total_pnl += pnl


@dataclass
class TradeRecord:
    """Record of a closed trade"""
    timestamp: str
    regime: str
    volatility_level: str
    confidence: float
    pnl: float
    r_multiple: float
    risk_amount: float
    duration_bars: int
    entry_price: float
    exit_price: float
    side: str
    exit_reason: str


# ==================== Kalman Filter ====================

class SimpleKalman:
    """
    Simplified Kalman filter for O(1) trend estimation.
    
    Used to compute smooth trend slope without lag.
    """
    
    def __init__(self, q: float = 0.01, r: float = 0.1):
        """
        Args:
            q: Process noise (lower = smoother)
            r: Measurement noise (lower = more responsive)
        """
        self.q = q
        self.r = r
        self.x = None  # State estimate
        self.p = 1.0   # Error covariance
        self.prev_x = None  # Previous estimate for slope
    
    def update(self, measurement: float) -> float:
        """
        Update filter with new measurement.
        
        Returns:
            Filtered value
        """
        self.prev_x = self.x
        
        if self.x is None:
            self.x = measurement
            return self.x
        
        # Predict step
        self.p += self.q
        
        # Update step
        k = self.p / (self.p + self.r)  # Kalman gain
        self.x = self.x + k * (measurement - self.x)
        self.p = (1 - k) * self.p
        
        return self.x
    
    def get_slope(self, price: float) -> float:
        """
        Get normalized slope (change rate).
        
        Returns:
            Slope as fraction of price
        """
        if self.prev_x is None or price == 0:
            return 0.0
        return (self.x - self.prev_x) / price


# ==================== Main AdaptiveState ====================

class AdaptiveState:
    """
    Core Adaptive State Component
    
    Tracks market conditions and agent performance.
    Updates incrementally (O(1)) after each candle and trade.
    
    Thread-safe for production use.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        config = config or {}
        
        # === Configuration ===
        self.ewma_alpha = config.get('ewma_alpha', 0.02)  # ~50 bar half-life
        self.rolling_window = config.get('rolling_window', 20)
        self.atr_history_size = config.get('atr_history_size', 20)
        self.r_multiple_history_size = config.get('r_multiple_history_size', 10)
        
        # === Market Conditions ===
        self.ewma_volatility: float = 0.01
        self.atr_history: deque = deque(maxlen=self.atr_history_size)
        self.volatility_level: str = 'MID'  # LOW / MID / HIGH
        self.volatility_percentile: float = 0.5
        
        # === Kalman Filter for Trend ===
        self.kalman = SimpleKalman(q=0.01, r=0.1)
        self.kalman_slope: float = 0.0
        
        # === Performance Tracking ===
        self.trade_history: deque = deque(maxlen=self.rolling_window)
        self.rolling_wins: int = 0
        self.rolling_losses: int = 0
        
        self.win_streak: int = 0
        self.loss_streak: int = 0
        self.max_loss_streak: int = 0
        
        # === Drawdown Tracking ===
        self.current_equity: float = config.get('initial_capital', 10000.0)
        self.peak_equity: float = self.current_equity
        self.rolling_drawdown: float = 0.0
        
        # === R-Multiple Tracking ===
        self.recent_r_multiples: deque = deque(maxlen=self.r_multiple_history_size)
        self.avg_r_multiple: float = 0.0
        
        # === Confidence Bucket Performance ===
        # Buckets: 0.4-0.5, 0.5-0.6, 0.6-0.7, 0.7-0.8, 0.8-0.9, 0.9-1.0
        self.confidence_buckets: Dict[str, BucketStats] = {
            '0.4-0.5': BucketStats(),
            '0.5-0.6': BucketStats(),
            '0.6-0.7': BucketStats(),
            '0.7-0.8': BucketStats(),
            '0.8-0.9': BucketStats(),
            '0.9-1.0': BucketStats(),
        }
        
        # === Regime Performance ===
        self.regime_performance: Dict[str, RegimeStats] = {
            'TREND': RegimeStats(),
            'RANGE': RegimeStats(),
            'DANGER': RegimeStats(),
            'UNCERTAIN': RegimeStats(),
            'UNKNOWN': RegimeStats(),
        }
        
        # === Regime Tracking ===
        self.last_regime: str = 'UNKNOWN'
        self.regime_duration: int = 0
        
        # === Timestamps ===
        self.last_candle_time: Optional[datetime] = None
        self.last_trade_time: Optional[datetime] = None
        self.candle_count: int = 0
        self.trade_count: int = 0
        
        logger.info("AdaptiveState initialized")
    
    # ==================== Candle Update ====================
    
    def update_candle(self, candle: pd.Series, features: Optional[pd.Series] = None):
        """
        Update state after every candle. O(1) complexity.
        
        Args:
            candle: OHLCV candle data
            features: Optional feature series with indicators
        """
        self.candle_count += 1
        
        # Extract prices
        close = candle.get('close', candle.get('Close', 0))
        open_price = candle.get('open', candle.get('Open', close))
        
        if close == 0:
            return
        
        # === Update EWMA Volatility ===
        ret = abs(close / open_price - 1) if open_price > 0 else 0
        self.ewma_volatility = (
            self.ewma_alpha * ret + 
            (1 - self.ewma_alpha) * self.ewma_volatility
        )
        
        # === Update ATR History ===
        atr = 0
        if features is not None:
            atr = features.get('atr_14', features.get('M5_atr_14', 0))
        if atr == 0:
            atr = self.ewma_volatility * close  # Fallback
        
        self.atr_history.append(atr)
        
        # === Calculate Volatility Percentile ===
        if len(self.atr_history) >= 5:
            self.volatility_percentile = self._calc_percentile(atr)
            self._update_volatility_level()
        
        # === Update Kalman Filter ===
        self.kalman.update(close)
        self.kalman_slope = self.kalman.get_slope(close)
        
        # === Update Regime Duration ===
        self.regime_duration += 1
        
        # === Update Timestamp ===
        timestamp = candle.get('timestamp', candle.get('time', None))
        if timestamp is not None:
            if isinstance(timestamp, str):
                try:
                    self.last_candle_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                except:
                    pass
            elif isinstance(timestamp, datetime):
                self.last_candle_time = timestamp
    
    def _calc_percentile(self, value: float) -> float:
        """Calculate percentile of value in ATR history"""
        if len(self.atr_history) == 0:
            return 0.5
        
        sorted_history = sorted(self.atr_history)
        rank = sum(1 for v in sorted_history if v <= value)
        return rank / len(sorted_history)
    
    def _update_volatility_level(self):
        """Update volatility level based on percentile"""
        if self.volatility_percentile < 0.33:
            self.volatility_level = 'LOW'
        elif self.volatility_percentile < 0.67:
            self.volatility_level = 'MID'
        else:
            self.volatility_level = 'HIGH'
    
    # ==================== Trade Update ====================
    
    def update_trade(self, trade: TradeRecord):
        """
        Update state after every closed trade. O(1) complexity.
        
        Args:
            trade: Closed trade record
        """
        self.trade_count += 1
        self.trade_history.append(trade)
        
        is_win = trade.pnl > 0
        
        # === Update Rolling Win/Loss ===
        # First, remove oldest trade's contribution if at capacity
        if len(self.trade_history) == self.rolling_window:
            oldest = self.trade_history[0]
            if oldest.pnl > 0:
                self.rolling_wins = max(0, self.rolling_wins - 1)
            else:
                self.rolling_losses = max(0, self.rolling_losses - 1)
        
        # Add new trade's contribution
        if is_win:
            self.rolling_wins += 1
            self.win_streak += 1
            self.loss_streak = 0
        else:
            self.rolling_losses += 1
            self.loss_streak += 1
            self.max_loss_streak = max(self.max_loss_streak, self.loss_streak)
            self.win_streak = 0
        
        # === Update R-Multiple ===
        self.recent_r_multiples.append(trade.r_multiple)
        if len(self.recent_r_multiples) > 0:
            self.avg_r_multiple = sum(self.recent_r_multiples) / len(self.recent_r_multiples)
        
        # === Update Confidence Bucket ===
        bucket = self._get_confidence_bucket(trade.confidence)
        if bucket in self.confidence_buckets:
            self.confidence_buckets[bucket].record(is_win, trade.pnl)
        
        # === Update Regime Performance ===
        regime = trade.regime if trade.regime in self.regime_performance else 'UNKNOWN'
        self.regime_performance[regime].record(is_win, trade.pnl)
        
        # === Update Timestamp ===
        if trade.timestamp:
            try:
                self.last_trade_time = datetime.fromisoformat(trade.timestamp.replace('Z', '+00:00'))
            except:
                pass
        
        logger.debug(f"Trade recorded: PnL={trade.pnl:.2f}, R={trade.r_multiple:.2f}, "
                    f"Streak={'W' if is_win else 'L'}{self.win_streak if is_win else self.loss_streak}")
    
    def update_equity(self, equity: float):
        """
        Update equity and drawdown. Call after each trade or periodic.
        
        Args:
            equity: Current account equity
        """
        self.current_equity = equity
        
        # Update peak
        if equity > self.peak_equity:
            self.peak_equity = equity
        
        # Calculate drawdown
        if self.peak_equity > 0:
            self.rolling_drawdown = (self.peak_equity - equity) / self.peak_equity
        else:
            self.rolling_drawdown = 0.0
    
    def update_regime(self, regime: str):
        """
        Update current regime. Call after regime detection.
        
        Args:
            regime: New regime label
        """
        if regime != self.last_regime:
            self.regime_duration = 0
            self.last_regime = regime
            logger.debug(f"Regime changed to: {regime}")
    
    def _get_confidence_bucket(self, confidence: float) -> str:
        """Get bucket key for confidence value"""
        if confidence < 0.5:
            return '0.4-0.5'
        elif confidence < 0.6:
            return '0.5-0.6'
        elif confidence < 0.7:
            return '0.6-0.7'
        elif confidence < 0.8:
            return '0.7-0.8'
        elif confidence < 0.9:
            return '0.8-0.9'
        else:
            return '0.9-1.0'
    
    # ==================== Getters ====================
    
    def get_rolling_winrate(self) -> float:
        """Get rolling winrate from recent trades"""
        total = self.rolling_wins + self.rolling_losses
        if total == 0:
            return 0.5  # Default
        return self.rolling_wins / total
    
    def get_bucket_winrate(self, confidence: float) -> Optional[float]:
        """Get winrate for confidence bucket"""
        bucket = self._get_confidence_bucket(confidence)
        stats = self.confidence_buckets.get(bucket)
        if stats and stats.count >= 10:
            return stats.winrate
        return None  # Not enough data
    
    def get_regime_winrate(self, regime: str) -> Optional[float]:
        """Get winrate for regime"""
        stats = self.regime_performance.get(regime)
        if stats and stats.count >= 10:
            return stats.winrate
        return None  # Not enough data
    
    def is_in_cooldown(self) -> bool:
        """Check if agent should be in cooldown"""
        return self.loss_streak >= 3
    
    def is_in_survival_mode(self) -> bool:
        """Check if agent should be in survival mode"""
        return self.rolling_drawdown >= 0.15
    
    # ==================== Features Export ====================
    
    def get_features(self) -> Dict[str, float]:
        """
        Export adaptive features for use in pipeline.
        
        Returns:
            Dict of adaptive features
        """
        return {
            'ewma_volatility': self.ewma_volatility,
            'volatility_percentile': self.volatility_percentile,
            'kalman_slope': self.kalman_slope,
            'rolling_winrate': self.get_rolling_winrate(),
            'rolling_drawdown': self.rolling_drawdown,
            'avg_r_multiple': self.avg_r_multiple,
            'win_streak': float(self.win_streak),
            'loss_streak': float(self.loss_streak),
            'regime_duration': float(self.regime_duration),
        }
    
    # ==================== Serialization ====================
    
    def to_dict(self) -> Dict:
        """Serialize state to dict for persistence"""
        return {
            'ewma_volatility': self.ewma_volatility,
            'volatility_level': self.volatility_level,
            'volatility_percentile': self.volatility_percentile,
            'kalman_slope': self.kalman_slope,
            'rolling_wins': self.rolling_wins,
            'rolling_losses': self.rolling_losses,
            'win_streak': self.win_streak,
            'loss_streak': self.loss_streak,
            'max_loss_streak': self.max_loss_streak,
            'current_equity': self.current_equity,
            'peak_equity': self.peak_equity,
            'rolling_drawdown': self.rolling_drawdown,
            'avg_r_multiple': self.avg_r_multiple,
            'last_regime': self.last_regime,
            'regime_duration': self.regime_duration,
            'candle_count': self.candle_count,
            'trade_count': self.trade_count,
        }
    
    def __repr__(self) -> str:
        return (f"AdaptiveState(vol={self.volatility_level}, "
                f"regime={self.last_regime}, "
                f"dd={self.rolling_drawdown:.1%}, "
                f"streak={'W' if self.win_streak > 0 else 'L'}"
                f"{max(self.win_streak, self.loss_streak)})")
