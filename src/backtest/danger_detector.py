"""
Danger Detection Engine

Detects portfolio-level danger conditions that warrant capital protection.
Implements the 5 danger signals from the Portfolio Shadow Engine design.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


class DangerLevel(Enum):
    """Portfolio danger levels."""
    SAFE = 'SAFE'           # Normal operation
    CAUTION = 'CAUTION'     # Reduced size
    DANGER = 'DANGER'       # Halt new trades
    CRITICAL = 'CRITICAL'   # Close all positions


@dataclass
class DangerSignal:
    """Individual danger signal detection."""
    name: str
    detected: bool
    severity: int  # 1 = low, 2 = medium, 3 = high
    details: str


@dataclass
class DangerAssessment:
    """Complete danger assessment result."""
    level: DangerLevel
    score: int
    signals: List[DangerSignal]
    timestamp: datetime
    
    def __str__(self):
        active = [s for s in self.signals if s.detected]
        signal_list = ", ".join(s.name for s in active) if active else "None"
        return f"[{self.level.value}] Score: {self.score} | Signals: {signal_list}"


class DangerDetector:
    """
    Detects portfolio-level danger conditions.
    
    Five Danger Signals:
    1. Volatility Spike: ATR > 2x 20-day average
    2. Abnormal Candle: Range > 3x ATR
    3. Correlated Losses: 2+ assets hit SL same session
    4. Regime Chaos: 3+ regime changes in 24h
    5. DD Threshold: Portfolio DD > 4%
    """
    
    def __init__(
        self,
        vol_spike_mult: float = 2.0,
        abnormal_candle_mult: float = 3.0,
        correlated_loss_threshold: int = 2,
        regime_chaos_threshold: int = 3,
        dd_caution: float = 0.03,
        dd_danger: float = 0.04,
        dd_critical: float = 0.06,
    ):
        self.vol_spike_mult = vol_spike_mult
        self.abnormal_candle_mult = abnormal_candle_mult
        self.correlated_loss_threshold = correlated_loss_threshold
        self.regime_chaos_threshold = regime_chaos_threshold
        self.dd_caution = dd_caution
        self.dd_danger = dd_danger
        self.dd_critical = dd_critical
        
        # State tracking
        self.regime_history: List[Tuple[datetime, str, str]] = []  # (time, symbol, regime)
        self.session_losses: List[Tuple[datetime, str]] = []  # (time, symbol)
    
    def assess(
        self,
        portfolio_dd: float,
        asset_data: Dict[str, pd.DataFrame],
        current_time: datetime,
    ) -> DangerAssessment:
        """
        Assess current danger level.
        
        Args:
            portfolio_dd: Current portfolio drawdown (0-1)
            asset_data: Dict of symbol -> OHLC DataFrame with ATR
            current_time: Current timestamp
            
        Returns:
            DangerAssessment with level, score, and active signals
        """
        signals = []
        
        # 1. Volatility Spike
        vol_signal = self._check_volatility_spike(asset_data)
        signals.append(vol_signal)
        
        # 2. Abnormal Candle
        candle_signal = self._check_abnormal_candle(asset_data)
        signals.append(candle_signal)
        
        # 3. Correlated Losses
        corr_signal = self._check_correlated_losses(current_time)
        signals.append(corr_signal)
        
        # 4. Regime Chaos
        regime_signal = self._check_regime_chaos(current_time)
        signals.append(regime_signal)
        
        # 5. DD Threshold
        dd_signal = self._check_dd_threshold(portfolio_dd)
        signals.append(dd_signal)
        
        # Calculate total danger score
        score = sum(s.severity for s in signals if s.detected)
        
        # Determine danger level
        if score >= 5 or portfolio_dd >= self.dd_critical:
            level = DangerLevel.CRITICAL
        elif score >= 3 or portfolio_dd >= self.dd_danger:
            level = DangerLevel.DANGER
        elif score >= 1 or portfolio_dd >= self.dd_caution:
            level = DangerLevel.CAUTION
        else:
            level = DangerLevel.SAFE
        
        return DangerAssessment(
            level=level,
            score=score,
            signals=signals,
            timestamp=current_time,
        )
    
    def _check_volatility_spike(self, asset_data: Dict[str, pd.DataFrame]) -> DangerSignal:
        """Check if any asset has volatility spike."""
        for symbol, df in asset_data.items():
            if len(df) < 21:
                continue
            
            if 'atr' not in df.columns:
                # Calculate ATR if not present
                high = df['high'].values
                low = df['low'].values
                close = df['close'].values
                tr = np.maximum(
                    high[1:] - low[1:],
                    np.maximum(
                        np.abs(high[1:] - close[:-1]),
                        np.abs(low[1:] - close[:-1])
                    )
                )
                current_atr = np.mean(tr[-14:])
                avg_atr = np.mean(tr[-34:-14]) if len(tr) >= 34 else np.mean(tr)
            else:
                current_atr = df['atr'].iloc[-1]
                avg_atr = df['atr'].iloc[-21:-1].mean()
            
            if avg_atr > 0 and current_atr > avg_atr * self.vol_spike_mult:
                return DangerSignal(
                    name="Volatility Spike",
                    detected=True,
                    severity=1,
                    details=f"{symbol}: ATR {current_atr/avg_atr:.1f}x average"
                )
        
        return DangerSignal(
            name="Volatility Spike",
            detected=False,
            severity=0,
            details=""
        )
    
    def _check_abnormal_candle(self, asset_data: Dict[str, pd.DataFrame]) -> DangerSignal:
        """Check for abnormal candle (news-like spike)."""
        for symbol, df in asset_data.items():
            if len(df) < 15:
                continue
            
            # Current bar range
            current_range = df['high'].iloc[-1] - df['low'].iloc[-1]
            
            # Calculate ATR
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            tr = np.maximum(
                high[1:] - low[1:],
                np.maximum(
                    np.abs(high[1:] - close[:-1]),
                    np.abs(low[1:] - close[:-1])
                )
            )
            atr = np.mean(tr[-14:])
            
            if atr > 0 and current_range > atr * self.abnormal_candle_mult:
                return DangerSignal(
                    name="Abnormal Candle",
                    detected=True,
                    severity=2,
                    details=f"{symbol}: Range {current_range/atr:.1f}x ATR"
                )
        
        return DangerSignal(
            name="Abnormal Candle",
            detected=False,
            severity=0,
            details=""
        )
    
    def _check_correlated_losses(self, current_time: datetime) -> DangerSignal:
        """Check for correlated losses in same session."""
        # Look at losses in last 4 hours
        cutoff = current_time - timedelta(hours=4)
        recent = [s for t, s in self.session_losses if t >= cutoff]
        
        if len(recent) >= self.correlated_loss_threshold:
            return DangerSignal(
                name="Correlated Losses",
                detected=True,
                severity=2,
                details=f"{len(recent)} losses in 4h: {', '.join(recent)}"
            )
        
        return DangerSignal(
            name="Correlated Losses",
            detected=False,
            severity=0,
            details=""
        )
    
    def _check_regime_chaos(self, current_time: datetime) -> DangerSignal:
        """Check for excessive regime changes."""
        cutoff = current_time - timedelta(hours=24)
        recent = [r for t, _, r in self.regime_history if t >= cutoff]
        
        # Count unique regime transitions
        changes = 0
        for i in range(1, len(recent)):
            if recent[i] != recent[i-1]:
                changes += 1
        
        if changes >= self.regime_chaos_threshold:
            return DangerSignal(
                name="Regime Chaos",
                detected=True,
                severity=1,
                details=f"{changes} regime changes in 24h"
            )
        
        return DangerSignal(
            name="Regime Chaos",
            detected=False,
            severity=0,
            details=""
        )
    
    def _check_dd_threshold(self, portfolio_dd: float) -> DangerSignal:
        """Check drawdown against thresholds."""
        if portfolio_dd >= self.dd_critical:
            return DangerSignal(
                name="DD Critical",
                detected=True,
                severity=3,
                details=f"DD {portfolio_dd*100:.1f}% >= {self.dd_critical*100:.0f}%"
            )
        elif portfolio_dd >= self.dd_danger:
            return DangerSignal(
                name="DD Threshold",
                detected=True,
                severity=2,
                details=f"DD {portfolio_dd*100:.1f}% >= {self.dd_danger*100:.0f}%"
            )
        elif portfolio_dd >= self.dd_caution:
            return DangerSignal(
                name="DD Warning",
                detected=True,
                severity=1,
                details=f"DD {portfolio_dd*100:.1f}% >= {self.dd_caution*100:.0f}%"
            )
        
        return DangerSignal(
            name="DD Threshold",
            detected=False,
            severity=0,
            details=""
        )
    
    # =========================================================================
    # STATE UPDATES
    # =========================================================================
    
    def record_loss(self, symbol: str, timestamp: datetime):
        """Record a losing trade for correlation tracking."""
        self.session_losses.append((timestamp, symbol))
        
        # Keep only last 24 hours
        cutoff = timestamp - timedelta(hours=24)
        self.session_losses = [(t, s) for t, s in self.session_losses if t >= cutoff]
    
    def record_regime(self, symbol: str, regime: str, timestamp: datetime):
        """Record regime detection for chaos tracking."""
        self.regime_history.append((timestamp, symbol, regime))
        
        # Keep only last 48 hours
        cutoff = timestamp - timedelta(hours=48)
        self.regime_history = [(t, s, r) for t, s, r in self.regime_history if t >= cutoff]
    
    def reset(self):
        """Reset state for new simulation."""
        self.regime_history.clear()
        self.session_losses.clear()
