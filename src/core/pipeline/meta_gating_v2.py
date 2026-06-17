"""
Meta-Gating V2

Upgraded Stage 8 pipeline component.

Responsibilities:
- Dynamic pod enable/disable
- Confidence weighting
- Performance-based throttling
- Regime-aware routing
- Drawdown protection
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from ..interfaces import (
    AlphaSignal, SignalDirection, DecisionAction
)

logger = logging.getLogger(__name__)


class MetaGatingV2:
    """
    Meta-Gating V2.
    
    Advanced signal filtering with:
    - Pod health scoring
    - Rolling performance tracking
    - Confidence aggregation
    - Adaptive capital routing
    - Regime-aware filtering
    """
    
    VERSION = "2.0.0"
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Meta-Gating V2.
        
        Args:
            config: Configuration
        """
        self.config = config
        
        # Thresholds
        self.min_confidence = config.get('min_confidence', 0.30)
        self.min_sharpe = config.get('min_sharpe', 0.5)
        self.max_drawdown = config.get('max_drawdown', 0.15)
        
        # Rolling windows
        self.performance_window = config.get('performance_window', 20)
        self.sharpe_window = config.get('sharpe_window', 20)
        
        # Pod tracking
        self.pod_health: Dict[str, Dict] = {}
        self.pod_disabled: Dict[str, bool] = {}
        self.pod_decay: Dict[str, float] = {}
        
        # History
        self.decision_history: List[Dict] = []
        
        logger.info(f"MetaGatingV2 initialized v{self.VERSION}")
    
    def gate_signals(self,
                    signals: List[AlphaSignal],
                    regime: str,
                    current_drawdown: float,
                    pod_performance: Dict[str, Dict]) -> Tuple[List[AlphaSignal], List[str]]:
        """
        Gate signals based on health, performance, and regime.
        
        Args:
            signals: Raw signals from pods
            regime: Current market regime
            current_drawdown: Current drawdown
            pod_performance: Performance metrics per pod
            
        Returns:
            (approved_signals, block_reasons)
        """
        approved = []
        block_reasons = []
        
        # Update pod health
        self._update_pod_health(pod_performance, current_drawdown)
        
        for signal in signals:
            pod_name = signal.source
            
            # Check if pod disabled
            if self.pod_disabled.get(pod_name, False):
                block_reasons.append(f"{pod_name}: POD_DISABLED")
                continue
            
            # Check confidence
            if signal.confidence < self.min_confidence:
                block_reasons.append(f"{signal.symbol}: LOW_CONFIDENCE")
                continue
            
            # Check pod health
            health = self.pod_health.get(pod_name, {})
            if health.get('score', 1.0) < 0.5:
                block_reasons.append(f"{pod_name}: POOR_HEALTH")
                continue
            
            # Check drawdown
            if current_drawdown > self.max_drawdown:
                block_reasons.append(f"{signal.symbol}: HIGH_DRAWDOWN")
                continue
            
            # Check regime compatibility
            if not self._check_regime_compatibility(signal, regime):
                block_reasons.append(f"{signal.symbol}: REGIME_MISMATCH")
                continue
            
            # Apply confidence weighting
            signal = self._weight_by_confidence(signal, health)
            
            approved.append(signal)
        
        return approved, block_reasons
    
    def _update_pod_health(self,
                          pod_performance: Dict[str, Dict],
                          current_drawdown: float):
        """Update pod health scores."""
        for pod_name, perf in pod_performance.items():
            sharpe = perf.get('sharpe', 0)
            returns = perf.get('returns', [])
            
            # Calculate health score (0-1)
            health_score = 1.0
            
            # Penalize low Sharpe
            if sharpe < self.min_sharpe:
                health_score *= 0.5
            elif sharpe < 1.0:
                health_score *= 0.8
            
            # Penalize recent losses
            if len(returns) >= 5:
                recent = returns[-5:]
                if np.mean(recent) < 0:
                    health_score *= 0.7
            
            # Penalize high drawdown contribution
            if current_drawdown > 0.10:
                health_score *= 0.6
            
            self.pod_health[pod_name] = {
                'score': health_score,
                'sharpe': sharpe,
                'last_update': datetime.now(),
                'disabled': health_score < 0.3
            }
            
            # Auto-disable if health too low
            if health_score < 0.2:
                self.pod_disabled[pod_name] = True
                logger.warning(f"Auto-disabled {pod_name} due to poor health")
    
    def _check_regime_compatibility(self,
                                   signal: AlphaSignal,
                                   regime: str) -> bool:
        """Check if signal compatible with regime."""
        if regime == "danger":
            # Block all signals in danger regime
            return False
        
        if regime == "volatile":
            # Only high confidence signals
            return signal.confidence > 0.7
        
        return True
    
    def _weight_by_confidence(self,
                             signal: AlphaSignal,
                             health: Dict) -> AlphaSignal:
        """Adjust signal weight by confidence and health."""
        health_score = health.get('score', 1.0)
        
        # Reduce size if health poor
        signal.recommended_size *= health_score
        
        return signal
    
    def calculate_pod_weights(self,
                            pod_signals: Dict[str, List[AlphaSignal]],
                            pod_performance: Dict[str, Dict]) -> Dict[str, float]:
        """
        Calculate adaptive weights for pods.
        
        Based on:
        - Recent Sharpe
        - Confidence
        - Drawdown
        - Regime fit
        """
        weights = {}
        
        for pod_name, signals in pod_signals.items():
            if not signals:
                weights[pod_name] = 0
                continue
            
            # Base weight from confidence
            avg_conf = np.mean([s.confidence for s in signals])
            weight = avg_conf
            
            # Adjust by performance
            perf = pod_performance.get(pod_name, {})
            sharpe = perf.get('sharpe', 0)
            
            if sharpe > 1.5:
                weight *= 1.2
            elif sharpe < 0:
                weight *= 0.5
            
            # Adjust by health
            health = self.pod_health.get(pod_name, {}).get('score', 1.0)
            weight *= health
            
            weights[pod_name] = weight
        
        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        
        return weights
    
    def get_confidence_aggregation(self,
                                 signals: List[AlphaSignal],
                                 method: str = "weighted") -> float:
        """
        Aggregate confidence across signals.
        
        Args:
            signals: Signals to aggregate
            method: "weighted", "majority", "best"
            
        Returns:
            Aggregated confidence
        """
        if not signals:
            return 0.0
        
        if method == "weighted":
            weights = [s.confidence for s in signals]
            confidences = [s.confidence for s in signals]
            return np.average(confidences, weights=weights)
        
        elif method == "majority":
            longs = sum(1 for s in signals if s.direction == SignalDirection.LONG)
            shorts = sum(1 for s in signals if s.direction == SignalDirection.SHORT)
            
            majority = max(longs, shorts) / len(signals)
            return majority
        
        elif method == "best":
            return max(s.confidence for s in signals)
        
        else:
            return np.mean([s.confidence for s in signals])
    
    def detect_pod_decay(self,
                        pod_name: str,
                        recent_returns: List[float]) -> bool:
        """
        Detect if pod performance is decaying.
        
        Args:
            pod_name: Pod name
            recent_returns: Recent returns
            
        Returns:
            bool: Decay detected
        """
        if len(recent_returns) < 20:
            return False
        
        # Calculate rolling Sharpe
        returns = np.array(recent_returns)
        
        # Recent vs historical
        recent_sharpe = np.mean(returns[-10:]) / (np.std(returns[-10:]) + 1e-9) * np.sqrt(252)
        historical_sharpe = np.mean(returns[:-10]) / (np.std(returns[:-10]) + 1e-9) * np.sqrt(252)
        
        # Decay if recent < 50% of historical
        if recent_sharpe < historical_sharpe * 0.5:
            self.pod_decay[pod_name] = recent_sharpe / max(historical_sharpe, 0.01)
            return True
        
        return False
    
    def get_throttle_level(self,
                         current_drawdown: float,
                         pod_health: Dict[str, float]) -> float:
        """
        Calculate global throttle level.
        
        Based on:
        - Drawdown
        - Average pod health
        """
        # Drawdown component
        if current_drawdown < 0.05:
            dd_throttle = 1.0
        elif current_drawdown < 0.10:
            dd_throttle = 0.8
        elif current_drawdown < 0.15:
            dd_throttle = 0.5
        else:
            dd_throttle = 0.25
        
        # Health component
        if pod_health:
            avg_health = np.mean(list(pod_health.values()))
            health_throttle = 0.5 + 0.5 * avg_health
        else:
            health_throttle = 1.0
        
        # Combined
        throttle = min(dd_throttle, health_throttle)
        
        return throttle
    
    def get_status(self) -> Dict[str, Any]:
        """Get meta-gating status."""
        return {
            'version': self.VERSION,
            'min_confidence': self.min_confidence,
            'min_sharpe': self.min_sharpe,
            'max_drawdown': self.max_drawdown,
            'pod_health': self.pod_health,
            'pod_disabled': self.pod_disabled,
            'pod_decay': self.pod_decay,
            'num_decisions': len(self.decision_history)
        }
