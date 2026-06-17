"""
Pod Performance Intelligence

Track and monitor alpha pod performance.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PodMetrics:
    """Pod performance metrics."""
    pod_name: str
    
    # Returns
    returns: List[float] = field(default_factory=list)
    cumulative_return: float = 0.0
    
    # Risk
    volatility: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    
    # Performance
    sharpe: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0
    
    # Hit rate
    wins: int = 0
    losses: int = 0
    hit_rate: float = 0.0
    
    # Recent
    recent_returns: List[float] = field(default_factory=list)
    rolling_sharpe: float = 0.0
    rolling_hit_rate: float = 0.0
    
    # Regime
    regime_performance: Dict[str, List[float]] = field(default_factory=dict)
    
    # Health
    health_score: float = 1.0
    last_update: Optional[datetime] = None


class PodPerformanceTracker:
    """
    Track pod performance metrics.
    
    Monitors:
    - Rolling Sharpe
    - Rolling hit rate
    - Drawdown
    - Volatility
    - Turnover
    - Regime dependence
    - Live vs expected drift
    """
    
    VERSION = "1.0.0"
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize tracker.
        
        Args:
            config: Configuration
        """
        self.config = config
        
        # Windows
        self.sharpe_window = config.get('sharpe_window', 20)
        self.hit_rate_window = config.get('hit_rate_window', 20)
        
        # Thresholds
        self.sharpe_alert_threshold = config.get('sharpe_alert', 0.5)
        self.drawdown_alert_threshold = config.get('drawdown_alert', 0.15)
        
        # Storage
        self.pods: Dict[str, PodMetrics] = {}
        self.trade_history: List[Dict] = []
        
        # Alert state
        self.alerts: List[str] = []
        
        logger.info(f"PodPerformanceTracker initialized v{self.VERSION}")
    
    def register_pod(self, pod_name: str):
        """Register a pod for tracking."""
        if pod_name not in self.pods:
            self.pods[pod_name] = PodMetrics(pod_name=pod_name)
            logger.info(f"Registered pod: {pod_name}")
    
    def record_trade(self,
                    pod_name: str,
                    pnl: float,
                    regime: str = "unknown",
                    metadata: Dict = None):
        """
        Record trade P&L.
        
        Args:
            pod_name: Pod name
            pnl: Trade P&L
            regime: Market regime
            metadata: Additional data
        """
        if pod_name not in self.pods:
            self.register_pod(pod_name)
        
        metrics = self.pods[pod_name]
        
        # Update returns
        metrics.returns.append(pnl)
        metrics.recent_returns.append(pnl)
        
        # Keep recent window
        if len(metrics.recent_returns) > self.sharpe_window * 2:
            metrics.recent_returns = metrics.recent_returns[-self.sharpe_window * 2:]
        
        # Update cumulative
        metrics.cumulative_return = sum(metrics.returns)
        
        # Update win/loss
        if pnl > 0:
            metrics.wins += 1
        else:
            metrics.losses += 1
        
        # Update hit rate
        total = metrics.wins + metrics.losses
        metrics.hit_rate = metrics.wins / total if total > 0 else 0
        
        # Regime tracking
        if regime not in metrics.regime_performance:
            metrics.regime_performance[regime] = []
        metrics.regime_performance[regime].append(pnl)
        
        # Update timestamp
        metrics.last_update = datetime.now()
        
        # Calculate metrics
        self._calculate_metrics(pod_name)
        
        # Check alerts
        self._check_alerts(pod_name)
        
        # Log
        self.trade_history.append({
            'pod': pod_name,
            'pnl': pnl,
            'regime': regime,
            'timestamp': datetime.now(),
            'metadata': metadata
        })
    
    def _calculate_metrics(self, pod_name: str):
        """Calculate all metrics for a pod."""
        metrics = self.pods[pod_name]
        returns = np.array(metrics.returns)
        
        if len(returns) < 2:
            return
        
        # Volatility
        metrics.volatility = np.std(returns) * np.sqrt(252)
        
        # Drawdown
        cumulative = np.cumsum(returns)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - peak) / (peak + 1e-9)
        metrics.max_drawdown = np.min(drawdown)
        metrics.current_drawdown = drawdown[-1]
        
        # Sharpe
        if metrics.volatility > 0:
            metrics.sharpe = np.mean(returns) / (np.std(returns) + 1e-9) * np.sqrt(252)
        
        # Sortino
        downside = returns[returns < 0]
        if len(downside) > 0:
            downside_std = np.std(downside) * np.sqrt(252)
            metrics.sortino = np.mean(returns) / (downside_std + 1e-9) * np.sqrt(252)
        
        # Calmar
        if abs(metrics.max_drawdown) > 0:
            metrics.calmar = (np.mean(returns) * 252) / abs(metrics.max_drawdown)
        
        # Rolling Sharpe
        recent = np.array(metrics.recent_returns[-self.sharpe_window:])
        if len(recent) >= self.sharpe_window:
            metrics.rolling_sharpe = np.mean(recent) / (np.std(recent) + 1e-9) * np.sqrt(252)
        
        # Rolling hit rate
        recent_trades = len(metrics.returns[-self.hit_rate_window:])
        if recent_trades >= self.hit_rate_window:
            recent_wins = sum(1 for r in metrics.returns[-self.hit_rate_window:] if r > 0)
            metrics.rolling_hit_rate = recent_wins / recent_trades
        
        # Health score
        self._calculate_health_score(pod_name)
    
    def _calculate_health_score(self, pod_name: str):
        """Calculate health score for pod."""
        metrics = self.pods[pod_name]
        
        score = 1.0
        
        # Penalize low Sharpe
        if metrics.sharpe < 0:
            score *= 0.3
        elif metrics.sharpe < 0.5:
            score *= 0.6
        elif metrics.sharpe < 1.0:
            score *= 0.8
        
        # Penalize drawdown
        if metrics.current_drawdown < -0.20:
            score *= 0.1
        elif metrics.current_drawdown < -0.15:
            score *= 0.3
        elif metrics.current_drawdown < -0.10:
            score *= 0.6
        
        # Penalize low hit rate
        if metrics.hit_rate < 0.50:
            score *= 0.7
        
        metrics.health_score = max(score, 0.0)
    
    def _check_alerts(self, pod_name: str):
        """Check for alerts."""
        metrics = self.pods[pod_name]
        
        # Sharpe alert
        if metrics.sharpe < self.sharpe_alert_threshold:
            alert = f"{pod_name}: Low Sharpe {metrics.sharpe:.2f}"
            if alert not in self.alerts:
                self.alerts.append(alert)
                logger.warning(alert)
        
        # Drawdown alert
        if metrics.current_drawdown < -self.drawdown_alert_threshold:
            alert = f"{pod_name}: High Drawdown {metrics.current_drawdown:.1%}"
            if alert not in self.alerts:
                self.alerts.append(alert)
                logger.warning(alert)
    
    def detect_degradation(self, pod_name: str) -> Tuple[bool, str]:
        """
        Detect if pod is degrading.
        
        Args:
            pod_name: Pod name
            
        Returns:
            (is_degrading, reason)
        """
        if pod_name not in self.pods:
            return False, "Not tracked"
        
        metrics = self.pods[pod_name]
        
        # Check rolling vs historical Sharpe
        if len(metrics.returns) < 40:
            return False, "Insufficient data"
        
        recent_sharpe = metrics.rolling_sharpe
        
        # Compare to overall Sharpe
        if recent_sharpe < metrics.sharpe * 0.5:
            return True, f"Rolling Sharpe ({recent_sharpe:.2f}) degraded from historical ({metrics.sharpe:.2f})"
        
        # Check health score
        if metrics.health_score < 0.3:
            return True, f"Health score {metrics.health_score:.1%} below threshold"
        
        return False, "Performance stable"
    
    def get_regime_analysis(self, pod_name: str) -> Dict[str, Any]:
        """
        Analyze pod performance by regime.
        
        Args:
            pod_name: Pod name
            
        Returns:
            Dict of regime stats
        """
        if pod_name not in self.pods:
            return {}
        
        metrics = self.pods[pod_name]
        
        analysis = {}
        for regime, returns in metrics.regime_performance.items():
            if len(returns) < 5:
                continue
            
            r = np.array(returns)
            analysis[regime] = {
                'trades': len(returns),
                'win_rate': sum(1 for x in r if x > 0) / len(returns),
                'avg_return': np.mean(r),
                'sharpe': np.mean(r) / (np.std(r) + 1e-9) * np.sqrt(252),
                'total_pnl': sum(r)
            }
        
        return analysis
    
    def get_pod_status(self, pod_name: str) -> Dict[str, Any]:
        """Get pod status."""
        if pod_name not in self.pods:
            return {}
        
        metrics = self.pods[pod_name]
        
        return {
            'pod_name': metrics.pod_name,
            'sharpe': metrics.sharpe,
            'rolling_sharpe': metrics.rolling_sharpe,
            'hit_rate': metrics.hit_rate,
            'rolling_hit_rate': metrics.rolling_hit_rate,
            'max_drawdown': metrics.max_drawdown,
            'current_drawdown': metrics.current_drawdown,
            'volatility': metrics.volatility,
            'calmar': metrics.calmar,
            'health_score': metrics.health_score,
            'total_trades': metrics.wins + metrics.losses,
            'wins': metrics.wins,
            'losses': metrics.losses,
            'last_update': metrics.last_update.isoformat() if metrics.last_update else None
        }
    
    def get_all_status(self) -> Dict[str, Dict]:
        """Get status for all pods."""
        return {name: self.get_pod_status(name) for name in self.pods}
    
    def get_degradation_report(self) -> str:
        """Get degradation report."""
        report = f"Pod Degradation Report ({datetime.now()})\n{'='*60}\n"
        
        for pod_name in self.pods:
            degrading, reason = self.detect_degradation(pod_name)
            status = "⚠️  DEGRADING" if degrading else "✅ Stable"
            report += f"{pod_name:20s}: {status} - {reason}\n"
        
        return report
    
    def get_status(self) -> Dict[str, Any]:
        """Get tracker status."""
        return {
            'version': self.VERSION,
            'tracked_pods': len(self.pods),
            'total_trades': len(self.trade_history),
            'alerts': self.alerts[-10:],  # Last 10
            'pod_status': self.get_all_status()
        }
