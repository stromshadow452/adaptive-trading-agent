"""
Portfolio Brain V1

Capital allocation across multiple alpha pods.

Rules-based allocation with:
- Risk-adjusted weighting
- Diversification control
- Exposure balancing
- Pod performance tracking
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd
import numpy as np

from ..interfaces import AlphaSignal, Decision, SignalDirection

logger = logging.getLogger(__name__)


@dataclass
class PodAllocation:
    """Allocation for a single pod."""
    pod_name: str
    weight: float  # 0-1
    target_exposure: float
    current_exposure: float
    risk_contribution: float
    expected_return: float
    volatility: float


@dataclass
class PortfolioState:
    """Portfolio state."""
    timestamp: datetime
    total_value: float = 1.0
    cash: float = 0.0
    
    # Exposures
    gross_exposure: float = 0.0
    net_exposure: float = 0.0
    long_exposure: float = 0.0
    short_exposure: float = 0.0
    
    # By pod
    pod_exposures: Dict[str, float] = field(default_factory=dict)
    
    # Risk
    portfolio_volatility: float = 0.0
    max_drawdown: float = 0.0


class PortfolioBrain:
    """
    Portfolio Brain V1.
    
    Manages capital allocation across alpha pods.
    
    Logic:
    1. Equal risk weighting initially
    2. Adjust based on recent performance
    3. Cap concentration at 50% per pod
    4. Maintain target volatility
    """
    
    VERSION = "1.0.0"
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize portfolio brain.
        
        Args:
            config: Configuration
        """
        self.config = config
        
        # Limits
        self.max_pod_weight = config.get('max_pod_weight', 0.50)
        self.min_pod_weight = config.get('min_pod_weight', 0.10)
        self.target_volatility = config.get('target_volatility', 0.10)
        self.max_gross_exposure = config.get('max_gross_exposure', 2.0)
        self.max_net_exposure = config.get('max_net_exposure', 1.0)
        
        # State
        self.state = PortfolioState(timestamp=datetime.now())
        
        # Pod tracking
        self.pod_allocations: Dict[str, PodAllocation] = {}
        self.pod_performance: Dict[str, Dict] = {}
        
        # History
        self.allocation_history: List[Dict] = []
        
        logger.info(f"PortfolioBrain initialized v{self.VERSION}")
        logger.info(f"  Max pod weight: {self.max_pod_weight:.0%}")
        logger.info(f"  Target vol: {self.target_volatility:.0%}")
    
    def allocate(self,
                signals: List[AlphaSignal],
                current_portfolio: Optional[Dict] = None) -> Dict[str, float]:
        """
        Allocate capital across pods.
        
        Args:
            signals: Signals from all pods
            current_portfolio: Current portfolio state
            
        Returns:
            Dict of pod_name → weight
        """
        if not signals:
            return {}
        
        # Group signals by pod
        pod_signals = {}
        for signal in signals:
            if signal.source not in pod_signals:
                pod_signals[signal.source] = []
            pod_signals[signal.source].append(signal)
        
        # Calculate base weights (equal risk)
        weights = self._calculate_base_weights(pod_signals)
        
        # Adjust for performance
        weights = self._adjust_for_performance(weights)
        
        # Apply diversification constraints
        weights = self._apply_constraints(weights)
        
        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        
        # Update state
        self._update_state(weights, signals)
        
        return weights
    
    def _calculate_base_weights(self, 
                              pod_signals: Dict[str, List[AlphaSignal]]) -> Dict[str, float]:
        """
        Calculate base weights (inverse volatility).
        
        Higher weight to lower volatility pods.
        """
        weights = {}
        
        for pod_name, signals in pod_signals.items():
            if not signals:
                continue
            
            # Calculate pod volatility
            vols = [s.volatility for s in signals if s.volatility > 0]
            if vols:
                avg_vol = np.mean(vols)
            else:
                avg_vol = 0.10
            
            # Weight inversely proportional to volatility
            weight = 1.0 / max(avg_vol, 0.05)
            weights[pod_name] = weight
        
        return weights
    
    def _adjust_for_performance(self, 
                               weights: Dict[str, float]) -> Dict[str, float]:
        """
        Adjust weights based on recent performance.
        
        Increase weight to better performing pods.
        """
        adjusted = {}
        
        for pod_name, base_weight in weights.items():
            # Get recent performance
            perf = self.pod_performance.get(pod_name, {})
            sharpe = perf.get('sharpe', 0)
            
            # Adjust based on Sharpe
            if sharpe > 1.0:
                multiplier = 1.2
            elif sharpe > 0.5:
                multiplier = 1.1
            elif sharpe > 0:
                multiplier = 1.0
            else:
                multiplier = 0.8
            
            adjusted[pod_name] = base_weight * multiplier
        
        return adjusted
    
    def _apply_constraints(self, 
                          weights: Dict[str, float]) -> Dict[str, float]:
        """
        Apply diversification constraints.
        
        - Max weight per pod: 50%
        - Min weight per pod: 10%
        """
        constrained = {}
        
        for pod_name, weight in weights.items():
            # Cap at max
            weight = min(weight, self.max_pod_weight)
            
            # Floor at min (if allocated)
            if weight > 0:
                weight = max(weight, self.min_pod_weight)
            
            constrained[pod_name] = weight
        
        return constrained
    
    def calculate_position_sizes(self,
                                signals: List[AlphaSignal],
                                weights: Dict[str, float],
                                portfolio_value: float) -> Dict[str, float]:
        """
        Calculate position sizes considering portfolio allocation.
        
        Args:
            signals: Signals from pods
            weights: Pod weights
            portfolio_value: Current portfolio value
            
        Returns:
            Dict of symbol → position size
        """
        positions = {}
        
        for signal in signals:
            pod_name = signal.source
            if pod_name not in weights:
                continue
            
            pod_weight = weights[pod_name]
            
            # Scale signal by pod weight
            base_size = signal.recommended_size
            scaled_size = base_size * pod_weight
            
            positions[signal.symbol] = scaled_size
        
        return positions
    
    def check_diversification(self,
                           positions: Dict[str, float]) -> Tuple[bool, str]:
        """
        Check if portfolio is properly diversified.
        
        Args:
            positions: Current positions
            
        Returns:
            (is_diversified, reason)
        """
        if not positions:
            return True, "No positions"
        
        # Check concentration
        total = sum(abs(v) for v in positions.values())
        if total == 0:
            return True, "No positions"
        
        for symbol, size in positions.items():
            concentration = abs(size) / total
            if concentration > 0.50:
                return False, f"High concentration in {symbol}: {concentration:.1%}"
        
        return True, "Diversified"
    
    def calculate_risk_contribution(self,
                                  positions: Dict[str, float],
                                  correlations: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """
        Calculate risk contribution per position.
        
        Args:
            positions: Current positions
            correlations: Correlation matrix
            
        Returns:
            Dict of symbol → risk contribution
        """
        if not positions or correlations is None:
            return {s: 0.0 for s in positions}
        
        contributions = {}
        
        for symbol, size in positions.items():
            # Simplified: assume 10% vol per position
            vol = 0.10
            
            # Risk contribution
            contribution = size * vol
            contributions[symbol] = contribution
        
        return contributions
    
    def rebalance_needed(self,
                        current_positions: Dict[str, float],
                        target_positions: Dict[str, float],
                        threshold: float = 0.05) -> bool:
        """
        Check if rebalancing is needed.
        
        Args:
            current_positions: Current positions
            target_positions: Target positions
            threshold: Rebalance threshold
            
        Returns:
            bool: Rebalance needed
        """
        for symbol, target in target_positions.items():
            current = current_positions.get(symbol, 0)
            diff = abs(target - current)
            
            if diff > threshold:
                return True
        
        return False
    
    def get_rebalance_trades(self,
                           current_positions: Dict[str, float],
                           target_positions: Dict[str, float]) -> List[Dict]:
        """
        Generate rebalance trades.
        
        Args:
            current_positions: Current positions
            target_positions: Target positions
            
        Returns:
            List of trade dicts
        """
        trades = []
        
        all_symbols = set(current_positions.keys()) | set(target_positions.keys())
        
        for symbol in all_symbols:
            current = current_positions.get(symbol, 0)
            target = target_positions.get(symbol, 0)
            delta = target - current
            
            if abs(delta) > 0.01:  # 1% threshold
                trades.append({
                    'symbol': symbol,
                    'current': current,
                    'target': target,
                    'delta': delta,
                    'action': 'BUY' if delta > 0 else 'SELL'
                })
        
        return trades
    
    def update_performance(self,
                          pod_name: str,
                          returns: List[float],
                          period: str = '1M'):
        """
        Update pod performance metrics.
        
        Args:
            pod_name: Pod name
            returns: List of returns
            period: Period label
        """
        if not returns:
            return
        
        returns = np.array(returns)
        
        # Calculate metrics
        sharpe = np.mean(returns) / (np.std(returns) + 1e-9) * np.sqrt(252)
        
        self.pod_performance[pod_name] = {
            'sharpe': sharpe,
            'returns': returns[-1],
            'volatility': np.std(returns) * np.sqrt(252),
            'period': period,
            'updated': datetime.now()
        }
        
        logger.debug(f"Updated {pod_name} performance: Sharpe={sharpe:.2f}")
    
    def _update_state(self,
                    weights: Dict[str, float],
                    signals: List[AlphaSignal]):
        """Update portfolio state."""
        self.state.timestamp = datetime.now()
        
        # Calculate exposures
        gross = sum(abs(s.recommended_size) for s in signals)
        net = sum(s.recommended_size for s in signals if s.direction == SignalDirection.LONG) \
              - sum(s.recommended_size for s in signals if s.direction == SignalDirection.SHORT)
        
        self.state.gross_exposure = gross
        self.state.net_exposure = net
        
        # Pod exposures
        for signal in signals:
            pod = signal.source
            if pod not in self.state.pod_exposures:
                self.state.pod_exposures[pod] = 0
            self.state.pod_exposures[pod] += signal.recommended_size
        
        # Store history
        self.allocation_history.append({
            'timestamp': datetime.now(),
            'weights': weights,
            'exposures': self.state.pod_exposures.copy()
        })
    
    def get_status(self) -> Dict[str, Any]:
        """Get portfolio status."""
        return {
            'version': self.VERSION,
            'timestamp': self.state.timestamp.isoformat(),
            'gross_exposure': self.state.gross_exposure,
            'net_exposure': self.state.net_exposure,
            'pod_weights': {
                name: alloc.weight
                for name, alloc in self.pod_allocations.items()
            },
            'pod_performance': self.pod_performance,
            'num_pods': len(self.pod_allocations)
        }
    
    def get_allocation_report(self) -> str:
        """Get formatted allocation report."""
        report = f"""
Portfolio Allocation Report ({datetime.now()})
{'='*60}
Gross Exposure: {self.state.gross_exposure:.1%}
Net Exposure: {self.state.net_exposure:.1%}

Pod Allocations:
{'-'*60}
"""
        
        for pod_name, alloc in self.pod_allocations.items():
            report += f"  {pod_name:20s}: {alloc.weight:.1%}\n"
        
        report += f"{'='*60}\n"
        return report
