"""
Alpha Pod Registry

Manages registration, lifecycle, and access to alpha pods.
"""

import logging
from typing import Dict, List, Optional, Type, Any
from datetime import datetime
from dataclasses import dataclass, field
import json

from ..interfaces import AlphaPod, AlphaSignal, MarketData

logger = logging.getLogger(__name__)


@dataclass
class PodRegistration:
    """Pod registration record."""
    pod: AlphaPod
    registered_at: datetime
    is_active: bool = True
    last_signal: Optional[datetime] = None
    signal_count: int = 0
    error_count: int = 0
    performance: Dict[str, float] = field(default_factory=dict)


class AlphaPodRegistry:
    """
    Registry for alpha pods.
    
    Manages:
    - Pod registration
    - Lifecycle
    - Health monitoring
    - Performance tracking
    """
    
    def __init__(self):
        self._pods: Dict[str, PodRegistration] = {}
        self._by_symbol: Dict[str, List[str]] = {}  # symbol -> pod names
        self._by_timeframe: Dict[str, List[str]] = {}  # timeframe -> pod names
        logger.info("AlphaPodRegistry initialized")
    
    # =========================================================================
    # Registration
    # =========================================================================
    
    def register(self, pod: AlphaPod) -> bool:
        """
        Register an alpha pod.
        
        Args:
            pod: AlphaPod instance
            
        Returns:
            bool: Success
        """
        try:
            # Validate
            if not self._validate_pod(pod):
                logger.error(f"Pod {pod.name} validation failed")
                return False
            
            # Check if already registered
            if pod.name in self._pods:
                logger.warning(f"Pod {pod.name} already registered, updating")
                self.unregister(pod.name)
            
            # Create registration
            registration = PodRegistration(
                pod=pod,
                registered_at=datetime.now()
            )
            
            # Add to registry
            self._pods[pod.name] = registration
            
            # Index by symbol
            for symbol in pod.universe:
                if symbol not in self._by_symbol:
                    self._by_symbol[symbol] = []
                self._by_symbol[symbol].append(pod.name)
            
            # Index by timeframe
            tf = pod.timeframe
            if tf not in self._by_timeframe:
                self._by_timeframe[tf] = []
            self._by_timeframe[tf].append(pod.name)
            
            # Initialize pod
            if not pod.initialize():
                logger.error(f"Pod {pod.name} initialization failed")
                self.unregister(pod.name)
                return False
            
            logger.info(f"Pod registered: {pod.name} v{pod.version}")
            logger.info(f"  Universe: {pod.universe}")
            logger.info(f"  Timeframe: {pod.timeframe}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to register pod {pod.name}: {e}")
            return False
    
    def unregister(self, pod_name: str) -> bool:
        """
        Unregister an alpha pod.
        
        Args:
            pod_name: Pod name
            
        Returns:
            bool: Success
        """
        try:
            if pod_name not in self._pods:
                logger.warning(f"Pod {pod_name} not registered")
                return False
            
            pod = self._pods[pod_name].pod
            
            # Remove from indices
            for symbol in pod.universe:
                if symbol in self._by_symbol:
                    self._by_symbol[symbol] = [
                        n for n in self._by_symbol[symbol] if n != pod_name
                    ]
            
            tf = pod.timeframe
            if tf in self._by_timeframe:
                self._by_timeframe[tf] = [
                    n for n in self._by_timeframe[tf] if n != pod_name
                ]
            
            # Shutdown pod
            pod.shutdown()
            
            # Remove from registry
            del self._pods[pod_name]
            
            logger.info(f"Pod unregistered: {pod_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unregister pod {pod_name}: {e}")
            return False
    
    def _validate_pod(self, pod: AlphaPod) -> bool:
        """
        Validate alpha pod.
        
        Args:
            pod: Pod to validate
            
        Returns:
            bool: Valid
        """
        # Check required properties
        required = ['name', 'version', 'universe', 'timeframe']
        for prop in required:
            if not hasattr(pod, prop):
                logger.error(f"Pod missing required property: {prop}")
                return False
            if getattr(pod, prop) is None:
                logger.error(f"Pod property is None: {prop}")
                return False
        
        # Check required methods
        required_methods = ['generate_signal', 'get_features']
        for method in required_methods:
            if not hasattr(pod, method):
                logger.error(f"Pod missing required method: {method}")
                return False
        
        # Check universe not empty
        if not pod.universe:
            logger.error(f"Pod {pod.name} has empty universe")
            return False
        
        return True
    
    # =========================================================================
    # Access
    # =========================================================================
    
    def get_pod(self, pod_name: str) -> Optional[AlphaPod]:
        """
        Get pod by name.
        
        Args:
            pod_name: Pod name
            
        Returns:
            AlphaPod or None
        """
        if pod_name in self._pods:
            return self._pods[pod_name].pod
        return None
    
    def get_pods(self, 
                 active_only: bool = True,
                 symbol: Optional[str] = None,
                 timeframe: Optional[str] = None) -> List[AlphaPod]:
        """
        Get pods matching criteria.
        
        Args:
            active_only: Only active pods
            symbol: Filter by symbol
            timeframe: Filter by timeframe
            
        Returns:
            List of AlphaPods
        """
        pods = []
        
        # Get candidate names
        if symbol and timeframe:
            # Intersection
            by_sym = set(self._by_symbol.get(symbol, []))
            by_tf = set(self._by_timeframe.get(timeframe, []))
            names = by_sym & by_tf
        elif symbol:
            names = self._by_symbol.get(symbol, [])
        elif timeframe:
            names = self._by_timeframe.get(timeframe, [])
        else:
            names = list(self._pods.keys())
        
        # Filter
        for name in names:
            reg = self._pods.get(name)
            if reg:
                if not active_only or reg.is_active:
                    pods.append(reg.pod)
        
        return pods
    
    def get_all_pods(self, active_only: bool = True) -> List[AlphaPod]:
        """
        Get all pods.
        
        Args:
            active_only: Only active pods
            
        Returns:
            List of AlphaPods
        """
        pods = []
        for reg in self._pods.values():
            if not active_only or reg.is_active:
                pods.append(reg.pod)
        return pods
    
    def get_registration(self, pod_name: str) -> Optional[PodRegistration]:
        """Get registration record."""
        return self._pods.get(pod_name)
    
    # =========================================================================
    # Lifecycle
    # =========================================================================
    
    def activate(self, pod_name: str) -> bool:
        """Activate pod."""
        if pod_name in self._pods:
            self._pods[pod_name].is_active = True
            logger.info(f"Pod activated: {pod_name}")
            return True
        return False
    
    def deactivate(self, pod_name: str, reason: str = "") -> bool:
        """Deactivate pod."""
        if pod_name in self._pods:
            self._pods[pod_name].is_active = False
            logger.info(f"Pod deactivated: {pod_name} ({reason})")
            return True
        return False
    
    def activate_all(self):
        """Activate all pods."""
        for name in self._pods:
            self._pods[name].is_active = True
        logger.info("All pods activated")
    
    def deactivate_all(self, reason: str = ""):
        """Deactivate all pods."""
        for name in self._pods:
            self._pods[name].is_active = False
        logger.info(f"All pods deactivated ({reason})")
    
    # =========================================================================
    # Signal Tracking
    # =========================================================================
    
    def record_signal(self, pod_name: str, signal: AlphaSignal):
        """
        Record signal from pod.
        
        Args:
            pod_name: Pod name
            signal: Generated signal
        """
        if pod_name in self._pods:
            reg = self._pods[pod_name]
            reg.last_signal = datetime.now()
            reg.signal_count += 1
    
    def record_error(self, pod_name: str, error: str):
        """
        Record error from pod.
        
        Args:
            pod_name: Pod name
            error: Error message
        """
        if pod_name in self._pods:
            self._pods[pod_name].error_count += 1
            logger.error(f"Pod {pod_name} error: {error}")
    
    # =========================================================================
    # Health & Status
    # =========================================================================
    
    def health_check(self) -> Dict[str, Any]:
        """
        Run health check on all pods.
        
        Returns:
            Health status
        """
        status = {
            'total': len(self._pods),
            'active': 0,
            'inactive': 0,
            'healthy': 0,
            'pods': {}
        }
        
        for name, reg in self._pods.items():
            pod_status = reg.pod.get_status()
            pod_status['registered_at'] = reg.registered_at.isoformat()
            pod_status['is_active'] = reg.is_active
            pod_status['signal_count'] = reg.signal_count
            pod_status['error_count'] = reg.error_count
            
            status['pods'][name] = pod_status
            
            if reg.is_active:
                status['active'] += 1
            else:
                status['inactive'] += 1
            
            if reg.error_count < 10:
                status['healthy'] += 1
        
        return status
    
    def get_status(self) -> Dict[str, Any]:
        """Get registry status."""
        return {
            'total_pods': len(self._pods),
            'active_pods': sum(1 for r in self._pods.values() if r.is_active),
            'by_symbol': {k: len(v) for k, v in self._by_symbol.items()},
            'by_timeframe': {k: len(v) for k, v in self._by_timeframe.items()}
        }
    
    # =========================================================================
    # Persistence
    # =========================================================================
    
    def save_state(self, filepath: str):
        """Save registry state."""
        state = {
            'pods': {
                name: {
                    'registered_at': reg.registered_at.isoformat(),
                    'is_active': reg.is_active,
                    'signal_count': reg.signal_count,
                    'error_count': reg.error_count
                }
                for name, reg in self._pods.items()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Registry state saved to {filepath}")
    
    def load_state(self, filepath: str):
        """Load registry state."""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            # Note: Cannot restore pods, just statistics
            logger.info(f"Registry state loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load registry state: {e}")
    
    # =========================================================================
    # Utilities
    # =========================================================================
    
    def __len__(self) -> int:
        """Number of registered pods."""
        return len(self._pods)
    
    def __contains__(self, pod_name: str) -> bool:
        """Check if pod is registered."""
        return pod_name in self._pods
    
    def __iter__(self):
        """Iterate over pod names."""
        return iter(self._pods.keys())


# =============================================================================
# FACTORY
# =============================================================================

class AlphaPodFactory:
    """
    Factory for creating alpha pods.
    
    Usage:
        factory = AlphaPodFactory()
        factory.register_pod_type('fx_momentum', FXMomentumAlphaPod)
        
        pod = factory.create('fx_momentum', config)
    """
    
    def __init__(self):
        self._pod_types: Dict[str, Type[AlphaPod]] = {}
        logger.info("AlphaPodFactory initialized")
    
    def register(self, pod_type: str, pod_class: Type[AlphaPod]):
        """
        Register pod type.
        
        Args:
            pod_type: Type identifier
            pod_class: Pod class
        """
        self._pod_types[pod_type] = pod_class
        logger.info(f"Pod type registered: {pod_type}")
    
    def create(self, pod_type: str, config: Dict[str, Any]) -> Optional[AlphaPod]:
        """
        Create pod instance.
        
        Args:
            pod_type: Type identifier
            config: Pod configuration
            
        Returns:
            AlphaPod instance or None
        """
        if pod_type not in self._pod_types:
            logger.error(f"Unknown pod type: {pod_type}")
            return None
        
        try:
            pod_class = self._pod_types[pod_type]
            pod = pod_class(config)
            return pod
        except Exception as e:
            logger.error(f"Failed to create pod {pod_type}: {e}")
            return None
    
    def get_available_types(self) -> List[str]:
        """Get available pod types."""
        return list(self._pod_types.keys())
