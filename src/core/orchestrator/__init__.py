"""
Adaptive Trading Orchestrator

Central command and control for the trading system.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path

# Core interfaces
from ..interfaces import (
    AlphaPod, AlphaSignal, Decision, MarketData, ExecutionResult,
    SignalDirection, DecisionAction, PipelineStage,
    DataServiceInterface, RiskServiceInterface, 
    ExecutionServiceInterface, MonitoringServiceInterface,
    OrchestratorError, RiskLimitError, ExecutionError
)

# Registry
from ..registry import AlphaPodRegistry, AlphaPodFactory

# Services (to be implemented)
# from ..services import DataService, RiskService, ExecutionService, MonitoringService

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorConfig:
    """Orchestrator configuration."""
    mode: str = "backtest"  # backtest, paper, live
    tick_interval: float = 1.0
    max_active_pods: int = 10
    
    # Kill switch
    max_drawdown: float = 0.20
    max_daily_loss: float = 0.05
    error_threshold: int = 10
    circuit_breaker_enabled: bool = True
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    # Feature flags
    use_alpha_pods: bool = True
    use_ml_brain: bool = True
    use_rl_brain: bool = False  # Disabled by default (stub)
    use_portfolio_brain: bool = False  # Disabled by default
    use_unified_pipeline: bool = False
    
    # Ensemble
    ensemble_method: str = "weighted"  # weighted, majority, best


@dataclass
class OrchestratorState:
    """Orchestrator runtime state."""
    is_running: bool = False
    is_paused: bool = False
    is_initialized: bool = False
    
    # Market state
    current_regime: str = "unknown"
    last_data_timestamp: Optional[datetime] = None
    
    # Pod state
    active_pods: int = 0
    total_signals_today: int = 0
    last_signal_time: Optional[datetime] = None
    
    # Execution state
    last_decision_time: Optional[datetime] = None
    decisions_today: int = 0
    trades_today: int = 0
    
    # Error tracking
    error_count: int = 0
    last_error: Optional[str] = None
    
    # Performance
    daily_pnl: float = 0.0
    total_pnl: float = 0.0
    peak_equity: float = 0.0
    current_drawdown: float = 0.0


class AdaptiveTradingOrchestrator:
    """
    Main orchestrator for Adaptive Trading OS.
    
    Responsibilities:
    - Manage alpha pod lifecycle
    - Coordinate shared services
    - Run unified pipeline
    - Handle events
    - Manage kill switches
    """
    
    def __init__(self, config: Optional[OrchestratorConfig] = None):
        """
        Initialize orchestrator.
        
        Args:
            config: Orchestrator configuration
        """
        self.config = config or OrchestratorConfig()
        self.state = OrchestratorState()
        
        # Core components
        self.registry = AlphaPodRegistry()
        self.factory = AlphaPodFactory()
        
        # Services (initialized later)
        self.data_service: Optional[DataServiceInterface] = None
        self.risk_service: Optional[RiskServiceInterface] = None
        self.execution_service: Optional[ExecutionServiceInterface] = None
        self.monitoring_service: Optional[MonitoringServiceInterface] = None
        
        # Pipeline (simplified - full implementation later)
        from src.pipeline.adapter import UnifiedPipelineAdapter
        self.pipeline = UnifiedPipelineAdapter(self.config)
        
        # Event handlers
        self._event_handlers: Dict[str, List[Callable]] = {}
        self._running = False
        
        # Setup logging
        self._setup_logging()
        
        logger.info("AdaptiveTradingOrchestrator initialized")
        logger.info(f"  Mode: {self.config.mode}")
        logger.info(f"  Alpha pods: {self.config.use_alpha_pods}")
        logger.info(f"  ML brain: {self.config.use_ml_brain}")
        logger.info(f"  Portfolio brain: {self.config.use_portfolio_brain}")
    
    def _setup_logging(self):
        """Setup logging."""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        if self.config.log_file:
            handler = logging.FileHandler(self.config.log_file)
            handler.setLevel(getattr(logging, self.config.log_level))
            logger.addHandler(handler)
    
    # ========================================================================
    # SERVICE INJECTION
    # ========================================================================
    
    def set_data_service(self, service: DataServiceInterface):
        """Set data service."""
        self.data_service = service
        logger.info("Data service registered")
    
    def set_risk_service(self, service: RiskServiceInterface):
        """Set risk service."""
        self.risk_service = service
        logger.info("Risk service registered")
    
    def set_execution_service(self, service: ExecutionServiceInterface):
        """Set execution service."""
        self.execution_service = service
        logger.info("Execution service registered")
    
    def set_monitoring_service(self, service: MonitoringServiceInterface):
        """Set monitoring service."""
        self.monitoring_service = service
        logger.info("Monitoring service registered")
    
    # ========================================================================
    # ALPHA POD MANAGEMENT
    # ========================================================================
    
    def register_alpha_pod(self, pod: AlphaPod) -> bool:
        """
        Register an alpha pod.
        
        Args:
            pod: AlphaPod instance
            
        Returns:
            bool: Success
        """
        # Inject services
        if self.data_service:
            pod.set_data_service(self.data_service)
        if self.risk_service:
            pod.set_risk_service(self.risk_service)
        
        # Register
        success = self.registry.register(pod)
        
        if success:
            self.state.active_pods = len(self.registry)
            self._emit_event('pod_registered', {
                'pod_name': pod.name,
                'universe': pod.universe
            })
        
        return success
    
    def register_pod_type(self, pod_type: str, pod_class: Any):
        """Register pod type with factory."""
        self.factory.register(pod_type, pod_class)
    
    def create_and_register_pod(self, pod_type: str, config: Dict) -> bool:
        """Create and register pod from factory."""
        pod = self.factory.create(pod_type, config)
        if pod:
            return self.register_alpha_pod(pod)
        return False
    
    def unregister_alpha_pod(self, pod_name: str) -> bool:
        """Unregister an alpha pod."""
        success = self.registry.unregister(pod_name)
        if success:
            self.state.active_pods = len(self.registry)
        return success
    
    def get_alpha_pods(self, active_only: bool = True) -> List[AlphaPod]:
        """Get registered alpha pods."""
        return self.registry.get_pods(active_only=active_only)
    
    def get_pod(self, pod_name: str) -> Optional[AlphaPod]:
        """Get specific pod."""
        return self.registry.get_pod(pod_name)
    
    # ========================================================================
    # LIFECYCLE MANAGEMENT
    # ========================================================================
    
    async def initialize(self) -> bool:
        """
        Initialize orchestrator.
        
        Returns:
            bool: Success
        """
        try:
            logger.info("Initializing orchestrator...")
            
            # Validate services
            if not self._validate_services():
                raise OrchestratorError("Service validation failed")
            
            # Initialize services
            if self.data_service:
                self.data_service.initialize()
            if self.risk_service:
                await self.risk_service.initialize()
            if self.execution_service:
                await self.execution_service.initialize()
            if self.monitoring_service:
                await self.monitoring_service.initialize()
            
            self.state.is_initialized = True
            logger.info("Orchestrator initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            self.state.last_error = str(e)
            self.state.error_count += 1
            return False
    
    def _validate_services(self) -> bool:
        """Validate required services."""
        required = ['data_service', 'risk_service', 'execution_service']
        
        for service_name in required:
            service = getattr(self, service_name)
            if service is None:
                logger.warning(f"Service not set: {service_name}")
                # Don't fail - some services optional in backtest
        
        return True
    
    async def start(self) -> bool:
        """Start trading."""
        if self.state.is_running:
            logger.warning("Orchestrator already running")
            return True
        
        try:
            logger.info("Starting orchestrator...")
            
            # Initialize if needed
            if not self.state.is_initialized:
                if not await self.initialize():
                    return False
            
            self.state.is_running = True
            self._running = True
            
            # Start event loop
            await self._event_loop()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start: {e}")
            self.state.is_running = False
            self._running = False
            return False
    
    async def stop(self, graceful: bool = True):
        """Stop trading."""
        logger.info("Stopping orchestrator...")
        
        self._running = False
        self.state.is_running = False
        
        if graceful and self.execution_service:
            # Close positions
            await self._close_all_positions()
        
        # Shutdown services
        if self.execution_service:
            await self.execution_service.shutdown()
        if self.data_service:
            self.data_service.shutdown()
        
        logger.info("Orchestrator stopped")
    
    async def pause(self):
        """Pause trading."""
        self.state.is_paused = True
        logger.info("Orchestrator paused")
    
    async def resume(self):
        """Resume trading."""
        self.state.is_paused = False
        logger.info("Orchestrator resumed")
    
    # ========================================================================
    # EVENT LOOP
    # ========================================================================
    
    async def _event_loop(self):
        """Main event loop."""
        logger.info("Event loop started")
        
        while self._running:
            try:
                # Check if paused
                if self.state.is_paused:
                    await asyncio.sleep(1)
                    continue
                
                # Get market data
                if self.data_service:
                    data = self.data_service.get_latest("EURUSD")  # Default
                    self.state.last_data_timestamp = datetime.now()
                else:
                    logger.warning("No data service")
                    await asyncio.sleep(1)
                    continue
                
                # Process
                await self._process_tick(data)
                
                # Sleep
                await asyncio.sleep(self.config.tick_interval)
                
            except Exception as e:
                logger.error(f"Event loop error: {e}")
                self.state.error_count += 1
                self.state.last_error = str(e)
                
                if self.state.error_count > self.config.error_threshold:
                    await self._trigger_kill_switch("ERROR_THRESHOLD")
                    break
        
        logger.info("Event loop stopped")
    
    async def _process_tick(self, data: MarketData):
        """Process single tick."""
        
        # Collect signals from alpha pods
        signals = []
        
        if self.config.use_alpha_pods:
            for pod in self.registry.get_pods(active_only=True):
                try:
                    signal = pod.generate_signal(data)
                    if signal:
                        signals.append(signal)
                        self.registry.record_signal(pod.name, signal)
                        self.state.total_signals_today += 1
                except Exception as e:
                    logger.error(f"Pod {pod.name} error: {e}")
                    self.registry.record_error(pod.name, str(e))
        
        self.state.last_signal_time = datetime.now()
        
        if not signals:
            return
        
        # Run pipeline
        decision = self.pipeline.process(signals, data)
        
        # Check kill switch
        if self._should_kill_switch(decision):
            await self._trigger_kill_switch("PIPELINE")
            return
        
        # Execute or pass
        if decision.action != DecisionAction.PASS:
            await self._execute_decision(decision, signals)
        
        self.state.last_decision_time = datetime.now()
    
    # ========================================================================
    # PIPELINE
    # ========================================================================
    
    def process_signals(self, 
                       signals: List[AlphaSignal], 
                       data: MarketData) -> Decision:
        """
        Process signals through pipeline.
        
        Args:
            signals: List of alpha signals
            data: Market data
            
        Returns:
            Decision
        """
        return self.pipeline.process(signals, data)
    
    # ========================================================================
    # EXECUTION
    # ========================================================================
    
    async def _execute_decision(self, 
                                decision: Decision, 
                                signals: List[AlphaSignal]):
        """Execute decision."""
        
        # Risk check
        if self.risk_service:
            allowed, reason = self.risk_service.check_limits(decision)
            if not allowed:
                logger.warning(f"Risk check blocked: {reason}")
                return
        
        # Execute
        if self.execution_service:
            result = await self.execution_service.execute(decision)
            
            # Log
            if self.monitoring_service:
                self.monitoring_service.log_decision(decision, signals, result)
            
            # Update state
            self.state.decisions_today += 1
            if result.status == "filled":
                self.state.trades_today += 1
            
            # Emit event
            self._emit_event('trade_executed', {
                'decision': decision,
                'result': result
            })
            
            # Notify pods
            for signal in signals:
                pod = self.registry.get_pod(signal.source)
                if pod:
                    pod.on_trade_executed(result)
    
    async def _close_all_positions(self):
        """Close all positions."""
        if not self.execution_service:
            return
        
        positions = await self.execution_service.get_positions()
        for pos in positions:
            await self.execution_service.close_position(pos['id'])
        
        logger.info(f"Closed {len(positions)} positions")
    
    # ========================================================================
    # KILL SWITCH
    # ========================================================================
    
    def _should_kill_switch(self, decision: Decision) -> bool:
        """Check if kill switch should trigger."""
        
        # Circuit breaker
        if self.config.circuit_breaker_enabled:
            if self.state.current_drawdown > self.config.max_drawdown:
                return True
        
        # Daily loss
        if self.state.daily_pnl < -self.config.max_daily_loss:
            return True
        
        # Error threshold
        if self.state.error_count > self.config.error_threshold:
            return True
        
        return False
    
    async def _trigger_kill_switch(self, reason: str):
        """Trigger kill switch."""
        logger.critical(f"KILL SWITCH TRIGGERED: {reason}")
        
        # Stop
        self._running = False
        
        # Close positions
        await self._close_all_positions()
        
        # Alert
        if self.monitoring_service:
            self.monitoring_service.alert_critical(f"Kill switch: {reason}")
        
        # Emit event
        self._emit_event('kill_switch', {'reason': reason})
    
    def manual_kill_switch(self, reason: str = "manual"):
        """Manual kill switch."""
        asyncio.create_task(self._trigger_kill_switch(reason))
    
    # ========================================================================
    # EVENTS
    # ========================================================================
    
    def on_event(self, event_type: str, handler: Callable):
        """Register event handler."""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)
    
    def _emit_event(self, event_type: str, data: Dict):
        """Emit event."""
        for handler in self._event_handlers.get(event_type, []):
            try:
                handler(data)
            except Exception as e:
                logger.error(f"Event handler error: {e}")
    
    # ========================================================================
    # STATUS
    # ========================================================================
    
    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status."""
        return {
            'running': self.state.is_running,
            'paused': self.state.is_paused,
            'initialized': self.state.is_initialized,
            'active_pods': self.state.active_pods,
            'total_signals_today': self.state.total_signals_today,
            'decisions_today': self.state.decisions_today,
            'trades_today': self.state.trades_today,
            'error_count': self.state.error_count,
            'current_drawdown': self.state.current_drawdown,
            'last_decision': self.state.last_decision_time.isoformat() if self.state.last_decision_time else None
        }
    
    def get_health(self) -> Dict[str, Any]:
        """Get health status."""
        return {
            'orchestrator': self.state.is_running,
            'services': {
                'data': self.data_service is not None,
                'risk': self.risk_service is not None,
                'execution': self.execution_service is not None,
                'monitoring': self.monitoring_service is not None
            },
            'pods': self.registry.health_check(),
            'errors': self.state.error_count
        }


# =============================================================================
# SIMPLE PIPELINE (Placeholder for full implementation)
# =============================================================================

class SimplePipeline:
    """Simplified pipeline for initial implementation."""
    
    def __init__(self, config: OrchestratorConfig):
        self.config = config
    
    def process(self, signals: List[AlphaSignal], data: MarketData) -> Decision:
        """
        Process signals through simplified pipeline.
        
        Stages:
        1. Ensemble signals (Stage 3)
        2. Risk sizing (Stage 7)
        3. Decision (Stage 12)
        """
        
        if not signals:
            return Decision(
                action=DecisionAction.PASS,
                symbol="",
                side="",
                size=0,
                entry_price=0,
                stop_loss=0,
                take_profit=0
            )
        
        # Stage 3: Ensemble
        best_signal = self._ensemble(signals)
        
        # Stage 7: Risk
        size = self._calculate_size(best_signal)
        sl = self._calculate_stop(best_signal)
        tp = self._calculate_target(best_signal)
        
        # Stage 12: Decision
        action = DecisionAction.BUY if best_signal.direction == SignalDirection.LONG else DecisionAction.SELL
        
        return Decision(
            action=action,
            symbol=best_signal.symbol,
            side=best_signal.direction.value,
            size=size,
            entry_price=data.close,
            stop_loss=sl,
            take_profit=tp,
            sources=[s.source for s in signals],
            signals=signals,
            confidence=best_signal.confidence,
            timestamp=datetime.now()
        )
    
    def _ensemble(self, signals: List[AlphaSignal]) -> AlphaSignal:
        """Combine signals."""
        # Simple: pick highest confidence
        return max(signals, key=lambda s: s.confidence)
    
    def _calculate_size(self, signal: AlphaSignal) -> float:
        """Calculate position size."""
        # Simple: use recommended size
        return min(signal.recommended_size, 0.1)  # Max 10%
    
    def _calculate_stop(self, signal: AlphaSignal) -> float:
        """Calculate stop loss."""
        if signal.stop_loss:
            return signal.stop_loss
        # Default: 2% stop
        return signal.entry_price * 0.98 if hasattr(signal, 'entry_price') else 0
    
    def _calculate_target(self, signal: AlphaSignal) -> float:
        """Calculate take profit."""
        if signal.take_profit:
            return signal.take_profit
        # Default: 4% target (2:1 RR)
        return signal.entry_price * 1.04 if hasattr(signal, 'entry_price') else 0
