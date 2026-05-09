"""
Autonomous Runtime - Central Nervous System

Continuously running autonomous adaptive trading runtime.

This is the PRIMARY runtime entrypoint for the Adaptive Trading OS.

Usage:
    python -m src.runtime.autonomous_runtime

Features:
- Continuous market data processing
- Pipeline orchestration
- Meta-gating integration
- Risk control enforcement
- Execution coordination
- Telemetry and logging
- Health monitoring
- Fault tolerance
- Graceful shutdown
- Restart recovery
"""

import asyncio
import logging
import signal
import sys
import time
import json
import psutil
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
from enum import Enum

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            f"logs/runtime_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
    ]
)
logger = logging.getLogger(__name__)


class RuntimeMode(Enum):
    """Runtime operating modes."""
    PAPER = "paper"
    LIVE = "live"
    BACKTEST = "backtest"


class RuntimeHealth(Enum):
    """Runtime health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    PAUSED = "paused"


@dataclass
class RuntimeConfig:
    """Runtime configuration."""
    # Symbols to trade
    symbols: List[str] = field(default_factory=lambda: [
        "EURUSD", "GBPUSD", "USDJPY", "XAGUSD"
    ])
    
    # Timeframes
    timeframes: List[str] = field(default_factory=lambda: ["1H", "4H", "D1"])
    
    # Polling
    polling_interval_sec: float = 1.0
    bar_check_interval_sec: float = 5.0
    
    # Mode
    mode: RuntimeMode = RuntimeMode.PAPER
    
    # Error handling
    max_errors_per_symbol: int = 10
    max_consecutive_errors: int = 3
    error_cooldown_sec: float = 60.0
    
    # Heartbeat
    heartbeat_interval_sec: float = 30.0
    
    # Paths
    log_dir: str = "logs/runtime"
    state_dir: str = "state/runtime"
    telemetry_dir: str = "logs/telemetry"
    
    # Circuit breaker
    enable_circuit_breaker: bool = True
    circuit_breaker_threshold: int = 5
    circuit_breaker_duration_sec: float = 300.0
    
    # Pipeline
    enable_pipeline_v2: bool = True
    enable_meta_gating: bool = True
    
    # Execution
    enable_executor_v2: bool = True
    prevent_duplicate_executions: bool = True
    execution_cooldown_sec: float = 300.0  # 5 min between executions
    
    # Monitoring
    enable_drift_monitor: bool = True
    enable_health_monitor: bool = True
    
    def __post_init__(self):
        """Ensure directories exist."""
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        Path(self.state_dir).mkdir(parents=True, exist_ok=True)
        Path(self.telemetry_dir).mkdir(parents=True, exist_ok=True)


@dataclass
class SymbolState:
    """Per-symbol runtime state."""
    symbol: str
    last_bar_time: Optional[datetime] = None
    last_processed_bar: Optional[datetime] = None
    processed_count: int = 0
    error_count: int = 0
    consecutive_errors: int = 0
    last_error_time: Optional[datetime] = None
    health: str = "healthy"
    latency_ms: float = 0.0
    
    def can_process(self, bar_time: datetime) -> bool:
        """Check if bar can be processed (no duplicates)."""
        if self.last_processed_bar is None:
            return True
        return bar_time > self.last_processed_bar


@dataclass
class RuntimeState:
    """Global runtime state."""
    # Boot info
    start_time: datetime = field(default_factory=datetime.now)
    boot_count: int = 0
    
    # Loop metrics
    loop_count: int = 0
    total_bars_processed: int = 0
    total_signals_generated: int = 0
    total_executions: int = 0
    
    # Timing
    last_loop_time: Optional[datetime] = None
    last_heartbeat_time: Optional[datetime] = None
    last_bar_processed_time: Optional[datetime] = None
    
    # Errors
    total_errors: int = 0
    consecutive_errors: int = 0
    
    # Symbol states
    symbol_states: Dict[str, SymbolState] = field(default_factory=dict)
    
    # Health
    health: str = "healthy"
    circuit_breaker_active: bool = False
    circuit_breaker_until: Optional[datetime] = None
    
    # Latency tracking
    avg_loop_latency_ms: float = 0.0
    max_loop_latency_ms: float = 0.0
    p95_loop_latency_ms: float = 0.0
    
    def __post_init__(self):
        """Initialize symbol states."""
        if not self.symbol_states:
            self.symbol_states = {}
    
    def get_symbol_state(self, symbol: str) -> SymbolState:
        """Get or create symbol state."""
        if symbol not in self.symbol_states:
            self.symbol_states[symbol] = SymbolState(symbol=symbol)
        return self.symbol_states[symbol]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'start_time': self.start_time.isoformat(),
            'boot_count': self.boot_count,
            'loop_count': self.loop_count,
            'total_bars_processed': self.total_bars_processed,
            'total_signals_generated': self.total_signals_generated,
            'total_executions': self.total_executions,
            'total_errors': self.total_errors,
            'health': self.health,
            'circuit_breaker_active': self.circuit_breaker_active,
            'symbol_states': {
                s: {
                    'last_processed_bar': state.last_processed_bar.isoformat() if state.last_processed_bar else None,
                    'processed_count': state.processed_count,
                    'error_count': state.error_count,
                    'health': state.health
                }
                for s, state in self.symbol_states.items()
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RuntimeState':
        """Create from dictionary."""
        state = cls()
        state.start_time = datetime.fromisoformat(data.get('start_time', datetime.now().isoformat()))
        state.boot_count = data.get('boot_count', 0)
        state.loop_count = data.get('loop_count', 0)
        state.total_bars_processed = data.get('total_bars_processed', 0)
        state.total_signals_generated = data.get('total_signals_generated', 0)
        state.total_executions = data.get('total_executions', 0)
        state.total_errors = data.get('total_errors', 0)
        state.health = data.get('health', 'healthy')
        state.circuit_breaker_active = data.get('circuit_breaker_active', False)
        
        # Restore symbol states
        for symbol, symbol_data in data.get('symbol_states', {}).items():
            ss = SymbolState(symbol=symbol)
            if symbol_data.get('last_processed_bar'):
                ss.last_processed_bar = datetime.fromisoformat(symbol_data['last_processed_bar'])
            ss.processed_count = symbol_data.get('processed_count', 0)
            ss.error_count = symbol_data.get('error_count', 0)
            ss.health = symbol_data.get('health', 'healthy')
            state.symbol_states[symbol] = ss
        
        return state


class MarketDataCoordinator:
    """Coordinates market data fetching and caching."""
    
    def __init__(self, config: RuntimeConfig):
        self.config = config
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._data_fetcher = None  # Would initialize with real data source
        
        logger.info("MarketDataCoordinator initialized")
        logger.info(f"  Symbols: {config.symbols}")
        logger.info(f"  Timeframes: {config.timeframes}")
    
    async def fetch_latest_bars(self) -> Dict[str, Dict[str, Any]]:
        """Fetch latest bars for all symbols."""
        bars = {}
        
        for symbol in self.config.symbols:
            try:
                for tf in self.config.timeframes:
                    bar = await self._fetch_bar(symbol, tf)
                    if bar:
                        key = f"{symbol}_{tf}"
                        bars[key] = bar
                        self._cache[key] = bar
            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")
        
        return bars
    
    async def _fetch_bar(self, symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
        """Fetch single bar."""
        # Placeholder - would integrate with actual data source
        # For now, return simulated bar
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'timestamp': datetime.now().replace(second=0, microsecond=0),
            'open': 1.0,
            'high': 1.1,
            'low': 0.9,
            'close': 1.05,
            'volume': 1000
        }
    
    def get_cached_bar(self, symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
        """Get cached bar."""
        key = f"{symbol}_{tf}"
        return self._cache.get(key)
    
    def is_new_bar(self, symbol: str, timeframe: str, 
                   bar_time: datetime, last_processed: Optional[datetime]) -> bool:
        """Check if bar is new (not yet processed)."""
        if last_processed is None:
            return True
        return bar_time > last_processed


class RuntimeHeartbeat:
    """Writes periodic runtime health updates."""
    
    def __init__(self, config: RuntimeConfig, state: RuntimeState):
        self.config = config
        self.state = state
        self._log_path = Path(config.log_dir) / "runtime_health.json"
        self._latency_history: List[float] = []
        
        logger.info(f"RuntimeHeartbeat initialized: {self._log_path}")
    
    async def write(self):
        """Write current health status."""
        try:
            # Calculate metrics
            uptime_sec = (datetime.now() - self.state.start_time).total_seconds()
            memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
            
            health_data = {
                'timestamp': datetime.now().isoformat(),
                'uptime_seconds': uptime_sec,
                'uptime_hours': uptime_sec / 3600,
                'mode': self.config.mode.value,
                'health': self.state.health,
                'circuit_breaker_active': self.state.circuit_breaker_active,
                'metrics': {
                    'loops_completed': self.state.loop_count,
                    'bars_processed': self.state.total_bars_processed,
                    'signals_generated': self.state.total_signals_generated,
                    'executions': self.state.total_executions,
                    'errors': self.state.total_errors
                },
                'memory_mb': memory_mb,
                'latency_ms': {
                    'avg': self.state.avg_loop_latency_ms,
                    'max': self.state.max_loop_latency_ms,
                    'p95': self.state.p95_loop_latency_ms
                },
                'symbol_health': {
                    symbol: {
                        'processed': ss.processed_count,
                        'errors': ss.error_count,
                        'health': ss.health,
                        'last_processed': ss.last_processed_bar.isoformat() if ss.last_processed_bar else None
                    }
                    for symbol, ss in self.state.symbol_states.items()
                }
            }
            
            # Write to file
            with open(self._log_path, 'w') as f:
                json.dump(health_data, f, indent=2)
            
            self.state.last_heartbeat_time = datetime.now()
            
        except Exception as e:
            logger.error(f"Failed to write heartbeat: {e}")
    
    def record_latency(self, latency_ms: float):
        """Record loop latency."""
        self._latency_history.append(latency_ms)
        
        # Keep last 100 measurements
        if len(self._latency_history) > 100:
            self._latency_history = self._latency_history[-100:]
        
        # Update statistics
        if self._latency_history:
            import numpy as np
            self.state.avg_loop_latency_ms = float(np.mean(self._latency_history))
            self.state.max_loop_latency_ms = float(np.max(self._latency_history))
            self.state.p95_loop_latency_ms = float(np.percentile(self._latency_history, 95))


class PipelineCoordinator:
    """Coordinates pipeline execution."""
    
    def __init__(self, config: RuntimeConfig, state: RuntimeState):
        self.config = config
        self.state = state
        self._pipeline = None  # Would initialize PipelineV2
        
        logger.info("PipelineCoordinator initialized")
    
    async def process_bar(self, symbol: str, timeframe: str, 
                         bar_data: Dict[str, Any]) -> Optional[Any]:
        """Process a single bar through pipeline."""
        try:
            start_time = time.time()
            
            # Get symbol state
            symbol_state = self.state.get_symbol_state(symbol)
            
            # Check for duplicate
            bar_time = bar_data.get('timestamp')
            if isinstance(bar_time, str):
                bar_time = datetime.fromisoformat(bar_time)
            
            if not symbol_state.can_process(bar_time):
                logger.debug(f"Skipping duplicate bar: {symbol} {timeframe} {bar_time}")
                return None
            
            # TODO: Integrate with actual PipelineV2
            # result = await self._pipeline.process_bar(symbol, timeframe, bar_data)
            
            # Simulate pipeline execution
            await asyncio.sleep(0.01)  # Simulate processing time
            
            result = {
                'symbol': symbol,
                'timeframe': timeframe,
                'timestamp': bar_time,
                'signal': 'neutral',
                'confidence': 0.5,
                'bar_result': bar_data
            }
            
            # Update state
            symbol_state.last_processed_bar = bar_time
            symbol_state.processed_count += 1
            symbol_state.latency_ms = (time.time() - start_time) * 1000
            
            self.state.total_bars_processed += 1
            self.state.last_bar_processed_time = datetime.now()
            
            # Reset error tracking on success
            symbol_state.consecutive_errors = 0
            
            return result
            
        except Exception as e:
            logger.error(f"Pipeline error for {symbol}: {e}")
            await self._handle_pipeline_error(symbol, e)
            return None
    
    async def _handle_pipeline_error(self, symbol: str, error: Exception):
        """Handle pipeline error."""
        symbol_state = self.state.get_symbol_state(symbol)
        symbol_state.error_count += 1
        symbol_state.consecutive_errors += 1
        symbol_state.last_error_time = datetime.now()
        
        self.state.total_errors += 1
        self.state.consecutive_errors += 1
        
        # Mark symbol as degraded if too many errors
        if symbol_state.consecutive_errors >= self.config.max_consecutive_errors:
            symbol_state.health = "degraded"
            logger.warning(f"Symbol {symbol} marked as degraded due to errors")


class ExecutionCoordinator:
    """Coordinates trade execution."""
    
    def __init__(self, config: RuntimeConfig, state: RuntimeState):
        self.config = config
        self.state = state
        self._executor = None  # Would initialize ExecutorV2
        self._last_execution_time: Dict[str, datetime] = {}
        self._fills_log = Path(config.telemetry_dir) / "fills.jsonl"
        
        logger.info("ExecutionCoordinator initialized")
    
    async def execute_signal(self, signal: Dict[str, Any]) -> bool:
        """Execute approved signal."""
        try:
            symbol = signal.get('symbol')
            
            # Check for duplicate execution
            if self._is_duplicate_execution(symbol):
                logger.debug(f"Skipping duplicate execution: {symbol}")
                return False
            
            # TODO: Integrate with actual ExecutorV2
            # result = await self._executor.execute(signal)
            
            # Simulate execution
            await asyncio.sleep(0.005)
            
            # Log fill
            fill_record = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'signal': signal,
                'status': 'filled',
                'mode': self.config.mode.value
            }
            
            with open(self._fills_log, 'a') as f:
                f.write(json.dumps(fill_record) + '\n')
            
            self._last_execution_time[symbol] = datetime.now()
            self.state.total_executions += 1
            
            logger.info(f"Executed: {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Execution error: {e}")
            return False
    
    def _is_duplicate_execution(self, symbol: str) -> bool:
        """Check if execution would be duplicate."""
        if not self.config.prevent_duplicate_executions:
            return False
        
        last_exec = self._last_execution_time.get(symbol)
        if not last_exec:
            return False
        
        elapsed = (datetime.now() - last_exec).total_seconds()
        return elapsed < self.config.execution_cooldown_sec


class DriftAndHealthMonitor:
    """Monitors drift and runtime health."""
    
    def __init__(self, config: RuntimeConfig, state: RuntimeState):
        self.config = config
        self.state = state
        self._error_history: List[datetime] = []
        
        logger.info("DriftAndHealthMonitor initialized")
    
    async def check(self) -> bool:
        """Check health and return True if should continue."""
        try:
            now = datetime.now()
            
            # Check circuit breaker
            if self.state.circuit_breaker_active:
                if self.state.circuit_breaker_until and now < self.state.circuit_breaker_until:
                    logger.warning("Circuit breaker active - runtime paused")
                    self.state.health = "paused"
                    return False
                else:
                    # Circuit breaker expired
                    self.state.circuit_breaker_active = False
                    logger.info("Circuit breaker expired - resuming")
            
            # Check error rate
            recent_errors = [
                e for e in self._error_history
                if (now - e).total_seconds() < 300  # Last 5 min
            ]
            
            if len(recent_errors) >= self.config.circuit_breaker_threshold:
                if self.config.enable_circuit_breaker:
                    logger.critical("Error threshold exceeded - activating circuit breaker")
                    self.state.circuit_breaker_active = True
                    self.state.circuit_breaker_until = now + timedelta(
                        seconds=self.config.circuit_breaker_duration_sec
                    )
                    self.state.health = "critical"
                    return False
            
            # Check overall health
            if self.state.consecutive_errors >= self.config.max_consecutive_errors:
                self.state.health = "degraded"
            else:
                self.state.health = "healthy"
            
            return True
            
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return True  # Continue on error
    
    def record_error(self):
        """Record an error occurrence."""
        self._error_history.append(datetime.now())


class StatePersistence:
    """Handles runtime state persistence."""
    
    def __init__(self, state_dir: str):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.state_dir / "runtime_state.json"
        
        logger.info(f"StatePersistence: {self.state_file}")
    
    def save(self, state: RuntimeState):
        """Save runtime state."""
        try:
            with open(self.state_file, 'w') as f:
                json.dump(state.to_dict(), f, indent=2)
            logger.debug("Runtime state saved")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def load(self) -> Optional[RuntimeState]:
        """Load runtime state."""
        try:
            if not self.state_file.exists():
                logger.info("No previous state found - starting fresh")
                return None
            
            with open(self.state_file, 'r') as f:
                data = json.load(f)
            
            state = RuntimeState.from_dict(data)
            state.boot_count += 1  # Increment boot count
            logger.info(f"Runtime state restored (boot #{state.boot_count})")
            return state
            
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return None


class AutonomousRuntime:
    """
    Main autonomous runtime class.
    
    Continuously orchestrates the adaptive trading system.
    """
    
    def __init__(self, config: Optional[RuntimeConfig] = None):
        """Initialize autonomous runtime."""
        self.config = config or RuntimeConfig()
        
        # Load or create state
        self.state_persistence = StatePersistence(self.config.state_dir)
        self.state = self.state_persistence.load() or RuntimeState()
        
        # Initialize coordinators
        self.data_coordinator = MarketDataCoordinator(self.config)
        self.heartbeat = RuntimeHeartbeat(self.config, self.state)
        self.pipeline = PipelineCoordinator(self.config, self.state)
        self.execution = ExecutionCoordinator(self.config, self.state)
        self.health_monitor = DriftAndHealthMonitor(self.config, self.state)
        
        # Runtime control
        self._running = False
        self._shutdown_requested = False
        
        # Setup signal handlers
        self._setup_signal_handlers()
        
        logger.info("=" * 60)
        logger.info("AUTONOMOUS RUNTIME INITIALIZED")
        logger.info("=" * 60)
        logger.info(f"Mode: {self.config.mode.value.upper()}")
        logger.info(f"Symbols: {self.config.symbols}")
        logger.info(f"Timeframes: {self.config.timeframes}")
        logger.info(f"State: {self.state.health}")
        logger.info("=" * 60)
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def handle_shutdown(signum, frame):
            logger.info(f"Received signal {signum} - initiating graceful shutdown")
            self._shutdown_requested = True
        
        signal.signal(signal.SIGINT, handle_shutdown)
        signal.signal(signal.SIGTERM, handle_shutdown)
    
    async def run(self):
        """Run the autonomous runtime loop."""
        self._running = True
        
        logger.info("Starting autonomous runtime...")
        logger.info("Press Ctrl+C to shutdown gracefully")
        
        # Main loop
        while self._running and not self._shutdown_requested:
            loop_start = time.time()
            
            try:
                self.state.loop_count += 1
                self.state.last_loop_time = datetime.now()
                
                # 1. Health check
                should_continue = await self.health_monitor.check()
                if not should_continue:
                    await asyncio.sleep(1)
                    continue
                
                # 2. Fetch market data
                bars = await self.data_coordinator.fetch_latest_bars()
                
                # 3. Process bars
                for key, bar_data in bars.items():
                    if self._shutdown_requested:
                        break
                    
                    parts = key.rsplit('_', 1)
                    if len(parts) == 2:
                        symbol, timeframe = parts
                        
                        # Check if new bar
                        symbol_state = self.state.get_symbol_state(symbol)
                        bar_time = bar_data.get('timestamp')
                        if isinstance(bar_time, str):
                            bar_time = datetime.fromisoformat(bar_time)
                        
                        if self.data_coordinator.is_new_bar(
                            symbol, timeframe, bar_time, 
                            symbol_state.last_processed_bar
                        ):
                            # Process through pipeline
                            result = await self.pipeline.process_bar(
                                symbol, timeframe, bar_data
                            )
                            
                            if result:
                                # Check for execution
                                if result.get('signal') in ['long', 'short']:
                                    await self.execution.execute_signal(result)
                        
                        # Brief yield to allow other tasks
                        await asyncio.sleep(0)
                
                # 4. Heartbeat (periodic)
                if self.state.loop_count % int(
                    self.config.heartbeat_interval_sec / self.config.polling_interval_sec
                ) == 0:
                    await self.heartbeat.write()
                
                # 5. Record latency
                loop_latency_ms = (time.time() - loop_start) * 1000
                self.heartbeat.record_latency(loop_latency_ms)
                
                # 6. Sleep until next cycle
                elapsed = time.time() - loop_start
                sleep_time = max(0, self.config.polling_interval_sec - elapsed)
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Runtime loop error: {e}")
                self.health_monitor.record_error()
                self.state.consecutive_errors += 1
                await asyncio.sleep(1)
        
        # Shutdown
        await self._shutdown()
    
    async def _shutdown(self):
        """Graceful shutdown."""
        logger.info("=" * 60)
        logger.info("SHUTTING DOWN AUTONOMOUS RUNTIME")
        logger.info("=" * 60)
        
        # Save state
        self.state_persistence.save(self.state)
        
        # Final heartbeat
        await self.heartbeat.write()
        
        logger.info(f"Total loops: {self.state.loop_count}")
        logger.info(f"Bars processed: {self.state.total_bars_processed}")
        logger.info(f"Signals generated: {self.state.total_signals_generated}")
        logger.info(f"Executions: {self.state.total_executions}")
        logger.info(f"Errors: {self.state.total_errors}")
        logger.info("=" * 60)
        logger.info("SHUTDOWN COMPLETE")
        
        self._running = False


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Adaptive Trading Runtime')
    parser.add_argument('--mode', default='paper', 
                       choices=['paper', 'live', 'backtest'],
                       help='Runtime mode')
    parser.add_argument('--symbols', nargs='+',
                       default=['EURUSD', 'GBPUSD', 'USDJPY', 'XAGUSD'],
                       help='Symbols to trade')
    parser.add_argument('--interval', type=float, default=1.0,
                       help='Polling interval (seconds)')
    
    args = parser.parse_args()
    
    # Create config
    config = RuntimeConfig(
        mode=RuntimeMode(args.mode),
        symbols=args.symbols,
        polling_interval_sec=args.interval
    )
    
    # Create and run runtime
    runtime = AutonomousRuntime(config)
    
    try:
        asyncio.run(runtime.run())
    except KeyboardInterrupt:
        logger.info("Runtime interrupted by user")
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
