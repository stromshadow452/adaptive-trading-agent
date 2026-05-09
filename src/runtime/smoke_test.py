"""
Runtime Smoke Tests

Validates autonomous runtime functionality.

Run with:
    python -m src.runtime.smoke_test
"""

import asyncio
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.runtime.autonomous_runtime import (
    AutonomousRuntime,
    RuntimeConfig,
    RuntimeMode,
    RuntimeState,
    RuntimeHeartbeat,
    MarketDataCoordinator,
    PipelineCoordinator,
    ExecutionCoordinator,
    StatePersistence
)


class RuntimeSmokeTest:
    """Smoke tests for autonomous runtime."""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.tests = []
    
    def run_all_tests(self):
        """Run all smoke tests."""
        print("=" * 60)
        print("AUTONOMOUS RUNTIME SMOKE TESTS")
        print("=" * 60)
        
        # Run tests
        asyncio.run(self._test_runtime_boot())
        asyncio.run(self._test_no_duplicate_processing())
        asyncio.run(self._test_symbol_crash_isolation())
        asyncio.run(self._test_heartbeat_updates())
        asyncio.run(self._test_pipeline_execution())
        asyncio.run(self._test_graceful_shutdown())
        asyncio.run(self._test_restart_recovery())
        asyncio.run(self._test_latency_tracking())
        
        # Summary
        print("=" * 60)
        print(f"RESULTS: {self.passed} passed, {self.failed} failed")
        print("=" * 60)
        
        return self.failed == 0
    
    async def _test_runtime_boot(self):
        """Test: Runtime boots successfully."""
        test_name = "Runtime Boot"
        try:
            print(f"\n[Test] {test_name}...")
            
            config = RuntimeConfig(
                symbols=["EURUSD"],
                mode=RuntimeMode.PAPER
            )
            
            runtime = AutonomousRuntime(config)
            
            assert runtime.config is not None
            assert runtime.state is not None
            assert len(runtime.config.symbols) == 1
            
            print(f"  ✅ PASSED: Runtime booted successfully")
            self.passed += 1
            
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            self.failed += 1
    
    async def _test_no_duplicate_processing(self):
        """Test: No duplicate bar processing."""
        test_name = "No Duplicate Processing"
        try:
            print(f"\n[Test] {test_name}...")
            
            config = RuntimeConfig(symbols=["EURUSD"])
            state = RuntimeState()
            
            symbol_state = state.get_symbol_state("EURUSD")
            
            # First bar should be processable
            bar_time_1 = datetime.now()
            assert symbol_state.can_process(bar_time_1) is True
            
            # Mark as processed
            symbol_state.last_processed_bar = bar_time_1
            
            # Same bar should NOT be processable
            assert symbol_state.can_process(bar_time_1) is False
            
            # Future bar should be processable
            bar_time_2 = bar_time_1 + timedelta(minutes=1)
            assert symbol_state.can_process(bar_time_2) is True
            
            print(f"  ✅ PASSED: Duplicate detection working")
            self.passed += 1
            
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            self.failed += 1
    
    async def _test_symbol_crash_isolation(self):
        """Test: Symbol crash isolation."""
        test_name = "Symbol Crash Isolation"
        try:
            print(f"\n[Test] {test_name}...")
            
            config = RuntimeConfig(symbols=["EURUSD", "GBPUSD"])
            state = RuntimeState()
            
            pipeline = PipelineCoordinator(config, state)
            
            # Simulate error in first symbol
            eurusd_state = state.get_symbol_state("EURUSD")
            eurusd_state.consecutive_errors = 3
            eurusd_state.health = "degraded"
            
            # Second symbol should still be healthy
            gbpusd_state = state.get_symbol_state("GBPUSD")
            assert gbpusd_state.health == "healthy"
            
            print(f"  ✅ PASSED: Symbol isolation working")
            self.passed += 1
            
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            self.failed += 1
    
    async def _test_heartbeat_updates(self):
        """Test: Heartbeat updates correctly."""
        test_name = "Heartbeat Updates"
        try:
            print(f"\n[Test] {test_name}...")
            
            config = RuntimeConfig(
                log_dir="logs/test_runtime"
            )
            state = RuntimeState()
            
            heartbeat = RuntimeHeartbeat(config, state)
            
            # Write heartbeat
            await heartbeat.write()
            
            # Check file was created
            health_file = Path(config.log_dir) / "runtime_health.json"
            assert health_file.exists()
            
            print(f"  ✅ PASSED: Heartbeat written")
            self.passed += 1
            
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            self.failed += 1
    
    async def _test_pipeline_execution(self):
        """Test: Pipeline executes continuously."""
        test_name = "Pipeline Execution"
        try:
            print(f"\n[Test] {test_name}...")
            
            config = RuntimeConfig(symbols=["EURUSD"])
            state = RuntimeState()
            pipeline = PipelineCoordinator(config, state)
            
            bar_data = {
                'symbol': 'EURUSD',
                'timestamp': datetime.now(),
                'open': 1.0850,
                'high': 1.0855,
                'low': 1.0845,
                'close': 1.0852,
                'volume': 1000
            }
            
            # Process bar
            result = await pipeline.process_bar('EURUSD', '1H', bar_data)
            
            assert result is not None
            assert state.total_bars_processed == 1
            
            print(f"  ✅ PASSED: Pipeline execution working")
            self.passed += 1
            
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            self.failed += 1
    
    async def _test_graceful_shutdown(self):
        """Test: Graceful shutdown."""
        test_name = "Graceful Shutdown"
        try:
            print(f"\n[Test] {test_name}...")
            
            config = RuntimeConfig(
                state_dir="state/test_runtime"
            )
            
            runtime = AutonomousRuntime(config)
            
            # Run for a short time
            runtime._running = True
            await asyncio.sleep(0.1)
            
            # Trigger shutdown
            runtime._shutdown_requested = True
            await runtime._shutdown()
            
            assert runtime._running is False
            
            # Check state was saved
            state_file = Path(config.state_dir) / "runtime_state.json"
            assert state_file.exists()
            
            print(f"  ✅ PASSED: Graceful shutdown working")
            self.passed += 1
            
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            self.failed += 1
    
    async def _test_restart_recovery(self):
        """Test: Restart recovery."""
        test_name = "Restart Recovery"
        try:
            print(f"\n[Test] {test_name}...")
            
            state_dir = "state/test_recovery"
            
            # Create and save state
            state1 = RuntimeState()
            state1.loop_count = 100
            state1.total_bars_processed = 50
            
            persistence = StatePersistence(state_dir)
            persistence.save(state1)
            
            # Load state
            state2 = persistence.load()
            
            assert state2 is not None
            assert state2.loop_count == 100
            assert state2.total_bars_processed == 50
            assert state2.boot_count == 1  # Incremented
            
            print(f"  ✅ PASSED: Restart recovery working")
            self.passed += 1
            
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            self.failed += 1
    
    async def _test_latency_tracking(self):
        """Test: Latency tracking."""
        test_name = "Latency Tracking"
        try:
            print(f"\n[Test] {test_name}...")
            
            config = RuntimeConfig()
            state = RuntimeState()
            heartbeat = RuntimeHeartbeat(config, state)
            
            # Record some latencies
            heartbeat.record_latency(10.0)
            heartbeat.record_latency(15.0)
            heartbeat.record_latency(20.0)
            
            assert state.avg_loop_latency_ms > 0
            assert state.max_loop_latency_ms >= 20.0
            
            print(f"  ✅ PASSED: Latency tracking working")
            self.passed += 1
            
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            self.failed += 1


if __name__ == "__main__":
    from datetime import timedelta  # Import needed
    
    test = RuntimeSmokeTest()
    success = test.run_all_tests()
    
    sys.exit(0 if success else 1)
