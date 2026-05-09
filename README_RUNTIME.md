# AUTONOMOUS RUNTIME README

## Overview

The **Autonomous Runtime** is the central nervous system of the Adaptive Trading OS.

This is the PRIMARY runtime entrypoint that continuously orchestrates:
- Market data processing
- Pipeline execution
- Meta-gating
- Risk controls
- Trade execution
- Telemetry and monitoring

## Usage

### Basic Usage

```bash
# Run in paper mode (default)
python -m src.runtime.autonomous_runtime

# Run with specific symbols
python -m src.runtime.autonomous_runtime --symbols EURUSD GBPUSD USDJPY

# Run in live mode (NOT for initial testing)
python -m src.runtime.autonomous_runtime --mode live

# Adjust polling interval
python -m src.runtime.autonomous_runtime --interval 5.0
```

### Programmatic Usage

```python
from src.runtime import AutonomousRuntime, RuntimeConfig, RuntimeMode

# Create configuration
config = RuntimeConfig(
    symbols=["EURUSD", "GBPUSD", "USDJPY", "XAGUSD"],
    mode=RuntimeMode.PAPER,
    polling_interval_sec=1.0
)

# Create and run runtime
runtime = AutonomousRuntime(config)
await runtime.run()
```

## Architecture

### Core Components

```
AutonomousRuntime
│
├── MarketDataCoordinator
│   └── Fetches and caches market data
│
├── PipelineCoordinator
│   └── Executes pipeline_v2.process_bar()
│
├── ExecutionCoordinator
│   └── Routes signals to executor_v2
│
├── RuntimeHeartbeat
│   └── Writes periodic health updates
│
├── DriftAndHealthMonitor
│   └── Monitors health and activates circuit breaker
│
└── StatePersistence
    └── Saves/recovers runtime state
```

### Main Loop

```
1. Fetch market data
2. Detect newly completed bars
3. Run pipeline_v2.process_bar()
4. Apply meta gating
5. Apply portfolio/risk controls
6. Route approved signals to executor_v2
7. Write telemetry/logs
8. Update readiness metrics
9. Monitor runtime health
10. Sleep/repeat
```

## Configuration

### RuntimeConfig Options

```python
@dataclass
class RuntimeConfig:
    # Symbols and timeframes
    symbols: List[str] = ["EURUSD", "GBPUSD", "USDJPY", "XAGUSD"]
    timeframes: List[str] = ["1H", "4H", "D1"]
    
    # Polling
    polling_interval_sec: float = 1.0
    bar_check_interval_sec: float = 5.0
    
    # Mode
    mode: RuntimeMode = RuntimeMode.PAPER  # PAPER, LIVE, BACKTEST
    
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
    execution_cooldown_sec: float = 300.0
```

## State Management

### RuntimeState

The runtime maintains state including:
- Loop count
- Processed bars
- Symbol states
- Error tracking
- Health metrics
- Latency statistics

### State Persistence

State is automatically saved to `state/runtime/runtime_state.json`:
- On graceful shutdown
- Periodically during operation

On restart, state is restored to:
- Continue from where left off
- Avoid duplicate processing
- Maintain statistics

## Key Features

### 1. No Duplicate Processing

```python
# Tracks last processed bar per symbol
symbol_state.last_processed_bar = bar_time

# Checks before processing
if symbol_state.can_process(bar_time):
    process_bar()
```

### 2. Fault Isolation

If one symbol fails, the runtime continues processing other symbols.

```python
for symbol in symbols:
    try:
        await process_symbol(symbol)
    except Exception:
        # Log error, continue to next symbol
        log_error()
```

### 3. Safe Continuous Loop

```python
while running:
    try:
        await process_cycle()
    except Exception:
        # Log error, continue
        log_error()
        await sleep(1)
```

### 4. Circuit Breaker

Automatically pauses runtime if too many errors:

```python
if error_count >= threshold:
    activate_circuit_breaker(duration=300)
    pause_runtime()
```

### 5. Telemetry

Continuously writes:
- `logs/runtime/runtime_health.json` - Health status
- `logs/telemetry/fills.jsonl` - Execution fills
- `logs/runtime/runtime_state.json` - State snapshot

### 6. Graceful Shutdown

Handles:
- KeyboardInterrupt (Ctrl+C)
- SIGTERM

Saves state before exit.

### 7. Restart Recovery

On boot:
- Loads previous state
- Continues from last processed bar
- Avoids duplicate executions

### 8. Multi-Symbol Support

Initial symbols:
- EURUSD
- GBPUSD
- USDJPY
- XAGUSD

Architecture supports easy expansion.

## Monitoring

### Runtime Health JSON

```json
{
  "timestamp": "2026-01-15T10:30:00",
  "uptime_seconds": 86400,
  "health": "healthy",
  "mode": "paper",
  "metrics": {
    "loops_completed": 86400,
    "bars_processed": 3456,
    "signals_generated": 120,
    "executions": 45,
    "errors": 2
  },
  "memory_mb": 256.5,
  "latency_ms": {
    "avg": 25.3,
    "max": 120.5,
    "p95": 45.2
  },
  "symbol_health": {
    "EURUSD": {
      "processed": 864,
      "errors": 0,
      "health": "healthy"
    }
  }
}
```

### Logs

- `logs/runtime_YYYYMMDD_HHMMSS.log` - Runtime logs
- `logs/runtime/runtime_health.json` - Health snapshots
- `logs/telemetry/fills.jsonl` - Execution records

## Testing

### Smoke Tests

```bash
python -m src.runtime.smoke_test
```

Validates:
- Runtime boot
- No duplicate processing
- Symbol crash isolation
- Heartbeat updates
- Pipeline execution
- Graceful shutdown
- Restart recovery
- Latency tracking

### Integration Tests

```bash
# Test with mock data
python -m src.runtime.autonomous_runtime --mode backtest

# Test paper mode (no real orders)
python -m src.runtime.autonomous_runtime --mode paper
```

## File Structure

```
src/runtime/
├── __init__.py              # Package init
├── autonomous_runtime.py    # Main runtime
└── smoke_test.py           # Smoke tests

logs/runtime/
├── runtime_health.json     # Health snapshots
└── runtime_*.log           # Runtime logs

state/runtime/
└── runtime_state.json      # Persisted state

logs/telemetry/
└── fills.jsonl             # Execution records
```

## Integration Points

### Pipeline V2

```python
result = await pipeline.process_bar(symbol, timeframe, bar_data)
```

### Meta Gating

Applied after pipeline, before execution.

### Executor V2

```python
await execution.execute_signal(signal)
```

### Risk Controls

- Position limits
- Portfolio heat
- Circuit breaker

## Safety Features

### Circuit Breaker

- Activates on error threshold
- Pauses runtime temporarily
- Prevents cascading failures

### Symbol Isolation

- Errors in one symbol don't affect others
- Independent error tracking
- Per-symbol health status

### Duplicate Prevention

- Tracks last processed bar
- Never processes same bar twice
- Prevents duplicate executions

### Graceful Degradation

- Continues on non-critical errors
- Logs and alerts on issues
- Maintains operation

## Performance Targets

| Metric | Target |
|--------|--------|
| Polling Interval | 1 second |
| Pipeline Latency | < 50ms |
| Execution Latency | < 100ms |
| Heartbeat Interval | 30 seconds |
| Uptime | > 99% |
| Memory Usage | < 500MB |

## Troubleshooting

### Runtime Won't Start

1. Check state directory permissions
2. Verify log directory exists
3. Check symbol configuration

### High Error Rate

1. Check data source connectivity
2. Review error logs
3. Verify pipeline configuration
4. Check circuit breaker status

### Memory Issues

1. Monitor memory usage in health JSON
2. Reduce symbol count if needed
3. Check for memory leaks in logs

### Duplicate Processing

1. Check state persistence
2. Verify restart recovery
3. Review symbol state tracking

## Development

### Adding New Symbols

```python
config = RuntimeConfig(
    symbols=["EURUSD", "GBPUSD", "USDJPY", "XAGUSD", "AUDUSD"]
)
```

### Custom Configuration

```python
config = RuntimeConfig(
    polling_interval_sec=5.0,
    max_errors_per_symbol=20,
    circuit_breaker_threshold=10
)
```

### Extending Runtime

Extend `AutonomousRuntime` class:

```python
class CustomRuntime(AutonomousRuntime):
    async def custom_pre_processing(self):
        # Add custom logic
        pass
```

## Production Deployment

### Pre-Flight Checklist

- [ ] Paper mode validated
- [ ] Smoke tests passing
- [ ] Configuration reviewed
- [ ] Monitoring enabled
- [ ] Circuit breaker configured
- [ ] State persistence verified
- [ ] Graceful shutdown tested
- [ ] Restart recovery validated

### Deployment Steps

1. Deploy in paper mode
2. Monitor for 24 hours
3. Review health metrics
4. Switch to live mode (if approved)

### Monitoring

- Watch `runtime_health.json`
- Monitor error rates
- Check latency metrics
- Review fills log
- Alert on circuit breaker

## License

Part of Adaptive Trading OS.
