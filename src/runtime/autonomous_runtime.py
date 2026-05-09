"""
SCOPUS Autonomous Runtime
Continuous orchestration runtime for adaptive trading.
"""

import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Dict, List, Optional, Any
from enum import Enum

import pandas as pd
import psutil

from src.market_data.yfinance_adapter import YFinanceAdapter
from src.pipeline.pipeline_v2 import PipelineV2, PipelineConfig


# ============================================================
# NULL EXECUTOR STUB  (FIX #3)
# Satisfies pipeline_v2.process_bar(executor=...) interface
# without requiring a live broker connection.
# ============================================================

class NullExecutor:
    """No-op executor used when no broker is connected.
    Satisfies the interface expected by PipelineV2.process_bar().
    """

    def check_sl_tp(self, symbol, high, low, bar_time=None):
        """No-op: no open positions to check."""
        return

    def get_open_positions(self):
        """Return empty set — no positions open."""
        return set()

    def execute(self, signal: dict):
        """Log the signal but do not route to a broker."""
        logger.info(
            f"[NullExecutor] SIGNAL CAPTURED | "
            f"symbol={signal.get('symbol')} | "
            f"side={signal.get('side')} | "
            f"confidence={signal.get('confidence', 0.0):.4f} | "
            f"strategy={signal.get('strategy')} | "
            f"price={signal.get('price')}"
        )
        return None


# ============================================================
# DIRECTORIES
# ============================================================

Path("logs").mkdir(exist_ok=True)
Path("logs/runtime").mkdir(parents=True, exist_ok=True)
Path("state/runtime").mkdir(parents=True, exist_ok=True)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            f"logs/runtime/runtime_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
    ]
)

logger = logging.getLogger(__name__)


# ============================================================
# ENUMS
# ============================================================

class RuntimeMode(Enum):
    PAPER = "paper"
    LIVE = "live"


class RuntimeHealth(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    PAUSED = "paused"


# ============================================================
# CONFIG
# ============================================================

@dataclass
class RuntimeConfig:

    symbols: List[str] = field(default_factory=lambda: [
        "EURUSD",
        "GBPUSD",
        "USDJPY",
        "XAGUSD"
    ])

    timeframes: List[str] = field(default_factory=lambda: [
        "1H",
        "4H",
        "D1"
    ])

    polling_interval_sec: float = 5.0

    mode: RuntimeMode = RuntimeMode.PAPER

    log_dir: str = "logs/runtime"
    state_dir: str = "state/runtime"

    enable_circuit_breaker: bool = True
    circuit_breaker_threshold: int = 5

    def __post_init__(self):
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        Path(self.state_dir).mkdir(parents=True, exist_ok=True)


# ============================================================
# STATE
# ============================================================

@dataclass
class SymbolState:
    symbol: str
    processed_count: int = 0
    error_count: int = 0
    consecutive_errors: int = 0
    latency_ms: float = 0.0
    last_processed_bar: Optional[datetime] = None


@dataclass
class RuntimeState:

    start_time: datetime = field(default_factory=datetime.now)

    loop_count: int = 0
    total_bars_processed: int = 0
    total_signals_generated: int = 0
    total_executions: int = 0
    total_errors: int = 0

    health: str = RuntimeHealth.HEALTHY.value

    symbol_states: Dict[str, SymbolState] = field(default_factory=dict)

    def get_symbol_state(self, symbol: str) -> SymbolState:

        if symbol not in self.symbol_states:
            self.symbol_states[symbol] = SymbolState(symbol=symbol)

        return self.symbol_states[symbol]


# ============================================================
# MARKET DATA
# ============================================================

class MarketDataCoordinator:

    def __init__(self, config: RuntimeConfig):

        self.config = config
        self.adapter = YFinanceAdapter()

        logger.info("MarketDataCoordinator initialized")
        logger.info(f"Symbols: {config.symbols}")
        logger.info(f"Timeframes: {config.timeframes}")

    async def fetch_latest_bars(self):

        bars = {}

        for symbol in self.config.symbols:
            for timeframe in self.config.timeframes:

                try:
                    bar = await self._fetch_bar(symbol, timeframe)

                    if bar:
                        bars[f"{symbol}_{timeframe}"] = bar

                except Exception as e:
                    logger.error(
                        f"Failed to fetch {symbol} {timeframe}: {e}"
                    )

        return bars

    async def _fetch_bar(self, symbol: str, timeframe: str):

        try:
            # FIX #1: Fetch full history (250 bars) so feature pipeline has
            # enough data.  Pipeline requires min_bars=150 before computing
            # features, so a single-row DataFrame always returns None.
            df = self.adapter.get_latest_candles(
                symbol=symbol,
                timeframe=timeframe,
                bars=250
            )

            if df.empty:
                return None

            latest = df.iloc[-1]

            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "timestamp": df.index[-1].to_pydatetime(),
                "open": float(latest["Open"]),
                "high": float(latest["High"]),
                "low": float(latest["Low"]),
                "close": float(latest["Close"]),
                "volume": float(latest["Volume"]),
                # Carry the full OHLCV history for the feature pipeline
                "_raw_df": df,
            }

        except Exception as e:
            logger.error(
                f"YFinance fetch failed {symbol} {timeframe}: {e}"
            )
            return None


# ============================================================
# PIPELINE
# ============================================================

class PipelineCoordinator:

    def __init__(self, config: RuntimeConfig, state: RuntimeState):

        self.config = config
        self.state = state

        pipeline_cfg = PipelineConfig()
        self.pipeline = PipelineV2(pipeline_cfg)
        self._executor = NullExecutor()   # FIX #3: reusable no-op executor

        logger.info("PipelineCoordinator initialized")

    async def process_bar(
        self,
        symbol: str,
        timeframe: str,
        bar_data: Dict[str, Any]
    ):

        try:
            symbol_state = self.state.get_symbol_state(symbol)

            bar_time = bar_data["timestamp"]

            if symbol_state.last_processed_bar:

                current_bar = bar_time
                last_bar = symbol_state.last_processed_bar

                if current_bar.tzinfo is not None:
                    current_bar = current_bar.replace(tzinfo=None)

                if last_bar.tzinfo is not None:
                    last_bar = last_bar.replace(tzinfo=None)

                if current_bar <= last_bar:
                    logger.debug(
                        f"Skipping duplicate bar {symbol} {timeframe}"
                    )
                    return None

            # FIX #1: Use the full history DataFrame carried by bar_data.
            # The feature pipeline requires min_bars=150; a single-row
            # DataFrame always returns None from build_full_feature_set().
            raw_df: Optional[pd.DataFrame] = bar_data.get("_raw_df")
            if raw_df is None or raw_df.empty:
                logger.warning(
                    f"[PipelineCoordinator] No history for {symbol} "
                    f"{timeframe} — skipping bar"
                )
                return None

            logger.debug(
                f"[DEBUG] {symbol} {timeframe} | raw_df rows={len(raw_df)} "
                f"cols={list(raw_df.columns)}"
            )

            # FIX #2 / #5: Wrap bar_data dict as an object with attribute
            # access so pipeline_v2 can use bar.high, bar.low, bar.close
            # without AttributeError.
            bar_obj = SimpleNamespace(
                open=bar_data["open"],
                high=bar_data["high"],
                low=bar_data["low"],
                close=bar_data["close"],
                volume=bar_data["volume"],
                time=bar_data["timestamp"],
                timestamp=bar_data["timestamp"],
            )

            result = self.pipeline.process_bar(
                symbol=symbol,
                bar=bar_obj,
                raw_df=raw_df,
                executor=self._executor   # FIX #3: pass real NullExecutor
            )

            # ── DEBUG: log full pipeline internals ───────────────────────
            logger.info(
                f"[DEBUG] {symbol} {timeframe} | "
                f"feature_count={result.feature_count} | "
                f"regime={result.regime} | "
                f"regime_confidence={result.confidence:.4f} | "
                f"errors={result.errors or 'none'}"
            )
            if result.signal:
                logger.info(
                    f"[DEBUG] {symbol} SIGNAL | "
                    f"side={result.signal.get('side')} | "
                    f"strategy={result.signal.get('strategy')} | "
                    f"signal_confidence={result.signal.get('confidence', 0.0):.4f} | "
                    f"price={result.signal.get('price')} | "
                    f"sl={result.signal.get('sl')} | "
                    f"tp={result.signal.get('tp')}"
                )
            if result.ensemble_decision:
                ed = result.ensemble_decision
                logger.info(
                    f"[DEBUG] {symbol} ENSEMBLE | "
                    f"should_trade={ed.should_trade} | "
                    f"side={ed.side} | "
                    f"score={ed.ensemble_score:.4f} | "
                    f"corr_guard={ed.correlation_guard}"
                )

            # FIX #4: BarResult has no 'decision' attribute.
            # Map signal presence → BUY/SELL/HOLD correctly.
            sig = result.signal
            if sig is not None and sig.get("side") in ("buy", "sell"):
                decision = sig["side"].upper()
                confidence = float(sig.get("confidence", result.confidence))
            else:
                decision = "HOLD"
                confidence = float(result.confidence)

            logger.info(
                f"PIPELINE | {symbol} | "
                f"decision={decision} | "
                f"confidence={confidence:.4f}"
            )

            symbol_state.last_processed_bar = bar_time
            symbol_state.processed_count += 1

            self.state.total_bars_processed += 1

            if decision != "HOLD":
                self.state.total_signals_generated += 1

            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "decision": decision,
                "confidence": confidence,
                "timestamp": bar_time,
                "signal": sig,
            }

        except Exception as e:
            logger.exception(
                f"Pipeline error {symbol} {timeframe}: {e}"
            )

            self.state.total_errors += 1
            return None


# ============================================================
# EXECUTION
# ============================================================

class ExecutionCoordinator:

    def __init__(self, config: RuntimeConfig, state: RuntimeState):

        self.config = config
        self.state = state

        logger.info("ExecutionCoordinator initialized")

    async def execute_signal(self, signal: Dict[str, Any]):

        try:
            if not signal:
                return False

            decision = signal.get("decision", "HOLD")

            if decision == "HOLD":
                return False

            logger.info(
                f"EXECUTION | "
                f"symbol={signal['symbol']} | "
                f"decision={decision}"
            )

            self.state.total_executions += 1

            return True

        except Exception as e:
            logger.error(f"Execution error: {e}")
            return False


# ============================================================
# HEARTBEAT
# ============================================================

class RuntimeHeartbeat:

    def __init__(self, config: RuntimeConfig, state: RuntimeState):

        self.config = config
        self.state = state

        self.path = Path(config.log_dir) / "runtime_health.json"

        logger.info(f"RuntimeHeartbeat initialized: {self.path}")

    async def write(self):

        uptime_sec = (
            datetime.now() - self.state.start_time
        ).total_seconds()

        heartbeat = {
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": uptime_sec,
            "loop_count": self.state.loop_count,
            "bars_processed": self.state.total_bars_processed,
            "signals_generated": self.state.total_signals_generated,
            "executions": self.state.total_executions,
            "errors": self.state.total_errors,
            "memory_mb": psutil.Process().memory_info().rss / 1024 / 1024,
            "health": self.state.health
        }

        with open(self.path, "w") as f:
            json.dump(heartbeat, f, indent=2)


# ============================================================
# RUNTIME
# ============================================================

class AutonomousRuntime:

    def __init__(self, config: RuntimeConfig):

        self.config = config
        self.state = RuntimeState()

        self.market_data = MarketDataCoordinator(config)
        self.pipeline = PipelineCoordinator(config, self.state)
        self.execution = ExecutionCoordinator(config, self.state)
        self.heartbeat = RuntimeHeartbeat(config, self.state)

        self.running = True

        logger.info("=" * 60)
        logger.info("AUTONOMOUS RUNTIME INITIALIZED")
        logger.info("=" * 60)
        logger.info(f"Mode: {config.mode.value}")
        logger.info(f"Symbols: {config.symbols}")
        logger.info(f"Timeframes: {config.timeframes}")

    async def run(self):

        logger.info("Starting autonomous runtime...")

        while self.running:

            loop_start = time.time()

            try:
                bars = await self.market_data.fetch_latest_bars()

                for _, bar in bars.items():

                    result = await self.pipeline.process_bar(
                        symbol=bar["symbol"],
                        timeframe=bar["timeframe"],
                        bar_data=bar
                    )

                    if result:
                        await self.execution.execute_signal(result)

                self.state.loop_count += 1

                await self.heartbeat.write()

            except Exception as e:
                logger.error(f"Runtime loop error: {e}")
                self.state.total_errors += 1

            latency_ms = (time.time() - loop_start) * 1000

            logger.info(
                f"LOOP={self.state.loop_count} | "
                f"latency={latency_ms:.2f}ms | "
                f"bars={self.state.total_bars_processed} | "
                f"signals={self.state.total_signals_generated} | "
                f"executions={self.state.total_executions}"
            )

            await asyncio.sleep(self.config.polling_interval_sec)

    async def shutdown(self):

        logger.info("=" * 60)
        logger.info("SHUTTING DOWN AUTONOMOUS RUNTIME")
        logger.info("=" * 60)

        self.running = False

        logger.info(f"Total loops: {self.state.loop_count}")
        logger.info(f"Bars processed: {self.state.total_bars_processed}")
        logger.info(f"Signals generated: {self.state.total_signals_generated}")
        logger.info(f"Executions: {self.state.total_executions}")
        logger.info(f"Errors: {self.state.total_errors}")


# ============================================================
# ENTRYPOINT
# ============================================================

async def async_main():

    config = RuntimeConfig()

    runtime = AutonomousRuntime(config)

    try:
        await runtime.run()

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")

    finally:
        await runtime.shutdown()


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()

