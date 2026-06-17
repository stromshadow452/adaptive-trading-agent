"""
src/market_data/mt5_source.py
==============================
MT5 Live Data Source — Week 6 Stub (ready-to-wire implementation).

Implements DataSourceABC using the MetaTrader5 Python API.
The pipeline sees zero difference vs HistoricalCSVSource.

Prerequisites:
    pip install MetaTrader5
    MT5 terminal must be running on the machine.

Usage:
    from src.market_data.mt5_source import MT5LiveSource
    source = MT5LiveSource(login=12345, password="...", server="ICMarketsSC-Demo")
    bar = source.get_latest_bar("EURUSD", Timeframe.H1)

Switching from backtest to shadow:
    # BEFORE:
    data_source = HistoricalCSVSource("data/")
    # AFTER (one line change):
    data_source = MT5LiveSource(login=..., password=..., server=...)
"""
from __future__ import annotations

import time
import logging
from datetime import datetime, timezone
from typing import Dict, Iterator, Optional

from src.market_data.types import (
    DataSource, DataSourceABC, OHLCVBar, OHLCVFrame, Timeframe
)
import pandas as pd

logger = logging.getLogger(__name__)

# MT5 timeframe mapping
_TF_MAP = {
    Timeframe.M1:  1,      # mt5.TIMEFRAME_M1
    Timeframe.M5:  5,      # mt5.TIMEFRAME_M5
    Timeframe.M15: 15,
    Timeframe.M30: 30,
    Timeframe.H1:  16385,  # mt5.TIMEFRAME_H1
    Timeframe.H4:  16388,  # mt5.TIMEFRAME_H4
    Timeframe.D1:  16408,  # mt5.TIMEFRAME_D1
}

_TF_SECONDS = {
    Timeframe.M1: 60, Timeframe.M5: 300, Timeframe.M15: 900,
    Timeframe.M30: 1800, Timeframe.H1: 3600, Timeframe.H4: 14400,
    Timeframe.D1: 86400,
}


class MT5LiveSource(DataSourceABC):
    """
    Live MT5 data source. Implements DataSourceABC so the pipeline
    needs zero changes to switch from historical CSV to live data.

    If MetaTrader5 is not installed, instantiation raises ImportError
    with a clear message.
    """

    def __init__(
        self,
        login:    int,
        password: str,
        server:   str,
        path:     Optional[str] = None,   # MT5 terminal executable path
    ):
        try:
            import MetaTrader5 as mt5
            self._mt5 = mt5
        except ImportError:
            raise ImportError(
                "MetaTrader5 Python package not installed. "
                "Run: pip install MetaTrader5\n"
                "The MT5 terminal must also be running on this machine."
            )

        ok = self._mt5.initialize(
            login=login,
            password=password,
            server=server,
            **({"path": path} if path else {}),
        )
        if not ok:
            err = self._mt5.last_error()
            raise ConnectionError(f"MT5 initialize failed: {err}")

        acc = self._mt5.account_info()
        logger.info(
            f"[MT5LiveSource] Connected: server={server} "
            f"account={acc.login if acc else 'unknown'} "
            f"balance={acc.balance if acc else 'unknown'}"
        )
        self._last_ts:  Dict[str, datetime] = {}
        self._login    = login
        self._server   = server

    @property
    def source_type(self) -> DataSource:
        return DataSource.MT5

    # ------------------------------------------------------------------ #
    # DataSourceABC interface                                              #
    # ------------------------------------------------------------------ #

    def get_latest_bar(self, symbol: str, timeframe: Timeframe) -> Optional[OHLCVBar]:
        """Return the most recent completed bar from MT5."""
        mt5_tf = _TF_MAP.get(timeframe)
        if mt5_tf is None:
            logger.warning(f"[MT5LiveSource] Unsupported timeframe: {timeframe}")
            return None
        try:
            rates = self._mt5.copy_rates_from_pos(symbol, mt5_tf, 1, 1)
            if rates is None or len(rates) == 0:
                return None
            r = rates[0]
            return OHLCVBar(
                timestamp=datetime.fromtimestamp(int(r["time"]), tz=timezone.utc),
                symbol=symbol,
                timeframe=timeframe,
                open=float(r["open"]),
                high=float(r["high"]),
                low=float(r["low"]),
                close=float(r["close"]),
                volume=float(r["tick_volume"]),
                source=DataSource.MT5,
            )
        except Exception as e:
            logger.error(f"[MT5LiveSource] get_latest_bar {symbol}: {e}")
            return None

    def get_history(self, symbol: str, timeframe: Timeframe, n_bars: int) -> OHLCVFrame:
        """Return the last n_bars as a DataFrame (ascending timestamp)."""
        mt5_tf = _TF_MAP.get(timeframe)
        if mt5_tf is None:
            return pd.DataFrame()
        try:
            rates = self._mt5.copy_rates_from_pos(symbol, mt5_tf, 0, n_bars)
            if rates is None or len(rates) == 0:
                return pd.DataFrame()
            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
            df = df.rename(columns={
                "time": "timestamp", "tick_volume": "volume",
                "open": "open", "high": "high", "low": "low", "close": "close",
            })
            df = df[["timestamp", "open", "high", "low", "close", "volume"]]
            df = df.sort_values("timestamp").reset_index(drop=True)
            return df
        except Exception as e:
            logger.error(f"[MT5LiveSource] get_history {symbol}: {e}")
            return pd.DataFrame()

    def stream(self, symbol: str, timeframe: Timeframe) -> Iterator[OHLCVBar]:
        """
        Block until each bar closes, then yield.
        Polls every 1/4 of the timeframe period.
        Safe for threading — uses MT5 copy_rates_from_pos.
        """
        poll_secs = max(5.0, _TF_SECONDS.get(timeframe, 3600) / 4)
        logger.info(
            f"[MT5LiveSource] Streaming {symbol} {timeframe.value} "
            f"(poll every {poll_secs:.0f}s)"
        )
        while True:
            bar = self.get_latest_bar(symbol, timeframe)
            if bar is not None:
                last = self._last_ts.get(f"{symbol}_{timeframe.value}")
                if last is None or bar.timestamp > last:
                    self._last_ts[f"{symbol}_{timeframe.value}"] = bar.timestamp
                    yield bar
            time.sleep(poll_secs)

    # ------------------------------------------------------------------ #
    # Connection management                                                #
    # ------------------------------------------------------------------ #

    def reconnect(self):
        """Attempt to reconnect if connection drops."""
        logger.warning("[MT5LiveSource] Attempting reconnect...")
        try:
            self._mt5.shutdown()
        except Exception:
            pass
        ok = self._mt5.initialize(login=self._login, server=self._server)
        if ok:
            logger.info("[MT5LiveSource] Reconnected successfully")
        else:
            raise ConnectionError(f"MT5 reconnect failed: {self._mt5.last_error()}")

    def shutdown(self):
        """Cleanly shut down MT5 connection."""
        try:
            self._mt5.shutdown()
            logger.info("[MT5LiveSource] MT5 connection closed")
        except Exception as e:
            logger.warning(f"[MT5LiveSource] Shutdown error: {e}")

    def get_spread_pips(self, symbol: str) -> Optional[float]:
        """Return current spread in pips (useful for slippage model calibration)."""
        try:
            info = self._mt5.symbol_info(symbol)
            if info is None:
                return None
            # spread is in points; points_per_pip depends on symbol digits
            pip_points = 10 if info.digits in (5, 3) else 1
            return float(info.spread) / pip_points
        except Exception:
            return None
