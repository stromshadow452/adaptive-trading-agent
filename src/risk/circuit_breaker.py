"""
src/risk/circuit_breaker.py
===========================
In-memory Circuit Breaker — Stage 9/11 Safety Gate.

REFACTOR NOTE (2026-03-12):
  Previous version read/wrote JSON on EVERY check_gate() call.
  This caused ~156 disk reads/minute in live 13-pair trading and
  created race conditions under multi-process load.

  This version keeps state entirely in-memory (O(1) dict lookup on
  check_gate). State is persisted to disk ONLY on trip() and reset()
  — never during the hot check path.

  Public interface is IDENTICAL to the old version:
    check_gate(symbol)      → True or raises RuntimeError
    trip(symbol, duration, reason)
    reset(symbol)

  Callers require NO changes.
"""

import os
import json
import time
import threading
import logging

logger = logging.getLogger(__name__)

_lock = threading.Lock()


class CircuitBreaker:
    """
    In-memory circuit breaker with optional disk persistence.

    State is loaded from disk once at startup, kept in-memory
    for the lifetime of the process, and written back only on
    trip() / reset() events.
    """

    def __init__(self, state_file: str = "config/circuit_breakers.json"):
        self.state_file = state_file

        # In-memory state (primary source of truth during runtime)
        self._global_trip: bool = False
        self._global_reason: str = ""
        self._symbols: dict = {}

        # Load persisted state at startup
        self._load_from_disk()

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _load_from_disk(self):
        """Load persisted breaker state at startup ONLY. Never called in hot path."""
        if not os.path.exists(self.state_file):
            return
        try:
            with open(self.state_file, "r") as f:
                state = json.load(f)
            self._global_trip   = bool(state.get("global_trip", False))
            self._global_reason = str(state.get("global_reason", ""))
            self._symbols       = dict(state.get("symbols", {}))
            logger.info(f"[CircuitBreaker] Loaded state from {self.state_file}")
        except Exception as e:
            logger.warning(f"[CircuitBreaker] Could not load state from disk: {e}. Starting fresh.")

    def _persist(self):
        """
        Write current in-memory state to disk.
        Called ONLY after trip() or reset() — NOT on every check_gate().
        Uses atomic rename to prevent partial writes.
        """
        state = {
            "global_trip":   self._global_trip,
            "global_reason": self._global_reason,
            "symbols":       self._symbols,
        }
        tmp = self.state_file + ".tmp"
        try:
            os.makedirs(os.path.dirname(self.state_file) or ".", exist_ok=True)
            with open(tmp, "w") as f:
                json.dump(state, f, indent=2)
            os.replace(tmp, self.state_file)
        except Exception as e:
            logger.error(f"[CircuitBreaker] Could not persist state: {e}")

    # ------------------------------------------------------------------ #
    # Public API (same interface as old file-based version)               #
    # ------------------------------------------------------------------ #

    def check_gate(self, symbol: str) -> bool:
        """
        Check if trading is allowed.
        Returns True if allowed. Raises RuntimeError if tripped.

        NO disk I/O in this method — pure in-memory dict lookup.
        """
        with _lock:
            # Global trip check
            if self._global_trip:
                raise RuntimeError(
                    f"GLOBAL CIRCUIT BREAKER TRIPPED: {self._global_reason}"
                )

            # Symbol-level trip check
            sym_state = self._symbols.get(symbol, {})
            if sym_state.get("tripped", False):
                if time.time() < sym_state.get("reset_time", 0):
                    raise RuntimeError(
                        f"CIRCUIT BREAKER TRIPPED FOR {symbol}: "
                        f"{sym_state.get('reason', 'no reason given')}"
                    )
                else:
                    # Duration expired → auto-reset in memory
                    self._symbols[symbol]["tripped"] = False
                    logger.info(f"[CircuitBreaker] Auto-reset expired trip for {symbol}")

        return True

    def trip(self, symbol: str = None, duration: int = 3600, reason: str = "Unknown"):
        """
        Trip the circuit breaker.

        Args:
            symbol:   Symbol to trip (None = global trip)
            duration: How long to remain tripped in seconds (default 1h)
            reason:   Human-readable reason string
        """
        with _lock:
            if symbol:
                self._symbols[symbol] = {
                    "tripped":    True,
                    "reset_time": time.time() + duration,
                    "reason":     reason,
                }
                logger.warning(
                    f"[CircuitBreaker] TRIPPED {symbol} for {duration}s: {reason}"
                )
            else:
                self._global_trip   = True
                self._global_reason = reason
                logger.critical(f"[CircuitBreaker] GLOBAL TRIP: {reason}")

        # Persist only on trip events
        self._persist()

    def reset(self, symbol: str = None):
        """
        Reset a tripped circuit breaker.

        Args:
            symbol: Symbol to reset (None = reset global)
        """
        with _lock:
            if symbol:
                if symbol in self._symbols:
                    self._symbols[symbol]["tripped"] = False
                    logger.info(f"[CircuitBreaker] Reset {symbol}")
            else:
                self._global_trip   = False
                self._global_reason = ""
                logger.info("[CircuitBreaker] Global reset")

        # Persist only on reset events
        self._persist()

    def is_tripped(self, symbol: str = None) -> bool:
        """
        Check if a specific symbol (or global) is currently tripped.
        Returns True if tripped, False otherwise. Does NOT raise.
        """
        with _lock:
            if self._global_trip:
                return True
            if symbol:
                sym_state = self._symbols.get(symbol, {})
                if sym_state.get("tripped", False):
                    return time.time() < sym_state.get("reset_time", 0)
        return False

    def get_status(self) -> dict:
        """Return full breaker status (for monitoring/API)."""
        with _lock:
            return {
                "global_trip":   self._global_trip,
                "global_reason": self._global_reason,
                "symbols":       dict(self._symbols),
            }
