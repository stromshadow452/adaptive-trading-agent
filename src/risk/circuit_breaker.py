import os
import json
import time
import threading

# Use a simple file lock mechanism since fcntl is not on Windows
# For production Windows, use 'portalocker'. Here we use a basic mutex for thread safety
# and atomic file writes for process safety (mostly).
_lock = threading.Lock()

class CircuitBreaker:
    """
    Stage 9/11: Persistent Safety Gates
    """
    def __init__(self, state_file="config/circuit_breakers.json"):
        self.state_file = state_file
        if not os.path.exists(state_file):
            self._save_state({"global_trip": False, "symbols": {}})

    def _load_state(self):
        with open(self.state_file, "r") as f:
            return json.load(f)

    def _save_state(self, state):
        # Atomic write
        tmp = self.state_file + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, self.state_file)

    def check_gate(self, symbol: str):
        with _lock:
            state = self._load_state()
            
            if state["global_trip"]:
                raise RuntimeError("GLOBAL CIRCUIT BREAKER TRIPPED")
            
            sym_state = state["symbols"].get(symbol, {})
            if sym_state.get("tripped", False):
                if time.time() < sym_state["reset_time"]:
                    raise RuntimeError(f"CIRCUIT BREAKER TRIPPED FOR {symbol}")
                else:
                    # Auto-reset
                    sym_state["tripped"] = False
                    state["symbols"][symbol] = sym_state
                    self._save_state(state)
                    return True
        return True

    def trip(self, symbol=None, duration=3600, reason="Unknown"):
        with _lock:
            state = self._load_state()
            if symbol:
                state["symbols"][symbol] = {
                    "tripped": True,
                    "reset_time": time.time() + duration,
                    "reason": reason
                }
            else:
                state["global_trip"] = True
                state["global_reason"] = reason
            self._save_state(state)
