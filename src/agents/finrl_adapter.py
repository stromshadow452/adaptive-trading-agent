# src/agents/finrl_adapter.py
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np

class FinRLAdapter:
    """
    Thin wrapper to load SB3 policies trained via FinRL.
    Works in-process if stable_baselines3 is importable; otherwise becomes a no-op.
    """

    def __init__(self, policies_dir: str | os.PathLike, algo: str = "PPO"):
        self.policies_dir = Path(policies_dir)
        self.algo = algo.upper()
        self.sb3 = None
        self.PolicyCls = None
        self.available = False
        self._try_import()

    def _try_import(self) -> None:
        try:
            import stable_baselines3 as sb3  # type: ignore
            self.sb3 = sb3
            # map common algos
            algo_map = {
                "PPO": sb3.PPO,
                "A2C": sb3.A2C,
                "DDPG": getattr(sb3, "DDPG", None),
                "SAC": getattr(sb3, "SAC", None),
                "TD3": getattr(sb3, "TD3", None),
            }
            self.PolicyCls = algo_map.get(self.algo, sb3.PPO)
            if self.PolicyCls is None:
                self.PolicyCls = sb3.PPO
            self.available = True
        except Exception:
            # SB3 not available in this interpreter; operate in disabled mode
            self.available = False
            self.sb3 = None
            self.PolicyCls = None

    def load_policy(self, path: str | os.PathLike):
        """Load a trained policy (.zip) from policies_dir or absolute path."""
        if not self.available:
            return None
        p = Path(path)
        if not p.is_absolute():
            p = self.policies_dir / p
        if not p.exists():
            return None
        try:
            model = self.PolicyCls.load(str(p))
            return model
        except Exception:
            return None

    def predict_action(self, model, obs: np.ndarray) -> int:
        """
        Return {-1, 0, +1} as short/flat/long.
        We assume discrete actions {0,1,2} or continuous with sign.
        """
        if not self.available or model is None:
            return 0
        try:
            action, _ = model.predict(obs, deterministic=True)
            # discrete policy (0,1,2) -> (-1,0,1)
            if np.isscalar(action):
                a = int(action)
                if a <= 0:  # 0
                    return -1
                elif a == 1:
                    return 0
                else:
                    return 1
            # continuous: use sign of first dim
            a = float(np.asarray(action).ravel()[0])
            return 1 if a > 0 else (-1 if a < 0 else 0)
        except Exception:
            return 0

    def is_ready(self) -> bool:
        return self.available
