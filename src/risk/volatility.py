import pandas as pd
import numpy as np
from arch import arch_model

class VolatilityOracle:
    """
    Stage 5: Volatility Brain -> Hybrid Estimator
    Combines Real-time ATR (fast) with Daily GARCH (predictive).
    """
    def __init__(self, window=14):
        self.window = window

    def estimate(self, returns: pd.Series):
        # 1. Real-time ATR/HV (Historical Volatility)
        if len(returns) < self.window:
            return {"hv": 0.0, "garch": 0.0, "regime": "UNCERTAIN"}
            
        hv = returns.rolling(self.window).std().iloc[-1] * np.sqrt(252)
        
        # 2. GARCH Forecast (Daily gating)
        # Only run this periodically (e.g. daily), not per tick
        # For demo, we run it if we have enough data, but catch errors
        garch_vol = 0.0
        try:
            if len(returns) > 200:
                # Scale returns for numerical stability
                am = arch_model(returns * 100, vol='Garch', p=1, q=1)
                res = am.fit(disp='off')
                garch_vol = np.sqrt(res.forecast(horizon=1).variance.values[-1, 0]) / 100
            else:
                garch_vol = hv
        except:
            garch_vol = hv

        return {
            "hv": hv,
            "garch": garch_vol,
            "regime": "HIGH_VOL" if garch_vol > 0.015 else "NORMAL" # Example threshold
        }
