import numpy as np
import pandas as pd

def classify_regime(feat: pd.DataFrame) -> pd.Series:
    vol_hi = feat['garch_vol'].quantile(0.80)
    slope_thr = feat['kama_slope'].abs().quantile(0.75)
    regime = np.where(feat['garch_vol'] >= vol_hi, 2,
               np.where(feat['kama_slope'].abs() >= slope_thr, 1, 0))
    return pd.Series(regime, index=feat.index, name='regime')
