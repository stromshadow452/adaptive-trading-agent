import numpy as np
import pandas as pd
try:
    from arch import arch_model
    _HAVE_ARCH = True
except Exception:
    _HAVE_ARCH = False

def compute_kama(close, er_period=10, fast=2, slow=30):
    close = close.astype(float)
    change = close.diff(er_period).abs()
    vol = close.diff().abs().rolling(er_period).sum()
    er = np.where(vol.values == 0, 0, (change / vol).values)
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1))**2
    kama = np.zeros(len(close))
    kama[:er_period] = close.iloc[:er_period].values
    for i in range(er_period, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close.iloc[i] - kama[i-1])
    return pd.Series(kama, index=close.index, name='kama')

def kalman_1d(series, R=0.01, Q=1e-5):
    xhat, P = series.iloc[0], 1.0
    xhat_list = []
    for z in series.astype(float).values:
        xhat_minus = xhat
        P_minus = P + Q
        K = P_minus / (P_minus + R)
        xhat = xhat_minus + K * (z - xhat_minus)
        P = (1 - K) * P_minus
        xhat_list.append(xhat)
    return pd.Series(xhat_list, index=series.index, name='kalman')

def garch_vol(close, ann_factor=252, min_obs=300):
    rets = np.log(close.astype(float)).diff().dropna()
    if not _HAVE_ARCH or len(rets) < min_obs:
        ewm = rets.ewm(span=60, adjust=False).std() * np.sqrt(ann_factor)
        return ewm.reindex(close.index).rename('garch_vol')
    try:
        am = arch_model(rets * 100, p=1, q=1, vol='Garch', dist='normal', mean='Zero')
        res = am.fit(disp='off')
        cond_vol = res.conditional_volatility / 100.0 * np.sqrt(ann_factor)
        return cond_vol.reindex(close.index).rename('garch_vol')
    except Exception:
        ewm = rets.ewm(span=60, adjust=False).std() * np.sqrt(ann_factor)
        return ewm.reindex(close.index).rename('garch_vol')

def make_feature_df(df):
    out = pd.DataFrame(index=df.index)
    out['close'] = df['Close'].astype(float)
    out['ret'] = np.log(out['close']).diff().fillna(0.0)
    out['kama'] = compute_kama(out['close'])
    out['kama_slope'] = out['kama'].diff().fillna(0.0)
    out['kalman'] = kalman_1d(out['close'])
    out['garch_vol'] = garch_vol(out['close'])
    out['atr'] = (df['High'] - df['Low']).rolling(14).mean()
    out = out.dropna()
    return out
