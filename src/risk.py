import numpy as np

def compute_position_size(equity, risk_per_trade, atr, value_per_point=1.0, clamp=(0.0,1.0)):
    units = (equity * risk_per_trade) / (atr * value_per_point + 1e-12)
    return max(clamp[0], min(clamp[1], units))

def risk_checks(garch_vol, max_vol, current_dd, max_dd_allowed, loss_streak, max_loss_streak):
    if garch_vol > max_vol: return False, 'high_vol'
    if current_dd > max_dd_allowed: return False, 'max_dd'
    if loss_streak >= max_loss_streak: return False, 'loss_streak'
    return True, 'ok'
