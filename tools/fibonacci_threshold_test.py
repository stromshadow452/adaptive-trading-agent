"""
FIBONACCI THRESHOLD A/B TEST
=============================

Compares BASELINE thresholds vs FIBONACCI thresholds.
No other changes allowed.
"""

import sys
import json
import time
from pathlib import Path
from copy import deepcopy

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from tqdm import tqdm

from src.market_data.unified_loader import load_unified


# ============================================================================
# THRESHOLDS
# ============================================================================

BASELINE_THRESHOLDS = {
    "ML_CONF_THRESHOLD": 0.55,
    "ML_HIGH_CONF": 0.70,
    "RSI_EXTREME_LOW": 25,
    "RSI_EXTREME_HIGH": 75,
    "RSI_NEUTRAL_SIZE": 0.85,
}

FIBONACCI_THRESHOLDS = {
    "ML_CONF_THRESHOLD": 0.57,   # MICRO-TUNED: +0.02
    "ML_HIGH_CONF": 0.72,        # MICRO-TUNED: +0.02
    "RSI_EXTREME_LOW": 25,       # unchanged
    "RSI_EXTREME_HIGH": 75,      # unchanged
    "RSI_NEUTRAL_SIZE": 0.80,    # MICRO-TUNED: -0.05
}


# ============================================================================
# INLINE EXECUTION CORE (to allow threshold injection)
# ============================================================================

import joblib
import logging

logger = logging.getLogger(__name__)


class MLBrain:
    FEATURE_NAMES = [
        "open", "high", "low", "close", "volume", "returns",
        "sma_20", "sma_50", "ema_12", "ema_26", "trend_flag",
        "rsi_14", "macd", "macd_signal", "macd_hist",
        "atr_14", "bb_middle", "bb_upper", "bb_lower", "bb_width", "volatility"
    ]
    
    def __init__(self, model_path=None):
        self.model = None
        self.model_loaded = False
        
        if model_path is None:
            model_path = PROJECT_ROOT / "models/xgb_primary.joblib"
        
        if model_path.exists():
            try:
                self.model = joblib.load(model_path)
                self.model_loaded = True
            except Exception as e:
                pass
    
    def predict(self, features):
        if not self.model_loaded:
            return "HOLD", 0.0
        
        try:
            feature_vector = [features.get(name, 0.0) for name in self.FEATURE_NAMES]
            X = np.array([feature_vector])
            
            if hasattr(self.model, 'predict_proba'):
                probs = self.model.predict_proba(X)[0]
                prob_bearish = probs[0]
                prob_bullish = probs[1]
                
                if prob_bullish > prob_bearish:
                    return "BUY", prob_bullish
                else:
                    return "SELL", prob_bearish
            else:
                pred = self.model.predict(X)[0]
                return "BUY" if pred == 1 else "SELL", 0.6
        except:
            return "HOLD", 0.0


class ThresholdTestCore:
    """Execution core with injectable thresholds."""
    
    MIN_BARS_BETWEEN_TRADES = 12
    ATR_SL_MULT = 1.5
    ATR_TP_RR = 1.5
    MIN_RR = 1.41
    
    def __init__(self, thresholds, initial_equity=10000.0):
        self.thresholds = thresholds
        self.ml_brain = MLBrain()
        
        # State
        self.equity = initial_equity
        self.initial_equity = initial_equity
        self.peak_equity = initial_equity
        self.max_dd = 0.0
        
        self.position_open = False
        self.position_side = ""
        self.position_entry = 0.0
        self.position_size = 0.0
        self.position_sl = 0.0
        self.position_tp = 0.0
        self.position_entry_idx = 0
        self.position_regime = ""
        self.position_ml_conf = 0.0
        self.position_decision_src = ""
        
        self.mark2_memory = 1.0
        self.mark2_cooldown_until = 0
        self.last_trade_index = -999
        self.consecutive_losses = 0
        
        self.trades = []
        self.decision_counts = {}
    
    def run(self, df, symbol, features_obj):
        for i in range(len(df)):
            row = df.iloc[i]
            feat = features_obj.get(i)
            regime = features_obj.get_regime(i)
            
            self._process_candle(
                timestamp=row["timestamp"],
                symbol=symbol,
                high=row["high"],
                low=row["low"],
                close=row["close"],
                idx=i,
                features=feat,
                regime=regime,
            )
        
        return self._get_results()
    
    def _process_candle(self, timestamp, symbol, high, low, close, idx, features, regime):
        # Check SL/TP
        if self.position_open:
            self._check_sl_tp(high, low, idx, timestamp)
        
        # DANGER block
        if regime == "DANGER":
            return
        
        # MARK-2 gate
        if idx < self.mark2_cooldown_until:
            return
        if idx - self.last_trade_index < self.MIN_BARS_BETWEEN_TRADES:
            return
        if self.mark2_memory < 0.20:
            return
        
        if self.position_open:
            return
        
        # ML inference
        ml_signal, ml_conf = self.ml_brain.predict(features)
        
        # Decision logic with thresholds
        final_signal, decision_src, size_mod = self._evaluate_signal(
            ml_signal, ml_conf, features, regime
        )
        
        if final_signal == "HOLD":
            return
        
        # Risk calc
        atr = features.get("atr_14", features.get("atr", 0.001))
        if atr < 0.0001:
            return
        
        sl, tp, size = self._calc_risk(close, final_signal, atr, regime)
        size *= size_mod
        
        # RR check
        sl_dist = abs(close - sl)
        tp_dist = abs(close - tp)
        rr = tp_dist / sl_dist if sl_dist > 0.00001 else 0
        if rr < self.MIN_RR:
            return
        
        # Open position
        self.position_open = True
        self.position_side = final_signal
        self.position_entry = close
        self.position_size = size
        self.position_sl = sl
        self.position_tp = tp
        self.position_entry_idx = idx
        self.position_regime = regime
        self.position_ml_conf = ml_conf
        self.position_decision_src = decision_src
        self.position_entry_time = timestamp
        
        self.mark2_cooldown_until = idx + self.MIN_BARS_BETWEEN_TRADES
        
        # Track decision source
        self.decision_counts[decision_src] = self.decision_counts.get(decision_src, 0) + 1
    
    def _evaluate_signal(self, ml_signal, ml_conf, features, regime):
        t = self.thresholds
        rsi = features.get("rsi_14", features.get("rsi", 50))
        
        if ml_conf < t["ML_CONF_THRESHOLD"]:
            return "HOLD", "ML_LOW_CONF", 1.0
        
        if ml_conf >= t["ML_HIGH_CONF"]:
            return ml_signal, "ML_HIGH_CONF", 1.0
        
        if regime == "RANGE":
            if ml_signal == "BUY":
                if rsi >= t["RSI_EXTREME_HIGH"]:
                    return "HOLD", "ML_BLOCKED_RSI", 1.0
                elif rsi <= t["RSI_EXTREME_LOW"]:
                    return ml_signal, "ML+RSI_CONFIRM", 1.0
                else:
                    return ml_signal, "ML_RSI_NEUTRAL", t["RSI_NEUTRAL_SIZE"]
            elif ml_signal == "SELL":
                if rsi <= t["RSI_EXTREME_LOW"]:
                    return "HOLD", "ML_BLOCKED_RSI", 1.0
                elif rsi >= t["RSI_EXTREME_HIGH"]:
                    return ml_signal, "ML+RSI_CONFIRM", 1.0
                else:
                    return ml_signal, "ML_RSI_NEUTRAL", t["RSI_NEUTRAL_SIZE"]
        
        if regime == "TREND":
            return ml_signal, "ML_TREND", 1.0
        
        return ml_signal, "ML", 1.0
    
    def _calc_risk(self, entry, side, atr, regime):
        regime_mult = {"RANGE": 0.7, "TREND": 1.0}.get(regime, 0.8)
        sl_dist = atr * self.ATR_SL_MULT * regime_mult
        tp_dist = sl_dist * self.ATR_TP_RR
        
        if side == "BUY":
            sl = entry - sl_dist
            tp = entry + tp_dist
        else:
            sl = entry + sl_dist
            tp = entry - tp_dist
        
        risk_pct = 0.01 * self.mark2_memory
        risk_amount = self.equity * risk_pct
        size = risk_amount / sl_dist if sl_dist > 0 else 0
        
        return sl, tp, size
    
    def _check_sl_tp(self, high, low, idx, timestamp):
        if not self.position_open:
            return
        
        sl = self.position_sl
        tp = self.position_tp
        side = self.position_side
        
        exit_price = None
        exit_reason = None
        
        if side == "BUY":
            if low <= sl:
                exit_price = sl
                exit_reason = "SL_HIT"
            elif high >= tp:
                exit_price = tp
                exit_reason = "TP_HIT"
        else:
            if high >= sl:
                exit_price = sl
                exit_reason = "SL_HIT"
            elif low <= tp:
                exit_price = tp
                exit_reason = "TP_HIT"
        
        if exit_price is not None:
            self._close_position(exit_price, exit_reason, idx, timestamp)
    
    def _close_position(self, exit_price, exit_reason, idx, timestamp):
        if self.position_side == "BUY":
            pnl = (exit_price - self.position_entry) * self.position_size
        else:
            pnl = (self.position_entry - exit_price) * self.position_size
        
        is_win = pnl > 0
        sl_dist = abs(self.position_entry - self.position_sl)
        r_mult = pnl / (sl_dist * self.position_size) if sl_dist > 0 and self.position_size > 0 else 0
        
        self.trades.append({
            "side": self.position_side,
            "entry": self.position_entry,
            "exit": exit_price,
            "pnl": pnl,
            "r_mult": r_mult,
            "exit_reason": exit_reason,
            "regime": self.position_regime,
            "ml_conf": self.position_ml_conf,
            "decision_src": self.position_decision_src,
            "is_win": is_win,
        })
        
        self.equity += pnl
        
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
        else:
            dd = (self.peak_equity - self.equity) / self.peak_equity
            if dd > self.max_dd:
                self.max_dd = dd
        
        self.position_open = False
        self.last_trade_index = idx
        
        if is_win:
            self.consecutive_losses = 0
            self.mark2_memory = min(1.0, self.mark2_memory * 1.02)
        else:
            self.consecutive_losses += 1
            self.mark2_memory *= 0.95
            if self.consecutive_losses >= 2:
                self.mark2_cooldown_until = idx + self.MIN_BARS_BETWEEN_TRADES * 2
    
    def _get_results(self):
        total = len(self.trades)
        wins = sum(1 for t in self.trades if t["is_win"])
        
        pnls = [t["pnl"] for t in self.trades]
        gains = sum(p for p in pnls if p > 0)
        losses = abs(sum(p for p in pnls if p < 0))
        
        return {
            "total_trades": total,
            "wins": wins,
            "losses": total - wins,
            "winrate": wins / total if total > 0 else 0,
            "total_return": (self.equity - self.initial_equity) / self.initial_equity,
            "max_drawdown": self.max_dd,
            "profit_factor": gains / losses if losses > 0 else float('inf'),
            "avg_r": np.mean([t["r_mult"] for t in self.trades]) if self.trades else 0,
            "decision_counts": self.decision_counts,
            "ml_loaded": self.ml_brain.model_loaded,
        }


# ============================================================================
# FEATURE COMPUTATION (same as fast_mode.py)
# ============================================================================

class Features:
    def __init__(self, data):
        self.data = data
        self.n = len(data)
        self._compute()
    
    def _compute(self):
        self.open = self.data["open"].values
        self.high = self.data["high"].values
        self.low = self.data["low"].values
        self.close = self.data["close"].values
        self.volume = self.data.get("volume", pd.Series(np.zeros(self.n))).values
        
        self.returns = np.zeros(self.n)
        self.returns[1:] = (self.close[1:] - self.close[:-1]) / self.close[:-1]
        
        self._sma(20, "sma_20")
        self._sma(50, "sma_50")
        self._ema(12, "ema_12")
        self._ema(26, "ema_26")
        self.trend_flag = np.where(self.ema_12 > self.ema_26, 1, -1)
        self._rsi(14)
        self._macd()
        self._atr(14)
        self._bollinger(20)
        self._volatility(20)
        self._regime()
    
    def _sma(self, p, name):
        r = np.zeros(self.n)
        for i in range(p-1, self.n):
            r[i] = np.mean(self.close[i-p+1:i+1])
        r[:p-1] = self.close[:p-1]
        setattr(self, name, r)
    
    def _ema(self, p, name):
        r = np.zeros(self.n)
        m = 2/(p+1)
        r[0] = self.close[0]
        for i in range(1, self.n):
            r[i] = self.close[i]*m + r[i-1]*(1-m)
        setattr(self, name, r)
    
    def _rsi(self, p):
        d = np.diff(self.close, prepend=self.close[0])
        g = np.where(d>0, d, 0)
        l = np.where(d<0, -d, 0)
        self.rsi_14 = np.zeros(self.n)
        for i in range(p, self.n):
            ag = np.mean(g[i-p+1:i+1])
            al = np.mean(l[i-p+1:i+1])
            if al == 0:
                self.rsi_14[i] = 100
            else:
                self.rsi_14[i] = 100 - (100/(1+ag/al))
        self.rsi_14[:p] = 50
    
    def _macd(self):
        self.macd = self.ema_12 - self.ema_26
        self.macd_signal = np.zeros(self.n)
        m = 2/10
        self.macd_signal[0] = self.macd[0]
        for i in range(1, self.n):
            self.macd_signal[i] = self.macd[i]*m + self.macd_signal[i-1]*(1-m)
        self.macd_hist = self.macd - self.macd_signal
    
    def _atr(self, p):
        tr = np.zeros(self.n)
        tr[1:] = np.maximum(self.high[1:]-self.low[1:], np.maximum(np.abs(self.high[1:]-self.close[:-1]), np.abs(self.low[1:]-self.close[:-1])))
        self.atr_14 = np.zeros(self.n)
        for i in range(p, self.n):
            self.atr_14[i] = np.mean(tr[i-p+1:i+1])
        self.atr_14[:p] = 0.001
    
    def _bollinger(self, p):
        self.bb_middle = np.zeros(self.n)
        self.bb_std = np.zeros(self.n)
        for i in range(p-1, self.n):
            w = self.close[i-p+1:i+1]
            self.bb_middle[i] = np.mean(w)
            self.bb_std[i] = np.std(w)
        self.bb_upper = self.bb_middle + 2*self.bb_std
        self.bb_lower = self.bb_middle - 2*self.bb_std
        self.bb_width = np.nan_to_num((self.bb_upper-self.bb_lower)/self.bb_middle, nan=0.0)
    
    def _volatility(self, p):
        self.volatility = np.zeros(self.n)
        for i in range(p, self.n):
            self.volatility[i] = np.std(self.returns[i-p+1:i+1])
    
    def _regime(self):
        self.regime = ["RANGE"]*self.n
        for i in range(20, self.n):
            if self.bb_width[i] > 0.02:
                self.regime[i] = "DANGER"
            elif self.rsi_14[i] > 55 or self.rsi_14[i] < 45:
                self.regime[i] = "TREND"
    
    def get(self, i):
        return {
            "open": self.open[i], "high": self.high[i], "low": self.low[i], "close": self.close[i],
            "volume": self.volume[i], "returns": self.returns[i],
            "sma_20": self.sma_20[i], "sma_50": self.sma_50[i],
            "ema_12": self.ema_12[i], "ema_26": self.ema_26[i], "trend_flag": self.trend_flag[i],
            "rsi_14": self.rsi_14[i], "macd": self.macd[i], "macd_signal": self.macd_signal[i], "macd_hist": self.macd_hist[i],
            "atr_14": self.atr_14[i], "bb_middle": self.bb_middle[i], "bb_upper": self.bb_upper[i],
            "bb_lower": self.bb_lower[i], "bb_width": self.bb_width[i], "volatility": self.volatility[i],
            "atr": self.atr_14[i], "rsi": self.rsi_14[i],
        }
    
    def get_regime(self, i):
        return self.regime[i]


# ============================================================================
# MAIN TEST
# ============================================================================

def run_test(name, thresholds, df, symbol):
    print(f"\n{'='*60}")
    print(f" Running: {name}")
    print(f"{'='*60}")
    print(f" Thresholds: {thresholds}")
    
    features = Features(df)
    core = ThresholdTestCore(thresholds)
    results = core.run(df, symbol, features)
    
    print(f" ML Model Loaded: {results['ml_loaded']}")
    print(f" Total Trades: {results['total_trades']}")
    print(f" Win Rate: {results['winrate']*100:.1f}%")
    print(f" Return: {results['total_return']*100:+.2f}%")
    print(f" Max DD: {results['max_drawdown']*100:.2f}%")
    print(f" Profit Factor: {results['profit_factor']:.2f}")
    print(f" Avg R: {results['avg_r']:.2f}")
    print(f" Decision Sources: {results['decision_counts']}")
    
    return results


def main():
    symbol = "EURUSD"
    start = "2022-01-01"
    end = "2022-06-30"  # 6 months
    
    print("="*60)
    print(" FIBONACCI THRESHOLD A/B TEST")
    print("="*60)
    print(f" Symbol: {symbol}")
    print(f" Period: {start} to {end}")
    
    # Load data
    print("\nLoading data...")
    df, _ = load_unified(symbol, "M15", start, end)
    print(f"Loaded {len(df):,} candles")
    months = len(df) / (96 * 22)  # approx trading days per month
    
    # Run tests
    baseline = run_test("BASELINE", BASELINE_THRESHOLDS, df, symbol)
    fibonacci = run_test("FIBONACCI", FIBONACCI_THRESHOLDS, df, symbol)
    
    # Comparison
    print("\n" + "="*60)
    print(" COMPARISON SUMMARY")
    print("="*60)
    
    wr_change = fibonacci["winrate"] - baseline["winrate"]
    trades_change = fibonacci["total_trades"] - baseline["total_trades"]
    dd_change = fibonacci["max_drawdown"] - baseline["max_drawdown"]
    r_change = fibonacci["avg_r"] - baseline["avg_r"]
    
    print(f"\n Win Rate: {baseline['winrate']*100:.1f}% → {fibonacci['winrate']*100:.1f}% ({wr_change*100:+.1f}%)")
    print(f" Trades: {baseline['total_trades']} → {fibonacci['total_trades']} ({trades_change:+d})")
    print(f" Max DD: {baseline['max_drawdown']*100:.2f}% → {fibonacci['max_drawdown']*100:.2f}% ({dd_change*100:+.2f}%)")
    print(f" Avg R: {baseline['avg_r']:.2f} → {fibonacci['avg_r']:.2f} ({r_change:+.2f})")
    
    # Verdict
    wr_improved = wr_change > 0
    dd_controlled = dd_change <= 0.01  # DD didn't increase by more than 1%
    trades_reasonable = fibonacci["total_trades"] >= 3  # At least 3 trades
    
    quality_verdict = "IMPROVED" if (wr_improved and dd_controlled and trades_reasonable) else "SAME" if (abs(wr_change) < 0.05) else "WORSE"
    
    safe_to_adopt = wr_improved and dd_controlled and trades_reasonable
    
    print("\n" + "="*60)
    print(" FINAL VERDICT")
    print("="*60)
    print(f"\n Fibonacci thresholds improve win rate? {'YES' if wr_improved else 'NO'}")
    print(f" Safe to adopt? {'YES' if safe_to_adopt else 'NO'}")
    print(f" Quality Verdict: {quality_verdict}")
    
    if fibonacci["total_trades"] < 3:
        print(" ⚠️ Hidden Risk: Very few trades - may be too conservative")
    if fibonacci["max_drawdown"] > baseline["max_drawdown"] * 1.5:
        print(" ⚠️ Hidden Risk: Drawdown increased significantly")
    
    # Save results
    results = {
        "baseline": baseline,
        "fibonacci": fibonacci,
        "comparison": {
            "wr_change": wr_change,
            "trades_change": trades_change,
            "dd_change": dd_change,
            "r_change": r_change,
            "quality_verdict": quality_verdict,
            "safe_to_adopt": safe_to_adopt,
        }
    }
    
    with open("fibonacci_audit_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n Results saved to fibonacci_audit_results.json")


if __name__ == "__main__":
    main()
