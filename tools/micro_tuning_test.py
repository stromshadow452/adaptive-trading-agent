"""
SAFE MICRO-TUNING A/B TEST
===========================

Comparing BASELINE vs minimal threshold adjustments.
"""

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import joblib

from src.market_data.unified_loader import load_unified


BASELINE = {
    "ML_CONF_THRESHOLD": 0.55,
    "ML_HIGH_CONF": 0.70,
    "RSI_EXTREME_LOW": 25,
    "RSI_EXTREME_HIGH": 75,
    "RSI_NEUTRAL_SIZE": 0.85,
}

MICRO_TUNED = {
    "ML_CONF_THRESHOLD": 0.57,  # +0.02
    "ML_HIGH_CONF": 0.72,        # +0.02
    "RSI_EXTREME_LOW": 25,       # unchanged
    "RSI_EXTREME_HIGH": 75,      # unchanged
    "RSI_NEUTRAL_SIZE": 0.80,    # -0.05
}


class MLBrain:
    FEATURES = ["open","high","low","close","volume","returns","sma_20","sma_50",
                "ema_12","ema_26","trend_flag","rsi_14","macd","macd_signal",
                "macd_hist","atr_14","bb_middle","bb_upper","bb_lower","bb_width","volatility"]
    
    def __init__(self):
        self.model = None
        try:
            self.model = joblib.load(PROJECT_ROOT / "models/xgb_primary.joblib")
        except:
            pass
    
    def predict(self, f):
        if self.model is None:
            return "HOLD", 0.0
        try:
            X = np.array([[f.get(n, 0.0) for n in self.FEATURES]])
            if hasattr(self.model, 'predict_proba'):
                p = self.model.predict_proba(X)[0]
                return ("BUY", p[1]) if p[1] > p[0] else ("SELL", p[0])
            return "HOLD", 0.5
        except:
            return "HOLD", 0.0


class TestCore:
    def __init__(self, thresholds, equity=10000.0):
        self.t = thresholds
        self.ml = MLBrain()
        self.equity = equity
        self.initial = equity
        self.peak = equity
        self.max_dd = 0.0
        self.pos_open = False
        self.pos_side = ""
        self.pos_entry = 0.0
        self.pos_size = 0.0
        self.pos_sl = 0.0
        self.pos_tp = 0.0
        self.pos_idx = 0
        self.pos_regime = ""
        self.pos_conf = 0.0
        self.pos_src = ""
        self.mark2_mem = 1.0
        self.cooldown = 0
        self.last_idx = -999
        self.losses = 0
        self.trades = []
        self.sources = {}
    
    def run(self, df, features, symbol):
        for i in range(len(df)):
            self._candle(df.iloc[i], i, features.get(i), features.regime[i])
        return self._results()
    
    def _candle(self, row, i, f, regime):
        if self.pos_open:
            self._check_exit(row["high"], row["low"], i)
        
        if regime == "DANGER":
            return
        if i < self.cooldown or i - self.last_idx < 12 or self.mark2_mem < 0.20:
            return
        if self.pos_open:
            return
        
        sig, conf = self.ml.predict(f)
        final, src, sz = self._eval(sig, conf, f, regime)
        
        if final == "HOLD":
            return
        
        atr = f.get("atr_14", f.get("atr", 0.001))
        if atr < 0.0001:
            return
        
        sl, tp, size = self._risk(row["close"], final, atr, regime)
        size *= sz
        
        sd = abs(row["close"] - sl)
        td = abs(row["close"] - tp)
        if sd > 0.00001 and td / sd < 1.41:
            return
        
        self.pos_open = True
        self.pos_side = final
        self.pos_entry = row["close"]
        self.pos_size = size
        self.pos_sl = sl
        self.pos_tp = tp
        self.pos_idx = i
        self.pos_regime = regime
        self.pos_conf = conf
        self.pos_src = src
        self.cooldown = i + 12
        self.sources[src] = self.sources.get(src, 0) + 1
    
    def _eval(self, sig, conf, f, regime):
        rsi = f.get("rsi_14", f.get("rsi", 50))
        
        if conf < self.t["ML_CONF_THRESHOLD"]:
            return "HOLD", "LOW_CONF", 1.0
        if conf >= self.t["ML_HIGH_CONF"]:
            return sig, "HIGH_CONF", 1.0
        
        if regime == "RANGE":
            if sig == "BUY":
                if rsi >= self.t["RSI_EXTREME_HIGH"]:
                    return "HOLD", "RSI_BLOCK", 1.0
                elif rsi <= self.t["RSI_EXTREME_LOW"]:
                    return sig, "RSI_CONFIRM", 1.0
                else:
                    return sig, "RSI_NEUTRAL", self.t["RSI_NEUTRAL_SIZE"]
            elif sig == "SELL":
                if rsi <= self.t["RSI_EXTREME_LOW"]:
                    return "HOLD", "RSI_BLOCK", 1.0
                elif rsi >= self.t["RSI_EXTREME_HIGH"]:
                    return sig, "RSI_CONFIRM", 1.0
                else:
                    return sig, "RSI_NEUTRAL", self.t["RSI_NEUTRAL_SIZE"]
        
        return sig, "TREND", 1.0
    
    def _risk(self, entry, side, atr, regime):
        m = {"RANGE": 0.7, "TREND": 1.0}.get(regime, 0.8)
        sl_d = atr * 1.5 * m
        tp_d = sl_d * 1.5
        if side == "BUY":
            sl, tp = entry - sl_d, entry + tp_d
        else:
            sl, tp = entry + sl_d, entry - tp_d
        risk = self.equity * 0.01 * self.mark2_mem
        size = risk / sl_d if sl_d > 0 else 0
        return sl, tp, size
    
    def _check_exit(self, high, low, i):
        sl, tp = self.pos_sl, self.pos_tp
        exit_p, reason = None, None
        
        if self.pos_side == "BUY":
            if low <= sl:
                exit_p, reason = sl, "SL_HIT"
            elif high >= tp:
                exit_p, reason = tp, "TP_HIT"
        else:
            if high >= sl:
                exit_p, reason = sl, "SL_HIT"
            elif low <= tp:
                exit_p, reason = tp, "TP_HIT"
        
        if exit_p:
            self._close(exit_p, reason, i)
    
    def _close(self, exit_p, reason, i):
        if self.pos_side == "BUY":
            pnl = (exit_p - self.pos_entry) * self.pos_size
        else:
            pnl = (self.pos_entry - exit_p) * self.pos_size
        
        win = pnl > 0
        sl_d = abs(self.pos_entry - self.pos_sl)
        r = pnl / (sl_d * self.pos_size) if sl_d > 0 and self.pos_size > 0 else 0
        
        self.trades.append({"pnl": pnl, "r": r, "win": win, "src": self.pos_src})
        self.equity += pnl
        
        if self.equity > self.peak:
            self.peak = self.equity
        else:
            dd = (self.peak - self.equity) / self.peak
            if dd > self.max_dd:
                self.max_dd = dd
        
        self.pos_open = False
        self.last_idx = i
        
        if win:
            self.losses = 0
            self.mark2_mem = min(1.0, self.mark2_mem * 1.02)
        else:
            self.losses += 1
            self.mark2_mem *= 0.95
            if self.losses >= 2:
                self.cooldown = i + 24
    
    def _results(self):
        n = len(self.trades)
        w = sum(1 for t in self.trades if t["win"])
        pnls = [t["pnl"] for t in self.trades]
        g = sum(p for p in pnls if p > 0)
        l = abs(sum(p for p in pnls if p < 0))
        return {
            "trades": n,
            "wins": w,
            "winrate": w / n if n > 0 else 0,
            "return": (self.equity - self.initial) / self.initial,
            "max_dd": self.max_dd,
            "pf": g / l if l > 0 else float('inf'),
            "avg_r": np.mean([t["r"] for t in self.trades]) if self.trades else 0,
            "sources": self.sources,
        }


class Features:
    def __init__(self, df):
        self.n = len(df)
        self.close = df["close"].values
        self.high = df["high"].values
        self.low = df["low"].values
        self.open = df["open"].values
        self.volume = df.get("volume", pd.Series(np.zeros(self.n))).values
        
        self.returns = np.zeros(self.n)
        self.returns[1:] = (self.close[1:] - self.close[:-1]) / self.close[:-1]
        
        self.sma_20 = self._sma(20)
        self.sma_50 = self._sma(50)
        self.ema_12 = self._ema(12)
        self.ema_26 = self._ema(26)
        self.trend_flag = np.where(self.ema_12 > self.ema_26, 1, -1)
        self.rsi_14 = self._rsi(14)
        self.macd = self.ema_12 - self.ema_26
        self.macd_signal = self._ema_arr(self.macd, 9)
        self.macd_hist = self.macd - self.macd_signal
        self.atr_14 = self._atr(14)
        self.bb_middle, self.bb_upper, self.bb_lower, self.bb_width = self._bb(20)
        self.volatility = self._vol(20)
        self.regime = self._regime()
    
    def _sma(self, p):
        r = np.zeros(self.n)
        for i in range(p - 1, self.n):
            r[i] = np.mean(self.close[i - p + 1:i + 1])
        r[:p - 1] = self.close[:p - 1]
        return r
    
    def _ema(self, p):
        r = np.zeros(self.n)
        m = 2 / (p + 1)
        r[0] = self.close[0]
        for i in range(1, self.n):
            r[i] = self.close[i] * m + r[i - 1] * (1 - m)
        return r
    
    def _ema_arr(self, arr, p):
        r = np.zeros(self.n)
        m = 2 / (p + 1)
        r[0] = arr[0]
        for i in range(1, self.n):
            r[i] = arr[i] * m + r[i - 1] * (1 - m)
        return r
    
    def _rsi(self, p):
        d = np.diff(self.close, prepend=self.close[0])
        g = np.where(d > 0, d, 0)
        l = np.where(d < 0, -d, 0)
        r = np.zeros(self.n)
        for i in range(p, self.n):
            ag = np.mean(g[i - p + 1:i + 1])
            al = np.mean(l[i - p + 1:i + 1])
            r[i] = 100 if al == 0 else 100 - (100 / (1 + ag / al))
        r[:p] = 50
        return r
    
    def _atr(self, p):
        tr = np.zeros(self.n)
        tr[1:] = np.maximum(self.high[1:] - self.low[1:],
                           np.maximum(np.abs(self.high[1:] - self.close[:-1]),
                                     np.abs(self.low[1:] - self.close[:-1])))
        r = np.zeros(self.n)
        for i in range(p, self.n):
            r[i] = np.mean(tr[i - p + 1:i + 1])
        r[:p] = 0.001
        return r
    
    def _bb(self, p):
        mid = np.zeros(self.n)
        std = np.zeros(self.n)
        for i in range(p - 1, self.n):
            w = self.close[i - p + 1:i + 1]
            mid[i] = np.mean(w)
            std[i] = np.std(w)
        upper = mid + 2 * std
        lower = mid - 2 * std
        width = np.zeros(self.n)
        mask = mid > 0
        width[mask] = (upper[mask] - lower[mask]) / mid[mask]
        return mid, upper, lower, width
    
    def _vol(self, p):
        r = np.zeros(self.n)
        for i in range(p, self.n):
            r[i] = np.std(self.returns[i - p + 1:i + 1])
        return r
    
    def _regime(self):
        r = ["RANGE"] * self.n
        for i in range(20, self.n):
            if self.bb_width[i] > 0.02:
                r[i] = "DANGER"
            elif self.rsi_14[i] > 55 or self.rsi_14[i] < 45:
                r[i] = "TREND"
        return r
    
    def get(self, i):
        return {
            "open": self.open[i], "high": self.high[i], "low": self.low[i],
            "close": self.close[i], "volume": self.volume[i], "returns": self.returns[i],
            "sma_20": self.sma_20[i], "sma_50": self.sma_50[i],
            "ema_12": self.ema_12[i], "ema_26": self.ema_26[i],
            "trend_flag": self.trend_flag[i], "rsi_14": self.rsi_14[i],
            "macd": self.macd[i], "macd_signal": self.macd_signal[i],
            "macd_hist": self.macd_hist[i], "atr_14": self.atr_14[i],
            "bb_middle": self.bb_middle[i], "bb_upper": self.bb_upper[i],
            "bb_lower": self.bb_lower[i], "bb_width": self.bb_width[i],
            "volatility": self.volatility[i],
            "atr": self.atr_14[i], "rsi": self.rsi_14[i],
        }


def main():
    print("=" * 60)
    print(" SAFE MICRO-TUNING AUDIT")
    print("=" * 60)
    
    symbol = "EURUSD"
    start = "2022-01-01"
    end = "2022-06-30"
    
    print(f" Symbol: {symbol}")
    print(f" Period: {start} to {end}")
    
    print("\nLoading data...")
    df, _ = load_unified(symbol, "M15", start, end)
    print(f"Loaded {len(df):,} candles")
    
    features = Features(df)
    
    # Run BASELINE
    print("\n" + "-" * 40)
    print(" BASELINE")
    print("-" * 40)
    baseline = TestCore(BASELINE)
    b = baseline.run(df, features, symbol)
    print(f" Trades: {b['trades']}")
    print(f" Win Rate: {b['winrate']*100:.1f}%")
    print(f" Return: {b['return']*100:+.2f}%")
    print(f" Max DD: {b['max_dd']*100:.2f}%")
    print(f" PF: {b['pf']:.2f}")
    print(f" Avg R: {b['avg_r']:.2f}")
    print(f" Sources: {b['sources']}")
    
    # Run MICRO-TUNED
    print("\n" + "-" * 40)
    print(" MICRO-TUNED")
    print("-" * 40)
    micro = TestCore(MICRO_TUNED)
    m = micro.run(df, features, symbol)
    print(f" Trades: {m['trades']}")
    print(f" Win Rate: {m['winrate']*100:.1f}%")
    print(f" Return: {m['return']*100:+.2f}%")
    print(f" Max DD: {m['max_dd']*100:.2f}%")
    print(f" PF: {m['pf']:.2f}")
    print(f" Avg R: {m['avg_r']:.2f}")
    print(f" Sources: {m['sources']}")
    
    # Comparison
    print("\n" + "=" * 60)
    print(" COMPARISON")
    print("=" * 60)
    
    wr_chg = m['winrate'] - b['winrate']
    tr_chg = m['trades'] - b['trades']
    dd_chg = m['max_dd'] - b['max_dd']
    
    print(f" Win Rate: {b['winrate']*100:.1f}% → {m['winrate']*100:.1f}% ({wr_chg*100:+.1f}%)")
    print(f" Trades: {b['trades']} → {m['trades']} ({tr_chg:+d})")
    print(f" Max DD: {b['max_dd']*100:.2f}% → {m['max_dd']*100:.2f}% ({dd_chg*100:+.2f}%)")
    print(f" PF: {b['pf']:.2f} → {m['pf']:.2f}")
    print(f" Avg R: {b['avg_r']:.2f} → {m['avg_r']:.2f}")
    
    # Verdict
    wr_improved = wr_chg >= 0.02
    dd_ok = dd_chg <= 0.005
    pf_ok = m['pf'] >= b['pf'] * 0.95
    trades_ok = m['trades'] >= b['trades'] * 0.5
    
    stability = "IMPROVED" if (wr_improved and dd_ok) else "SAME" if abs(wr_chg) < 0.02 else "WORSE"
    safe = wr_improved and dd_ok and pf_ok and trades_ok
    
    print("\n" + "=" * 60)
    print(" FINAL DECISION")
    print("=" * 60)
    print(f" Stability Verdict: {stability}")
    print(f" Safe to adopt micro-tuning? {'YES' if safe else 'NO'}")
    
    if m['trades'] < b['trades'] * 0.5:
        print(" ⚠️ Risk: Trade count collapsed")
    if dd_chg > 0.01:
        print(" ⚠️ Risk: Drawdown increased")
    if m['pf'] < 1:
        print(" ⚠️ Risk: Profit factor below 1")
    
    # Save
    with open("micro_tuning_results.json", "w") as f:
        json.dump({"baseline": b, "micro": m}, f, indent=2, default=str)
    print("\n Results saved to micro_tuning_results.json")


if __name__ == "__main__":
    main()
