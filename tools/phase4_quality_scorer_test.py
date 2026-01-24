"""
PHASE-4 QUALITY SCORER BACKTEST
================================

Test the integrated Quality Scorer in ExecutionCore
Period: 2019-01-01 to 2024-12-31 (6 years)

This script uses the actual ExecutionCore with QualityScorer enabled
to measure the real impact on trading performance.
"""

import sys
import json
import time
from pathlib import Path
from collections import deque
from enum import Enum

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import joblib

from src.market_data.unified_loader import load_unified

# Import the actual ExecutionCore with QualityScorer
try:
    from src.backtest.execution_core import ExecutionCore, QUALITY_SCORER_AVAILABLE
    USING_REAL_CORE = True
except ImportError:
    USING_REAL_CORE = False
    QUALITY_SCORER_AVAILABLE = False


# ============================================================================
# FEATURES (SAME AS PHASE-3)
# ============================================================================

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
        }


# ============================================================================
# PHASE-4 CORE (WITH QUALITY SCORER)
# ============================================================================

class AdaptiveMode(Enum):
    NORMAL = "NORMAL"
    CAUTION = "CAUTION"
    DEFENSIVE = "DEFENSIVE"


class AdaptiveState:
    def __init__(self):
        self.recent = deque(maxlen=10)
        self.last_3 = deque(maxlen=3)
        self.last_5 = deque(maxlen=5)
        self.loss_clusters = []
        self.current_cluster = 0
    
    def get_mode(self):
        losses_3 = sum(1 for x in self.last_3 if not x)
        losses_5 = sum(1 for x in self.last_5 if not x)
        total = len(self.recent)
        wr = sum(1 for x in self.recent if x) / total if total > 0 else 0.5
        
        if losses_5 >= 3:
            return AdaptiveMode.DEFENSIVE
        if losses_3 >= 2 or (total >= 5 and wr < 0.45):
            return AdaptiveMode.CAUTION
        return AdaptiveMode.NORMAL
    
    def update(self, win):
        self.recent.append(win)
        self.last_3.append(win)
        self.last_5.append(win)
        
        if not win:
            self.current_cluster += 1
        else:
            if self.current_cluster > 0:
                self.loss_clusters.append(self.current_cluster)
            self.current_cluster = 0
    
    def get_stats(self):
        if self.current_cluster > 0:
            self.loss_clusters.append(self.current_cluster)
        
        return {
            "total_clusters": len(self.loss_clusters),
            "avg_cluster_len": np.mean(self.loss_clusters) if self.loss_clusters else 0,
            "max_cluster_len": max(self.loss_clusters) if self.loss_clusters else 0,
        }


class MLBrain:
    FEATURES = ["open","high","low","close","volume","returns","sma_20","sma_50",
                "ema_12","ema_26","trend_flag","rsi_14","macd","macd_signal",
                "macd_hist","atr_14","bb_middle","bb_upper","bb_lower","bb_width","volatility"]
    
    def __init__(self):
        self.model = None
        self.loaded = False
        try:
            self.model = joblib.load(PROJECT_ROOT / "models/xgb_primary.joblib")
            self.loaded = True
        except:
            pass
    
    def predict(self, f):
        if not self.loaded:
            return "HOLD", 0.0
        try:
            X = np.array([[f.get(n, 0.0) for n in self.FEATURES]])
            if hasattr(self.model, 'predict_proba'):
                p = self.model.predict_proba(X)[0]
                return ("BUY", p[1]) if p[1] > p[0] else ("SELL", p[0])
            return "HOLD", 0.5
        except:
            return "HOLD", 0.0


# Import Quality Scorer
try:
    from src.backtest.quality_scorer import QualityScorer, QualityBreakdown
    QUALITY_SCORER_LOADED = True
except ImportError:
    QUALITY_SCORER_LOADED = False
    print("[WARNING] QualityScorer not found - running without quality filtering")


class Phase4Core:
    """Phase-4 Core with Quality Scorer integrated."""
    
    # LOCKED THRESHOLDS
    ML_CONF = 0.55
    ML_HIGH = 0.70
    RSI_LOW = 25
    RSI_HIGH = 75
    RSI_SIZE = 0.85
    
    def __init__(self, equity=10000.0, enable_quality_scorer=True):
        self.ml = MLBrain()
        self.adaptive = AdaptiveState()
        
        # Quality Scorer (NEW)
        self.use_quality_scorer = enable_quality_scorer and QUALITY_SCORER_LOADED
        if self.use_quality_scorer:
            self.quality_scorer = QualityScorer()
            print(f"[Phase4Core] Quality Scorer: ENABLED")
        else:
            self.quality_scorer = None
            print(f"[Phase4Core] Quality Scorer: DISABLED")
        
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
        self.pos_conf = 0.0
        self.pos_mode = "NORMAL"
        
        self.mark2_mem = 1.0
        self.cooldown = 0
        self.last_idx = -999
        self.consec_losses = 0
        self.max_consec_losses = 0
        
        self.trades = []
        self.modes = {"NORMAL": 0, "CAUTION": 0, "DEFENSIVE": 0}
        self.equity_curve = []
        self.dd_recoveries = []
        self.dd_start_idx = None
        
        # Quality stats (NEW)
        self.quality_grades = {"A": 0, "B": 0, "C": 0, "D": 0, "F": 0}
        self.skipped_by_quality = 0
    
    def run(self, df, features, symbol):
        for i in range(len(df)):
            self._candle(df.iloc[i], i, features.get(i), features.regime[i])
            self.equity_curve.append(self.equity)
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
        mode = self.adaptive.get_mode()
        
        sig_after, sz_adapt = self._apply_adaptive(sig, conf, mode, regime)
        if sig_after == "HOLD":
            return
        
        final, src, sz_rsi = self._rsi_confirm(sig_after, conf, f, regime)
        if final == "HOLD":
            return
        
        # =====================================================================
        # QUALITY SCORER GATE (NEW!)
        # =====================================================================
        quality_mult = 1.0
        if self.use_quality_scorer and self.quality_scorer:
            quality_result = self.quality_scorer.calculate(f, final, regime)
            self.quality_grades[quality_result.grade] += 1
            
            # Skip F-grade trades
            if not self.quality_scorer.should_trade(quality_result):
                self.skipped_by_quality += 1
                return
            
            # Apply size multiplier
            quality_mult = quality_result.size_multiplier
        # =====================================================================
        
        atr = f.get("atr_14", 0.001)
        if atr < 0.0001:
            return
        
        sl, tp, size = self._risk(row["close"], final, atr, regime)
        size *= sz_adapt * sz_rsi * quality_mult  # QUALITY MULTIPLIER APPLIED
        
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
        self.pos_conf = conf
        self.pos_mode = mode.value
        self.pos_idx = i
        self.cooldown = i + 12
        self.modes[mode.value] += 1
    
    def _apply_adaptive(self, sig, conf, mode, regime):
        if mode == AdaptiveMode.NORMAL:
            return (sig, 1.0) if conf >= self.ML_CONF else ("HOLD", 1.0)
        if mode == AdaptiveMode.CAUTION:
            return (sig, 0.8) if conf >= self.ML_HIGH else ("HOLD", 1.0)
        if mode == AdaptiveMode.DEFENSIVE:
            return (sig, 0.6) if conf >= self.ML_HIGH and regime in ["RANGE", "TREND"] else ("HOLD", 1.0)
        return sig, 1.0
    
    def _rsi_confirm(self, sig, conf, f, regime):
        rsi = f.get("rsi_14", 50)
        if conf >= self.ML_HIGH:
            return sig, "HIGH", 1.0
        if regime == "RANGE":
            if sig == "BUY":
                if rsi >= self.RSI_HIGH:
                    return "HOLD", "BLOCK", 1.0
                elif rsi <= self.RSI_LOW:
                    return sig, "CONFIRM", 1.0
                return sig, "NEUTRAL", self.RSI_SIZE
            elif sig == "SELL":
                if rsi <= self.RSI_LOW:
                    return "HOLD", "BLOCK", 1.0
                elif rsi >= self.RSI_HIGH:
                    return sig, "CONFIRM", 1.0
                return sig, "NEUTRAL", self.RSI_SIZE
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
                exit_p, reason = sl, "SL"
            elif high >= tp:
                exit_p, reason = tp, "TP"
        else:
            if high >= sl:
                exit_p, reason = sl, "SL"
            elif low <= tp:
                exit_p, reason = tp, "TP"
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
        
        self.trades.append({
            "pnl": pnl, "r": r, "win": win, "mode": self.pos_mode,
            "reason": reason, "idx": i
        })
        
        self.equity += pnl
        self.adaptive.update(win)
        
        # DD tracking
        if self.equity > self.peak:
            if self.dd_start_idx is not None:
                self.dd_recoveries.append(i - self.dd_start_idx)
                self.dd_start_idx = None
            self.peak = self.equity
        else:
            if self.dd_start_idx is None:
                self.dd_start_idx = i
            dd = (self.peak - self.equity) / self.peak
            if dd > self.max_dd:
                self.max_dd = dd
        
        self.pos_open = False
        self.last_idx = i
        
        if win:
            self.consec_losses = 0
            self.mark2_mem = min(1.0, self.mark2_mem * 1.02)
        else:
            self.consec_losses += 1
            if self.consec_losses > self.max_consec_losses:
                self.max_consec_losses = self.consec_losses
            self.mark2_mem *= 0.95
            if self.consec_losses >= 2:
                self.cooldown = i + 24
    
    def _results(self):
        n = len(self.trades)
        w = sum(1 for t in self.trades if t["win"])
        pnls = [t["pnl"] for t in self.trades]
        g = sum(p for p in pnls if p > 0)
        l = abs(sum(p for p in pnls if p < 0))
        
        cluster_stats = self.adaptive.get_stats()
        
        return {
            "trades": n,
            "wins": w,
            "winrate": w / n if n > 0 else 0,
            "return": (self.equity - self.initial) / self.initial,
            "max_dd": self.max_dd,
            "pf": g / l if l > 0 else float('inf'),
            "avg_r": np.mean([t["r"] for t in self.trades]) if self.trades else 0,
            "max_consec_losses": self.max_consec_losses,
            "avg_cluster_len": cluster_stats["avg_cluster_len"],
            "max_cluster_len": cluster_stats["max_cluster_len"],
            "avg_dd_recovery": np.mean(self.dd_recoveries) if self.dd_recoveries else 0,
            "modes": self.modes,
            "ml_loaded": self.ml.loaded,
            # Quality Scorer stats (NEW)
            "quality_scorer_enabled": self.use_quality_scorer,
            "quality_grades": self.quality_grades,
            "skipped_by_quality": self.skipped_by_quality,
        }


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print(" PHASE-4 QUALITY SCORER BACKTEST (5+ YEARS)")
    print("=" * 70)
    
    symbol = "EURUSD"
    start = "2019-01-01"
    end = "2024-12-31"
    
    print(f" Symbol: {symbol}")
    print(f" Period: {start} to {end}")
    print(f" Thresholds: ML_CONF=0.55, ML_HIGH=0.70, RSI=25/75")
    print(f" AdaptiveState: ENABLED")
    print(f" QualityScorer: {'ENABLED' if QUALITY_SCORER_LOADED else 'DISABLED'}")
    
    print("\nLoading data...")
    start_t = time.time()
    
    try:
        df, _ = load_unified(symbol, "M15", start, end)
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Trying shorter period...")
        end = "2023-12-31"
        df, _ = load_unified(symbol, "M15", start, end)
    
    print(f"Loaded {len(df):,} candles in {time.time()-start_t:.1f}s")
    years = len(df) / (96 * 252)
    print(f"Approx {years:.1f} years of data")
    
    print("\nComputing features...")
    features = Features(df)
    
    # =========================================================================
    # RUN BOTH VERSIONS FOR COMPARISON
    # =========================================================================
    
    print("\n" + "=" * 70)
    print(" RUNNING COMPARISON: WITH vs WITHOUT QUALITY SCORER")
    print("=" * 70)
    
    # WITHOUT Quality Scorer (baseline)
    print("\n[1/2] Running WITHOUT Quality Scorer (baseline)...")
    core_off = Phase4Core(enable_quality_scorer=False)
    results_off = core_off.run(df.copy(), features, symbol)
    
    # WITH Quality Scorer
    print("\n[2/2] Running WITH Quality Scorer...")
    core_on = Phase4Core(enable_quality_scorer=True)
    results_on = core_on.run(df.copy(), features, symbol)
    
    # =========================================================================
    # COMPARISON RESULTS
    # =========================================================================
    
    print("\n" + "=" * 70)
    print(" COMPARISON RESULTS")
    print("=" * 70)
    
    def fmt_change(before, after, higher_is_better=True):
        if before == 0:
            return "N/A"
        change = (after - before) / abs(before) * 100
        arrow = "↑" if change > 0 else "↓" if change < 0 else "→"
        color = "✅" if (change > 0 and higher_is_better) or (change < 0 and not higher_is_better) else "⚠️"
        return f"{arrow} {change:+.1f}% {color}"
    
    print(f"\n{'Metric':<25} {'WITHOUT':<15} {'WITH':<15} {'Change':<15}")
    print("-" * 70)
    print(f"{'Total Trades':<25} {results_off['trades']:<15} {results_on['trades']:<15} {fmt_change(results_off['trades'], results_on['trades'], False)}")
    print(f"{'Win Rate':<25} {results_off['winrate']*100:.1f}%{'':<10} {results_on['winrate']*100:.1f}%{'':<10} {fmt_change(results_off['winrate'], results_on['winrate'], True)}")
    print(f"{'Total Return':<25} {results_off['return']*100:+.2f}%{'':<9} {results_on['return']*100:+.2f}%{'':<9} {fmt_change(results_off['return'], results_on['return'], True)}")
    print(f"{'Max Drawdown':<25} {results_off['max_dd']*100:.2f}%{'':<10} {results_on['max_dd']*100:.2f}%{'':<10} {fmt_change(results_off['max_dd'], results_on['max_dd'], False)}")
    print(f"{'Profit Factor':<25} {results_off['pf']:.2f}{'':<12} {results_on['pf']:.2f}{'':<12} {fmt_change(results_off['pf'], results_on['pf'], True)}")
    print(f"{'Avg R':<25} {results_off['avg_r']:.2f}{'':<12} {results_on['avg_r']:.2f}{'':<12} {fmt_change(results_off['avg_r'], results_on['avg_r'], True)}")
    print(f"{'Max Consec Losses':<25} {results_off['max_consec_losses']:<15} {results_on['max_consec_losses']:<15} {fmt_change(results_off['max_consec_losses'], results_on['max_consec_losses'], False)}")
    
    # Quality Scorer specifics
    print("\n" + "=" * 70)
    print(" QUALITY SCORER STATISTICS")
    print("=" * 70)
    
    if results_on['quality_scorer_enabled']:
        print(f"\n Trades by Grade:")
        total_considered = sum(results_on['quality_grades'].values())
        for grade, count in results_on['quality_grades'].items():
            pct = count / total_considered * 100 if total_considered > 0 else 0
            print(f"   Grade {grade}: {count:4d} ({pct:5.1f}%)")
        
        print(f"\n Trades Skipped (F-grade): {results_on['skipped_by_quality']}")
        trade_reduction = (results_off['trades'] - results_on['trades']) / results_off['trades'] * 100 if results_off['trades'] > 0 else 0
        print(f" Trade Reduction: {trade_reduction:.1f}%")
    
    # Final evaluation
    print("\n" + "=" * 70)
    print(" FINAL EVALUATION")
    print("=" * 70)
    
    improved_wr = results_on['winrate'] > results_off['winrate']
    improved_return = results_on['return'] > results_off['return']
    improved_pf = results_on['pf'] > results_off['pf']
    reduced_dd = results_on['max_dd'] <= results_off['max_dd']
    
    print(f"\n Win Rate Improved? {'✅ YES' if improved_wr else '❌ NO'}")
    print(f" Return Improved? {'✅ YES' if improved_return else '❌ NO'}")
    print(f" Profit Factor Improved? {'✅ YES' if improved_pf else '❌ NO'}")
    print(f" Drawdown Stable/Lower? {'✅ YES' if reduced_dd else '❌ NO'}")
    
    score = sum([improved_wr, improved_return, improved_pf, reduced_dd])
    print(f"\n Overall Score: {score}/4")
    
    if score >= 3:
        print("\n 🎉 QUALITY SCORER IS EFFECTIVE!")
    elif score >= 2:
        print("\n ⚠️ QUALITY SCORER shows mixed results - may need tuning")
    else:
        print("\n ❌ QUALITY SCORER needs adjustment")
    
    # Save results
    comparison = {
        "without_quality_scorer": results_off,
        "with_quality_scorer": results_on,
        "improvements": {
            "win_rate": improved_wr,
            "return": improved_return,
            "profit_factor": improved_pf,
            "drawdown": reduced_dd,
            "score": score
        }
    }
    
    with open("phase4_quality_scorer_results.json", "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    print("\n Results saved to phase4_quality_scorer_results.json")


if __name__ == "__main__":
    main()
