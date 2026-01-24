"""
FAST MODE RUNNER - ML-DRIVEN
=============================

Thin runner that provides features to ExecutionCore.
ALL decisions made inside ExecutionCore.

PHASE-2: Computes all 21 features required by xgb_primary model.
"""

import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.execution_core import ExecutionCore

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# VECTORIZED FEATURE CALCULATOR (21 features for ML)
# ============================================================================

class VectorizedFeatures:
    """
    Pre-compute ALL 21 features required by xgb_primary model.
    
    Feature list:
    open, high, low, close, volume, returns,
    sma_20, sma_50, ema_12, ema_26, trend_flag,
    rsi_14, macd, macd_signal, macd_hist,
    atr_14, bb_middle, bb_upper, bb_lower, bb_width, volatility
    """
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.n = len(data)
        self._compute_all()
    
    def _compute_all(self):
        """Compute all 21 features."""
        logger.info("Pre-computing 21 ML features...")
        start = time.time()
        
        # Basic OHLCV
        self.open = self.data["open"].values
        self.high = self.data["high"].values
        self.low = self.data["low"].values
        self.close = self.data["close"].values
        self.volume = self.data.get("volume", pd.Series(np.zeros(self.n))).values
        
        # Returns
        self.returns = np.zeros(self.n)
        self.returns[1:] = (self.close[1:] - self.close[:-1]) / self.close[:-1]
        
        # Moving averages
        self._compute_sma(20, "sma_20")
        self._compute_sma(50, "sma_50")
        self._compute_ema(12, "ema_12")
        self._compute_ema(26, "ema_26")
        
        # Trend flag
        self.trend_flag = np.where(self.ema_12 > self.ema_26, 1, -1)
        
        # RSI
        self._compute_rsi(14)
        
        # MACD
        self._compute_macd()
        
        # ATR
        self._compute_atr(14)
        
        # Bollinger Bands
        self._compute_bollinger(20)
        
        # Volatility
        self._compute_volatility(20)
        
        # Regime
        self._compute_regime()
        
        elapsed = time.time() - start
        logger.info(f"Features pre-computed in {elapsed:.2f}s")
    
    def _compute_sma(self, period: int, name: str):
        result = np.zeros(self.n)
        for i in range(period - 1, self.n):
            result[i] = np.mean(self.close[i - period + 1:i + 1])
        result[:period - 1] = self.close[:period - 1]
        setattr(self, name, result)
    
    def _compute_ema(self, period: int, name: str):
        result = np.zeros(self.n)
        mult = 2 / (period + 1)
        result[0] = self.close[0]
        for i in range(1, self.n):
            result[i] = self.close[i] * mult + result[i - 1] * (1 - mult)
        setattr(self, name, result)
    
    def _compute_rsi(self, period: int):
        delta = np.diff(self.close, prepend=self.close[0])
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)
        
        self.rsi_14 = np.zeros(self.n)
        for i in range(period, self.n):
            avg_gain = np.mean(gains[i - period + 1:i + 1])
            avg_loss = np.mean(losses[i - period + 1:i + 1])
            if avg_loss == 0:
                self.rsi_14[i] = 100
            else:
                rs = avg_gain / avg_loss
                self.rsi_14[i] = 100 - (100 / (1 + rs))
        self.rsi_14[:period] = 50
    
    def _compute_macd(self):
        self.macd = self.ema_12 - self.ema_26
        
        # Signal line (9-period EMA of MACD)
        self.macd_signal = np.zeros(self.n)
        mult = 2 / 10
        self.macd_signal[0] = self.macd[0]
        for i in range(1, self.n):
            self.macd_signal[i] = self.macd[i] * mult + self.macd_signal[i - 1] * (1 - mult)
        
        self.macd_hist = self.macd - self.macd_signal
    
    def _compute_atr(self, period: int):
        tr = np.zeros(self.n)
        tr[1:] = np.maximum(
            self.high[1:] - self.low[1:],
            np.maximum(
                np.abs(self.high[1:] - self.close[:-1]),
                np.abs(self.low[1:] - self.close[:-1])
            )
        )
        
        self.atr_14 = np.zeros(self.n)
        for i in range(period, self.n):
            self.atr_14[i] = np.mean(tr[i - period + 1:i + 1])
        self.atr_14[:period] = 0.001
    
    def _compute_bollinger(self, period: int):
        self.bb_middle = np.zeros(self.n)
        self.bb_std = np.zeros(self.n)
        
        for i in range(period - 1, self.n):
            window = self.close[i - period + 1:i + 1]
            self.bb_middle[i] = np.mean(window)
            self.bb_std[i] = np.std(window)
        
        self.bb_upper = self.bb_middle + 2 * self.bb_std
        self.bb_lower = self.bb_middle - 2 * self.bb_std
        self.bb_width = (self.bb_upper - self.bb_lower) / self.bb_middle
        self.bb_width = np.nan_to_num(self.bb_width, nan=0.0)
    
    def _compute_volatility(self, period: int):
        self.volatility = np.zeros(self.n)
        for i in range(period, self.n):
            self.volatility[i] = np.std(self.returns[i - period + 1:i + 1])
    
    def _compute_regime(self):
        """Compute regime based on volatility and trend."""
        self.regime = ["RANGE"] * self.n
        
        for i in range(20, self.n):
            bb_width = self.bb_width[i]
            
            if bb_width > 0.02:
                self.regime[i] = "DANGER"
            elif abs(self.trend_flag[i]) == 1 and self.rsi_14[i] > 55 or self.rsi_14[i] < 45:
                self.regime[i] = "TREND"
            else:
                self.regime[i] = "RANGE"
    
    def get(self, index: int) -> Dict:
        """Get all 21 features for a specific candle index."""
        return {
            "open": self.open[index],
            "high": self.high[index],
            "low": self.low[index],
            "close": self.close[index],
            "volume": self.volume[index],
            "returns": self.returns[index],
            "sma_20": self.sma_20[index],
            "sma_50": self.sma_50[index],
            "ema_12": self.ema_12[index],
            "ema_26": self.ema_26[index],
            "trend_flag": self.trend_flag[index],
            "rsi_14": self.rsi_14[index],
            "macd": self.macd[index],
            "macd_signal": self.macd_signal[index],
            "macd_hist": self.macd_hist[index],
            "atr_14": self.atr_14[index],
            "bb_middle": self.bb_middle[index],
            "bb_upper": self.bb_upper[index],
            "bb_lower": self.bb_lower[index],
            "bb_width": self.bb_width[index],
            "volatility": self.volatility[index],
            # Aliases for backward compatibility
            "atr": self.atr_14[index],
            "rsi": self.rsi_14[index],
        }
    
    def get_regime(self, index: int) -> str:
        return self.regime[index]


# ============================================================================
# THIN FAST MODE RUNNER
# ============================================================================

class FastModeRunner:
    """Thin runner that only provides features to ExecutionCore."""
    
    def __init__(self, initial_capital: float = 10000.0, output_dir: str = "fast_mode_logs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.core = ExecutionCore(initial_equity=initial_capital)
    
    def run(self, symbols: List[str], start_date: str, end_date: str) -> Dict:
        start_time = time.time()
        total_candles = 0
        
        for symbol in symbols:
            candles = self._run_symbol(symbol, start_date, end_date)
            total_candles += candles
        
        execution_time = time.time() - start_time
        
        results = self.core.get_results()
        results["symbols"] = symbols
        results["start_date"] = start_date
        results["end_date"] = end_date
        results["execution_time"] = execution_time
        results["candles_processed"] = total_candles
        results["candles_per_second"] = total_candles / execution_time if execution_time > 0 else 0
        
        return results
    
    def _run_symbol(self, symbol: str, start_date: str, end_date: str) -> int:
        logger.info(f"Loading {symbol} data...")
        df = self._load_data(symbol, start_date, end_date)
        
        if len(df) == 0:
            logger.warning(f"No data for {symbol}")
            return 0
        
        logger.info(f"Loaded {len(df):,} candles for {symbol}")
        
        features = VectorizedFeatures(df)
        
        logger.info(f"Running {symbol}...")
        
        for i in tqdm(range(len(df)), desc=f"🧠 {symbol}", unit="candle"):
            row = df.iloc[i]
            feat = features.get(i)
            regime = features.get_regime(i)
            
            self.core.on_candle(
                timestamp=row["timestamp"],
                symbol=symbol,
                open_price=row["open"],
                high_price=row["high"],
                low_price=row["low"],
                close_price=row["close"],
                candle_index=i,
                features=feat,
                regime=regime,
                mode="FAST",
            )
        
        return len(df)
    
    def _load_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        try:
            from src.market_data.unified_loader import load_unified
            df, _ = load_unified(symbol, "M15", start_date, end_date)
            return df
        except ImportError:
            pass
        
        csv_path = PROJECT_ROOT / f"data/raw/forex_kaggle_multiTF/{symbol}_M15.csv"
        if not csv_path.exists():
            return pd.DataFrame()
        
        df = pd.read_csv(csv_path)
        df.columns = [c.lower() for c in df.columns]
        if "date" in df.columns:
            df = df.rename(columns={"date": "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        start = pd.to_datetime(start_date, utc=True)
        end = pd.to_datetime(end_date, utc=True)
        df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]
        return df.reset_index(drop=True)
    
    def save_results(self, results: Dict):
        if results.get("trades"):
            trades_df = pd.DataFrame(results["trades"])
            trades_df.to_csv(self.output_dir / "trades.csv", index=False)
        
        summary = {k: v for k, v in results.items() if k != "trades"}
        with open(self.output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Results saved to {self.output_dir}")


def print_results(results: Dict):
    print("\n" + "🧠" * 30)
    print(" ML-DRIVEN FAST MODE COMPLETE")
    print("🧠" * 30)
    print()
    print(f" ML Model Loaded: {'✅ YES' if results.get('ml_model_loaded') else '❌ NO'}")
    print()
    print(f" Period:       {results['start_date']} to {results['end_date']}")
    print(f" Symbols:      {', '.join(results['symbols'])}")
    print()
    print(f" Final Equity: ${results['final_equity']:,.2f}")
    print(f" Total Return: {results['total_return']*100:+.2f}%")
    print(f" Max Drawdown: {results['max_drawdown']*100:.2f}%")
    print()
    print(f" Total Trades: {results['total_trades']}")
    print(f" Win Rate:     {results['winrate']*100:.1f}%")
    print()
    
    if results.get('decision_sources'):
        print(" Decision Sources:")
        for src, count in results['decision_sources'].items():
            print(f"   {src}: {count}")
    print()
    
    print(f" Candles:      {results['candles_processed']:,}")
    print(f" Time:         {results['execution_time']:.2f}s")
    print(f" Speed:        {results['candles_per_second']:,.0f} candles/sec")
    print("🧠" * 30)


def main():
    parser = argparse.ArgumentParser(description="ML-Driven Fast Mode")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--symbols", default="EURUSD", help="Comma-separated symbols")
    parser.add_argument("--capital", type=float, default=10000.0, help="Initial capital")
    parser.add_argument("--output", default="fast_mode_logs", help="Output directory")
    
    args = parser.parse_args()
    symbols = [s.strip() for s in args.symbols.split(",")]
    
    runner = FastModeRunner(initial_capital=args.capital, output_dir=args.output)
    results = runner.run(symbols=symbols, start_date=args.start, end_date=args.end)
    runner.save_results(results)
    print_results(results)


if __name__ == "__main__":
    main()
