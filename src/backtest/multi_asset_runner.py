"""
Multi-Asset Runner - PHASE 3
============================

Main orchestration loop for multi-asset trading.
Coordinates: Data → Screener → Pipeline → Portfolio → Execute

Flow:
1. Load candles for ALL pairs
2. Run screener (fast filter)
3. Run full pipeline for TRADEABLE pairs only
4. Portfolio Brain selects best trades
5. Execute approved trades only
"""

import sys
import json
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.multi_asset_screener import (
    MultiAssetScreener, ScreenerResult, get_screener
)
from src.backtest.portfolio_brain import (
    PortfolioBrain, CandidateSignal, OpenPosition, PortfolioDecision,
    get_portfolio_brain
)

logger = logging.getLogger(__name__)


# =============================================================================
# DATA TYPES
# =============================================================================

@dataclass
class MultiAssetConfig:
    """Configuration for multi-asset runner."""
    # Pairs to trade
    pairs: List[str] = field(default_factory=lambda: ["EURUSD", "GBPUSD", "USDJPY", "GBPJPY"])
    
    # Timeframe
    timeframe: str = "M15"
    
    # Backtest period
    start_date: str = "2019-01-01"
    end_date: str = "2024-12-31"
    
    # Portfolio limits
    max_open_trades: int = 2
    max_total_risk_pct: float = 0.03
    
    # ML thresholds (locked)
    ml_conf_threshold: float = 0.55
    ml_high_conf: float = 0.70
    
    # Initial equity
    initial_equity: float = 10000.0
    
    # Risk per trade
    risk_per_trade: float = 0.01
    
    # Logging
    verbose: bool = True


@dataclass
class TradeRecord:
    """Record of a completed trade."""
    trade_id: str
    symbol: str
    side: str
    entry_time: Any
    exit_time: Any
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    r_multiple: float
    exit_reason: str
    is_win: bool
    regime: str
    confidence: float


@dataclass
class RunnerState:
    """State of the multi-asset runner."""
    equity: float
    initial_equity: float
    peak_equity: float
    max_drawdown: float
    open_positions: List[OpenPosition] = field(default_factory=list)
    trades: List[TradeRecord] = field(default_factory=list)
    bar_index: int = 0
    
    # Statistics
    total_screened: int = 0
    total_tradeable: int = 0
    total_candidates: int = 0
    total_approved: int = 0
    
    # Mode distribution
    mode_distribution: Dict[str, int] = field(default_factory=lambda: {
        "NORMAL": 0, "CAUTION": 0, "DEFENSIVE": 0
    })


# =============================================================================
# SIMPLE FEATURE CALCULATOR
# =============================================================================

class SimpleFeatureCalculator:
    """Calculate features for screener and pipeline."""
    
    @staticmethod
    def calculate(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all features for a dataframe."""
        df = df.copy()
        
        # Price features
        df["returns"] = df["close"].pct_change()
        
        # Moving averages
        df["sma_20"] = df["close"].rolling(20).mean()
        df["sma_50"] = df["close"].rolling(50).mean()
        df["ema_12"] = df["close"].ewm(span=12).mean()
        df["ema_26"] = df["close"].ewm(span=26).mean()
        
        # RSI
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["rsi_14"] = 100 - (100 / (1 + rs))
        
        # ATR
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr_14"] = tr.rolling(14).mean()
        
        # Bollinger Bands
        df["bb_middle"] = df["close"].rolling(20).mean()
        bb_std = df["close"].rolling(20).std()
        df["bb_upper"] = df["bb_middle"] + (2 * bb_std)
        df["bb_lower"] = df["bb_middle"] - (2 * bb_std)
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
        
        # Volatility
        df["volatility"] = df["returns"].rolling(20).std()
        
        return df
    
    @staticmethod
    def get_features_at(df: pd.DataFrame, idx: int) -> Dict:
        """Extract features at a specific index."""
        row = df.iloc[idx]
        return {
            "open": row.get("open", 0),
            "high": row.get("high", 0),
            "low": row.get("low", 0),
            "close": row.get("close", 0),
            "returns": row.get("returns", 0),
            "sma_20": row.get("sma_20", 0),
            "sma_50": row.get("sma_50", 0),
            "ema_12": row.get("ema_12", 0),
            "ema_26": row.get("ema_26", 0),
            "rsi_14": row.get("rsi_14", 50),
            "atr_14": row.get("atr_14", 0),
            "bb_middle": row.get("bb_middle", 0),
            "bb_upper": row.get("bb_upper", 0),
            "bb_lower": row.get("bb_lower", 0),
            "bb_width": row.get("bb_width", 0),
            "volatility": row.get("volatility", 0),
        }


# =============================================================================
# SIMPLE ML BRAIN (for testing)
# =============================================================================

class SimpleMLBrain:
    """Simple ML brain for backtest (uses existing model or heuristic)."""
    
    def __init__(self, model_path: Optional[Path] = None):
        self.model = None
        self.model_loaded = False
        
        if model_path is None:
            model_path = PROJECT_ROOT / "models/xgb_primary.joblib"
        
        if model_path.exists():
            try:
                import joblib
                self.model = joblib.load(model_path)
                self.model_loaded = True
                logger.info(f"[MLBrain] Loaded model from {model_path}")
            except Exception as e:
                logger.warning(f"[MLBrain] Failed to load: {e}")
    
    def predict(self, features: Dict) -> Tuple[str, float]:
        """Generate signal and confidence."""
        if not self.model_loaded:
            return self._heuristic_predict(features)
        
        try:
            # Create feature vector
            feature_names = [
                "open", "high", "low", "close", "returns",
                "sma_20", "sma_50", "ema_12", "ema_26",
                "rsi_14", "atr_14", "bb_middle", "bb_upper", "bb_lower", "bb_width", "volatility"
            ]
            X = np.array([[features.get(n, 0) for n in feature_names]])
            
            if hasattr(self.model, 'predict_proba'):
                probs = self.model.predict_proba(X)[0]
                if len(probs) >= 2:
                    if probs[1] > probs[0]:
                        return "BUY", float(probs[1])
                    else:
                        return "SELL", float(probs[0])
            
            return "HOLD", 0.5
        except Exception as e:
            logger.warning(f"[MLBrain] Prediction error: {e}")
            return self._heuristic_predict(features)
    
    def _heuristic_predict(self, features: Dict) -> Tuple[str, float]:
        """Simple heuristic prediction."""
        rsi = features.get("rsi_14", 50)
        sma_20 = features.get("sma_20", 0)
        sma_50 = features.get("sma_50", 0)
        close = features.get("close", 0)
        
        signal = "HOLD"
        confidence = 0.5
        
        # Trend + RSI heuristic
        if sma_20 > 0 and sma_50 > 0:
            if sma_20 > sma_50 and rsi < 70:
                signal = "BUY"
                confidence = 0.5 + (sma_20 / sma_50 - 1) * 10
            elif sma_20 < sma_50 and rsi > 30:
                signal = "SELL"
                confidence = 0.5 + (1 - sma_20 / sma_50) * 10
        
        return signal, min(0.9, max(0.3, confidence))


# =============================================================================
# MULTI-ASSET RUNNER
# =============================================================================

class MultiAssetRunner:
    """
    Main orchestration loop for multi-asset trading.
    
    Flow:
    1. Load data for all pairs
    2. For each candle:
       a. Run screener on all pairs
       b. Run pipeline on tradeable pairs
       c. Portfolio Brain selects trades
       d. Execute approved trades
       e. Check SL/TP for open positions
    """
    
    def __init__(self, config: MultiAssetConfig):
        self.config = config
        self.state = RunnerState(
            equity=config.initial_equity,
            initial_equity=config.initial_equity,
            peak_equity=config.initial_equity,
            max_drawdown=0.0,
        )
        
        self.screener = MultiAssetScreener({"pairs": config.pairs})
        self.portfolio = PortfolioBrain({
            "max_open_trades": config.max_open_trades,
            "max_total_risk_pct": config.max_total_risk_pct,
        })
        self.ml_brain = SimpleMLBrain()
        self.feature_calc = SimpleFeatureCalculator()
        
        self.all_data: Dict[str, pd.DataFrame] = {}
        self._trade_counter = 0
        
        logger.info(f"[Runner] Initialized for {len(config.pairs)} pairs")
    
    def load_data(self) -> bool:
        """Load data for all pairs."""
        try:
            from src.market_data.unified_loader import load_unified
            
            for symbol in self.config.pairs:
                logger.info(f"[Runner] Loading {symbol}...")
                df, _ = load_unified(
                    symbol,
                    self.config.timeframe,
                    self.config.start_date,
                    self.config.end_date
                )
                
                # Calculate features
                df = self.feature_calc.calculate(df)
                self.all_data[symbol] = df
                logger.info(f"[Runner] {symbol}: {len(df)} candles loaded")
            
            return True
            
        except Exception as e:
            logger.error(f"[Runner] Failed to load data: {e}")
            return False
    
    def run(self) -> Dict:
        """Run the multi-asset backtest."""
        logger.info(f"\n{'='*60}")
        logger.info("MULTI-ASSET BACKTEST")
        logger.info(f"{'='*60}")
        logger.info(f"Pairs: {self.config.pairs}")
        logger.info(f"Period: {self.config.start_date} to {self.config.end_date}")
        
        start_time = time.time()
        
        # Load data
        if not self.load_data():
            return {"error": "Failed to load data"}
        
        # Get the minimum length across all pairs
        min_len = min(len(df) for df in self.all_data.values())
        logger.info(f"Running on {min_len} candles...")
        
        # Main loop
        for i in range(50, min_len):  # Start at 50 for indicator warmup
            self.state.bar_index = i
            
            # Get timestamp from first pair
            first_df = list(self.all_data.values())[0]
            timestamp = first_df.index[i] if hasattr(first_df.index, '__getitem__') else None
            
            # Step 1: Check SL/TP for open positions
            self._check_open_positions(i)
            
            # Step 2: Run screener
            all_features = {}
            all_atr_avg = {}
            
            for symbol in self.config.pairs:
                df = self.all_data[symbol]
                features = self.feature_calc.get_features_at(df, i)
                all_features[symbol] = features
                
                # Calculate ATR average (last 50 candles)
                atr_col = df["atr_14"].iloc[max(0, i-50):i+1]
                all_atr_avg[symbol] = atr_col.mean() if len(atr_col) > 0 else 0
            
            # Update screener with open positions
            open_symbols = [p.symbol for p in self.state.open_positions]
            self.screener.set_open_positions(open_symbols)
            
            screener_results = self.screener.screen_all(all_features, timestamp, all_atr_avg)
            self.state.total_screened += len(screener_results)
            
            tradeable = [r for r in screener_results if r.tradeable]
            self.state.total_tradeable += len(tradeable)
            
            # Step 3: Run pipeline for tradeable pairs
            candidates = []
            
            for result in tradeable[:3]:  # Top 3 only
                symbol = result.symbol
                features = all_features[symbol]
                df = self.all_data[symbol]
                close = df["close"].iloc[i]
                atr = features.get("atr_14", 0.001)
                
                # ML prediction
                signal, confidence = self.ml_brain.predict(features)
                
                if signal not in ["BUY", "SELL"]:
                    continue
                if confidence < self.config.ml_conf_threshold:
                    continue
                
                # Calculate SL/TP
                sl_dist = atr * 1.5
                tp_dist = sl_dist * 1.5
                
                if signal == "BUY":
                    sl = close - sl_dist
                    tp = close + tp_dist
                else:
                    sl = close + sl_dist
                    tp = close - tp_dist
                
                # Calculate size
                risk_amount = self.state.equity * self.config.risk_per_trade
                size = risk_amount / sl_dist if sl_dist > 0 else 0
                
                candidate = CandidateSignal(
                    symbol=symbol,
                    side=signal,
                    confidence=confidence,
                    entry_price=close,
                    sl_price=sl,
                    tp_price=tp,
                    size=size,
                    risk_pct=self.config.risk_per_trade,
                    regime=result.regime,
                    screener_rank=result.rank,
                    timestamp=timestamp,
                )
                candidates.append(candidate)
            
            self.state.total_candidates += len(candidates)
            
            # Step 4: Portfolio decision
            if candidates:
                decision = self.portfolio.decide(
                    candidates,
                    self.state.open_positions
                )
                
                # Step 5: Execute approved trades
                for signal in decision.approved:
                    self._open_position(signal, i, timestamp)
                    self.state.total_approved += 1
            
            # Progress logging
            if i % 5000 == 0 and self.config.verbose:
                logger.info(f"[Runner] Bar {i}/{min_len}, "
                           f"Trades: {len(self.state.trades)}, "
                           f"Equity: ${self.state.equity:.2f}")
        
        elapsed = time.time() - start_time
        
        return self._get_results(elapsed)
    
    def _check_open_positions(self, bar_idx: int):
        """Check SL/TP for all open positions."""
        closed = []
        
        for pos in self.state.open_positions:
            df = self.all_data[pos.symbol]
            high = df["high"].iloc[bar_idx]
            low = df["low"].iloc[bar_idx]
            
            # Find matching position data (we need to store more info)
            exit_price = None
            exit_reason = None
            
            # Get SL/TP from position metadata
            sl = getattr(pos, 'sl_price', 0)
            tp = getattr(pos, 'tp_price', 0)
            
            if pos.side == "BUY":
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
            
            if exit_price:
                self._close_position(pos, exit_price, exit_reason, bar_idx)
                closed.append(pos)
        
        # Remove closed positions
        for pos in closed:
            self.state.open_positions.remove(pos)
    
    def _open_position(self, signal: CandidateSignal, bar_idx: int, timestamp: Any):
        """Open a new position."""
        # Create extended position with SL/TP
        pos = OpenPosition(
            symbol=signal.symbol,
            side=signal.side,
            size=signal.size,
            risk_pct=signal.risk_pct,
            entry_time=timestamp,
        )
        
        # Add SL/TP as attributes
        pos.sl_price = signal.sl_price
        pos.tp_price = signal.tp_price
        pos.entry_price = signal.entry_price
        pos.entry_bar = bar_idx
        pos.confidence = signal.confidence
        pos.regime = signal.regime
        
        self.state.open_positions.append(pos)
        self._trade_counter += 1
        
        if self.config.verbose:
            logger.info(f"[OPEN] {signal.symbol} {signal.side} @ {signal.entry_price:.5f} "
                       f"(conf={signal.confidence:.2f})")
    
    def _close_position(self, pos: OpenPosition, exit_price: float, reason: str, bar_idx: int):
        """Close a position and record the trade."""
        entry_price = getattr(pos, 'entry_price', exit_price)
        
        # Calculate PnL
        if pos.side == "BUY":
            pnl = (exit_price - entry_price) * pos.size
        else:
            pnl = (entry_price - exit_price) * pos.size
        
        is_win = pnl > 0
        
        # R-multiple
        sl_dist = abs(entry_price - getattr(pos, 'sl_price', entry_price))
        r_mult = pnl / (sl_dist * pos.size) if sl_dist > 0 and pos.size > 0 else 0
        
        # Record trade
        trade = TradeRecord(
            trade_id=f"T{self._trade_counter:05d}",
            symbol=pos.symbol,
            side=pos.side,
            entry_time=pos.entry_time,
            exit_time=bar_idx,
            entry_price=entry_price,
            exit_price=exit_price,
            size=pos.size,
            pnl=pnl,
            r_multiple=r_mult,
            exit_reason=reason,
            is_win=is_win,
            regime=getattr(pos, 'regime', 'UNKNOWN'),
            confidence=getattr(pos, 'confidence', 0.5),
        )
        self.state.trades.append(trade)
        
        # Update equity
        self.state.equity += pnl
        
        if self.state.equity > self.state.peak_equity:
            self.state.peak_equity = self.state.equity
        else:
            dd = (self.state.peak_equity - self.state.equity) / self.state.peak_equity
            if dd > self.state.max_drawdown:
                self.state.max_drawdown = dd
        
        if self.config.verbose:
            win_str = "✅" if is_win else "❌"
            logger.info(f"[CLOSE] {win_str} {pos.symbol} {reason} @ {exit_price:.5f} "
                       f"PnL: {pnl:+.2f} (R: {r_mult:+.2f})")
    
    def _get_results(self, elapsed: float) -> Dict:
        """Compile final results."""
        trades = self.state.trades
        n = len(trades)
        wins = sum(1 for t in trades if t.is_win)
        
        pnls = [t.pnl for t in trades]
        gross_profit = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p < 0))
        
        # Per-symbol breakdown
        symbol_stats = {}
        for symbol in self.config.pairs:
            sym_trades = [t for t in trades if t.symbol == symbol]
            sym_wins = sum(1 for t in sym_trades if t.is_win)
            symbol_stats[symbol] = {
                "trades": len(sym_trades),
                "wins": sym_wins,
                "winrate": sym_wins / len(sym_trades) if sym_trades else 0,
            }
        
        return {
            "summary": {
                "total_trades": n,
                "winning_trades": wins,
                "losing_trades": n - wins,
                "winrate": wins / n if n > 0 else 0,
                "total_return": (self.state.equity - self.state.initial_equity) / self.state.initial_equity,
                "final_equity": self.state.equity,
                "max_drawdown": self.state.max_drawdown,
                "profit_factor": gross_profit / gross_loss if gross_loss > 0 else float('inf'),
                "avg_r": np.mean([t.r_multiple for t in trades]) if trades else 0,
            },
            "funnel": {
                "total_screened": self.state.total_screened,
                "total_tradeable": self.state.total_tradeable,
                "total_candidates": self.state.total_candidates,
                "total_approved": self.state.total_approved,
                "conversion_rate": self.state.total_approved / self.state.total_screened if self.state.total_screened > 0 else 0,
            },
            "by_symbol": symbol_stats,
            "config": {
                "pairs": self.config.pairs,
                "timeframe": self.config.timeframe,
                "period": f"{self.config.start_date} to {self.config.end_date}",
            },
            "elapsed_seconds": elapsed,
        }


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run multi-asset backtest."""
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    config = MultiAssetConfig(
        pairs=["EURUSD", "GBPUSD", "USDJPY", "GBPJPY"],
        timeframe="M15",
        start_date="2019-01-01",
        end_date="2024-12-31",
        max_open_trades=2,
        max_total_risk_pct=0.03,
        initial_equity=10000.0,
        verbose=True,
    )
    
    runner = MultiAssetRunner(config)
    results = runner.run()
    
    # Print results
    print("\n" + "=" * 60)
    print(" MULTI-ASSET BACKTEST RESULTS")
    print("=" * 60)
    
    summary = results["summary"]
    print(f"\n PERFORMANCE:")
    print(f"   Total Trades: {summary['total_trades']}")
    print(f"   Win Rate: {summary['winrate']*100:.1f}%")
    print(f"   Total Return: {summary['total_return']*100:+.2f}%")
    print(f"   Max Drawdown: {summary['max_drawdown']*100:.2f}%")
    print(f"   Profit Factor: {summary['profit_factor']:.2f}")
    print(f"   Avg R: {summary['avg_r']:.2f}")
    
    funnel = results["funnel"]
    print(f"\n FUNNEL:")
    print(f"   Screened: {funnel['total_screened']}")
    print(f"   Tradeable: {funnel['total_tradeable']} ({funnel['total_tradeable']/funnel['total_screened']*100:.1f}%)")
    print(f"   Candidates: {funnel['total_candidates']}")
    print(f"   Approved: {funnel['total_approved']}")
    
    print(f"\n BY SYMBOL:")
    for sym, stats in results["by_symbol"].items():
        print(f"   {sym}: {stats['trades']} trades, {stats['winrate']*100:.1f}% WR")
    
    # Save results
    with open("multi_asset_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to multi_asset_results.json")


if __name__ == "__main__":
    main()
