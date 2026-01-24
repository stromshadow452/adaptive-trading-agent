"""
Historical Training Executor - Rolling Window Method
=====================================================

Trains the adaptive trading agent on 24 years of historical data (2001-2025)
using a rolling-window approach that prevents lookahead bias.

Key Features:
- 6-month lookback window (agent never sees future)
- Day-by-day progression
- ERM + MARK-2 + Size Ladder evolution
- Checkpointing (daily/monthly/yearly)
- Phase-based risk tier unlocking

Usage:
    python tools/historical_trainer.py --start 2001-07-01 --end 2025-08-31
"""

import os
import sys
import json
import pickle
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from tqdm import tqdm

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.market_data.unified_loader import UnifiedDataLoader, load_unified

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class TrainingConfig:
    """Training configuration."""
    # Time settings
    lookback_days: int = 180  # 6 months
    start_date: str = "2001-07-01"
    end_date: str = "2025-08-31"
    
    # Symbols
    symbols: List[str] = field(default_factory=lambda: [
        "EURUSD", "GBPUSD", "USDJPY", "USDCAD", "USDCHF", "EURJPY"
    ])
    timeframe: str = "M15"
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_daily: bool = False
    save_monthly: bool = True
    save_yearly: bool = True
    
    # Phase settings
    cold_start_days: int = 180
    phase1_end: str = "2001-12-31"  # Learning
    phase2_end: str = "2002-12-31"  # TIER-1 unlock
    phase3_end: str = "2004-12-31"  # TIER-2 unlock
    phase4_end: str = "2010-12-31"  # Mature
    
    # Risk settings
    initial_capital: float = 10000.0
    enable_sniper: bool = False
    
    def get_phase(self, date: datetime) -> int:
        """Get current phase based on date."""
        if date < datetime.strptime(self.phase1_end, "%Y-%m-%d"):
            return 1
        elif date < datetime.strptime(self.phase2_end, "%Y-%m-%d"):
            return 2
        elif date < datetime.strptime(self.phase3_end, "%Y-%m-%d"):
            return 3
        elif date < datetime.strptime(self.phase4_end, "%Y-%m-%d"):
            return 4
        else:
            return 5


# ============================================================================
# TRAINING STATE
# ============================================================================

@dataclass
class TrainingState:
    """Persistent training state."""
    # Core metrics
    current_date: str = ""
    current_phase: int = 0
    trading_day: int = 0
    
    # Equity
    equity: float = 10000.0
    peak_equity: float = 10000.0
    max_drawdown: float = 0.0
    
    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    
    # ERM state
    erm_patterns: Dict = field(default_factory=dict)
    erm_observations: int = 0
    
    # MARK-2 state
    mark2_memory: float = 1.0
    mark2_ego: float = 1.0
    mark2_cooldown_until: str = ""
    
    # Size ladder state
    tier_distribution: Dict = field(default_factory=lambda: {
        "BASE": 0, "TIER_1": 0, "TIER_2": 0, "TIER_3": 0
    })
    
    # Monthly tracking
    monthly_pnl: float = 0.0
    monthly_trades: int = 0
    monthly_wins: int = 0
    
    # History
    equity_curve: List[float] = field(default_factory=list)
    monthly_returns: List[float] = field(default_factory=list)
    
    @property
    def winrate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def save(self, filepath: Path):
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(filepath: Path) -> "TrainingState":
        with open(filepath, "rb") as f:
            return pickle.load(f)


# ============================================================================
# MAIN TRAINER
# ============================================================================

class HistoricalTrainer:
    """
    Rolling-window historical trainer.
    
    Simulates real trading conditions:
    - Agent only sees past 6 months
    - Decisions made for current day only
    - No lookahead bias possible
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.loader = UnifiedDataLoader()
        self.state = TrainingState(equity=config.initial_capital)
        
        # Create checkpoint directories
        self.checkpoint_base = Path(config.checkpoint_dir)
        (self.checkpoint_base / "daily").mkdir(parents=True, exist_ok=True)
        (self.checkpoint_base / "monthly").mkdir(parents=True, exist_ok=True)
        (self.checkpoint_base / "yearly").mkdir(parents=True, exist_ok=True)
        
        # Load data for all symbols
        self.data: Dict[str, pd.DataFrame] = {}
        self._load_all_data()
    
    def _load_all_data(self):
        """Pre-load all symbol data."""
        logger.info("Loading data for all symbols...")
        
        for symbol in self.config.symbols:
            df, audit = self.loader.load(symbol, self.config.timeframe)
            if len(df) > 0:
                self.data[symbol] = df
                logger.info(f"  {symbol}: {len(df):,} candles ({audit.total_years:.1f}y)")
            else:
                logger.warning(f"  {symbol}: No data loaded!")
    
    def _get_window(
        self,
        symbol: str,
        current_date: datetime,
    ) -> pd.DataFrame:
        """Get lookback window for a date (no future data)."""
        if symbol not in self.data:
            return pd.DataFrame()
        
        df = self.data[symbol]
        
        # Calculate window bounds
        start_date = current_date - timedelta(days=self.config.lookback_days)
        
        # Filter to window (NO future data - critical)
        mask = (df["timestamp"] >= pd.Timestamp(start_date, tz="UTC")) & \
               (df["timestamp"] < pd.Timestamp(current_date, tz="UTC"))
        
        return df[mask].copy()
    
    def _get_day_candles(
        self,
        symbol: str,
        date: datetime,
    ) -> pd.DataFrame:
        """Get candles for a specific day."""
        if symbol not in self.data:
            return pd.DataFrame()
        
        df = self.data[symbol]
        
        # Get candles for this day only
        day_start = pd.Timestamp(date, tz="UTC")
        day_end = day_start + timedelta(days=1)
        
        mask = (df["timestamp"] >= day_start) & (df["timestamp"] < day_end)
        return df[mask].copy()
    
    def _simulate_day(
        self,
        date: datetime,
    ) -> Dict:
        """Simulate one trading day across all symbols."""
        day_results = {
            "date": date.strftime("%Y-%m-%d"),
            "trades": [],
            "pnl": 0.0,
        }
        
        for symbol in self.config.symbols:
            # Get lookback window (no future)
            window = self._get_window(symbol, date)
            if len(window) < 100:
                continue
            
            # Get today's candles
            day_candles = self._get_day_candles(symbol, date)
            if len(day_candles) == 0:
                continue
            
            # Simulate trading for each candle
            for _, candle in day_candles.iterrows():
                trade = self._evaluate_candle(symbol, window, candle, date)
                if trade:
                    day_results["trades"].append(trade)
                    day_results["pnl"] += trade["pnl"]
        
        return day_results
    
    def _evaluate_candle(
        self,
        symbol: str,
        window: pd.DataFrame,
        candle: pd.Series,
        date: datetime,
    ) -> Optional[Dict]:
        """
        Evaluate a single candle for trading opportunity.
        
        This is a placeholder - actual implementation should use:
        - ML Brain for signal
        - MARK-2 for permission
        - ERM for size adjustment
        - Risk Brain for SL/TP
        """
        # Phase determines allowed behavior
        phase = self.config.get_phase(date)
        
        # Phase 0 (cold start): No trading
        if self.state.trading_day < self.config.cold_start_days:
            return None
        
        # Simplified simulation for now
        # In real implementation, this calls the full pipeline
        
        # Random trade decision for demonstration (REPLACE with real logic)
        # This is just to show the structure
        if np.random.random() > 0.99:  # ~1% chance per candle
            is_win = np.random.random() > 0.45  # 55% win rate
            r_mult = 1.5 if is_win else -1.0
            size = 100 * (1.0 + (phase - 1) * 0.05)  # Size grows with phase
            pnl = size * r_mult * 0.001  # Simplified PnL
            
            return {
                "symbol": symbol,
                "timestamp": str(candle["timestamp"]),
                "side": "BUY" if np.random.random() > 0.5 else "SELL",
                "entry": candle["close"],
                "size": size,
                "pnl": pnl,
                "r_multiple": r_mult,
                "is_win": is_win,
                "phase": phase,
            }
        
        return None
    
    def _update_state(self, day_results: Dict, date: datetime):
        """Update training state after a day."""
        self.state.trading_day += 1
        self.state.current_date = date.strftime("%Y-%m-%d")
        self.state.current_phase = self.config.get_phase(date)
        
        # Update equity
        self.state.equity += day_results["pnl"]
        self.state.total_pnl += day_results["pnl"]
        
        # Track peak and drawdown
        if self.state.equity > self.state.peak_equity:
            self.state.peak_equity = self.state.equity
        else:
            dd = (self.state.peak_equity - self.state.equity) / self.state.peak_equity
            if dd > self.state.max_drawdown:
                self.state.max_drawdown = dd
        
        # Update trade counts
        for trade in day_results["trades"]:
            self.state.total_trades += 1
            self.state.monthly_trades += 1
            if trade["is_win"]:
                self.state.winning_trades += 1
                self.state.monthly_wins += 1
            else:
                self.state.losing_trades += 1
        
        # Monthly PnL
        self.state.monthly_pnl += day_results["pnl"]
        
        # Equity curve
        self.state.equity_curve.append(self.state.equity)
    
    def _checkpoint(self, date: datetime, force_type: str = None):
        """Save checkpoint if needed."""
        year = date.strftime("%Y")
        month = date.strftime("%Y-%m")
        day = date.strftime("%Y-%m-%d")
        
        # Daily checkpoint
        if self.config.save_daily or force_type == "daily":
            path = self.checkpoint_base / "daily" / year / f"{day}.pkl"
            path.parent.mkdir(parents=True, exist_ok=True)
            self.state.save(path)
        
        # Monthly checkpoint (on month end)
        is_month_end = (date + timedelta(days=1)).month != date.month
        if (is_month_end and self.config.save_monthly) or force_type == "monthly":
            # Record monthly return
            monthly_return = self.state.monthly_pnl / (self.state.equity - self.state.monthly_pnl)
            self.state.monthly_returns.append(monthly_return)
            
            # Save
            path = self.checkpoint_base / "monthly" / f"{month}.pkl"
            self.state.save(path)
            
            logger.info(
                f"[MONTH END] {month}: "
                f"Equity={self.state.equity:.2f}, "
                f"Trades={self.state.monthly_trades}, "
                f"PnL={self.state.monthly_pnl:+.2f} ({monthly_return*100:+.2f}%)"
            )
            
            # Reset monthly counters
            self.state.monthly_pnl = 0.0
            self.state.monthly_trades = 0
            self.state.monthly_wins = 0
        
        # Yearly checkpoint
        is_year_end = (date + timedelta(days=1)).year != date.year
        if (is_year_end and self.config.save_yearly) or force_type == "yearly":
            path = self.checkpoint_base / "yearly" / f"{year}.pkl"
            self.state.save(path)
            
            logger.info(f"[YEAR END] {year}: Equity={self.state.equity:.2f}")
    
    def run(self):
        """Execute the full training run."""
        logger.info("=" * 70)
        logger.info(" HISTORICAL TRAINING - STARTING")
        logger.info("=" * 70)
        logger.info(f" Period: {self.config.start_date} to {self.config.end_date}")
        logger.info(f" Symbols: {self.config.symbols}")
        logger.info(f" Lookback: {self.config.lookback_days} days")
        logger.info("=" * 70)
        
        # Generate trading days
        start = datetime.strptime(self.config.start_date, "%Y-%m-%d")
        end = datetime.strptime(self.config.end_date, "%Y-%m-%d")
        
        trading_days = []
        current = start
        while current <= end:
            # Skip weekends
            if current.weekday() < 5:
                trading_days.append(current)
            current += timedelta(days=1)
        
        logger.info(f" Trading days: {len(trading_days):,}")
        logger.info("")
        
        # Main training loop
        for date in tqdm(trading_days, desc="Training", unit="day"):
            try:
                # Simulate day
                day_results = self._simulate_day(date)
                
                # Update state
                self._update_state(day_results, date)
                
                # Checkpoint
                self._checkpoint(date)
                
            except Exception as e:
                logger.error(f"Error on {date}: {e}")
                continue
        
        # Final checkpoint
        self._checkpoint(end, force_type="yearly")
        
        # Summary
        self._print_summary()
    
    def _print_summary(self):
        """Print training summary."""
        print("\n" + "=" * 70)
        print(" TRAINING COMPLETE")
        print("=" * 70)
        print()
        print(f" Final Equity:    ${self.state.equity:,.2f}")
        print(f" Total Return:    {(self.state.equity / self.config.initial_capital - 1) * 100:.2f}%")
        print(f" Total Trades:    {self.state.total_trades:,}")
        print(f" Win Rate:        {self.state.winrate * 100:.1f}%")
        print(f" Max Drawdown:    {self.state.max_drawdown * 100:.2f}%")
        print(f" Trading Days:    {self.state.trading_day:,}")
        print()
        
        if self.state.monthly_returns:
            returns = np.array(self.state.monthly_returns)
            print(f" Avg Monthly:     {np.mean(returns) * 100:.2f}%")
            print(f" Sharpe (monthly): {np.mean(returns) / np.std(returns) * np.sqrt(12):.2f}")
        
        print("=" * 70)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Historical Training Executor")
    parser.add_argument("--start", default="2001-07-01", help="Start date")
    parser.add_argument("--end", default="2025-08-31", help="End date")
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--resume", help="Resume from checkpoint file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        ]
    )
    
    # Config
    config = TrainingConfig(
        start_date=args.start,
        end_date=args.end,
        checkpoint_dir=args.checkpoint_dir,
    )
    
    # Create trainer
    trainer = HistoricalTrainer(config)
    
    # Resume if specified
    if args.resume:
        checkpoint_path = Path(args.resume)
        if checkpoint_path.exists():
            trainer.state = TrainingState.load(checkpoint_path)
            logger.info(f"Resumed from {checkpoint_path}")
    
    # Run
    trainer.run()


if __name__ == "__main__":
    main()
