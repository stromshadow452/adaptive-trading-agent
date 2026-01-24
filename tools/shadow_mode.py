"""
MARK-2 SHADOW MODE RUNNER

Paper trading with real market data simulation.
All MARK-2 modules active with full observability.

PURPOSE: Observe agent survival behavior, NOT profits.

DAILY QUESTIONS TO ANSWER:
1. Size reducing? Why? (Memory/Ego/Regime)
2. After loss, agent slowing down? (Cooldown/Size)
3. After win, agent overtrading? (Ego control working?)
4. In DANGER, agent going silent? (Or forcing trades?)

"First make sure it survives the battlefield. Winning wars comes later."
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass, asdict

# Add project root to PATH for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('shadow_mode.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class ShadowTradeLog:
    """Log entry for each shadow trade."""
    timestamp: str
    symbol: str
    side: str
    entry_price: float
    sl_price: float
    tp_price: float
    size: float
    regime: str
    regime_strength: float
    
    # MARK-2 modifiers
    memory_mod: float
    ego_mod: float
    regime_mod: float
    final_size_mod: float
    
    # Ego state
    ego_score: float
    win_streak: int
    cooldown_active: bool
    
    # Memory state
    in_pain_zone: bool
    cluster_active: bool
    
    # Decision
    risk_decision: str
    confidence: float


@dataclass 
class DailyObservation:
    """Daily observation for MARK-2 behavior analysis."""
    date: str
    
    # Question 1: Size reducing?
    avg_size_modifier: float
    size_reductions_memory: int
    size_reductions_ego: int
    size_reductions_regime: int
    
    # Question 2: After loss, slowing down?
    losses_today: int
    cooldowns_triggered: int
    post_loss_size_reduction: float
    
    # Question 3: After win, overtrading?
    wins_today: int
    max_ego_score: float
    ego_blocks_triggered: int
    
    # Question 4: DANGER regime behavior
    danger_minutes: int
    trades_in_danger: int
    silent_in_danger: bool
    
    # Summary
    total_trades: int
    total_signals: int
    trade_rate: float  # trades per signal


class ShadowModeTracker:
    """
    Tracks MARK-2 behavior in shadow mode.
    Answers the 4 daily questions.
    """
    
    def __init__(self, output_dir: str = "shadow_mode_logs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.trade_logs: List[ShadowTradeLog] = []
        self.current_date: Optional[str] = None
        self.daily_observations: List[DailyObservation] = []
        
        # Daily counters
        self._reset_daily_counters()
        
        logger.info(f"ShadowModeTracker initialized - logs at {self.output_dir}")
    
    def _reset_daily_counters(self):
        """Reset daily observation counters."""
        self.signals_today = 0
        self.trades_today = 0
        self.losses_today = 0
        self.wins_today = 0
        self.cooldowns_triggered = 0
        self.ego_blocks = 0
        self.danger_minutes = 0
        self.trades_in_danger = 0
        self.size_reductions = {'memory': 0, 'ego': 0, 'regime': 0}
        self.size_modifiers = []
        self.max_ego_score = 0.0
        self.post_loss_sizes = []
    
    def on_signal(
        self,
        timestamp: datetime,
        symbol: str,
        side: str,
        confidence: float,
        regime: str,
        mark2_output: dict,
        risk_decision: str,
        entry_price: float = 0.0,
        sl_price: float = 0.0,
        tp_price: float = 0.0,
        size: float = 0.0
    ):
        """
        Called on every trading signal.
        Logs MARK-2 state whether trade executes or not.
        """
        date_str = timestamp.strftime('%Y-%m-%d')
        
        # Check for day change
        if self.current_date != date_str:
            if self.current_date is not None:
                self._save_daily_observation()
            self.current_date = date_str
            self._reset_daily_counters()
        
        self.signals_today += 1
        
        # Update danger tracking
        if regime == "DANGER":
            self.danger_minutes += 5  # Assuming M5 timeframe
        
        # Track size modifiers
        memory_mod = mark2_output.get('memory_mod', 1.0)
        ego_mod = mark2_output.get('ego_mod', 1.0)
        regime_mod = mark2_output.get('regime_mod', 1.0)
        final_size_mod = mark2_output.get('final_size_modifier', 1.0)
        
        self.size_modifiers.append(final_size_mod)
        
        # Track which module reduced size
        if memory_mod < 0.95:
            self.size_reductions['memory'] += 1
        if ego_mod < 0.95:
            self.size_reductions['ego'] += 1
        if regime_mod < 0.95:
            self.size_reductions['regime'] += 1
        
        # Track ego score
        ego_score = mark2_output.get('ego_score', 0.0)
        self.max_ego_score = max(self.max_ego_score, ego_score)
        
        # Track cooldown
        if not mark2_output.get('can_trade', True):
            self.cooldowns_triggered += 1
        
        # Log if trade executed
        if risk_decision != "BLOCK" and size > 0:
            self.trades_today += 1
            
            if regime == "DANGER":
                self.trades_in_danger += 1
            
            log_entry = ShadowTradeLog(
                timestamp=timestamp.isoformat(),
                symbol=symbol,
                side=side,
                entry_price=entry_price,
                sl_price=sl_price,
                tp_price=tp_price,
                size=size,
                regime=regime,
                regime_strength=mark2_output.get('regime_strength', 0.5),
                memory_mod=memory_mod,
                ego_mod=ego_mod,
                regime_mod=regime_mod,
                final_size_mod=final_size_mod,
                ego_score=ego_score,
                win_streak=mark2_output.get('win_streak', 0),
                cooldown_active=not mark2_output.get('can_trade', True),
                in_pain_zone=memory_mod < 0.9,
                cluster_active=mark2_output.get('cluster_active', False),
                risk_decision=risk_decision,
                confidence=confidence
            )
            
            self.trade_logs.append(log_entry)
            
            # Print observation
            self._print_trade_observation(log_entry)
    
    def on_trade_result(self, is_win: bool, size_after: float = 0.0):
        """Called when trade closes."""
        if is_win:
            self.wins_today += 1
        else:
            self.losses_today += 1
            if size_after > 0:
                self.post_loss_sizes.append(size_after)
    
    def _print_trade_observation(self, log: ShadowTradeLog):
        """Print observation in readable format."""
        print("\n" + "="*60)
        print(f"📊 SHADOW TRADE: {log.symbol} {log.side}")
        print(f"   Time: {log.timestamp}")
        print(f"   Regime: {log.regime} (strength: {log.regime_strength:.2f})")
        print()
        print(f"   🧠 MARK-2 MODIFIERS:")
        print(f"      Memory:  {log.memory_mod:.2f} {'⚠️ PAIN ZONE' if log.in_pain_zone else ''}")
        print(f"      Ego:     {log.ego_mod:.2f} (score: {log.ego_score:.2f})")
        print(f"      Regime:  {log.regime_mod:.2f}")
        print(f"      FINAL:   {log.final_size_mod:.2f}")
        print()
        print(f"   📈 Trade: size={log.size:.4f}, conf={log.confidence:.2f}")
        print("="*60)
    
    def _save_daily_observation(self):
        """Save daily observation to file."""
        avg_size = sum(self.size_modifiers) / len(self.size_modifiers) if self.size_modifiers else 1.0
        avg_post_loss = sum(self.post_loss_sizes) / len(self.post_loss_sizes) if self.post_loss_sizes else 1.0
        
        obs = DailyObservation(
            date=self.current_date,
            avg_size_modifier=avg_size,
            size_reductions_memory=self.size_reductions['memory'],
            size_reductions_ego=self.size_reductions['ego'],
            size_reductions_regime=self.size_reductions['regime'],
            losses_today=self.losses_today,
            cooldowns_triggered=self.cooldowns_triggered,
            post_loss_size_reduction=avg_post_loss,
            wins_today=self.wins_today,
            max_ego_score=self.max_ego_score,
            ego_blocks_triggered=self.ego_blocks,
            danger_minutes=self.danger_minutes,
            trades_in_danger=self.trades_in_danger,
            silent_in_danger=(self.trades_in_danger == 0 and self.danger_minutes > 30),
            total_trades=self.trades_today,
            total_signals=self.signals_today,
            trade_rate=self.trades_today / self.signals_today if self.signals_today > 0 else 0
        )
        
        self.daily_observations.append(obs)
        
        # Save to file
        obs_file = self.output_dir / f"observation_{self.current_date}.json"
        with open(obs_file, 'w') as f:
            json.dump(asdict(obs), f, indent=2)
        
        # Print daily summary
        self._print_daily_summary(obs)
    
    def _print_daily_summary(self, obs: DailyObservation):
        """Print daily summary answering the 4 questions."""
        print("\n" + "🌅"*30)
        print(f"📋 DAILY MARK-2 OBSERVATION: {obs.date}")
        print("🌅"*30)
        
        print("\n1️⃣ SIZE REDUCING? WHY?")
        print(f"   Avg Size Modifier: {obs.avg_size_modifier:.2f}")
        print(f"   Reductions by Memory: {obs.size_reductions_memory}")
        print(f"   Reductions by Ego: {obs.size_reductions_ego}")
        print(f"   Reductions by Regime: {obs.size_reductions_regime}")
        q1_ok = obs.avg_size_modifier < 1.0 and (obs.size_reductions_memory > 0 or obs.size_reductions_ego > 0)
        print(f"   ✅ MARK-2 WORKING" if q1_ok else "   ⚪ No reductions today")
        
        print("\n2️⃣ AFTER LOSS, SLOWING DOWN?")
        print(f"   Losses: {obs.losses_today}")
        print(f"   Cooldowns Triggered: {obs.cooldowns_triggered}")
        print(f"   Post-Loss Size Avg: {obs.post_loss_size_reduction:.2f}")
        q2_ok = obs.losses_today == 0 or obs.cooldowns_triggered > 0 or obs.post_loss_size_reduction < 0.9
        print(f"   ✅ SLOWING DOWN" if q2_ok else "   ⚠️ NOT SLOWING")
        
        print("\n3️⃣ AFTER WIN, OVERTRADING?")
        print(f"   Wins: {obs.wins_today}")
        print(f"   Max Ego Score: {obs.max_ego_score:.2f}")
        print(f"   Trade Rate: {obs.trade_rate:.1%}")
        q3_ok = obs.max_ego_score < 0.7 or obs.trade_rate < 0.5
        print(f"   ✅ EGO CONTROLLED" if q3_ok else "   ⚠️ POSSIBLE OVERTRADE")
        
        print("\n4️⃣ DANGER REGIME BEHAVIOR?")
        print(f"   Time in DANGER: {obs.danger_minutes} min")
        print(f"   Trades in DANGER: {obs.trades_in_danger}")
        q4_ok = obs.silent_in_danger or obs.trades_in_danger == 0 or obs.danger_minutes < 30
        print(f"   ✅ CAUTIOUS IN DANGER" if q4_ok else "   ⚠️ FORCING TRADES")
        
        print("\n📊 SUMMARY")
        print(f"   Total Signals: {obs.total_signals}")
        print(f"   Total Trades: {obs.total_trades}")
        print(f"   Trade Rate: {obs.trade_rate:.1%}")
        
        all_ok = q1_ok and q2_ok and q3_ok and q4_ok
        if all_ok:
            print("\n✅ MARK-2 SURVIVAL MODE: ACTIVE")
        else:
            print("\n⚠️ MARK-2 NEEDS REVIEW")
        
        print("🌅"*30 + "\n")
    
    def save_all_logs(self):
        """Save all trade logs to file."""
        if self.current_date:
            self._save_daily_observation()
        
        trades_file = self.output_dir / "all_shadow_trades.json"
        with open(trades_file, 'w') as f:
            json.dump([asdict(t) for t in self.trade_logs], f, indent=2)
        
        logger.info(f"Saved {len(self.trade_logs)} shadow trades to {trades_file}")
    
    def get_summary(self) -> dict:
        """Get overall shadow mode summary."""
        if not self.daily_observations:
            return {}
        
        return {
            'days_tracked': len(self.daily_observations),
            'total_trades': sum(o.total_trades for o in self.daily_observations),
            'total_signals': sum(o.total_signals for o in self.daily_observations),
            'avg_size_modifier': sum(o.avg_size_modifier for o in self.daily_observations) / len(self.daily_observations),
            'total_losses': sum(o.losses_today for o in self.daily_observations),
            'total_wins': sum(o.wins_today for o in self.daily_observations),
            'cooldowns_triggered': sum(o.cooldowns_triggered for o in self.daily_observations),
        }


# ============================================================================
# MAIN RUNNER
# ============================================================================

def run_shadow_mode_backtest(
    start_date: str = "2023-03-01",
    end_date: str = "2023-03-13",
    symbols: List[str] = None
):
    """
    Run MARK-2 in shadow mode using historical data.
    This simulates what would happen with real market data.
    """
    symbols = symbols or ["EURUSD"]
    
    print("\n" + "🔮"*30)
    print("MARK-2 SHADOW MODE STARTING")
    print("🔮"*30)
    print(f"\nPeriod: {start_date} to {end_date}")
    print(f"Symbols: {symbols}")
    print("\n⚠️ PAPER TRADING - NO REAL MONEY")
    print("📊 OBSERVING SURVIVAL BEHAVIOR, NOT PROFITS\n")
    
    tracker = ShadowModeTracker()
    
    # Run the actual backtest with MARK-2
    try:
        from src.backtest.engine import run_backtest
        from src.backtest.mark2_intelligence import MARK2Intelligence
        
        print("Running backtest with MARK-3 active...")
        print("=" * 60)
        
        import subprocess
        
        # Stream output in real-time (enables tqdm progress bar)
        process = subprocess.Popen(
            [
                sys.executable, "-m", "src.backtest.engine",
                "--config", "config/mvp_v1.yaml",
                "--symbols", ",".join(symbols),
                "--start", start_date,
                "--end", end_date,
                "--output", "shadow_mode_logs/backtest"
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',  # Handle Unicode properly (emojis etc)
            errors='replace',  # Replace undecodable chars
            bufsize=1,  # Line buffered
            cwd=os.getcwd()
        )
        
        # Stream output line by line
        for line in process.stdout:
            print(line, end='', flush=True)
        
        process.wait()
        
        print("=" * 60)

        
    except Exception as e:
        logger.error(f"Shadow mode error: {e}")
        raise
    
    tracker.save_all_logs()
    
    print("\n" + "🎯"*30)
    print("SHADOW MODE COMPLETE")
    print("🎯"*30)
    print(f"\nLogs saved to: shadow_mode_logs/")
    print("\nNEXT: Review daily observations and answer the 4 questions!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MARK-2 Shadow Mode")
    parser.add_argument("--start", default="2023-03-01", help="Start date")
    parser.add_argument("--end", default="2023-03-13", help="End date")
    parser.add_argument("--symbols", default="EURUSD", help="Comma-separated symbols")
    
    args = parser.parse_args()
    
    run_shadow_mode_backtest(
        start_date=args.start,
        end_date=args.end,
        symbols=args.symbols.split(",")
    )
