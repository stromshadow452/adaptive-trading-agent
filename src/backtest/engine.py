"""
SCOPUS Backtest Engine - Candle-by-Candle Simulation

Real backtest engine that:
- Iterates through historical data candle by candle
- Uses BacktestBroker for position tracking
- Detects SL/TP hits
- Generates detailed trade logs
- Calculates comprehensive metrics
"""
import os
import argparse
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict, field
from pathlib import Path
import json
import logging
import time

from .broker import BacktestBroker
from .pipeline import TradingPipeline, PipelineConfigV2, Decision
from src.market_data import MarketDataStore, Symbol, Timeframe

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Backtest result container"""
    start_date: str
    end_date: str
    symbols: List[str]
    initial_capital: float
    final_equity: float
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    winrate: float
    total_pnl: float
    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    profit_factor: float
    avg_r_multiple: float
    avg_trade_duration_minutes: float
    exposure_pct: float
    total_candles: int
    
    # Detailed data
    trades: List[Dict]
    equity_curve: List[Dict]  # [{timestamp, equity, drawdown}]
    
    # Breakdowns
    regime_breakdown: Dict[str, Dict]
    decision_source_breakdown: Dict[str, Dict]
    
    # Metadata
    execution_time: float
    config: Dict


def load_and_filter_csv(
    csv_path: str,
    start_date: datetime,
    end_date: datetime
) -> pd.DataFrame:
    """
    Load CSV and filter by date range.
    
    Args:
        csv_path: Path to CSV file
        start_date: Start date
        end_date: End date
    
    Returns:
        Filtered DataFrame with OHLCV data
    """
    try:
        df = pd.read_csv(csv_path)
        
        # Detect timestamp column
        timestamp_col = None
        for col in ['timestamp', 'date', 'time', 'datetime']:
            if col in df.columns:
                timestamp_col = col
                break
        
        if timestamp_col is None:
            logger.error(f"No timestamp column found in {csv_path}")
            return pd.DataFrame()
        
        # Convert to datetime
        df['timestamp'] = pd.to_datetime(df[timestamp_col])
        
        # Filter by date range
        df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Ensure required columns exist
        required = ['open', 'high', 'low', 'close']
        for col in required:
            if col not in df.columns:
                logger.error(f"Missing required column '{col}' in {csv_path}")
                return pd.DataFrame()
        
        logger.info(f"Loaded {len(df)} candles from {os.path.basename(csv_path)}")
        
        return df
    
    except Exception as e:
        logger.error(f"Error loading {csv_path}: {e}")
        return pd.DataFrame()


# NOTE: simple_decision_logic has been REPLACED with the real 13-stage pipeline
# The pipeline is now initialized in run_backtest() and called via pipeline.decide()
# This function is kept for backward compatibility but is NO LONGER USED

def simple_decision_logic(
    candle: pd.Series,
    symbol: str,
    enable_meta_gating: bool = False,
    enable_portfolio: bool = False,
    enable_rl_fallback: bool = False
) -> Optional[Dict]:
    """
    DEPRECATED: This function is no longer used.
    The backtest now uses the real 13-stage pipeline from pipeline.py
    
    Kept for backward compatibility only.
    """
    logger.warning("simple_decision_logic is deprecated - using real pipeline instead")
    return None


def calculate_metrics(
    trades: List[Dict],
    equity_curve: List[Dict],
    initial_capital: float,
    total_candles: int
) -> Dict:
    """
    Calculate comprehensive backtest metrics.
    
    Args:
        trades: List of closed trades
        equity_curve: Equity history
        initial_capital: Starting capital
        total_candles: Total number of candles processed
    
    Returns:
        Dict of metrics
    """
    if not trades:
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'winrate': 0.0,
            'total_pnl': 0.0,
            'total_return': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'profit_factor': 0.0,
            'avg_r_multiple': 0.0,
            'avg_trade_duration_minutes': 0.0,
            'exposure_pct': 0.0
        }
    
    # Extract PnLs
    pnls = np.array([t['pnl'] for t in trades])
    
    # Win/Loss
    winning = sum(1 for p in pnls if p > 0)
    losing = sum(1 for p in pnls if p < 0)
    winrate = winning / len(trades)
    
    # Returns
    total_pnl = pnls.sum()
    final_equity = equity_curve[-1]['equity'] if equity_curve else initial_capital
    total_return = (final_equity - initial_capital) / initial_capital
    
    # Drawdown
    equities = np.array([e['equity'] for e in equity_curve])
    running_max = np.maximum.accumulate(equities)
    drawdowns = (equities - running_max) / running_max
    max_drawdown = abs(drawdowns.min()) if len(drawdowns) > 0 else 0.0
    
    # Sharpe ratio (annualized, assuming daily returns)
    if len(pnls) > 1 and pnls.std() > 0:
        sharpe = (pnls.mean() / pnls.std()) * np.sqrt(252)
    else:
        sharpe = 0.0
    
    # Sortino ratio
    downside = pnls[pnls < 0]
    if len(downside) > 1 and downside.std() > 0:
        sortino = (pnls.mean() / downside.std()) * np.sqrt(252)
    else:
        sortino = 0.0
    
    # Profit factor
    gross_profit = pnls[pnls > 0].sum()
    gross_loss = abs(pnls[pnls < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
    
    # R-multiples
    r_multiples = [t['r_multiple'] for t in trades]
    avg_r = np.mean(r_multiples) if r_multiples else 0.0
    
    # Duration
    durations = [t['duration_minutes'] for t in trades]
    avg_duration = np.mean(durations) if durations else 0.0
    
    # Exposure (rough estimate: total trade duration / total time)
    total_trade_minutes = sum(durations)
    total_minutes = total_candles * 15  # Assuming M15 data
    exposure_pct = total_trade_minutes / total_minutes if total_minutes > 0 else 0.0
    
    return {
        'total_trades': len(trades),
        'winning_trades': winning,
        'losing_trades': losing,
        'winrate': winrate,
        'total_pnl': float(total_pnl),
        'total_return': float(total_return),
        'max_drawdown': float(max_drawdown),
        'sharpe_ratio': float(sharpe),
        'sortino_ratio': float(sortino),
        'profit_factor': float(profit_factor),
        'avg_r_multiple': float(avg_r),
        'avg_trade_duration_minutes': float(avg_duration),
        'exposure_pct': float(exposure_pct)
    }


def calculate_breakdowns(trades: List[Dict]) -> tuple:
    """Calculate regime and decision source breakdowns"""
    regime_breakdown = {}
    decision_breakdown = {}
    
    for trade in trades:
        # Regime breakdown
        regime = trade['regime']
        if regime not in regime_breakdown:
            regime_breakdown[regime] = {'trades': 0, 'wins': 0, 'total_pnl': 0.0}
        
        regime_breakdown[regime]['trades'] += 1
        regime_breakdown[regime]['total_pnl'] += trade['pnl']
        if trade['pnl'] > 0:
            regime_breakdown[regime]['wins'] += 1
            
        # Decision source breakdown
        source = trade['decision_source']
        if source not in decision_breakdown:
            decision_breakdown[source] = {'trades': 0, 'wins': 0, 'total_pnl': 0.0}
        
        decision_breakdown[source]['trades'] += 1
        decision_breakdown[source]['total_pnl'] += trade['pnl']
        if trade['pnl'] > 0:
            decision_breakdown[source]['wins'] += 1
    
    # Calculate winrates and avgs
    for r in regime_breakdown.values():
        r['winrate'] = r['wins'] / r['trades'] if r['trades'] > 0 else 0.0
        r['avg_pnl'] = r['total_pnl'] / r['trades'] if r['trades'] > 0 else 0.0
        
    for s in decision_breakdown.values():
        s['winrate'] = s['wins'] / s['trades'] if s['trades'] > 0 else 0.0
        s['avg_pnl'] = s['total_pnl'] / s['trades'] if s['trades'] > 0 else 0.0
        
    return regime_breakdown, decision_breakdown


def run_backtest(
    symbols: List[str],
    start_date: datetime,
    end_date: datetime,
    initial_capital: float = 10000.0,
    enable_meta_gating: bool = False,
    enable_portfolio_brain: bool = False,
    enable_slicer: bool = False,
    enable_rl_fallback: bool = False,
    csv_price_dir: str = "data/raw/forex_kaggle_multiTF",
    output_dir: str = "logs/backtest",
    primary_model_path: Optional[str] = None,
    finrl_policies_path: Optional[str] = None,
    use_real_pipeline: bool = True
) -> BacktestResult:
    """
    Run candle-by-candle backtest with REAL 13-stage pipeline.
    
    Args:
        symbols: List of symbols to backtest
        start_date: Start date
        end_date: End date
        initial_capital: Starting capital
        enable_meta_gating: Enable Meta-Gating Brain (Stage 8)
        enable_portfolio_brain: Enable Portfolio Brain (Stage 9)
        enable_slicer: Enable Execution Slicer
        enable_rl_fallback: Enable RL fallback (Stage 4)
        csv_price_dir: Directory containing CSV files
        output_dir: Output directory for results
        primary_model_path: Path to primary ML model
        finrl_policies_path: Path to FinRL policies
        use_real_pipeline: Use real 13-stage pipeline (True) or legacy logic (False)
    
    Returns:
        BacktestResult object
    """
    start_time = time.time()
    
    logger.info(f"[SCOPUS BACKTEST] Starting backtest with {'REAL PIPELINE' if use_real_pipeline else 'LEGACY LOGIC'}")
    logger.info(f"  Symbols: {symbols}")
    logger.info(f"  Date range: {start_date.date()} to {end_date.date()}")
    logger.info(f"  Initial capital: ${initial_capital:,.2f}")
    logger.info(f"  Meta-Gating: {enable_meta_gating}")
    logger.info(f"  Portfolio Brain: {enable_portfolio_brain}")
    logger.info(f"  RL Fallback: {enable_rl_fallback}")
    
    # Initialize MarketDataStore with multiple data roots (including D1 backup)
    store = MarketDataStore(data_roots=[
        Path(csv_price_dir),
        Path("data/raw/forex_backup_2020_2025"),  # D1/Daily data
    ])
    logger.info(f"  MarketDataStore initialized with {len(store.registry.files)} files")
    
    # Make dates timezone-aware (UTC) to match DataFrame index
    import pytz
    if start_date.tzinfo is None:
        start_date = pytz.UTC.localize(start_date)
    if end_date.tzinfo is None:
        end_date = pytz.UTC.localize(end_date)
    
    # Initialize 13-stage pipeline
    pipeline = None
    if use_real_pipeline:
        pipeline_config = PipelineConfigV2(
            primary_model_path=primary_model_path,
            finrl_policies_path=finrl_policies_path,
            enable_meta_gating=enable_meta_gating,
            enable_portfolio_brain=enable_portfolio_brain,
            enable_rl_fallback=enable_rl_fallback,
            verbose=False
        )
        pipeline = TradingPipeline(pipeline_config)
        logger.info(f"  Pipeline initialized with primary_model={primary_model_path}")
    
    # Initialize broker
    broker = BacktestBroker(initial_capital=initial_capital)
    
    total_candles = 0
    
    # Process each symbol
    for symbol_str in symbols:
        # Convert to type-safe Symbol enum
        try:
            symbol = Symbol(symbol_str)
        except ValueError:
            logger.warning(f"Unknown symbol: {symbol_str}, skipping")
            continue
        
        # Load data using MarketDataStore
        df = store.load_ohlcv(
            symbol=symbol,
            timeframe=Timeframe.M15,
            start=start_date,
            end=end_date
        )
        
        if df.empty:
            logger.warning(f"No data for {symbol.value} in date range")
            continue
        
        # Normalize column names (store returns lowercase)
        df = df.reset_index()  # timestamp becomes column
        df = df.rename(columns={'index': 'timestamp'} if 'index' in df.columns else {})
        
        logger.info(f"Processing {symbol.value}: {len(df)} candles")
        total_candles += len(df)
        
        # Candle-by-candle iteration with progress bar
        try:
            from tqdm import tqdm
            pbar = tqdm(
                df.iterrows(),
                total=len(df),
                desc=f"🔮 {symbol.value}",
                unit="candle",
                miniters=50,  # Update every 50 candles for performance
                ncols=80,     # Fixed width for clean display
                leave=True    # Keep bar after completion
            )
        except ImportError:
            pbar = df.iterrows()
            
        for idx, candle in pbar:
            timestamp = candle['timestamp']
            
            # Update existing positions (check SL/TP hits)
            current_prices = {
                symbol.value: {  # Use symbol string for broker
                    'high': candle['high'],
                    'low': candle['low'],
                    'close': candle['close']
                }
            }
            
            closed_trades = broker.update_positions(current_prices, timestamp)
            
            # === MARK-2: Record trade results ===
            if use_real_pipeline and pipeline and closed_trades:
                for trade in closed_trades:
                    is_win = trade['pnl'] > 0
                    pipeline.mark2.record_trade_result(
                        entry_price=trade['entry_price'],
                        atr=abs(trade['entry_price'] - trade['sl_price']),  # Approximate ATR from SL distance
                        regime=trade.get('regime', 'RANGE'),
                        side=trade['side'].upper(),
                        is_win=is_win,
                        confidence=trade.get('confidence', 0.5),
                        r_multiple=trade.get('r_multiple', 0.0),
                        loss_streak=0  # Will be calculated by Ego Control internally
                    )

            
            # Make decision for new trade (if no position open for this symbol)
            if symbol.value not in broker.positions:
                decision_obj = None
                
                if use_real_pipeline and pipeline:
                    # Calculate state metrics for gating
                    # 1. Trades today
                    current_date = timestamp.date()
                    symbol_trades = [t for t in broker.closed_trades if t['symbol'] == symbol.value]
                    
                    trades_today = 0
                    # Count closed trades opened today
                    for t in symbol_trades:
                        entry_dt = datetime.fromisoformat(t['timestamp_entry'])
                        if entry_dt.date() == current_date:
                            trades_today += 1
                    
                    # 2. Bars since last trade
                    bars_since_last_trade = 9999
                    if symbol_trades:
                        last_trade = symbol_trades[-1]
                        last_entry = datetime.fromisoformat(last_trade['timestamp_entry'])
                        delta = timestamp - last_entry
                        bars_since_last_trade = int(delta.total_seconds() / (15 * 60))
                    
                    # 3. Consecutive losses & time since last loss
                    consecutive_losses = 0
                    bars_since_last_loss = 9999
                    
                    # Sort by exit time descending
                    sorted_trades = sorted(symbol_trades, key=lambda x: x['timestamp_exit'], reverse=True)
                    
                    for t in sorted_trades:
                        if t['pnl'] < 0:
                            consecutive_losses += 1
                        else:
                            break
                    
                    if sorted_trades and sorted_trades[0]['pnl'] < 0:
                        last_loss_exit = datetime.fromisoformat(sorted_trades[0]['timestamp_exit'])
                        delta_loss = timestamp - last_loss_exit
                        bars_since_last_loss = int(delta_loss.total_seconds() / (15 * 60))

                    # Use REAL 13-stage pipeline
                    context = {
                        'symbol': symbol.value,
                        'timeframe': 'M15',
                        'history': df.iloc[:idx+1],  # Historical data up to current candle
                        'open_positions': list(broker.positions.keys()),
                        'trades_today': trades_today,
                        'bars_since_last_trade': bars_since_last_trade,
                        'consecutive_losses': consecutive_losses,
                        'bars_since_last_loss': bars_since_last_loss
                    }
                    
                    decision_obj = pipeline.decide(candle, context)
                else:
                    # Fallback to legacy logic (deprecated)
                    legacy_decision = simple_decision_logic(
                        candle=candle,
                        symbol=symbol.value,
                        enable_meta_gating=enable_meta_gating,
                        enable_portfolio=enable_portfolio_brain,
                        enable_rl_fallback=enable_rl_fallback
                    )
                    # Convert to Decision object
                    if legacy_decision and legacy_decision['action'] == 'open':
                        decision_obj = Decision(
                            action=legacy_decision['action'],
                            side=legacy_decision['side'],
                            entry_price=legacy_decision['entry_price'],
                            size=legacy_decision['size'],
                            sl_price=legacy_decision['sl_price'],
                            tp_price=legacy_decision['tp_price'],
                            decision_source=legacy_decision['decision_source'],
                            regime=legacy_decision['regime']
                        )
                
                if decision_obj and decision_obj.action == 'open':
                    # Open position
                    broker.open_position(
                        symbol=symbol.value,
                        side=decision_obj.side,
                        entry_price=decision_obj.entry_price,
                        size=decision_obj.size,
                        sl_price=decision_obj.sl_price,
                        tp_price=decision_obj.tp_price,
                        timestamp=timestamp,
                        metadata={
                            'decision_source': decision_obj.decision_source,
                            'regime': decision_obj.regime,
                            'confidence': decision_obj.confidence
                        }
                    )
    
    # Close any remaining open positions at end
    for symbol_str in list(broker.positions.keys()):
        try:
            symbol = Symbol(symbol_str)
        except ValueError:
            logger.warning(f"Unknown symbol in positions: {symbol_str}")
            continue
        
        df = store.load_ohlcv(symbol, Timeframe.M15, start_date, end_date)
        if not df.empty:
            df = df.reset_index()
            last_candle = df.iloc[-1]
            broker.close_position(
                symbol=symbol_str,
                exit_price=last_candle['close'],
                timestamp=last_candle['timestamp'] if 'timestamp' in last_candle else last_candle.name,
                reason='END_OF_BACKTEST'
            )
    
    # Get results
    trades = broker.get_trades()
    equity_history = broker.get_equity_curve()
    
    # Convert equity history to list of dicts
    equity_curve = []
    for ts, equity in equity_history:
        equity_curve.append({
            'timestamp': ts.isoformat(),
            'equity': equity,
            'drawdown': 0.0  # Will calculate below
        })
    
    # Calculate drawdown for equity curve
    if equity_curve:
        equities = [e['equity'] for e in equity_curve]
        running_max = np.maximum.accumulate(equities)
        for i, e in enumerate(equity_curve):
            e['drawdown'] = (e['equity'] - running_max[i]) / running_max[i] if running_max[i] > 0 else 0.0
            e['drawdown_pct'] = e['drawdown'] * 100
    
    # Calculate metrics
    metrics = calculate_metrics(trades, equity_curve, initial_capital, total_candles)
    
    # Calculate breakdowns
    regime_breakdown, decision_breakdown = calculate_breakdowns(trades)
    
    execution_time = time.time() - start_time
    
    logger.info(f"[SCOPUS BACKTEST] Completed in {execution_time:.2f}s")
    logger.info(f"  Total trades: {metrics['total_trades']}")
    logger.info(f"  Winrate: {metrics['winrate']*100:.1f}%")
    logger.info(f"  Total return: {metrics['total_return']*100:.2f}%")
    logger.info(f"  Max drawdown: {metrics['max_drawdown']*100:.2f}%")
    logger.info(f"  Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
    
    # Create result object
    result = BacktestResult(
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        symbols=symbols,
        initial_capital=initial_capital,
        final_equity=broker.cash,
        total_trades=metrics['total_trades'],
        winning_trades=metrics['winning_trades'],
        losing_trades=metrics['losing_trades'],
        winrate=metrics['winrate'],
        total_pnl=metrics['total_pnl'],
        total_return=metrics['total_return'],
        max_drawdown=metrics['max_drawdown'],
        sharpe_ratio=metrics['sharpe_ratio'],
        sortino_ratio=metrics['sortino_ratio'],
        profit_factor=metrics['profit_factor'],
        avg_r_multiple=metrics['avg_r_multiple'],
        avg_trade_duration_minutes=metrics['avg_trade_duration_minutes'],
        exposure_pct=metrics['exposure_pct'],
        trades=trades,
        equity_curve=equity_curve,
        regime_breakdown=regime_breakdown,
        decision_source_breakdown=decision_breakdown,
        execution_time=execution_time,
        total_candles=total_candles,
        config={
            'enable_meta_gating': enable_meta_gating,
            'enable_portfolio_brain': enable_portfolio_brain,
            'enable_slicer': enable_slicer,
            'enable_rl_fallback': enable_rl_fallback
        }
    )
    
    # Save outputs
    save_backtest_outputs(result, output_dir)
    
    return result


def save_backtest_outputs(result: BacktestResult, output_dir: str):
    """Save backtest results to files"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save trades.csv
    if result.trades:
        trades_df = pd.DataFrame(result.trades)
        trades_csv = os.path.join(output_dir, 'trades.csv')
        trades_df.to_csv(trades_csv, index=False)
        logger.info(f"Saved {len(result.trades)} trades to {trades_csv}")
    
    # Save equity.csv
    if result.equity_curve:
        equity_df = pd.DataFrame(result.equity_curve)
        equity_csv = os.path.join(output_dir, 'equity.csv')
        equity_df.to_csv(equity_csv, index=False)
        logger.info(f"Saved equity curve to {equity_csv}")
    
    # Save summary.json
    summary = {
        'start_date': result.start_date,
        'end_date': result.end_date,
        'symbols': result.symbols,
        'initial_capital': result.initial_capital,
        'final_equity': result.final_equity,
        'total_trades': result.total_trades,
        'winning_trades': result.winning_trades,
        'losing_trades': result.losing_trades,
        'winrate': result.winrate,
        'total_return': result.total_return,
        'max_drawdown': result.max_drawdown,
        'sharpe_ratio': result.sharpe_ratio,
        'sortino_ratio': result.sortino_ratio,
        'profit_factor': result.profit_factor,
        'avg_r_multiple': result.avg_r_multiple,
        'avg_trade_duration_minutes': result.avg_trade_duration_minutes,
        'exposure_pct': result.exposure_pct,
        'regime_breakdown': result.regime_breakdown,
        'decision_source_breakdown': result.decision_source_breakdown,
        'execution_time': result.execution_time,
        'config': result.config
    }
    
    summary_json = os.path.join(output_dir, 'summary.json')
    with open(summary_json, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary to {summary_json}")
    
    # Print FINAL_METRICS for instant SSE streaming (parsed by async_runner)
    # This eliminates disk I/O latency after subprocess completes
    final_metrics = {
        'total_trades': result.total_trades,
        'winrate': result.winrate,
        'total_pnl': result.final_equity - result.initial_capital,
        'final_equity': result.final_equity,
        'sharpe_ratio': result.sharpe_ratio,
        'max_drawdown': result.max_drawdown,
        'profit_factor': result.profit_factor,
        # Include top 50 trades for chart markers (reduces API call)
        'trades': result.trades[:50] if result.trades else []
    }
    print(f"FINAL_METRICS:{json.dumps(final_metrics)}", flush=True)


if __name__ == "__main__":
    import pytz
    parser = argparse.ArgumentParser(description="SCOPUS Backtest Engine")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--symbols", type=str, default="EURUSD", help="Comma-separated symbols")
    parser.add_argument("--start", type=str, default="2023-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2023-01-07", help="End date (YYYY-MM-DD)")
    parser.add_argument("--capital", type=float, default=10000.0, help="Initial capital")
    parser.add_argument("--output", type=str, default="backtest_results", help="Output directory")
    
    args = parser.parse_args()
    
    # Parse config
    config = PipelineConfigV2() # Default
    # TODO: Load config from file if provided
    
    symbols = args.symbols.split(",")
    # Parse dates with UTC timezone
    start_date = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=pytz.UTC)
    end_date = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=pytz.UTC)
    
    print(f"[CLI] Starting backtest: {symbols}, {start_date} to {end_date}", flush=True)
    
    run_backtest(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        initial_capital=args.capital,
        output_dir=args.output,
        use_real_pipeline=True,
        csv_price_dir="data/raw/forex_kaggle_multiTF"
    )
