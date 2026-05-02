"""
SCOPUS Backtest Broker - In-Memory Position Tracking

Simulates broker for backtesting with:
- Position tracking (open/close)
- SL/TP hit detection
- Equity management
- Trade recording
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


def _normalize_side(side: str) -> Optional[str]:
    """Normalize side to broker-safe lowercase values."""
    if side is None:
        return None
    normalized = str(side).strip().lower()
    if normalized in {"buy", "sell"}:
        return normalized
    return None


@dataclass
class Position:
    """Represents an open position"""
    symbol: str
    side: str  # 'buy' or 'sell'
    entry_price: float
    size: float
    sl_price: float
    tp_price: float
    entry_timestamp: datetime
    metadata: Dict  # decision_source, regime, etc.
    
    # Trailing tracking
    initial_sl_price: float = None
    max_r_multiple: float = 0.0
    trailing_active: bool = False
    last_trailing_r: float = 0.0
    locked_r_level: float = 0.0
    atr_trailing_active: bool = False
    
    # Track excursions
    max_favorable_price: float = None
    max_adverse_price: float = None
    
    def __post_init__(self):
        if self.max_favorable_price is None:
            self.max_favorable_price = self.entry_price
        if self.max_adverse_price is None:
            self.max_adverse_price = self.entry_price
        if self.initial_sl_price is None:
            self.initial_sl_price = self.sl_price
    
    def update_excursions(self, current_price: float):
        """Update MAE/MFE tracking"""
        if self.side == 'buy':
            # For long: favorable = higher, adverse = lower
            self.max_favorable_price = max(self.max_favorable_price, current_price)
            self.max_adverse_price = min(self.max_adverse_price, current_price)
        else:
            # For short: favorable = lower, adverse = higher
            self.max_favorable_price = min(self.max_favorable_price, current_price)
            self.max_adverse_price = max(self.max_adverse_price, current_price)
    
    def get_mfe(self) -> float:
        """Max Favorable Excursion in price units"""
        if self.side == 'buy':
            return self.max_favorable_price - self.entry_price
        else:
            return self.entry_price - self.max_favorable_price
    
    def get_mae(self) -> float:
        """Max Adverse Excursion in price units"""
        if self.side == 'buy':
            return self.entry_price - self.max_adverse_price
        else:
            return self.max_adverse_price - self.entry_price
    
    def get_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L"""
        if self.side == 'buy':
            return (current_price - self.entry_price) * self.size
        else:
            return (self.entry_price - current_price) * self.size


class BacktestBroker:
    """
    In-memory broker for backtesting.
    
    Handles:
    - Opening/closing positions
    - SL/TP hit detection
    - Equity tracking
    - Trade recording
    """
    
    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}  # {symbol: Position}
        self.closed_trades: List[Dict] = []
        self.equity_history: List[Tuple[datetime, float]] = []
        
        # Track first timestamp
        self.start_timestamp = None
    
    def open_position(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        size: float,
        sl_price: float,
        tp_price: float,
        timestamp: datetime,
        metadata: Dict
    ) -> bool:
        """
        Open a new position.
        
        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            entry_price: Entry price
            size: Position size (in lots/units)
            sl_price: Stop loss price
            tp_price: Take profit price
            timestamp: Entry timestamp
            metadata: Additional info (decision_source, regime, etc.)
        
        Returns:
            True if position opened successfully
        """
        if self.start_timestamp is None:
            self.start_timestamp = timestamp

        normalized_side = _normalize_side(side)
        if normalized_side is None:
            logger.warning(f"Invalid trade side for {symbol}: {side!r}")
            return False
        
        # Check if position already exists for this symbol
        if symbol in self.positions:
            logger.warning(f"Position already exists for {symbol}, skipping")
            return False
        
        # Create position
        position = Position(
            symbol=symbol,
            side=normalized_side,
            entry_price=entry_price,
            size=size,
            sl_price=sl_price,
            tp_price=tp_price,
            entry_timestamp=timestamp,
            metadata=metadata
        )
        
        self.positions[symbol] = position
        
        logger.debug(f"[BACKTEST] Opened {side} {symbol} @ {entry_price:.5f}, "
                    f"SL={sl_price:.5f}, TP={tp_price:.5f}")
        
        return True
    
    def update_positions(
        self,
        current_prices: Dict[str, Dict],
        timestamp: datetime
    ) -> List[Dict]:
        """
        Update all positions and check for SL/TP hits.
        
        Args:
            current_prices: {symbol: {'high': X, 'low': Y, 'close': Z}}
            timestamp: Current candle timestamp
        
        Returns:
            List of closed trades (if any SL/TP hit)
        """
        closed_this_update = []
        symbols_to_close = []
        
        for symbol, position in self.positions.items():
            if symbol not in current_prices:
                continue
            
            candle = current_prices[symbol]
            high = candle.get('high', candle.get('close'))
            low = candle.get('low', candle.get('close'))
            close = candle['close']
            
            # Update excursions with intrabar extremes so MFE/MAE reflect what the candle actually touched
            position.update_excursions(high if position.side == 'buy' else low)
            position.update_excursions(low if position.side == 'buy' else high)
            
            # --- Strict Exit Logic: structure lock first, ATR trail only after 2R ---
            initial_risk = abs(position.entry_price - position.initial_sl_price)
            if initial_risk > 0:
                locked_r_level = position.locked_r_level
                atr_value = abs(float(position.metadata.get('atr_value', 0.0) or 0.0))
                if position.side == 'buy':
                    # Step 1-2: intrabar R and max R tracking for longs.
                    current_r = (high - position.entry_price) / initial_risk
                    position.max_r_multiple = max(position.max_r_multiple, current_r)

                    # Step 3: structure lock.
                    if position.max_r_multiple >= 1.8:
                        position.trailing_active = True
                        position.sl_price = max(position.sl_price, position.entry_price)
                        locked_r_level = max(locked_r_level, 0.0)
                    if position.max_r_multiple >= 2.0:
                        position.sl_price = max(position.sl_price, position.entry_price + initial_risk)
                        locked_r_level = max(locked_r_level, 1.0)
                        position.atr_trailing_active = True
                    if position.max_r_multiple >= 3.0:
                        position.sl_price = max(position.sl_price, position.entry_price + 2.0 * initial_risk)
                        locked_r_level = max(locked_r_level, 2.0)

                    # Step 4-5: ATR trailing activates only after 2R and can only tighten.
                    if position.max_r_multiple >= 2.0 and atr_value > 0:
                        position.atr_trailing_active = True
                        atr_buffer = atr_value * 2.0
                        dynamic_sl = close - atr_buffer
                        position.sl_price = max(position.sl_price, dynamic_sl)
                else:  # short
                    # Step 1-2: intrabar R and max R tracking for shorts.
                    current_r = (position.entry_price - low) / initial_risk
                    position.max_r_multiple = max(position.max_r_multiple, current_r)

                    # Step 3: structure lock.
                    if position.max_r_multiple >= 1.8:
                        position.trailing_active = True
                        position.sl_price = min(position.sl_price, position.entry_price)
                        locked_r_level = max(locked_r_level, 0.0)
                    if position.max_r_multiple >= 2.0:
                        position.sl_price = min(position.sl_price, position.entry_price - initial_risk)
                        locked_r_level = max(locked_r_level, 1.0)
                        position.atr_trailing_active = True
                    if position.max_r_multiple >= 3.0:
                        position.sl_price = min(position.sl_price, position.entry_price - 2.0 * initial_risk)
                        locked_r_level = max(locked_r_level, 2.0)

                    # Step 4-5: ATR trailing activates only after 2R and can only tighten.
                    if position.max_r_multiple >= 2.0 and atr_value > 0:
                        position.atr_trailing_active = True
                        atr_buffer = atr_value * 2.0
                        dynamic_sl = close + atr_buffer
                        position.sl_price = min(position.sl_price, dynamic_sl)

                position.locked_r_level = locked_r_level
                position.last_trailing_r = locked_r_level
                position.metadata['atr_trailing_active'] = position.atr_trailing_active
                logger.debug(
                    f"[TRAIL] {symbol}: side={position.side} max_r_multiple={position.max_r_multiple:.2f} "
                    f"sl_price={position.sl_price:.5f} locked_r_level={position.locked_r_level:.1f} "
                    f"atr_trailing_active={position.atr_trailing_active}"
                )
            
            # Check SL/TP hits
            sl_hit = False
            tp_hit = False
            
            if position.side == 'buy':
                # Long position
                if low <= position.sl_price:
                    sl_hit = True
                    exit_price = position.sl_price
                    if position.sl_price >= position.entry_price:
                        exit_reason = 'TRAILING_PROFIT'
                        exit_type = 'PROFIT_LOCK'
                    else:
                        exit_reason = 'SL_HIT'
                        exit_type = 'LOSS'
                elif high >= position.tp_price and not position.trailing_active:
                    tp_hit = True
                    exit_price = position.tp_price
                    exit_reason = 'TP_HIT'
                    exit_type = 'PROFIT'
            else:
                # Short position
                if high >= position.sl_price:
                    sl_hit = True
                    exit_price = position.sl_price
                    if position.sl_price <= position.entry_price:
                        exit_reason = 'TRAILING_PROFIT'
                        exit_type = 'PROFIT_LOCK'
                    else:
                        exit_reason = 'SL_HIT'
                        exit_type = 'LOSS'
                elif low <= position.tp_price and not position.trailing_active:
                    tp_hit = True
                    exit_price = position.tp_price
                    exit_reason = 'TP_HIT'
                    exit_type = 'PROFIT'
            
            if sl_hit or tp_hit:
                # Close position
                trade = self._close_position_internal(
                    symbol=symbol,
                    exit_price=exit_price,
                    exit_timestamp=timestamp,
                    exit_reason=exit_reason,
                    exit_type=exit_type
                )
                closed_this_update.append(trade)
                symbols_to_close.append(symbol)
        
        # Remove closed positions
        for symbol in symbols_to_close:
            del self.positions[symbol]
        
        # Update equity history
        equity = self.get_equity(current_prices)
        self.equity_history.append((timestamp, equity))
        
        return closed_this_update
    
    def close_position(
        self,
        symbol: str,
        exit_price: float,
        timestamp: datetime,
        reason: str = 'MANUAL'
    ) -> Optional[Dict]:
        """
        Manually close a position.
        
        Args:
            symbol: Symbol to close
            exit_price: Exit price
            timestamp: Exit timestamp
            reason: Exit reason
        
        Returns:
            Trade dict if closed, None if no position
        """
        if symbol not in self.positions:
            logger.warning(f"No position to close for {symbol}")
            return None
        
        trade = self._close_position_internal(symbol, exit_price, timestamp, reason)
        del self.positions[symbol]
        
        return trade
    
    def _close_position_internal(
        self,
        symbol: str,
        exit_price: float,
        exit_timestamp: datetime,
        exit_reason: str,
        exit_type: str = 'UNKNOWN'
    ) -> Dict:
        """Internal method to close position and record trade"""
        position = self.positions[symbol]
        
        # Calculate P&L
        if position.side == 'buy':
            pnl = (exit_price - position.entry_price) * position.size
        else:
            pnl = (position.entry_price - exit_price) * position.size
        
        # Calculate percentage P&L
        pnl_pct = (pnl / (position.entry_price * position.size)) * 100
        
        # Calculate R-multiple
        risk = abs(position.entry_price - position.sl_price) * position.size
        r_multiple = pnl / risk if risk > 0 else 0.0
        
        # Calculate duration
        duration = exit_timestamp - position.entry_timestamp
        duration_minutes = duration.total_seconds() / 60
        
        # Get MAE/MFE
        mfe = position.get_mfe()
        mae = position.get_mae()
        
        # Create trade record
        trade = {
            'timestamp_entry': position.entry_timestamp.isoformat(),
            'timestamp_exit': exit_timestamp.isoformat(),
            'symbol': symbol,
            'side': position.side,
            'entry_price': position.entry_price,
            'exit_price': exit_price,
            'size': position.size,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'r_multiple': r_multiple,
            'max_r_multiple_reached': position.max_r_multiple,
            'locked_r_level': position.locked_r_level,
            'trailing_active': position.trailing_active,
            'final_sl_price': position.sl_price,
            'sl_price': position.sl_price,
            'tp_price': position.tp_price,
            'exit_reason': exit_reason,
            'exit_type': exit_type,
            'decision_source': position.metadata.get('decision_source', 'UNKNOWN'),
            'strategy': position.metadata.get('strategy', 'UNKNOWN'),
            'regime': position.metadata.get('regime', 'UNKNOWN'),
            'regime_reason': position.metadata.get('regime_reason'),
            'ml_conf_raw': position.metadata.get('ml_conf_raw'),
            'ml_conf_calibrated': position.metadata.get('ml_conf_calibrated'),
            'edge_score': position.metadata.get('edge_score'),
            'skip_reason': position.metadata.get('skip_reason'),
            'cooldown_active': position.metadata.get('cooldown_active', False),
            'soft_filter_score': position.metadata.get('soft_filter_score'),
            'signal_quality': position.metadata.get('signal_quality'),
            'probability_of_success': position.metadata.get('probability_of_success'),
            'ml_filter_pass': position.metadata.get('ml_filter_pass'),
            'ml_filter_enabled': position.metadata.get('ml_filter_enabled'),
            'shadow_logged': position.metadata.get('shadow_logged', False),
            'virtual_r_multiple': position.metadata.get('virtual_r_multiple'),
            'atr_trailing_active': position.atr_trailing_active or position.metadata.get('atr_trailing_active', False),
            'boll_z': position.metadata.get('boll_z'),
            'atr_pctile': position.metadata.get('atr_pctile'),
            'adx': position.metadata.get('adx'),
            'session': position.metadata.get('session'),
            'atr_value': position.metadata.get('atr_value'),
            'neutral_regime_size_reduction': position.metadata.get('neutral_regime_size_reduction', False),
            'danger_size_reduction': position.metadata.get('danger_size_reduction', False),
            'danger_reason': position.metadata.get('danger_reason'),
            'danger_duration_bars': position.metadata.get('danger_duration_bars', 0),
            'mfe': mfe,
            'mae': mae,
            'duration_minutes': int(duration_minutes),
            'duration_bars': position.metadata.get('duration_bars', 0)
        }
        
        self.closed_trades.append(trade)
        self.cash += pnl
        
        # CRITICAL: Validate trade sanity BEFORE continuing
        try:
            from .trade_sanity import validate_trade
            validate_trade(trade)
        except ImportError:
            pass  # Sanity module not installed
        
        logger.debug(f"[BACKTEST] Closed {position.side} {symbol} @ {exit_price:.5f}, "
                    f"PnL=${pnl:.2f} ({pnl_pct:.2f}%), R={r_multiple:.2f}, "
                    f"Reason={exit_reason}")
        
        return trade
    
    def get_equity(self, current_prices: Dict[str, Dict] = None) -> float:
        """
        Get current equity (cash + unrealized P&L).
        
        Args:
            current_prices: {symbol: {'close': X}}
        
        Returns:
            Total equity
        """
        equity = self.cash
        
        if current_prices:
            for symbol, position in self.positions.items():
                if symbol in current_prices:
                    current_price = current_prices[symbol]['close']
                    unrealized_pnl = position.get_unrealized_pnl(current_price)
                    equity += unrealized_pnl
        
        return equity
    
    def get_trades(self) -> List[Dict]:
        """Get all closed trades"""
        return self.closed_trades.copy()
    
    def get_equity_curve(self) -> List[Tuple[datetime, float]]:
        """Get equity history"""
        return self.equity_history.copy()
    
    def get_summary(self) -> Dict:
        """Get summary statistics"""
        if not self.closed_trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'winrate': 0.0,
                'total_pnl': 0.0,
                'final_equity': self.initial_capital
            }
        
        pnls = [t['pnl'] for t in self.closed_trades]
        winning = sum(1 for p in pnls if p > 0)
        losing = sum(1 for p in pnls if p < 0)
        
        return {
            'total_trades': len(self.closed_trades),
            'winning_trades': winning,
            'losing_trades': losing,
            'winrate': winning / len(self.closed_trades) if self.closed_trades else 0.0,
            'total_pnl': sum(pnls),
            'final_equity': self.cash
        }
