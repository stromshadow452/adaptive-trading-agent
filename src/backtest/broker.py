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
    
    # Track excursions
    max_favorable_price: float = None
    max_adverse_price: float = None
    
    def __post_init__(self):
        if self.max_favorable_price is None:
            self.max_favorable_price = self.entry_price
        if self.max_adverse_price is None:
            self.max_adverse_price = self.entry_price
    
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
        
        # Check if position already exists for this symbol
        if symbol in self.positions:
            logger.warning(f"Position already exists for {symbol}, skipping")
            return False
        
        # Create position
        position = Position(
            symbol=symbol,
            side=side,
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
            
            # Update excursions with current close
            position.update_excursions(close)
            
            # Check SL/TP hits
            sl_hit = False
            tp_hit = False
            
            if position.side == 'buy':
                # Long position
                if low <= position.sl_price:
                    sl_hit = True
                    exit_price = position.sl_price
                    exit_reason = 'SL_HIT'
                elif high >= position.tp_price:
                    tp_hit = True
                    exit_price = position.tp_price
                    exit_reason = 'TP_HIT'
            else:
                # Short position
                if high >= position.sl_price:
                    sl_hit = True
                    exit_price = position.sl_price
                    exit_reason = 'SL_HIT'
                elif low <= position.tp_price:
                    tp_hit = True
                    exit_price = position.tp_price
                    exit_reason = 'TP_HIT'
            
            if sl_hit or tp_hit:
                # Close position
                trade = self._close_position_internal(
                    symbol=symbol,
                    exit_price=exit_price,
                    exit_timestamp=timestamp,
                    exit_reason=exit_reason
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
        exit_reason: str
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
            'sl_price': position.sl_price,
            'tp_price': position.tp_price,
            'exit_reason': exit_reason,
            'decision_source': position.metadata.get('decision_source', 'UNKNOWN'),
            'regime': position.metadata.get('regime', 'UNKNOWN'),
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
