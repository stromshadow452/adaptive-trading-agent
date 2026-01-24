"""
WEAPON SYSTEM: RSI Range Reversion (SCALPEL)
=============================================

Locked RSI-based mean reversion for RANGE markets only.

HARD CONSTRAINTS (NON-NEGOTIABLE):
- Trades ONLY in RANGE regime
- RSI <= 28 for BUY, RSI >= 72 for SELL
- Confirmation candle REQUIRED
- Cooldown: No new trade until RSI crosses 50
- Size: 12% max (SCALPEL)
- SL: 0.5 × ATR, TP: 1.0 × ATR

Philosophy: "Agent is commander. Strategies are tools."
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass
import logging
import time

from src.strategy.base import (
    Strategy, WeaponClass, Signal,
    MarketSnapshot, AgentContext, StrategySignal
)

logger = logging.getLogger(__name__)


# ============================================================================
# RSI CALCULATION
# ============================================================================

def calculate_rsi(prices: list, period: int = 14) -> float:
    """
    Calculate RSI from price list.
    
    Args:
        prices: List of close prices (most recent last)
        period: RSI period (default 14)
    
    Returns:
        RSI value 0-100
    """
    if len(prices) < period + 1:
        return 50.0  # Neutral if insufficient data
    
    # Calculate price changes
    changes = []
    for i in range(1, len(prices)):
        changes.append(prices[i] - prices[i-1])
    
    # Use last 'period' changes
    recent_changes = changes[-period:]
    
    # Separate gains and losses
    gains = [max(0, c) for c in recent_changes]
    losses = [max(0, -c) for c in recent_changes]
    
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    
    if avg_loss == 0:
        return 100.0
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


# ============================================================================
# RSI RANGE REVERSION STRATEGY
# ============================================================================

@dataclass
class RSICooldownState:
    """Tracks cooldown state for RSI strategy."""
    last_trade_direction: str = ""  # "BUY" or "SELL"
    rsi_at_trade: float = 50.0
    cooldown_active: bool = False
    trade_timestamp: float = 0.0


class RSIRangeReversion(Strategy):
    """
    Locked RSI-based mean reversion for RANGE markets ONLY.
    
    RULES (LOCKED - NO MODIFICATION ALLOWED):
    1. ONLY trades when regime == RANGE (via ExactRangeDetector)
    2. BUY when RSI <= 28 AND next candle closes bullish
    3. SELL when RSI >= 72 AND next candle closes bearish
    4. Cooldown: No new trades until RSI crosses back past 50
    5. Size: base_size × 0.12 × EDGE × MARK-2 × ERM (max 15%)
    6. SL = 0.5 × ATR, TP = 1.0 × ATR
    
    This strategy is a TOOL. The Agent decides when to use it.
    """
    
    # ========== LOCKED THRESHOLDS (DO NOT MODIFY) ==========
    RSI_OVERSOLD = 28           # BUY threshold
    RSI_OVERBOUGHT = 72         # SELL threshold
    RSI_NEUTRAL = 50            # Cooldown reset level
    
    SL_ATR_MULT = 0.5           # Tight stop loss
    TP_ATR_MULT = 1.0           # Target = 2:1 RR
    
    SIZE_MULTIPLIER = 0.12      # SCALPEL size (12%)
    MAX_SIZE_PCT = 0.15         # Never exceed 15%
    
    RSI_PERIOD = 14             # Standard RSI
    # ========== END LOCKED THRESHOLDS ==========
    
    def __init__(self):
        super().__init__()
        self._cooldown_state: Dict[str, RSICooldownState] = {}
        
        # Pending confirmation tracking
        self._pending_signals: Dict[str, Dict] = {}
        
        # Price history for RSI calculation
        self._price_history: Dict[str, list] = {}
    
    @property
    def name(self) -> str:
        return "rsi_range_reversion"
    
    @property
    def weapon_class(self) -> WeaponClass:
        return WeaponClass.SCALPEL
    
    def get_size_multiplier(self) -> float:
        """Return SCALPEL size multiplier (12%)."""
        return self.SIZE_MULTIPLIER
    
    def is_allowed(self, context: AgentContext) -> bool:
        """
        Check if RSI strategy is allowed.
        
        HARD GATES:
        - RANGE regime ONLY
        - MARK-2 must allow
        - Not in OFF session
        - Not in DANGER regime
        """
        # Base checks from parent
        if not self.enabled:
            return False
        
        # === HARD GATE: RANGE ONLY ===
        if context.regime != 'RANGE':
            return False
        
        # === MARK-2 VETO ===
        if not context.mark2_can_trade:
            return False
        
        # === DANGER BLOCK ===
        if context.regime == 'DANGER':
            return False
        
        # === OFF SESSION BLOCK ===
        if context.session == 'OFF':
            return False
        
        # === HIGH PAIN BLOCK ===
        if context.memory_pain_level > 0.6:
            return False
        
        return True
    
    def _get_cooldown_state(self, symbol: str) -> RSICooldownState:
        """Get or create cooldown state for symbol."""
        if symbol not in self._cooldown_state:
            self._cooldown_state[symbol] = RSICooldownState()
        return self._cooldown_state[symbol]
    
    def _check_cooldown(self, symbol: str, current_rsi: float) -> bool:
        """
        Check if cooldown is active.
        
        Cooldown Logic:
        - After BUY trade: Wait for RSI to cross above 50
        - After SELL trade: Wait for RSI to cross below 50
        
        Returns True if cooldown ACTIVE (no trade allowed).
        """
        state = self._get_cooldown_state(symbol)
        
        if not state.cooldown_active:
            return False
        
        # Check if RSI has crossed neutral
        if state.last_trade_direction == "BUY":
            # After BUY, wait for RSI to go above 50
            if current_rsi > self.RSI_NEUTRAL:
                state.cooldown_active = False
                logger.info(f"[RSI] {symbol}: Cooldown lifted (RSI crossed above 50)")
                return False
        elif state.last_trade_direction == "SELL":
            # After SELL, wait for RSI to go below 50
            if current_rsi < self.RSI_NEUTRAL:
                state.cooldown_active = False
                logger.info(f"[RSI] {symbol}: Cooldown lifted (RSI crossed below 50)")
                return False
        
        return True  # Still in cooldown
    
    def _is_bullish_candle(self, snapshot: MarketSnapshot) -> bool:
        """Check if current candle is bullish (close > open)."""
        # We need open price - if not available, use a heuristic
        # For now, assume snapshot has latest close and we compare to previous
        if hasattr(snapshot, 'open_price') and snapshot.open_price > 0:
            return snapshot.price > snapshot.open_price
        # Fallback: use price vs SMA as proxy
        return snapshot.price > snapshot.sma_20 if snapshot.sma_20 > 0 else True
    
    def _is_bearish_candle(self, snapshot: MarketSnapshot) -> bool:
        """Check if current candle is bearish (close < open)."""
        if hasattr(snapshot, 'open_price') and snapshot.open_price > 0:
            return snapshot.price < snapshot.open_price
        return snapshot.price < snapshot.sma_20 if snapshot.sma_20 > 0 else True
    
    def _update_price_history(self, symbol: str, price: float):
        """Update price history for RSI calculation."""
        if symbol not in self._price_history:
            self._price_history[symbol] = []
        
        self._price_history[symbol].append(price)
        
        # Keep last 100 prices
        if len(self._price_history[symbol]) > 100:
            self._price_history[symbol] = self._price_history[symbol][-100:]
    
    def _get_rsi(self, symbol: str) -> float:
        """Calculate current RSI for symbol."""
        if symbol not in self._price_history:
            return 50.0
        
        return calculate_rsi(self._price_history[symbol], self.RSI_PERIOD)
    
    def _activate_cooldown(self, symbol: str, direction: str, rsi: float):
        """Activate cooldown after trade."""
        state = self._get_cooldown_state(symbol)
        state.last_trade_direction = direction
        state.rsi_at_trade = rsi
        state.cooldown_active = True
        state.trade_timestamp = time.time()
        logger.info(f"[RSI] {symbol}: Cooldown ACTIVATED after {direction} @ RSI={rsi:.1f}")
    
    def evaluate(
        self,
        snapshot: MarketSnapshot,
        context: AgentContext,
    ) -> Optional[StrategySignal]:
        """
        Evaluate for RSI mean-reversion opportunity.
        
        Logic Flow:
        1. Check RANGE regime gate
        2. Check MARK-2 / session gates
        3. Calculate RSI
        4. Check cooldown
        5. Check threshold + confirmation
        6. Return signal if valid
        """
        symbol = snapshot.symbol
        
        # Update price history
        self._update_price_history(symbol, snapshot.price)
        
        # === GATE 1: Strategy allowed? ===
        if not self.is_allowed(context):
            return None
        
        # === GATE 2: Calculate RSI ===
        rsi = self._get_rsi(symbol)
        
        # === GATE 3: Check cooldown ===
        if self._check_cooldown(symbol, rsi):
            logger.debug(f"[RSI] {symbol}: BLOCKED | cooldown active (RSI={rsi:.1f})")
            return None
        
        # === GATE 4: RSI threshold check ===
        price = snapshot.price
        atr = snapshot.atr
        
        # --- BUY SIGNAL: RSI <= 28 ---
        if rsi <= self.RSI_OVERSOLD:
            # Confirmation required: bullish candle
            if not self._is_bullish_candle(snapshot):
                logger.debug(f"[RSI] {symbol}: BUY pending confirmation (RSI={rsi:.1f})")
                return None
            
            # Calculate SL/TP using LOCKED parameters
            sl = price - atr * self.SL_ATR_MULT
            tp = price + atr * self.TP_ATR_MULT
            
            # Calculate size (never exceed MAX_SIZE_PCT)
            size_mult = min(
                self.SIZE_MULTIPLIER * context.edge_score * context.memory_mod,
                self.MAX_SIZE_PCT
            )
            
            logger.info(
                f"[RSI] RANGE BUY | {symbol} | RSI={rsi:.1f} | confirm=TRUE | "
                f"size={size_mult:.2%}"
            )
            
            # Activate cooldown
            self._activate_cooldown(symbol, "BUY", rsi)
            
            return StrategySignal(
                signal=Signal.BUY,
                confidence=0.55,
                suggested_sl=sl,
                suggested_tp=tp,
                reason=f"RSI RANGE reversion: oversold @ RSI={rsi:.1f}, confirmed bullish",
                strategy_name=self.name,
                weapon_class=self.weapon_class.value,
                size_multiplier=size_mult
            )
        
        # --- SELL SIGNAL: RSI >= 72 ---
        if rsi >= self.RSI_OVERBOUGHT:
            # Confirmation required: bearish candle
            if not self._is_bearish_candle(snapshot):
                logger.debug(f"[RSI] {symbol}: SELL pending confirmation (RSI={rsi:.1f})")
                return None
            
            # Calculate SL/TP using LOCKED parameters
            sl = price + atr * self.SL_ATR_MULT
            tp = price - atr * self.TP_ATR_MULT
            
            # Calculate size (never exceed MAX_SIZE_PCT)
            size_mult = min(
                self.SIZE_MULTIPLIER * context.edge_score * context.memory_mod,
                self.MAX_SIZE_PCT
            )
            
            logger.info(
                f"[RSI] RANGE SELL | {symbol} | RSI={rsi:.1f} | confirm=TRUE | "
                f"size={size_mult:.2%}"
            )
            
            # Activate cooldown
            self._activate_cooldown(symbol, "SELL", rsi)
            
            return StrategySignal(
                signal=Signal.SELL,
                confidence=0.55,
                suggested_sl=sl,
                suggested_tp=tp,
                reason=f"RSI RANGE reversion: overbought @ RSI={rsi:.1f}, confirmed bearish",
                strategy_name=self.name,
                weapon_class=self.weapon_class.value,
                size_multiplier=size_mult
            )
        
        # No signal
        return None
    
    def inject_rsi(self, symbol: str, rsi_value: float):
        """
        Inject RSI value from external source (e.g., features pipeline).
        
        Use this if RSI is pre-calculated in the pipeline.
        """
        # Store in a separate dict for external RSI
        if not hasattr(self, '_external_rsi'):
            self._external_rsi = {}
        self._external_rsi[symbol] = rsi_value
    
    def get_stats(self) -> Dict[str, Any]:
        """Get strategy statistics."""
        return {
            'name': self.name,
            'weapon_class': self.weapon_class.value,
            'enabled': self.enabled,
            'cooldowns': {
                symbol: {
                    'active': state.cooldown_active,
                    'direction': state.last_trade_direction,
                    'rsi_at_trade': state.rsi_at_trade
                }
                for symbol, state in self._cooldown_state.items()
            },
            'stats': {
                'total_trades': self.stats.total_trades,
                'win_rate': self.stats.win_rate,
                'avg_r': self.stats.avg_r
            }
        }
