"""
SCOPUS Adaptive Risk Brain V2 - Fibonacci Edition

Execution-first, adaptive risk management with:
- Fibonacci-based RR thresholds (1.13, 1.27, 1.41, 1.62, 2.00)
- Degradative position sizing with Fib deltas
- Confidence-based RR forgiveness (step down one Fib level)
- Exploration override (never block learning trades)
- ATR-based SL/TP

CORE PHILOSOPHY: RISK SHOULD DEGRADE, NOT BLOCK.
Blocking is LAST RESORT.

WHY FIBONACCI RATIOS?
- Non-linear confidence scaling matches market asymmetry
- Natural mathematical relationship mirrors price behavior
- Prevents abrupt decision cliffs (smooth degradation)
- Golden ratio (1.618) aligns with market structure patterns
"""

import pandas as pd
from typing import Dict, Optional, TYPE_CHECKING
from dataclasses import dataclass
import logging

if TYPE_CHECKING:
    from src.backtest.adaptive_state import AdaptiveState

# Optional: Size Ladder for Risk Evolution
try:
    from src.risk.size_ladder import get_size_ladder, SizeLadderManager
    HAS_SIZE_LADDER = True
except ImportError:
    HAS_SIZE_LADDER = False

logger = logging.getLogger(__name__)


# ============================================================================
# FIBONACCI CONSTANTS
# ============================================================================
FIB_LEVELS = [1.13, 1.27, 1.41, 1.62, 2.00]  # Ordered Fib RR anchors
FIB_DELTA_SMALL = 0.14  # One Fib step for degradation
FIB_DELTA_LARGE = 0.28  # Two Fib steps for degradation


# ============================================================================
# ADAPTIVE MIN RR TABLE (FIBONACCI-BASED)
# ============================================================================
MIN_RR_TABLE = {
    # LEARNING mode - Lower thresholds for exploration
    ("LEARNING", "RANGE"): 1.27,    # Fib: ~61.8% extension
    ("LEARNING", "TREND"): 1.41,    # Fib: √2 ratio
    ("LEARNING", "DANGER"): 1.13,   # Fib: Very shallow for learning
    ("LEARNING", "UNCERTAIN"): 1.27,
    
    # CONFIRMATION mode - Higher thresholds for quality
    ("CONFIRMATION", "RANGE"): 1.41,
    ("CONFIRMATION", "TREND"): 1.62,  # Golden ratio
    ("CONFIRMATION", "DANGER"): 1.27,
    ("CONFIRMATION", "UNCERTAIN"): 1.41,
}

DEFAULT_MIN_RR = 1.41  # Default to √2 ratio


def fib_step_down(rr: float) -> float:
    """
    Step down one Fibonacci level for RR forgiveness.
    Example: 1.62 → 1.41, 1.41 → 1.27
    """
    for i, level in enumerate(FIB_LEVELS):
        if rr <= level:
            return FIB_LEVELS[max(0, i - 1)]
    return FIB_LEVELS[-2]  # If above max, step to second highest


@dataclass
class RiskDecision:
    """Structured risk decision output."""
    sl_price: float
    tp_price: float
    position_size: float
    rr: float
    risk_decision: str  # "FULL" | "REDUCED" | "BLOCK"
    risk_reason: str
    
    def to_dict(self) -> Dict:
        return {
            "sl_price": self.sl_price,
            "tp_price": self.tp_price,
            "position_size": self.position_size,
            "rr": self.rr,
            "risk_decision": self.risk_decision,
            "risk_reason": self.risk_reason
        }


class RiskBrainV1:
    """
    Adaptive Risk Brain V2 - Fibonacci Edition
    
    CORE PHILOSOPHY: RISK SHOULD DEGRADE, NOT BLOCK.
    
    Features:
    - Fibonacci-based adaptive RR thresholds
    - Degradative sizing with Fib deltas (0.14, 0.28)
    - Confidence-based RR forgiveness (step down one Fib)
    - Exploration override (min_rr = 1.27, never block)
    - ATR-based SL/TP calculation
    
    Blocking is ABSOLUTE LAST RESORT.
    """
    
    def __init__(self, config: dict = None):
        config = config or {}
        
        # Base parameters
        self.base_sl_atr_mult = config.get('base_sl_atr_mult', 2.0)
        self.base_tp_atr_mult = config.get('base_tp_atr_mult', 3.0)
        self.base_risk_pct = config.get('base_risk_pct', 0.01)  # 1% base
        
        # Hard caps
        self.min_risk_pct = config.get('min_risk_pct', 0.002)  # 0.2%
        self.max_risk_pct = config.get('max_risk_pct', 0.03)   # 3%
        
        # Regime multipliers for SL/TP width
        self.regime_multipliers = config.get('regime_multipliers', {
            'TREND': {'sl': 1.0, 'tp': 1.5},
            'RANGE': {'sl': 0.7, 'tp': 1.0},  # Keep TP at 1.5R even in RANGE
            'DANGER': {'sl': 0.5, 'tp': 0.5},
            'UNCERTAIN': {'sl': 0.8, 'tp': 0.9}
        })
        
        # Drawdown thresholds
        self.survival_threshold = config.get('survival_threshold', 0.20)
        self.caution_threshold = config.get('caution_threshold', 0.10)
        
        # Cooldown settings
        self.loss_streak_cooldown = config.get('loss_streak_cooldown', 3)
        
        # Exploration settings (Fibonacci-aligned)
        self.exploration_min_rr = config.get('exploration_min_rr', 1.27)  # Fib level
        self.exploration_size_cap = config.get('exploration_size_cap', 0.30)
        self.manipulation_block_threshold = config.get('manipulation_block_threshold', 0.70)
        
        # State
        self.recent_drawdown = 0.0
        self.recent_winrate = 0.5
        
        logger.info("RiskBrainV2-Fibonacci initialized - Degrade before Block")
    
    def calculate_sl_tp(
        self,
        entry_price: float,
        side: str,
        atr: float,
        regime: str,
        context: dict,
        adaptive_state: Optional['AdaptiveState'] = None,
        exploration_trade: bool = False,
        context_output: Optional[dict] = None
    ) -> Dict:
        """
        Calculate SL, TP, and position size with Fibonacci-based adaptive logic.
        
        PHILOSOPHY: Degrade size before blocking. Block only as last resort.
        
        Args:
            entry_price: Entry price
            side: 'BUY' or 'SELL'
            atr: ATR value
            regime: Market regime (RANGE/TREND/DANGER/UNCERTAIN)
            context: Trading context
            adaptive_state: Optional AdaptiveState for enhanced adaptation
            exploration_trade: If True, use exploration override
            context_output: Context Brain output with mode, manipulation_risk, etc.
        
        Returns:
            Dict with sl_price, tp_price, position_size, rr, risk_decision, risk_reason
        """
        # Normalize inputs
        side = side.upper() if isinstance(side, str) else 'BUY'
        regime = regime.upper() if isinstance(regime, str) else 'UNCERTAIN'
        
        # Get operating mode
        operating_mode = 'LEARNING'
        if context_output:
            operating_mode = context_output.get('operating_mode', 'LEARNING')
        
        # === STEP 1: Calculate base SL/TP ===
        sl_mult, tp_mult = self._get_multipliers(regime, context.get('volatility', 0.01))
        
        if atr <= 0 or atr < 0.00001:
            logger.warning(f"ATR too small: {atr}")
            return RiskDecision(
                sl_price=0.0,
                tp_price=0.0,
                position_size=0.0,
                rr=0.0,
                risk_decision="BLOCK",
                risk_reason="ATR too small for valid SL/TP"
            ).to_dict()
        
        if side == 'BUY':
            sl_price = entry_price - (atr * sl_mult)
            tp_price = entry_price + (atr * tp_mult)
        else:  # SELL
            sl_price = entry_price + (atr * sl_mult)
            tp_price = entry_price - (atr * tp_mult)
        
        # === STEP 2: Calculate RR ratio ===
        tp_distance = abs(tp_price - entry_price)
        sl_distance = abs(sl_price - entry_price)
        rr_ratio = tp_distance / sl_distance if sl_distance > 0 else 0.0
        
        context['rr_ratio'] = rr_ratio
        
        # === STEP 3: Get Fibonacci-based adaptive minimum RR ===
        min_rr = self._get_fibonacci_min_rr(
            operating_mode=operating_mode,
            regime=regime,
            ml_confidence=context.get('confidence', 0.5),
            context_output=context_output,
            exploration_trade=exploration_trade
        )
        
        # === STEP 4: Apply degradative sizing (Fibonacci steps) ===
        size_multiplier, risk_decision, risk_reason = self._apply_fib_degradative_sizing(
            rr_ratio=rr_ratio,
            min_rr=min_rr,
            operating_mode=operating_mode,
            exploration_trade=exploration_trade,
            context_output=context_output
        )
        
        # === STEP 5: Check for hard block ===
        if risk_decision == "BLOCK":
            context['rejected_reason'] = risk_reason
            logger.info(f"RiskBrain BLOCK: {risk_reason}")
            return RiskDecision(
                sl_price=sl_price,
                tp_price=tp_price,
                position_size=0.0,
                rr=rr_ratio,
                risk_decision="BLOCK",
                risk_reason=risk_reason
            ).to_dict()
        
        # === STEP 6: Calculate base position size ===
        risk_pct = self._calculate_adaptive_risk(
            confidence=context.get('confidence', 0.5),
            regime=regime,
            context=context,
            adaptive_state=adaptive_state
        )
        
        base_size = self._calculate_position_size(
            entry_price,
            sl_price,
            context.get('balance', 10000),
            risk_pct
        )
        
        # === STEP 7: Apply Fibonacci size multiplier ===
        position_size = base_size * size_multiplier
        
        # === STEP 8: Apply exploration cap ===
        if exploration_trade:
            position_size = min(position_size, base_size * self.exploration_size_cap)
            context['exploration_trade'] = True
            logger.info(f"EXPLORATION: size capped to {self.exploration_size_cap:.0%}")
        
        # === STEP 9: Apply learning mode cap ===
        if operating_mode == 'LEARNING' and not exploration_trade:
            learning_cap = 0.62  # Fibonacci ratio
            position_size = position_size * learning_cap
        
        context['risk_pct'] = risk_pct
        
        logger.info(
            f"RiskBrain {risk_decision}: RR={rr_ratio:.2f} vs min={min_rr:.2f}, "
            f"size_mult={size_multiplier:.2f}, final_size={position_size:.4f}"
        )
        
        return RiskDecision(
            sl_price=sl_price,
            tp_price=tp_price,
            position_size=position_size,
            rr=rr_ratio,
            risk_decision=risk_decision,
            risk_reason=risk_reason
        ).to_dict()
    
    def _get_fibonacci_min_rr(
        self,
        operating_mode: str,
        regime: str,
        ml_confidence: float,
        context_output: Optional[dict],
        exploration_trade: bool
    ) -> float:
        """
        Get Fibonacci-based adaptive minimum RR.
        
        Implements:
        1. MIN_RR table lookup (Fib-based)
        2. Confidence forgiveness (step down one Fib level)
        3. Exploration override (min_rr = 1.27)
        """
        # Exploration override - use 1.27 (low Fib level)
        if exploration_trade:
            return self.exploration_min_rr
        
        # Lookup from Fibonacci table
        key = (operating_mode, regime)
        base_min_rr = MIN_RR_TABLE.get(key, DEFAULT_MIN_RR)
        
        # === CONFIDENCE-BASED RR FORGIVENESS ===
        # If high quality signal, step down ONE Fibonacci level
        if context_output and ml_confidence >= 0.85:
            manipulation_risk = context_output.get('manipulation_risk', 0.5)
            sr_strength = context_output.get('sr_strength', 0.0)
            
            if manipulation_risk <= 0.4 and sr_strength >= 0.5:
                # Step down one Fib level
                forgiven_rr = fib_step_down(base_min_rr)
                logger.debug(f"Fib forgiveness: {base_min_rr:.2f} → {forgiven_rr:.2f}")
                base_min_rr = forgiven_rr
        
        return max(base_min_rr, 1.13)  # Never below lowest Fib
    
    def _apply_fib_degradative_sizing(
        self,
        rr_ratio: float,
        min_rr: float,
        operating_mode: str,
        exploration_trade: bool,
        context_output: Optional[dict]
    ) -> tuple:
        """
        Apply Fibonacci-based degradative sizing.
        
        Degradation steps:
        - RR >= min_rr           → FULL (1.0)
        - RR >= min_rr - 0.14    → REDUCED (0.62 - Fib ratio)
        - RR >= min_rr - 0.28    → REDUCED (0.38 - Fib ratio)
        - RR < min_rr - 0.28     → BLOCK (last resort)
        
        Returns:
            (size_multiplier, risk_decision, risk_reason)
        """
        # === EXPLORATION OVERRIDE (SACRED RULE) ===
        if exploration_trade:
            # Check manipulation hard-block
            if context_output:
                manipulation_risk = context_output.get('manipulation_risk', 0.0)
                if manipulation_risk > self.manipulation_block_threshold:
                    return 0.0, "BLOCK", f"Exploration blocked: manipulation={manipulation_risk:.2f} > {self.manipulation_block_threshold}"
            
            # Exploration trades NEVER blocked for RR
            if rr_ratio >= min_rr:
                return 1.0, "FULL", f"Exploration FULL: RR={rr_ratio:.2f} >= {min_rr:.2f}"
            elif rr_ratio >= min_rr - FIB_DELTA_SMALL:
                return 0.62, "REDUCED", f"Exploration REDUCED-62%: RR={rr_ratio:.2f}"
            else:
                # Still execute with minimum Fib ratio
                return 0.38, "REDUCED", f"Exploration REDUCED-38%: RR={rr_ratio:.2f}"
        
        # === STANDARD FIBONACCI DEGRADATION ===
        if rr_ratio >= min_rr:
            return 1.0, "FULL", f"FULL: RR={rr_ratio:.2f} >= {min_rr:.2f}"
        
        elif rr_ratio >= min_rr - FIB_DELTA_SMALL:
            # One Fib step down: 62% size
            return 0.62, "REDUCED", f"REDUCED-62%: RR={rr_ratio:.2f} >= {min_rr - FIB_DELTA_SMALL:.2f}"
        
        elif rr_ratio >= min_rr - FIB_DELTA_LARGE:
            # Two Fib steps down: 38% size
            return 0.38, "REDUCED", f"REDUCED-38%: RR={rr_ratio:.2f} >= {min_rr - FIB_DELTA_LARGE:.2f}"
        
        else:
            # BLOCK - Absolute last resort
            return 0.0, "BLOCK", f"BLOCK: RR={rr_ratio:.2f} < {min_rr - FIB_DELTA_LARGE:.2f} (min={min_rr:.2f})"
    
    def _calculate_adaptive_risk(
        self,
        confidence: float,
        regime: str,
        context: dict,
        adaptive_state: Optional['AdaptiveState'] = None
    ) -> float:
        """
        Calculate adaptive risk percentage based on conditions.
        Uses Fibonacci-aligned multipliers where possible.
        """
        risk = self.base_risk_pct
        
        if adaptive_state:
            vol_level = adaptive_state.volatility_level
            drawdown = adaptive_state.rolling_drawdown
            loss_streak = adaptive_state.loss_streak
            win_streak = adaptive_state.win_streak
        else:
            vol_level = context.get('volatility_level', 'MID')
            drawdown = context.get('drawdown', 0)
            loss_streak = context.get('loss_streak', 0)
            win_streak = context.get('win_streak', 0)
        
        # Confidence multiplier (Fib-aligned)
        conf_mult = 0.76 + (confidence - 0.5) * 0.48  # Range: 0.76 - 1.0
        conf_mult = max(0.76, min(1.0, conf_mult))
        
        # Volatility multiplier
        vol_mult = {'LOW': 1.0, 'MID': 0.85, 'HIGH': 0.62}.get(vol_level, 0.85)
        
        # Regime multiplier (Fib-aligned)
        regime_mult = {
            'TREND': 1.0,
            'RANGE': 0.85,
            'DANGER': 0.38,  # Fibonacci ratio
            'UNCERTAIN': 0.62  # Fibonacci ratio
        }.get(regime, 0.85)
        
        # Drawdown multiplier (survival mode)
        if drawdown >= self.survival_threshold:
            dd_mult = 0.23  # Fib-ish
            logger.warning(f"SURVIVAL MODE: DD={drawdown:.1%}")
        elif drawdown >= 0.15:
            dd_mult = 0.38
        elif drawdown >= self.caution_threshold:
            dd_mult = 0.62
        elif drawdown >= 0.05:
            dd_mult = 0.85
        else:
            dd_mult = 1.0
        
        # Streak multiplier
        if loss_streak >= self.loss_streak_cooldown:
            streak_mult = 0.38 - (loss_streak - self.loss_streak_cooldown) * 0.08
            streak_mult = max(0.15, streak_mult)
            logger.info(f"COOLDOWN: Loss streak {loss_streak}")
        elif win_streak >= 3:
            streak_mult = 1.0 + min(win_streak - 3, 3) * 0.05
            streak_mult = min(1.15, streak_mult)
        else:
            streak_mult = 1.0
        
        # Apply all multipliers
        risk = risk * conf_mult * vol_mult * regime_mult * dd_mult * streak_mult
        
        # Hard caps
        return max(self.min_risk_pct, min(self.max_risk_pct, risk))
    
    def _get_multipliers(self, regime: str, volatility: float) -> tuple:
        """Get SL/TP multipliers based on regime and volatility."""
        sl_mult = self.base_sl_atr_mult
        tp_mult = self.base_tp_atr_mult
        
        regime_mult = self.regime_multipliers.get(regime, {'sl': 1.0, 'tp': 1.0})
        sl_mult *= regime_mult['sl']
        tp_mult *= regime_mult['tp']
        
        if volatility > 0.02:
            sl_mult *= 1.2
        elif volatility < 0.005:
            sl_mult *= 0.9
        
        return sl_mult, tp_mult
    
    def _calculate_position_size(
        self,
        entry_price: float,
        sl_price: float,
        balance: float,
        risk_pct: float
    ) -> float:
        """Calculate position size based on risk percentage."""
        risk_amount = balance * risk_pct
        stop_distance = abs(entry_price - sl_price)
        
        if stop_distance > 0:
            size = risk_amount / stop_distance
        else:
            size = 0.01
        
        max_size = balance * 0.1
        return max(min(size, max_size), 0.01)
    
    def update_state(self, drawdown: float, winrate: float):
        """Update internal state (for backward compatibility)"""
        self.recent_drawdown = drawdown
        self.recent_winrate = winrate
    
    def get_risk_context(self) -> dict:
        """Get current risk context for logging"""
        return {
            'base_risk_pct': self.base_risk_pct,
            'min_risk_pct': self.min_risk_pct,
            'max_risk_pct': self.max_risk_pct,
            'survival_threshold': self.survival_threshold,
            'exploration_min_rr': self.exploration_min_rr,
            'fib_levels': FIB_LEVELS,
        }
