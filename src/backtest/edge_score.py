"""
MARK-3 EDGE_SCORE Module

Profit acceleration layer that ONLY affects position sizing.
Does NOT create trades, does NOT override safety systems.

PHILOSOPHY:
- Protect survival first (MARK-2 always wins)
- Increase profit only on A-grade setups
- Fewer trades, higher quality
- Size boost is a REWARD, not a right

"MARK-2 decides IF we trade. MARK-3 decides HOW MUCH we commit."
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

# Component weights (must sum to 1.0)
WEIGHT_ML_CONFIDENCE = 0.30
WEIGHT_REGIME_STRENGTH = 0.25
WEIGHT_STRUCTURE_QUALITY = 0.25
WEIGHT_VOLATILITY_ALIGN = 0.20

# Edge score thresholds
EDGE_BLOCK_THRESHOLD = 0.40      # Below this = minimum size or skip
EDGE_REDUCED_THRESHOLD = 0.60   # Below this = reduced size
EDGE_NORMAL_THRESHOLD = 0.75    # Above this = size boost allowed

# Size multipliers
SIZE_MULT_MINIMUM = 0.3         # EDGE < 0.40
SIZE_MULT_REDUCED = 0.6         # EDGE 0.40-0.60
SIZE_MULT_NORMAL = 1.0          # EDGE 0.60-0.75
SIZE_MULT_BOOST_MIN = 1.3       # EDGE > 0.75 (if allowed)
SIZE_MULT_BOOST_MAX = 1.6       # Maximum boost cap

# Safety caps
STRUCTURE_QUALITY_FLOOR = 0.40  # Below this = cap edge score
EDGE_CAP_LOW_STRUCTURE = 0.50   # Max edge when structure is weak


@dataclass
class EdgeScoreOutput:
    """Output from EDGE_SCORE computation."""
    
    edge_score: float           # 0.0 - 1.0
    size_multiplier: float      # 0.3 - 1.6
    boost_allowed: bool         # True if size boost is legal
    quality_tier: str           # "A", "B", "C", "D"
    
    # Component breakdown (for logging)
    ml_component: float
    regime_component: float
    structure_component: float
    volatility_component: float
    
    # Flags
    structure_capped: bool      # True if low structure capped edge
    danger_blocked: bool        # True if DANGER blocked boost
    
    def to_dict(self) -> dict:
        return {
            'edge_score': self.edge_score,
            'size_multiplier': self.size_multiplier,
            'boost_allowed': self.boost_allowed,
            'quality_tier': self.quality_tier,
            'ml_component': self.ml_component,
            'regime_component': self.regime_component,
            'structure_component': self.structure_component,
            'volatility_component': self.volatility_component,
            'structure_capped': self.structure_capped,
            'danger_blocked': self.danger_blocked,
        }


# ============================================================================
# EDGE SCORE COMPUTATION
# ============================================================================

class EdgeScoreModule:
    """
    MARK-3 Edge Score Calculator
    
    Computes a quality score for each trade opportunity:
    - 0.0 = Terrible setup (minimum size or skip)
    - 0.5 = Average setup (reduced size)
    - 0.75 = Good setup (normal size)
    - 1.0 = A-grade setup (boost allowed)
    
    CRITICAL RULES:
    1. NEVER creates trades
    2. NEVER overrides Memory/Ego/RiskBrain
    3. DANGER regime = NO SIZE BOOST
    4. Low structure = capped edge score
    """
    
    def __init__(self, config: dict = None):
        config = config or {}
        
        # Weights (configurable)
        self.w_ml = config.get('weight_ml', WEIGHT_ML_CONFIDENCE)
        self.w_regime = config.get('weight_regime', WEIGHT_REGIME_STRENGTH)
        self.w_structure = config.get('weight_structure', WEIGHT_STRUCTURE_QUALITY)
        self.w_volatility = config.get('weight_volatility', WEIGHT_VOLATILITY_ALIGN)
        
        # Normalize weights
        total = self.w_ml + self.w_regime + self.w_structure + self.w_volatility
        self.w_ml /= total
        self.w_regime /= total
        self.w_structure /= total
        self.w_volatility /= total
        
        logger.info(
            f"EdgeScoreModule initialized: "
            f"ML={self.w_ml:.0%}, Regime={self.w_regime:.0%}, "
            f"Structure={self.w_structure:.0%}, Vol={self.w_volatility:.0%}"
        )
    
    def compute(
        self,
        ml_confidence: float,
        regime_strength: float,
        structure_quality: float,
        volatility_alignment: float,
        regime: str,
        is_exploration: bool = False
    ) -> EdgeScoreOutput:
        """
        Compute EDGE_SCORE from input signals.
        
        Args:
            ml_confidence: ML model confidence (0.0-1.0)
            regime_strength: Current regime strength (0.0-1.0)
            structure_quality: S/R and structure quality (0.0-1.0)
            volatility_alignment: ATR/volatility alignment (0.0-1.0)
            regime: Current regime label ("RANGE", "TREND", "DANGER")
            is_exploration: True if this is an exploration trade
        
        Returns:
            EdgeScoreOutput with score, multiplier, and breakdown
        """
        # === STEP 1: Normalize inputs ===
        ml_conf = self._clamp(ml_confidence)
        regime_str = self._clamp(regime_strength)
        structure = self._clamp(structure_quality)
        vol_align = self._clamp(volatility_alignment)
        
        # === STEP 2: Calculate weighted components ===
        ml_component = ml_conf * self.w_ml
        regime_component = regime_str * self.w_regime
        structure_component = structure * self.w_structure
        volatility_component = vol_align * self.w_volatility
        
        # === STEP 3: Calculate raw edge score ===
        raw_edge = ml_component + regime_component + structure_component + volatility_component
        
        # === STEP 4: Apply structure quality cap ===
        structure_capped = False
        if structure < STRUCTURE_QUALITY_FLOOR:
            # Low structure = cap the edge score
            raw_edge = min(raw_edge, EDGE_CAP_LOW_STRUCTURE)
            structure_capped = True
        
        # === STEP 5: Clamp final edge score ===
        edge_score = self._clamp(raw_edge)
        
        # === STEP 6: Determine if boost is allowed ===
        danger_blocked = False
        boost_allowed = edge_score > EDGE_NORMAL_THRESHOLD
        
        # CRITICAL: No boost in DANGER regime
        if regime.upper() == "DANGER":
            boost_allowed = False
            danger_blocked = True
        
        # No boost for exploration trades (keep them conservative)
        if is_exploration:
            boost_allowed = False
        
        # === STEP 7: Calculate size multiplier ===
        size_multiplier = self._calculate_size_multiplier(edge_score, boost_allowed)
        
        # === STEP 8: Determine quality tier ===
        quality_tier = self._get_quality_tier(edge_score)
        
        # Log significant decisions
        if edge_score > 0.7 or edge_score < 0.4:
            logger.debug(
                f"EDGE: {edge_score:.2f} ({quality_tier}) "
                f"mult={size_multiplier:.2f} boost={boost_allowed}"
            )
        
        return EdgeScoreOutput(
            edge_score=edge_score,
            size_multiplier=size_multiplier,
            boost_allowed=boost_allowed,
            quality_tier=quality_tier,
            ml_component=ml_component,
            regime_component=regime_component,
            structure_component=structure_component,
            volatility_component=volatility_component,
            structure_capped=structure_capped,
            danger_blocked=danger_blocked
        )
    
    def _calculate_size_multiplier(
        self,
        edge_score: float,
        boost_allowed: bool
    ) -> float:
        """
        Calculate size multiplier from edge score.
        
        Mapping:
        - EDGE < 0.40 → 0.3x (minimum)
        - EDGE 0.40-0.60 → 0.3x to 0.6x (linear interpolation)
        - EDGE 0.60-0.75 → 0.6x to 1.0x (linear interpolation)
        - EDGE > 0.75 → 1.0x to 1.6x (if boost allowed)
        """
        if edge_score < EDGE_BLOCK_THRESHOLD:
            # D-tier: Minimum size
            return SIZE_MULT_MINIMUM
        
        elif edge_score < EDGE_REDUCED_THRESHOLD:
            # C-tier: Reduced size (linear interpolation)
            t = (edge_score - EDGE_BLOCK_THRESHOLD) / (EDGE_REDUCED_THRESHOLD - EDGE_BLOCK_THRESHOLD)
            return SIZE_MULT_MINIMUM + t * (SIZE_MULT_REDUCED - SIZE_MULT_MINIMUM)
        
        elif edge_score < EDGE_NORMAL_THRESHOLD:
            # B-tier: Normal size (linear interpolation)
            t = (edge_score - EDGE_REDUCED_THRESHOLD) / (EDGE_NORMAL_THRESHOLD - EDGE_REDUCED_THRESHOLD)
            return SIZE_MULT_REDUCED + t * (SIZE_MULT_NORMAL - SIZE_MULT_REDUCED)
        
        else:
            # A-tier: Boost allowed (if permitted)
            if not boost_allowed:
                return SIZE_MULT_NORMAL
            
            # Linear interpolation from 1.0x to 1.6x
            t = min(1.0, (edge_score - EDGE_NORMAL_THRESHOLD) / (1.0 - EDGE_NORMAL_THRESHOLD))
            boost = SIZE_MULT_BOOST_MIN + t * (SIZE_MULT_BOOST_MAX - SIZE_MULT_BOOST_MIN)
            return min(SIZE_MULT_BOOST_MAX, boost)
    
    def _get_quality_tier(self, edge_score: float) -> str:
        """Get quality tier label from edge score."""
        if edge_score >= EDGE_NORMAL_THRESHOLD:
            return "A"
        elif edge_score >= EDGE_REDUCED_THRESHOLD:
            return "B"
        elif edge_score >= EDGE_BLOCK_THRESHOLD:
            return "C"
        else:
            return "D"
    
    def _clamp(self, value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """Clamp value to range."""
        if value is None or (isinstance(value, float) and value != value):  # NaN check
            return 0.5  # Default to neutral
        return max(min_val, min(max_val, float(value)))


# ============================================================================
# INTEGRATION WITH SIZING PIPELINE
# ============================================================================

def apply_edge_score_to_sizing(
    base_size: float,
    edge_output: EdgeScoreOutput,
    memory_mod: float,
    ego_mod: float,
    regime_mod: float
) -> Tuple[float, dict]:
    """
    Apply EDGE_SCORE to sizing pipeline.
    
    PIPELINE ORDER (CRITICAL):
    1. base_size (from RiskBrain)
    2. EDGE_SCORE multiplier (MARK-3 boost/reduce)
    3. Memory modifier (pain zones)
    4. Ego modifier (overconfidence)
    5. Regime modifier (regime strength)
    
    SAFETY: MARK-2 modifiers are applied AFTER edge boost,
    so they can always reduce the size back down.
    
    Args:
        base_size: Base position size from RiskBrain
        edge_output: EdgeScoreOutput from EdgeScoreModule
        memory_mod: Memory module modifier (0.1-1.0)
        ego_mod: Ego control modifier (0.5-1.0)
        regime_mod: Regime strength modifier (0.5-1.0)
    
    Returns:
        (final_size, info_dict)
    """
    # Step 1: Apply edge score multiplier
    edge_adjusted = base_size * edge_output.size_multiplier
    
    # Step 2: Apply MARK-2 safety modifiers (ALWAYS applied after edge)
    after_memory = edge_adjusted * memory_mod
    after_ego = after_memory * ego_mod
    after_regime = after_ego * regime_mod
    
    # Step 3: Apply floor (never zero)
    final_size = max(base_size * 0.1, after_regime)
    
    # Build info dict
    info = {
        'base_size': base_size,
        'edge_multiplier': edge_output.size_multiplier,
        'edge_adjusted': edge_adjusted,
        'after_memory': after_memory,
        'after_ego': after_ego,
        'after_regime': after_regime,
        'final_size': final_size,
        'edge_score': edge_output.edge_score,
        'quality_tier': edge_output.quality_tier,
        'boost_applied': edge_output.size_multiplier > 1.0,
        'safety_reduced': final_size < edge_adjusted,
    }
    
    # Log if boost was applied but then reduced by safety
    if edge_output.size_multiplier > 1.0 and final_size < edge_adjusted:
        logger.info(
            f"EDGE: Boost {edge_output.size_multiplier:.2f}x applied but "
            f"reduced by safety: {edge_adjusted:.2f} → {final_size:.2f}"
        )
    
    return final_size, info


# ============================================================================
# VOLATILITY ALIGNMENT CALCULATOR
# ============================================================================

def calculate_volatility_alignment(
    current_atr: float,
    avg_atr: float,
    atr_expansion_threshold: float = 1.5,
    atr_contraction_threshold: float = 0.7
) -> float:
    """
    Calculate volatility alignment score.
    
    Best alignment (1.0): ATR is slightly above average (expansion favorable)
    Worst alignment (0.0): ATR is too extreme (either direction)
    
    Args:
        current_atr: Current ATR value
        avg_atr: Average ATR (e.g., 20-period SMA of ATR)
        atr_expansion_threshold: ATR ratio above which is "too volatile"
        atr_contraction_threshold: ATR ratio below which is "too quiet"
    
    Returns:
        Volatility alignment score (0.0-1.0)
    """
    if avg_atr <= 0 or current_atr <= 0:
        return 0.5  # Neutral if invalid
    
    ratio = current_atr / avg_atr
    
    if ratio > atr_expansion_threshold:
        # Too volatile - reduce alignment
        excess = (ratio - atr_expansion_threshold) / atr_expansion_threshold
        return max(0.0, 0.5 - excess * 0.5)
    
    elif ratio < atr_contraction_threshold:
        # Too quiet - reduce alignment
        deficit = (atr_contraction_threshold - ratio) / atr_contraction_threshold
        return max(0.0, 0.5 - deficit * 0.5)
    
    else:
        # Goldilocks zone - good alignment
        # Peak at ratio = 1.1 (slight expansion)
        optimal = 1.1
        distance = abs(ratio - optimal) / (atr_expansion_threshold - atr_contraction_threshold)
        return min(1.0, 0.7 + (1.0 - distance) * 0.3)


# ============================================================================
# STRUCTURE QUALITY CALCULATOR
# ============================================================================

def calculate_structure_quality(
    sr_strength: float,
    trend_alignment: float,
    pattern_quality: float = 0.5
) -> float:
    """
    Calculate structure quality score.
    
    Args:
        sr_strength: Support/resistance strength (0.0-1.0)
        trend_alignment: How aligned with trend (0.0-1.0)
        pattern_quality: Chart pattern quality (0.0-1.0)
    
    Returns:
        Structure quality score (0.0-1.0)
    """
    # Weighted average
    score = (
        sr_strength * 0.40 +
        trend_alignment * 0.35 +
        pattern_quality * 0.25
    )
    
    return max(0.0, min(1.0, score))


# ============================================================================
# SAFEGUARDS
# ============================================================================

class EdgeScoreSafeguards:
    """
    Safety checks to prevent edge score abuse.
    """
    
    # Maximum boost trades per day
    MAX_BOOST_TRADES_PER_DAY = 3
    
    # Minimum time between boost trades (minutes)
    MIN_BOOST_INTERVAL_MINUTES = 60
    
    # Consecutive boost limit
    MAX_CONSECUTIVE_BOOSTS = 2
    
    def __init__(self):
        self.boost_trades_today = 0
        self.consecutive_boosts = 0
        self.last_boost_time = None
        self.current_date = None
    
    def can_boost(self, timestamp) -> Tuple[bool, str]:
        """
        Check if boost is allowed based on safeguards.
        
        Returns:
            (allowed, reason)
        """
        from datetime import datetime
        
        # Reset daily counter
        current_date = timestamp.date() if hasattr(timestamp, 'date') else None
        if current_date != self.current_date:
            self.boost_trades_today = 0
            self.current_date = current_date
        
        # Check daily limit
        if self.boost_trades_today >= self.MAX_BOOST_TRADES_PER_DAY:
            return False, f"Daily boost limit ({self.MAX_BOOST_TRADES_PER_DAY}) reached"
        
        # Check consecutive limit
        if self.consecutive_boosts >= self.MAX_CONSECUTIVE_BOOSTS:
            return False, f"Consecutive boost limit ({self.MAX_CONSECUTIVE_BOOSTS}) reached"
        
        # Check time interval
        if self.last_boost_time is not None:
            if hasattr(timestamp, 'timestamp'):
                minutes_since = (timestamp - self.last_boost_time).total_seconds() / 60
            else:
                minutes_since = self.MIN_BOOST_INTERVAL_MINUTES + 1
            
            if minutes_since < self.MIN_BOOST_INTERVAL_MINUTES:
                return False, f"Boost interval not met ({minutes_since:.0f} < {self.MIN_BOOST_INTERVAL_MINUTES} min)"
        
        return True, "Boost allowed"
    
    def record_trade(self, was_boosted: bool, timestamp):
        """Record trade for safeguard tracking."""
        if was_boosted:
            self.boost_trades_today += 1
            self.consecutive_boosts += 1
            self.last_boost_time = timestamp
        else:
            self.consecutive_boosts = 0
    
    def reset(self):
        """Reset safeguards."""
        self.boost_trades_today = 0
        self.consecutive_boosts = 0
        self.last_boost_time = None
        self.current_date = None


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_edge_score_module(config: dict = None) -> EdgeScoreModule:
    """Factory function to create EdgeScoreModule."""
    return EdgeScoreModule(config)
