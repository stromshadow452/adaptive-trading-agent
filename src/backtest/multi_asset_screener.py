"""
Multi-Asset Screener - PHASE 1
==============================

Fast filter that determines which pairs are TRADEABLE.
Does NOT generate BUY/SELL signals - only PASS/FAIL + Rank.

Screener Criteria:
1. Regime != DANGER
2. Volatility in acceptable range (0.5-2x normal ATR)
3. Session active for pair
4. Minimum data available
5. No existing open position for this pair

Output:
- TRADEABLE / NOT_TRADEABLE
- Rank (1 = best opportunity, 4 = worst among tradeable)
- Reason for decision
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# DATA TYPES
# =============================================================================

class ScreenerDecision(Enum):
    TRADEABLE = "TRADEABLE"
    NOT_TRADEABLE = "NOT_TRADEABLE"


@dataclass
class ScreenerResult:
    """Output from screener for a single pair."""
    symbol: str
    decision: ScreenerDecision
    rank: int  # 1 = best, higher = worse. 0 if NOT_TRADEABLE
    score: float  # 0.0 - 1.0, used for ranking
    reason: str
    regime: str
    volatility_ratio: float
    session: str
    timestamp: Optional[datetime] = None
    
    @property
    def tradeable(self) -> bool:
        return self.decision == ScreenerDecision.TRADEABLE
    
    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "decision": self.decision.value,
            "tradeable": self.tradeable,
            "rank": self.rank,
            "score": round(self.score, 3),
            "reason": self.reason,
            "regime": self.regime,
            "volatility_ratio": round(self.volatility_ratio, 3),
            "session": self.session,
        }


# =============================================================================
# SESSION DETECTION
# =============================================================================

# Session hours (UTC)
SESSIONS = {
    "ASIAN": (0, 8),      # 00:00 - 08:00 UTC
    "LONDON": (8, 16),    # 08:00 - 16:00 UTC
    "NEWYORK": (13, 21),  # 13:00 - 21:00 UTC
    "OFF": (21, 24),      # 21:00 - 00:00 UTC (low liquidity)
}

# Which pairs trade best in which session
PAIR_SESSIONS = {
    "EURUSD": ["LONDON", "NEWYORK"],
    "GBPUSD": ["LONDON", "NEWYORK"],
    "USDJPY": ["ASIAN", "NEWYORK"],
    "GBPJPY": ["ASIAN", "LONDON"],
    "AUDUSD": ["ASIAN", "NEWYORK"],
    "USDCAD": ["NEWYORK"],
    "AUDJPY": ["ASIAN"],
    "NZDUSD": ["ASIAN", "NEWYORK"],
    "EURJPY": ["ASIAN", "LONDON"],
    "EURGBP": ["LONDON"],
}


def detect_session(timestamp) -> str:
    """Detect current trading session from timestamp."""
    if timestamp is None:
        return "UNKNOWN"
    
    # Handle different timestamp types
    try:
        if hasattr(timestamp, 'hour'):
            hour = timestamp.hour
        else:
            # Convert to pandas Timestamp
            ts = pd.Timestamp(timestamp)
            hour = ts.hour
    except Exception:
        return "UNKNOWN"
    
    # Check overlaps first (most active)
    if 13 <= hour < 16:  # London-NY overlap
        return "LONDON_NY"
    if 8 <= hour < 16:
        return "LONDON"
    if 13 <= hour < 21:
        return "NEWYORK"
    if 0 <= hour < 8:
        return "ASIAN"
    
    return "OFF"



def is_session_active(symbol: str, session: str) -> bool:
    """Check if pair is active in current session."""
    if session == "UNKNOWN":
        return True  # Allow if unknown
    if session == "OFF":
        return False  # Block off-hours
    if session == "LONDON_NY":
        return True  # Most active, allow all
    
    preferred = PAIR_SESSIONS.get(symbol, ["LONDON", "NEWYORK"])
    return session in preferred


# =============================================================================
# VOLATILITY ANALYSIS
# =============================================================================

def calculate_volatility_ratio(atr: float, atr_avg: float) -> float:
    """
    Calculate current volatility relative to average.
    
    Ratio interpretation:
    - 0.5 = very low volatility (50% of normal)
    - 1.0 = normal volatility
    - 2.0 = high volatility (2x normal)
    """
    if atr_avg <= 0:
        return 1.0
    return atr / atr_avg


def is_volatility_acceptable(vol_ratio: float, 
                             min_ratio: float = 0.5, 
                             max_ratio: float = 2.5) -> Tuple[bool, str]:
    """
    Check if volatility is in acceptable range.
    
    Too low = no movement, waste of capital
    Too high = dangerous, unpredictable
    """
    if vol_ratio < min_ratio:
        return False, f"Vol too low ({vol_ratio:.2f}x)"
    if vol_ratio > max_ratio:
        return False, f"Vol too high ({vol_ratio:.2f}x)"
    return True, "Vol OK"


# =============================================================================
# REGIME DETECTION (Simple version for screener)
# =============================================================================

def detect_regime_simple(features: Dict) -> str:
    """
    Quick regime detection for screener.
    Not as sophisticated as full regime engine - just filter DANGER.
    """
    bb_width = features.get("bb_width", 0)
    volatility = features.get("volatility", 0)
    rsi = features.get("rsi_14", features.get("rsi", 50))
    
    # DANGER: Very high volatility or extreme RSI
    if bb_width > 0.03:  # BB very wide
        return "DANGER"
    if volatility > 0.025:  # Very high returns volatility
        return "DANGER"
    if rsi < 15 or rsi > 85:  # Extreme RSI
        return "DANGER"
    
    # TREND: Clear direction
    sma_20 = features.get("sma_20", 0)
    sma_50 = features.get("sma_50", 0)
    if sma_20 > 0 and sma_50 > 0:
        if sma_20 > sma_50 * 1.005 or sma_20 < sma_50 * 0.995:
            return "TREND"
    
    # Default: RANGE
    return "RANGE"


# =============================================================================
# SCREENER SCORING
# =============================================================================

def calculate_screener_score(
    regime: str,
    vol_ratio: float,
    session_active: bool,
    features: Dict
) -> float:
    """
    Calculate opportunity score (0-1) for ranking.
    Higher = better opportunity.
    
    Scoring factors:
    - Regime: TREND > RANGE > DANGER
    - Volatility: Sweet spot around 1.0-1.5x
    - Session: Active > Inactive
    - RSI: Neutral range preferred
    """
    score = 0.0
    
    # Regime score (40%)
    regime_scores = {"TREND": 0.40, "RANGE": 0.25, "DANGER": 0.0}
    score += regime_scores.get(regime, 0.10)
    
    # Volatility score (30%) - bell curve around 1.2x
    if 0.8 <= vol_ratio <= 1.8:
        vol_score = 0.30 - abs(vol_ratio - 1.2) * 0.15
        score += max(0, vol_score)
    elif 0.5 <= vol_ratio < 0.8:
        score += 0.10
    
    # Session score (20%)
    if session_active:
        score += 0.20
    else:
        score += 0.05
    
    # RSI score (10%) - prefer neutral
    rsi = features.get("rsi_14", features.get("rsi", 50))
    if 35 <= rsi <= 65:
        score += 0.10
    elif 25 <= rsi < 35 or 65 < rsi <= 75:
        score += 0.05
    
    return min(1.0, max(0.0, score))


# =============================================================================
# MAIN SCREENER CLASS
# =============================================================================

class MultiAssetScreener:
    """
    Multi-Asset Screener - filters pairs for trading.
    
    Does NOT generate BUY/SELL signals!
    Only determines: TRADEABLE / NOT_TRADEABLE + Rank
    """
    
    # Configuration
    DEFAULT_PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "GBPJPY"]
    MIN_VOL_RATIO = 0.5
    MAX_VOL_RATIO = 2.5
    
    def __init__(self, config: Optional[Dict] = None):
        config = config or {}
        self.pairs = config.get("pairs", self.DEFAULT_PAIRS)
        self.min_vol_ratio = config.get("min_vol_ratio", self.MIN_VOL_RATIO)
        self.max_vol_ratio = config.get("max_vol_ratio", self.MAX_VOL_RATIO)
        self.open_positions = set()  # Track which pairs have open positions
        
        logger.info(f"[Screener] Initialized for {len(self.pairs)} pairs: {self.pairs}")
    
    def set_open_positions(self, symbols: List[str]):
        """Update set of pairs with open positions."""
        self.open_positions = set(symbols)
    
    def screen(
        self,
        symbol: str,
        features: Dict,
        timestamp: Optional[datetime] = None,
        atr_avg: Optional[float] = None,
    ) -> ScreenerResult:
        """
        Screen a single pair.
        
        Args:
            symbol: Pair symbol (e.g., "EURUSD")
            features: Technical features dict
            timestamp: Current candle timestamp
            atr_avg: Average ATR for volatility comparison
            
        Returns:
            ScreenerResult with decision and score
        """
        reasons = []
        
        # === CHECK 1: Open position ===
        if symbol in self.open_positions:
            return ScreenerResult(
                symbol=symbol,
                decision=ScreenerDecision.NOT_TRADEABLE,
                rank=0,
                score=0.0,
                reason="Position already open",
                regime="N/A",
                volatility_ratio=0.0,
                session="N/A",
                timestamp=timestamp,
            )
        
        # === CHECK 2: Regime ===
        regime = detect_regime_simple(features)
        if regime == "DANGER":
            return ScreenerResult(
                symbol=symbol,
                decision=ScreenerDecision.NOT_TRADEABLE,
                rank=0,
                score=0.0,
                reason="DANGER regime",
                regime=regime,
                volatility_ratio=0.0,
                session="N/A",
                timestamp=timestamp,
            )
        
        # === CHECK 3: Session ===
        session = detect_session(timestamp) if timestamp else "UNKNOWN"
        session_active = is_session_active(symbol, session)
        if session == "OFF":
            return ScreenerResult(
                symbol=symbol,
                decision=ScreenerDecision.NOT_TRADEABLE,
                rank=0,
                score=0.0,
                reason="Session OFF (low liquidity)",
                regime=regime,
                volatility_ratio=0.0,
                session=session,
                timestamp=timestamp,
            )
        
        # === CHECK 4: Volatility ===
        atr = features.get("atr_14", features.get("atr", 0))
        if atr_avg is None or atr_avg <= 0:
            atr_avg = atr  # Use current as baseline
        
        vol_ratio = calculate_volatility_ratio(atr, atr_avg)
        vol_ok, vol_reason = is_volatility_acceptable(
            vol_ratio, self.min_vol_ratio, self.max_vol_ratio
        )
        
        if not vol_ok:
            return ScreenerResult(
                symbol=symbol,
                decision=ScreenerDecision.NOT_TRADEABLE,
                rank=0,
                score=0.0,
                reason=vol_reason,
                regime=regime,
                volatility_ratio=vol_ratio,
                session=session,
                timestamp=timestamp,
            )
        
        # === PASSED: Calculate score ===
        score = calculate_screener_score(regime, vol_ratio, session_active, features)
        
        return ScreenerResult(
            symbol=symbol,
            decision=ScreenerDecision.TRADEABLE,
            rank=0,  # Will be set by screen_all
            score=score,
            reason=f"PASS: {regime}, vol={vol_ratio:.1f}x",
            regime=regime,
            volatility_ratio=vol_ratio,
            session=session,
            timestamp=timestamp,
        )
    
    def screen_all(
        self,
        all_features: Dict[str, Dict],
        timestamp: Optional[datetime] = None,
        all_atr_avg: Optional[Dict[str, float]] = None,
    ) -> List[ScreenerResult]:
        """
        Screen all pairs and rank tradeable ones.
        
        Args:
            all_features: Dict mapping symbol -> features dict
            timestamp: Current candle timestamp
            all_atr_avg: Dict mapping symbol -> average ATR
            
        Returns:
            List of ScreenerResult, sorted by rank (tradeable first)
        """
        results = []
        all_atr_avg = all_atr_avg or {}
        
        # Screen each pair
        for symbol in self.pairs:
            features = all_features.get(symbol, {})
            if not features:
                logger.warning(f"[Screener] No features for {symbol}")
                continue
            
            atr_avg = all_atr_avg.get(symbol)
            result = self.screen(symbol, features, timestamp, atr_avg)
            results.append(result)
        
        # Rank tradeable pairs by score
        tradeable = [r for r in results if r.tradeable]
        tradeable.sort(key=lambda x: x.score, reverse=True)
        
        for i, result in enumerate(tradeable):
            result.rank = i + 1  # 1 = best
        
        # Non-tradeable get rank 0
        non_tradeable = [r for r in results if not r.tradeable]
        
        # Return tradeable first, then non-tradeable
        return tradeable + non_tradeable
    
    def get_top_pairs(
        self,
        results: List[ScreenerResult],
        top_n: int = 3
    ) -> List[str]:
        """Get top N tradeable pairs by rank."""
        tradeable = [r for r in results if r.tradeable]
        return [r.symbol for r in tradeable[:top_n]]
    
    def get_summary(self, results: List[ScreenerResult]) -> Dict:
        """Get screening summary."""
        tradeable = [r for r in results if r.tradeable]
        blocked = [r for r in results if not r.tradeable]
        
        return {
            "total_pairs": len(results),
            "tradeable": len(tradeable),
            "blocked": len(blocked),
            "top_pairs": [r.symbol for r in tradeable[:3]],
            "block_reasons": {r.symbol: r.reason for r in blocked},
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_screener_instance: Optional[MultiAssetScreener] = None


def get_screener(config: Optional[Dict] = None) -> MultiAssetScreener:
    """Get or create screener singleton."""
    global _screener_instance
    if _screener_instance is None:
        _screener_instance = MultiAssetScreener(config)
    return _screener_instance


def reset_screener():
    """Reset screener singleton (for testing)."""
    global _screener_instance
    _screener_instance = None


# =============================================================================
# MAIN (Testing)
# =============================================================================

if __name__ == "__main__":
    # Quick test
    screener = MultiAssetScreener()
    
    # Simulate features for testing
    test_features = {
        "EURUSD": {"rsi_14": 55, "bb_width": 0.015, "volatility": 0.01, 
                   "sma_20": 1.0850, "sma_50": 1.0800, "atr_14": 0.0012},
        "GBPUSD": {"rsi_14": 45, "bb_width": 0.018, "volatility": 0.012,
                   "sma_20": 1.2650, "sma_50": 1.2700, "atr_14": 0.0015},
        "USDJPY": {"rsi_14": 72, "bb_width": 0.025, "volatility": 0.015,
                   "sma_20": 150.50, "sma_50": 150.00, "atr_14": 0.0018},
        "GBPJPY": {"rsi_14": 88, "bb_width": 0.035, "volatility": 0.030,
                   "sma_20": 190.00, "sma_50": 188.00, "atr_14": 0.0025},
    }
    
    from datetime import datetime
    timestamp = datetime(2024, 1, 15, 10, 30)  # London session
    
    results = screener.screen_all(test_features, timestamp)
    
    print("\n=== SCREENER RESULTS ===")
    for r in results:
        status = "✅" if r.tradeable else "❌"
        print(f"{status} {r.symbol}: {r.decision.value} (rank={r.rank}, score={r.score:.2f}) - {r.reason}")
    
    print(f"\nTop pairs: {screener.get_top_pairs(results)}")
    print(f"Summary: {screener.get_summary(results)}")
