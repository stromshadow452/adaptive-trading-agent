"""
Portfolio Brain - PHASE 2
=========================

Final decision maker for multi-asset trading.
Selects which candidate trades to actually execute.

Responsibilities:
1. Correlation management - don't take correlated trades together
2. Risk budget - total exposure ≤ 3%
3. Trade selection - pick best candidates
4. Position limits - max 2 open trades

Input: Candidate signals from pipeline
Output: Approved trades for execution
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# DATA TYPES
# =============================================================================

@dataclass
class CandidateSignal:
    """A candidate trade from the pipeline."""
    symbol: str
    side: str  # "BUY" or "SELL"
    confidence: float  # 0.0 - 1.0
    entry_price: float
    sl_price: float
    tp_price: float
    size: float
    risk_pct: float  # Risk as % of equity
    regime: str
    screener_rank: int  # From screener
    timestamp: Any = None
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "confidence": round(self.confidence, 3),
            "entry_price": self.entry_price,
            "sl_price": self.sl_price,
            "tp_price": self.tp_price,
            "size": round(self.size, 4),
            "risk_pct": round(self.risk_pct, 4),
            "regime": self.regime,
            "screener_rank": self.screener_rank,
        }


@dataclass
class PortfolioDecision:
    """Output from Portfolio Brain."""
    approved: List[CandidateSignal]
    rejected: List[Tuple[CandidateSignal, str]]  # (signal, reason)
    total_risk: float
    explanation: str
    
    def to_dict(self) -> Dict:
        return {
            "approved_count": len(self.approved),
            "rejected_count": len(self.rejected),
            "approved": [s.symbol for s in self.approved],
            "rejected": {s.symbol: reason for s, reason in self.rejected},
            "total_risk": round(self.total_risk, 4),
            "explanation": self.explanation,
        }


@dataclass
class OpenPosition:
    """Currently open position."""
    symbol: str
    side: str
    size: float
    risk_pct: float
    entry_time: Any = None


# =============================================================================
# CORRELATION MATRIX
# =============================================================================

# Pre-defined correlations between major pairs
# Values: -1.0 (inverse) to +1.0 (same direction)
# Pairs with |correlation| > 0.70 should not be traded together
CORRELATION_MATRIX = {
    # EUR pairs
    ("EURUSD", "GBPUSD"): 0.85,   # Both USD shorts - highly correlated
    ("EURUSD", "USDCAD"): -0.65,  # Inverse (EUR up, CAD down usually)
    ("EURUSD", "USDJPY"): -0.30,  # Weak inverse
    ("EURUSD", "AUDUSD"): 0.70,   # Similar risk sentiment
    
    # GBP pairs
    ("GBPUSD", "GBPJPY"): 0.75,   # GBP as base
    ("GBPUSD", "EURGBP"): -0.80,  # GBP on different sides
    
    # JPY pairs
    ("USDJPY", "GBPJPY"): 0.80,   # Both JPY shorts
    ("USDJPY", "EURJPY"): 0.75,   # Both JPY shorts
    ("GBPJPY", "EURJPY"): 0.85,   # Both JPY shorts
    
    # AUD pairs
    ("AUDUSD", "NZDUSD"): 0.90,   # Very similar economies
    ("AUDUSD", "AUDJPY"): 0.60,   # AUD as base
    
    # CAD
    ("USDCAD", "AUDUSD"): -0.55,  # Commodity currencies inverse
}


def get_correlation(pair1: str, pair2: str) -> float:
    """
    Get correlation between two pairs.
    Returns 0 if unknown, which means "assume uncorrelated".
    """
    if pair1 == pair2:
        return 1.0
    
    # Check both orderings
    key1 = (pair1, pair2)
    key2 = (pair2, pair1)
    
    if key1 in CORRELATION_MATRIX:
        return CORRELATION_MATRIX[key1]
    if key2 in CORRELATION_MATRIX:
        return CORRELATION_MATRIX[key2]
    
    return 0.0  # Unknown = uncorrelated


def are_correlated(pair1: str, pair2: str, threshold: float = 0.70) -> bool:
    """Check if two pairs are too correlated to trade together."""
    corr = abs(get_correlation(pair1, pair2))
    return corr >= threshold


def check_side_correlation(sig1: CandidateSignal, sig2: CandidateSignal) -> bool:
    """
    Check if two signals compound risk due to correlation.
    
    Examples:
    - EURUSD BUY + GBPUSD BUY with correlation 0.85 = COMPOUND RISK
    - EURUSD BUY + GBPUSD SELL with correlation 0.85 = HEDGING (allowed)
    """
    corr = get_correlation(sig1.symbol, sig2.symbol)
    
    # Same direction signals with positive correlation = risk
    if sig1.side == sig2.side and corr > 0:
        return abs(corr) >= 0.70
    
    # Opposite direction signals with positive correlation = hedging (ok)
    # Same direction signals with negative correlation = hedging (ok)
    # Opposite direction signals with negative correlation = risk
    if sig1.side != sig2.side and corr < 0:
        return abs(corr) >= 0.70
    
    return False


# =============================================================================
# PORTFOLIO BRAIN
# =============================================================================

class PortfolioBrain:
    """
    Final decision maker for multi-asset trading.
    
    Constraints:
    - Max 2 open trades at any time
    - Total risk ≤ 3% of equity
    - No correlated trades (|corr| > 0.70)
    - Prefer higher confidence signals
    """
    
    # === LOCKED CONFIGURATION ===
    MAX_OPEN_TRADES = 2
    MAX_TOTAL_RISK_PCT = 0.03  # 3%
    CORRELATION_THRESHOLD = 0.70
    MIN_CONFIDENCE = 0.55  # Must meet ML threshold
    
    def __init__(self, config: Optional[Dict] = None):
        config = config or {}
        self.max_open_trades = config.get("max_open_trades", self.MAX_OPEN_TRADES)
        self.max_risk_pct = config.get("max_total_risk_pct", self.MAX_TOTAL_RISK_PCT)
        self.corr_threshold = config.get("correlation_threshold", self.CORRELATION_THRESHOLD)
        self.min_confidence = config.get("min_confidence", self.MIN_CONFIDENCE)
        
        logger.info(f"[PortfolioBrain] Initialized: max_trades={self.max_open_trades}, "
                   f"max_risk={self.max_risk_pct:.1%}, corr_thresh={self.corr_threshold}")
    
    def decide(
        self,
        candidates: List[CandidateSignal],
        open_positions: List[OpenPosition],
    ) -> PortfolioDecision:
        """
        Decide which candidates to approve.
        
        Args:
            candidates: List of candidate signals from pipeline
            open_positions: Currently open positions
            
        Returns:
            PortfolioDecision with approved and rejected lists
        """
        approved = []
        rejected = []
        explanations = []
        
        # Calculate current state
        current_trades = len(open_positions)
        current_risk = sum(p.risk_pct for p in open_positions)
        open_symbols = {p.symbol for p in open_positions}
        
        logger.info(f"[PortfolioBrain] Evaluating {len(candidates)} candidates, "
                   f"current: {current_trades} trades, {current_risk:.2%} risk")
        
        # === CHECK: Any room for new trades? ===
        if current_trades >= self.max_open_trades:
            for c in candidates:
                rejected.append((c, "Max trades reached"))
            return PortfolioDecision(
                approved=[],
                rejected=rejected,
                total_risk=current_risk,
                explanation="Max trades limit reached"
            )
        
        # === STEP 1: Filter out low confidence ===
        qualified = []
        for c in candidates:
            if c.confidence < self.min_confidence:
                rejected.append((c, f"Low confidence ({c.confidence:.2f})"))
            else:
                qualified.append(c)
        
        # === STEP 2: Filter out symbols with open positions ===
        available = []
        for c in qualified:
            if c.symbol in open_symbols:
                rejected.append((c, "Position already open"))
            else:
                available.append(c)
        
        # === STEP 3: Sort by confidence (best first) ===
        available.sort(key=lambda x: x.confidence, reverse=True)
        
        # === STEP 4: Filter correlated with open positions ===
        not_corr_with_open = []
        corr_blocked = set()
        
        for c in available:
            is_correlated = False
            for pos in open_positions:
                if are_correlated(c.symbol, pos.symbol, self.corr_threshold):
                    # Check if it's same-direction correlation risk
                    if c.side == pos.side:
                        is_correlated = True
                        corr_blocked.add(c.symbol)
                        rejected.append((c, f"Correlated with open {pos.symbol}"))
                        break
            
            if not is_correlated:
                not_corr_with_open.append(c)
        
        # === STEP 5: Filter correlated among candidates ===
        # Keep higher confidence when pairs are correlated
        final_candidates = []
        seen_groups = set()  # Track which correlation groups we've used
        
        for c in not_corr_with_open:
            skip = False
            
            for approved_c in final_candidates:
                if check_side_correlation(c, approved_c):
                    # This candidate is correlated with one we already approved
                    rejected.append((c, f"Correlated with {approved_c.symbol} (keeping higher conf)"))
                    skip = True
                    break
            
            if not skip:
                final_candidates.append(c)
        
        # === STEP 6: Take trades within risk budget ===
        available_risk = self.max_risk_pct - current_risk
        trades_available = self.max_open_trades - current_trades
        
        for c in final_candidates:
            # Check trade count
            if len(approved) >= trades_available:
                rejected.append((c, "Trade limit reached"))
                continue
            
            # Check risk budget
            if current_risk + c.risk_pct > self.max_risk_pct:
                rejected.append((c, f"Exceeds risk budget ({c.risk_pct:.2%})"))
                continue
            
            # APPROVED!
            approved.append(c)
            current_risk += c.risk_pct
            explanations.append(f"{c.symbol} {c.side} (conf={c.confidence:.2f})")
        
        # Build explanation
        if approved:
            explanation = f"Approved {len(approved)}: " + ", ".join(explanations)
        else:
            explanation = "No trades approved"
        
        return PortfolioDecision(
            approved=approved,
            rejected=rejected,
            total_risk=current_risk,
            explanation=explanation,
        )
    
    def get_correlation_groups(self, symbols: List[str]) -> List[List[str]]:
        """
        Group symbols by correlation for analysis.
        Returns list of correlated groups.
        """
        groups = []
        processed = set()
        
        for sym in symbols:
            if sym in processed:
                continue
            
            group = [sym]
            for other in symbols:
                if other != sym and other not in processed:
                    if are_correlated(sym, other, self.corr_threshold):
                        group.append(other)
            
            if len(group) > 1:
                for s in group:
                    processed.add(s)
                groups.append(group)
            else:
                processed.add(sym)
        
        return groups
    
    def get_risk_status(
        self,
        open_positions: List[OpenPosition]
    ) -> Dict:
        """Get current portfolio risk status."""
        total_risk = sum(p.risk_pct for p in open_positions)
        trade_count = len(open_positions)
        
        return {
            "trade_count": trade_count,
            "max_trades": self.max_open_trades,
            "trades_available": self.max_open_trades - trade_count,
            "total_risk_pct": total_risk,
            "max_risk_pct": self.max_risk_pct,
            "risk_available_pct": self.max_risk_pct - total_risk,
            "can_trade": trade_count < self.max_open_trades and total_risk < self.max_risk_pct,
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_portfolio_brain: Optional[PortfolioBrain] = None


def get_portfolio_brain(config: Optional[Dict] = None) -> PortfolioBrain:
    """Get or create portfolio brain singleton."""
    global _portfolio_brain
    if _portfolio_brain is None:
        _portfolio_brain = PortfolioBrain(config)
    return _portfolio_brain


def reset_portfolio_brain():
    """Reset singleton (for testing)."""
    global _portfolio_brain
    _portfolio_brain = None


# =============================================================================
# MAIN (Testing)
# =============================================================================

if __name__ == "__main__":
    brain = PortfolioBrain()
    
    # Test candidates
    candidates = [
        CandidateSignal(
            symbol="EURUSD", side="BUY", confidence=0.72,
            entry_price=1.0850, sl_price=1.0820, tp_price=1.0900,
            size=0.5, risk_pct=0.01, regime="TREND", screener_rank=1
        ),
        CandidateSignal(
            symbol="GBPUSD", side="BUY", confidence=0.65,
            entry_price=1.2650, sl_price=1.2620, tp_price=1.2700,
            size=0.4, risk_pct=0.008, regime="RANGE", screener_rank=2
        ),
        CandidateSignal(
            symbol="USDJPY", side="SELL", confidence=0.68,
            entry_price=150.50, sl_price=151.00, tp_price=149.50,
            size=0.3, risk_pct=0.007, regime="TREND", screener_rank=3
        ),
    ]
    
    # No open positions
    open_positions = []
    
    decision = brain.decide(candidates, open_positions)
    
    print("\n=== PORTFOLIO DECISION ===")
    print(f"Approved: {[s.symbol for s in decision.approved]}")
    print(f"Rejected: {[(s.symbol, r) for s, r in decision.rejected]}")
    print(f"Total Risk: {decision.total_risk:.2%}")
    print(f"Explanation: {decision.explanation}")
    
    # Test with correlated open position
    print("\n=== TEST WITH OPEN EURUSD ===")
    open_positions = [
        OpenPosition(symbol="EURUSD", side="BUY", size=0.5, risk_pct=0.01)
    ]
    
    decision2 = brain.decide(candidates, open_positions)
    print(f"Approved: {[s.symbol for s in decision2.approved]}")
    print(f"Rejected: {[(s.symbol, r) for s, r in decision2.rejected]}")
    
    # Check correlation
    print(f"\nEURUSD-GBPUSD correlation: {get_correlation('EURUSD', 'GBPUSD')}")
    print(f"Are correlated: {are_correlated('EURUSD', 'GBPUSD')}")
