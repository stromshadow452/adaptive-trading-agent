"""
EXPERIENCE REASONING MODULE (ERM)
==================================

Fast adaptive reasoning from past mistakes.

Features:
- O(1) pattern lookup
- Probability-based decisions
- Automatic behavior correction
- Time-decay for old experiences

Actions:
- EXECUTE: Full trade (success_rate >= 60%)
- REDUCE: Scaled trade (success_rate 40-60%)
- IGNORE: Skip trade (success_rate < 40% or unknown)

Note: ERM never blocks trades directly. MARK-2 has final veto.
"""

import pickle
import time
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


# ============================================================================
# FEATURE BUCKETING
# ============================================================================

REGIME_MAP = {'RANGE': 0, 'TREND': 1, 'DANGER': 2}
SESSION_MAP = {'OFF': 0, 'SYDNEY': 1, 'TOKYO': 2, 'LONDON': 3, 'NEW_YORK': 4}
VOL_MAP = {'LOW': 0, 'MID': 1, 'HIGH': 2}
EDGE_TIER_MAP = {'D': 0, 'C': 1, 'B': 2, 'A': 3}


def bucket_confidence(conf: float) -> int:
    """Bucket ML confidence: 0-0.5=0, 0.5-0.65=1, 0.65-0.8=2, 0.8+=3"""
    if conf < 0.5:
        return 0
    if conf < 0.65:
        return 1
    if conf < 0.8:
        return 2
    return 3


def bucket_regime_strength(rs: float) -> int:
    """Bucket regime strength: 0-0.4=0, 0.4-0.7=1, 0.7+=2"""
    if rs < 0.4:
        return 0
    if rs < 0.7:
        return 1
    return 2


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass(frozen=True)
class ProblemPattern:
    """
    Immutable pattern for O(1) hash lookup.
    Represents a normalized trading context.
    """
    session: int         # 0-4 (OFF/SYDNEY/TOKYO/LONDON/NEW_YORK)
    regime: int          # 0-2 (RANGE/TREND/DANGER)
    regime_bucket: int   # 0-2 (low/mid/high strength)
    edge_tier: int       # 0-3 (D/C/B/A)
    conf_bucket: int     # 0-3 (confidence bucket)
    vol_bucket: int      # 0-2 (LOW/MID/HIGH)
    
    def __hash__(self) -> int:
        return hash((self.session, self.regime, self.regime_bucket,
                     self.edge_tier, self.conf_bucket, self.vol_bucket))
    
    def to_string(self) -> str:
        """Human-readable representation."""
        return f"{self.session}-{self.regime}-{self.regime_bucket}-{self.edge_tier}-{self.conf_bucket}-{self.vol_bucket}"


@dataclass
class SolutionAttempt:
    """Tracks effectiveness of a solution type."""
    solution_type: str        # EXECUTE / REDUCE / WAIT
    size_multiplier: float    # Applied size multiplier
    success_count: int = 0
    failure_count: int = 0
    total_r: float = 0.0
    last_updated: float = 0.0  # Unix timestamp
    
    @property
    def total_trades(self) -> int:
        return self.success_count + self.failure_count
    
    @property
    def success_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.success_count / self.total_trades
    
    @property
    def avg_r(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.total_r / self.total_trades


@dataclass
class ExperienceRecord:
    """Complete experience for a pattern."""
    pattern: ProblemPattern
    solutions: Dict[str, SolutionAttempt] = field(default_factory=dict)
    occurrence_count: int = 0
    first_seen: float = 0.0
    last_seen: float = 0.0
    decay_weight: float = 1.0
    
    @property
    def best_solution(self) -> Optional[SolutionAttempt]:
        """Dynamic best solution based on success rate and avg R."""
        valid = [s for s in self.solutions.values() if s.total_trades >= 3]
        if not valid:
            return None
        # Score = success_rate * max(0.1, avg_r) to handle negative avg_r
        return max(valid, key=lambda s: s.success_rate * max(0.1, s.avg_r + 1))


@dataclass
class ERMDecision:
    """Output from ERM evaluation."""
    action: str              # EXECUTE / REDUCE / IGNORE
    size_multiplier: float   # 0.0-1.0
    confidence: float        # 0.0-1.0
    reason: str              # Human-readable


# ============================================================================
# CONTEXT ENCODER
# ============================================================================

def encode_context(ctx: Dict) -> ProblemPattern:
    """
    Convert raw context to hashable pattern. O(1).
    
    Expected context keys:
    - session: TOKYO/LONDON/NEW_YORK/SYDNEY/OFF
    - regime: RANGE/TREND/DANGER
    - regime_strength: 0.0-1.0
    - edge_tier: D/C/B/A
    - ml_confidence: 0.0-1.0
    - volatility_bucket: LOW/MID/HIGH
    """
    return ProblemPattern(
        session=SESSION_MAP.get(ctx.get('session', 'OFF'), 0),
        regime=REGIME_MAP.get(ctx.get('regime', 'RANGE'), 0),
        regime_bucket=bucket_regime_strength(ctx.get('regime_strength', 0.5)),
        edge_tier=EDGE_TIER_MAP.get(ctx.get('edge_tier', 'C'), 1),
        conf_bucket=bucket_confidence(ctx.get('ml_confidence', 0.5)),
        vol_bucket=VOL_MAP.get(ctx.get('volatility_bucket', 'MID'), 1)
    )


# ============================================================================
# MEMORY STORE
# ============================================================================

class ExperienceMemory:
    """
    Fast O(1) pattern lookup with LRU eviction.
    
    Features:
    - Hash-based storage
    - LRU eviction when over capacity
    - Time decay for old patterns
    """
    
    MAX_PATTERNS = 10000
    DECAY_HALF_LIFE_DAYS = 30
    
    def __init__(self):
        self.records: Dict[ProblemPattern, ExperienceRecord] = {}
        self.access_order: OrderedDict = OrderedDict()
    
    def get(self, pattern: ProblemPattern) -> Optional[ExperienceRecord]:
        """O(1) lookup."""
        if pattern in self.records:
            self.access_order.move_to_end(pattern)
            return self.records[pattern]
        return None
    
    def upsert(self, pattern: ProblemPattern, record: ExperienceRecord):
        """Insert or update, with LRU eviction."""
        self.records[pattern] = record
        self.access_order[pattern] = True
        
        # Evict LRU if over capacity
        while len(self.records) > self.MAX_PATTERNS:
            oldest = next(iter(self.access_order))
            del self.records[oldest]
            del self.access_order[oldest]
            logger.debug(f"[ERM] Evicted LRU pattern: {oldest.to_string()}")
    
    def apply_decay(self):
        """Apply time decay to all records. Run daily/offline."""
        now = time.time()
        half_life_seconds = self.DECAY_HALF_LIFE_DAYS * 86400
        
        to_prune = []
        for pattern, record in self.records.items():
            days_since = (now - record.last_seen) / 86400
            decay = 0.5 ** (days_since / self.DECAY_HALF_LIFE_DAYS)
            record.decay_weight = decay
            
            # Prune if decay too low
            if record.decay_weight < 0.1:
                to_prune.append(pattern)
        
        for pattern in to_prune:
            del self.records[pattern]
            if pattern in self.access_order:
                del self.access_order[pattern]
        
        if to_prune:
            logger.info(f"[ERM] Pruned {len(to_prune)} decayed patterns")
    
    def stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            'total_patterns': len(self.records),
            'capacity': self.MAX_PATTERNS,
            'utilization': len(self.records) / self.MAX_PATTERNS
        }


# ============================================================================
# REASONING ENGINE
# ============================================================================

class ReasoningEngine:
    """
    Core decision logic. O(1) runtime.
    
    Decision thresholds:
    - EXECUTE: success_rate >= 60% and avg_r > 0
    - REDUCE: success_rate 40-60%
    - IGNORE: success_rate < 40% or unknown pattern
    """
    
    EXECUTE_THRESHOLD = 0.60
    REDUCE_THRESHOLD = 0.40
    MAX_FAILURES = 5
    MIN_TRADES_FOR_DECISION = 3
    
    def decide(self, record: Optional[ExperienceRecord]) -> ERMDecision:
        """Make decision based on experience record."""
        
        # Case 1: No experience → default to cautious REDUCE
        if record is None:
            return ERMDecision(
                action="REDUCE",
                size_multiplier=0.5,
                confidence=0.0,
                reason="No prior experience for this pattern"
            )
        
        best = record.best_solution
        
        # Case 2: Insufficient data (< 3 trades)
        if best is None or best.total_trades < self.MIN_TRADES_FOR_DECISION:
            return ERMDecision(
                action="REDUCE",
                size_multiplier=0.5,
                confidence=0.3,
                reason=f"Insufficient data ({record.occurrence_count} occurrences)"
            )
        
        # Case 3: Repeated failures → IGNORE
        if best.failure_count >= self.MAX_FAILURES and best.success_rate < self.REDUCE_THRESHOLD:
            return ERMDecision(
                action="IGNORE",
                size_multiplier=0.0,
                confidence=0.8,
                reason=f"Pattern has {best.failure_count} failures, SR={best.success_rate:.1%}"
            )
        
        # Case 4: High success rate → EXECUTE
        if best.success_rate >= self.EXECUTE_THRESHOLD and best.avg_r > 0:
            return ERMDecision(
                action="EXECUTE",
                size_multiplier=min(1.0, best.size_multiplier),
                confidence=best.success_rate,
                reason=f"Pattern SR={best.success_rate:.1%}, AvgR={best.avg_r:.2f}"
            )
        
        # Case 5: Medium success rate → REDUCE
        if best.success_rate >= self.REDUCE_THRESHOLD:
            # Linear interpolation: 40% → 0.3, 60% → 0.6
            reduce_mult = 0.3 + (best.success_rate - 0.4) * 1.5
            reduce_mult = max(0.3, min(0.6, reduce_mult))
            
            return ERMDecision(
                action="REDUCE",
                size_multiplier=reduce_mult,
                confidence=best.success_rate,
                reason=f"Pattern SR={best.success_rate:.1%} < 60%, reducing size"
            )
        
        # Case 6: Low success rate → IGNORE
        return ERMDecision(
            action="IGNORE",
            size_multiplier=0.0,
            confidence=1.0 - best.success_rate,
            reason=f"Pattern SR={best.success_rate:.1%} too low"
        )


# ============================================================================
# EXPERIENCE LEARNER
# ============================================================================

class ExperienceLearner:
    """Updates memory after trade closes. Non-blocking."""
    
    def update(
        self,
        memory: ExperienceMemory,
        pattern: ProblemPattern,
        solution_used: str,
        size_mult: float,
        r_multiple: float
    ):
        """
        Update experience after trade outcome.
        
        Args:
            memory: Experience memory store
            pattern: The pattern that was traded
            solution_used: EXECUTE / REDUCE / IGNORE
            size_mult: Size multiplier that was applied
            r_multiple: Trade result in R
        """
        now = time.time()
        
        # Get or create record
        record = memory.get(pattern)
        if record is None:
            record = ExperienceRecord(
                pattern=pattern,
                solutions={},
                first_seen=now
            )
        
        # Update occurrence
        record.occurrence_count += 1
        record.last_seen = now
        record.decay_weight = 1.0  # Reset decay on activity
        
        # Get or create solution attempt
        if solution_used not in record.solutions:
            record.solutions[solution_used] = SolutionAttempt(
                solution_type=solution_used,
                size_multiplier=size_mult
            )
        
        solution = record.solutions[solution_used]
        
        # Update solution stats
        if r_multiple > 0:
            solution.success_count += 1
        else:
            solution.failure_count += 1
        
        solution.total_r += r_multiple
        solution.last_updated = now
        
        # Update size multiplier with exponential moving average
        alpha = 0.2
        solution.size_multiplier = alpha * size_mult + (1 - alpha) * solution.size_multiplier
        
        # Save back to memory
        memory.upsert(pattern, record)
        
        logger.info(
            f"[ERM] Updated pattern {pattern.to_string()}: "
            f"R={r_multiple:.2f}, SR={solution.success_rate:.1%}, "
            f"AvgR={solution.avg_r:.2f}"
        )


# ============================================================================
# MAIN ERM CLASS
# ============================================================================

class ExperienceReasoningModule:
    """
    Fast adaptive reasoning from past mistakes.
    
    Constraints:
    - Decision time: ≤ 1ms
    - Never blocks trades (only EXECUTE/REDUCE/IGNORE)
    - MARK-2 still has final veto
    
    Usage:
        erm = ExperienceReasoningModule()
        
        # Before trade
        decision = erm.evaluate(context)
        if decision.action == "IGNORE":
            # Consider skipping or reducing
        else:
            size = base_size * decision.size_multiplier
        
        # After trade
        erm.update(context, "EXECUTE", 1.0, r_multiple=-1.2)
    """
    
    def __init__(self, persistence_path: Path = None):
        self.memory = ExperienceMemory()
        self.engine = ReasoningEngine()
        self.learner = ExperienceLearner()
        self.persistence_path = persistence_path or Path("data/erm_memory.pkl")
        
        # Stats
        self.total_evaluations = 0
        self.total_updates = 0
        
        # Load existing memory
        if self.persistence_path.exists():
            self._load()
            logger.info(f"[ERM] Loaded {len(self.memory.records)} patterns from disk")
    
    def evaluate(self, context: Dict) -> ERMDecision:
        """
        Main entry point. O(1) complexity.
        
        Args:
            context: Trading context with session, regime, etc.
        
        Returns:
            ERMDecision with action, size_multiplier, confidence, reason
        """
        self.total_evaluations += 1
        
        pattern = encode_context(context)
        record = self.memory.get(pattern)
        decision = self.engine.decide(record)
        
        # Log significant decisions
        if decision.action != "EXECUTE" or decision.confidence < 0.6:
            logger.debug(
                f"[ERM] {pattern.to_string()} → {decision.action} "
                f"(conf={decision.confidence:.2f}): {decision.reason}"
            )
        
        return decision
    
    def update(
        self,
        context: Dict,
        solution_used: str,
        size_mult: float,
        r_multiple: float
    ):
        """
        Update after trade closes.
        
        Args:
            context: Original trading context
            solution_used: What action was taken (EXECUTE/REDUCE/IGNORE)
            size_mult: Size multiplier used
            r_multiple: Trade result in R
        """
        self.total_updates += 1
        
        pattern = encode_context(context)
        self.learner.update(self.memory, pattern, solution_used, size_mult, r_multiple)
    
    def decay_all(self):
        """Apply time decay. Call daily."""
        self.memory.apply_decay()
    
    def save(self):
        """Persist to disk."""
        self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.persistence_path, 'wb') as f:
            pickle.dump({
                'records': self.memory.records,
                'access_order': dict(self.memory.access_order)
            }, f)
        logger.info(f"[ERM] Saved {len(self.memory.records)} patterns to {self.persistence_path}")
    
    def _load(self):
        """Load from disk."""
        try:
            with open(self.persistence_path, 'rb') as f:
                data = pickle.load(f)
                self.memory.records = data.get('records', {})
                self.memory.access_order = OrderedDict(data.get('access_order', {}))
        except Exception as e:
            logger.warning(f"[ERM] Failed to load: {e}")
    
    def stats(self) -> Dict[str, Any]:
        """Get ERM statistics."""
        return {
            'total_evaluations': self.total_evaluations,
            'total_updates': self.total_updates,
            'memory_stats': self.memory.stats()
        }
    
    def print_stats(self):
        """Print human-readable stats."""
        stats = self.stats()
        print("\n" + "=" * 50)
        print("EXPERIENCE REASONING MODULE STATS")
        print("=" * 50)
        print(f"Total Evaluations: {stats['total_evaluations']}")
        print(f"Total Updates: {stats['total_updates']}")
        print(f"Patterns in Memory: {stats['memory_stats']['total_patterns']}")
        print(f"Memory Utilization: {stats['memory_stats']['utilization']:.1%}")
        print("=" * 50)


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_erm_instance: Optional[ExperienceReasoningModule] = None


def get_erm() -> ExperienceReasoningModule:
    """Get or create the ERM singleton."""
    global _erm_instance
    if _erm_instance is None:
        _erm_instance = ExperienceReasoningModule()
    return _erm_instance


def reset_erm():
    """Reset the ERM singleton (for testing)."""
    global _erm_instance
    _erm_instance = None
