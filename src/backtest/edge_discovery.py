"""
MARK-3.1: A-GRADE EDGE DISCOVERY
================================

READ-ONLY observation module for discovering true A-grade edges.

Philosophy: "Observe first. Upgrade later."

This module:
- Logs ALL signals (executed OR blocked) with full context
- Tags A-candidates (EDGE >= 0.72) for analysis
- Outputs to structured CSV/JSONL for post-run analysis
- Does NOT change any trading behavior

Integration Point:
- AFTER EDGE_SCORE calculation
- BEFORE MARK-2 modifiers are applied
- Does NOT alter pipeline order or decisions
"""

import csv
import json
import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

A_CANDIDATE_THRESHOLD = 0.72  # Tag as A-candidate if EDGE >= this
OUTPUT_DIR = Path("logs/edge_discovery")


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class EdgeContextLog:
    """
    Complete context for every signal evaluated.
    One row per signal (executed OR blocked).
    """
    # Identification
    timestamp: str  # UTC ISO format
    symbol: str
    
    # Session context
    session: str  # SYDNEY / TOKYO / LONDON / NEW_YORK / OFF
    
    # EDGE_SCORE (MARK-3)
    edge_score: float  # 0.0-1.0
    edge_tier: str  # C / B / A-candidate
    is_a_candidate: bool  # True if EDGE >= 0.72
    
    # Regime context (MARK-2)
    regime_strength: float  # 0.0-1.0
    regime_type: str  # RANGE / TREND / DANGER
    
    # Risk/Reward context
    rr_ratio: float
    rr_bucket: str  # 1.13 / 1.27 / 1.41 / 1.62
    
    # Volatility context
    atr: float
    atr_avg: float
    volatility_bucket: str  # LOW / MID / HIGH
    
    # Structure context
    distance_from_htf_sr: str  # NEAR / MID / FAR
    sr_strength: float
    
    # Decision
    decision: str  # ALLOW / BLOCK
    block_reason: str = ""
    
    # Outcome (filled post-trade if executed)
    outcome_r: Optional[float] = None
    trade_pnl: Optional[float] = None
    was_executed: bool = False
    
    # ML confidence
    ml_confidence: float = 0.0
    ml_side: str = ""
    
    # Size modifiers
    edge_multiplier: float = 1.0
    mark2_modifier: float = 1.0
    session_modifier: float = 1.0
    final_size: float = 0.0


# ============================================================================
# EDGE DISCOVERY MODULE
# ============================================================================

class EdgeDiscoveryModule:
    """
    READ-ONLY observation module for A-grade edge discovery.
    
    Records every signal with full context for post-analysis.
    Does NOT change any trading behavior.
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize edge discovery module."""
        self.output_dir = output_dir or OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Session file paths
        self.session_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.csv_path = self.output_dir / f"edge_discovery_{self.session_id}.csv"
        self.jsonl_path = self.output_dir / f"edge_discovery_{self.session_id}.jsonl"
        
        # In-memory buffer
        self.logs: List[EdgeContextLog] = []
        self.a_candidates: List[EdgeContextLog] = []
        
        # Stats
        self.total_signals = 0
        self.total_a_candidates = 0
        self.a_candidate_by_session: Dict[str, int] = {}
        
        # CSV header written flag
        self._csv_header_written = False
        
        logger.info(f"[EDGE_DISCOVERY] Initialized - output: {self.csv_path}")
    
    def log_signal(
        self,
        timestamp: datetime,
        symbol: str,
        session: str,
        edge_score: float,
        regime_strength: float,
        regime_type: str,
        rr_ratio: float,
        atr: float,
        atr_avg: float,
        sr_strength: float,
        decision: str,
        block_reason: str = "",
        ml_confidence: float = 0.0,
        ml_side: str = "",
        edge_multiplier: float = 1.0,
        mark2_modifier: float = 1.0,
        session_modifier: float = 1.0,
        final_size: float = 0.0,
    ) -> EdgeContextLog:
        """
        Log a signal with full context.
        
        This is the main entry point for edge discovery.
        Call this for EVERY signal, executed or blocked.
        
        Returns:
            EdgeContextLog for further enrichment
        """
        # Determine edge tier
        if edge_score >= A_CANDIDATE_THRESHOLD:
            edge_tier = "A-candidate"
            is_a_candidate = True
        elif edge_score >= 0.60:
            edge_tier = "B"
            is_a_candidate = False
        elif edge_score >= 0.50:
            edge_tier = "C"
            is_a_candidate = False
        else:
            edge_tier = "D"
            is_a_candidate = False
        
        # Determine RR bucket
        rr_bucket = self._classify_rr(rr_ratio)
        
        # Determine volatility bucket
        volatility_bucket = self._classify_volatility(atr, atr_avg)
        
        # Determine distance from HTF S/R
        distance_from_htf_sr = self._classify_sr_distance(sr_strength)
        
        # Create log entry
        log = EdgeContextLog(
            timestamp=timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
            symbol=symbol,
            session=session,
            edge_score=round(edge_score, 4),
            edge_tier=edge_tier,
            is_a_candidate=is_a_candidate,
            regime_strength=round(regime_strength, 4),
            regime_type=regime_type,
            rr_ratio=round(rr_ratio, 4),
            rr_bucket=rr_bucket,
            atr=round(atr, 6),
            atr_avg=round(atr_avg, 6),
            volatility_bucket=volatility_bucket,
            distance_from_htf_sr=distance_from_htf_sr,
            sr_strength=round(sr_strength, 4),
            decision=decision,
            block_reason=block_reason,
            ml_confidence=round(ml_confidence, 4),
            ml_side=ml_side,
            edge_multiplier=round(edge_multiplier, 4),
            mark2_modifier=round(mark2_modifier, 4),
            session_modifier=round(session_modifier, 4),
            final_size=round(final_size, 4),
        )
        
        # Store in memory
        self.logs.append(log)
        self.total_signals += 1
        
        # Track A-candidates
        if is_a_candidate:
            self.a_candidates.append(log)
            self.total_a_candidates += 1
            self.a_candidate_by_session[session] = self.a_candidate_by_session.get(session, 0) + 1
            
            # Log A-candidate probe
            logger.info(
                f"[EDGE_PROBE] {symbol} | {session} | "
                f"score={edge_score:.3f} | regime={regime_strength:.2f} | "
                f"RR={rr_ratio:.2f} | {decision}"
            )
        
        # Write to files incrementally
        self._write_log(log)
        
        return log
    
    def record_outcome(self, log: EdgeContextLog, outcome_r: float, pnl: float):
        """Record trade outcome for executed trades."""
        log.outcome_r = round(outcome_r, 4)
        log.trade_pnl = round(pnl, 4)
        log.was_executed = True
    
    def _classify_rr(self, rr: float) -> str:
        """Classify RR ratio into Fibonacci buckets."""
        if rr >= 1.55:
            return "1.62"
        elif rr >= 1.34:
            return "1.41"
        elif rr >= 1.20:
            return "1.27"
        else:
            return "1.13"
    
    def _classify_volatility(self, atr: float, atr_avg: float) -> str:
        """Classify volatility based on ATR ratio."""
        if atr_avg <= 0:
            return "MID"
        
        ratio = atr / atr_avg
        if ratio < 0.8:
            return "LOW"
        elif ratio > 1.2:
            return "HIGH"
        else:
            return "MID"
    
    def _classify_sr_distance(self, sr_strength: float) -> str:
        """Classify distance from HTF S/R based on strength."""
        if sr_strength >= 0.7:
            return "NEAR"
        elif sr_strength >= 0.4:
            return "MID"
        else:
            return "FAR"
    
    def _to_python(self, obj):
        """Convert numpy types to Python native types for JSON serialization."""
        import numpy as np
        if isinstance(obj, dict):
            return {k: self._to_python(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._to_python(v) for v in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    def _write_log(self, log: EdgeContextLog):
        """Write log entry to CSV and JSONL files."""
        try:
            log_dict = asdict(log)
            
            # Write to CSV (native dict works fine)
            with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=log_dict.keys())
                if not self._csv_header_written:
                    writer.writeheader()
                    self._csv_header_written = True
                writer.writerow(log_dict)
            
            # Write to JSONL (convert numpy types first)
            safe_dict = self._to_python(log_dict)
            with open(self.jsonl_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(safe_dict) + '\n')
                
        except Exception as e:
            logger.warning(f"[EDGE_DISCOVERY] Failed to write log: {e}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for post-run analysis."""
        return {
            "total_signals": self.total_signals,
            "total_a_candidates": self.total_a_candidates,
            "a_candidate_rate": self.total_a_candidates / max(1, self.total_signals),
            "a_candidates_by_session": self.a_candidate_by_session,
            "output_csv": str(self.csv_path),
            "output_jsonl": str(self.jsonl_path),
        }
    
    def print_summary(self):
        """Print end-of-run summary."""
        summary = self.get_summary()
        
        print("\n" + "=" * 60)
        print("MARK-3.1 EDGE DISCOVERY SUMMARY")
        print("=" * 60)
        print(f"Total Signals Logged: {summary['total_signals']}")
        print(f"A-Candidates Found:   {summary['total_a_candidates']} ({summary['a_candidate_rate']*100:.1f}%)")
        print("\nA-Candidates by Session:")
        for session, count in summary['a_candidates_by_session'].items():
            print(f"  {session}: {count}")
        print(f"\nOutput Files:")
        print(f"  CSV:   {summary['output_csv']}")
        print(f"  JSONL: {summary['output_jsonl']}")
        print("=" * 60 + "\n")


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_discovery_instance: Optional[EdgeDiscoveryModule] = None


def get_edge_discovery() -> EdgeDiscoveryModule:
    """Get or create the edge discovery singleton."""
    global _discovery_instance
    if _discovery_instance is None:
        _discovery_instance = EdgeDiscoveryModule()
    return _discovery_instance


def reset_edge_discovery():
    """Reset the edge discovery module (for new backtest runs)."""
    global _discovery_instance
    _discovery_instance = None
