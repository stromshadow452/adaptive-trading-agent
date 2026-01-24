"""
WEAPON SYSTEM: Decision Logging
================================

Logs all routing decisions for post-run analysis.
Output: CSV/JSONL to logs/weapon_system/
"""

import csv
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class WeaponDecisionLog:
    """Log entry for every routing decision."""
    timestamp: str
    symbol: str
    session: str
    
    # Strategy selection
    strategy_selected: str       # RIFLE / SCALPEL / NONE
    strategy_name: str           # e.g., "range_micro_mr"
    weapon_class: str            # RIFLE / SCALPEL / SHIELD
    signal: str                  # BUY / SELL / HOLD
    decision_reason: str
    
    # Context
    ml_confidence: float
    ml_prediction: str
    edge_score: float
    edge_tier: str
    regime: str
    regime_strength: float
    
    # MARK-2
    mark2_can_trade: bool
    mark2_modifier: float
    
    # Sizing
    edge_multiplier: float
    strategy_size_mult: float
    final_size: float
    
    # Block info
    was_blocked: bool
    block_reason: str


class WeaponDecisionLogger:
    """Logs weapon system decisions to CSV/JSONL files."""
    
    def __init__(self, output_dir: Path = None, format: str = 'jsonl'):
        self.output_dir = output_dir or Path("logs/weapon_system")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.format = format
        self.session_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        
        # File paths
        self.csv_path = self.output_dir / f"weapon_decisions_{self.session_id}.csv"
        self.jsonl_path = self.output_dir / f"weapon_decisions_{self.session_id}.jsonl"
        
        # Stats
        self.total_decisions = 0
        self.rifle_count = 0
        self.scalpel_count = 0
        self.blocked_count = 0
        
        self._csv_header_written = False
        
        logger.info(f"[WEAPON_LOG] Initialized - output: {self.output_dir}")
    
    def log_decision(
        self,
        timestamp: datetime,
        symbol: str,
        session: str,
        strategy_selected: str,
        strategy_name: str,
        weapon_class: str,
        signal: str,
        decision_reason: str,
        ml_confidence: float,
        ml_prediction: str,
        edge_score: float,
        edge_tier: str,
        regime: str,
        regime_strength: float,
        mark2_can_trade: bool,
        mark2_modifier: float,
        edge_multiplier: float,
        strategy_size_mult: float,
        final_size: float,
        was_blocked: bool,
        block_reason: str = ""
    ):
        """Log a routing decision."""
        
        log = WeaponDecisionLog(
            timestamp=timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
            symbol=symbol,
            session=session,
            strategy_selected=strategy_selected,
            strategy_name=strategy_name,
            weapon_class=weapon_class,
            signal=signal,
            decision_reason=decision_reason,
            ml_confidence=round(ml_confidence, 4),
            ml_prediction=ml_prediction,
            edge_score=round(edge_score, 4),
            edge_tier=edge_tier,
            regime=regime,
            regime_strength=round(regime_strength, 4),
            mark2_can_trade=mark2_can_trade,
            mark2_modifier=round(mark2_modifier, 4),
            edge_multiplier=round(edge_multiplier, 4),
            strategy_size_mult=round(strategy_size_mult, 4),
            final_size=round(final_size, 4),
            was_blocked=was_blocked,
            block_reason=block_reason
        )
        
        # Update stats
        self.total_decisions += 1
        if was_blocked:
            self.blocked_count += 1
        elif weapon_class == 'RIFLE':
            self.rifle_count += 1
        elif weapon_class == 'SCALPEL':
            self.scalpel_count += 1
        
        # Write to files
        self._write_log(log)
        
        # Console log for significant events
        if strategy_selected == 'SCALPEL':
            logger.info(
                f"[WEAPON] {symbol} SCALPEL:{strategy_name} | "
                f"{signal} | size×{strategy_size_mult:.2f}"
            )
    
    def _write_log(self, log: WeaponDecisionLog):
        """Write log to file."""
        try:
            log_dict = asdict(log)
            
            # Convert any numpy types
            for k, v in log_dict.items():
                if hasattr(v, 'item'):
                    log_dict[k] = v.item()
            
            # Write to JSONL
            with open(self.jsonl_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_dict) + '\n')
            
            # Write to CSV
            with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=log_dict.keys())
                if not self._csv_header_written:
                    writer.writeheader()
                    self._csv_header_written = True
                writer.writerow(log_dict)
                
        except Exception as e:
            logger.warning(f"[WEAPON_LOG] Failed to write: {e}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get logging summary."""
        return {
            'total_decisions': self.total_decisions,
            'rifle_count': self.rifle_count,
            'scalpel_count': self.scalpel_count,
            'blocked_count': self.blocked_count,
            'scalpel_rate': self.scalpel_count / max(1, self.total_decisions),
            'output_files': {
                'csv': str(self.csv_path),
                'jsonl': str(self.jsonl_path)
            }
        }
    
    def print_summary(self):
        """Print end-of-run summary."""
        summary = self.get_summary()
        print("\n" + "=" * 50)
        print("WEAPON SYSTEM LOGGING SUMMARY")
        print("=" * 50)
        print(f"Total Decisions: {summary['total_decisions']}")
        print(f"RIFLE Trades: {summary['rifle_count']}")
        print(f"SCALPEL Trades: {summary['scalpel_count']} ({summary['scalpel_rate']*100:.1f}%)")
        print(f"Blocked: {summary['blocked_count']}")
        print(f"\nOutput: {summary['output_files']['jsonl']}")
        print("=" * 50)
