"""
Shadow Validation Framework

Compare old standalone FX system vs new orchestrated alpha pod.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Validation comparison result."""
    timestamp: datetime
    
    # Old system outputs
    old_signals: List[Dict] = field(default_factory=list)
    old_rankings: Optional[pd.Series] = None
    old_positions: Dict = field(default_factory=dict)
    old_returns: float = 0.0
    
    # New system outputs
    new_signals: List[Dict] = field(default_factory=list)
    new_rankings: Optional[pd.Series] = None
    new_positions: Dict = field(default_factory=dict)
    new_returns: float = 0.0
    
    # Comparison
    signal_match: bool = False
    ranking_match: bool = False
    position_match: bool = False
    return_diff: float = 0.0
    return_diff_pct: float = 0.0
    
    # Mismatch details
    mismatches: List[str] = field(default_factory=list)


class ShadowValidationRunner:
    """
    Run old and new systems in parallel for validation.
    
    Compares:
    - Signals
    - Rankings
    - Positions
    - Returns
    - Risk outputs
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize shadow validation.
        
        Args:
            config: Configuration
        """
        self.config = config
        
        # Tolerance for matching
        self.tolerance_bps = config.get('tolerance_bps', 1.0)  # 1 basis point
        self.tolerance_rank = config.get('tolerance_rank', 0)  # Exact match
        
        # Results storage
        self.results: List[ValidationResult] = []
        self.mismatch_count = 0
        
        logger.info("ShadowValidationRunner initialized")
    
    def compare_outputs(self,
                       old_output: Dict[str, Any],
                       new_output: Dict[str, Any]) -> ValidationResult:
        """
        Compare old vs new outputs.
        
        Args:
            old_output: Old standalone system output
            new_output: New orchestrated system output
            
        Returns:
            ValidationResult
        """
        result = ValidationResult(timestamp=datetime.now())
        
        # Extract signals
        result.old_signals = old_output.get('signals', [])
        result.new_signals = new_output.get('signals', [])
        
        # Compare signals
        result.signal_match = self._compare_signals(
            result.old_signals,
            result.new_signals
        )
        
        if not result.signal_match:
            result.mismatches.append("SIGNAL_MISMATCH")
        
        # Compare rankings
        result.old_rankings = old_output.get('rankings')
        result.new_rankings = new_output.get('rankings')
        
        if result.old_rankings is not None and result.new_rankings is not None:
            result.ranking_match = self._compare_rankings(
                result.old_rankings,
                result.new_rankings
            )
            
            if not result.ranking_match:
                result.mismatches.append("RANKING_MISMATCH")
        
        # Compare positions
        result.old_positions = old_output.get('positions', {})
        result.new_positions = new_output.get('positions', {})
        result.position_match = self._compare_positions(
            result.old_positions,
            result.new_positions
        )
        
        if not result.position_match:
            result.mismatches.append("POSITION_MISMATCH")
        
        # Compare returns
        result.old_returns = old_output.get('returns', 0.0)
        result.new_returns = new_output.get('returns', 0.0)
        result.return_diff = result.new_returns - result.old_returns
        
        if result.old_returns != 0:
            result.return_diff_pct = result.return_diff / abs(result.old_returns)
        
        # Log result
        if result.mismatches:
            self.mismatch_count += 1
            logger.warning(f"Validation mismatches: {result.mismatches}")
        else:
            logger.info("Validation: OK")
        
        self.results.append(result)
        return result
    
    def _compare_signals(self, 
                        old_signals: List[Dict],
                        new_signals: List[Dict]) -> bool:
        """
        Compare signal outputs.
        
        Args:
            old_signals: Old system signals
            new_signals: New system signals
            
        Returns:
            bool: Match
        """
        if len(old_signals) != len(new_signals):
            return False
        
        for old, new in zip(old_signals, new_signals):
            # Compare symbol
            if old.get('symbol') != new.get('symbol'):
                return False
            
            # Compare direction
            if old.get('direction') != new.get('direction'):
                return False
            
            # Compare size (within tolerance)
            old_size = old.get('size', 0)
            new_size = new.get('size', 0)
            
            if old_size > 0:
                size_diff_pct = abs(new_size - old_size) / old_size
                if size_diff_pct > self.tolerance_bps / 10000:
                    return False
        
        return True
    
    def _compare_rankings(self,
                         old_rankings: pd.Series,
                         new_rankings: pd.Series) -> bool:
        """Compare pair rankings."""
        if len(old_rankings) != len(new_rankings):
            return False
        
        if not all(old_rankings.index == new_rankings.index):
            return False
        
        # Compare values within tolerance
        for pair in old_rankings.index:
            old_val = old_rankings[pair]
            new_val = new_rankings[pair]
            
            if abs(new_val - old_val) > self.tolerance_bps / 100:
                return False
        
        return True
    
    def _compare_positions(self,
                          old_positions: Dict,
                          new_positions: Dict) -> bool:
        """Compare positions."""
        if set(old_positions.keys()) != set(new_positions.keys()):
            return False
        
        for symbol in old_positions:
            old_pos = old_positions[symbol]
            new_pos = new_positions[symbol]
            
            if old_pos != new_pos:
                return False
        
        return True
    
    def get_summary(self) -> Dict[str, Any]:
        """Get validation summary."""
        if not self.results:
            return {}
        
        total = len(self.results)
        mismatches = self.mismatch_count
        
        return {
            'total_validations': total,
            'mismatches': mismatches,
            'match_rate': (total - mismatches) / total if total > 0 else 0,
            'avg_return_diff': np.mean([r.return_diff for r in self.results]),
            'max_return_diff': np.max([abs(r.return_diff) for r in self.results])
        }
    
    def is_migration_safe(self, threshold: float = 0.99) -> bool:
        """
        Check if migration is safe.
        
        Args:
            threshold: Minimum match rate (default 99%)
            
        Returns:
            bool: Safe to migrate
        """
        summary = self.get_summary()
        match_rate = summary.get('match_rate', 0)
        
        return match_rate >= threshold


class ShadowRunner:
    """
    Run both systems in parallel.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.validator = ShadowValidationRunner(config)
        
        # Mode
        self.shadow_mode = config.get('shadow_mode', True)
        
        logger.info("ShadowRunner initialized")
        logger.info(f"  Shadow mode: {self.shadow_mode}")
    
    async def run_iteration(self,
                         old_system,  # Old standalone
                         new_system,  # New orchestrated
                         data):
        """
        Run one iteration of both systems.
        
        Args:
            old_system: Old standalone system
            new_system: New orchestrated system
            data: Market data
        """
        # Run old system
        old_output = await old_system.process(data)
        
        # Run new system
        new_output = await new_system.process(data)
        
        # Compare
        result = self.validator.compare_outputs(old_output, new_output)
        
        # If shadow mode, only use old system for trading
        if self.shadow_mode:
            return old_output
        else:
            # Use new system
            return new_output
    
    def get_validation_report(self) -> str:
        """Get validation report."""
        summary = self.validator.get_summary()
        
        report = f"""
Shadow Validation Report
{'='*60}
Total Validations: {summary.get('total_validations', 0)}
Mismatches: {summary.get('mismatches', 0)}
Match Rate: {summary.get('match_rate', 0):.2%}

Return Comparison:
  Avg Difference: {summary.get('avg_return_diff', 0):.6f}
  Max Difference: {summary.get('max_return_diff', 0):.6f}

Migration Status: {'SAFE' if self.validator.is_migration_safe() else 'NOT SAFE'}
{'='*60}
"""
        return report
