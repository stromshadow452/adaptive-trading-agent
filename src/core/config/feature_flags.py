"""
Feature Flags for Safe Migration

Controls gradual rollout of unified orchestrator.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class FeatureFlags:
    """
    Feature flags for safe migration.
    
    All flags default to False for backward compatibility.
    Enable incrementally during migration.
    """
    
    # Core orchestrator
    USE_UNIFIED_ORCHESTRATOR: bool = False
    """Enable new orchestrator (Phase 0)"""
    
    # Alpha pods
    USE_ALPHA_PODS: bool = False
    """Enable alpha pod architecture (Phase 1)"""
    
    USE_FX_MOMENTUM_POD: bool = False
    """Enable FX momentum as alpha pod (Phase 1)"""
    
    USE_EURUSD_ML_POD: bool = False
    """Enable EURUSD ML as alpha pod (Phase 2)"""
    
    # Pipeline
    USE_UNIFIED_PIPELINE: bool = False
    """Enable unified 13-stage pipeline"""
    
    USE_PIPELINE_ENSEMBLE: bool = False
    """Enable alpha signal ensemble in pipeline"""
    
    # Services
    USE_SHARED_DATA_SERVICE: bool = False
    """Enable shared data service"""
    
    USE_SHARED_RISK_SERVICE: bool = False
    """Enable shared risk service"""
    
    USE_SHARED_EXECUTION_SERVICE: bool = False
    """Enable shared execution service"""
    
    # Portfolio
    USE_PORTFOLIO_BRAIN: bool = False
    """Enable portfolio optimization (Stage 9)"""
    
    USE_MULTI_ASSET_ALLOCATION: bool = False
    """Enable multi-asset allocation"""
    
    # Safety
    ENABLE_KILL_SWITCH: bool = True
    """Always enable kill switch"""
    
    ENABLE_CIRCUIT_BREAKER: bool = True
    """Always enable circuit breaker"""
    
    # Legacy compatibility
    KEEP_LEGACY_SYSTEM: bool = True
    """Keep legacy system running parallel (during migration)"""
    
    SHADOW_MODE: bool = True
    """Run new system in shadow mode (log only, don't trade)"""
    
    COMPARE_OUTPUTS: bool = True
    """Compare new vs old outputs"""
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'FeatureFlags':
        """Create from dict."""
        return cls(**{
            k: v for k, v in config.items()
            if k in cls.__dataclass_fields__
        })
    
    def to_dict(self) -> Dict[str, bool]:
        """Convert to dict."""
        return {
            k: getattr(self, k)
            for k in self.__dataclass_fields__
        }
    
    def validate(self) -> bool:
        """
        Validate feature flag combinations.
        
        Returns:
            bool: Valid
        """
        # Can't use alpha pods without unified orchestrator
        if self.USE_ALPHA_PODS and not self.USE_UNIFIED_ORCHESTRATOR:
            return False
        
        # Can't use FX pod without alpha pods
        if self.USE_FX_MOMENTUM_POD and not self.USE_ALPHA_PODS:
            return False
        
        # Can't use unified pipeline without orchestrator
        if self.USE_UNIFIED_PIPELINE and not self.USE_UNIFIED_ORCHESTRATOR:
            return False
        
        # Can't use shared services without orchestrator
        if self.USE_SHARED_DATA_SERVICE and not self.USE_UNIFIED_ORCHESTRATOR:
            return False
        
        return True
    
    def get_active_pods(self) -> Dict[str, bool]:
        """Get active pod flags."""
        return {
            'fx_momentum': self.USE_FX_MOMENTUM_POD,
            'eurusd_ml': self.USE_EURUSD_ML_POD
        }


# Migration phases
PHASE_0_FLAGS = FeatureFlags(
    USE_UNIFIED_ORCHESTRATOR=True,
    KEEP_LEGACY_SYSTEM=True,
    SHADOW_MODE=True
)

PHASE_1_FLAGS = FeatureFlags(
    USE_UNIFIED_ORCHESTRATOR=True,
    USE_ALPHA_PODS=True,
    USE_FX_MOMENTUM_POD=True,
    USE_UNIFIED_PIPELINE=True,
    USE_PIPELINE_ENSEMBLE=True,
    USE_SHARED_DATA_SERVICE=True,
    USE_SHARED_RISK_SERVICE=True,
    USE_SHARED_EXECUTION_SERVICE=True,
    KEEP_LEGACY_SYSTEM=True,
    SHADOW_MODE=True,
    COMPARE_OUTPUTS=True
)

PHASE_2_FLAGS = FeatureFlags(
    USE_UNIFIED_ORCHESTRATOR=True,
    USE_ALPHA_PODS=True,
    USE_FX_MOMENTUM_POD=True,
    USE_EURUSD_ML_POD=True,
    USE_UNIFIED_PIPELINE=True,
    USE_PIPELINE_ENSEMBLE=True,
    USE_PORTFOLIO_BRAIN=True,
    USE_MULTI_ASSET_ALLOCATION=True,
    USE_SHARED_DATA_SERVICE=True,
    USE_SHARED_RISK_SERVICE=True,
    USE_SHARED_EXECUTION_SERVICE=True,
    KEEP_LEGACY_SYSTEM=True,
    SHADOW_MODE=False,  # Live mode
    COMPARE_OUTPUTS=True
)

PRODUCTION_FLAGS = FeatureFlags(
    USE_UNIFIED_ORCHESTRATOR=True,
    USE_ALPHA_PODS=True,
    USE_FX_MOMENTUM_POD=True,
    USE_EURUSD_ML_POD=True,
    USE_UNIFIED_PIPELINE=True,
    USE_PIPELINE_ENSEMBLE=True,
    USE_PORTFOLIO_BRAIN=True,
    USE_MULTI_ASSET_ALLOCATION=True,
    USE_SHARED_DATA_SERVICE=True,
    USE_SHARED_RISK_SERVICE=True,
    USE_SHARED_EXECUTION_SERVICE=True,
    KEEP_LEGACY_SYSTEM=False,  # Full migration complete
    SHADOW_MODE=False,
    COMPARE_OUTPUTS=False,
    ENABLE_KILL_SWITCH=True,
    ENABLE_CIRCUIT_BREAKER=True
)


# Current flags (update as migration progresses)
CURRENT_FLAGS = PHASE_0_FLAGS
