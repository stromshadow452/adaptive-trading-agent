"""
Asset Tier Configuration

Professional selective activation based on ML Brain performance.
Trade where the brain is strong, block where it's weak.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum


class AssetTier(Enum):
    """Asset trading tiers."""
    BLOCKED = 'BLOCKED'      # No signals, veto mode
    SHADOW = 'SHADOW'        # Signal-only, no execution
    REDUCED = 'REDUCED'      # Trade at 25% size
    PRIMARY = 'PRIMARY'      # Full activation
    HOT = 'HOT'              # High confidence, can boost size


@dataclass
class AssetConfig:
    """Configuration for a single asset."""
    tier: AssetTier
    size_multiplier: float = 1.0
    min_confidence: float = 0.55
    # Choppy engine configuration
    choppy_enabled: bool = False
    choppy_size_mult: float = 0.25  # 25% of normal size for choppy
    choppy_min_confidence: float = 0.50
    notes: str = ""


# ============================================================================
# ASSET TIER CONFIGURATION (Based on Phase-2B backtest results)
# ============================================================================

ASSET_TIERS: Dict[str, AssetConfig] = {
    # =========================================================================
    # PRODUCTION ASSETS (Validated in Phase-2B backtesting)
    # =========================================================================
    
    # HOT: Star performer (holdout asset that generalized well)
    'EURJPY': AssetConfig(
        tier=AssetTier.HOT,
        size_multiplier=1.0,
        min_confidence=0.50,
        notes="PF 1.25, WR 47.2%, DD 13.4% - Best overall performer"
    ),
    
    # PRIMARY: Strong performer
    'AUDUSD': AssetConfig(
        tier=AssetTier.PRIMARY,
        size_multiplier=1.0,
        min_confidence=0.50,
        notes="PF 1.10, WR 40.2%, DD 16.1% - Consistent"
    ),
    
    # =========================================================================
    # SHADOW ASSETS (Not ready for production)
    # =========================================================================
    
    # SHADOW: Weak in TREND but testing CHOPPY engine
    'EURUSD': AssetConfig(
        tier=AssetTier.SHADOW,
        size_multiplier=0.0,  # No trend trading
        min_confidence=0.60,
        choppy_enabled=True,  # Enable choppy engine
        choppy_size_mult=0.25,  # Very conservative
        choppy_min_confidence=0.55,
        notes="PF 0.91 in trend - testing choppy mean-reversion"
    ),
    
    'GBPUSD': AssetConfig(
        tier=AssetTier.SHADOW,
        size_multiplier=0.0,
        min_confidence=0.55,
        notes="PF 0.86-1.01 - Inconsistent, high DD"
    ),
    
    'GBPJPY': AssetConfig(
        tier=AssetTier.SHADOW,
        size_multiplier=0.0,
        min_confidence=0.55,
        notes="PF 0.95-0.99 - Marginal, needs more validation"
    ),
    
    # =========================================================================
    # BLOCKED ASSETS (Not tested)
    # =========================================================================
    
    'USDJPY': AssetConfig(
        tier=AssetTier.BLOCKED,
        size_multiplier=0.0,
        min_confidence=0.60,
        notes="Not validated in Phase-2B"
    ),
    
    'USDCAD': AssetConfig(
        tier=AssetTier.BLOCKED,
        size_multiplier=0.0,
        min_confidence=0.60,
        notes="Not validated in Phase-2B"
    ),
    
    'USDCHF': AssetConfig(
        tier=AssetTier.BLOCKED,
        size_multiplier=0.0,
        min_confidence=0.60,
        notes="Not validated in Phase-2B"
    ),
    
    'NZDUSD': AssetConfig(
        tier=AssetTier.BLOCKED,
        size_multiplier=0.0,
        min_confidence=0.60,
        notes="Not validated in Phase-2B"
    ),
}


# Default for unknown assets
DEFAULT_CONFIG = AssetConfig(
    tier=AssetTier.BLOCKED,
    size_multiplier=0.0,
    min_confidence=0.65,
    notes="Unknown asset - blocked by default"
)


# ============================================================================
# CONFIGURATION FUNCTIONS
# ============================================================================

def get_asset_config(symbol: str) -> AssetConfig:
    """Get configuration for an asset."""
    return ASSET_TIERS.get(symbol, DEFAULT_CONFIG)


def is_tradeable(symbol: str) -> bool:
    """Check if asset is tradeable (not BLOCKED or SHADOW)."""
    config = get_asset_config(symbol)
    return config.tier in [AssetTier.PRIMARY, AssetTier.HOT, AssetTier.REDUCED]


def get_size_multiplier(symbol: str) -> float:
    """Get position size multiplier for asset."""
    config = get_asset_config(symbol)
    return config.size_multiplier


def get_min_confidence(symbol: str) -> float:
    """Get minimum ML confidence for asset."""
    config = get_asset_config(symbol)
    return config.min_confidence


def get_tradeable_assets() -> List[str]:
    """Get list of tradeable assets."""
    return [s for s, c in ASSET_TIERS.items() if c.tier in [AssetTier.PRIMARY, AssetTier.HOT, AssetTier.REDUCED]]


def get_primary_assets() -> List[str]:
    """Get list of primary/hot assets."""
    return [s for s, c in ASSET_TIERS.items() if c.tier in [AssetTier.PRIMARY, AssetTier.HOT]]


def print_tier_summary():
    """Print current tier configuration."""
    print("\n" + "=" * 60)
    print(" ASSET TIER CONFIGURATION")
    print("=" * 60)
    
    for tier in AssetTier:
        assets = [s for s, c in ASSET_TIERS.items() if c.tier == tier]
        if assets:
            print(f"\n {tier.value}:")
            for asset in assets:
                config = ASSET_TIERS[asset]
                print(f"   {asset}: size={config.size_multiplier:.0%}, min_conf={config.min_confidence}")
                if config.notes:
                    print(f"      ({config.notes})")


# ============================================================================
# MAIN (for testing)
# ============================================================================

if __name__ == '__main__':
    print_tier_summary()
    
    print("\n" + "-" * 60)
    print(" TRADEABLE ASSETS:")
    print(f"   {', '.join(get_tradeable_assets())}")
    
    print("\n PRIMARY/HOT ASSETS:")
    print(f"   {', '.join(get_primary_assets())}")
