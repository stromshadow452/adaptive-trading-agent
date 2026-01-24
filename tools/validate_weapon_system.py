"""
WEAPON SYSTEM: Shadow Mode Validation
=====================================

Tests the strategy router in shadow mode without real trades.
"""

import logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test imports
try:
    from src.strategy import (
        StrategyRouter, MarketSnapshot, AgentContext,
        RangeMicroMR, LiquiditySweepFade, MLPrimaryStrategy
    )
    print("✅ All strategy imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    exit(1)


def test_router_decisions():
    """Test router decision tree."""
    print("\n" + "=" * 60)
    print("STRATEGY ROUTER DECISION TREE TESTS")
    print("=" * 60)
    
    router = StrategyRouter(enable_micro=True)
    
    # Test snapshot
    snap = MarketSnapshot(
        timestamp=datetime.now(),
        symbol='EURUSD',
        price=1.0800,
        atr=0.0015,
        atr_avg=0.0012,
        high_20=1.0820,
        low_20=1.0780,
        sma_20=1.0800,
        bb_upper=1.0830,
        bb_lower=1.0770
    )
    
    # Test 1: High confidence → RIFLE
    print("\n1. High confidence (0.70) + B-tier → RIFLE")
    ctx = AgentContext(
        ml_confidence=0.70,
        ml_prediction='BUY',
        edge_score=0.68,
        edge_tier='B',
        regime='TREND',
        regime_strength=0.6,
        session='LONDON',
        mark2_can_trade=True,
        base_size=1000,
        rr_ratio=1.5
    )
    strat, sig = router.route(snap, ctx)
    print(f"   Result: {strat.name if strat else 'None'}, Signal: {sig.signal.value if sig else 'None'}")
    assert strat and strat.name == 'ml_primary', "Expected ml_primary strategy"
    print("   ✅ PASSED")
    
    # Test 2: MARK-2 veto
    print("\n2. MARK-2 veto (can_trade=False) → BLOCKED")
    ctx2 = AgentContext(
        ml_confidence=0.80,
        ml_prediction='BUY',
        edge_score=0.75,
        edge_tier='A',
        regime='TREND',
        regime_strength=0.5,
        session='LONDON',
        mark2_can_trade=False  # VETO
    )
    strat2, sig2 = router.route(snap, ctx2)
    print(f"   Result: {strat2.name if strat2 else 'None'}")
    assert strat2 is None, "Expected MARK-2 veto"
    print("   ✅ PASSED")
    
    # Test 3: DANGER regime
    print("\n3. DANGER regime → BLOCKED")
    ctx3 = AgentContext(
        ml_confidence=0.70,
        ml_prediction='BUY',
        edge_score=0.68,
        edge_tier='B',
        regime='DANGER',  # DANGER
        regime_strength=0.9,
        session='LONDON',
        mark2_can_trade=True
    )
    strat3, sig3 = router.route(snap, ctx3)
    print(f"   Result: {strat3.name if strat3 else 'None'}")
    assert strat3 is None, "Expected DANGER block"
    print("   ✅ PASSED")
    
    # Test 4: OFF session
    print("\n4. OFF session → BLOCKED")
    ctx4 = AgentContext(
        ml_confidence=0.70,
        ml_prediction='BUY',
        edge_score=0.68,
        edge_tier='B',
        regime='RANGE',
        regime_strength=0.5,
        session='OFF',  # OFF
        mark2_can_trade=True
    )
    strat4, sig4 = router.route(snap, ctx4)
    print(f"   Result: {strat4.name if strat4 else 'None'}")
    assert strat4 is None, "Expected OFF session block"
    print("   ✅ PASSED")
    
    # Test 5: Low confidence C-tier → Try SCALPEL
    print("\n5. Low confidence (0.55) + C-tier + extremes → SCALPEL")
    snap_extreme = MarketSnapshot(
        timestamp=datetime.now(),
        symbol='EURUSD',
        price=1.0828,  # Near upper BB
        atr=0.0015,
        atr_avg=0.0012,
        high_20=1.0820,
        low_20=1.0780,
        sma_20=1.0800,
        bb_upper=1.0830,
        bb_lower=1.0770
    )
    ctx5 = AgentContext(
        ml_confidence=0.55,
        ml_prediction='HOLD',
        edge_score=0.55,
        edge_tier='C',
        regime='RANGE',
        regime_strength=0.4,
        session='TOKYO',
        mark2_can_trade=True,
        memory_pain_level=0.2
    )
    strat5, sig5 = router.route(snap_extreme, ctx5)
    print(f"   Result: {strat5.name if strat5 else 'None'}, Signal: {sig5.signal.value if sig5 else 'None'}")
    if strat5:
        print(f"   Size mult: {strat5.get_size_multiplier()}")
        print("   ✅ PASSED (Scalpel triggered)")
    else:
        print("   ⚠️ No scalpel triggered (extremes not detected)")
    
    # Test 6: Cooldown active
    print("\n6. MARK-2 cooldown active → BLOCKED")
    ctx6 = AgentContext(
        ml_confidence=0.70,
        ml_prediction='BUY',
        edge_score=0.68,
        edge_tier='B',
        regime='RANGE',
        regime_strength=0.5,
        session='LONDON',
        mark2_can_trade=True,
        mark2_cooldown_min=5.0  # Cooldown active
    )
    strat6, sig6 = router.route(snap, ctx6)
    print(f"   Result: {strat6.name if strat6 else 'None'}")
    assert strat6 is None, "Expected cooldown block"
    print("   ✅ PASSED")
    
    # Print final stats
    router.print_stats()
    
    print("\n" + "=" * 60)
    print("ALL DECISION TREE TESTS PASSED ✅")
    print("=" * 60)


def test_size_multipliers():
    """Test size multipliers for different strategies."""
    print("\n" + "=" * 60)
    print("SIZE MULTIPLIER TESTS")
    print("=" * 60)
    
    from src.strategy import RangeMicroMR, LiquiditySweepFade, MLPrimaryStrategy
    
    micro_mr = RangeMicroMR()
    micro_sweep = LiquiditySweepFade()
    rifle = MLPrimaryStrategy()
    
    print(f"\nRangeMicroMR size mult: {micro_mr.get_size_multiplier()} (expected 0.12)")
    assert micro_mr.get_size_multiplier() == 0.12
    print("✅ PASSED")
    
    print(f"LiquiditySweepFade size mult: {micro_sweep.get_size_multiplier()} (expected 0.12)")
    assert micro_sweep.get_size_multiplier() == 0.12
    print("✅ PASSED")
    
    print(f"MLPrimaryStrategy size mult: {rifle.get_size_multiplier()} (expected 1.0)")
    assert rifle.get_size_multiplier() == 1.0
    print("✅ PASSED")
    
    print("\n" + "=" * 60)
    print("ALL SIZE TESTS PASSED ✅")
    print("=" * 60)


if __name__ == "__main__":
    test_router_decisions()
    test_size_multipliers()
    print("\n🎯 WEAPON SYSTEM VALIDATION COMPLETE 🎯")
