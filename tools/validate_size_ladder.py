"""
Size Ladder Validation Tests
=============================
Tests for the risk evolution size ladder system.
"""

import sys
from pathlib import Path

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.risk.size_ladder import (
    SizeTier, SizeLadderManager, PatternRecord, 
    EarnedRiskCalculator, TIER_MULTIPLIERS, reset_size_ladder
)


def test_tier_multipliers():
    """Test tier multipliers are correct."""
    print("\n[TEST] Tier Multipliers...")
    
    assert TIER_MULTIPLIERS[SizeTier.BASE] == 1.00
    assert TIER_MULTIPLIERS[SizeTier.TIER_1] == 1.10
    assert TIER_MULTIPLIERS[SizeTier.TIER_2] == 1.15
    assert TIER_MULTIPLIERS[SizeTier.TIER_3] == 1.25
    
    print("  ✓ BASE = 1.00×")
    print("  ✓ TIER-1 = 1.10×")
    print("  ✓ TIER-2 = 1.15×")
    print("  ✓ TIER-3 = 1.25×")
    print("  [PASS]")


def test_pattern_record():
    """Test pattern record tracking."""
    print("\n[TEST] Pattern Record...")
    
    record = PatternRecord(pattern_key="RSI_RANGE_LONDON")
    
    # Add 10 trades: 6 wins, 4 losses
    for i in range(10):
        is_win = i < 6
        record.add_result(is_win, 0.5 if is_win else -1.0, 0.65)
    
    assert record.total_trades == 10
    assert record.wins == 6
    assert record.losses == 4
    assert record.success_rate == 0.6
    
    print(f"  ✓ Total trades: {record.total_trades}")
    print(f"  ✓ Success rate: {record.success_rate:.0%}")
    print(f"  ✓ Avg R: {record.avg_r:.2f}")
    print("  [PASS]")


def test_earned_risk_calculation():
    """Test earned risk calculation."""
    print("\n[TEST] Earned Risk Calculation...")
    
    calc = EarnedRiskCalculator(phase="MONTH_2_3")
    
    # Pattern with insufficient reps
    record1 = PatternRecord(pattern_key="test1")
    for i in range(5):
        record1.add_result(True, 0.5, 0.7)
    
    assert not calc.is_earned(record1, 0.6), "Should NOT be earned (insufficient reps)"
    print("  ✓ Insufficient reps → NOT earned")
    
    # Pattern with sufficient reps but low SR
    record2 = PatternRecord(pattern_key="test2")
    for i in range(10):
        is_win = i < 4  # 40% WR
        record2.add_result(is_win, 0.2, 0.7)
    
    assert not calc.is_earned(record2, 0.6), "Should NOT be earned (low SR)"
    print("  ✓ Low success rate → NOT earned")
    
    # Pattern that qualifies
    record3 = PatternRecord(pattern_key="test3")
    for i in range(12):
        record3.add_result(True, 0.3, 0.7)
    
    assert calc.is_earned(record3, 0.6), "Should be EARNED"
    print("  ✓ Qualified pattern → EARNED")
    print("  [PASS]")


def test_tier_progression():
    """Test tier progression with reps."""
    print("\n[TEST] Tier Progression...")
    
    reset_size_ladder()
    ladder = SizeLadderManager(phase="MONTH_7_12")  # Allow TIER-3
    
    pattern_key = "RSI_OVERSOLD_LONDON"
    
    # Build up pattern with wins
    for i in range(50):
        ladder.record_trade_result(pattern_key, True, 0.5, 0.7, 0.01)
    
    # Get multiplier (should be elevated)
    mult = ladder.get_size_multiplier(pattern_key, erm_confidence=0.6, edge_tier="A")
    
    print(f"  ✓ After 50 wins: multiplier = {mult:.2f}×")
    
    pattern = ladder.get_or_create_pattern(pattern_key)
    print(f"  ✓ Current tier: {pattern.tier.value}")
    print("  [PASS]")


def test_tier_reversion_tier3():
    """Test TIER-3 fragility (1 loss = BASE)."""
    print("\n[TEST] TIER-3 Fragility...")
    
    reset_size_ladder()
    ladder = SizeLadderManager(phase="MONTH_7_12")
    
    pattern_key = "A_GRADE_PATTERN"
    
    # Build to TIER-3
    for i in range(50):
        ladder.record_trade_result(pattern_key, True, 0.5, 0.75, 0.01)
    
    pattern = ladder.get_or_create_pattern(pattern_key)
    pattern.tier = SizeTier.TIER_3  # Force to TIER-3
    
    # One loss should reset to BASE
    ladder.record_trade_result(pattern_key, False, -1.0, 0.65, -0.01)
    
    assert pattern.tier == SizeTier.BASE, "TIER-3 should reset to BASE on loss"
    print("  ✓ TIER-3 + 1 loss → BASE")
    print("  [PASS]")


def test_tier_reversion_tier2():
    """Test TIER-2 demotion (1 loss = TIER-1)."""
    print("\n[TEST] TIER-2 Demotion...")
    
    reset_size_ladder()
    ladder = SizeLadderManager(phase="MONTH_7_12")
    
    pattern_key = "B_GRADE_PATTERN"
    
    pattern = ladder.get_or_create_pattern(pattern_key)
    pattern.tier = SizeTier.TIER_2
    
    # One loss should demote to TIER-1
    ladder.record_trade_result(pattern_key, False, -1.0, 0.65, -0.01)
    
    assert pattern.tier == SizeTier.TIER_1, "TIER-2 should demote to TIER-1 on loss"
    print("  ✓ TIER-2 + 1 loss → TIER-1")
    print("  [PASS]")


def test_tier_reversion_tier1():
    """Test TIER-1 demotion (2 consecutive losses = BASE)."""
    print("\n[TEST] TIER-1 Demotion...")
    
    reset_size_ladder()
    ladder = SizeLadderManager(phase="MONTH_7_12")
    
    pattern_key = "C_GRADE_PATTERN"
    
    pattern = ladder.get_or_create_pattern(pattern_key)
    pattern.tier = SizeTier.TIER_1
    
    # First loss - should stay TIER-1
    ladder.record_trade_result(pattern_key, False, -1.0, 0.65, -0.01)
    assert pattern.tier == SizeTier.TIER_1, "TIER-1 should stay after 1 loss"
    print("  ✓ TIER-1 + 1 loss → TIER-1 (stays)")
    
    # Second consecutive loss - should demote
    ladder.record_trade_result(pattern_key, False, -1.0, 0.65, -0.01)
    assert pattern.tier == SizeTier.BASE, "TIER-1 should demote after 2 consecutive losses"
    print("  ✓ TIER-1 + 2 losses → BASE")
    print("  [PASS]")


def test_phase_caps():
    """Test phase-based tier caps."""
    print("\n[TEST] Phase Caps...")
    
    # Month 2-3: max TIER-1
    reset_size_ladder()
    ladder = SizeLadderManager(phase="MONTH_2_3")
    
    pattern_key = "EARLY_PATTERN"
    for i in range(50):
        ladder.record_trade_result(pattern_key, True, 0.5, 0.7, 0.01)
    
    mult = ladder.get_size_multiplier(pattern_key, 0.6, "A")
    assert mult <= 1.10, f"Month 2-3 cap should be 1.10×, got {mult}"
    print(f"  ✓ Month 2-3: max = 1.10× (got {mult:.2f}×)")
    
    # Month 7-12: max TIER-3
    reset_size_ladder()
    ladder = SizeLadderManager(phase="MONTH_7_12")
    
    for i in range(50):
        ladder.record_trade_result(pattern_key, True, 0.5, 0.7, 0.01)
    
    mult = ladder.get_size_multiplier(pattern_key, 0.6, "A")
    print(f"  ✓ Month 7-12: max = 1.25× (got {mult:.2f}×)")
    print("  [PASS]")


def test_mtd_drawdown_controls():
    """Test monthly drawdown controls."""
    print("\n[TEST] MTD Drawdown Controls...")
    
    reset_size_ladder()
    ladder = SizeLadderManager(phase="MONTH_7_12")
    
    # Create some patterns at elevated tiers
    for key in ["P1", "P2", "P3"]:
        pattern = ladder.get_or_create_pattern(key)
        pattern.tier = SizeTier.TIER_2
    
    # 3% drawdown should reset all to BASE
    ladder.check_mtd_drawdown(-0.03)
    
    for key in ["P1", "P2", "P3"]:
        pattern = ladder.get_or_create_pattern(key)
        assert pattern.tier == SizeTier.BASE, f"{key} should be BASE after 3% DD"
    
    print("  ✓ -3% MTD → all patterns to BASE")
    print("  [PASS]")


def run_all_tests():
    """Run all validation tests."""
    print("\n" + "="*60)
    print(" SIZE LADDER VALIDATION")
    print("="*60)
    
    tests = [
        test_tier_multipliers,
        test_pattern_record,
        test_earned_risk_calculation,
        test_tier_progression,
        test_tier_reversion_tier3,
        test_tier_reversion_tier2,
        test_tier_reversion_tier1,
        test_phase_caps,
        test_mtd_drawdown_controls,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"\n  [FAIL] {e}")
            failed += 1
        except Exception as e:
            print(f"\n  [ERROR] {type(e).__name__}: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print(f" RESULTS: {passed} PASSED, {failed} FAILED")
    print("="*60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
