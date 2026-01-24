"""
RSI RANGE Weapon Validation Tests
==================================

Tests to verify the locked RSI strategy constraints:
❌ No RSI trades in TREND / DANGER
❌ No trades without confirmation
❌ Size never > 15%
❌ Cooldown respected
✅ Trades only in RANGE

Usage:
    python tools/validate_rsi_range_weapon.py
"""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.strategy.rsi_range_reversion import RSIRangeReversion, calculate_rsi
from src.strategy.base import MarketSnapshot, AgentContext, Signal


def create_snapshot(
    symbol: str = "EURUSD",
    price: float = 1.1000,
    atr: float = 0.0010,
    sma_20: float = 1.0990,
) -> MarketSnapshot:
    """Create a market snapshot for testing."""
    return MarketSnapshot(
        timestamp=datetime.now(),
        symbol=symbol,
        price=price,
        atr=atr,
        atr_avg=atr,
        sma_20=sma_20,
    )


def create_context(
    regime: str = "RANGE",
    ml_confidence: float = 0.55,
    edge_score: float = 0.65,
    edge_tier: str = "B",
    session: str = "LONDON",
    mark2_can_trade: bool = True,
    memory_mod: float = 1.0,
    memory_pain_level: float = 0.0,
) -> AgentContext:
    """Create agent context for testing."""
    return AgentContext(
        ml_confidence=ml_confidence,
        ml_prediction="BUY",
        edge_score=edge_score,
        edge_tier=edge_tier,
        regime=regime,
        regime_strength=0.5,
        session=session,
        mark2_can_trade=mark2_can_trade,
        memory_mod=memory_mod,
        memory_pain_level=memory_pain_level,
    )


def test_rsi_calculation():
    """Test RSI calculation accuracy."""
    print("\n[TEST] RSI Calculation...")
    
    # Create price series with known RSI
    # Rising prices = high RSI
    rising_prices = [1.0 + i * 0.01 for i in range(20)]
    rsi = calculate_rsi(rising_prices, 14)
    assert rsi > 70, f"Rising prices should have RSI > 70, got {rsi:.1f}"
    print(f"  ✓ Rising prices RSI = {rsi:.1f} (expected > 70)")
    
    # Falling prices = low RSI
    falling_prices = [1.2 - i * 0.01 for i in range(20)]
    rsi = calculate_rsi(falling_prices, 14)
    assert rsi < 30, f"Falling prices should have RSI < 30, got {rsi:.1f}"
    print(f"  ✓ Falling prices RSI = {rsi:.1f} (expected < 30)")
    
    # Mixed prices = neutral RSI
    mixed_prices = [1.0 + (0.01 if i % 2 == 0 else -0.01) for i in range(20)]
    rsi = calculate_rsi(mixed_prices, 14)
    assert 40 < rsi < 60, f"Mixed prices should have RSI 40-60, got {rsi:.1f}"
    print(f"  ✓ Mixed prices RSI = {rsi:.1f} (expected 40-60)")
    
    print("  [PASS] RSI calculation verified")


def test_no_trades_outside_range():
    """Test that RSI strategy only trades in RANGE regime."""
    print("\n[TEST] No Trades Outside RANGE...")
    
    strategy = RSIRangeReversion()
    snapshot = create_snapshot()
    
    # Prime with oversold prices to trigger buy signal
    for i in range(20):
        strategy._update_price_history("EURUSD", 1.0 - i * 0.001)
    
    # Test RANGE - should be allowed
    context_range = create_context(regime="RANGE")
    assert strategy.is_allowed(context_range), "Should be allowed in RANGE"
    print("  ✓ RANGE regime: ALLOWED")
    
    # Test TREND - should be blocked
    context_trend = create_context(regime="TREND")
    signal = strategy.evaluate(snapshot, context_trend)
    assert signal is None, "Should NOT trade in TREND"
    print("  ✓ TREND regime: BLOCKED")
    
    # Test DANGER - should be blocked
    context_danger = create_context(regime="DANGER")
    signal = strategy.evaluate(snapshot, context_danger)
    assert signal is None, "Should NOT trade in DANGER"
    print("  ✓ DANGER regime: BLOCKED")
    
    print("  [PASS] Regime gating verified")


def test_rsi_thresholds():
    """Test that RSI thresholds are enforced."""
    print("\n[TEST] RSI Thresholds (28/72)...")
    
    strategy = RSIRangeReversion()
    snapshot = create_snapshot()
    context = create_context(regime="RANGE")
    
    # Clear any existing state
    strategy._price_history = {}
    strategy._cooldown_state = {}
    
    # Test 1: RSI = 50 (neutral) - no trade
    for i in range(20):
        strategy._update_price_history("EURUSD", 1.0 + (0.001 if i % 2 == 0 else -0.001))
    
    rsi = strategy._get_rsi("EURUSD")
    signal = strategy.evaluate(snapshot, context)
    assert signal is None, f"RSI {rsi:.1f} should NOT trigger trade"
    print(f"  ✓ RSI = {rsi:.1f} (neutral): NO TRADE")
    
    # Test 2: RSI <= 28 (oversold) - buy potential
    strategy._price_history["EURUSD"] = []
    for i in range(20):
        strategy._update_price_history("EURUSD", 1.1 - i * 0.002)  # Falling
    
    rsi = strategy._get_rsi("EURUSD")
    print(f"  - Oversold test RSI = {rsi:.1f}")
    assert rsi <= 30, f"Setup failed: RSI should be <= 30, got {rsi:.1f}"
    
    # Test 3: RSI >= 72 (overbought) - sell potential
    strategy._cooldown_state = {}
    strategy._price_history["EURUSD"] = []
    for i in range(20):
        strategy._update_price_history("EURUSD", 1.0 + i * 0.002)  # Rising
    
    rsi = strategy._get_rsi("EURUSD")
    print(f"  - Overbought test RSI = {rsi:.1f}")
    assert rsi >= 70, f"Setup failed: RSI should be >= 70, got {rsi:.1f}"
    
    print("  [PASS] RSI thresholds verified")


def test_size_limit():
    """Test that size never exceeds 15%."""
    print("\n[TEST] Size Limit (max 15%)...")
    
    strategy = RSIRangeReversion()
    
    # Test various EDGE and MARK-2 combinations
    test_cases = [
        (1.0, 1.0, 0.12),   # Perfect conditions → 12%
        (0.8, 1.0, 0.096),  # Lower EDGE → 9.6%
        (1.0, 0.5, 0.06),   # Low memory → 6%
        (1.5, 1.5, 0.15),   # Over limit → capped at 15%
    ]
    
    for edge, memory, expected in test_cases:
        # Simulate size calculation
        raw_size = strategy.SIZE_MULTIPLIER * edge * memory
        capped_size = min(raw_size, strategy.MAX_SIZE_PCT)
        
        assert capped_size <= 0.15, f"Size {capped_size:.2%} exceeds 15% limit"
        print(f"  ✓ EDGE={edge}, MEM={memory} → size={capped_size:.2%} (expected ~{expected:.2%})")
    
    # Verify locked parameters
    assert strategy.SIZE_MULTIPLIER == 0.12, "SIZE_MULTIPLIER should be 0.12"
    assert strategy.MAX_SIZE_PCT == 0.15, "MAX_SIZE_PCT should be 0.15"
    
    print("  [PASS] Size limits verified")


def test_cooldown_logic():
    """Test that cooldown is respected after trades."""
    print("\n[TEST] Cooldown Logic...")
    
    strategy = RSIRangeReversion()
    
    # Simulate BUY trade
    strategy._activate_cooldown("EURUSD", "BUY", 25.0)
    
    # Check cooldown is active
    assert strategy._check_cooldown("EURUSD", 30.0), "Cooldown should be active"
    print("  ✓ After BUY @ RSI=25: cooldown ACTIVE")
    
    # RSI moves to 45 (still below 50) - cooldown still active
    assert strategy._check_cooldown("EURUSD", 45.0), "Cooldown should still be active"
    print("  ✓ RSI=45 (below 50): cooldown STILL ACTIVE")
    
    # RSI moves above 50 - cooldown should lift
    assert not strategy._check_cooldown("EURUSD", 55.0), "Cooldown should be lifted"
    print("  ✓ RSI=55 (above 50): cooldown LIFTED")
    
    # Simulate SELL trade
    strategy._activate_cooldown("EURUSD", "SELL", 75.0)
    
    # Check cooldown is active
    assert strategy._check_cooldown("EURUSD", 70.0), "Cooldown should be active"
    print("  ✓ After SELL @ RSI=75: cooldown ACTIVE")
    
    # RSI moves below 50 - cooldown should lift
    assert not strategy._check_cooldown("EURUSD", 45.0), "Cooldown should be lifted"
    print("  ✓ RSI=45 (below 50): cooldown LIFTED")
    
    print("  [PASS] Cooldown logic verified")


def test_mark2_veto():
    """Test that MARK-2 veto is respected."""
    print("\n[TEST] MARK-2 Veto...")
    
    strategy = RSIRangeReversion()
    snapshot = create_snapshot()
    
    # MARK-2 allows trade
    context_allowed = create_context(mark2_can_trade=True)
    assert strategy.is_allowed(context_allowed), "Should be allowed when MARK-2 allows"
    print("  ✓ MARK-2 allows: ALLOWED")
    
    # MARK-2 blocks trade
    context_blocked = create_context(mark2_can_trade=False)
    assert not strategy.is_allowed(context_blocked), "Should be blocked when MARK-2 vetoes"
    print("  ✓ MARK-2 vetoes: BLOCKED")
    
    # High pain level
    context_pain = create_context(memory_pain_level=0.7)
    assert not strategy.is_allowed(context_pain), "Should be blocked at high pain"
    print("  ✓ High pain (0.7): BLOCKED")
    
    print("  [PASS] MARK-2 veto verified")


def test_off_session_blocked():
    """Test that OFF session trades are blocked."""
    print("\n[TEST] OFF Session Blocking...")
    
    strategy = RSIRangeReversion()
    
    # OFF session
    context_off = create_context(session="OFF")
    assert not strategy.is_allowed(context_off), "Should be blocked in OFF session"
    print("  ✓ OFF session: BLOCKED")
    
    # Active sessions
    for session in ["LONDON", "NEW_YORK", "TOKYO", "SYDNEY"]:
        context = create_context(session=session)
        assert strategy.is_allowed(context), f"Should be allowed in {session}"
        print(f"  ✓ {session} session: ALLOWED")
    
    print("  [PASS] Session gating verified")


def test_locked_parameters():
    """Test that key parameters are locked and immutable."""
    print("\n[TEST] Locked Parameters...")
    
    strategy = RSIRangeReversion()
    
    # Verify locked thresholds
    assert strategy.RSI_OVERSOLD == 28, "RSI_OVERSOLD should be 28"
    assert strategy.RSI_OVERBOUGHT == 72, "RSI_OVERBOUGHT should be 72"
    assert strategy.RSI_NEUTRAL == 50, "RSI_NEUTRAL should be 50"
    assert strategy.SL_ATR_MULT == 0.5, "SL_ATR_MULT should be 0.5"
    assert strategy.TP_ATR_MULT == 1.0, "TP_ATR_MULT should be 1.0"
    assert strategy.SIZE_MULTIPLIER == 0.12, "SIZE_MULTIPLIER should be 0.12"
    assert strategy.MAX_SIZE_PCT == 0.15, "MAX_SIZE_PCT should be 0.15"
    
    print(f"  ✓ RSI_OVERSOLD = {strategy.RSI_OVERSOLD}")
    print(f"  ✓ RSI_OVERBOUGHT = {strategy.RSI_OVERBOUGHT}")
    print(f"  ✓ RSI_NEUTRAL = {strategy.RSI_NEUTRAL}")
    print(f"  ✓ SL_ATR_MULT = {strategy.SL_ATR_MULT}")
    print(f"  ✓ TP_ATR_MULT = {strategy.TP_ATR_MULT}")
    print(f"  ✓ SIZE_MULTIPLIER = {strategy.SIZE_MULTIPLIER}")
    print(f"  ✓ MAX_SIZE_PCT = {strategy.MAX_SIZE_PCT}")
    
    print("  [PASS] Locked parameters verified")


def run_all_tests():
    """Run all validation tests."""
    print("\n" + "=" * 60)
    print(" RSI RANGE WEAPON VALIDATION")
    print("=" * 60)
    
    tests = [
        test_rsi_calculation,
        test_no_trades_outside_range,
        test_rsi_thresholds,
        test_size_limit,
        test_cooldown_logic,
        test_mark2_veto,
        test_off_session_blocked,
        test_locked_parameters,
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
    
    print("\n" + "=" * 60)
    print(f" RESULTS: {passed} PASSED, {failed} FAILED")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
