"""
ERM Validation Tests
====================

Tests the Experience Reasoning Module functionality.
"""

import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)

# Test imports
try:
    from src.reasoning import (
        ExperienceReasoningModule,
        encode_context,
        ERMDecision
    )
    print("✅ ERM imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    exit(1)


def test_pattern_encoding():
    """Test context to pattern encoding."""
    print("\n" + "=" * 50)
    print("TEST 1: PATTERN ENCODING")
    print("=" * 50)
    
    ctx = {
        'session': 'LONDON',
        'regime': 'TREND',
        'regime_strength': 0.72,
        'edge_tier': 'B',
        'ml_confidence': 0.68,
        'volatility_bucket': 'MID'
    }
    
    pattern = encode_context(ctx)
    print(f"Context: session=LONDON, regime=TREND, strength=0.72, tier=B, conf=0.68")
    print(f"Pattern: {pattern.to_string()}")
    
    # Verify buckets
    assert pattern.session == 3, "LONDON should be 3"
    assert pattern.regime == 1, "TREND should be 1"
    assert pattern.regime_bucket == 2, "0.72 should be bucket 2"
    assert pattern.edge_tier == 2, "B should be 2"
    assert pattern.conf_bucket == 2, "0.68 should be bucket 2"
    assert pattern.vol_bucket == 1, "MID should be 1"
    
    print("✅ Pattern encoding PASSED")


def test_reasoning_engine():
    """Test reasoning engine decisions."""
    print("\n" + "=" * 50)
    print("TEST 2: REASONING ENGINE DECISIONS")
    print("=" * 50)
    
    erm = ExperienceReasoningModule()
    
    # Test 1: Unknown pattern → REDUCE (cautious)
    ctx1 = {
        'session': 'TOKYO',
        'regime': 'RANGE',
        'regime_strength': 0.45,
        'edge_tier': 'C',
        'ml_confidence': 0.55,
        'volatility_bucket': 'LOW'
    }
    
    decision1 = erm.evaluate(ctx1)
    print(f"\n1. Unknown pattern:")
    print(f"   Action: {decision1.action}")
    print(f"   Size mult: {decision1.size_multiplier}")
    print(f"   Reason: {decision1.reason}")
    assert decision1.action == "REDUCE", "Unknown pattern should be REDUCE"
    print("   ✅ PASSED")
    
    # Add some experience
    pattern = encode_context(ctx1)
    
    # Simulate 5 trades: 3 wins, 2 losses
    for i, r in enumerate([1.2, -0.8, 1.5, -1.0, 0.8]):
        erm.update(ctx1, "EXECUTE", 1.0, r)
    
    # Test 2: Pattern with 60% win rate
    decision2 = erm.evaluate(ctx1)
    print(f"\n2. After 5 trades (3W/2L = 60%):")
    print(f"   Action: {decision2.action}")
    print(f"   Size mult: {decision2.size_multiplier}")
    print(f"   Confidence: {decision2.confidence:.1%}")
    print(f"   Reason: {decision2.reason}")
    assert decision2.action == "EXECUTE", "60% SR should be EXECUTE"
    print("   ✅ PASSED")
    
    # Test 3: Pattern with many failures
    ctx3 = {
        'session': 'SYDNEY',
        'regime': 'DANGER',
        'regime_strength': 0.85,
        'edge_tier': 'C',
        'ml_confidence': 0.45,
        'volatility_bucket': 'HIGH'
    }
    
    # Simulate 7 losses
    for i in range(7):
        erm.update(ctx3, "EXECUTE", 1.0, -1.0)
    
    decision3 = erm.evaluate(ctx3)
    print(f"\n3. After 7 losses (0%):")
    print(f"   Action: {decision3.action}")
    print(f"   Reason: {decision3.reason}")
    assert decision3.action == "IGNORE", "7 failures should be IGNORE"
    print("   ✅ PASSED")
    
    print("\n" + "=" * 50)
    print("ALL REASONING TESTS PASSED ✅")
    print("=" * 50)


def test_memory_performance():
    """Test O(1) lookup performance."""
    print("\n" + "=" * 50)
    print("TEST 3: MEMORY PERFORMANCE")
    print("=" * 50)
    
    import time
    
    erm = ExperienceReasoningModule()
    
    # Pre-populate with many patterns
    sessions = ['LONDON', 'TOKYO', 'NEW_YORK', 'SYDNEY']
    regimes = ['RANGE', 'TREND']
    
    count = 0
    for session in sessions:
        for regime in regimes:
            for strength in [0.3, 0.5, 0.7]:
                for tier in ['B', 'C']:
                    for conf in [0.5, 0.6, 0.7]:
                        ctx = {
                            'session': session,
                            'regime': regime,
                            'regime_strength': strength,
                            'edge_tier': tier,
                            'ml_confidence': conf,
                            'volatility_bucket': 'MID'
                        }
                        erm.update(ctx, "EXECUTE", 1.0, 0.5)
                        count += 1
    
    print(f"Populated {count} patterns")
    
    # Benchmark lookup
    ctx = {
        'session': 'LONDON',
        'regime': 'TREND',
        'regime_strength': 0.65,
        'edge_tier': 'B',
        'ml_confidence': 0.68,
        'volatility_bucket': 'MID'
    }
    
    iterations = 10000
    start = time.perf_counter()
    for _ in range(iterations):
        _ = erm.evaluate(ctx)
    elapsed = time.perf_counter() - start
    
    avg_time_us = (elapsed / iterations) * 1_000_000
    print(f"Lookup time: {avg_time_us:.2f} µs/eval ({iterations} iterations)")
    assert avg_time_us < 100, "Lookup should be < 100µs"
    print("✅ Performance PASSED (<100µs per eval)")
    
    # Print stats
    erm.print_stats()


def test_learning_example():
    """Walk through the learning example from the plan."""
    print("\n" + "=" * 50)
    print("TEST 4: LEARNING WALKTHROUGH")
    print("=" * 50)
    
    erm = ExperienceReasoningModule()
    
    # The context from the plan
    ctx = {
        'session': 'LONDON',
        'regime': 'TREND',
        'regime_strength': 0.72,
        'edge_tier': 'B',
        'ml_confidence': 0.68,
        'volatility_bucket': 'MID'
    }
    
    print("\n--- Trade 1: Initial encounter ---")
    decision1 = erm.evaluate(ctx)
    print(f"ERM Decision: {decision1.action}, reason: {decision1.reason}")
    print("Trade executed, result: -1.0R (loss)")
    erm.update(ctx, "EXECUTE", 1.0, -1.0)
    
    print("\n--- Trade 2: Same pattern returns ---")
    decision2 = erm.evaluate(ctx)
    print(f"ERM Decision: {decision2.action}, size×{decision2.size_multiplier:.2f}")
    print(f"Reason: {decision2.reason}")
    print("Trade executed with reduced size, result: +1.2R")
    erm.update(ctx, "REDUCE", 0.5, 1.2)
    
    print("\n--- Trade 3-6: More data ---")
    outcomes = [0.8, -0.5, 1.1, -0.3]
    for i, r in enumerate(outcomes, start=3):
        erm.update(ctx, "EXECUTE", 1.0, r)
        print(f"  Trade {i}: R={r:+.1f}")
    
    print("\n--- Trade 7: After learning ---")
    decision7 = erm.evaluate(ctx)
    print(f"ERM Decision: {decision7.action}")
    print(f"Size multiplier: {decision7.size_multiplier:.2f}")
    print(f"Confidence: {decision7.confidence:.1%}")
    print(f"Reason: {decision7.reason}")
    
    print("\n" + "=" * 50)
    print("LEARNING WALKTHROUGH COMPLETE ✅")
    print("=" * 50)


if __name__ == "__main__":
    test_pattern_encoding()
    test_reasoning_engine()
    test_memory_performance()
    test_learning_example()
    
    print("\n🎯 ALL ERM VALIDATION TESTS PASSED 🎯")
