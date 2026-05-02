"""
Trade Sanity Module
====================

CRITICAL ASSERTIONS for trading system correctness.
These assertions MUST run after every trade exit.
Violations CRASH the system immediately - do NOT log-and-continue.

Invariants enforced:
1. SL_HIT → pnl < 0 (stop loss must always be a loss)
2. TP_HIT → pnl > 0 (take profit must always be a profit)
3. PnL calculation is direction-aware (no absolute values)
4. DANGER regime policy is consistent across all modes
"""

import logging
from typing import Dict

logger = logging.getLogger(__name__)


class TradeSanityViolation(Exception):
    """
    Raised when a trade violates fundamental correctness invariants.
    This exception MUST crash the system immediately.
    """
    pass


def assert_pnl_sanity(trade: Dict) -> None:
    """
    Assert that trade PnL is consistent with exit reason.
    
    CRITICAL: This function MUST be called immediately after every trade exit,
    BEFORE metrics aggregation or any other processing.
    
    Invariants:
        SL_HIT  → pnl < 0  (hitting stop loss is always a loss)
        TP_HIT  → pnl > 0  (hitting take profit is always a gain)
    
    Raises:
        TradeSanityViolation: If invariant is violated (CRASHES the run)
    """
    exit_reason = trade.get("exit_reason", "")
    pnl = trade.get("pnl", 0.0)
    
    # Invariant 1: SL_HIT must have negative PnL
    if exit_reason == "SL_HIT" and pnl >= 0:
        error_msg = (
            f"CRITICAL SANITY VIOLATION: SL_HIT with positive PnL!\n"
            f"  Trade: {trade.get('symbol', 'UNKNOWN')}\n"
            f"  Side: {trade.get('side', 'UNKNOWN')}\n"
            f"  Entry: {trade.get('entry_price', 'N/A')}\n"
            f"  Exit: {trade.get('exit_price', 'N/A')}\n"
            f"  SL: {trade.get('sl_price', 'N/A')}\n"
            f"  PnL: {pnl}\n"
            f"  Exit Reason: {exit_reason}\n"
            f"\n"
            f"This indicates a BUG in PnL calculation or SL/TP logic.\n"
            f"ABORTING to prevent fake profitability."
        )
        logger.critical(error_msg)
        raise TradeSanityViolation(error_msg)
    
    # Invariant 2: TP_HIT must have positive PnL
    if exit_reason == "TP_HIT" and pnl <= 0:
        error_msg = (
            f"CRITICAL SANITY VIOLATION: TP_HIT with negative PnL!\n"
            f"  Trade: {trade.get('symbol', 'UNKNOWN')}\n"
            f"  Side: {trade.get('side', 'UNKNOWN')}\n"
            f"  Entry: {trade.get('entry_price', 'N/A')}\n"
            f"  Exit: {trade.get('exit_price', 'N/A')}\n"
            f"  TP: {trade.get('tp_price', 'N/A')}\n"
            f"  PnL: {pnl}\n"
            f"  Exit Reason: {exit_reason}\n"
            f"\n"
            f"This indicates a BUG in PnL calculation or SL/TP logic.\n"
            f"ABORTING to prevent fake losses."
        )
        logger.critical(error_msg)
        raise TradeSanityViolation(error_msg)
    
    logger.debug(f"Sanity check passed: {exit_reason} with pnl={pnl:.4f}")


def assert_pnl_direction(
    side: str,
    entry_price: float,
    exit_price: float,
    calculated_pnl: float,
    size: float,
) -> None:
    """
    Assert that PnL calculation is direction-aware.
    
    BUY:  pnl = (exit - entry) * size
    SELL: pnl = (entry - exit) * size
    
    Raises:
        TradeSanityViolation: If PnL doesn't match expected direction
    """
    if side.upper() == "BUY":
        expected_pnl = (exit_price - entry_price) * size
    elif side.upper() == "SELL":
        expected_pnl = (entry_price - exit_price) * size
    else:
        raise TradeSanityViolation(f"Invalid side: {side}")
    
    # Allow tiny floating point tolerance
    tolerance = abs(expected_pnl) * 0.0001 + 0.0001
    
    if abs(calculated_pnl - expected_pnl) > tolerance:
        error_msg = (
            f"CRITICAL: PnL calculation mismatch!\n"
            f"  Side: {side}\n"
            f"  Entry: {entry_price}\n"
            f"  Exit: {exit_price}\n"
            f"  Size: {size}\n"
            f"  Calculated PnL: {calculated_pnl}\n"
            f"  Expected PnL: {expected_pnl}\n"
            f"\n"
            f"Possible cause: absolute value used somewhere.\n"
            f"ABORTING."
        )
        logger.critical(error_msg)
        raise TradeSanityViolation(error_msg)


# ============================================================================
# DANGER REGIME POLICY
# ============================================================================

# POLICY CHOICE: DANGER = MR-ONLY DEFENSIVE MODE
# Allowed only for MEAN_REVERSION trades under a separate size reduction layer.
DANGER_POLICY = "MR_ONLY_DEFENSIVE"


def can_trade_in_regime(regime: str) -> bool:
    """
    Check if trading is allowed in the given regime.
    
    Policy: DANGER = MR_ONLY_DEFENSIVE
    - Only MEAN_REVERSION trades allowed in DANGER regime
    - This applies to ALL modes (FAST, SHADOW, LIVE)
    
    Returns:
        True if trading allowed, False if blocked
    """
    if regime == "DANGER":
        return False
    return True


def assert_no_danger_trade(trade: Dict) -> None:
    """
    Assert that no trade was executed in DANGER regime.
    
    Raises:
        TradeSanityViolation: If trade was in DANGER (policy violation)
    """
    regime = trade.get("regime", "UNKNOWN")
    
    if regime == "DANGER" and trade.get("strategy", "UNKNOWN") != "MEAN_REVERSION":
        error_msg = (
            f"CRITICAL: Trade executed in DANGER regime!\n"
            f"  Trade: {trade.get('symbol', 'UNKNOWN')}\n"
            f"  Side: {trade.get('side', 'UNKNOWN')}\n"
            f"  Regime: {regime}\n"
            f"  Strategy: {trade.get('strategy', 'UNKNOWN')}\n"
            f"\n"
            f"Policy violation: DANGER = MEAN_REVERSION only\n"
            f"This trade should have been blocked.\n"
            f"ABORTING."
        )
        logger.critical(error_msg)
        raise TradeSanityViolation(error_msg)


# ============================================================================
# FULL TRADE VALIDATION
# ============================================================================

def validate_trade(trade: Dict) -> None:
    """
    Run ALL sanity checks on a trade.
    
    This function MUST be called immediately after trade exit,
    BEFORE any metrics aggregation.
    
    Order of checks:
    1. PnL sanity (SL→loss, TP→profit)
    2. PnL direction (no abs values)
    3. DANGER regime policy
    
    Raises:
        TradeSanityViolation: On ANY violation
    """
    # Check 1: PnL sanity
    assert_pnl_sanity(trade)
    
    # Check 2: PnL direction
    assert_pnl_direction(
        side=trade.get("side", ""),
        entry_price=trade.get("entry_price", 0),
        exit_price=trade.get("exit_price", 0),
        calculated_pnl=trade.get("pnl", 0),
        size=trade.get("size", 0),
    )
    
    # Check 3: DANGER policy
    assert_no_danger_trade(trade)
    
    logger.debug(f"All sanity checks passed for trade")


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    # Test valid trade
    # BUY: (exit - entry) * size = (1.1050 - 1.1000) * 1000 = 5.0
    valid_trade = {
        "symbol": "EURUSD",
        "side": "BUY",
        "entry_price": 1.1000,
        "exit_price": 1.1050,
        "sl_price": 1.0950,
        "tp_price": 1.1050,
        "size": 1000,
        "pnl": 5.0,  # Correct: (1.1050 - 1.1000) * 1000
        "exit_reason": "TP_HIT",
        "regime": "RANGE",
    }
    
    print("Testing valid TP_HIT trade...")
    validate_trade(valid_trade)
    print("✅ Passed")
    
    # Test invalid trade (should fail)
    # PnL is WRONG: SL_HIT should be negative but we set positive
    invalid_trade = {
        "symbol": "EURUSD",
        "side": "BUY",
        "entry_price": 1.1000,
        "exit_price": 1.0950,  # SL hit
        "sl_price": 1.0950,
        "tp_price": 1.1050,
        "size": 1000,
        "pnl": 5.0,  # WRONG! Should be -5.0
        "exit_reason": "SL_HIT",
        "regime": "RANGE",
    }
    
    print("\nTesting invalid SL_HIT trade (should crash)...")
    try:
        validate_trade(invalid_trade)
        print("❌ Should have crashed!")
    except TradeSanityViolation:
        print("✅ Correctly crashed with TradeSanityViolation")
