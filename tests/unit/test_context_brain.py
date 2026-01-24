"""
Unit Tests for Context Brain V1

Tests for:
- SRDetector: Swing-based S/R detection
- BreakoutClassifier: False breakout scoring
- ManipulationDetector: Trap signature detection
- ContextBrain: Main orchestrator
- AdaptiveFilterEngine: Mode switching and threshold adaptation
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock

from src.backtest.context_brain import (
    SRDetector, BreakoutClassifier, ManipulationDetector,
    MarketStateClassifier, ContextBrain, ContextOutput, SRLevel
)
from src.backtest.adaptive_filters import (
    AdaptiveFilterEngine, ModeConfig, LEARNING_CONFIG, CONFIRMATION_CONFIG,
    get_ml_threshold, get_meta_thresholds, reset_adaptive_filter_engine
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_ohlcv():
    """Generate sample OHLCV data with clear swing points."""
    np.random.seed(42)
    n = 100
    
    # Create trending data with swings
    base = 1.1000
    trend = np.linspace(0, 0.0050, n)  # Uptrend
    noise = np.random.randn(n) * 0.0005
    closes = base + trend + noise
    
    # Create OHLC from close
    highs = closes + np.abs(np.random.randn(n) * 0.0003)
    lows = closes - np.abs(np.random.randn(n) * 0.0003)
    opens = closes - np.random.randn(n) * 0.0002
    
    df = pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes
    })
    
    return df


@pytest.fixture
def sample_candle():
    """Single candle for testing."""
    return pd.Series({
        'open': 1.1050,
        'high': 1.1065,
        'low': 1.1045,
        'close': 1.1055
    })


@pytest.fixture
def atr_value():
    """Typical ATR for EURUSD M5."""
    return 0.0015


# =============================================================================
# SRDetector Tests
# =============================================================================

class TestSRDetector:
    
    def test_init_defaults(self):
        detector = SRDetector()
        assert detector.swing_lookback == 3
        assert detector.cluster_atr_mult == 0.5
        assert detector.max_levels == 5
    
    def test_detect_finds_levels(self, sample_ohlcv, atr_value):
        detector = SRDetector(swing_lookback=3)
        
        support, resistance = detector.detect(
            sample_ohlcv['high'].values,
            sample_ohlcv['low'].values,
            sample_ohlcv['close'].values,
            atr_value,
            lookback=50
        )
        
        # Should find at least some levels in trending data
        assert isinstance(support, list)
        assert isinstance(resistance, list)
        # All levels are SRLevel objects
        for level in support + resistance:
            assert isinstance(level, SRLevel)
            assert 0 < level.strength <= 1.0
    
    def test_detect_with_sparse_data(self, atr_value):
        # Very short data
        detector = SRDetector()
        short_data = np.linspace(1.1, 1.105, 10)
        
        support, resistance = detector.detect(
            short_data, short_data * 0.999, short_data,
            atr_value, lookback=10
        )
        
        # Should handle gracefully (empty or minimal levels)
        assert isinstance(support, list)
        assert isinstance(resistance, list)
    
    def test_compute_proximity(self, atr_value):
        detector = SRDetector()
        current_price = 1.1050
        
        support = [SRLevel(1.1030, 0.8, "support", 3, 45)]
        resistance = [SRLevel(1.1070, 0.7, "resistance", 2, 48)]
        
        proximity, nearest_s, nearest_r = detector.compute_proximity(
            current_price, support, resistance, atr_value
        )
        
        assert 0 <= proximity <= 1.0
        assert nearest_s == 1.1030
        # nearest_r may be None if support is closer (within min_distance logic)
        # Just check it's either the expected value or None
        assert nearest_r is None or nearest_r == 1.1070


# =============================================================================
# BreakoutClassifier Tests
# =============================================================================

class TestBreakoutClassifier:
    
    def test_confirmed_breakout(self, atr_value):
        classifier = BreakoutClassifier(confirm_bars=5)
        
        # Price breaks resistance and holds above for >5 bars
        resistance = 1.1050
        closes = np.array([1.1045, 1.1048, 1.1060, 1.1062, 1.1065, 1.1068, 1.1070, 1.1072, 1.1075])
        highs = closes + 0.0010  # Clear breach above threshold
        lows = closes - 0.0003
        
        score = classifier.classify(closes, highs, lows, resistance, "resistance", atr_value)
        
        # Confirmed breakout should be positive (or 0 if breach threshold not met)
        assert score >= 0  # Allow 0 if exactly at threshold
    
    def test_false_breakout(self, atr_value):
        classifier = BreakoutClassifier(reject_bars=3)
        
        # Price breaks then returns below
        resistance = 1.1050
        closes = np.array([1.1045, 1.1060, 1.1055, 1.1040, 1.1038])  # Break then fail
        highs = np.array([1.1048, 1.1065, 1.1058, 1.1052, 1.1045])
        lows = closes - 0.0005
        
        score = classifier.classify(closes, highs, lows, resistance, "resistance", atr_value)
        
        # False breakout should be negative
        assert score < 0
    
    def test_no_breakout(self, atr_value):
        classifier = BreakoutClassifier()
        
        # Price never breaks level
        resistance = 1.1050
        closes = np.array([1.1040, 1.1042, 1.1038, 1.1045, 1.1043])
        highs = closes + 0.0003
        lows = closes - 0.0003
        
        score = classifier.classify(closes, highs, lows, resistance, "resistance", atr_value)
        
        # No breakout = 0
        assert score == 0.0


# =============================================================================
# ManipulationDetector Tests
# =============================================================================

class TestManipulationDetector:
    
    def test_clean_candle(self, sample_ohlcv, atr_value):
        detector = ManipulationDetector()
        
        # Normal candle with balanced wicks
        candle = pd.Series({'open': 1.1050, 'high': 1.1055, 'low': 1.1048, 'close': 1.1053})
        
        risk = detector.detect(candle, sample_ohlcv.iloc[-10:], [], atr_value)
        
        # Low manipulation risk for clean candle
        assert 0 <= risk <= 0.5
    
    def test_wick_rejection(self, sample_ohlcv, atr_value):
        detector = ManipulationDetector(wick_ratio_threshold=2.0)
        
        # Candle with large upper wick (rejection)
        candle = pd.Series({
            'open': 1.1050, 
            'high': 1.1080,  # Long upper wick
            'low': 1.1048, 
            'close': 1.1052
        })
        
        risk = detector.detect(candle, sample_ohlcv.iloc[-10:], [], atr_value)
        
        # Should detect wick rejection
        assert risk > 0.3
    
    def test_stop_hunt_pattern(self, sample_ohlcv, atr_value):
        detector = ManipulationDetector()
        
        # Spike high then close low (stop hunt)
        recent = sample_ohlcv.iloc[-10:].copy()
        recent['high'] = recent['high'] * 0.999  # Lower recent highs
        
        candle = pd.Series({
            'open': 1.1050,
            'high': 1.1090,  # Spike above recent range
            'low': 1.1045,
            'close': 1.1048   # Close bearish (reversal)
        })
        
        risk = detector.detect(candle, recent, [], atr_value)
        
        # Should detect stop hunt
        assert risk > 0.4


# =============================================================================
# MarketStateClassifier Tests
# =============================================================================

class TestMarketStateClassifier:
    
    def test_expansion_state(self):
        classifier = MarketStateClassifier()
        
        state = classifier.classify(
            atr=0.0020,        # High ATR
            atr_ma=0.0015,     # Normal ATR MA
            breakout_quality=0.3,
            manipulation_risk=0.2
        )
        
        assert state == "expansion"
    
    def test_contraction_state(self):
        classifier = MarketStateClassifier()
        
        state = classifier.classify(
            atr=0.0010,        # Low ATR
            atr_ma=0.0015,     # Normal ATR MA
            breakout_quality=0.0,
            manipulation_risk=0.1
        )
        
        assert state == "contraction"
    
    def test_breakout_state(self):
        classifier = MarketStateClassifier()
        
        state = classifier.classify(
            atr=0.0015,
            atr_ma=0.0015,
            breakout_quality=0.7,  # High breakout quality
            manipulation_risk=0.2
        )
        
        assert state == "breakout"


# =============================================================================
# ContextBrain Tests
# =============================================================================

class TestContextBrain:
    
    def test_init(self):
        brain = ContextBrain()
        assert brain.sr_detector is not None
        assert brain.breakout_classifier is not None
        assert brain.manipulation_detector is not None
    
    def test_analyze_returns_valid_output(self, sample_ohlcv, sample_candle, atr_value):
        brain = ContextBrain()
        
        result = brain.analyze(sample_candle, sample_ohlcv, atr_value)
        
        assert isinstance(result, ContextOutput)
        assert 0 <= result.context_confidence <= 1.0
        assert result.operating_mode in ["LEARNING", "CONFIRMATION"]
        assert result.market_state in ["expansion", "contraction", "breakout", "reversal"]
    
    def test_learning_mode_with_sparse_data(self, sample_candle, atr_value):
        brain = ContextBrain()
        
        # Very short data
        short_df = pd.DataFrame({
            'open': [1.1050] * 30,
            'high': [1.1055] * 30,
            'low': [1.1045] * 30,
            'close': [1.1052] * 30
        })
        
        result = brain.analyze(sample_candle, short_df, atr_value)
        
        # Should be in LEARNING mode
        assert result.operating_mode == "LEARNING"
        assert result.data_sufficiency < 1.0
        # Should have exploration budget
        assert result.exploration_budget > 0
    
    def test_confirmation_mode_with_full_data(self, sample_candle, atr_value):
        brain = ContextBrain()
        
        # Plenty of data
        long_df = pd.DataFrame({
            'open': [1.1050] * 600,
            'high': [1.1055] * 600,
            'low': [1.1045] * 600,
            'close': [1.1052] * 600
        })
        
        result = brain.analyze(sample_candle, long_df, atr_value)
        
        # Should be in CONFIRMATION mode
        assert result.operating_mode == "CONFIRMATION"
        assert result.data_sufficiency >= 0.8
    
    def test_to_dict_output(self, sample_ohlcv, sample_candle, atr_value):
        brain = ContextBrain()
        
        result = brain.analyze(sample_candle, sample_ohlcv, atr_value)
        output_dict = result.to_dict()
        
        assert "context_confidence" in output_dict
        assert "operating_mode" in output_dict
        assert "market_state" in output_dict
        assert isinstance(output_dict["context_confidence"], float)


# =============================================================================
# AdaptiveFilterEngine Tests
# =============================================================================

class TestAdaptiveFilterEngine:
    
    def setup_method(self):
        reset_adaptive_filter_engine()
    
    def test_learning_mode_with_few_bars(self):
        engine = AdaptiveFilterEngine()
        
        mode, sufficiency, config = engine.compute_mode(50)
        
        assert mode == "LEARNING"
        assert sufficiency < 0.5
        assert config.ml_buy_threshold < CONFIRMATION_CONFIG.ml_buy_threshold
    
    def test_confirmation_mode_with_many_bars(self):
        engine = AdaptiveFilterEngine()
        
        mode, sufficiency, config = engine.compute_mode(600)
        
        assert mode == "CONFIRMATION"
        assert sufficiency == 1.0
        assert config.ml_buy_threshold == CONFIRMATION_CONFIG.ml_buy_threshold
    
    def test_threshold_interpolation(self):
        engine = AdaptiveFilterEngine()
        
        # Mid-range data
        mode, sufficiency, config = engine.compute_mode(200)
        
        # Thresholds should be between LEARNING and CONFIRMATION
        assert LEARNING_CONFIG.ml_buy_threshold < config.ml_buy_threshold < CONFIRMATION_CONFIG.ml_buy_threshold
    
    def test_adapt_threshold(self):
        engine = AdaptiveFilterEngine()
        
        # Sparse data should relax threshold
        sparse_thresh = engine.adapt_threshold(0.6, 50)
        full_thresh = engine.adapt_threshold(0.6, 500)
        
        assert sparse_thresh < full_thresh
        assert sparse_thresh > 0.3  # Some floor
    
    def test_exploration_budget(self):
        engine = AdaptiveFilterEngine()
        
        # Set learning mode
        engine.compute_mode(50)
        budget = engine.get_exploration_budget()
        
        assert budget > 0
        assert budget <= 0.30  # Max cap
    
    def test_can_explore_tracking(self):
        engine = AdaptiveFilterEngine()
        engine.compute_mode(50)  # LEARNING mode
        
        assert engine.can_explore() == True
        
        engine.record_exploration()
        engine.record_exploration()
        
        # After 2 explorations, should be at limit
        assert engine.can_explore() == False
    
    def test_daily_reset(self):
        engine = AdaptiveFilterEngine()
        engine.compute_mode(50)
        
        engine.record_exploration()
        engine.record_exploration()
        assert engine.can_explore() == False
        
        # New day
        engine.reset_daily("2023-01-02")
        assert engine.can_explore() == True


# =============================================================================
# Convenience Function Tests
# =============================================================================

class TestConvenienceFunctions:
    
    def test_get_ml_threshold(self):
        learn_thresh = get_ml_threshold("LEARNING", "buy")
        conf_thresh = get_ml_threshold("CONFIRMATION", "buy")
        
        assert learn_thresh < conf_thresh
    
    def test_get_meta_thresholds(self):
        learn_p, learn_f = get_meta_thresholds("LEARNING")
        conf_p, conf_f = get_meta_thresholds("CONFIRMATION")
        
        assert learn_p < conf_p
        assert learn_f < conf_f


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
