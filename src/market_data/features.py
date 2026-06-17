"""
SCOPUS Data Layer - Canonical Feature Definitions

Defines the standard feature set for each timeframe.
This ensures consistency across backtesting, live trading, and model training.
"""

from typing import Dict, List
from .types import Timeframe


# Canonical feature list per timeframe
CANONICAL_FEATURES: Dict[Timeframe, List[str]] = {
    # M5: High-frequency features for entry timing
    Timeframe.M5: [
        # Raw OHLCV
        "open", "high", "low", "close", "volume",
        
        # Momentum
        "rsi_14",           # RSI(14) - overbought/oversold
        "rsi_7",            # Fast RSI for quick reversals
        
        # Volatility
        "atr_14",           # ATR(14) - position sizing
        "bb_upper",         # Bollinger upper band
        "bb_middle",        # Bollinger middle (SMA 20)
        "bb_lower",         # Bollinger lower band
        "bb_width",         # Band width (volatility proxy)
        
        # Trend
        "trend_flag",       # 1=up, -1=down, 0=neutral (SMA 20/50 cross)
        "ema_9",            # Fast EMA
        "ema_21",           # Slow EMA
        
        # MACD
        "macd",             # MACD line
        "macd_signal",      # Signal line
        "macd_hist",        # Histogram
        
        # Price action
        "candle_body_pct",  # Body size as % of range
        "upper_wick_pct",   # Upper wick as % of range
        "lower_wick_pct",   # Lower wick as % of range
    ],
    
    # M15: Entry confirmation + micro-structure
    Timeframe.M15: [
        # Raw OHLCV
        "open", "high", "low", "close", "volume",
        
        # Momentum
        "rsi_14",
        "rsi_21",
        
        # Volatility
        "atr_14",
        "bb_upper",
        "bb_middle",
        "bb_lower",
        "bb_width",
        
        # Trend
        "trend_flag",
        "sma_20",
        "sma_50",
        "ema_12",
        "ema_26",
        
        # MACD
        "macd",
        "macd_signal",
        "macd_hist",
        
        # Volume
        "volume_sma_20",    # Average volume
        "volume_ratio",     # Current / average
    ],
    
    # H1: Intraday context + regime
    Timeframe.H1: [
        # Raw OHLCV
        "open", "high", "low", "close", "volume",
        
        # Momentum
        "rsi_14",
        
        # Volatility
        "atr_14",
        "atr_ratio",        # Current ATR / SMA(ATR, 20)
        
        # Trend
        "trend_flag",
        "sma_50",
        "sma_100",
        "sma_200",
        
        # MACD
        "macd",
        "macd_signal",
        "macd_hist",
        
        # Support/Resistance
        "sr_zone_id",       # Nearest S/R zone ID
        "sr_distance",      # Distance to nearest S/R (pips)
        "sr_strength",      # Zone strength (touches)
        
        # Regime
        "volatility_regime", # 0=low, 1=normal, 2=high
        "trend_strength",    # ADX or similar
    ],
    
    # H4: Swing context + major levels
    Timeframe.H4: [
        # Raw OHLCV
        "open", "high", "low", "close", "volume",
        
        # Momentum
        "rsi_14",
        
        # Volatility
        "atr_14",
        "atr_percentile",   # ATR percentile (0-100)
        
        # Trend
        "trend_flag",
        "sma_50",
        "sma_100",
        "sma_200",
        
        # Support/Resistance
        "sr_zone_id",
        "sr_distance",
        "sr_strength",
        "pivot_point",      # Daily pivot
        "pivot_r1",         # Resistance 1
        "pivot_s1",         # Support 1
        
        # Regime
        "volatility_regime",
        "trend_strength",
        "market_phase",     # 0=accumulation, 1=markup, 2=distribution, 3=markdown
    ],
    
    # D1: Position context + macro structure
    Timeframe.D1: [
        # Raw OHLCV
        "open", "high", "low", "close", "volume",
        
        # Momentum
        "rsi_14",
        
        # Volatility
        "atr_14",
        "atr_percentile",
        
        # Trend
        "trend_flag",
        "sma_50",
        "sma_100",
        "sma_200",
        "price_vs_sma200",  # % above/below SMA(200)
        
        # Support/Resistance
        "sr_zone_id",
        "sr_distance",
        "sr_strength",
        "weekly_high",      # Distance to weekly high
        "weekly_low",       # Distance to weekly low
        "monthly_high",     # Distance to monthly high
        "monthly_low",      # Distance to monthly low
        
        # Regime
        "volatility_regime",
        "trend_strength",
        "market_phase",
        
        # Seasonality
        "day_of_week",      # 0-6 (Mon-Sun)
        "week_of_month",    # 1-5
    ],
}


# Feature groups for easy filtering
FEATURE_GROUPS = {
    "ohlcv": ["open", "high", "low", "close", "volume"],
    "momentum": ["rsi_14", "rsi_21", "rsi_7"],
    "volatility": ["atr_14", "bb_upper", "bb_middle", "bb_lower", "bb_width", "atr_ratio", "atr_percentile"],
    "trend": ["trend_flag", "sma_20", "sma_50", "sma_100", "sma_200", "ema_9", "ema_12", "ema_21", "ema_26"],
    "macd": ["macd", "macd_signal", "macd_hist"],
    "sr": ["sr_zone_id", "sr_distance", "sr_strength"],
    "regime": ["volatility_regime", "trend_strength", "market_phase"],
    "price_action": ["candle_body_pct", "upper_wick_pct", "lower_wick_pct"],
}


def get_features_for_timeframe(timeframe: Timeframe) -> List[str]:
    """
    Get canonical feature list for a timeframe.
    
    Args:
        timeframe: Timeframe enum
    
    Returns:
        List of feature names
    """
    return CANONICAL_FEATURES.get(timeframe, [])


def get_feature_columns_for_mtf(
    base_timeframe: Timeframe,
    higher_timeframes: List[Timeframe]
) -> List[str]:
    """
    Get all feature column names for a multi-timeframe context.
    
    Returns columns in format: {TF}_{feature}
    
    Args:
        base_timeframe: Base timeframe
        higher_timeframes: List of higher timeframes
    
    Returns:
        List of column names (e.g., ["M5_open", "M5_rsi_14", "H1_trend_flag", ...])
    """
    all_timeframes = [base_timeframe] + higher_timeframes
    columns = []
    
    for tf in all_timeframes:
        features = get_features_for_timeframe(tf)
        for feature in features:
            columns.append(f"{tf.value}_{feature}")
    
    return columns


def validate_features(df_columns: List[str], required_timeframes: List[Timeframe]) -> Dict[str, List[str]]:
    """
    Validate that a DataFrame has all required features.
    
    Args:
        df_columns: List of column names in DataFrame
        required_timeframes: List of timeframes that should be present
    
    Returns:
        Dict with 'missing' and 'extra' column lists
    """
    expected = []
    for tf in required_timeframes:
        features = get_features_for_timeframe(tf)
        expected.extend([f"{tf.value}_{feat}" for feat in features])
    
    expected_set = set(expected)
    actual_set = set(df_columns)
    
    return {
        "missing": sorted(list(expected_set - actual_set)),
        "extra": sorted(list(actual_set - expected_set))
    }


# Feature computation metadata
FEATURE_METADATA = {
    "rsi_14": {
        "description": "Relative Strength Index (14 periods)",
        "range": (0, 100),
        "overbought": 70,
        "oversold": 30,
    },
    "atr_14": {
        "description": "Average True Range (14 periods)",
        "range": (0, float('inf')),
        "unit": "price_units",
    },
    "trend_flag": {
        "description": "Trend direction based on SMA crossover",
        "range": (-1, 1),
        "values": {-1: "downtrend", 0: "neutral", 1: "uptrend"},
    },
    "volatility_regime": {
        "description": "Volatility regime classification",
        "range": (0, 2),
        "values": {0: "low", 1: "normal", 2: "high"},
    },
    "market_phase": {
        "description": "Market phase (Wyckoff-inspired)",
        "range": (0, 3),
        "values": {0: "accumulation", 1: "markup", 2: "distribution", 3: "markdown"},
    },
}
