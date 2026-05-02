"""
SCOPUS MTF System - H4 Brain

Strategic decision-making on H4 timeframe.
Outputs: Direction (BUY/SELL/NO-TRADE) + Permission (ALLOW/BLOCK)

Target: 6 decisions per day, ~80% NO-TRADE.
"""

import pandas as pd
import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Tuple


# =============================================================================
# ENUMS
# =============================================================================

class Direction(Enum):
    BUY = "BUY"
    SELL = "SELL"
    NO_TRADE = "NO_TRADE"


class Permission(Enum):
    ALLOW = "ALLOW"
    BLOCK = "BLOCK"


class Regime(Enum):
    TREND = "TREND"
    CHOPPY = "CHOPPY"
    DEAD = "DEAD"
    TRANSITION = "TRANSITION"


class Confidence(Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


# =============================================================================
# DECISION OUTPUT
# =============================================================================

@dataclass
class H4Decision:
    """Output from H4 Brain."""
    direction: Direction
    permission: Permission
    regime: Regime
    confidence: Confidence
    trend_strength: float
    timestamp: pd.Timestamp


# =============================================================================
# H4 BRAIN
# =============================================================================

class H4Brain:
    """
    Strategic brain operating on H4 timeframe.
    
    Uses simple, robust rules to determine:
    1. Market regime (trend vs choppy vs dead)
    2. Trade direction (if any)
    3. Permission for M15 to trade
    
    Philosophy: Most of the time, do NOTHING.
    """
    
    # Thresholds - STRENGTHENED for quality
    TREND_WEAK = 1.2       # Below this = no trend (was 0.8)
    TREND_MODERATE = 1.8   # Above this = tradable trend (was 1.2)
    TREND_STRONG = 2.5     # Above this = strong trend (was 2.0)
    
    VOLATILITY_DEAD = 0.7  # Below this = dead market (was 0.6)
    VOLATILITY_HIGH = 1.5  # Above this = volatile
    
    RSI_OVERBOUGHT = 70
    RSI_OVERSOLD = 30
    
    def __init__(self):
        self.last_decision: Optional[H4Decision] = None
    
    def decide(self, h4_features: Dict[str, float], timestamp: pd.Timestamp) -> H4Decision:
        """
        Make strategic decision based on H4 features.
        
        Features should be computed from H4 bars (aggregated from M15).
        All features must be LAGGED (computed from previous bar).
        """
        # Extract features
        trend_50 = h4_features.get('trend_50_zscore', 0)
        trend_200 = h4_features.get('trend_200_zscore', 0)
        atr_ratio = h4_features.get('atr_ratio_14_50', 1.0)
        rsi = h4_features.get('rsi_14', 50)
        bb_width = h4_features.get('bb_width_zscore', 0)
        
        # Step 1: Detect regime
        regime = self._detect_regime(trend_50, trend_200, atr_ratio, bb_width)
        
        # Step 2: If not trending, block trading
        if regime in [Regime.CHOPPY, Regime.DEAD]:
            decision = H4Decision(
                direction=Direction.NO_TRADE,
                permission=Permission.BLOCK,
                regime=regime,
                confidence=Confidence.LOW,
                trend_strength=abs(trend_50),
                timestamp=timestamp,
            )
            self.last_decision = decision
            return decision
        
        # Step 3: Determine direction based on trend strength
        direction, confidence = self._determine_direction(trend_50, trend_200, rsi)
        
        # Step 4: Set permission
        if direction == Direction.NO_TRADE:
            permission = Permission.BLOCK
        else:
            permission = Permission.ALLOW
        
        decision = H4Decision(
            direction=direction,
            permission=permission,
            regime=regime,
            confidence=confidence,
            trend_strength=abs(trend_50),
            timestamp=timestamp,
        )
        
        self.last_decision = decision
        return decision
    
    def _detect_regime(
        self,
        trend_50: float,
        trend_200: float,
        atr_ratio: float,
        bb_width: float,
    ) -> Regime:
        """Detect market regime."""
        # Dead market: Very low volatility
        if atr_ratio < self.VOLATILITY_DEAD:
            return Regime.DEAD
        
        # Choppy: No clear trend
        if abs(trend_50) < self.TREND_WEAK and abs(trend_200) < self.TREND_WEAK:
            return Regime.CHOPPY
        
        # Transition: Trend forming but not confirmed
        if abs(trend_50) < self.TREND_MODERATE:
            return Regime.TRANSITION
        
        # Trend: Clear directional move
        return Regime.TREND
    
    def _determine_direction(
        self,
        trend_50: float,
        trend_200: float,
        rsi: float,
    ) -> Tuple[Direction, Confidence]:
        """Determine trade direction."""
        # Need moderate trend to trade
        if abs(trend_50) < self.TREND_MODERATE:
            return Direction.NO_TRADE, Confidence.LOW
        
        # Check trend alignment
        trends_aligned = (trend_50 > 0 and trend_200 > 0) or (trend_50 < 0 and trend_200 < 0)
        
        # BUY conditions
        if trend_50 > self.TREND_MODERATE and trend_200 > 0:
            # Don't buy if overbought
            if rsi > self.RSI_OVERBOUGHT:
                return Direction.NO_TRADE, Confidence.LOW
            
            if trend_50 > self.TREND_STRONG and trends_aligned:
                return Direction.BUY, Confidence.HIGH
            else:
                return Direction.BUY, Confidence.MEDIUM
        
        # SELL conditions
        if trend_50 < -self.TREND_MODERATE and trend_200 < 0:
            # Don't sell if oversold
            if rsi < self.RSI_OVERSOLD:
                return Direction.NO_TRADE, Confidence.LOW
            
            if trend_50 < -self.TREND_STRONG and trends_aligned:
                return Direction.SELL, Confidence.HIGH
            else:
                return Direction.SELL, Confidence.MEDIUM
        
        return Direction.NO_TRADE, Confidence.LOW


# =============================================================================
# H4 FEATURE AGGREGATOR
# =============================================================================

class H4FeatureAggregator:
    """
    Aggregates M15 data into H4 bars and computes features.
    
    H4 = 4-hour bars = 16 M15 bars per H4 bar.
    """
    
    def aggregate_to_h4(self, m15_df: pd.DataFrame) -> pd.DataFrame:
        """Convert M15 bars to H4 bars."""
        # Resample to 4H
        h4_df = m15_df.resample('4h').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum' if 'volume' in m15_df.columns else 'first',
        }).dropna()
        
        return h4_df
    
    def compute_h4_features(self, h4_df: pd.DataFrame) -> pd.DataFrame:
        """Compute features on H4 bars."""
        df = h4_df.copy()
        
        # EMAs
        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
        
        # Trend z-scores
        def zscore(x, window=50):
            return (x - x.rolling(window).mean()) / x.rolling(window).std()
        
        df['trend_50_zscore'] = zscore(df['close'] - df['ema_50'])
        df['trend_200_zscore'] = zscore(df['close'] - df['ema_200'])
        
        # ATR
        tr = pd.concat([
            df['high'] - df['low'],
            (df['high'] - df['close'].shift(1)).abs(),
            (df['low'] - df['close'].shift(1)).abs(),
        ], axis=1).max(axis=1)
        
        df['atr_14'] = tr.rolling(14).mean()
        df['atr_50'] = tr.rolling(50).mean()
        df['atr_ratio_14_50'] = df['atr_14'] / df['atr_50']
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Bollinger Width
        std_20 = df['close'].rolling(20).std()
        df['bb_width'] = std_20 / df['ema_20']
        df['bb_width_zscore'] = zscore(df['bb_width'])
        
        # Apply 1-bar lag (CRITICAL)
        feature_cols = [
            'trend_50_zscore', 'trend_200_zscore',
            'atr_ratio_14_50', 'rsi_14', 'bb_width_zscore',
        ]
        
        for col in feature_cols:
            df[col] = df[col].shift(1)
        
        return df
    
    def get_h4_bar_at_time(
        self,
        h4_features: pd.DataFrame,
        timestamp: pd.Timestamp,
    ) -> Optional[Dict[str, float]]:
        """Get H4 features valid at a given timestamp."""
        # Find the most recent H4 bar before this timestamp
        valid_bars = h4_features[h4_features.index <= timestamp]
        
        if len(valid_bars) == 0:
            return None
        
        latest = valid_bars.iloc[-1]
        
        return {
            'trend_50_zscore': latest.get('trend_50_zscore', 0),
            'trend_200_zscore': latest.get('trend_200_zscore', 0),
            'atr_ratio_14_50': latest.get('atr_ratio_14_50', 1),
            'rsi_14': latest.get('rsi_14', 50),
            'bb_width_zscore': latest.get('bb_width_zscore', 0),
        }


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    # Quick test
    brain = H4Brain()
    
    # Simulated features
    test_features = {
        'trend_50_zscore': 1.5,
        'trend_200_zscore': 0.8,
        'atr_ratio_14_50': 1.2,
        'rsi_14': 55,
        'bb_width_zscore': 0.5,
    }
    
    decision = brain.decide(test_features, pd.Timestamp.now())
    
    print(f"Direction: {decision.direction.value}")
    print(f"Permission: {decision.permission.value}")
    print(f"Regime: {decision.regime.value}")
    print(f"Confidence: {decision.confidence.value}")
    print(f"Trend Strength: {decision.trend_strength:.2f}")
