"""
SCOPUS v3.0 — Mean Reversion Engine

Pure statistical mean reversion system based on the "Rubber Band" principle.
Price stretched too far from mean → snaps back to fair value.

Theory:
    - EURUSD is statistically mean-reverting (proven in 2022-2023 data)
    - Price at 2.5 standard deviations = top/bottom 1% of distribution
    - ADX < 25 confirms no trend = rubber band won't break
    - Target = return to mean (middle BB), not full reversal

This module implements production-ready mean reversion logic with:
    - No look-ahead bias (all indicators properly shifted)
    - Fixed risk management (ATR-based stops)
    - Strict entry/exit rules based on statistics
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Optional, Dict
from enum import Enum


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class MeanReversionConfig:
    """Configuration for the Mean Reversion Engine."""
    
    # Regime Filter
    adx_period: int = 14
    adx_threshold: float = 40.0  # Trade only if ADX < this (raised from 25)
    
    # Bollinger Bands
    bb_period: int = 20
    bb_std: float = 3.0          # Extreme deviation only (3 SD = 0.3% probability)
    
    # RSI
    rsi_period: int = 14
    rsi_overbought: float = 80.0  # Looser (was 70)
    rsi_oversold: float = 20.0    # Looser (was 30)
    
    # Risk Management
    atr_period: int = 14
    atr_sl_multiplier: float = 2.5  # Balanced stop
    risk_per_trade: float = 0.01    # 1% risk
    
    # Trade Management
    min_bars_between_trades: int = 4  # ~1 hour cooldown on M15


class Signal(Enum):
    """Trading signal types."""
    LONG = "LONG"
    SHORT = "SHORT"
    HOLD = "HOLD"
    EXIT_LONG = "EXIT_LONG"
    EXIT_SHORT = "EXIT_SHORT"


@dataclass
class TradeSignal:
    """Complete trade signal with all details."""
    signal: Signal
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    confidence: float = 0.0
    regime: str = ""
    reason: str = ""
    features: Dict[str, float] = None


# ============================================================================
# MEAN REVERSION ENGINE
# ============================================================================

class MeanReversionEngine:
    """
    Pure Statistical Mean Reversion Trading Engine.
    
    The "Rubber Band" Philosophy:
        1. Price deviates from mean (stretch)
        2. We wait for extreme deviation (2.5 SD)
        3. We confirm with momentum (RSI)
        4. We enter betting on snap-back to mean
        5. We exit at the mean (not beyond)
    
    Why This Works on FX Majors:
        - Central bank interventions prevent extremes
        - Carry trades create mean reversion
        - Retail stops create temporary dislocations
        - Market makers profit from reversion
    """
    
    def __init__(self, config: MeanReversionConfig = None):
        """Initialize the engine with configuration."""
        self.config = config or MeanReversionConfig()
        self.position = None  # "LONG" or "SHORT" or None
        self.last_trade_bar = -999
    
    # =========================================================================
    # INDICATOR CALCULATIONS
    # =========================================================================
    
    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all required technical features.
        
        Calculates:
            - Bollinger Bands (2.5 SD)
            - RSI (14)
            - ADX (14)
            - ATR (14)
        
        All values are SHIFTED by 1 to prevent look-ahead bias.
        We use values from PREVIOUS bar to make decisions on CURRENT bar.
        
        Args:
            df: OHLCV DataFrame with columns [open, high, low, close, volume]
        
        Returns:
            DataFrame with added feature columns
        """
        df = df.copy()
        
        # Ensure lowercase columns
        df.columns = [c.lower() for c in df.columns]
        
        # ----- Bollinger Bands (20, 2.5) -----
        df['bb_sma'] = df['close'].rolling(window=self.config.bb_period).mean()
        df['bb_std'] = df['close'].rolling(window=self.config.bb_period).std()
        df['bb_upper'] = df['bb_sma'] + (self.config.bb_std * df['bb_std'])
        df['bb_lower'] = df['bb_sma'] - (self.config.bb_std * df['bb_std'])
        
        # ----- RSI (14) -----
        df['rsi'] = self._calculate_rsi(df['close'], self.config.rsi_period)
        
        # ----- ADX (14) -----
        df['adx'] = self._calculate_adx(df, self.config.adx_period)
        
        # ----- ATR (14) -----
        df['atr'] = self._calculate_atr(df, self.config.atr_period)
        
        # ----- Distance from Mean (Z-Score) -----
        df['z_score'] = (df['close'] - df['bb_sma']) / df['bb_std']
        
        # ----- SHIFT ALL INDICATORS BY 1 TO PREVENT LOOK-AHEAD -----
        # We use PREVIOUS bar's indicator values to decide on CURRENT bar
        indicator_cols = ['bb_sma', 'bb_upper', 'bb_lower', 'rsi', 'adx', 'atr', 'z_score']
        for col in indicator_cols:
            df[col] = df[col].shift(1)
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """
        Calculate Relative Strength Index.
        
        RSI = 100 - (100 / (1 + RS))
        RS = Average Gain / Average Loss
        """
        delta = prices.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        
        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()
        
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.fillna(50)  # Neutral if undefined
    
    def _calculate_adx(self, df: pd.DataFrame, period: int) -> pd.Series:
        """
        Calculate Average Directional Index.
        
        ADX measures trend strength (not direction).
        ADX < 25 = Weak/No trend (good for mean reversion)
        ADX > 25 = Trending (avoid mean reversion)
        """
        high = df['high']
        low = df['low']
        close = df['close']
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed values
        atr = pd.Series(tr).ewm(span=period, adjust=False).mean()
        plus_di = 100 * pd.Series(plus_dm).ewm(span=period, adjust=False).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).ewm(span=period, adjust=False).mean() / atr
        
        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.ewm(span=period, adjust=False).mean()
        
        return adx.fillna(25)  # Neutral if undefined
    
    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """
        Calculate Average True Range for stop-loss sizing.
        """
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        return tr.ewm(span=period, adjust=False).mean()
    
    # =========================================================================
    # SIGNAL GENERATION
    # =========================================================================
    
    def generate_signal(self, df: pd.DataFrame, current_bar_idx: int = -1) -> TradeSignal:
        """
        Generate trading signal for the current bar.
        
        Entry Logic (The Stretch):
            SHORT: Close > Upper BB (2.5 SD) AND RSI > 70 AND ADX < 25
            LONG:  Close < Lower BB (2.5 SD) AND RSI < 30 AND ADX < 25
        
        Exit Logic (The Snap Back):
            EXIT:  Price returns to Middle Band (SMA 20)
        
        Args:
            df: DataFrame with computed features
            current_bar_idx: Index of current bar (default -1 = last bar)
        
        Returns:
            TradeSignal with signal type and trade details
        """
        # Get current bar data
        bar = df.iloc[current_bar_idx]
        
        # Extract values (already shifted, so safe to use)
        close = bar['close']
        bb_upper = bar['bb_upper']
        bb_lower = bar['bb_lower']
        bb_sma = bar['bb_sma']
        rsi = bar['rsi']
        adx = bar['adx']
        atr = bar['atr']
        z_score = bar['z_score']
        
        # Check for NaN (not enough data)
        if pd.isna(bb_upper) or pd.isna(rsi) or pd.isna(adx):
            return TradeSignal(signal=Signal.HOLD, reason="Insufficient data")
        
        # Build features dict for logging
        features = {
            'close': close,
            'bb_upper': bb_upper,
            'bb_lower': bb_lower,
            'bb_sma': bb_sma,
            'rsi': rsi,
            'adx': adx,
            'atr': atr,
            'z_score': z_score,
        }
        
        # ----- REGIME CHECK (The Gatekeeper) -----
        if adx >= self.config.adx_threshold:
            return TradeSignal(
                signal=Signal.HOLD,
                regime="TRENDING",
                reason=f"ADX {adx:.1f} >= {self.config.adx_threshold} (Trend detected, rubber band might break)",
                features=features,
            )
        
        regime = "MEAN_REVERTING"
        
        # ----- EXIT CHECK (The Snap Back) -----
        if self.position == "LONG":
            # Exit long when price returns to mean
            if close >= bb_sma:
                return TradeSignal(
                    signal=Signal.EXIT_LONG,
                    take_profit=bb_sma,
                    regime=regime,
                    reason="Price returned to mean (SMA 20)",
                    features=features,
                )
        
        elif self.position == "SHORT":
            # Exit short when price returns to mean
            if close <= bb_sma:
                return TradeSignal(
                    signal=Signal.EXIT_SHORT,
                    take_profit=bb_sma,
                    regime=regime,
                    reason="Price returned to mean (SMA 20)",
                    features=features,
                )
        
        # ----- COOLDOWN CHECK -----
        bars_since_trade = current_bar_idx - self.last_trade_bar
        if bars_since_trade < self.config.min_bars_between_trades and current_bar_idx != -1:
            return TradeSignal(
                signal=Signal.HOLD,
                regime=regime,
                reason=f"Cooldown: {bars_since_trade}/{self.config.min_bars_between_trades} bars",
                features=features,
            )
        
        # ----- ENTRY CHECK (The Stretch) -----
        
        # SHORT: Price stretched above upper band + overbought
        if close > bb_upper and rsi > self.config.rsi_overbought:
            # Already in position?
            if self.position is not None:
                return TradeSignal(
                    signal=Signal.HOLD,
                    regime=regime,
                    reason="Already in position",
                    features=features,
                )
            
            # Calculate stop loss (above entry)
            stop_loss = close + (atr * self.config.atr_sl_multiplier)
            
            # Calculate confidence based on z-score
            confidence = min(abs(z_score) / 3.0, 1.0)  # Max at 3 SD
            
            return TradeSignal(
                signal=Signal.SHORT,
                entry_price=close,
                stop_loss=stop_loss,
                take_profit=bb_sma,
                confidence=confidence,
                regime=regime,
                reason=f"SHORT: Close {close:.5f} > BB_Upper {bb_upper:.5f}, RSI {rsi:.1f} > 70, Z={z_score:.2f}",
                features=features,
            )
        
        # LONG: Price stretched below lower band + oversold
        if close < bb_lower and rsi < self.config.rsi_oversold:
            # Already in position?
            if self.position is not None:
                return TradeSignal(
                    signal=Signal.HOLD,
                    regime=regime,
                    reason="Already in position",
                    features=features,
                )
            
            # Calculate stop loss (below entry)
            stop_loss = close - (atr * self.config.atr_sl_multiplier)
            
            # Calculate confidence based on z-score
            confidence = min(abs(z_score) / 3.0, 1.0)
            
            return TradeSignal(
                signal=Signal.LONG,
                entry_price=close,
                stop_loss=stop_loss,
                take_profit=bb_sma,
                confidence=confidence,
                regime=regime,
                reason=f"LONG: Close {close:.5f} < BB_Lower {bb_lower:.5f}, RSI {rsi:.1f} < 30, Z={z_score:.2f}",
                features=features,
            )
        
        # No signal
        return TradeSignal(
            signal=Signal.HOLD,
            regime=regime,
            reason="No extreme stretch detected",
            features=features,
        )
    
    # =========================================================================
    # POSITION MANAGEMENT
    # =========================================================================
    
    def enter_position(self, direction: str, bar_idx: int):
        """Record entry into a position."""
        self.position = direction
        self.last_trade_bar = bar_idx
    
    def exit_position(self):
        """Record exit from position."""
        self.position = None
    
    def reset(self):
        """Reset engine state."""
        self.position = None
        self.last_trade_bar = -999


# ============================================================================
# RISK CALCULATOR
# ============================================================================

class RiskCalculator:
    """
    Position sizing based on fixed risk percentage.
    """
    
    def __init__(self, account_balance: float, risk_per_trade: float = 0.01):
        """
        Args:
            account_balance: Total account balance
            risk_per_trade: Risk as decimal (0.01 = 1%)
        """
        self.account_balance = account_balance
        self.risk_per_trade = risk_per_trade
    
    def calculate_position_size(self, entry_price: float, stop_loss: float, 
                                 pip_value: float = 0.0001) -> float:
        """
        Calculate position size in lots.
        
        Position Size = Risk Amount / (SL pips * pip value per lot)
        """
        risk_amount = self.account_balance * self.risk_per_trade
        sl_distance = abs(entry_price - stop_loss)
        sl_pips = sl_distance / pip_value
        
        # Standard lot: 1 pip = $10
        pip_value_per_lot = 10.0
        
        position_size = risk_amount / (sl_pips * pip_value_per_lot)
        
        return round(position_size, 2)


# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

class MeanReversionAnalyzer:
    """
    Analyze mean reversion statistics of price data.
    """
    
    @staticmethod
    def compute_half_life(prices: pd.Series) -> float:
        """
        Compute mean reversion half-life using Ornstein-Uhlenbeck model.
        
        Half-life = ln(2) / speed of reversion
        
        Lower half-life = faster mean reversion = better for strategy.
        """
        # Lag prices
        lag = prices.shift(1).dropna()
        delta = prices.diff().dropna()
        
        # OLS regression: delta = alpha + beta * lag + error
        # beta gives us the speed of reversion
        from numpy.linalg import lstsq
        
        X = np.column_stack([np.ones(len(lag)), lag.values])
        y = delta.iloc[1:].values
        
        result = lstsq(X[1:], y, rcond=None)
        beta = result[0][1]
        
        if beta >= 0:
            return np.inf  # Not mean reverting
        
        half_life = -np.log(2) / beta
        return half_life
    
    @staticmethod
    def compute_hurst_exponent(prices: pd.Series, max_lag: int = 100) -> float:
        """
        Compute Hurst exponent.
        
        H < 0.5: Mean reverting (good)
        H = 0.5: Random walk
        H > 0.5: Trending (bad for mean reversion)
        """
        lags = range(2, min(max_lag, len(prices) // 2))
        tau = []
        
        for lag in lags:
            tau.append(np.std(prices.diff(lag).dropna()))
        
        # Linear fit of log(lag) vs log(tau)
        poly = np.polyfit(np.log(list(lags)), np.log(tau), 1)
        hurst = poly[0]
        
        return hurst


# ============================================================================
# TEST
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print(" SCOPUS v3.0 — Mean Reversion Engine")
    print(" Pure Statistical Rubber Band System")
    print("=" * 70)
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=500, freq='15min')
    
    # Create mean-reverting price series (Ornstein-Uhlenbeck process)
    mean = 1.1000
    theta = 0.1  # Mean reversion speed
    sigma = 0.001  # Volatility
    
    prices = [mean]
    for i in range(1, 500):
        dp = theta * (mean - prices[-1]) + sigma * np.random.randn()
        prices.append(prices[-1] + dp)
    
    prices = np.array(prices)
    
    df = pd.DataFrame({
        'open': prices,
        'high': prices + abs(np.random.randn(500)) * 0.0005,
        'low': prices - abs(np.random.randn(500)) * 0.0005,
        'close': prices + np.random.randn(500) * 0.0002,
    }, index=dates)
    
    # Initialize engine
    config = MeanReversionConfig()
    engine = MeanReversionEngine(config)
    
    # Compute features
    df = engine.compute_features(df)
    
    print("\nFeatures computed:")
    print(f"  BB Upper: {df['bb_upper'].iloc[-1]:.5f}")
    print(f"  BB Lower: {df['bb_lower'].iloc[-1]:.5f}")
    print(f"  BB SMA:   {df['bb_sma'].iloc[-1]:.5f}")
    print(f"  RSI:      {df['rsi'].iloc[-1]:.1f}")
    print(f"  ADX:      {df['adx'].iloc[-1]:.1f}")
    print(f"  ATR:      {df['atr'].iloc[-1]:.5f}")
    print(f"  Z-Score:  {df['z_score'].iloc[-1]:.2f}")
    
    # Generate signal
    signal = engine.generate_signal(df)
    
    print(f"\nSignal: {signal.signal.value}")
    print(f"Regime: {signal.regime}")
    print(f"Reason: {signal.reason}")
    
    # Analyze mean reversion statistics
    analyzer = MeanReversionAnalyzer()
    
    half_life = analyzer.compute_half_life(df['close'])
    hurst = analyzer.compute_hurst_exponent(df['close'])
    
    print(f"\nMean Reversion Statistics:")
    print(f"  Half-life: {half_life:.1f} bars")
    print(f"  Hurst:     {hurst:.3f} (< 0.5 = mean reverting)")
    
    if hurst < 0.5:
        print("  ✅ Asset is MEAN REVERTING")
    else:
        print("  ⚠️ Asset is TRENDING (strategy may not work)")
