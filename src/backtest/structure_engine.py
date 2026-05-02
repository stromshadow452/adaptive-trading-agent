"""
SCOPUS v2.0 — Market Structure Engine

Pure price structure analysis without indicators.
Based on Smart Money Concepts (SMC).

Components:
    - RangeDetector: Identifies valid trading ranges
    - LiquidityScanner: Finds EQH/EQL stop pools
    - StructureAnalyzer: BOS/CHOCH detection
    - ZoneCalculator: Premium/Discount zones
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from enum import Enum


# ============================================================================
# ENUMS
# ============================================================================

class StructureType(Enum):
    """Market structure classification."""
    RANGE = "RANGE"
    UPTREND = "UPTREND"
    DOWNTREND = "DOWNTREND"
    CHOCH_BULLISH = "CHOCH_BULLISH"
    CHOCH_BEARISH = "CHOCH_BEARISH"
    UNDEFINED = "UNDEFINED"


class Zone(Enum):
    """Price zone classification."""
    PREMIUM = "PREMIUM"       # Top 30% - SELL zone
    EQUILIBRIUM = "EQUILIBRIUM"   # Middle 40% - NO TRADE
    DISCOUNT = "DISCOUNT"     # Bottom 30% - BUY zone


class LiquidityType(Enum):
    """Liquidity pool classification."""
    EQH = "EQUAL_HIGHS"       # Buy stops above
    EQL = "EQUAL_LOWS"        # Sell stops below
    SWEPT = "SWEPT"           # Liquidity taken


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class SwingPoint:
    """Represents a swing high or swing low."""
    index: int
    price: float
    is_high: bool
    timestamp: pd.Timestamp = None


@dataclass
class RangeInfo:
    """Information about a detected range."""
    valid: bool
    high: float = 0.0
    low: float = 0.0
    width_pips: float = 0.0
    bars_in_range: int = 0
    high_touches: int = 0
    low_touches: int = 0
    

@dataclass
class LiquidityPool:
    """Represents a liquidity pool (clustered stops)."""
    level: float
    liquidity_type: LiquidityType
    touches: int
    first_touch_idx: int
    last_touch_idx: int
    swept: bool = False
    sweep_idx: Optional[int] = None


@dataclass
class StructureBreak:
    """Represents a BOS or CHOCH."""
    break_type: str  # "BOS" or "CHOCH"
    direction: str   # "BULLISH" or "BEARISH"
    break_level: float
    break_idx: int
    timestamp: pd.Timestamp = None


# ============================================================================
# RANGE DETECTOR
# ============================================================================

class RangeDetector:
    """
    Detects valid trading ranges from pure price structure.
    
    A range is valid when:
        - 2+ touches of high zone (within tolerance)
        - 2+ touches of low zone (within tolerance)
        - Exists for minimum bars
        - Width between min and max pips
    """
    
    # Configuration - LOOSENED for more setups
    TOUCH_TOLERANCE = 0.003     # 0.3% = ~30 pips (was 0.1%)
    MIN_RANGE_BARS = 30         # ~7.5 hours on M15 (was 50)
    MIN_RANGE_PIPS = 20         # (was 30)
    MAX_RANGE_PIPS = 200        # (was 150)
    SWING_LOOKBACK = 2          # Faster swing detection (was 3)
    
    def __init__(self, pip_value: float = 0.0001):
        """
        Args:
            pip_value: Pip value for the pair (0.0001 for most, 0.01 for JPY)
        """
        self.pip_value = pip_value
    
    def detect_swings(self, df: pd.DataFrame, lookback: int = None) -> Tuple[List[SwingPoint], List[SwingPoint]]:
        """
        Identify swing highs and swing lows using pure price structure.
        
        Swing High: bar.high > neighbors for lookback bars on both sides
        Swing Low: bar.low < neighbors for lookback bars on both sides
        """
        lookback = lookback or self.SWING_LOOKBACK
        highs = []
        lows = []
        
        high_vals = df['high'].values
        low_vals = df['low'].values
        
        for i in range(lookback, len(df) - lookback):
            # Check swing high
            is_swing_high = True
            for j in range(1, lookback + 1):
                if high_vals[i] <= high_vals[i - j] or high_vals[i] <= high_vals[i + j]:
                    is_swing_high = False
                    break
            
            if is_swing_high:
                highs.append(SwingPoint(
                    index=i,
                    price=high_vals[i],
                    is_high=True,
                    timestamp=df.index[i] if hasattr(df.index, '__getitem__') else None
                ))
            
            # Check swing low
            is_swing_low = True
            for j in range(1, lookback + 1):
                if low_vals[i] >= low_vals[i - j] or low_vals[i] >= low_vals[i + j]:
                    is_swing_low = False
                    break
            
            if is_swing_low:
                lows.append(SwingPoint(
                    index=i,
                    price=low_vals[i],
                    is_high=False,
                    timestamp=df.index[i] if hasattr(df.index, '__getitem__') else None
                ))
        
        return highs, lows
    
    def find_clusters(self, points: List[SwingPoint], tolerance: float = None) -> List[Dict]:
        """
        Find clusters of similar price levels (potential EQH/EQL).
        
        Returns list of clusters with level, touches, and indices.
        """
        tolerance = tolerance or self.TOUCH_TOLERANCE
        if not points:
            return []
        
        clusters = []
        used = set()
        
        for i, p1 in enumerate(points):
            if i in used:
                continue
            
            cluster_points = [p1]
            cluster_indices = [p1.index]
            
            for j, p2 in enumerate(points):
                if j <= i or j in used:
                    continue
                
                # Check if within tolerance
                if abs(p2.price - p1.price) / p1.price <= tolerance:
                    cluster_points.append(p2)
                    cluster_indices.append(p2.index)
                    used.add(j)
            
            if len(cluster_points) >= 2:
                avg_price = np.mean([p.price for p in cluster_points])
                clusters.append({
                    'level': avg_price,
                    'touches': len(cluster_points),
                    'indices': cluster_indices,
                    'first_idx': min(cluster_indices),
                    'last_idx': max(cluster_indices),
                })
        
        return clusters
    
    def detect_range(self, df: pd.DataFrame) -> RangeInfo:
        """
        Detect if current price is in a valid trading range.
        
        Returns RangeInfo with range details.
        """
        if len(df) < self.MIN_RANGE_BARS:
            return RangeInfo(valid=False)
        
        # Get recent data for range detection
        recent = df.iloc[-self.MIN_RANGE_BARS * 2:]
        
        # Find swing highs and lows
        swing_highs, swing_lows = self.detect_swings(recent)
        
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return RangeInfo(valid=False)
        
        # Find clusters
        high_clusters = self.find_clusters(swing_highs, self.TOUCH_TOLERANCE)
        low_clusters = self.find_clusters(swing_lows, self.TOUCH_TOLERANCE)
        
        if not high_clusters or not low_clusters:
            return RangeInfo(valid=False)
        
        # Use the most recent valid cluster for each
        best_high = max(high_clusters, key=lambda x: x['touches'])
        best_low = max(low_clusters, key=lambda x: x['touches'])
        
        range_high = best_high['level']
        range_low = best_low['level']
        range_width = range_high - range_low
        range_width_pips = range_width / self.pip_value
        
        # Validate range width
        if range_width_pips < self.MIN_RANGE_PIPS or range_width_pips > self.MAX_RANGE_PIPS:
            return RangeInfo(valid=False)
        
        # Check if price is currently within range
        current_price = df['close'].iloc[-1]
        buffer = range_width * 0.1  # 10% buffer
        
        if current_price < range_low - buffer or current_price > range_high + buffer:
            return RangeInfo(valid=False)
        
        return RangeInfo(
            valid=True,
            high=range_high,
            low=range_low,
            width_pips=range_width_pips,
            bars_in_range=best_high['last_idx'] - best_low['first_idx'],
            high_touches=best_high['touches'],
            low_touches=best_low['touches'],
        )


# ============================================================================
# LIQUIDITY SCANNER
# ============================================================================

class LiquidityScanner:
    """
    Scans for liquidity pools (EQH/EQL) and detects sweeps.
    
    Equal Highs (EQH): Clustered swing highs = buy stops above
    Equal Lows (EQL): Clustered swing lows = sell stops below
    Sweep: Price takes liquidity then closes back inside
    """
    
    CLUSTER_TOLERANCE = 0.002  # ~20 pips for 1.0 price (was 5 pips)
    MIN_TOUCHES = 2
    SWEEP_CONFIRM_BARS = 5      # More bars to confirm (was 3)
    
    def __init__(self, range_detector: RangeDetector):
        self.range_detector = range_detector
    
    def scan_liquidity(self, df: pd.DataFrame, range_info: RangeInfo) -> List[LiquidityPool]:
        """
        Scan for liquidity pools within and around the range.
        """
        if not range_info.valid:
            return []
        
        pools = []
        
        # Get swing points
        swing_highs, swing_lows = self.range_detector.detect_swings(df)
        
        # Find EQH (equal highs - liquidity above)
        high_clusters = self.range_detector.find_clusters(swing_highs, self.CLUSTER_TOLERANCE)
        for cluster in high_clusters:
            if cluster['touches'] >= self.MIN_TOUCHES:
                # Check if near range high
                if abs(cluster['level'] - range_info.high) / range_info.high < 0.005:
                    pools.append(LiquidityPool(
                        level=cluster['level'],
                        liquidity_type=LiquidityType.EQH,
                        touches=cluster['touches'],
                        first_touch_idx=cluster['first_idx'],
                        last_touch_idx=cluster['last_idx'],
                    ))
        
        # Find EQL (equal lows - liquidity below)
        low_clusters = self.range_detector.find_clusters(swing_lows, self.CLUSTER_TOLERANCE)
        for cluster in low_clusters:
            if cluster['touches'] >= self.MIN_TOUCHES:
                # Check if near range low
                if abs(cluster['level'] - range_info.low) / range_info.low < 0.005:
                    pools.append(LiquidityPool(
                        level=cluster['level'],
                        liquidity_type=LiquidityType.EQL,
                        touches=cluster['touches'],
                        first_touch_idx=cluster['first_idx'],
                        last_touch_idx=cluster['last_idx'],
                    ))
        
        return pools
    
    def detect_sweep(self, df: pd.DataFrame, pool: LiquidityPool) -> bool:
        """
        Detect if a liquidity pool has been swept.
        
        Sweep = price trades through level, then closes back inside.
        """
        if len(df) < self.SWEEP_CONFIRM_BARS:
            return False
        
        recent = df.iloc[-self.SWEEP_CONFIRM_BARS:]
        
        if pool.liquidity_type == LiquidityType.EQH:
            # Check if any bar's high exceeded the level
            swept = any(recent['high'] > pool.level)
            # Check if current close is back below
            rejected = recent['close'].iloc[-1] < pool.level
            return swept and rejected
        
        elif pool.liquidity_type == LiquidityType.EQL:
            # Check if any bar's low went below the level
            swept = any(recent['low'] < pool.level)
            # Check if current close is back above
            rejected = recent['close'].iloc[-1] > pool.level
            return swept and rejected
        
        return False


# ============================================================================
# STRUCTURE ANALYZER
# ============================================================================

class StructureAnalyzer:
    """
    Analyzes market structure for BOS (Break of Structure) and CHOCH (Change of Character).
    
    BOS: Continuation - HH after HL (bullish), LL after LH (bearish)
    CHOCH: Reversal - First break of trend structure
    """
    
    def __init__(self, range_detector: RangeDetector):
        self.range_detector = range_detector
    
    def identify_structure(self, df: pd.DataFrame) -> Tuple[StructureType, List[StructureBreak]]:
        """
        Identify current market structure and any recent breaks.
        """
        swing_highs, swing_lows = self.range_detector.detect_swings(df)
        
        if len(swing_highs) < 3 or len(swing_lows) < 3:
            return StructureType.UNDEFINED, []
        
        # Sort by index
        all_swings = sorted(swing_highs + swing_lows, key=lambda x: x.index)
        
        # Analyze structure
        breaks = []
        current_structure = StructureType.RANGE
        
        # Get last 4-6 swing points
        recent_swings = all_swings[-6:]
        
        # Separate into highs and lows
        recent_highs = [s for s in recent_swings if s.is_high]
        recent_lows = [s for s in recent_swings if not s.is_high]
        
        if len(recent_highs) >= 2 and len(recent_lows) >= 2:
            # Check for uptrend (HH-HL)
            if recent_highs[-1].price > recent_highs[-2].price and \
               recent_lows[-1].price > recent_lows[-2].price:
                current_structure = StructureType.UPTREND
            
            # Check for downtrend (LH-LL)
            elif recent_highs[-1].price < recent_highs[-2].price and \
                 recent_lows[-1].price < recent_lows[-2].price:
                current_structure = StructureType.DOWNTREND
            
            # Check for CHOCH
            # Bullish CHOCH: was making LH-LL, now broke above LH
            # Bearish CHOCH: was making HH-HL, now broke below HL
        
        return current_structure, breaks
    
    def check_bos(self, df: pd.DataFrame, direction: str) -> Optional[StructureBreak]:
        """
        Check for Break of Structure in given direction.
        
        Args:
            direction: "BULLISH" or "BEARISH"
        """
        swing_highs, swing_lows = self.range_detector.detect_swings(df)
        
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return None
        
        current_close = df['close'].iloc[-1]
        
        if direction == "BULLISH":
            # BOS bullish = current price above recent swing high
            recent_high = swing_highs[-1].price
            if current_close > recent_high:
                return StructureBreak(
                    break_type="BOS",
                    direction="BULLISH",
                    break_level=recent_high,
                    break_idx=len(df) - 1,
                )
        
        elif direction == "BEARISH":
            # BOS bearish = current price below recent swing low
            recent_low = swing_lows[-1].price
            if current_close < recent_low:
                return StructureBreak(
                    break_type="BOS",
                    direction="BEARISH",
                    break_level=recent_low,
                    break_idx=len(df) - 1,
                )
        
        return None


# ============================================================================
# ZONE CALCULATOR
# ============================================================================

class ZoneCalculator:
    """
    Calculates Premium/Discount zones from range.
    
    Premium: Top 30% - SELL zone only
    Equilibrium: Middle 40% - NO TRADE zone
    Discount: Bottom 30% - BUY zone only
    """
    
    PREMIUM_THRESHOLD = 0.70   # Above 70% = premium
    DISCOUNT_THRESHOLD = 0.30  # Below 30% = discount
    
    def get_zone(self, current_price: float, range_info: RangeInfo) -> Zone:
        """
        Determine which zone the current price is in.
        """
        if not range_info.valid:
            return Zone.EQUILIBRIUM  # Default to no-trade
        
        range_height = range_info.high - range_info.low
        if range_height <= 0:
            return Zone.EQUILIBRIUM
        
        # Position within range (0 = low, 1 = high)
        position = (current_price - range_info.low) / range_height
        
        if position >= self.PREMIUM_THRESHOLD:
            return Zone.PREMIUM
        elif position <= self.DISCOUNT_THRESHOLD:
            return Zone.DISCOUNT
        else:
            return Zone.EQUILIBRIUM
    
    def get_zone_levels(self, range_info: RangeInfo) -> Dict[str, Tuple[float, float]]:
        """
        Get exact price levels for each zone.
        
        Returns dict with zone name -> (low, high) tuple
        """
        if not range_info.valid:
            return {}
        
        range_height = range_info.high - range_info.low
        
        return {
            'premium': (
                range_info.low + range_height * self.PREMIUM_THRESHOLD,
                range_info.high
            ),
            'equilibrium': (
                range_info.low + range_height * self.DISCOUNT_THRESHOLD,
                range_info.low + range_height * self.PREMIUM_THRESHOLD
            ),
            'discount': (
                range_info.low,
                range_info.low + range_height * self.DISCOUNT_THRESHOLD
            ),
        }


# ============================================================================
# MARKET STRUCTURE ENGINE (FACADE)
# ============================================================================

class MarketStructureEngine:
    """
    Main facade for all market structure analysis.
    
    Combines:
        - Range detection
        - Liquidity scanning
        - Structure analysis
        - Zone calculation
    """
    
    def __init__(self, pip_value: float = 0.0001):
        self.range_detector = RangeDetector(pip_value)
        self.liquidity_scanner = LiquidityScanner(self.range_detector)
        self.structure_analyzer = StructureAnalyzer(self.range_detector)
        self.zone_calculator = ZoneCalculator()
        self.pip_value = pip_value
    
    def analyze(self, df: pd.DataFrame) -> Dict:
        """
        Complete market structure analysis.
        
        Returns dict with:
            - range: RangeInfo
            - liquidity: List[LiquidityPool]
            - structure: StructureType
            - zone: Zone
            - tradeable: bool
        """
        if len(df) < 100:
            return {
                'range': RangeInfo(valid=False),
                'liquidity': [],
                'structure': StructureType.UNDEFINED,
                'zone': Zone.EQUILIBRIUM,
                'tradeable': False,
                'reason': 'Insufficient data',
            }
        
        # 1. Detect range
        range_info = self.range_detector.detect_range(df)
        
        # 2. Scan liquidity pools
        liquidity_pools = []
        if range_info.valid:
            liquidity_pools = self.liquidity_scanner.scan_liquidity(df, range_info)
            # Check for sweeps
            for pool in liquidity_pools:
                if self.liquidity_scanner.detect_sweep(df, pool):
                    pool.swept = True
                    pool.sweep_idx = len(df) - 1
        
        # 3. Analyze structure
        structure_type, breaks = self.structure_analyzer.identify_structure(df)
        
        # 4. Calculate zone
        current_price = df['close'].iloc[-1]
        current_zone = self.zone_calculator.get_zone(current_price, range_info)
        
        # 5. Determine if tradeable
        tradeable = False
        reason = ""
        
        if not range_info.valid:
            reason = "No valid range"
        elif current_zone == Zone.EQUILIBRIUM:
            reason = "Price in equilibrium zone"
        elif structure_type in [StructureType.CHOCH_BULLISH, StructureType.CHOCH_BEARISH]:
            reason = "CHOCH detected - wait for retest"
        else:
            # Check for swept liquidity in the right zone
            swept_pools = [p for p in liquidity_pools if p.swept]
            if current_zone == Zone.DISCOUNT:
                if any(p.liquidity_type == LiquidityType.EQL for p in swept_pools):
                    tradeable = True
                    reason = "Discount zone + EQL swept"
                else:
                    reason = "Discount zone but no EQL sweep"
            elif current_zone == Zone.PREMIUM:
                if any(p.liquidity_type == LiquidityType.EQH for p in swept_pools):
                    tradeable = True
                    reason = "Premium zone + EQH swept"
                else:
                    reason = "Premium zone but no EQH sweep"
        
        return {
            'range': range_info,
            'liquidity': liquidity_pools,
            'structure': structure_type,
            'zone': current_zone,
            'tradeable': tradeable,
            'reason': reason,
            'current_price': current_price,
        }


# ============================================================================
# TEST
# ============================================================================

if __name__ == '__main__':
    print("Market Structure Engine v2.0")
    print("=" * 50)
    
    # Create sample data for testing
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=200, freq='15min')
    
    # Create range-bound price action
    base = 1.1000
    noise = np.random.randn(200) * 0.0010
    trend = np.sin(np.linspace(0, 4 * np.pi, 200)) * 0.0050
    prices = base + noise + trend
    
    df = pd.DataFrame({
        'open': prices,
        'high': prices + abs(np.random.randn(200)) * 0.0005,
        'low': prices - abs(np.random.randn(200)) * 0.0005,
        'close': prices + np.random.randn(200) * 0.0002,
    }, index=dates)
    
    # Analyze
    engine = MarketStructureEngine(pip_value=0.0001)
    result = engine.analyze(df)
    
    print(f"\nRange Valid: {result['range'].valid}")
    if result['range'].valid:
        print(f"  High: {result['range'].high:.5f}")
        print(f"  Low: {result['range'].low:.5f}")
        print(f"  Width: {result['range'].width_pips:.1f} pips")
    
    print(f"\nLiquidity Pools: {len(result['liquidity'])}")
    for pool in result['liquidity']:
        print(f"  {pool.liquidity_type.value} at {pool.level:.5f} ({pool.touches} touches)")
    
    print(f"\nStructure: {result['structure'].value}")
    print(f"Zone: {result['zone'].value}")
    print(f"Tradeable: {result['tradeable']}")
    print(f"Reason: {result['reason']}")
