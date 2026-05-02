"""
SCOPUS Phase-1: Asset Class Framework

Asset-class tagging without per-asset models.
Classes define:
- Default parameters (SL/TP ATR multiples)
- Size limits
- Behavioral characteristics

The agent uses the SAME logic for all assets, with only
parameter scaling by asset class.
"""

from dataclasses import dataclass
from typing import Dict, Optional, List
import re


# ============================================================================
# ASSET CLASS DEFINITIONS
# ============================================================================

@dataclass
class AssetClassConfig:
    """Configuration for an asset class."""
    name: str
    display_name: str
    assets: List[str]
    
    # Behavioral characteristics
    typical_atr_pct: float  # Typical daily ATR as % of price
    session_driven: bool     # Does it follow session patterns?
    mean_reverting: bool     # Tendency to mean revert
    liquidity: str           # HIGH, MEDIUM, LOW, VARIABLE
    
    # Trading parameters
    sl_atr_mult: float       # Default SL in ATR multiples
    tp_atr_mult: float       # Default TP in ATR multiples
    max_size_pct: float      # Max position size as % of equity
    min_rr_ratio: float      # Minimum R:R to accept trade
    
    # Risk adjustments
    weekend_risk: bool = False  # Can hold over weekend?
    news_sensitive: bool = True  # Reduce size around news?


# Define asset classes
ASSET_CLASSES: Dict[str, AssetClassConfig] = {
    
    'FX_MAJOR': AssetClassConfig(
        name='FX_MAJOR',
        display_name='FX Majors',
        assets=[
            'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF',
            'AUDUSD', 'USDCAD', 'NZDUSD'
        ],
        typical_atr_pct=0.5,
        session_driven=True,
        mean_reverting=True,
        liquidity='HIGH',
        sl_atr_mult=1.5,
        tp_atr_mult=2.5,
        max_size_pct=2.0,
        min_rr_ratio=1.5,
        weekend_risk=True,
        news_sensitive=True,
    ),
    
    'FX_CROSS': AssetClassConfig(
        name='FX_CROSS',
        display_name='FX Crosses',
        assets=[
            'EURGBP', 'EURJPY', 'GBPJPY', 'AUDNZD',
            'EURAUD', 'GBPAUD', 'AUDCAD', 'CADJPY'
        ],
        typical_atr_pct=0.7,
        session_driven=True,
        mean_reverting=True,
        liquidity='MEDIUM',
        sl_atr_mult=1.8,
        tp_atr_mult=2.5,
        max_size_pct=1.5,
        min_rr_ratio=1.5,
        weekend_risk=True,
        news_sensitive=True,
    ),
    
    'FX_VOLATILE': AssetClassConfig(
        name='FX_VOLATILE',
        display_name='FX Volatile',
        assets=[
            'GBPJPY', 'GBPNZD', 'GBPAUD', 'GBPCAD',
            'XAUUSD', 'XAGUSD'  # Gold/Silver included
        ],
        typical_atr_pct=1.0,
        session_driven=True,
        mean_reverting=False,
        liquidity='MEDIUM',
        sl_atr_mult=2.0,
        tp_atr_mult=3.0,
        max_size_pct=1.0,
        min_rr_ratio=1.5,
        weekend_risk=False,
        news_sensitive=True,
    ),
    
    'CRYPTO_MAJOR': AssetClassConfig(
        name='CRYPTO_MAJOR',
        display_name='Crypto Majors',
        assets=[
            'BTCUSD', 'ETHUSD', 'BTCUSDT', 'ETHUSDT',
            'XBTUSD', 'BTCEUR'
        ],
        typical_atr_pct=3.0,
        session_driven=False,  # 24/7
        mean_reverting=False,
        liquidity='VARIABLE',
        sl_atr_mult=2.5,
        tp_atr_mult=4.0,
        max_size_pct=0.5,
        min_rr_ratio=1.5,
        weekend_risk=True,  # Trading continues
        news_sensitive=False,
    ),
    
    'CRYPTO_ALT': AssetClassConfig(
        name='CRYPTO_ALT',
        display_name='Crypto Alts',
        assets=[
            'SOLUSD', 'ADAUSD', 'XRPUSD', 'DOTUSD',
            'BNBUSD', 'AVAXUSD', 'MATICUSD'
        ],
        typical_atr_pct=5.0,
        session_driven=False,
        mean_reverting=False,
        liquidity='LOW',
        sl_atr_mult=3.0,
        tp_atr_mult=5.0,
        max_size_pct=0.3,
        min_rr_ratio=1.5,
        weekend_risk=True,
        news_sensitive=False,
    ),
    
    'INDEX': AssetClassConfig(
        name='INDEX',
        display_name='Indices',
        assets=[
            'SPX500', 'NAS100', 'US30', 'DAX40',
            'FTSE100', 'SP500', 'NDX', 'DJI'
        ],
        typical_atr_pct=1.0,
        session_driven=True,
        mean_reverting=True,
        liquidity='HIGH',
        sl_atr_mult=1.5,
        tp_atr_mult=2.5,
        max_size_pct=1.0,
        min_rr_ratio=1.5,
        weekend_risk=False,
        news_sensitive=True,
    ),
    
    'STOCK_LARGE': AssetClassConfig(
        name='STOCK_LARGE',
        display_name='Large Cap Stocks',
        assets=[
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
            'META', 'TSLA', 'JPM', 'V', 'JNJ'
        ],
        typical_atr_pct=2.0,
        session_driven=True,
        mean_reverting=True,
        liquidity='HIGH',
        sl_atr_mult=2.0,
        tp_atr_mult=3.0,
        max_size_pct=1.0,
        min_rr_ratio=1.5,
        weekend_risk=False,
        news_sensitive=True,
    ),
    
    'COMMODITY': AssetClassConfig(
        name='COMMODITY',
        display_name='Commodities',
        assets=[
            'XAUUSD', 'XAGUSD', 'WTIUSD', 'BRENT',
            'NATGAS', 'COPPER'
        ],
        typical_atr_pct=1.5,
        session_driven=True,
        mean_reverting=False,
        liquidity='MEDIUM',
        sl_atr_mult=2.0,
        tp_atr_mult=3.0,
        max_size_pct=1.0,
        min_rr_ratio=1.5,
        weekend_risk=False,
        news_sensitive=True,
    ),
}

# Unknown/fallback class
UNKNOWN_CLASS = AssetClassConfig(
    name='UNKNOWN',
    display_name='Unknown',
    assets=[],
    typical_atr_pct=2.0,
    session_driven=True,
    mean_reverting=True,
    liquidity='MEDIUM',
    sl_atr_mult=2.0,
    tp_atr_mult=3.0,
    max_size_pct=0.5,  # Conservative
    min_rr_ratio=1.5,
    weekend_risk=False,
    news_sensitive=True,
)


# ============================================================================
# ASSET CLASS DETECTOR
# ============================================================================

class AssetClassDetector:
    """
    Detect asset class from symbol name.
    Uses exact match first, then pattern heuristics.
    """
    
    def __init__(self, classes: Dict[str, AssetClassConfig] = None):
        self.classes = classes or ASSET_CLASSES
        self._build_lookup()
    
    def _build_lookup(self):
        """Build symbol -> class lookup."""
        self._lookup = {}
        for class_name, config in self.classes.items():
            for asset in config.assets:
                self._lookup[asset.upper()] = class_name
    
    def detect(self, symbol: str) -> str:
        """
        Detect asset class from symbol.
        
        Args:
            symbol: Asset symbol (e.g., 'EURUSD', 'BTCUSD')
        
        Returns:
            Asset class name
        """
        symbol = symbol.upper().replace('/', '').replace('.', '')
        
        # Exact match
        if symbol in self._lookup:
            return self._lookup[symbol]
        
        # Heuristic patterns
        if self._is_crypto(symbol):
            if any(c in symbol for c in ['BTC', 'ETH', 'XBT']):
                return 'CRYPTO_MAJOR'
            return 'CRYPTO_ALT'
        
        if self._is_forex(symbol):
            if self._is_volatile_fx(symbol):
                return 'FX_VOLATILE'
            if self._is_major_fx(symbol):
                return 'FX_MAJOR'
            return 'FX_CROSS'
        
        if self._is_index(symbol):
            return 'INDEX'
        
        if self._is_commodity(symbol):
            return 'COMMODITY'
        
        if len(symbol) <= 5 and symbol.isalpha():
            return 'STOCK_LARGE'
        
        return 'UNKNOWN'
    
    def _is_crypto(self, symbol: str) -> bool:
        crypto_tokens = ['BTC', 'ETH', 'XRP', 'SOL', 'ADA', 'DOT', 'USDT', 'USDC']
        return any(token in symbol for token in crypto_tokens)
    
    def _is_forex(self, symbol: str) -> bool:
        currencies = ['USD', 'EUR', 'GBP', 'JPY', 'CHF', 'AUD', 'CAD', 'NZD']
        matches = sum(1 for c in currencies if c in symbol)
        return matches >= 2 and len(symbol) == 6
    
    def _is_major_fx(self, symbol: str) -> bool:
        majors = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD']
        return symbol in majors
    
    def _is_volatile_fx(self, symbol: str) -> bool:
        return 'GBP' in symbol and 'JPY' in symbol
    
    def _is_index(self, symbol: str) -> bool:
        index_patterns = ['SPX', 'NAS', 'NDX', 'DAX', 'FTSE', 'US30', 'DJI', 'CAC']
        return any(p in symbol for p in index_patterns)
    
    def _is_commodity(self, symbol: str) -> bool:
        commodities = ['XAU', 'XAG', 'WTI', 'BRENT', 'NATGAS', 'OIL', 'COPPER']
        return any(c in symbol for c in commodities)
    
    def get_config(self, symbol: str) -> AssetClassConfig:
        """Get full config for a symbol."""
        class_name = self.detect(symbol)
        return self.classes.get(class_name, UNKNOWN_CLASS)


# ============================================================================
# CLASS-SPECIFIC PARAMETER APPLIER
# ============================================================================

class ClassParameterApplier:
    """
    Apply asset-class-specific parameter adjustments.
    
    These are MULTIPLIERS on universal logic, not separate logic.
    """
    
    def __init__(self):
        self.detector = AssetClassDetector()
    
    def apply_to_trade(
        self,
        trade_params: Dict,
        symbol: str,
    ) -> Dict:
        """
        Apply class-specific adjustments to trade parameters.
        
        Args:
            trade_params: Dict with sl_atr, tp_atr, size_pct, etc.
            symbol: Asset symbol
        
        Returns:
            Adjusted trade parameters
        """
        config = self.detector.get_config(symbol)
        adjusted = trade_params.copy()
        
        # Adjust SL/TP if not already set
        if 'sl_atr' not in adjusted or adjusted['sl_atr'] is None:
            adjusted['sl_atr'] = config.sl_atr_mult
        
        if 'tp_atr' not in adjusted or adjusted['tp_atr'] is None:
            adjusted['tp_atr'] = config.tp_atr_mult
        
        # Cap position size by class limit
        if 'size_pct' in adjusted:
            adjusted['size_pct'] = min(adjusted['size_pct'], config.max_size_pct)
        
        # Enforce minimum R:R
        if 'rr_ratio' in adjusted:
            if adjusted['rr_ratio'] < config.min_rr_ratio:
                adjusted['_blocked'] = True
                adjusted['_block_reason'] = f'RR {adjusted["rr_ratio"]:.2f} < min {config.min_rr_ratio}'
        
        # Add class metadata
        adjusted['asset_class'] = config.name
        adjusted['class_config'] = {
            'typical_atr_pct': config.typical_atr_pct,
            'session_driven': config.session_driven,
            'mean_reverting': config.mean_reverting,
            'news_sensitive': config.news_sensitive,
        }
        
        return adjusted
    
    def get_symbol_info(self, symbol: str) -> Dict:
        """Get information about a symbol."""
        config = self.detector.get_config(symbol)
        return {
            'symbol': symbol,
            'asset_class': config.name,
            'display_name': config.display_name,
            'typical_atr_pct': config.typical_atr_pct,
            'sl_atr': config.sl_atr_mult,
            'tp_atr': config.tp_atr_mult,
            'max_size_pct': config.max_size_pct,
            'liquidity': config.liquidity,
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def detect_asset_class(symbol: str) -> str:
    """Detect asset class from symbol."""
    detector = AssetClassDetector()
    return detector.detect(symbol)


def get_asset_class_config(symbol: str) -> AssetClassConfig:
    """Get full config for a symbol."""
    detector = AssetClassDetector()
    return detector.get_config(symbol)


def get_class_parameters(symbol: str) -> Dict:
    """Get trading parameters for a symbol."""
    config = get_asset_class_config(symbol)
    return {
        'sl_atr': config.sl_atr_mult,
        'tp_atr': config.tp_atr_mult,
        'max_size_pct': config.max_size_pct,
        'min_rr': config.min_rr_ratio,
    }


def list_asset_classes() -> List[str]:
    """List all defined asset classes."""
    return list(ASSET_CLASSES.keys())


def list_assets_in_class(class_name: str) -> List[str]:
    """List all assets in a class."""
    config = ASSET_CLASSES.get(class_name)
    return config.assets if config else []
