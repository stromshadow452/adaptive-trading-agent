"""
SCOPUS MVP v1 - Safe Feature Reactor

Production-safe feature engineering with:
- Batch-based importance tracking (not per-trade)
- Regime-based pack selection
- Offline pruning (not live)
- NO feature value boosting

Follows Jarvis-approved architecture.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
import logging
from dataclasses import dataclass, field
import json

from src.market_data import Timeframe, Symbol, MarketDataStore, build_mtf_context, add_mtf_features

logger = logging.getLogger(__name__)


# Feature pack definitions
FEATURE_PACKS = {
    'ohlcv_base': [
        'open', 'high', 'low', 'close', 'volume'
    ],
    
    'trend_pack': [
        'sma_20', 'sma_50', 'ema_12', 'ema_26', 
        'trend_flag', 'slope_20'
    ],
    
    'momentum_pack': [
        'rsi_14', 'macd', 'macd_signal', 'macd_hist',
        'stoch_k', 'stoch_d'
    ],
    
    'volatility_pack': [
        'atr_14', 'bb_width', 'bb_upper', 'bb_lower',
        'volatility'
    ],
    
    'sr_pack': [
        'sr_zone_id', 'sr_distance', 'sr_strength',
        'pivot_point', 'pivot_r1', 'pivot_s1'
    ],
    
    # NEW: Adaptive features for adaptive trading
    'adaptive_pack': [
        'ewma_volatility', 'atr_normalized', 
        'volatility_percentile', 'kalman_slope'
    ]
}


@dataclass
class TradeRecord:
    """Record of a trade for batch importance computation"""
    features: Dict[str, float]
    outcome: str  # 'win' or 'loss'
    timestamp: str


class SafeFeatureImportanceTracker:
    """
    Batch-based feature importance tracker.
    
    Computes importance using permutation importance over batches,
    not noisy per-trade updates.
    """
    
    def __init__(self, batch_size: int = 100, save_path: Optional[Path] = None):
        self.batch_size = batch_size
        self.save_path = save_path or Path("logs/feature_importance.json")
        
        self.trade_buffer: List[TradeRecord] = []
        self.importance_scores: Dict[str, float] = {}
        self.batch_count = 0
        
        # Load existing scores if available
        self._load_scores()
    
    def record_trade(self, features: Dict[str, float], outcome: str, timestamp: str):
        """Record a trade for batch processing"""
        self.trade_buffer.append(TradeRecord(
            features=features,
            outcome=outcome,
            timestamp=timestamp
        ))
        
        # Process batch when full
        if len(self.trade_buffer) >= self.batch_size:
            self._compute_batch_importance()
            self.trade_buffer = []
            self.batch_count += 1
    
    def _compute_batch_importance(self):
        """Compute importance using permutation importance"""
        if len(self.trade_buffer) < 10:
            logger.warning("Not enough trades in batch for importance computation")
            return
        
        try:
            # Extract features and outcomes
            X_list = []
            y_list = []
            
            for record in self.trade_buffer:
                X_list.append(record.features)
                y_list.append(1 if record.outcome == 'win' else 0)
            
            X = pd.DataFrame(X_list)
            y = np.array(y_list)
            
            # Compute permutation importance
            from sklearn.inspection import permutation_importance
            from sklearn.ensemble import RandomForestClassifier
            
            # Train simple model on batch
            model = RandomForestClassifier(
                n_estimators=50,
                max_depth=5,
                random_state=42
            )
            model.fit(X, y)
            
            # Compute importance
            result = permutation_importance(
                model, X, y,
                n_repeats=10,
                random_state=42,
                n_jobs=-1
            )
            
            # Update scores with exponential moving average
            alpha = 0.3  # Smoothing factor
            for i, feat in enumerate(X.columns):
                new_score = result.importances_mean[i]
                
                if feat in self.importance_scores:
                    # EMA update
                    self.importance_scores[feat] = (
                        alpha * new_score + 
                        (1 - alpha) * self.importance_scores[feat]
                    )
                else:
                    self.importance_scores[feat] = new_score
            
            # Save scores
            self._save_scores()
            
            logger.info(f"Batch {self.batch_count}: Computed importance for {len(X.columns)} features")
            
        except Exception as e:
            logger.error(f"Error computing batch importance: {e}")
    
    def get_top_features(self, n: int = 100) -> List[str]:
        """Get top N features by importance"""
        if not self.importance_scores:
            return []
        
        sorted_feats = sorted(
            self.importance_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [f[0] for f in sorted_feats[:n]]
    
    def _save_scores(self):
        """Save importance scores to disk"""
        try:
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.save_path, 'w') as f:
                json.dump({
                    'importance_scores': self.importance_scores,
                    'batch_count': self.batch_count
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving importance scores: {e}")
    
    def _load_scores(self):
        """Load importance scores from disk"""
        try:
            if self.save_path.exists():
                with open(self.save_path, 'r') as f:
                    data = json.load(f)
                    self.importance_scores = data.get('importance_scores', {})
                    self.batch_count = data.get('batch_count', 0)
                    
                logger.info(f"Loaded {len(self.importance_scores)} feature importance scores")
        except Exception as e:
            logger.error(f"Error loading importance scores: {e}")


class OfflineFeaturePruner:
    """
    Offline feature pruner - NEVER prunes during live trading.
    
    Used during model retraining to remove weak features.
    """
    
    def __init__(self, min_importance: float = 0.01, min_features: int = 50):
        self.min_importance = min_importance
        self.min_features = min_features
    
    def prune(
        self,
        importance_scores: Dict[str, float],
        min_samples: int = 500
    ) -> List[str]:
        """
        Prune features - OFFLINE ONLY.
        
        Args:
            importance_scores: Feature importance scores
            min_samples: Minimum samples required before pruning
        
        Returns:
            List of features to keep
        """
        if len(importance_scores) < min_samples:
            logger.info(f"Not enough samples ({len(importance_scores)} < {min_samples}), keeping all features")
            return list(importance_scores.keys())
        
        # Keep features above threshold
        active = [
            feat for feat, score in importance_scores.items()
            if score >= self.min_importance
        ]
        
        # Always keep at least min_features
        if len(active) < self.min_features:
            sorted_feats = sorted(
                importance_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            active = [f[0] for f in sorted_feats[:self.min_features]]
            
            logger.info(f"Pruned to minimum {self.min_features} features")
        else:
            logger.info(f"Kept {len(active)} features above importance threshold")
        
        return active


class SafeFeatureSelector:
    """
    Safe feature selector - selects features, does NOT boost values.
    
    "Boosting" means selecting better subsets, not multiplying values.
    """
    
    def __init__(self, importance_scores: Dict[str, float]):
        self.importance_scores = importance_scores
    
    def select_top_features(self, n: int = 100) -> List[str]:
        """Select top N features by importance"""
        if not self.importance_scores:
            return []
        
        sorted_feats = sorted(
            self.importance_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [f[0] for f in sorted_feats[:n]]
    
    def select_by_pack(self, regime: str) -> List[str]:
        """
        Select feature packs by regime.
        
        This is the safe way to "boost" - choose relevant features,
        don't multiply their values.
        """
        pack_map = {
            'TREND': ['ohlcv_base', 'trend_pack', 'momentum_pack'],
            'RANGE': ['ohlcv_base', 'sr_pack', 'volatility_pack'],
            'CRASH': ['ohlcv_base', 'volatility_pack'],
            'UNCERTAIN': ['ohlcv_base']
        }
        
        selected_packs = pack_map.get(regime, ['ohlcv_base'])
        
        # Flatten pack features
        features = []
        for pack_name in selected_packs:
            features.extend(FEATURE_PACKS.get(pack_name, []))
        
        return list(set(features))  # Remove duplicates


class SafeFeatureReactor:
    """
    MVP v1 Feature Reactor - Production-safe.
    
    Features:
    - Batch-based importance tracking
    - Regime-based pack selection
    - NO live pruning
    - NO feature value boosting
    """
    
    def __init__(self, config: Optional[Dict] = None, store: Optional[MarketDataStore] = None):
        self.config = config or {}
        self.store = store
        
        # Initialize components
        self.importance_tracker = SafeFeatureImportanceTracker(
            batch_size=self.config.get('batch_size', 100)
        )
        
        self.feature_selector = SafeFeatureSelector(
            self.importance_tracker.importance_scores
        )
        
        # === HTF CACHE: Avoid reloading D1/H4/H1 on every candle ===
        # Key: (symbol, timeframe, date_str) -> DataFrame
        self._htf_cache: Dict[tuple, pd.DataFrame] = {}
        self._htf_cache_hits = 0
        self._htf_cache_misses = 0

    
    def extract_mtf(
        self,
        candle: pd.Series,
        context: Dict,
        regime: str
    ) -> pd.Series:
        """
        Extract MTF features for current candle.
        
        Args:
            candle: Current OHLCV candle
            context: Trading context (must include symbol)
            regime: Current market regime
            
        Returns:
            MTF Feature series
        """
        if not self.store:
            logger.warning("MarketDataStore not provided, cannot build MTF context")
            return pd.Series()
            
        symbol = context.get('symbol')
        if not symbol:
            logger.warning("Symbol not found in context")
            return pd.Series()
        
        # Convert string to Symbol enum if needed
        if isinstance(symbol, str):
            try:
                symbol = Symbol(symbol)
            except ValueError:
                logger.warning(f"Invalid symbol: {symbol}")
                return pd.Series()
            
        # Extract timestamp from candle
        # candle.name might be an integer index, so check for 'timestamp' column first
        if 'timestamp' in candle.index:
            timestamp = candle['timestamp']
        elif hasattr(candle, 'name') and isinstance(candle.name, pd.Timestamp):
            timestamp = candle.name
        else:
            logger.warning("Could not extract timestamp from candle")
            return pd.Series()
        
        # Build context window (enough for indicators)
        # D1 SMA(50) needs ~50 days, but we load D1 separately.
        # For M5 indicators, 5 days is plenty.
        start = timestamp - pd.Timedelta(days=5)
        end = timestamp
        
        # Build MTF context
        # Note: We assume M5 base for now as per MVP v1 MTF spec
        mtf_df = build_mtf_context(
            store=self.store,
            symbol=symbol,
            base_timeframe=Timeframe.M5,
            higher_timeframes=[Timeframe.M15, Timeframe.H1, Timeframe.H4, Timeframe.D1],
            start=start,
            end=end
        )
        
        if mtf_df.empty:
            return pd.Series()
            
        # Add features
        mtf_df = add_mtf_features(
            df=mtf_df,
            base_timeframe=Timeframe.M5,
            higher_timeframes=[Timeframe.M15, Timeframe.H1, Timeframe.H4, Timeframe.D1]
        )

        # Return last row (current candle features) with router-critical aliases.
        features = mtf_df.iloc[-1].copy()
        alias_map = {
            'M5_adx_14': ['adx_14', 'adx14'],
            'M5_atr_pctile': ['atr_pctile'],
            'M5_boll_z': ['boll_z'],
        }
        for source_col, aliases in alias_map.items():
            if source_col in features.index:
                for alias in aliases:
                    features[alias] = features[source_col]

        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(method='ffill').fillna(method='bfill')
        return features
    
    def extract(
        self,
        candle: pd.Series,
        context: Dict,
        regime: str
    ) -> pd.Series:
        """
        Extract features for current candle.
        
        Args:
            candle: Current OHLCV candle
            context: Trading context (history, etc.)
            regime: Current market regime
        
        Returns:
            Feature series
        """
        # Get historical data
        history = context.get('history')
        if history is None or len(history) < 50:
            logger.warning("Insufficient history for feature extraction")
            return pd.Series()
        
        # Compute technical indicators
        df_with_indicators = self._compute_indicators(history)
        
        # Get features from last row
        features = df_with_indicators.iloc[-1]
        
        # Select features by regime (pack selection)
        selected_features = self.feature_selector.select_by_pack(regime)
        
        # Filter to selected features only
        features = features[features.index.isin(selected_features)]
        
        return features
    
    def _compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute technical indicators including adaptive features"""
        df = df.copy()
        
        # Price-based
        df['returns'] = df['close'].pct_change()
        
        # Trend pack
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        df['trend_flag'] = (df['sma_20'] > df['sma_50']).astype(int) * 2 - 1
        df['slope_20'] = df['sma_20'].diff(5) / df['sma_20']
        
        # Momentum pack
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Stochastic
        low_14 = df['low'].rolling(14).min()
        high_14 = df['high'].rolling(14).max()
        df['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14)
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()
        
        # Volatility pack
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr_14'] = tr.rolling(14).mean()
        
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (2 * bb_std)
        df['bb_lower'] = df['bb_middle'] - (2 * bb_std)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        df['volatility'] = df['returns'].rolling(20).std()
        
        # === ADAPTIVE FEATURES ===
        
        # 1. EWMA Volatility (O(1) update approximation)
        df['ewma_volatility'] = df['returns'].abs().ewm(span=50, adjust=False).mean()
        
        # 2. Normalized ATR (ATR as % of price)
        df['atr_normalized'] = df['atr_14'] / df['close']
        
        # 3. Volatility Percentile (rolling rank over 20 periods)
        df['volatility_percentile'] = df['atr_14'].rolling(20).apply(
            lambda x: (x.values[-1] <= x.values).sum() / len(x) if len(x) > 0 else 0.5,
            raw=False
        )
        
        # 4. Kalman Trend Slope (approximated via EMA difference)
        kf_fast = df['close'].ewm(span=5, adjust=False).mean()
        kf_slow = df['close'].ewm(span=20, adjust=False).mean()
        df['kalman_slope'] = (kf_fast - kf_slow) / df['close']
        
        # S/R pack (placeholders for now)
        df['sr_zone_id'] = 0
        df['sr_distance'] = 0.0
        df['sr_strength'] = 0.0
        df['pivot_point'] = (df['high'] + df['low'] + df['close']) / 3
        df['pivot_r1'] = 2 * df['pivot_point'] - df['low']
        df['pivot_s1'] = 2 * df['pivot_point'] - df['high']
        
        return df
    
    def record_trade(self, features: Dict[str, float], outcome: str, timestamp: str):
        """Record trade for batch importance tracking"""
        self.importance_tracker.record_trade(features, outcome, timestamp)
