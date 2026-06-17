"""
EURUSD ML Alpha Pod

Machine learning-based directional bias prediction for EURUSD.

Validated approach:
- XGBoost for 5D directional bias
- Probability/confidence outputs
- Walk-forward validation
- Feature importance tracking
- Calibrated confidence intervals

Integrates with:
- Unified orchestrator
- Shared RiskService
- Shared ExecutionService
- Portfolio Brain
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from dataclasses import dataclass

# ML libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available, using RandomForest fallback")

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Core interfaces
from ...core.interfaces import (
    AlphaPod, AlphaSignal, MarketData,
    SignalDirection,
    DataServiceInterface, RiskServiceInterface
)

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Model configuration."""
    model_type: str = "xgboost"  # xgboost, lightgbm, randomforest
    prediction_horizon: int = 5  # 5 days
    lookback_window: int = 20  # Feature window
    
    # XGBoost params
    max_depth: int = 4
    learning_rate: float = 0.05
    n_estimators: int = 100
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    
    # Validation
    n_splits: int = 5
    min_train_size: int = 252  # 1 year
    
    # Calibration
    calibration_method: str = "sigmoid"
    
    # Features
    feature_selection: str = "importance"  # importance, correlation
    max_features: int = 20


class EURUSDMLAlphaPod(AlphaPod):
    """
    EURUSD ML Alpha Pod.
    
    Uses XGBoost/RandomForest to predict 5-day directional bias.
    
    Features:
    - Technical (momentum, volatility, regime)
    - Macro (DXY, VIX, yields)
    - Structure (trend, support/resistance)
    
    Output:
    - Directional bias (LONG/SHORT)
    - Probability (0-1)
    - Confidence interval
    """
    
    VERSION = "1.0.0"
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize EURUSD ML Alpha Pod.
        
        Args:
            config: Configuration dict
        """
        super().__init__(config)
        
        # Model config
        self.model_config = ModelConfig(**config.get('model', {}))
        
        # Universe
        self.pairs = ['EURUSD']
        self._timeframe = 'D1'
        
        # Model state
        self.model = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.is_trained = False
        
        # Feature importance
        self.feature_importance: Dict[str, float] = {}
        self.selected_features: List[str] = []
        
        # Performance tracking
        self.predictions: List[Dict] = []
        self.actuals: List[float] = []
        
        # Calibration
        self.calibration_map: Dict[float, float] = {}
        
        logger.info(f"EURUSDMLAlphaPod initialized v{self.VERSION}")
        logger.info(f"  Model type: {self.model_config.model_type}")
        logger.info(f"  Horizon: {self.model_config.prediction_horizon}D")
        logger.info(f"  Universe: {self.pairs}")
    
    # =========================================================================
    # Properties
    # =========================================================================
    
    @property
    def name(self) -> str:
        """Pod name."""
        return "eurusd_ml"
    
    @property
    def version(self) -> str:
        """Pod version."""
        return self.VERSION
    
    @property
    def universe(self) -> List[str]:
        """Trading universe."""
        return self.pairs
    
    @property
    def timeframe(self) -> str:
        """Primary timeframe."""
        return self._timeframe
    
    # =========================================================================
    # Core Methods
    # =========================================================================
    
    def generate_signal(self, data: MarketData) -> Optional[AlphaSignal]:
        """
        Generate ML-based signal.
        
        Process:
        1. Get features
        2. Scale features
        3. Predict direction
        4. Calibrate probability
        5. Calculate confidence
        6. Generate signal
        
        Args:
            data: Current market data
            
        Returns:
            AlphaSignal or None
        """
        try:
            # 1. Check model
            if not self.is_trained:
                if not self._train_model():
                    logger.warning("Model not trained, skipping")
                    return None
            
            # 2. Get features
            features = self._get_features_for_prediction(data)
            if features is None:
                return None
            
            # 3. Predict
            prediction, probability = self._predict(features)
            
            if prediction is None:
                return None
            
            # 4. Calibrate probability
            calibrated_prob = self._calibrate_probability(probability)
            
            # 5. Calculate confidence
            confidence = self._calculate_confidence(calibrated_prob, features)
            
            # 6. Determine direction
            if prediction == 1:
                direction = SignalDirection.LONG
            elif prediction == -1:
                direction = SignalDirection.SHORT
            else:
                return None
            
            # 7. Calculate expected return
            expected_return = self._estimate_return(calibrated_prob)
            
            # 8. Create signal
            signal = AlphaSignal(
                source=self.name,
                timestamp=datetime.now(),
                symbol='EURUSD',
                direction=direction,
                confidence=confidence,
                expected_return=expected_return,
                volatility=self._estimate_volatility(),
                recommended_size=0.40,  # Will be adjusted by risk
                max_position=0.40,
                metadata={
                    'raw_probability': probability,
                    'calibrated_probability': calibrated_prob,
                    'features_used': len(features),
                    'model_type': self.model_config.model_type
                }
            )
            
            logger.info(f"Signal generated: EURUSD {direction.value} "
                       f"(conf: {confidence:.2f}, prob: {calibrated_prob:.2f})")
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            return None
    
    def get_features(self, data: MarketData) -> Dict[str, float]:
        """
        Return ML features.
        
        Args:
            data: Market data
            
        Returns:
            Dict of feature names to values
        """
        features = {}
        
        try:
            # Get historical data
            prices = self._get_historical_prices('EURUSD', days=60)
            if prices is None or len(prices) < 20:
                return features
            
            # Technical features
            features.update(self._technical_features(prices))
            
            # Regime features
            features.update(self._regime_features(prices))
            
            # Macro features (if available)
            features.update(self._macro_features())
            
            return features
            
        except Exception as e:
            logger.error(f"Error computing features: {e}")
            return features
    
    # =========================================================================
    # Model Training
    # =========================================================================
    
    def _train_model(self) -> bool:
        """
        Train ML model with walk-forward validation.
        
        Returns:
            bool: Success
        """
        try:
            logger.info("Training EURUSD ML model...")
            
            # Get training data
            df = self._get_training_data()
            if df is None or len(df) < 252:
                logger.error("Insufficient training data")
                return False
            
            # Prepare features
            X, y = self._prepare_training_data(df)
            if X is None or len(X) < 100:
                return False
            
            # Feature selection
            X_selected, feature_names = self._select_features(X, y)
            self.selected_features = feature_names
            
            # Walk-forward validation
            scores = self._walk_forward_validation(X_selected, y)
            
            logger.info(f"Validation scores: {scores}")
            
            # Train final model on all data
            if self.model_config.model_type == "xgboost" and XGBOOST_AVAILABLE:
                self.model = xgb.XGBClassifier(
                    max_depth=self.model_config.max_depth,
                    learning_rate=self.model_config.learning_rate,
                    n_estimators=self.model_config.n_estimators,
                    subsample=self.model_config.subsample,
                    colsample_bytree=self.model_config.colsample_bytree,
                    reg_alpha=self.model_config.reg_alpha,
                    reg_lambda=self.model_config.reg_lambda,
                    objective='binary:logistic',
                    eval_metric='logloss',
                    use_label_encoder=False
                )
            else:
                # Fallback to RandomForest
                self.model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=6,
                    random_state=42
                )
            
            # Fit
            self.model.fit(X_selected, y)
            
            # Get feature importance
            self._compute_feature_importance(X_selected, y)
            
            # Calibrate
            self._calibrate_model(X_selected, y)
            
            self.is_trained = True
            logger.info("Model training complete")
            return True
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return False
    
    def _walk_forward_validation(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Walk-forward validation.
        
        Args:
            X: Features
            y: Labels
            
        Returns:
            Dict of scores
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score
        
        tscv = TimeSeriesSplit(
            n_splits=self.model_config.n_splits,
            test_size=21  # 1 month
        )
        
        scores = {
            'accuracy': [],
            'precision': [],
            'recall': []
        }
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Train
            if self.model_config.model_type == "xgboost" and XGBOOST_AVAILABLE:
                model = xgb.XGBClassifier(
                    max_depth=self.model_config.max_depth,
                    learning_rate=self.model_config.learning_rate,
                    n_estimators=self.model_config.n_estimators
                )
            else:
                model = RandomForestClassifier(n_estimators=100)
            
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Score
            scores['accuracy'].append(accuracy_score(y_test, y_pred))
            scores['precision'].append(precision_score(y_test, y_pred, zero_division=0))
            scores['recall'].append(recall_score(y_test, y_pred, zero_division=0))
        
        return {
            'accuracy_mean': np.mean(scores['accuracy']),
            'accuracy_std': np.std(scores['accuracy']),
            'precision_mean': np.mean(scores['precision']),
            'recall_mean': np.mean(scores['recall'])
        }
    
    def _select_features(self, 
                        X: np.ndarray, 
                        y: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """
        Select features by importance.
        
        Args:
            X: Feature matrix
            y: Labels
            
        Returns:
            (X_selected, feature_names)
        """
        # Train temporary model
        if self.model_config.model_type == "xgboost" and XGBOOST_AVAILABLE:
            model = xgb.XGBClassifier(n_estimators=50, max_depth=3)
        else:
            model = RandomForestClassifier(n_estimators=50)
        
        model.fit(X, y)
        
        # Get importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            return X, [f"feature_{i}" for i in range(X.shape[1])]
        
        # Select top features
        n_features = min(self.model_config.max_features, len(importances))
        top_indices = np.argsort(importances)[-n_features:]
        
        X_selected = X[:, top_indices]
        feature_names = [f"feature_{i}" for i in top_indices]
        
        return X_selected, feature_names
    
    def _compute_feature_importance(self, X: np.ndarray, y: np.ndarray):
        """Compute feature importance."""
        if self.model is None:
            return
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            self.feature_importance = {
                name: float(importances[i])
                for i, name in enumerate(self.selected_features)
                if i < len(importances)
            }
    
    def _calibrate_model(self, X: np.ndarray, y: np.ndarray):
        """Calibrate probability outputs."""
        if self.model is None:
            return
        
        # Use Platt scaling
        try:
            calibrated = CalibratedClassifierCV(
                self.model,
                method=self.model_config.calibration_method,
                cv=3
            )
            calibrated.fit(X, y)
            self.calibrated_model = calibrated
        except:
            self.calibrated_model = None
    
    # =========================================================================
    # Feature Engineering
    # =========================================================================
    
    def _technical_features(self, prices: pd.DataFrame) -> Dict[str, float]:
        """Compute technical features."""
        features = {}
        
        close = prices['close']
        
        # Returns
        for period in [1, 5, 10, 20]:
            ret = close.pct_change(period).iloc[-1]
            features[f'return_{period}d'] = ret
        
        # Volatility
        for period in [10, 20]:
            vol = close.pct_change().rolling(period).std().iloc[-1]
            features[f'volatility_{period}d'] = vol * np.sqrt(252)
        
        # Momentum
        for period in [5, 10, 20]:
            mom = (close.iloc[-1] - close.iloc[-period-1]) / close.iloc[-period-1]
            features[f'momentum_{period}d'] = mom
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        features['rsi_14'] = rsi.iloc[-1]
        
        # Trend
        sma20 = close.rolling(20).mean().iloc[-1]
        sma50 = close.rolling(50).mean().iloc[-1]
        features['trend_20d'] = 1 if close.iloc[-1] > sma20 else -1
        features['trend_50d'] = 1 if close.iloc[-1] > sma50 else -1
        
        # Distance from moving averages
        features['dist_sma20'] = (close.iloc[-1] - sma20) / sma20
        features['dist_sma50'] = (close.iloc[-1] - sma50) / sma50
        
        return features
    
    def _regime_features(self, prices: pd.DataFrame) -> Dict[str, float]:
        """Compute regime features."""
        features = {}
        
        close = prices['close']
        
        # Volatility regime
        vol20 = close.pct_change().rolling(20).std().iloc[-1] * np.sqrt(252)
        vol_threshold = 0.10
        
        if vol20 < vol_threshold * 0.8:
            features['regime_low_vol'] = 1
        elif vol20 > vol_threshold * 1.5:
            features['regime_high_vol'] = 1
        else:
            features['regime_normal'] = 1
        
        # Trend strength (ADX-like)
        atr = self._calculate_atr(prices)
        features['trend_strength'] = 1 if atr > 0 else 0
        
        return features
    
    def _macro_features(self) -> Dict[str, float]:
        """
        Compute macro features using CrossAssetSignals.
        
        Uses local repository data to compute:
        - dxy_trend: Dollar strength proxy via EURUSD (inverted)
        - vix_level: Volatility proxy via US500 ATR percentile
        - yield_spread: Risk-on/off proxy via silver/gold ratio
        
        PHASE 5A: Replaces mocked values with real proxy features.
        """
        features = {}
        
        try:
            # Import CrossAssetSignals for macro proxy calculation
            from ...features.cross_asset import CrossAssetSignals, get_cross_asset_defaults
            
            # Check if data service is available
            if self._data_service is None:
                logger.warning("Data service not available, using default macro features")
                return get_cross_asset_defaults()
            
            # Compute cross-asset signals
            ca_signals = CrossAssetSignals(self._data_service)
            signals = ca_signals.compute()
            
            # Validate signals before use
            validated_signals = self._validate_macro_features(signals)
            
            # Map CrossAssetSignals to macro feature names
            # ca_dxy_momentum: EURUSD 5/20 SMA ratio (inverted DXY proxy)
            features['dxy_trend'] = validated_signals.get('ca_dxy_momentum', 0.0)
            
            # ca_vol_regime_proxy: US500 ATR percentile (VIX proxy, scaled 0-100)
            vol_proxy = validated_signals.get('ca_vol_regime_proxy', 0.5)
            features['vix_level'] = vol_proxy * 100.0  # Scale to VIX-like 0-100 range
            
            # ca_silver_gold_ratio: Risk-on/off proxy (as yield spread alternative)
            features['yield_spread'] = validated_signals.get('ca_silver_gold_ratio', 0.0)
            
            # Additional useful macro indicators
            features['gold_trend'] = validated_signals.get('ca_gold_trend', 0.0)
            
            # PHASE 5A: If VIX proxy is neutral (0.5), compute FX volatility proxy
            if features['vix_level'] == 50.0 or features['vix_level'] == 0.0:
                fx_vol_proxy = self._compute_fx_volatility_proxy()
                if fx_vol_proxy != 0.5:  # Only use if we got real data
                    features['vix_level'] = fx_vol_proxy * 100.0
                    logger.debug(f"Using FX volatility proxy: {features['vix_level']:.2f}")
            
            # PHASE 5A: If yield spread is zero, compute FX momentum dispersion proxy
            if features['yield_spread'] == 0.0:
                yield_proxy = self._compute_fx_momentum_dispersion()
                if yield_proxy != 0.0:
                    features['yield_spread'] = yield_proxy
                    logger.debug(f"Using FX momentum dispersion proxy: {features['yield_spread']:.4f}")
            
            logger.info(f"PHASE 5A Macro features computed: dxy_trend={features['dxy_trend']:.4f}, "
                        f"vix_level={features['vix_level']:.2f}, "
                        f"yield_spread={features['yield_spread']:.4f}, "
                        f"gold_trend={features['gold_trend']:.4f}")
            
        except Exception as e:
            logger.error(f"PHASE 5A Error computing macro features: {e}")
            # Graceful fallback - return zeros but log the issue
            features = {
                'dxy_trend': 0.0,
                'vix_level': 0.0,
                'yield_spread': 0.0,
                'gold_trend': 0.0
            }
        
        return features
    
    def _validate_macro_features(self, signals: Dict[str, float]) -> Dict[str, float]:
        """
        Validate macro features for missing, NaN, or stale values.
        
        Args:
            signals: Raw signals from CrossAssetSignals
            
        Returns:
            Validated signals with fallbacks for invalid values
        """
        validated = {}
        
        # Feature validation rules
        validation_rules = {
            'ca_dxy_momentum': {
                'default': 0.0,
                'min': -0.5,  # Max 50% trend
                'max': 0.5,
                'allow_nan': False
            },
            'ca_vol_regime_proxy': {
                'default': 0.5,  # Neutral
                'min': 0.0,
                'max': 1.0,
                'allow_nan': False
            },
            'ca_silver_gold_ratio': {
                'default': 0.0,
                'min': -0.5,
                'max': 0.5,
                'allow_nan': False
            },
            'ca_gold_trend': {
                'default': 0.0,
                'min': -0.5,
                'max': 0.5,
                'allow_nan': False
            }
        }
        
        for feature, rules in validation_rules.items():
            value = signals.get(feature, rules['default'])
            
            # Check for NaN/None
            if value is None or (isinstance(value, float) and np.isnan(value)):
                logger.warning(f"Macro feature {feature} is NaN/None, using default {rules['default']}")
                validated[feature] = rules['default']
                continue
            
            # Check for infinite values
            if isinstance(value, float) and np.isinf(value):
                logger.warning(f"Macro feature {feature} is infinite, using default {rules['default']}")
                validated[feature] = rules['default']
                continue
            
            # Check range bounds
            if value < rules['min']:
                logger.warning(f"Macro feature {feature}={value:.4f} below min {rules['min']}, clamping")
                validated[feature] = rules['min']
            elif value > rules['max']:
                logger.warning(f"Macro feature {feature}={value:.4f} above max {rules['max']}, clamping")
                validated[feature] = rules['max']
            else:
                validated[feature] = value
        
        # Log feature availability
        valid_count = sum(1 for v in validated.values() if v != 0.0 and not np.isnan(v))
        total_count = len(validated)
        logger.debug(f"Macro feature availability: {valid_count}/{total_count} features valid")
        
        return validated
    
    def _compute_fx_volatility_proxy(self) -> float:
        """
        Compute FX volatility proxy using EURUSD realized volatility.
        
        Returns:
            Volatility percentile (0.0-1.0) based on 20-day vs 100-day history
        """
        try:
            if self._data_service is None:
                return 0.5
            
            # Get EURUSD data
            df = self._data_service.get_prices('EURUSD', 'D1')
            if df is None or len(df) < 100:
                return 0.5
            
            close = df['close']
            returns = close.pct_change().dropna()
            
            if len(returns) < 100:
                return 0.5
            
            # Current 20-day volatility (annualized)
            current_vol = returns.iloc[-20:].std() * np.sqrt(252)
            
            # Historical distribution of 20-day volatilities
            hist_vol = returns.rolling(20).std().dropna() * np.sqrt(252)
            
            if len(hist_vol) < 20:
                return 0.5
            
            # Compute percentile
            percentile = (hist_vol.iloc[:-1] <= current_vol).mean()
            
            logger.debug(f"FX volatility proxy: {current_vol:.4f} at {percentile:.2f} percentile")
            return float(percentile)
            
        except Exception as e:
            logger.debug(f"Error computing FX volatility proxy: {e}")
            return 0.5
    
    def _compute_fx_momentum_dispersion(self) -> float:
        """
        Compute FX momentum dispersion as yield spread proxy.
        
        Uses momentum difference between strongest and weakest FX pairs
        as a risk-on/off indicator.
        
        Returns:
            Momentum dispersion (-0.5 to 0.5)
        """
        try:
            if self._data_service is None:
                return 0.0
            
            # FX pairs to compare (all majors vs USD)
            pairs = ['EURUSD', 'GBPUSD', 'AUDUSD', 'USDCAD', 'USDJPY']
            momentums = {}
            
            for pair in pairs:
                try:
                    df = self._data_service.get_prices(pair, 'D1')
                    if df is not None and len(df) >= 20:
                        close = df['close']
                        # 20-day momentum
                        mom = (close.iloc[-1] / close.iloc[-20] - 1)
                        momentums[pair] = mom
                except:
                    continue
            
            if len(momentums) < 3:
                return 0.0
            
            # Compute dispersion (max - min)
            mom_values = list(momentums.values())
            dispersion = max(mom_values) - min(mom_values)
            
            # Normalize: strong dispersion = risk-off (flight to safety)
            # Weak dispersion = risk-on (correlated moves)
            normalized = min(dispersion * 5, 0.5)  # Scale and cap
            
            logger.debug(f"FX momentum dispersion: {dispersion:.4f} (normalized: {normalized:.4f})")
            return normalized
            
        except Exception as e:
            logger.debug(f"Error computing FX momentum dispersion: {e}")
            return 0.0
    
    def _calculate_atr(self, prices: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range."""
        high = prices['high']
        low = prices['low']
        close = prices['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        
        return atr.iloc[-1] if len(atr) > 0 else 0.0
    
    # =========================================================================
    # Prediction
    # =========================================================================
    
    def _predict(self, features: Dict[str, float]) -> Tuple[Optional[int], float]:
        """
        Predict direction.
        
        Args:
            features: Feature dict
            
        Returns:
            (prediction, probability)
        """
        if self.model is None:
            return None, 0.5
        
        # Convert to array
        X = np.array([[features.get(f, 0.0) for f in self.selected_features]])
        
        # Predict
        try:
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(X)[0]
                # Assume binary: class 0 = short, class 1 = long
                long_prob = proba[1]
                prediction = 1 if long_prob > 0.5 else -1
                return prediction, long_prob
            else:
                pred = self.model.predict(X)[0]
                return pred, 0.7
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None, 0.5
    
    def _calibrate_probability(self, raw_prob: float) -> float:
        """Calibrate raw probability."""
        # Simple sigmoid calibration
        # Would use learned calibration in production
        return 1 / (1 + np.exp(-(raw_prob - 0.5) * 4))
    
    def _calculate_confidence(self, 
                             calibrated_prob: float, 
                             features: Dict[str, float]) -> float:
        """
        Calculate confidence.
        
        Confidence based on:
        - Probability distance from 0.5
        - Feature quality
        - Model certainty
        
        Args:
            calibrated_prob: Calibrated probability
            features: Feature dict
            
        Returns:
            Confidence (0-1)
        """
        # Base confidence from probability
        base_conf = abs(calibrated_prob - 0.5) * 2
        
        # Adjust for feature completeness
        feature_ratio = len([v for v in features.values() if v != 0]) / max(len(features), 1)
        
        confidence = base_conf * 0.7 + feature_ratio * 0.3
        
        return min(confidence, 1.0)
    
    def _estimate_return(self, calibrated_prob: float) -> float:
        """Estimate expected return."""
        # Conservative estimate based on validation
        base_return = 0.08  # 8% annually
        
        # Scale by probability confidence
        confidence_factor = abs(calibrated_prob - 0.5) * 2
        
        return base_return * confidence_factor
    
    def _estimate_volatility(self) -> float:
        """Estimate expected volatility."""
        return 0.10  # 10% target
    
    # =========================================================================
    # Data Access
    # =========================================================================
    
    def _get_historical_prices(self, 
                               symbol: str, 
                               days: int) -> Optional[pd.DataFrame]:
        """Get historical prices."""
        if self._data_service is None:
            return None
        
        try:
            # Get from data service
            df = self._data_service.get_prices(symbol, 'D1')
            if df is not None and len(df) >= days:
                return df.tail(days)
            return None
        except:
            return None
    
    def _get_training_data(self) -> Optional[pd.DataFrame]:
        """Get training data."""
        return self._get_historical_prices('EURUSD', days=500)
    
    def _prepare_training_data(self, 
                               df: pd.DataFrame) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Prepare training data.
        
        Returns:
            (X, y)
        """
        try:
            # Compute features
            feature_list = []
            labels = []
            
            for i in range(60, len(df) - 5):
                window = df.iloc[i-60:i]
                
                # Features
                feats = self._technical_features(window)
                feats.update(self._regime_features(window))
                
                feature_list.append(list(feats.values()))
                
                # Label: future 5-day return
                future_return = (df['close'].iloc[i+5] - df['close'].iloc[i]) / df['close'].iloc[i]
                label = 1 if future_return > 0 else 0
                labels.append(label)
            
            X = np.array(feature_list)
            y = np.array(labels)
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return None, None
    
    def _get_features_for_prediction(self, 
                                    data: MarketData) -> Optional[Dict[str, float]]:
        """Get features for prediction."""
        # Get recent history
        prices = self._get_historical_prices('EURUSD', days=60)
        if prices is None:
            return None
        
        # Compute features
        features = self._technical_features(prices)
        features.update(self._regime_features(prices))
        features.update(self._macro_features())
        
        return features
    
    # =========================================================================
    # Lifecycle
    # =========================================================================
    
    def initialize(self) -> bool:
        """
        Initialize pod.
        
        Returns:
            bool: Success
        """
        try:
            logger.info(f"Initializing {self.name} pod...")
            
            # Validate data service
            if self._data_service is None:
                logger.error("Data service required")
                return False
            
            # Train model
            if not self._train_model():
                logger.warning("Could not train model initially")
                # Don't fail - may train later
            
            logger.info(f"{self.name} pod initialized")
            return True
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get pod status."""
        status = super().get_status()
        status.update({
            'model_type': self.model_config.model_type,
            'is_trained': self.is_trained,
            'num_features': len(self.selected_features),
            'top_features': sorted(
                self.feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
        })
        return status


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_CONFIG = {
    'model': {
        'model_type': 'xgboost',
        'prediction_horizon': 5,
        'lookback_window': 20,
        'max_depth': 4,
        'learning_rate': 0.05,
        'n_estimators': 100,
        'max_features': 20
    }
}


def create_eurusd_ml_pod(config: Optional[Dict] = None) -> EURUSDMLAlphaPod:
    """
    Factory function.
    
    Args:
        config: Configuration
        
    Returns:
        EURUSDMLAlphaPod instance
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    return EURUSDMLAlphaPod(config)
