"""
SCOPUS MVP v1 - ML Brain

The decision-making core that uses the trained XGBoost model 
to generate trading signals based on MTF features.

Enhanced with Context Brain integration for adaptive thresholds.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import logging

from src.backtest.feature_reactor_v1 import SafeFeatureReactor

logger = logging.getLogger(__name__)


class MLBrainV1:
    """
    ML Brain V1 - XGBoost-based decision engine.
    
    Responsibilities:
    1. Load trained model
    2. Request features from Feature Reactor
    3. Generate BUY/SELL/HOLD signals with confidence
    4. Adapt thresholds based on Context Brain output
    """
    
    def __init__(
        self,
        model_path: Path,
        feature_names_path: Path,
        feature_reactor: SafeFeatureReactor,
        config: Optional[Dict] = None
    ):
        self.config = config or {}
        self.feature_reactor = feature_reactor
        
        # Load model
        logger.info(f"Loading model from {model_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        self.model = joblib.load(model_path)
        
        # Load feature names
        logger.info(f"Loading feature names from {feature_names_path}")
        if not feature_names_path.exists():
            raise FileNotFoundError(f"Feature names not found: {feature_names_path}")
        
        with open(feature_names_path, 'r') as f:
            self.feature_names = [line.strip() for line in f.readlines() if line.strip()]
            
        logger.info(f"ML Brain initialized with {len(self.feature_names)} features")
        
        # Default thresholds (used in CONFIRMATION mode)
        self.buy_threshold = self.config.get('buy_threshold', 0.6)
        self.sell_threshold = self.config.get('sell_threshold', 0.6)
        
        # Adaptive threshold config
        self.learning_buy_threshold = self.config.get('learning_buy_threshold', 0.35)
        self.learning_sell_threshold = self.config.get('learning_sell_threshold', 0.35)
    
    def predict(
        self,
        candle: pd.Series,
        context: Dict,
        regime: str,
        features: Optional[pd.Series] = None,
        context_output: Optional[Dict] = None
    ) -> Tuple[str, float]:
        """
        Generate prediction for current candle.
        
        Args:
            candle: Current OHLCV candle
            context: Trading context (must include MarketDataStore access)
            regime: Current market regime
            features: Optional pre-calculated features (Stage 2 output)
            context_output: Optional output from Context Brain (Stage 2.5)
            
        Returns:
            (signal, confidence)
            signal: 'BUY', 'SELL', 'HOLD'
            confidence: 0.0 to 1.0
        """
        try:
            # 1. Extract MTF features if not provided
            if features is None:
                features = self.feature_reactor.extract_mtf(candle, context, regime)
            
            if features.empty:
                logger.warning("Empty features returned, defaulting to HOLD")
                return 'HOLD', 0.0
            
            # 2. Align features with model expectation
            # Create DataFrame with correct columns, filling missing with 0 (safe fallback)
            X = pd.DataFrame([features], columns=self.feature_names).fillna(0)
            
            # 3. Determine adaptive thresholds based on Context Brain output
            buy_threshold, sell_threshold = self._get_adaptive_thresholds(context_output)
            
            # 4. Predict probabilities
            # classes are [0, 1] -> [SELL/HOLD, BUY] usually, but check training mapping
            # In our training: 1 = BUY (future return > 0), 0 = SELL/HOLD (future return <= 0)
            # For this binary model:
            # Prob(1) > threshold -> BUY
            # Prob(1) < (1-threshold) -> SELL (i.e. Prob(0) > threshold)
            
            probs = self.model.predict_proba(X)[0]
            prob_0 = probs[0]  # Probability of Class 0 (Bearish/Neutral)
            prob_1 = probs[1]  # Probability of Class 1 (Bullish)
            
            # 5. Signal Logic with adaptive thresholds
            signal = 'HOLD'
            confidence = 0.0
            
            if prob_1 > buy_threshold:
                signal = 'BUY'
                confidence = prob_1
            elif prob_0 > sell_threshold:
                signal = 'SELL'
                confidence = prob_0
            else:
                signal = 'HOLD'
                confidence = max(prob_0, prob_1)
            
            # 6. Apply context confidence modulation (if available)
            if context_output and signal != 'HOLD':
                context_conf = context_output.get('context_confidence', 1.0)
                manipulation_risk = context_output.get('manipulation_risk', 0.0)
                
                # Reduce confidence if high manipulation risk
                if manipulation_risk > 0.5:
                    confidence *= (1.0 - manipulation_risk * 0.3)
                    logger.debug(f"Confidence reduced due to manipulation risk: {manipulation_risk:.2f}")
                
                # Boost confidence if context agrees
                if context_conf > 0.6:
                    confidence = min(1.0, confidence * (1.0 + (context_conf - 0.6) * 0.2))
            
            return signal, confidence
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return 'HOLD', 0.0
    
    def _get_adaptive_thresholds(self, context_output: Optional[Dict]) -> Tuple[float, float]:
        """
        Get adaptive thresholds based on Context Brain output.
        
        Returns:
            (buy_threshold, sell_threshold)
        """
        if context_output is None:
            # No context output - use default thresholds
            return self.buy_threshold, self.sell_threshold
        
        operating_mode = context_output.get('operating_mode', 'CONFIRMATION')
        data_sufficiency = context_output.get('data_sufficiency', 1.0)
        
        if operating_mode == 'LEARNING':
            # Interpolate between learning and confirmation thresholds
            # based on data sufficiency
            buy_t = self.learning_buy_threshold + (
                (self.buy_threshold - self.learning_buy_threshold) * data_sufficiency * 0.5
            )
            sell_t = self.learning_sell_threshold + (
                (self.sell_threshold - self.learning_sell_threshold) * data_sufficiency * 0.5
            )
            return buy_t, sell_t
        else:
            # CONFIRMATION mode - use strict thresholds
            return self.buy_threshold, self.sell_threshold

