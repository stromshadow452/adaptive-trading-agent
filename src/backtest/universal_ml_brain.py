"""
Universal ML Brain - Phase-2

LightGBM-based classifier using only normalized universal features.
Works across all FX pairs without asset-specific tuning.
"""

import sys
from pathlib import Path
from typing import Tuple, Dict, Optional
import json

import numpy as np

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "lgb_universal.txt"
METADATA_PATH = PROJECT_ROOT / "models" / "lgb_universal_metadata.json"


# ============================================================================
# UNIVERSAL ML BRAIN
# ============================================================================

class UniversalMLBrain:
    """
    Universal ML Brain using LightGBM.
    
    Input: Dictionary of normalized universal features
    Output: (direction, confidence) tuple
    
    No raw prices, no symbol info, no pip values.
    """
    
    # Default feature order (must match training)
    DEFAULT_FEATURES = [
        'range_ratio', 'atr_ratio_14_50', 'trend_200_zscore',
        'trend_20_zscore', 'trend_50_zscore', 'bb_width_zscore',
        'price_position_50', 'stoch_zscore', 'price_position_20',
        'rsi_zscore',
    ]
    
    # Thresholds
    MIN_CONFIDENCE = 0.40      # Minimum confidence to signal
    STRONG_CONFIDENCE = 0.55   # High confidence threshold
    
    def __init__(self, model_path: Optional[Path] = None):
        self.model = None
        self.features = self.DEFAULT_FEATURES
        self.loaded = False
        
        model_path = model_path or MODEL_PATH
        
        if not HAS_LGB:
            print("⚠️ LightGBM not installed. Using fallback rules.")
            return
        
        if model_path.exists():
            try:
                self.model = lgb.Booster(model_file=str(model_path))
                self.loaded = True
                
                # Load metadata if available
                metadata_path = model_path.parent / 'lgb_universal_metadata.json'
                if metadata_path.exists():
                    with open(metadata_path) as f:
                        metadata = json.load(f)
                    self.features = metadata.get('features', self.DEFAULT_FEATURES)
                
            except Exception as e:
                print(f"⚠️ Failed to load model: {e}")
    
    def predict(self, features: Dict[str, float]) -> Tuple[str, float]:
        """
        Predict trading signal from universal features.
        
        Args:
            features: Dictionary of normalized features
        
        Returns:
            Tuple of (direction, confidence)
            direction: 'BUY', 'SELL', or 'HOLD'
            confidence: 0.0-1.0 probability
        """
        if not self.loaded:
            return self._fallback_predict(features)
        
        try:
            # Build feature vector
            x = np.array([[features.get(f, 0.0) for f in self.features]])
            
            # Get probabilities [BUY, SELL, HOLD]
            probs = self.model.predict(x)[0]
            
            buy_prob = probs[0]
            sell_prob = probs[1]
            hold_prob = probs[2]
            
            # Decision logic
            if buy_prob > sell_prob and buy_prob > hold_prob:
                if buy_prob >= self.MIN_CONFIDENCE:
                    return 'BUY', float(buy_prob)
            
            elif sell_prob > buy_prob and sell_prob > hold_prob:
                if sell_prob >= self.MIN_CONFIDENCE:
                    return 'SELL', float(sell_prob)
            
            # Default to HOLD
            return 'HOLD', float(max(probs))
        
        except Exception as e:
            return self._fallback_predict(features)
    
    def predict_with_details(self, features: Dict[str, float]) -> Dict:
        """
        Predict with full probability details.
        
        Returns dict with all probabilities and metadata.
        """
        if not self.loaded:
            sig, conf = self._fallback_predict(features)
            return {
                'signal': sig,
                'confidence': conf,
                'buy_prob': 0.33,
                'sell_prob': 0.33,
                'hold_prob': 0.34,
                'model': 'fallback',
            }
        
        try:
            x = np.array([[features.get(f, 0.0) for f in self.features]])
            probs = self.model.predict(x)[0]
            
            sig, conf = self.predict(features)
            
            return {
                'signal': sig,
                'confidence': conf,
                'buy_prob': float(probs[0]),
                'sell_prob': float(probs[1]),
                'hold_prob': float(probs[2]),
                'model': 'lightgbm',
                'is_strong': conf >= self.STRONG_CONFIDENCE,
            }
        
        except Exception:
            return self.predict_with_details({})  # Fallback
    
    def _fallback_predict(self, features: Dict[str, float]) -> Tuple[str, float]:
        """
        Rule-based fallback when model unavailable.
        """
        # Trend indicators
        trend_20 = features.get('trend_20_zscore', 0)
        trend_50 = features.get('trend_50_zscore', 0)
        trend_200 = features.get('trend_200_zscore', 0)
        
        # Momentum
        rsi = features.get('rsi_zscore', 0)
        stoch = features.get('stoch_zscore', 0)
        
        # Volatility
        atr_ratio = features.get('atr_ratio_14_50', 1.0)
        
        # Simple scoring
        bull_score = 0
        bear_score = 0
        
        # Trend alignment
        if trend_20 > 0.5 and trend_50 > 0:
            bull_score += 1
        elif trend_20 < -0.5 and trend_50 < 0:
            bear_score += 1
        
        # Momentum
        if rsi < -0.5 and stoch < 0:
            bull_score += 0.5  # Oversold
        elif rsi > 0.5 and stoch > 0:
            bear_score += 0.5  # Overbought
        
        # Decision
        if bull_score > bear_score and bull_score >= 1.0:
            return 'BUY', min(0.45 + bull_score * 0.05, 0.6)
        elif bear_score > bull_score and bear_score >= 1.0:
            return 'SELL', min(0.45 + bear_score * 0.05, 0.6)
        
        return 'HOLD', 0.4


# ============================================================================
# FACTORY
# ============================================================================

def create_universal_brain(model_path: Optional[Path] = None) -> UniversalMLBrain:
    """Factory function to create Universal ML Brain."""
    return UniversalMLBrain(model_path)


# ============================================================================
# TEST
# ============================================================================

if __name__ == '__main__':
    # Quick test
    brain = UniversalMLBrain()
    
    print(f"Model loaded: {brain.loaded}")
    print(f"Features: {len(brain.features)}")
    
    # Test prediction
    test_features = {
        'range_ratio': 0.5,
        'atr_ratio_14_50': 1.2,
        'trend_200_zscore': 1.5,
        'trend_20_zscore': 0.8,
        'trend_50_zscore': 1.0,
        'bb_width_zscore': -0.5,
        'price_position_50': 0.7,
        'stoch_zscore': -0.3,
        'price_position_20': 0.8,
        'rsi_zscore': -0.2,
    }
    
    result = brain.predict_with_details(test_features)
    print(f"\nTest prediction:")
    print(f"  Signal: {result['signal']}")
    print(f"  Confidence: {result['confidence']:.2f}")
    print(f"  BUY: {result['buy_prob']:.2f}")
    print(f"  SELL: {result['sell_prob']:.2f}")
    print(f"  HOLD: {result['hold_prob']:.2f}")
