"""
FinRL Adapter for RL Brain Fallback (Stage 5)

Provides a clean interface to trained PPO/FinRL models with:
- HMAC integrity verification
- Feature parity validation
- Graceful degradation
"""
import os
import numpy as np
from typing import Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class FinRLAdapter:
    """
    Adapter for FinRL/PPO models with JARVIS safety checks.
    
    Usage:
        adapter = FinRLAdapter(model_path="models/finrl/EURUSD_M15_ppo.joblib", 
                               registry=model_registry,
                               primary_meta=primary_model_meta)
        
        if adapter.is_available():
            conf, action = adapter.predict_proba(feature_vector)
    """
    
    def __init__(self, 
                 model_path: Optional[str],
                 registry: Any,
                 primary_meta: dict):
        """
        Initialize FinRL adapter with safety checks.
        
        Args:
            model_path: Path to FinRL model file (.joblib)
            registry: ModelRegistry instance for HMAC verification
            primary_meta: Primary model metadata for feature parity check
        """
        self.model = None
        self.meta = {}
        self.available = False
        
        if not model_path or not os.path.exists(model_path):
            logger.warning(f"[RL-ADAPTER] Model path not found: {model_path}")
            return
        
        try:
            # Load model with HMAC integrity check
            self.model, self.meta = registry.load_model_from_path(model_path)
            
            # Feature parity check
            primary_hash = primary_meta.get("feature_hash") or primary_meta.get("feature_order_hash")
            rl_hash = self.meta.get("feature_hash") or self.meta.get("feature_order_hash")
            
            if primary_hash and rl_hash and primary_hash != rl_hash:
                logger.error(f"[RL-ADAPTER] Feature parity mismatch! Primary: {primary_hash[:8]}, RL: {rl_hash[:8]}")
                self.model = None
                return
            
            self.available = True
            logger.info(f"[RL-ADAPTER] Loaded FinRL model: features={len(self.meta.get('feature_names', []))}, hash={rl_hash[:8] if rl_hash else 'N/A'}")
            
        except Exception as e:
            logger.warning(f"[RL-ADAPTER] Failed to load RL model: {e}")
            self.model = None
            self.available = False
    
    def is_available(self) -> bool:
        """Check if RL model is loaded and ready."""
        return self.available and self.model is not None
    
    def predict_proba(self, features: np.ndarray) -> Tuple[float, int]:
        """
        Get RL model prediction with confidence.
        
        Args:
            features: Feature vector (1D numpy array)
        
        Returns:
            (confidence, action) where:
                - confidence: float in [0, 1], higher = stronger conviction
                - action: int in {-1, 0, +1} for sell/hold/buy
        
        Raises:
            RuntimeError: If model not available
        """
        if not self.is_available():
            raise RuntimeError("RL model not available")
        
        try:
            # Reshape for model input
            if len(features.shape) == 1:
                features = features.reshape(1, -1)
            
            # Get prediction from RL model
            # Try different interfaces (PPO, generic policy, etc.)
            if hasattr(self.model, 'predict'):
                # Stable-Baselines3 PPO interface
                action, _states = self.model.predict(features, deterministic=True)
                action = int(action[0]) if hasattr(action, '__len__') else int(action)
                
                # Estimate confidence from action value
                # For PPO, we can use value function or just high confidence for non-hold
                confidence = 0.75 if action != 0 else 0.50
                
            elif hasattr(self.model, 'predict_proba'):
                # Sklearn-like interface
                proba = self.model.predict_proba(features)
                
                # Convert probabilities to action and confidence
                if proba.shape[1] == 3:  # [sell_prob, hold_prob, buy_prob]
                    action = np.argmax(proba[0]) - 1  # Map 0,1,2 to -1,0,+1
                    confidence = float(np.max(proba[0]))
                elif proba.shape[1] == 2:  # [sell_prob, buy_prob]
                    action = 1 if proba[0, 1] > proba[0, 0] else -1
                    confidence = float(max(proba[0, 0], proba[0, 1]))
                else:
                    raise ValueError(f"Unexpected proba shape: {proba.shape}")
                    
            else:
                # Fallback: use raw model output
                output = self.model(features) if callable(self.model) else None
                if output is None:
                    raise AttributeError("Model has no predict/predict_proba method")
                
                # Simple heuristic
                action = 1 if float(output) > 0 else -1
                confidence = min(abs(float(output)), 1.0)
            
            # Clamp confidence to [0, 1]
            confidence = max(0.0, min(1.0, float(confidence)))
            
            # Clamp action to {-1, 0, +1}
            if action > 0:
                action = 1
            elif action < 0:
                action = -1
            else:
                action = 0
            
            return confidence, action
            
        except Exception as e:
            logger.error(f"[RL-ADAPTER] Prediction error: {e}")
            return 0.0, 0  # Safe fallback


# Backward compatibility helper
def load_finrl_adapter(model_path: Optional[str], 
                       registry: Any,
                       primary_meta: dict) -> Optional[FinRLAdapter]:
    """
    Factory function to create FinRLAdapter with error handling.
    
    Returns None if adapter cannot be created.
    """
    try:
        adapter = FinRLAdapter(model_path, registry, primary_meta)
        return adapter if adapter.is_available() else None
    except Exception as e:
        logger.warning(f"[RL-ADAPTER] Failed to create adapter: {e}")
        return None
