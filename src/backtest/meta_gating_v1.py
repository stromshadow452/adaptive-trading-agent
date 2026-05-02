"""
SCOPUS MVP v1 - Meta-Gating (Enhanced with MetaJudge)

Production-safe meta-supervision with:
- Signal quality scoring
- Environment stability check
- Simple threshold adjustment
- MetaJudge self-learning integration (optional)

Follows Jarvis-approved architecture.
"""

import pandas as pd
from typing import Optional, Dict, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from src.backtest.adaptive_state import AdaptiveState
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Optional MetaJudge import
try:
    from src.training.meta_judge_trainer import MetaJudge
    HAS_METAJUDGE = True
except ImportError:
    HAS_METAJUDGE = False
    logger.warning("MetaJudge not available - install training module for self-learning")


class MetaGatingV1:
    """
    MVP v1 Meta-Gating - Jarvis supervision layer.
    
    Features:
    - Signal quality scoring (ML-RL agreement, feature completeness)
    - Environment stability check (volatility, news, market hours)
    - Simple regime-based threshold adjustment
    
    NO model selection, NO pack selection in v1 (comes in v2).
    """
    
    def __init__(self, config: dict = None):
        config = config or {}
        
        # Fixed thresholds
        self.min_signal_quality = config.get('min_signal_quality', 0.6)
        self.min_env_stability = config.get('min_env_stability', 0.5)
        
        # Regime adjustments
        self.regime_adjustments = config.get('regime_adjustments', {
            'CRASH': 0.15,   # Raise threshold
            'TREND': -0.05,  # Lower threshold
            'RANGE': 0.0,
            'UNCERTAIN': 0.10
        })
        
        # MetaJudge integration (optional)
        self.meta_judge = None
        self.use_meta_judge = config.get('use_meta_judge', False)
        
        # MetaJudge thresholds for 3-tier decision
        self.meta_block_threshold = config.get('meta_block_threshold', 0.3)
        self.meta_reduce_threshold = config.get('meta_reduce_threshold', 0.6)
        
        if self.use_meta_judge and HAS_METAJUDGE:
            meta_judge_path = config.get('meta_judge_path', 'models/meta_judge_latest.joblib')
            if Path(meta_judge_path).exists():
                try:
                    self.meta_judge = MetaJudge(meta_judge_path)
                    logger.info(f"MetaJudge loaded from {meta_judge_path}")
                except Exception as e:
                    logger.error(f"Failed to load MetaJudge: {e}")
                    self.meta_judge = None
            else:
                logger.warning(f"MetaJudge model not found at {meta_judge_path}")
        
        logger.info(f"MetaGatingV1 initialized (MetaJudge: {'enabled' if self.meta_judge else 'disabled'})")
    
    def supervise(
        self,
        ml_decision: Optional[Dict],
        rl_decision: Optional[Dict],
        features: pd.Series,
        context: Dict
    ) -> Optional[Dict]:
        """
        Supervise and potentially block decisions.
        
        Args:
            ml_decision: ML Brain decision
            rl_decision: RL Brain decision
            features: Feature series
            context: Trading context
        
        Returns:
            Approval dict with meta_score and meta_decision, or None if blocked
        """
        # Score signal quality
        signal_score = self._score_signal_quality(
            ml_decision, rl_decision, features
        )
        
        # Check environment
        env_score = self._check_environment(context)
        
        # Adjust threshold by regime
        min_signal = self._adjust_threshold(context.get('regime', 'UNKNOWN'))
        
        # Block if signal quality too low
        if signal_score < min_signal:
            logger.info(f"Meta-Gating BLOCK: Signal quality {signal_score:.2f} < {min_signal:.2f}")
            return None
        
        # Block if environment unstable
        if env_score < self.min_env_stability:
            logger.info(f"Meta-Gating BLOCK: Environment score {env_score:.2f} < {self.min_env_stability:.2f}")
            return None
        
        # MetaJudge scoring (if enabled)
        meta_score = None
        meta_decision = 'NOT_EVALUATED'
        
        if self.meta_judge:
            try:
                # Build context for MetaJudge
                meta_context = self._build_meta_context(ml_decision, rl_decision, features, context)
                meta_score = self.meta_judge.score(meta_context)
                
                # 3-tier decision logic
                if meta_score < self.meta_block_threshold:
                    meta_decision = 'BLOCKED_BAD'
                    logger.info(f"MetaJudge BLOCK: Score {meta_score:.3f} < {self.meta_block_threshold}")
                    return None
                elif meta_score < self.meta_reduce_threshold:
                    meta_decision = 'ALLOWED_REDUCED'
                    logger.info(f"MetaJudge REDUCE: Score {meta_score:.3f} in [0.3, 0.6)")
                else:
                    meta_decision = 'ALLOWED_NORMAL'
                    logger.debug(f"MetaJudge ALLOW: Score {meta_score:.3f} >= {self.meta_reduce_threshold}")
                    
            except Exception as e:
                logger.error(f"MetaJudge scoring error: {e}")
                meta_score = None
                meta_decision = 'ERROR'
        
        # Approve
        approval = {
            'approved': True,
            'signal_score': signal_score,
            'env_score': env_score,
            'meta_score': meta_score,
            'meta_decision': meta_decision
        }
        
        return approval
    
    def _score_signal_quality(
        self,
        ml_decision: Optional[Dict],
        rl_decision: Optional[Dict],
        features: pd.Series
    ) -> float:
        """
        Score signal quality.
        
        Components:
        - ML-RL agreement (0.2)
        - Feature completeness (0.3)
        - Base score (0.5)
        """
        score = 0.5  # Base score
        
        # ML-RL agreement
        if ml_decision and rl_decision:
            ml_side = ml_decision.get('side')
            rl_action = rl_decision.get('action')
            
            # Map RL action to side
            rl_side = None
            if rl_action == 1:
                rl_side = 'buy'
            elif rl_action == 2:
                rl_side = 'sell'
            
            if ml_side and rl_side and ml_side == rl_side:
                score += 0.2  # Agreement bonus
        
        # Feature completeness
        if len(features) > 0:
            completeness = features.notna().sum() / len(features)
            score += completeness * 0.3
        
        return min(score, 1.0)
    
    def _check_environment(self, context: Dict) -> float:
        """
        Check environment stability.
        
        Penalties:
        - Extreme volatility: -0.3
        - News event: -0.5
        - Outside market hours: -0.4
        - Low liquidity: -0.3
        """
        score = 1.0
        
        # Extreme volatility
        volatility = context.get('volatility', 0)
        if volatility > 0.05:
            score -= 0.3
            logger.debug(f"High volatility {volatility:.3f}, env score penalty")
        
        # News event
        if context.get('news_event_in_5min', False):
            score -= 0.5
            logger.debug("News event detected, env score penalty")
        
        # Market hours (placeholder)
        if not context.get('is_market_hours', True):
            score -= 0.4
            logger.debug("Outside market hours, env score penalty")
        
        # Liquidity (placeholder)
        liquidity = context.get('liquidity_score', 1.0)
        if liquidity < 0.3:
            score -= 0.3
            logger.debug(f"Low liquidity {liquidity:.2f}, env score penalty")
        
        return max(score, 0.0)
    
    def _adjust_threshold(self, regime: str) -> float:
        """
        Adjust minimum signal threshold by regime.
        
        Simple rule-based adjustment.
        """
        adjustment = self.regime_adjustments.get(regime, 0.0)
        adjusted = self.min_signal_quality + adjustment
        
        return max(0.0, min(1.0, adjusted))
    
    def _build_meta_context(
        self,
        ml_decision: Optional[Dict],
        rl_decision: Optional[Dict],
        features: pd.Series,
        context: Dict
    ) -> Dict:
        """
        Build context dictionary for MetaJudge scoring.
        
        Extracts relevant features from ml_decision, rl_decision, features, and context.
        """
        meta_context = {}
        
        # ML confidence
        if ml_decision:
            meta_context['ml_confidence'] = ml_decision.get('confidence', 0.0)
        
        # RL confidence (if available)
        if rl_decision:
            meta_context['rl_confidence'] = rl_decision.get('confidence', 0.0)
        
        # Market features
        for key in ['volatility', 'atr_pct', 'rsi', 'bb_width']:
            if key in features.index:
                meta_context[key] = features[key]
        
        # Context features
        for key in ['regime', 'hour_of_day', 'day_of_week', 'recent_winrate_10', 
                    'consecutive_losses', 'bars_since_last_trade']:
            if key in context:
                meta_context[key] = context[key]
        
        # Side (from ml_decision)
        if ml_decision and 'side' in ml_decision:
            meta_context['side'] = ml_decision['side']
        
        return meta_context
    
    # ==================== ADAPTIVE GATING ====================
    
    def supervise_adaptive(
        self,
        decision: Dict,
        features: pd.Series,
        context: Dict,
        adaptive_state: 'AdaptiveState'
    ) -> Tuple[bool, str]:
        """
        Adaptive supervision with AdaptiveState integration.
        
        Blocks trades if:
        - regime = DANGER
        - HIGH volatility + LOW confidence
        - Expected value negative
        - Recent performance poor
        - Loss streak >= 3 (cooldown)
        
        Args:
            decision: Trade decision dict with 'confidence', 'side', etc.
            features: Feature series
            context: Trading context
            adaptive_state: AdaptiveState instance
        
        Returns:
            (allowed, reason) tuple
        """
        from src.backtest.adaptive_state import AdaptiveState
        
        regime = adaptive_state.last_regime
        vol_level = adaptive_state.volatility_level
        confidence = decision.get('confidence', 0.5)
        kalman_slope = features.get('kalman_slope', features.get('M5_kalman_slope', 0))
        
        # === WINRATE OPTIMIZATION: Progressive loss-streak tightening ===
        conf_penalty = 0.0
        if adaptive_state.loss_streak == 1:
            conf_penalty = 0.05
        elif adaptive_state.loss_streak == 2:
            conf_penalty = 0.10
        elif adaptive_state.loss_streak >= 3:
            conf_penalty = 0.15  # Plus cooldown below
        
        effective_min_conf = 0.65 + conf_penalty
        
        # === Block 0: Absolute Minimum Confidence Floor (0.65) ===
        if confidence < 0.65:
            logger.info(f"Meta BLOCK: Low confidence ({confidence:.2f} < 0.65)")
            return False, f'BLOCKED: Low confidence ({confidence:.2f})'
        
        # === Block 1: DANGER Regime → MR-only defensive mode ===
        if regime == 'DANGER':
            strategy = str(context.get('strategy_route', 'UNKNOWN')).upper()
            if strategy != 'MEAN_REVERSION':
                logger.info("Meta BLOCK: DANGER regime allows MEAN_REVERSION only")
                return False, 'BLOCKED: DANGER regime non-MR'
            if confidence < max(0.75, effective_min_conf):
                logger.info(f"Meta BLOCK: DANGER MR needs conf >= {max(0.75, effective_min_conf):.2f} ({confidence:.2f})")
                return False, 'BLOCKED: DANGER regime + LOW confidence'
            logger.info("Meta PASS: DANGER regime MR-only defensive mode")
        
        # === Block 2: RANGE Regime → Allow with HIGH confidence only ===
        if regime == 'RANGE' and confidence < 0.75:
            logger.info(f"Meta BLOCK: RANGE regime needs conf >= 0.75 ({confidence:.2f})")
            return False, 'BLOCKED: RANGE regime + LOW confidence'
        
        # === Block 3: HIGH Volatility → BLOCK ALL (unconditional) ===
        if vol_level == 'HIGH':
            logger.info("Meta BLOCK: HIGH volatility - noise too high")
            return False, 'BLOCKED: HIGH volatility (pause trading)'
        
        # === Block 4: MID Volatility + Confidence < 0.65 ===
        if vol_level == 'MID' and confidence < 0.65:
            logger.info(f"Meta BLOCK: MID vol requires conf >= 0.65 ({confidence:.2f})")
            return False, 'BLOCKED: MID vol + LOW confidence'
        
        # === Block 5: LOW Volatility + Confidence < effective_min ===
        if vol_level == 'LOW' and confidence < effective_min_conf:
            logger.info(f"Meta BLOCK: LOW vol requires conf >= {effective_min_conf:.2f} ({confidence:.2f})")
            return False, f'BLOCKED: LOW vol + LOW confidence (after streak penalty)'
        
        # === Block 6: Weak Trend Strength (Kalman Slope) - lowered threshold ===
        TREND_STRENGTH_MIN = 0.002  # 0.2% normalized slope (lowered from 0.5%)
        if abs(kalman_slope) < TREND_STRENGTH_MIN:
            logger.info(f"Meta BLOCK: Weak trend (|slope|={abs(kalman_slope):.4f} < {TREND_STRENGTH_MIN})")
            return False, f'BLOCKED: Weak trend strength ({abs(kalman_slope):.4f})'
        
        # === Block 7: Negative Expected Value ===
        ev = self._calc_expected_value(confidence, adaptive_state)
        if ev < 0:
            logger.info(f"Meta BLOCK: Negative EV ({ev:.3f})")
            return False, f'BLOCKED: Negative EV ({ev:.3f})'
        
        # === Block 8: Loss Streak Cooldown (3+ losses) ===
        if adaptive_state.is_in_cooldown():
            logger.info(f"Meta BLOCK: Cooldown (loss streak {adaptive_state.loss_streak})")
            return False, f'BLOCKED: Cooldown (loss streak {adaptive_state.loss_streak})'
        
        # === Block 9: Poor Recent Performance ===
        recent_winrate = adaptive_state.get_rolling_winrate()
        if recent_winrate < 0.30 and adaptive_state.trade_count >= 10:
            logger.info(f"Meta BLOCK: Poor performance (winrate={recent_winrate:.1%})")
            return False, 'BLOCKED: Poor recent performance'
        
        # === Block 10: Survival Mode (allow but warn) ===
        if adaptive_state.is_in_survival_mode():
            logger.warning(f"SURVIVAL MODE active (DD={adaptive_state.rolling_drawdown:.1%}) - allowing with reduced risk")
        
        # === All checks passed - ALLOW TRADE ===
        logger.info(f"Meta ALLOW: regime={regime}, vol={vol_level}, conf={confidence:.2f}, slope={kalman_slope:.4f}")
        return True, 'PASSED'
    
    def _calc_expected_value(
        self, 
        confidence: float, 
        adaptive_state: 'AdaptiveState'
    ) -> float:
        """
        Calculate expected value for trade decision.
        
        EV = (winrate * avg_win) - (lossrate * avg_loss)
        Simplified: EV = winrate * avg_r - (1-winrate) * 1
        """
        # Try to get bucket-specific winrate
        bucket_winrate = adaptive_state.get_bucket_winrate(confidence)
        
        if bucket_winrate is not None:
            winrate = bucket_winrate
        else:
            # Fallback to confidence as proxy
            winrate = confidence
        
        # Use avg R-multiple (default 1.0 if not enough data)
        avg_r = max(adaptive_state.avg_r_multiple, 1.0)
        
        # EV calculation
        ev = winrate * avg_r - (1 - winrate) * 1.0
        
        return ev
    
    def get_gate_context(self, adaptive_state: 'AdaptiveState') -> Dict:
        """Get current gating context for logging"""
        return {
            'regime': adaptive_state.last_regime,
            'volatility_level': adaptive_state.volatility_level,
            'rolling_winrate': adaptive_state.get_rolling_winrate(),
            'loss_streak': adaptive_state.loss_streak,
            'drawdown': adaptive_state.rolling_drawdown,
            'is_cooldown': adaptive_state.is_in_cooldown(),
            'is_survival': adaptive_state.is_in_survival_mode(),
        }

