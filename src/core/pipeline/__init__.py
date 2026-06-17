"""
Unified Pipeline Runtime

Full 13-stage pipeline implementation.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import pandas as pd
import numpy as np

from ..interfaces import (
    AlphaSignal, Decision, MarketData,
    SignalDirection, DecisionAction, PipelineStage,
    DataServiceInterface, RiskServiceInterface
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineContext:
    """Shared context across pipeline stages."""
    timestamp: datetime
    
    # Stage outputs
    raw_data: Optional[MarketData] = None
    features: Dict[str, float] = field(default_factory=dict)
    ml_signals: List[AlphaSignal] = field(default_factory=list)
    rl_allocation: Dict[str, float] = field(default_factory=dict)
    regime: str = "unknown"
    sentiment: float = 0.0
    risk_decisions: List[Any] = field(default_factory=list)
    meta_decision: Optional[Any] = None
    portfolio_allocations: Dict[str, float] = field(default_factory=dict)
    gated_decision: Optional[Decision] = None
    final_decision: Optional[Decision] = None
    execution_result: Optional[Any] = None
    
    # Metadata
    stage_times: Dict[int, float] = field(default_factory=dict)
    blocked: bool = False
    block_reason: str = ""


class UnifiedPipeline:
    """
    Unified 13-stage pipeline.
    
    Stages:
    1. Data Load - Load market data
    2. Features - Compute features
    3. ML Brain - Generate ML signals
    4. RL Brain - RL allocation (placeholder)
    5. Regime - Detect regime
    6. Sentiment - Sentiment analysis (placeholder)
    7. Risk - Risk sizing
    8. Meta-Gating - Quality gating
    9. Portfolio - Portfolio allocation
    10. Risk Gates - Hard limits
    11. Throttle Gates - Timing controls
    12. Decision - Final decision
    13. Log - Comprehensive logging
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        
        # Stage enablement
        self.enabled_stages = set(range(1, 14))  # All enabled by default
        
        # Stage configs
        self.stage_configs = config.get('stages', {})
        
        # Services (injected)
        self.data_service: Optional[DataServiceInterface] = None
        self.risk_service: Optional[RiskServiceInterface] = None
        
        # Ensemble config
        self.ensemble_method = config.get('ensemble_method', 'weighted')
        
        logger.info("UnifiedPipeline initialized")
        logger.info(f"  Enabled stages: {len(self.enabled_stages)}/13")
    
    def set_data_service(self, service: DataServiceInterface):
        """Set data service."""
        self.data_service = service
    
    def set_risk_service(self, service: RiskServiceInterface):
        """Set risk service."""
        self.risk_service = service
    
    def disable_stage(self, stage: int):
        """Disable a stage."""
        self.enabled_stages.discard(stage)
    
    def enable_stage(self, stage: int):
        """Enable a stage."""
        self.enabled_stages.add(stage)
    
    # ========================================================================
    # MAIN PROCESS
    # ========================================================================
    
    def process(self,
               signals: List[AlphaSignal],
               data: MarketData) -> Decision:
        """
        Process signals through full pipeline.
        
        Args:
            signals: Alpha signals from pods
            data: Market data
            
        Returns:
            Final decision
        """
        start_time = datetime.now()
        
        # Initialize context
        context = PipelineContext(timestamp=start_time, raw_data=data)
        
        try:
            # Stage 1: Data Load
            if self._is_enabled(1):
                context = self._stage_data_load(context)
            
            # Stage 2: Features
            if self._is_enabled(2) and not context.blocked:
                context = self._stage_features(context)
            
            # Stage 3: ML Brain (Ensemble)
            if self._is_enabled(3) and not context.blocked:
                context = self._stage_ml_brain(context, signals)
            
            # Stage 4: RL Brain (Placeholder)
            if self._is_enabled(4) and not context.blocked:
                context = self._stage_rl_brain(context)
            
            # Stage 5: Regime
            if self._is_enabled(5) and not context.blocked:
                context = self._stage_regime(context)
            
            # Stage 6: Sentiment (Placeholder)
            if self._is_enabled(6) and not context.blocked:
                context = self._stage_sentiment(context)
            
            # Stage 7: Risk
            if self._is_enabled(7) and not context.blocked:
                context = self._stage_risk(context)
            
            # Stage 8: Meta-Gating
            if self._is_enabled(8) and not context.blocked:
                context = self._stage_meta_gating(context)
            
            # Stage 9: Portfolio
            if self._is_enabled(9) and not context.blocked:
                context = self._stage_portfolio(context)
            
            # Stage 10: Risk Gates
            if self._is_enabled(10) and not context.blocked:
                context = self._stage_risk_gates(context)
            
            # Stage 11: Throttle Gates
            if self._is_enabled(11) and not context.blocked:
                context = self._stage_throttle_gates(context)
            
            # Stage 12: Decision
            if self._is_enabled(12) and not context.blocked:
                context = self._stage_decision(context)
            
            # Stage 13: Log
            if self._is_enabled(13):
                context = self._stage_log(context)
            
            # Return final decision
            if context.final_decision:
                return context.final_decision
            else:
                return self._create_pass_decision()
                
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            return self._create_pass_decision(reason=str(e))
    
    def _is_enabled(self, stage: int) -> bool:
        """Check if stage is enabled."""
        return stage in self.enabled_stages
    
    # ========================================================================
    # STAGE IMPLEMENTATIONS
    # ========================================================================
    
    def _stage_data_load(self, context: PipelineContext) -> PipelineContext:
        """Stage 1: Data Load"""
        # Data already loaded, just validate
        if context.raw_data is None:
            context.blocked = True
            context.block_reason = "NO_DATA"
        
        return context
    
    def _stage_features(self, context: PipelineContext) -> PipelineContext:
        """Stage 2: Features"""
        if self.data_service:
            features = self.data_service.get_features(
                context.raw_data.symbol
            )
            context.features = features
        
        return context
    
    def _stage_ml_brain(self, 
                       context: PipelineContext,
                       signals: List[AlphaSignal]) -> PipelineContext:
        """Stage 3: ML Brain (Ensemble)"""
        if not signals:
            context.blocked = True
            context.block_reason = "NO_SIGNALS"
            return context
        
        # Ensemble signals
        if len(signals) == 1:
            context.ml_signals = signals
        else:
            context.ml_signals = self._ensemble_signals(signals)
        
        return context
    
    def _ensemble_signals(self, signals: List[AlphaSignal]) -> List[AlphaSignal]:
        """Ensemble multiple signals."""
        if self.ensemble_method == "weighted":
            # Weight by confidence
            total_conf = sum(s.confidence for s in signals)
            
            if total_conf > 0:
                # Create ensemble signal
                avg_direction = self._majority_vote(signals)
                avg_confidence = total_conf / len(signals)
                avg_expected_return = np.average(
                    [s.expected_return for s in signals],
                    weights=[s.confidence for s in signals]
                )
                
                ensemble = AlphaSignal(
                    source="ensemble",
                    timestamp=datetime.now(),
                    symbol=signals[0].symbol,
                    direction=avg_direction,
                    confidence=avg_confidence,
                    expected_return=avg_expected_return,
                    volatility=signals[0].volatility,
                    metadata={'sources': [s.source for s in signals]}
                )
                
                return [ensemble]
        
        elif self.ensemble_method == "best":
            # Pick best Sharpe
            best = max(signals, key=lambda s: s.sharpe)
            return [best]
        
        return signals
    
    def _majority_vote(self, signals: List[AlphaSignal]) -> SignalDirection:
        """Majority vote on direction."""
        longs = sum(1 for s in signals if s.direction == SignalDirection.LONG)
        shorts = sum(1 for s in signals if s.direction == SignalDirection.SHORT)
        
        return SignalDirection.LONG if longs > shorts else SignalDirection.SHORT
    
    def _stage_rl_brain(self, context: PipelineContext) -> PipelineContext:
        """Stage 4: RL Brain (Placeholder)"""
        # RL allocation - placeholder for future
        # For now, just pass through
        context.rl_allocation = {s.symbol: 1.0 for s in context.ml_signals}
        return context
    
    def _stage_regime(self, context: PipelineContext) -> PipelineContext:
        """Stage 5: Regime Detection"""
        # Simple regime detection based on volatility
        vol = context.features.get('volatility', 0)
        
        if vol > 0.20:
            context.regime = "volatile"
        elif vol < 0.10:
            context.regime = "trending"
        else:
            context.regime = "ranging"
        
        return context
    
    def _stage_sentiment(self, context: PipelineContext) -> PipelineContext:
        """Stage 6: Sentiment (Placeholder)"""
        # Sentiment not implemented yet
        context.sentiment = 0.0
        return context
    
    def _stage_risk(self, context: PipelineContext) -> PipelineContext:
        """Stage 7: Risk Sizing"""
        if not context.ml_signals:
            return context
        
        if self.risk_service:
            for signal in context.ml_signals:
                size = self.risk_service.calculate_position_size(
                    signal, 1.0  # portfolio value
                )
                signal.recommended_size = size
        
        return context
    
    def _stage_meta_gating(self, context: PipelineContext) -> PipelineContext:
        """Stage 8: Meta-Gating"""
        if not context.ml_signals:
            context.blocked = True
            context.block_reason = "NO_SIGNALS_POST_RISK"
            return context
        
        # Simple gating
        for signal in context.ml_signals:
            if signal.confidence < 0.3:
                context.blocked = True
                context.block_reason = "LOW_CONFIDENCE"
                break
        
        return context
    
    def _stage_portfolio(self, context: PipelineContext) -> PipelineContext:
        """Stage 9: Portfolio Allocation"""
        # Simplified allocation
        if context.ml_signals:
            total_weight = sum(s.recommended_size for s in context.ml_signals)
            if total_weight > 0:
                context.portfolio_allocations = {
                    s.symbol: s.recommended_size / total_weight
                    for s in context.ml_signals
                }
        
        return context
    
    def _stage_risk_gates(self, context: PipelineContext) -> PipelineContext:
        """Stage 10: Risk Gates"""
        if self.risk_service:
            status = self.risk_service.get_status()
            
            # Check drawdown
            if status['drawdown'] > 0.20:
                context.blocked = True
                context.block_reason = "MAX_DRAWDOWN"
            
            # Check heat
            if status['portfolio_heat'] > 1.0:
                context.blocked = True
                context.block_reason = "MAX_HEAT"
        
        return context
    
    def _stage_throttle_gates(self, context: PipelineContext) -> PipelineContext:
        """Stage 11: Throttle Gates"""
        # Check if we should throttle
        if self.risk_service:
            status = self.risk_service.get_status()
            if status['throttle'] < 1.0:
                # Reduce position sizes
                for signal in context.ml_signals:
                    signal.recommended_size *= status['throttle']
        
        return context
    
    def _stage_decision(self, context: PipelineContext) -> PipelineContext:
        """Stage 12: Final Decision"""
        if not context.ml_signals:
            context.final_decision = self._create_pass_decision()
            return context
        
        signal = context.ml_signals[0]  # Use first signal
        
        # Determine action
        if signal.direction == SignalDirection.LONG:
            action = DecisionAction.BUY
            side = "long"
        elif signal.direction == SignalDirection.SHORT:
            action = DecisionAction.SELL
            side = "short"
        else:
            action = DecisionAction.PASS
            side = ""
        
        # Get stops from risk service
        if self.risk_service:
            sl, tp = self.risk_service.calculate_stops(
                context.raw_data.close,
                signal
            )
        else:
            sl = context.raw_data.close * 0.98
            tp = context.raw_data.close * 1.04
        
        decision = Decision(
            action=action,
            symbol=signal.symbol,
            side=side,
            size=signal.recommended_size,
            entry_price=context.raw_data.close,
            stop_loss=sl,
            take_profit=tp,
            sources=[signal.source],
            signals=context.ml_signals,
            regime=context.regime,
            confidence=signal.confidence,
            timestamp=datetime.now()
        )
        
        context.final_decision = decision
        return context
    
    def _stage_log(self, context: PipelineContext) -> PipelineContext:
        """Stage 13: Logging"""
        # Log pipeline execution
        logger.info(f"Pipeline completed: {context.timestamp}")
        logger.info(f"  Blocked: {context.blocked} ({context.block_reason})")
        if context.final_decision:
            logger.info(f"  Decision: {context.final_decision.action.value} "
                       f"{context.final_decision.symbol}")
        
        return context
    
    def _create_pass_decision(self, reason: str = "") -> Decision:
        """Create pass decision."""
        return Decision(
            action=DecisionAction.PASS,
            symbol="",
            side="",
            size=0,
            entry_price=0,
            stop_loss=0,
            take_profit=0,
            reason=reason
        )
    
    # ========================================================================
    # STATUS
    # ========================================================================
    
    def get_status(self) -> Dict[str, Any]:
        """Get pipeline status."""
        return {
            'enabled_stages': len(self.enabled_stages),
            'ensemble_method': self.ensemble_method,
            'data_service': self.data_service is not None,
            'risk_service': self.risk_service is not None
        }
