"""
REASONING MODULE
=================

Adaptive learning and reasoning components.

Components:
- ERM: Experience Reasoning Module (learn from mistakes)
"""

from src.reasoning.erm import (
    # Data structures
    ProblemPattern,
    SolutionAttempt,
    ExperienceRecord,
    ERMDecision,
    
    # Core functions
    encode_context,
    bucket_confidence,
    bucket_regime_strength,
    
    # Classes
    ExperienceMemory,
    ReasoningEngine,
    ExperienceLearner,
    ExperienceReasoningModule,
    
    # Singleton
    get_erm,
    reset_erm,
)

__all__ = [
    'ProblemPattern',
    'SolutionAttempt',
    'ExperienceRecord',
    'ERMDecision',
    'encode_context',
    'bucket_confidence',
    'bucket_regime_strength',
    'ExperienceMemory',
    'ReasoningEngine',
    'ExperienceLearner',
    'ExperienceReasoningModule',
    'get_erm',
    'reset_erm',
]
