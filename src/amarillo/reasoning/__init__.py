"""Reasoning and scoring modules for Amarillo."""

from .fluid_reasoning import (
    FluidReasoner,
    FluidResult,
    Solution,
    ExpertConfig,
    MultiExpertReasoner,
    create_simple_evaluator,
    DEFAULT_SOLVER_PROMPT,
    DEFAULT_FEEDBACK_PROMPT,
)
from .ensemble_reasoning import (
    EnsembleReasoner,
    ReasoningPhase,
    ReasoningResult,
    SoftScore,
    SoftScorer,
    FeedbackInjector,
    VotingMechanism,
    create_ensemble_reasoner,
    run_fluid_reasoning,
)
from .scoring import (
    SoftScorer as ScoringScorer,
    ScoreResult,
    ScoringCriteria,
    CodeExecutionScorer,
    code_syntax_scorer,
    code_length_scorer,
    keyword_presence_scorer,
    output_match_scorer,
    array_similarity_scorer,
    structural_scorer,
    create_composite_scorer,
    build_feedback,
)

__all__ = [
    # Fluid Reasoning
    "FluidReasoner",
    "FluidResult",
    "Solution",
    "ExpertConfig",
    "MultiExpertReasoner",
    "create_simple_evaluator",
    "DEFAULT_SOLVER_PROMPT",
    "DEFAULT_FEEDBACK_PROMPT",
    # Ensemble Reasoning
    "EnsembleReasoner",
    "ReasoningPhase",
    "ReasoningResult",
    "SoftScore",
    "SoftScorer",
    "FeedbackInjector",
    "VotingMechanism",
    "create_ensemble_reasoner",
    "run_fluid_reasoning",
    # Scoring
    "ScoringScorer",
    "ScoreResult",
    "ScoringCriteria",
    "CodeExecutionScorer",
    "code_syntax_scorer",
    "code_length_scorer",
    "keyword_presence_scorer",
    "output_match_scorer",
    "array_similarity_scorer",
    "structural_scorer",
    "create_composite_scorer",
    "build_feedback",
]
