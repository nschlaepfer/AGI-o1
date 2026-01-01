"""
ARC-AGI-2 Competition Module.

This module implements a record-breaking system for the ARC-AGI-2 benchmark,
combining the best techniques from winning solutions:

- Poetiq-style refinement loops (54% SOTA)
- NVARC test-time training (24% contest-constrained)
- ARChitects 2D-aware reasoning
- DSL-guided program synthesis
- Multi-expert ensemble with voting
- Synthetic data generation for pre-training

Target: >50% on ARC-AGI-2 semi-private evaluation set.

Key Components:
- Grid: Core data structure for ARC grids with transformations
- ARCTask/ARCDataset: Task and dataset management
- ARCRefinementLoop: Poetiq-style iterative solving
- ARCDSL: Domain-specific language for program synthesis
- TestTimeTrainer: Per-task model adaptation
- ARCSolver: Main orchestrator combining all strategies
- CompetitionRunner: Kaggle submission pipeline

Usage:
    from amarillo.arc import ARCSolver, ARCDataset

    # Load tasks
    dataset = ARCDataset.from_directory("path/to/arc/training")

    # Create solver
    solver = ARCSolver(llm_client)

    # Solve
    result = await solver.solve_task(task)
"""

from .grid import Grid, GridPair, BoundingBox, GridObject
from .dataset import ARCDataset, ARCTask, load_kaggle_data
from .evaluator import ARCEvaluator, EvaluationResult, CodeExecutionEvaluator
from .refinement import ARCRefinementLoop, RefinementResult, ParallelRefinementLoop
from .solver import ARCSolver, SolverConfig, SolveResult, CompetitionResult, ParallelSolver
from .competition import CompetitionRunner, CompetitionConfig, KaggleNotebookRunner

# DSL components
from .dsl import (
    ARCDSL,
    DSLPrimitive,
    DSLProgram,
    create_standard_dsl,
    BeamSearch,
    GeneticSearch,
    HybridSearch,
)

# Synthesis components
from .synthesis import (
    SyntheticGenerator,
    ConceptGenerator,
    augment_task,
    generate_augmentations,
)

# TTT components
from .ttt import (
    TestTimeTrainer,
    TTTConfig,
    LoRAAdapter,
)

__version__ = "0.1.0"

__all__ = [
    # Core
    "Grid",
    "GridPair",
    "BoundingBox",
    "GridObject",
    "ARCDataset",
    "ARCTask",
    "load_kaggle_data",
    # Evaluation
    "ARCEvaluator",
    "EvaluationResult",
    "CodeExecutionEvaluator",
    # Refinement
    "ARCRefinementLoop",
    "RefinementResult",
    "ParallelRefinementLoop",
    # Solver
    "ARCSolver",
    "SolverConfig",
    "SolveResult",
    "CompetitionResult",
    "ParallelSolver",
    # Competition
    "CompetitionRunner",
    "CompetitionConfig",
    "KaggleNotebookRunner",
    # DSL
    "ARCDSL",
    "DSLPrimitive",
    "DSLProgram",
    "create_standard_dsl",
    "BeamSearch",
    "GeneticSearch",
    "HybridSearch",
    # Synthesis
    "SyntheticGenerator",
    "ConceptGenerator",
    "augment_task",
    "generate_augmentations",
    # TTT
    "TestTimeTrainer",
    "TTTConfig",
    "LoRAAdapter",
]
