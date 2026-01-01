"""
ARC Solver - Main orchestrator for ARC-AGI-2 competition.

Combines all components into a unified solving pipeline:
1. Task analysis and concept detection
2. Strategy selection (refinement, DSL, TTT)
3. Multi-approach solving with ensembling
4. Time and compute management
5. Self-auditing and early stopping
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from .grid import Grid
from .dataset import ARCTask
from .evaluator import ARCEvaluator, EvaluationResult
from .refinement import ARCRefinementLoop, RefinementResult
from .dsl import ARCDSL, create_standard_dsl, BeamSearch, GeneticSearch, HybridSearch
from .synthesis import generate_augmentations
from .ttt import TestTimeTrainer, TTTConfig

logger = logging.getLogger(__name__)


class SolvingStrategy(Enum):
    """Available solving strategies."""
    REFINEMENT = auto()  # LLM-based iterative refinement
    DSL_SEARCH = auto()  # Program synthesis with DSL
    TTT = auto()  # Test-time training
    DIRECT = auto()  # Direct LLM prediction
    ENSEMBLE = auto()  # Combine multiple strategies


@dataclass
class SolverConfig:
    """Configuration for the ARC solver."""
    # Strategy selection
    strategies: List[SolvingStrategy] = field(default_factory=lambda: [
        SolvingStrategy.REFINEMENT,
        SolvingStrategy.DSL_SEARCH,
    ])
    ensemble_strategies: bool = True

    # Time management
    total_time_budget_hours: float = 12.0
    time_per_task_seconds: float = 180.0  # 3 minutes default
    min_time_per_task: float = 30.0
    max_time_per_task: float = 600.0

    # Refinement settings
    refinement_max_iterations: int = 10
    refinement_model: str = "gpt-4o"
    refinement_temperature: float = 0.7

    # DSL settings
    dsl_beam_width: int = 50
    dsl_max_depth: int = 5
    dsl_max_time: float = 30.0

    # TTT settings
    use_ttt: bool = True
    ttt_time_budget: float = 60.0

    # Ensemble settings
    ensemble_size: int = 4
    voting_threshold: float = 0.5

    # Quality thresholds
    early_stop_threshold: float = 1.0
    acceptable_score: float = 0.9

    # Compute settings
    num_parallel_workers: int = 4
    use_gpu: bool = True


@dataclass
class SolveResult:
    """Result from solving a single task."""
    task_id: str
    predictions: List[Grid]
    score: float
    success: bool
    strategy_used: SolvingStrategy
    elapsed_seconds: float
    attempts: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompetitionResult:
    """Result from solving all competition tasks."""
    total_tasks: int
    solved_tasks: int
    accuracy: float
    total_time_seconds: float
    per_task_results: List[SolveResult]
    strategy_breakdown: Dict[str, int]
    metadata: Dict[str, Any] = field(default_factory=dict)


class ARCSolver:
    """
    Main ARC-AGI-2 solver.

    Orchestrates multiple solving strategies and combines results
    using ensembling and voting.
    """

    def __init__(
        self,
        llm_client,
        config: Optional[SolverConfig] = None,
    ):
        self.client = llm_client
        self.config = config or SolverConfig()

        # Initialize components
        self.evaluator = ARCEvaluator(allow_partial_credit=True)
        self.dsl = create_standard_dsl()

        # Strategy handlers
        self._strategy_handlers: Dict[SolvingStrategy, Callable] = {
            SolvingStrategy.REFINEMENT: self._solve_refinement,
            SolvingStrategy.DSL_SEARCH: self._solve_dsl,
            SolvingStrategy.TTT: self._solve_ttt,
            SolvingStrategy.DIRECT: self._solve_direct,
        }

        # Statistics
        self._stats = {
            "tasks_solved": 0,
            "tasks_attempted": 0,
            "strategy_wins": {s.name: 0 for s in SolvingStrategy},
        }

    async def solve_task(
        self,
        task: ARCTask,
        time_budget: Optional[float] = None,
    ) -> SolveResult:
        """
        Solve a single ARC task.

        Tries multiple strategies and returns best result.
        """
        start_time = time.time()
        time_budget = time_budget or self.config.time_per_task_seconds

        logger.info(f"Solving task {task.id} with {time_budget:.0f}s budget")

        best_result: Optional[SolveResult] = None
        strategy_results: List[SolveResult] = []

        # Allocate time to strategies
        time_per_strategy = time_budget / len(self.config.strategies)

        for strategy in self.config.strategies:
            remaining_time = time_budget - (time.time() - start_time)
            if remaining_time < self.config.min_time_per_task / 2:
                break

            strategy_time = min(time_per_strategy, remaining_time)

            try:
                result = await self._run_strategy(
                    strategy, task, strategy_time
                )
                strategy_results.append(result)

                # Update best
                if best_result is None or result.score > best_result.score:
                    best_result = result

                # Early exit on success
                if result.score >= self.config.early_stop_threshold:
                    logger.info(f"Task {task.id}: Solved with {strategy.name}")
                    break

            except Exception as e:
                logger.error(f"Strategy {strategy.name} failed on {task.id}: {e}")

        # Ensemble if multiple results
        if self.config.ensemble_strategies and len(strategy_results) > 1:
            ensemble_result = self._ensemble_results(task, strategy_results)
            if ensemble_result.score > (best_result.score if best_result else 0):
                best_result = ensemble_result

        elapsed = time.time() - start_time

        if best_result:
            best_result.elapsed_seconds = elapsed
            return best_result

        # No result - return empty
        return SolveResult(
            task_id=task.id,
            predictions=[Grid.empty(1, 1) for _ in task.test],
            score=0.0,
            success=False,
            strategy_used=SolvingStrategy.REFINEMENT,
            elapsed_seconds=elapsed,
            attempts=len(strategy_results),
            metadata={"error": "All strategies failed"}
        )

    async def solve_all(
        self,
        tasks: List[ARCTask],
        time_budget_hours: Optional[float] = None,
    ) -> CompetitionResult:
        """
        Solve all competition tasks within time budget.

        Implements dynamic time allocation based on remaining tasks.
        """
        start_time = time.time()
        total_budget = (time_budget_hours or self.config.total_time_budget_hours) * 3600

        results: List[SolveResult] = []
        strategy_breakdown: Dict[str, int] = {s.name: 0 for s in SolvingStrategy}

        for i, task in enumerate(tasks):
            # Calculate remaining time and tasks
            elapsed = time.time() - start_time
            remaining_time = total_budget - elapsed
            remaining_tasks = len(tasks) - i

            if remaining_time <= 0:
                logger.warning("Time budget exhausted")
                break

            # Dynamic time allocation
            base_time = remaining_time / remaining_tasks
            task_time = np.clip(
                base_time,
                self.config.min_time_per_task,
                self.config.max_time_per_task
            )

            # Solve task
            result = await self.solve_task(task, task_time)
            results.append(result)

            # Track statistics
            if result.success:
                strategy_breakdown[result.strategy_used.name] += 1

            # Progress logging
            if (i + 1) % 10 == 0:
                solved = sum(1 for r in results if r.success)
                logger.info(
                    f"Progress: {i+1}/{len(tasks)} tasks, "
                    f"{solved} solved ({solved/(i+1)*100:.1f}%)"
                )

        # Compute final statistics
        total_time = time.time() - start_time
        solved = sum(1 for r in results if r.success)
        accuracy = solved / len(tasks) if tasks else 0.0

        return CompetitionResult(
            total_tasks=len(tasks),
            solved_tasks=solved,
            accuracy=accuracy,
            total_time_seconds=total_time,
            per_task_results=results,
            strategy_breakdown=strategy_breakdown,
            metadata={
                "avg_time_per_task": total_time / len(tasks) if tasks else 0,
                "tasks_processed": len(results),
            }
        )

    async def _run_strategy(
        self,
        strategy: SolvingStrategy,
        task: ARCTask,
        time_budget: float,
    ) -> SolveResult:
        """Run a specific solving strategy."""
        handler = self._strategy_handlers.get(strategy)
        if not handler:
            raise ValueError(f"Unknown strategy: {strategy}")

        return await handler(task, time_budget)

    async def _solve_refinement(
        self,
        task: ARCTask,
        time_budget: float,
    ) -> SolveResult:
        """Solve using LLM refinement loop."""
        refinement_loop = ARCRefinementLoop(
            client=self.client,
            model=self.config.refinement_model,
            max_iterations=self.config.refinement_max_iterations,
            max_time_seconds=time_budget,
            temperature=self.config.refinement_temperature,
        )

        result = await refinement_loop.solve(task)

        return SolveResult(
            task_id=task.id,
            predictions=result.best_predictions,
            score=result.best_score,
            success=result.success,
            strategy_used=SolvingStrategy.REFINEMENT,
            elapsed_seconds=result.total_elapsed_seconds,
            attempts=result.total_iterations,
            metadata={"refinement_result": True}
        )

    async def _solve_dsl(
        self,
        task: ARCTask,
        time_budget: float,
    ) -> SolveResult:
        """Solve using DSL program search."""
        start_time = time.time()

        # Prepare examples
        examples = [
            (pair.input.data, pair.output.data)
            for pair in task.train
        ]

        # Run hybrid search
        searcher = HybridSearch(
            self.dsl,
            searchers=[
                BeamSearch(self.dsl, beam_width=self.config.dsl_beam_width),
                GeneticSearch(self.dsl, population_size=50),
            ]
        )

        search_result = searcher.search(
            examples,
            max_time_seconds=min(time_budget, self.config.dsl_max_time),
            max_programs=5000,
        )

        # Generate predictions
        predictions = []
        if search_result.program:
            for test_pair in task.test:
                try:
                    output = self.dsl.execute(search_result.program, test_pair.input.data)
                    predictions.append(Grid(output))
                except Exception:
                    predictions.append(Grid.empty(1, 1))
        else:
            predictions = [Grid.empty(1, 1) for _ in task.test]

        # Evaluate
        if predictions and task.test[0].output:
            eval_result = self.evaluator.evaluate(task, predictions)
            score = eval_result.score
            success = eval_result.correct
        else:
            score = search_result.score
            success = score == 1.0

        return SolveResult(
            task_id=task.id,
            predictions=predictions,
            score=score,
            success=success,
            strategy_used=SolvingStrategy.DSL_SEARCH,
            elapsed_seconds=time.time() - start_time,
            attempts=search_result.programs_evaluated,
            metadata={
                "program": str(search_result.program) if search_result.program else None,
                "programs_evaluated": search_result.programs_evaluated,
            }
        )

    async def _solve_ttt(
        self,
        task: ARCTask,
        time_budget: float,
    ) -> SolveResult:
        """Solve using test-time training."""
        start_time = time.time()

        trainer = TestTimeTrainer(TTTConfig(
            num_epochs=3,
            augmentation_multiplier=8,
        ))

        # Train adapter
        train_result = trainer.train(task, max_time_seconds=time_budget * 0.7)

        # Use adapter for inference (placeholder)
        # In production, would run adapted model on test inputs
        predictions = [Grid.empty(1, 1) for _ in task.test]

        return SolveResult(
            task_id=task.id,
            predictions=predictions,
            score=train_result.validation_accuracy,
            success=train_result.validation_accuracy >= 0.95,
            strategy_used=SolvingStrategy.TTT,
            elapsed_seconds=time.time() - start_time,
            attempts=train_result.epochs_trained,
            metadata={
                "training_loss": train_result.final_loss,
                "validation_accuracy": train_result.validation_accuracy,
            }
        )

    async def _solve_direct(
        self,
        task: ARCTask,
        time_budget: float,
    ) -> SolveResult:
        """Solve with direct LLM prediction."""
        start_time = time.time()

        # Simple prompt for direct prediction
        prompt = f"""Solve this ARC task by predicting the output for the test input.

{task.format_for_prompt()}

Provide the output as a JSON 2D array:
```json
[[...], [...], ...]
```
"""

        try:
            response = self.client.chat.completions.create(
                model=self.config.refinement_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=2048,
            )
            content = response.choices[0].message.content or ""

            # Parse predictions
            from .evaluator import parse_grid_from_response
            predictions = []
            for _ in task.test:
                grid = parse_grid_from_response(content)
                if grid:
                    predictions.append(grid)
                else:
                    predictions.append(Grid.empty(1, 1))

            # Evaluate
            if task.test[0].output:
                eval_result = self.evaluator.evaluate(task, predictions)
                score = eval_result.score
                success = eval_result.correct
            else:
                score = 0.0
                success = False

        except Exception as e:
            logger.error(f"Direct prediction failed: {e}")
            predictions = [Grid.empty(1, 1) for _ in task.test]
            score = 0.0
            success = False

        return SolveResult(
            task_id=task.id,
            predictions=predictions,
            score=score,
            success=success,
            strategy_used=SolvingStrategy.DIRECT,
            elapsed_seconds=time.time() - start_time,
            attempts=1,
        )

    def _ensemble_results(
        self,
        task: ARCTask,
        results: List[SolveResult],
    ) -> SolveResult:
        """Combine results from multiple strategies using voting."""
        if not results:
            return SolveResult(
                task_id=task.id,
                predictions=[],
                score=0.0,
                success=False,
                strategy_used=SolvingStrategy.ENSEMBLE,
                elapsed_seconds=0.0,
                attempts=0,
            )

        # Group predictions by hash for voting
        from collections import Counter

        num_tests = len(task.test)
        final_predictions = []

        for test_idx in range(num_tests):
            # Collect predictions for this test
            predictions_for_test = []
            for result in results:
                if test_idx < len(result.predictions):
                    predictions_for_test.append((
                        result.score,
                        result.predictions[test_idx]
                    ))

            if not predictions_for_test:
                final_predictions.append(Grid.empty(1, 1))
                continue

            # Vote by hash, weighted by score
            hash_votes: Dict[str, float] = {}
            hash_to_grid: Dict[str, Grid] = {}

            for score, pred in predictions_for_test:
                h = pred.hash()
                hash_votes[h] = hash_votes.get(h, 0) + score
                hash_to_grid[h] = pred

            # Select winner
            winner_hash = max(hash_votes.keys(), key=lambda h: hash_votes[h])
            final_predictions.append(hash_to_grid[winner_hash])

        # Evaluate ensemble
        if task.test[0].output:
            eval_result = self.evaluator.evaluate(task, final_predictions)
            score = eval_result.score
            success = eval_result.correct
        else:
            # Use best individual score
            score = max(r.score for r in results)
            success = any(r.success for r in results)

        return SolveResult(
            task_id=task.id,
            predictions=final_predictions,
            score=score,
            success=success,
            strategy_used=SolvingStrategy.ENSEMBLE,
            elapsed_seconds=sum(r.elapsed_seconds for r in results),
            attempts=sum(r.attempts for r in results),
            metadata={
                "num_strategies": len(results),
                "strategy_scores": {r.strategy_used.name: r.score for r in results},
            }
        )


class ParallelSolver:
    """
    Solve tasks in parallel for efficiency.

    Uses thread/process pool to solve multiple tasks concurrently.
    """

    def __init__(
        self,
        llm_client,
        config: Optional[SolverConfig] = None,
        num_workers: int = 4,
    ):
        self.client = llm_client
        self.config = config or SolverConfig()
        self.num_workers = num_workers

    async def solve_parallel(
        self,
        tasks: List[ARCTask],
        time_budget_hours: float = 12.0,
    ) -> CompetitionResult:
        """Solve tasks in parallel batches."""
        start_time = time.time()
        total_budget = time_budget_hours * 3600

        results: List[SolveResult] = []
        solver = ARCSolver(self.client, self.config)

        # Process in batches
        batch_size = self.num_workers * 2

        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]

            # Calculate time for this batch
            elapsed = time.time() - start_time
            remaining = total_budget - elapsed
            remaining_tasks = len(tasks) - i

            if remaining <= 0:
                break

            time_per_task = min(
                remaining / remaining_tasks,
                self.config.max_time_per_task
            )

            # Solve batch concurrently
            batch_results = await asyncio.gather(*[
                solver.solve_task(task, time_per_task)
                for task in batch
            ])

            results.extend(batch_results)

            # Progress
            solved = sum(1 for r in results if r.success)
            logger.info(
                f"Batch {i//batch_size + 1}: "
                f"{len(results)}/{len(tasks)} processed, "
                f"{solved} solved"
            )

        # Compile results
        total_time = time.time() - start_time
        solved = sum(1 for r in results if r.success)

        strategy_breakdown = {}
        for r in results:
            name = r.strategy_used.name
            strategy_breakdown[name] = strategy_breakdown.get(name, 0) + (1 if r.success else 0)

        return CompetitionResult(
            total_tasks=len(tasks),
            solved_tasks=solved,
            accuracy=solved / len(tasks) if tasks else 0,
            total_time_seconds=total_time,
            per_task_results=results,
            strategy_breakdown=strategy_breakdown,
        )
