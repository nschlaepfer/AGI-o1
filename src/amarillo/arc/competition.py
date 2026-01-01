"""
Competition Runner for ARC-AGI-2.

Handles the full competition pipeline:
1. Load competition data
2. Manage compute/time budget
3. Run solver on all tasks
4. Format submission
5. Track and report progress
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .grid import Grid
from .dataset import ARCDataset, ARCTask, load_kaggle_data
from .solver import ARCSolver, SolverConfig, CompetitionResult, SolveResult

logger = logging.getLogger(__name__)


@dataclass
class CompetitionConfig:
    """Configuration for competition run."""
    # Time budget
    total_time_hours: float = 12.0
    reserve_time_minutes: float = 10.0  # Buffer for submission

    # Compute budget
    max_cost_usd: float = 50.0
    estimated_cost_per_task: float = 0.20

    # Paths
    challenges_path: Optional[Path] = None
    output_path: Optional[Path] = None

    # Strategy weights (for adaptive selection)
    strategy_weights: Dict[str, float] = field(default_factory=lambda: {
        "REFINEMENT": 1.0,
        "DSL_SEARCH": 0.8,
        "TTT": 0.5,
        "DIRECT": 0.3,
    })

    # Progress tracking
    checkpoint_interval: int = 10  # Checkpoint every N tasks
    verbose: bool = True


@dataclass
class SubmissionEntry:
    """A single submission entry for Kaggle."""
    task_id: str
    output_grid: List[List[int]]
    attempt: int = 1


class CompetitionRunner:
    """
    Main competition runner.

    Handles:
    - Loading competition data
    - Time/budget management
    - Running solver
    - Formatting and saving submission
    - Progress tracking and checkpointing
    """

    def __init__(
        self,
        llm_client,
        config: Optional[CompetitionConfig] = None,
        solver_config: Optional[SolverConfig] = None,
    ):
        self.client = llm_client
        self.config = config or CompetitionConfig()
        self.solver_config = solver_config or SolverConfig()

        # Adjust solver config based on competition constraints
        self._adjust_solver_config()

        self.solver = ARCSolver(llm_client, self.solver_config)

        # State
        self._start_time: Optional[float] = None
        self._results: List[SolveResult] = []
        self._checkpoints: List[Dict] = []

    def _adjust_solver_config(self) -> None:
        """Adjust solver config for competition constraints."""
        # Calculate time per task
        usable_time = (
            self.config.total_time_hours * 3600 -
            self.config.reserve_time_minutes * 60
        )
        # Assume ~240 tasks
        estimated_tasks = 240
        time_per_task = usable_time / estimated_tasks

        self.solver_config.time_per_task_seconds = time_per_task
        self.solver_config.total_time_budget_hours = self.config.total_time_hours

        logger.info(f"Adjusted time per task: {time_per_task:.1f}s")

    async def run(
        self,
        tasks: Optional[List[ARCTask]] = None,
        challenges_path: Optional[Path] = None,
    ) -> CompetitionResult:
        """
        Run the full competition.

        Args:
            tasks: Optional list of tasks (if not loading from file)
            challenges_path: Path to challenges JSON

        Returns:
            CompetitionResult with all results
        """
        self._start_time = time.time()

        # Load tasks
        if tasks is None:
            tasks = self._load_tasks(challenges_path)

        logger.info(f"Starting competition with {len(tasks)} tasks")
        logger.info(f"Time budget: {self.config.total_time_hours}h")

        # Run solver
        result = await self.solver.solve_all(
            tasks,
            time_budget_hours=self.config.total_time_hours
        )

        # Save results
        self._results = result.per_task_results

        # Generate submission
        if self.config.output_path:
            self._save_submission(result)

        # Final report
        self._print_summary(result)

        return result

    def _load_tasks(
        self,
        challenges_path: Optional[Path] = None
    ) -> List[ARCTask]:
        """Load competition tasks."""
        path = challenges_path or self.config.challenges_path

        if path is None:
            raise ValueError("No challenges path provided")

        logger.info(f"Loading tasks from {path}")

        if path.is_dir():
            dataset = ARCDataset.from_directory(path)
        else:
            dataset = ARCDataset.from_json_file(path)

        return list(dataset)

    def _save_submission(self, result: CompetitionResult) -> None:
        """Save submission in Kaggle format."""
        submission = {}

        for task_result in result.per_task_results:
            task_id = task_result.task_id
            predictions = task_result.predictions

            # Format: { "task_id": [ {"attempt_1": [[...]], "attempt_2": [[...]]} ] }
            task_submission = []

            for i, pred in enumerate(predictions):
                attempt_key = f"attempt_{i + 1}"
                task_submission.append({
                    attempt_key: pred.to_json()
                })

            submission[task_id] = task_submission

        # Save
        output_path = self.config.output_path or Path("submission.json")
        with open(output_path, "w") as f:
            json.dump(submission, f, indent=2)

        logger.info(f"Submission saved to {output_path}")

    def _print_summary(self, result: CompetitionResult) -> None:
        """Print competition summary."""
        print("\n" + "=" * 60)
        print("COMPETITION SUMMARY")
        print("=" * 60)
        print(f"Total tasks:     {result.total_tasks}")
        print(f"Solved tasks:    {result.solved_tasks}")
        print(f"Accuracy:        {result.accuracy:.1%}")
        print(f"Total time:      {result.total_time_seconds / 3600:.2f} hours")
        print(f"Avg time/task:   {result.total_time_seconds / result.total_tasks:.1f}s")
        print()
        print("Strategy breakdown:")
        for strategy, count in result.strategy_breakdown.items():
            print(f"  {strategy}: {count} solved")
        print("=" * 60)


class KaggleNotebookRunner:
    """
    Runner specifically designed for Kaggle notebook submission.

    Handles:
    - Kaggle-specific paths and formats
    - Offline execution constraints
    - GPU resource management
    - Progress tracking for 12-hour runs
    """

    def __init__(self):
        self._setup_paths()
        self._setup_logging()

    def _setup_paths(self) -> None:
        """Set up Kaggle-specific paths."""
        self.input_path = Path("/kaggle/input/arc-prize-2025")
        self.output_path = Path("/kaggle/working")

        # Check if running on Kaggle
        self.is_kaggle = self.input_path.exists()

        if not self.is_kaggle:
            logger.warning("Not running on Kaggle, using local paths")
            self.input_path = Path("data")
            self.output_path = Path("output")

    def _setup_logging(self) -> None:
        """Set up logging for long runs."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S"
        )

    def run(self) -> Dict[str, Any]:
        """
        Main entry point for Kaggle notebook.

        Returns submission dictionary.
        """
        start_time = time.time()

        print("=" * 60)
        print("ARC-AGI-2 Competition Solver")
        print("=" * 60)

        # Load challenges
        challenges_file = self.input_path / "arc-agi_test_challenges.json"
        if not challenges_file.exists():
            # Try alternative name
            challenges_file = self.input_path / "test_challenges.json"

        print(f"Loading challenges from: {challenges_file}")

        with open(challenges_file) as f:
            challenges = json.load(f)

        num_tasks = len(challenges)
        print(f"Loaded {num_tasks} tasks")

        # Calculate time budget
        total_time_hours = 11.5  # Leave 30 min buffer
        time_per_task = (total_time_hours * 3600) / num_tasks

        print(f"Time budget: {total_time_hours}h ({time_per_task:.1f}s per task)")

        # Initialize solver (placeholder - would use actual model)
        submission = {}

        for i, (task_id, task_data) in enumerate(challenges.items()):
            task_start = time.time()

            # Progress
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                remaining = (num_tasks - i - 1) * (elapsed / (i + 1))
                print(f"Progress: {i+1}/{num_tasks} ({elapsed/3600:.1f}h elapsed, {remaining/3600:.1f}h remaining)")

            # Parse task
            task = ARCTask.from_json(task_id, task_data)

            # Solve (placeholder)
            predictions = self._solve_task(task, time_per_task)

            # Format submission
            task_submission = []
            for j, pred in enumerate(predictions):
                task_submission.append({
                    f"attempt_{j+1}": pred
                })

            submission[task_id] = task_submission

            # Check time budget
            task_time = time.time() - task_start
            if task_time > time_per_task * 2:
                logger.warning(f"Task {task_id} took {task_time:.1f}s (budget: {time_per_task:.1f}s)")

        # Save submission
        submission_path = self.output_path / "submission.json"
        with open(submission_path, "w") as f:
            json.dump(submission, f)

        print(f"\nSubmission saved to: {submission_path}")

        # Final stats
        total_time = time.time() - start_time
        print(f"\nTotal time: {total_time/3600:.2f} hours")

        return submission

    def _solve_task(
        self,
        task: ARCTask,
        time_budget: float
    ) -> List[List[List[int]]]:
        """
        Solve a single task.

        Returns list of prediction grids (one per test input).
        """
        predictions = []

        for test_pair in task.test:
            # Placeholder: return input as-is
            # In production, would run actual solver
            pred = test_pair.input.to_json()
            predictions.append(pred)

        return predictions


def create_submission_template() -> str:
    """
    Create template code for Kaggle notebook submission.

    Returns Python code string.
    """
    return '''
"""
ARC-AGI-2 Competition Submission

This notebook implements a state-of-the-art ARC solver combining:
- Iterative refinement with LLM
- DSL-based program synthesis
- Test-time training
- Ensemble voting

Time budget: 12 hours on 4x L4 GPUs
Target: >25% on private evaluation set
"""

import os
import sys
import json
import time
from pathlib import Path

# Check environment
IS_KAGGLE = Path("/kaggle").exists()

if IS_KAGGLE:
    # Install dependencies
    os.system("pip install -q openai numpy scipy")

    # Set up paths
    INPUT_PATH = Path("/kaggle/input/arc-prize-2025")
    OUTPUT_PATH = Path("/kaggle/working")
else:
    INPUT_PATH = Path("data")
    OUTPUT_PATH = Path("output")
    OUTPUT_PATH.mkdir(exist_ok=True)

# Import solver (would be included in notebook)
# from amarillo.arc import ARCSolver, CompetitionRunner

def main():
    """Main entry point."""
    start_time = time.time()

    print("=" * 60)
    print("ARC-AGI-2 Solver")
    print("=" * 60)

    # Load challenges
    challenges_file = INPUT_PATH / "arc-agi_test_challenges.json"
    with open(challenges_file) as f:
        challenges = json.load(f)

    print(f"Loaded {len(challenges)} tasks")

    # Time management
    TOTAL_HOURS = 11.5  # Leave buffer
    time_per_task = (TOTAL_HOURS * 3600) / len(challenges)

    # Solve all tasks
    submission = {}

    for i, (task_id, task_data) in enumerate(challenges.items()):
        # Progress every 10 tasks
        if (i + 1) % 10 == 0:
            elapsed = (time.time() - start_time) / 3600
            print(f"Progress: {i+1}/{len(challenges)} ({elapsed:.1f}h)")

        # Solve task
        predictions = solve_task(task_data, time_per_task)
        submission[task_id] = predictions

    # Save submission
    with open(OUTPUT_PATH / "submission.json", "w") as f:
        json.dump(submission, f)

    print(f"\\nDone! Total time: {(time.time() - start_time) / 3600:.2f}h")

def solve_task(task_data, time_budget):
    """Solve a single task."""
    # Placeholder - implement actual solver
    predictions = []
    for test in task_data.get("test", []):
        # Return input as placeholder
        predictions.append({"attempt_1": test["input"]})
    return predictions

if __name__ == "__main__":
    main()
'''
