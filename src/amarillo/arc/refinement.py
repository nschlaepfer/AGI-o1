"""
ARC Refinement Loop - Poetiq-style iterative solving.

Implements the core refinement loop that drives state-of-the-art ARC solving:
1. Generate hypothesis (transformation code or direct output)
2. Validate against training examples
3. Analyze errors and generate feedback
4. Refine hypothesis using accumulated feedback
5. Repeat until success or budget exhausted
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

from .grid import Grid
from .dataset import ARCTask
from .evaluator import ARCEvaluator, CodeExecutionEvaluator, EvaluationResult, parse_grid_from_response

logger = logging.getLogger(__name__)


@dataclass
class SolutionAttempt:
    """A single attempt at solving an ARC task."""
    iteration: int
    approach: str  # "code" or "direct"
    content: str  # The code or grid prediction
    training_score: float
    test_score: Optional[float]
    feedback: str
    predictions: List[Grid]
    elapsed_seconds: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_correct(self) -> bool:
        return self.training_score == 1.0


@dataclass
class RefinementResult:
    """Result from a refinement loop session."""
    task_id: str
    success: bool
    best_score: float
    best_predictions: List[Grid]
    all_attempts: List[SolutionAttempt]
    total_iterations: int
    total_elapsed_seconds: float
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Prompt Templates
# ============================================================================

ANALYSIS_PROMPT = """You are an expert at solving ARC-AGI tasks. Your goal is to identify the transformation rule that converts input grids to output grids.

## Task
{task_json}

## Instructions
1. Carefully examine each training example
2. Identify patterns in how inputs relate to outputs
3. Consider: colors, shapes, positions, symmetries, counting, copying, rotation, scaling
4. Formulate a precise rule that explains ALL training examples

## Analysis Format
<analysis>
Describe the transformation pattern you observe.
Be specific about:
- What objects/patterns exist in the input
- How they change in the output
- Any conditions or special cases
</analysis>

<rule>
State the transformation rule concisely.
</rule>
"""

CODE_GENERATION_PROMPT = """You are an expert Python programmer solving an ARC-AGI task.

## Task
{task_json}

## Your Analysis
{analysis}

## Previous Attempts and Feedback
{feedback_block}

## Instructions
Write a Python function that implements the transformation. The function should:
1. Take a numpy array (2D, values 0-9) as input
2. Return a numpy array as output
3. Work correctly on ALL training examples

```python
import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    # Your implementation here
    pass
```

Important:
- Use numpy operations for efficiency
- Handle edge cases
- The grid values are integers 0-9 representing colors
- Return a new array, don't modify the input

Provide ONLY the Python code block, nothing else.
"""

DIRECT_PREDICTION_PROMPT = """You are an expert at solving ARC-AGI tasks.

## Task
{task_json}

## Your Analysis
{analysis}

## Previous Attempts and Feedback
{feedback_block}

## Instructions
Based on the transformation rule, predict the output for the test input(s).

For each test input, provide the output as a JSON 2D array:
```json
[[row1], [row2], ...]
```

Provide your predictions in order, one per test case.
"""

REFINEMENT_PROMPT = """You are refining a solution for an ARC-AGI task.

## Task
{task_json}

## Previous Attempt
```python
{previous_code}
```

## Evaluation Feedback
{evaluation_feedback}

## Specific Errors
{error_analysis}

## Instructions
Fix the code based on the feedback. Focus on:
1. Correcting the specific errors identified
2. Ensuring the logic handles all training examples
3. Not changing parts that already work correctly

Provide the complete fixed Python code:
```python
import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    # Fixed implementation
    pass
```
"""


class ARCRefinementLoop:
    """
    Main refinement loop for solving ARC tasks.

    Combines:
    - Multi-strategy generation (code vs direct)
    - Iterative refinement with feedback
    - Self-auditing (knows when to stop)
    - Ensemble over multiple attempts
    """

    def __init__(
        self,
        client: OpenAI,
        model: str = "gpt-4o",
        max_iterations: int = 10,
        max_time_seconds: float = 300.0,
        temperature: float = 0.7,
        use_code_approach: bool = True,
        use_direct_approach: bool = True,
        parallel_attempts: int = 1,
        reasoning_effort: Optional[str] = None,  # "low", "medium", "high"
    ):
        self.client = client
        self.model = model
        self.max_iterations = max_iterations
        self.max_time_seconds = max_time_seconds
        self.temperature = temperature
        self.use_code_approach = use_code_approach
        self.use_direct_approach = use_direct_approach
        self.parallel_attempts = parallel_attempts
        self.reasoning_effort = reasoning_effort

        self.arc_evaluator = ARCEvaluator(allow_partial_credit=True)
        self.code_evaluator = CodeExecutionEvaluator()

    async def solve(self, task: ARCTask) -> RefinementResult:
        """
        Solve an ARC task using the refinement loop.

        Returns the best solution found within constraints.
        """
        start_time = time.time()
        attempts: List[SolutionAttempt] = []
        best_score = -1.0
        best_predictions: List[Grid] = []

        # Phase 1: Initial analysis
        analysis = await self._analyze_task(task)
        logger.info(f"Task {task.id}: Analysis complete")

        # Phase 2: Iterative refinement
        previous_code: Optional[str] = None
        previous_feedback: str = ""

        for iteration in range(self.max_iterations):
            # Check time budget
            elapsed = time.time() - start_time
            if elapsed > self.max_time_seconds:
                logger.info(f"Task {task.id}: Time budget exhausted at iteration {iteration}")
                break

            remaining_time = self.max_time_seconds - elapsed

            # Build feedback block from previous attempts
            feedback_block = self._build_feedback_block(attempts[-3:] if attempts else [])

            # Try code approach
            if self.use_code_approach:
                attempt = await self._try_code_approach(
                    task, analysis, feedback_block, previous_code, iteration
                )
                attempts.append(attempt)

                if attempt.training_score == 1.0:
                    # Success on training - use for test
                    logger.info(f"Task {task.id}: Code approach succeeded at iteration {iteration}")
                    return RefinementResult(
                        task_id=task.id,
                        success=True,
                        best_score=1.0,
                        best_predictions=attempt.predictions,
                        all_attempts=attempts,
                        total_iterations=iteration + 1,
                        total_elapsed_seconds=time.time() - start_time,
                    )

                if attempt.training_score > best_score:
                    best_score = attempt.training_score
                    best_predictions = attempt.predictions
                    previous_code = attempt.content
                    previous_feedback = attempt.feedback

            # Try direct approach (especially useful for simple tasks)
            if self.use_direct_approach and iteration < 3:
                attempt = await self._try_direct_approach(
                    task, analysis, feedback_block, iteration
                )
                attempts.append(attempt)

                if attempt.training_score == 1.0:
                    logger.info(f"Task {task.id}: Direct approach succeeded at iteration {iteration}")
                    return RefinementResult(
                        task_id=task.id,
                        success=True,
                        best_score=1.0,
                        best_predictions=attempt.predictions,
                        all_attempts=attempts,
                        total_iterations=iteration + 1,
                        total_elapsed_seconds=time.time() - start_time,
                    )

                if attempt.training_score > best_score:
                    best_score = attempt.training_score
                    best_predictions = attempt.predictions

            # Self-audit: should we continue?
            if best_score >= 0.95 and iteration >= 3:
                # Close enough, might be evaluation noise
                logger.info(f"Task {task.id}: High score plateau, stopping early")
                break

            if iteration >= 5 and best_score < 0.3:
                # Not making progress, fundamental misunderstanding
                logger.info(f"Task {task.id}: Low score after 5 iterations, may need different approach")
                # Could switch strategies here

        return RefinementResult(
            task_id=task.id,
            success=best_score == 1.0,
            best_score=best_score,
            best_predictions=best_predictions,
            all_attempts=attempts,
            total_iterations=len(attempts),
            total_elapsed_seconds=time.time() - start_time,
        )

    async def _analyze_task(self, task: ARCTask) -> str:
        """Generate initial analysis of the task."""
        prompt = ANALYSIS_PROMPT.format(
            task_json=task.format_as_json_prompt()
        )

        response = await self._call_model(prompt)
        return response

    async def _try_code_approach(
        self,
        task: ARCTask,
        analysis: str,
        feedback_block: str,
        previous_code: Optional[str],
        iteration: int
    ) -> SolutionAttempt:
        """Generate and evaluate code-based solution."""
        start_time = time.time()

        if previous_code and iteration > 0:
            # Refinement mode
            prompt = REFINEMENT_PROMPT.format(
                task_json=task.format_as_json_prompt(),
                previous_code=previous_code,
                evaluation_feedback=feedback_block,
                error_analysis=self._extract_error_patterns(feedback_block)
            )
        else:
            # Initial generation
            prompt = CODE_GENERATION_PROMPT.format(
                task_json=task.format_as_json_prompt(),
                analysis=analysis,
                feedback_block=feedback_block if feedback_block else "No previous attempts."
            )

        response = await self._call_model(prompt)
        code = self._extract_code(response)

        if not code:
            return SolutionAttempt(
                iteration=iteration,
                approach="code",
                content=response,
                training_score=0.0,
                test_score=None,
                feedback="Failed to extract valid Python code from response",
                predictions=[],
                elapsed_seconds=time.time() - start_time,
            )

        # Evaluate
        eval_result = self.code_evaluator.evaluate_code(code, task)

        return SolutionAttempt(
            iteration=iteration,
            approach="code",
            content=code,
            training_score=eval_result.score,
            test_score=None,
            feedback=eval_result.feedback,
            predictions=eval_result.predictions,
            elapsed_seconds=time.time() - start_time,
        )

    async def _try_direct_approach(
        self,
        task: ARCTask,
        analysis: str,
        feedback_block: str,
        iteration: int
    ) -> SolutionAttempt:
        """Generate direct grid predictions."""
        start_time = time.time()

        prompt = DIRECT_PREDICTION_PROMPT.format(
            task_json=task.format_as_json_prompt(),
            analysis=analysis,
            feedback_block=feedback_block if feedback_block else "No previous attempts."
        )

        response = await self._call_model(prompt)

        # Parse predictions
        predictions = self._parse_multiple_grids(response, len(task.test))

        if not predictions:
            return SolutionAttempt(
                iteration=iteration,
                approach="direct",
                content=response,
                training_score=0.0,
                test_score=None,
                feedback="Failed to parse grid predictions from response",
                predictions=[],
                elapsed_seconds=time.time() - start_time,
            )

        # Evaluate (but we can only score against test if we have ground truth)
        # For training validation, we'd need the model to predict training outputs too
        # For now, we just return the predictions

        eval_result = self.arc_evaluator.evaluate(task, predictions)

        return SolutionAttempt(
            iteration=iteration,
            approach="direct",
            content=response,
            training_score=eval_result.score,
            test_score=None,
            feedback=eval_result.feedback,
            predictions=predictions,
            elapsed_seconds=time.time() - start_time,
        )

    def _build_feedback_block(self, attempts: List[SolutionAttempt]) -> str:
        """Build feedback block from previous attempts."""
        if not attempts:
            return ""

        blocks = []
        for i, attempt in enumerate(attempts, 1):
            block = f"""
<attempt_{i}>
<approach>{attempt.approach}</approach>
<score>{attempt.training_score:.2f}</score>
<feedback>
{attempt.feedback}
</feedback>
</attempt_{i}>
"""
            blocks.append(block.strip())

        return "\n\n".join(blocks)

    def _extract_code(self, response: str) -> Optional[str]:
        """Extract Python code from response."""
        import re

        # Try code block
        match = re.search(r"```python\s*(.*?)```", response, re.DOTALL | re.IGNORECASE)
        if match:
            code = match.group(1).strip()
            # Validate it has transform function
            if "def transform" in code or "def solve" in code:
                return code

        # Try any code block
        match = re.search(r"```\s*(.*?)```", response, re.DOTALL)
        if match:
            code = match.group(1).strip()
            if "def " in code:
                return code

        # Try to find function definition directly
        match = re.search(r"(def\s+\w+\s*\(.*?\).*?)(?=\n\n|\Z)", response, re.DOTALL)
        if match:
            return match.group(1).strip()

        return None

    def _parse_multiple_grids(self, response: str, expected_count: int) -> List[Grid]:
        """Parse multiple grid predictions from response."""
        import re

        grids = []

        # Find all JSON arrays
        json_blocks = re.findall(r"\[\[.*?\]\]", response, re.DOTALL)

        for block in json_blocks:
            try:
                data = json.loads(block)
                if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
                    grids.append(Grid(data))
                    if len(grids) >= expected_count:
                        break
            except json.JSONDecodeError:
                continue

        # Pad with empty grids if needed
        while len(grids) < expected_count:
            grids.append(Grid.empty(1, 1))

        return grids

    def _extract_error_patterns(self, feedback: str) -> str:
        """Extract specific error patterns from feedback for targeted fixes."""
        patterns = []

        # Shape errors
        if "Shape mismatch" in feedback:
            patterns.append("- Output shape is incorrect. Check grid dimensions.")

        # Color errors
        if "Missing colors" in feedback:
            patterns.append("- Some colors are missing. Review color transformation logic.")
        if "Extra colors" in feedback:
            patterns.append("- Unexpected colors in output. Check for color leakage.")

        # Pixel errors
        if "Pixel accuracy" in feedback:
            import re
            match = re.search(r"Pixel accuracy: ([\d.]+)%", feedback)
            if match:
                accuracy = float(match.group(1))
                if accuracy < 50:
                    patterns.append("- Major pixel errors. Fundamental logic may be wrong.")
                elif accuracy < 90:
                    patterns.append("- Moderate pixel errors. Check edge cases and boundaries.")
                else:
                    patterns.append("- Minor pixel errors. Fine-tune positioning or conditions.")

        if not patterns:
            patterns.append("- Review the transformation logic step by step.")

        return "\n".join(patterns)

    async def _call_model(self, prompt: str) -> str:
        """Call the language model."""
        try:
            # Try reasoning API first (for o1/o3 style models)
            if self.reasoning_effort and hasattr(self.client, 'responses'):
                response = self.client.responses.create(
                    model=self.model,
                    input=prompt,
                    reasoning={"effort": self.reasoning_effort}
                )
                return response.output_text
        except Exception:
            pass

        # Fall back to chat completions
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=4096,
        )
        return response.choices[0].message.content or ""


class ParallelRefinementLoop:
    """
    Run multiple refinement loops in parallel with different strategies.

    Combines results using voting and score-based selection.
    """

    def __init__(
        self,
        client: OpenAI,
        num_workers: int = 4,
        **refinement_kwargs
    ):
        self.client = client
        self.num_workers = num_workers
        self.refinement_kwargs = refinement_kwargs

    async def solve(self, task: ARCTask) -> RefinementResult:
        """Run parallel solvers and combine results."""
        # Create varied solver configurations
        configs = [
            {"temperature": 0.3, "use_direct_approach": False},
            {"temperature": 0.7, "use_direct_approach": True},
            {"temperature": 0.9, "use_code_approach": True},
            {"temperature": 0.5, "max_iterations": 15},
        ][:self.num_workers]

        # Run in parallel
        tasks = []
        for config in configs:
            kwargs = {**self.refinement_kwargs, **config}
            loop = ARCRefinementLoop(self.client, **kwargs)
            tasks.append(loop.solve(task))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions
        valid_results = [r for r in results if isinstance(r, RefinementResult)]

        if not valid_results:
            # All failed
            return RefinementResult(
                task_id=task.id,
                success=False,
                best_score=0.0,
                best_predictions=[],
                all_attempts=[],
                total_iterations=0,
                total_elapsed_seconds=0.0,
                metadata={"error": "All parallel solvers failed"}
            )

        # Find best result
        best = max(valid_results, key=lambda r: r.best_score)

        # If we have ties, use voting
        if best.best_score < 1.0:
            best = self._select_by_voting(valid_results)

        # Combine attempts from all solvers
        all_attempts = []
        for r in valid_results:
            all_attempts.extend(r.all_attempts)

        best.all_attempts = all_attempts
        best.metadata["num_parallel_solvers"] = len(valid_results)

        return best

    def _select_by_voting(self, results: List[RefinementResult]) -> RefinementResult:
        """Select best result using voting on predictions."""
        # Group by prediction hash
        from collections import Counter

        prediction_hashes = []
        for r in results:
            if r.best_predictions:
                h = tuple(p.hash() for p in r.best_predictions)
                prediction_hashes.append((h, r))

        if not prediction_hashes:
            return max(results, key=lambda r: r.best_score)

        # Count votes
        hash_counts = Counter(h for h, _ in prediction_hashes)
        most_common = hash_counts.most_common(1)[0][0]

        # Return result with most voted prediction
        for h, r in prediction_hashes:
            if h == most_common:
                return r

        return results[0]
