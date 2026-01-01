"""
GPT-5.2-Codex Optimized ARC Solver.

Leverages GPT-5.2-Codex's unique capabilities:
- 400K token context window for full solution history
- 128K max output for complex code generation
- Native context compaction for million-token sessions
- State-of-the-art agentic coding (56.4% SWE-Bench Pro)

This is the Poetiq-style refinement loop optimized for GPT-5.2-Codex.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from .grid import Grid
from .dataset import ARCTask
from .evaluator import ARCEvaluator, CodeExecutionEvaluator

logger = logging.getLogger(__name__)


# ============================================================================
# GPT-5.2-Codex Optimized Prompts
# ============================================================================

SYSTEM_PROMPT = """You are an expert ARC-AGI solver using GPT-5.2-Codex.

Your task is to analyze grid transformation puzzles and write Python code that correctly transforms inputs to outputs.

Key capabilities you should leverage:
1. Pattern Recognition: Identify geometric, color, and structural patterns
2. Code Generation: Write clean, efficient numpy-based transformations
3. Iterative Refinement: Learn from execution feedback to fix errors
4. Multi-step Reasoning: Break complex transforms into composable steps

Grid Format:
- 2D numpy arrays with integer values 0-9
- 0 = black (often background)
- 1-9 = colors (blue, red, green, yellow, grey, pink, orange, cyan, brown)

Your code must:
- Accept a numpy array as input
- Return a numpy array as output
- Handle edge cases gracefully
- Work on ALL training examples exactly
"""

ANALYSIS_PROMPT = """Analyze this ARC-AGI task and identify the transformation rule.

## Training Examples
{examples}

## Analysis Steps
1. Examine each input-output pair carefully
2. Identify what changes between input and output:
   - Shape changes (size, dimensions)
   - Color changes (which colors appear/disappear)
   - Structural changes (patterns, objects, positions)
3. Look for consistent rules across ALL examples
4. Consider: rotation, reflection, scaling, tiling, object extraction, color mapping

Provide your analysis in this format:
<observation>
What you see in the examples (be specific about grids, colors, positions)
</observation>

<pattern>
The consistent transformation pattern across all examples
</pattern>

<rule>
A precise, implementable rule for the transformation
</rule>
"""

CODE_GENERATION_PROMPT = """Write Python code to implement this ARC transformation.

## Task Analysis
{analysis}

## Training Examples
{examples}

## Previous Attempts
{attempts}

## Requirements
- Function signature: def transform(grid: np.ndarray) -> np.ndarray
- Use numpy operations for efficiency
- Handle the exact transformations shown in examples
- Return a new array (don't modify input)

Write the complete Python code:
```python
import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    # Your implementation
    pass
```
"""

REFINEMENT_PROMPT = """Fix the code based on execution feedback.

## Original Task
{examples}

## Your Previous Code
```python
{code}
```

## Execution Results
{results}

## Specific Errors
{errors}

## Instructions
1. Analyze each error carefully
2. Identify the root cause (not just symptoms)
3. Fix the logic while preserving what works
4. Test your mental model against ALL examples

Write the corrected code:
```python
import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    # Corrected implementation
    pass
```
"""


@dataclass
class CodexConfig:
    """Configuration for GPT-5.2-Codex solver."""

    # Model settings
    model: str = "gpt-5.2-codex"
    reasoning_model: str = "gpt-5.2"
    reasoning_effort: str = "high"

    # Generation settings
    temperature: float = 0.3  # Lower for more consistent code
    max_tokens: int = 8192

    # Refinement settings
    max_iterations: int = 15
    max_time_seconds: float = 300.0
    early_stop_threshold: float = 1.0

    # Context management
    use_compaction: bool = True
    max_context_tokens: int = 200000  # Leave headroom

    # Parallel settings
    parallel_attempts: int = 3

    # Cost tracking
    track_costs: bool = True


@dataclass
class SolutionAttempt:
    """A single solution attempt."""
    iteration: int
    code: str
    score: float
    feedback: str
    predictions: List[Grid]
    tokens_used: int
    elapsed_seconds: float


@dataclass
class CodexResult:
    """Result from Codex solver."""
    task_id: str
    success: bool
    score: float
    predictions: List[Grid]
    final_code: str
    attempts: List[SolutionAttempt]
    total_tokens: int
    total_cost: float
    elapsed_seconds: float


class GPT52CodexSolver:
    """
    ARC solver optimized for GPT-5.2-Codex.

    Key optimizations:
    1. Uses 400K context for full solution history
    2. Leverages reasoning API for complex analysis
    3. Uses Codex for code generation/refinement
    4. Implements native context compaction
    """

    def __init__(
        self,
        client: OpenAI,
        config: Optional[CodexConfig] = None,
    ):
        self.client = client
        self.config = config or CodexConfig()

        self.evaluator = ARCEvaluator(allow_partial_credit=True)
        self.code_evaluator = CodeExecutionEvaluator()

        # Token tracking
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

        # Session context (for compaction)
        self._session_context: List[Dict] = []

    async def solve(self, task: ARCTask) -> CodexResult:
        """
        Solve an ARC task using GPT-5.2-Codex refinement loop.
        """
        start_time = time.time()
        attempts: List[SolutionAttempt] = []

        # Phase 1: Deep analysis with reasoning model
        analysis = await self._analyze_task(task)
        logger.info(f"Task {task.id}: Analysis complete")

        # Phase 2: Iterative code refinement with Codex
        best_score = 0.0
        best_code = ""
        best_predictions: List[Grid] = []

        for iteration in range(self.config.max_iterations):
            elapsed = time.time() - start_time
            if elapsed > self.config.max_time_seconds:
                logger.info(f"Task {task.id}: Time budget exhausted at iteration {iteration}")
                break

            # Generate or refine code
            if iteration == 0:
                code = await self._generate_code(task, analysis, [])
            else:
                code = await self._refine_code(
                    task,
                    attempts[-1].code,
                    attempts[-1].feedback,
                    attempts[-3:] if len(attempts) >= 3 else attempts
                )

            if not code:
                logger.warning(f"Task {task.id}: Failed to generate code at iteration {iteration}")
                continue

            # Execute and evaluate
            eval_result = self.code_evaluator.evaluate_code(code, task)

            attempt = SolutionAttempt(
                iteration=iteration,
                code=code,
                score=eval_result.score,
                feedback=eval_result.feedback,
                predictions=eval_result.predictions,
                tokens_used=self._get_last_tokens(),
                elapsed_seconds=time.time() - start_time,
            )
            attempts.append(attempt)

            logger.info(f"Task {task.id}: Iteration {iteration}, score={eval_result.score:.2%}")

            # Track best
            if eval_result.score > best_score:
                best_score = eval_result.score
                best_code = code
                best_predictions = eval_result.predictions

            # Early exit on success
            if eval_result.score >= self.config.early_stop_threshold:
                logger.info(f"Task {task.id}: Solved at iteration {iteration}!")
                break

            # Apply compaction if context growing too large
            if self.config.use_compaction and self._should_compact():
                await self._compact_context()

        # Calculate cost
        total_cost = self._calculate_cost()

        return CodexResult(
            task_id=task.id,
            success=best_score >= self.config.early_stop_threshold,
            score=best_score,
            predictions=best_predictions,
            final_code=best_code,
            attempts=attempts,
            total_tokens=self.total_prompt_tokens + self.total_completion_tokens,
            total_cost=total_cost,
            elapsed_seconds=time.time() - start_time,
        )

    async def _analyze_task(self, task: ARCTask) -> str:
        """Deep analysis using GPT-5.2 reasoning model."""
        examples = self._format_examples(task)
        prompt = ANALYSIS_PROMPT.format(examples=examples)

        try:
            # Use reasoning API for deep analysis
            response = self.client.responses.create(
                model=self.config.reasoning_model,
                input=prompt,
                reasoning={"effort": self.config.reasoning_effort},
            )
            self._track_tokens(response)
            return response.output_text
        except Exception as e:
            # Fallback to chat completions
            logger.warning(f"Reasoning API failed, falling back: {e}")
            return await self._call_chat(prompt)

    async def _generate_code(
        self,
        task: ARCTask,
        analysis: str,
        previous_attempts: List[SolutionAttempt],
    ) -> Optional[str]:
        """Generate initial code with GPT-5.2-Codex."""
        examples = self._format_examples(task)
        attempts_str = self._format_attempts(previous_attempts)

        prompt = CODE_GENERATION_PROMPT.format(
            analysis=analysis,
            examples=examples,
            attempts=attempts_str or "No previous attempts.",
        )

        response = await self._call_codex(prompt)
        return self._extract_code(response)

    async def _refine_code(
        self,
        task: ARCTask,
        previous_code: str,
        feedback: str,
        previous_attempts: List[SolutionAttempt],
    ) -> Optional[str]:
        """Refine code based on feedback with GPT-5.2-Codex."""
        examples = self._format_examples(task)

        # Build detailed error analysis
        errors = self._analyze_errors(previous_attempts)
        results = self._format_execution_results(previous_attempts[-1] if previous_attempts else None)

        prompt = REFINEMENT_PROMPT.format(
            examples=examples,
            code=previous_code,
            results=results,
            errors=errors,
        )

        response = await self._call_codex(prompt)
        return self._extract_code(response)

    async def _call_codex(self, prompt: str) -> str:
        """Call GPT-5.2-Codex for code generation."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        # Add session context for continuity
        if self._session_context:
            messages = [messages[0]] + self._session_context + [messages[-1]]

        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        self._track_chat_tokens(response)

        result = response.choices[0].message.content or ""

        # Add to session context
        self._session_context.append({"role": "user", "content": prompt})
        self._session_context.append({"role": "assistant", "content": result})

        return result

    async def _call_chat(self, prompt: str) -> str:
        """Fallback chat completion."""
        response = self.client.chat.completions.create(
            model=self.config.reasoning_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=4096,
        )
        self._track_chat_tokens(response)
        return response.choices[0].message.content or ""

    async def _compact_context(self) -> None:
        """Use GPT-5.2's native context compaction."""
        if not self._session_context:
            return

        try:
            # Build conversation for compaction
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
            ] + self._session_context

            # Call compaction endpoint
            response = self.client.responses.compact(
                model=self.config.model,
                input=messages,
            )

            # Replace context with compacted version
            self._session_context = [
                {"role": "assistant", "content": f"[Compacted context: {response.summary}]"}
            ]

            logger.info(f"Context compacted: {response.tokens_before} -> {response.tokens_after}")

        except Exception as e:
            logger.warning(f"Compaction failed: {e}")
            # Fallback: keep recent context only
            self._session_context = self._session_context[-6:]

    def _should_compact(self) -> bool:
        """Check if context needs compaction."""
        # Estimate tokens in session context
        estimated_tokens = sum(
            len(msg.get("content", "")) // 4
            for msg in self._session_context
        )
        return estimated_tokens > self.config.max_context_tokens

    def _format_examples(self, task: ARCTask) -> str:
        """Format task examples for prompts."""
        examples = []
        for i, pair in enumerate(task.train):
            example = f"""Example {i + 1}:
Input:
{self._grid_to_str(pair.input.data)}

Output:
{self._grid_to_str(pair.output.data)}
"""
            examples.append(example)
        return "\n".join(examples)

    def _grid_to_str(self, grid) -> str:
        """Convert grid to string representation."""
        return "\n".join(" ".join(str(int(x)) for x in row) for row in grid)

    def _format_attempts(self, attempts: List[SolutionAttempt]) -> str:
        """Format previous attempts for context."""
        if not attempts:
            return ""

        parts = []
        for attempt in attempts:
            part = f"""Attempt {attempt.iteration + 1} (score: {attempt.score:.2%}):
```python
{attempt.code}
```
Feedback: {attempt.feedback}
"""
            parts.append(part)
        return "\n".join(parts)

    def _format_execution_results(self, attempt: Optional[SolutionAttempt]) -> str:
        """Format execution results for refinement."""
        if not attempt:
            return "No previous execution."
        return f"Score: {attempt.score:.2%}\n{attempt.feedback}"

    def _analyze_errors(self, attempts: List[SolutionAttempt]) -> str:
        """Analyze error patterns across attempts."""
        if not attempts:
            return "No errors to analyze."

        errors = []
        for attempt in attempts:
            if attempt.score < 1.0:
                errors.append(f"Iteration {attempt.iteration}: {attempt.feedback}")

        if not errors:
            return "Previous attempts had issues but no specific errors captured."

        return "\n".join(errors[-3:])  # Last 3 errors

    def _extract_code(self, response: str) -> Optional[str]:
        """Extract Python code from response."""
        import re

        # Try python code block
        match = re.search(r"```python\s*(.*?)```", response, re.DOTALL | re.IGNORECASE)
        if match:
            code = match.group(1).strip()
            if "def transform" in code:
                return code

        # Try any code block
        match = re.search(r"```\s*(.*?)```", response, re.DOTALL)
        if match:
            code = match.group(1).strip()
            if "def " in code:
                return code

        return None

    def _track_tokens(self, response) -> None:
        """Track tokens from responses API."""
        if hasattr(response, 'usage'):
            self.total_prompt_tokens += getattr(response.usage, 'input_tokens', 0)
            self.total_completion_tokens += getattr(response.usage, 'output_tokens', 0)

    def _track_chat_tokens(self, response) -> None:
        """Track tokens from chat completions."""
        if response.usage:
            self.total_prompt_tokens += response.usage.prompt_tokens
            self.total_completion_tokens += response.usage.completion_tokens

    def _get_last_tokens(self) -> int:
        """Get tokens used in last call."""
        return self.total_prompt_tokens + self.total_completion_tokens

    def _calculate_cost(self) -> float:
        """Calculate total cost in USD."""
        # GPT-5.2-Codex pricing: $1.75/M input, $14/M output
        input_cost = self.total_prompt_tokens * 1.75 / 1_000_000
        output_cost = self.total_completion_tokens * 14 / 1_000_000
        return input_cost + output_cost


class ParallelCodexSolver:
    """
    Run multiple Codex solvers in parallel with different strategies.

    Combines results via voting for robustness.
    """

    def __init__(
        self,
        client: OpenAI,
        num_workers: int = 3,
    ):
        self.client = client
        self.num_workers = num_workers

    async def solve(self, task: ARCTask) -> CodexResult:
        """Solve with parallel strategies and voting."""

        # Different configurations for diversity
        configs = [
            CodexConfig(temperature=0.2, max_iterations=15),
            CodexConfig(temperature=0.5, max_iterations=12),
            CodexConfig(temperature=0.7, max_iterations=10),
        ][:self.num_workers]

        # Run in parallel
        solvers = [GPT52CodexSolver(self.client, cfg) for cfg in configs]
        results = await asyncio.gather(*[
            solver.solve(task) for solver in solvers
        ], return_exceptions=True)

        # Filter out exceptions
        valid_results = [r for r in results if isinstance(r, CodexResult)]

        if not valid_results:
            # All failed
            return CodexResult(
                task_id=task.id,
                success=False,
                score=0.0,
                predictions=[],
                final_code="",
                attempts=[],
                total_tokens=0,
                total_cost=0.0,
                elapsed_seconds=0.0,
            )

        # Select by voting
        return self._vote(valid_results)

    def _vote(self, results: List[CodexResult]) -> CodexResult:
        """Vote on best result."""
        from collections import Counter

        # If any succeeded, pick highest scoring success
        successes = [r for r in results if r.success]
        if successes:
            return max(successes, key=lambda r: r.score)

        # Otherwise, vote by prediction hash
        if results[0].predictions:
            hash_to_result = {}
            hash_counts = Counter()

            for result in results:
                if result.predictions:
                    h = tuple(p.hash() for p in result.predictions)
                    hash_counts[h] += 1
                    if h not in hash_to_result or result.score > hash_to_result[h].score:
                        hash_to_result[h] = result

            if hash_counts:
                best_hash = hash_counts.most_common(1)[0][0]
                return hash_to_result[best_hash]

        # Fallback: highest score
        return max(results, key=lambda r: r.score)
