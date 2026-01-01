"""
Fluid Reasoning Module - Iterative Refinement with Feedback Accumulation.

Inspired by Poetiq's ARC-AGI solver, this module implements:
- Iterative hypothesis generation and testing
- Feedback accumulation across iterations
- Soft scoring for partial credit
- Solution pool with probabilistic selection
- Early termination on success

This enables "fluid intelligence" - adaptive reasoning that learns from
its own mistakes within a single problem-solving session.
"""

import asyncio
import json
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from openai import OpenAI

logger = logging.getLogger(__name__)


@dataclass
class Solution:
    """A single solution attempt with its evaluation."""
    code: str
    feedback: str
    score: float
    iteration: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FluidResult:
    """Result from a fluid reasoning session."""
    final_output: str
    solutions: List[Solution]
    best_score: float
    iteration_count: int
    success: bool
    prompt_tokens: int = 0
    completion_tokens: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExpertConfig:
    """Configuration for a single reasoning expert."""
    solver_prompt: str
    feedback_prompt: str
    model_id: str = "gpt-5.2"
    temperature: float = 0.7
    max_iterations: int = 10
    max_solutions_in_context: int = 5
    selection_probability: float = 1.0
    seed: int = 0
    improving_order: bool = True  # Show solutions worst-first to guide improvement
    return_best_on_failure: bool = True
    reasoning_effort: str = "high"
    timeout_seconds: float = 60.0


class FluidReasoner:
    """
    Implements fluid intelligence through iterative refinement.

    Key patterns from Poetiq:
    1. Generate hypothesis (solution)
    2. Evaluate against criteria
    3. Build detailed feedback
    4. Accumulate solutions with scores
    5. Use feedback to guide next iteration
    6. Early exit on success
    """

    def __init__(
        self,
        client: OpenAI,
        evaluate_fn: Callable[[str, Dict[str, Any]], Tuple[bool, float, str]],
        config: Optional[ExpertConfig] = None,
    ):
        """
        Initialize the fluid reasoner.

        Args:
            client: OpenAI client instance
            evaluate_fn: Function that evaluates a solution.
                         Takes (solution_text, context) and returns (success, score, feedback)
            config: Expert configuration
        """
        self.client = client
        self.evaluate_fn = evaluate_fn
        self.config = config or ExpertConfig(
            solver_prompt=DEFAULT_SOLVER_PROMPT,
            feedback_prompt=DEFAULT_FEEDBACK_PROMPT,
        )
        self.solutions: List[Solution] = []
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    async def solve(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
        success_threshold: float = 1.0,
    ) -> FluidResult:
        """
        Solve a task through iterative refinement.

        Args:
            task: The task description
            context: Additional context for evaluation
            success_threshold: Score threshold to consider the task solved

        Returns:
            FluidResult with the solution and metadata
        """
        context = context or {}
        self.solutions = []
        best_score = -1.0
        best_solution: Optional[Solution] = None
        rng = random.Random(self.config.seed)

        for iteration in range(self.config.max_iterations):
            # Build prompt with task and selected prior solutions
            prompt = self._build_prompt(task, iteration, rng)

            try:
                # Generate solution
                response = await self._call_model(prompt)
                solution_text = self._extract_solution(response)

                if not solution_text:
                    logger.warning(f"Iteration {iteration}: No solution extracted")
                    continue

                # Evaluate the solution
                success, score, feedback = self.evaluate_fn(solution_text, context)

                # Create solution record
                solution = Solution(
                    code=solution_text,
                    feedback=feedback,
                    score=score,
                    iteration=iteration + 1,
                    metadata={"success": success},
                )
                self.solutions.append(solution)

                # Track best
                if score > best_score:
                    best_score = score
                    best_solution = solution

                # Early exit on success
                if success or score >= success_threshold:
                    logger.info(f"Success at iteration {iteration + 1}")
                    return FluidResult(
                        final_output=solution_text,
                        solutions=self.solutions,
                        best_score=score,
                        iteration_count=iteration + 1,
                        success=True,
                        prompt_tokens=self.total_prompt_tokens,
                        completion_tokens=self.total_completion_tokens,
                        metadata={"early_exit": True},
                    )

            except Exception as e:
                logger.error(f"Iteration {iteration} failed: {e}")
                continue

        # Return best solution if configured
        if self.config.return_best_on_failure and best_solution:
            return FluidResult(
                final_output=best_solution.code,
                solutions=self.solutions,
                best_score=best_score,
                iteration_count=self.config.max_iterations,
                success=False,
                prompt_tokens=self.total_prompt_tokens,
                completion_tokens=self.total_completion_tokens,
                metadata={"returned_best": True},
            )

        # Return last solution
        last_solution = self.solutions[-1] if self.solutions else None
        return FluidResult(
            final_output=last_solution.code if last_solution else "",
            solutions=self.solutions,
            best_score=best_score,
            iteration_count=self.config.max_iterations,
            success=False,
            prompt_tokens=self.total_prompt_tokens,
            completion_tokens=self.total_completion_tokens,
        )

    def _build_prompt(self, task: str, iteration: int, rng: random.Random) -> str:
        """Build the prompt with task and selected prior solutions."""
        prompt = self.config.solver_prompt.replace("$$task$$", task)

        if self.solutions:
            # Probabilistically select solutions to include
            mask = [rng.random() < self.config.selection_probability
                    for _ in self.solutions]
            selected = [s for s, keep in zip(self.solutions, mask) if keep]

            if selected:
                feedback_block = self._format_solutions(selected)
                feedback_prompt = self.config.feedback_prompt.replace(
                    "$$feedback$$", feedback_block
                )
                prompt += "\n\n" + feedback_prompt

        return prompt

    def _format_solutions(self, solutions: List[Solution]) -> str:
        """Format solutions for feedback context."""
        # Sort by score
        sorted_solutions = sorted(solutions, key=lambda s: s.score, reverse=True)

        # Limit to max in context
        sorted_solutions = sorted_solutions[:self.config.max_solutions_in_context]

        # Optionally reverse for improving order (worst first)
        if self.config.improving_order:
            sorted_solutions = sorted_solutions[::-1]

        blocks = []
        for i, sol in enumerate(sorted_solutions, 1):
            block = f"""
<solution_{i}>
<solution_attempt>
{sol.code}
</solution_attempt>
<evaluation_feedback>
{sol.feedback}
</evaluation_feedback>
<score>{sol.score:.2f}</score>
</solution_{i}>
"""
            blocks.append(block.strip())

        return "\n\n".join(blocks)

    async def _call_model(self, prompt: str) -> str:
        """Call the model and return the response text."""
        try:
            # Try responses API first (for reasoning models)
            request_args = {
                "model": self.config.model_id,
                "input": prompt,
            }
            if self.config.reasoning_effort:
                request_args["reasoning"] = {"effort": self.config.reasoning_effort}

            response = self.client.responses.create(**request_args)

            # Track tokens
            if hasattr(response, 'usage'):
                self.total_prompt_tokens += getattr(response.usage, 'prompt_tokens', 0)
                self.total_completion_tokens += getattr(response.usage, 'completion_tokens', 0)

            return response.output_text
        except Exception:
            # Fallback to chat completions
            response = self.client.chat.completions.create(
                model=self.config.model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
            )

            # Track tokens
            if response.usage:
                self.total_prompt_tokens += response.usage.prompt_tokens
                self.total_completion_tokens += response.usage.completion_tokens

            return response.choices[0].message.content or ""

    def _extract_solution(self, response: str) -> Optional[str]:
        """Extract the solution from the response."""
        # Try to find code block
        import re

        # Look for python code block
        match = re.search(r"```python\s*(.*?)```", response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Look for any code block
        match = re.search(r"```\s*(.*?)```", response, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Look for FINAL ANSWER marker
        if "FINAL ANSWER:" in response:
            return response.split("FINAL ANSWER:")[-1].strip()

        # Return full response as solution
        return response.strip()


class MultiExpertReasoner:
    """
    Run multiple experts in parallel and use voting to select best result.

    Inspired by Poetiq's solve_parallel_coding:
    - Run N experts concurrently with different seeds
    - Group results by output
    - Rank by vote count (agreement = confidence)
    """

    def __init__(
        self,
        client: OpenAI,
        evaluate_fn: Callable[[str, Dict[str, Any]], Tuple[bool, float, str]],
        num_experts: int = 3,
        base_config: Optional[ExpertConfig] = None,
    ):
        self.client = client
        self.evaluate_fn = evaluate_fn
        self.num_experts = num_experts
        self.base_config = base_config or ExpertConfig(
            solver_prompt=DEFAULT_SOLVER_PROMPT,
            feedback_prompt=DEFAULT_FEEDBACK_PROMPT,
        )

    async def solve(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
        success_threshold: float = 1.0,
    ) -> List[FluidResult]:
        """
        Solve using multiple experts in parallel.

        Returns results sorted by:
        1. Success (successful first)
        2. Vote count (agreement)
        3. Best score
        """
        context = context or {}

        # Create expert configs with different seeds
        configs = []
        for i in range(self.num_experts):
            config = ExpertConfig(
                solver_prompt=self.base_config.solver_prompt,
                feedback_prompt=self.base_config.feedback_prompt,
                model_id=self.base_config.model_id,
                temperature=self.base_config.temperature,
                max_iterations=self.base_config.max_iterations,
                max_solutions_in_context=self.base_config.max_solutions_in_context,
                selection_probability=self.base_config.selection_probability,
                seed=self.base_config.seed + i * self.base_config.max_iterations,
                improving_order=self.base_config.improving_order,
                return_best_on_failure=self.base_config.return_best_on_failure,
                reasoning_effort=self.base_config.reasoning_effort,
            )
            configs.append(config)

        # Run experts in parallel
        tasks = []
        for config in configs:
            reasoner = FluidReasoner(self.client, self.evaluate_fn, config)
            tasks.append(reasoner.solve(task, context, success_threshold))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions
        valid_results = [r for r in results if isinstance(r, FluidResult)]

        # Group by output
        output_groups: Dict[str, List[FluidResult]] = {}
        for result in valid_results:
            key = result.final_output.strip()[:200]  # Use first 200 chars as key
            if key not in output_groups:
                output_groups[key] = []
            output_groups[key].append(result)

        # Sort groups by vote count (descending) then by best score
        sorted_groups = sorted(
            output_groups.values(),
            key=lambda group: (
                -sum(1 for r in group if r.success),  # Successful first
                -len(group),  # More votes = more confidence
                -max(r.best_score for r in group),  # Higher score
            ),
        )

        # Flatten: take one from each group for diversity, then rest
        ordered_results = []
        for group in sorted_groups:
            ordered_results.append(group[0])
        for group in sorted_groups:
            for result in group[1:]:
                ordered_results.append(result)

        return ordered_results


# Default prompts inspired by Poetiq's approach

DEFAULT_SOLVER_PROMPT = """You are an expert problem solver. Your goal is to solve the given task through careful analysis and iterative refinement.

**Task:**
$$task$$

**Approach:**
1. Carefully analyze the task requirements
2. Formulate a clear hypothesis for your solution
3. Implement the solution step by step
4. Verify your solution meets all requirements

**Output Format:**
- Provide a brief explanation of your approach
- Include your solution (code if applicable) in a code block
- End with "FINAL ANSWER:" followed by your final response

Remember: Be thorough, precise, and verify your work.
"""

DEFAULT_FEEDBACK_PROMPT = """**Prior Solution Attempts:**

The following are previous attempts at solving this task, along with their evaluation feedback and scores. Study these carefully to understand what worked, what didn't, and how to improve.

$$feedback$$

**Instructions:**
- Learn from the feedback on previous attempts
- Avoid repeating the same mistakes
- Build on partial successes to achieve a complete solution
- If a prior attempt scored highly, consider refining that approach
"""


def create_simple_evaluator(
    success_keywords: List[str] = None,
    failure_keywords: List[str] = None,
    score_fn: Callable[[str], float] = None,
) -> Callable[[str, Dict[str, Any]], Tuple[bool, float, str]]:
    """
    Create a simple evaluator function.

    Args:
        success_keywords: Keywords that indicate success
        failure_keywords: Keywords that indicate failure
        score_fn: Custom scoring function
    """
    success_keywords = success_keywords or []
    failure_keywords = failure_keywords or []

    def evaluate(solution: str, context: Dict[str, Any]) -> Tuple[bool, float, str]:
        solution_lower = solution.lower()

        # Check for failure keywords
        for keyword in failure_keywords:
            if keyword.lower() in solution_lower:
                return False, 0.2, f"Solution contains problematic element: '{keyword}'"

        # Check for success keywords
        success_count = sum(1 for kw in success_keywords if kw.lower() in solution_lower)
        if success_keywords:
            keyword_score = success_count / len(success_keywords)
        else:
            keyword_score = 0.5

        # Apply custom scoring
        if score_fn:
            custom_score = score_fn(solution)
            final_score = (keyword_score + custom_score) / 2
        else:
            final_score = keyword_score

        success = final_score >= 0.9

        feedback_parts = []
        if success_count > 0:
            feedback_parts.append(f"Found {success_count}/{len(success_keywords)} expected elements.")
        if final_score < 1.0:
            feedback_parts.append(f"Score: {final_score:.2f} - room for improvement.")

        feedback = " ".join(feedback_parts) if feedback_parts else "Evaluation complete."

        return success, final_score, feedback

    return evaluate
