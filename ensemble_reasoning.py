"""
Ensemble Reasoning for AGI-o1 - Multi-Expert Parallel Reasoning with Voting.

Implements fluid intelligence patterns from poetiq-arc-agi-solver:
- Multi-expert parallel solving with different seeds/prompts
- Soft-scoring with partial credit (gradient signal for refinement)
- Hierarchical prompt cascade (analysis → hypothesis → implementation)
- Probabilistic solution reuse (prevents overfitting to early solutions)
- Diversity-first voting mechanism (wisdom of crowds)
- Feedback injection loop (in-context learning from past attempts)

Key patterns borrowed from poetiq-arc-agi-solver:
- solve_parallel_coding.py: Parallel expert orchestration
- solve_coding.py: Iterative refinement loop with feedback
- prompts.py: Hierarchical prompt cascade
- types.py: Solution state management

References:
- ARC-AGI: Abstraction and Reasoning Corpus
- Fluid Intelligence: Novel problem-solving without prior experience
"""

import asyncio
import hashlib
import json
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar
from uuid import uuid4

# Type alias for LLM call function
T = TypeVar('T')
LLMCallable = Callable[[str, Dict[str, Any]], str]


class ReasoningPhase(Enum):
    """
    Hierarchical reasoning phases (from poetiq prompt cascade).
    """
    ANALYSIS = "analysis"      # Understand the problem
    HYPOTHESIS = "hypothesis"  # Generate hypotheses
    IMPLEMENTATION = "implementation"  # Execute solution
    VERIFICATION = "verification"  # Validate results
    REFINEMENT = "refinement"  # Improve based on feedback


@dataclass
class Solution:
    """
    A single solution attempt with scoring and feedback.

    Modeled after poetiq's ARCAGISolution type.
    """
    id: str
    content: str  # The solution output
    code: Optional[str] = None  # If code was generated
    score: float = 0.0  # Soft score [0, 1]
    feedback: str = ""  # Detailed feedback for refinement
    phase: ReasoningPhase = ReasoningPhase.IMPLEMENTATION
    iteration: int = 1
    expert_id: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash(self.id)


@dataclass
class ExpertConfig:
    """
    Configuration for a single expert in the ensemble.

    Different experts can use different prompts, temperatures, seeds.
    """
    id: str
    name: str
    prompt_template: str
    temperature: float = 0.7
    seed: Optional[int] = None
    reasoning_effort: str = "high"
    max_tokens: int = 4096
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningResult:
    """
    Result of an ensemble reasoning session.
    """
    task: str
    solutions: List[Solution]
    best_solution: Optional[Solution]
    consensus_solution: Optional[Solution]
    phase: ReasoningPhase
    iterations: int
    total_time: float
    token_usage: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SoftScore:
    """
    Soft scoring result with breakdown.

    Implements poetiq's soft-scoring for partial credit.
    """
    total: float  # Overall score [0, 1]
    correctness: float  # How correct is the answer
    completeness: float  # How complete is the answer
    coherence: float  # How coherent is the reasoning
    novelty: float  # How novel/creative is the approach
    feedback: str  # Detailed feedback
    breakdown: Dict[str, float] = field(default_factory=dict)


# ---- Hierarchical Prompt Templates ----

ANALYSIS_PROMPT = """
You are an expert analyst. Your task is to deeply understand the problem before proposing solutions.

## Problem
{problem}

## Analysis Instructions
1. Identify the core challenge or question
2. Break down into sub-problems if complex
3. Identify relevant patterns, constraints, and edge cases
4. List assumptions that need validation
5. Consider what information is missing

## Output Format
Provide a structured analysis with:
- Core Problem: [one sentence summary]
- Key Constraints: [list]
- Sub-Problems: [if any]
- Critical Considerations: [list]
- Recommended Approach: [brief strategy]
"""

HYPOTHESIS_PROMPT = """
You are a hypothesis generator. Based on the analysis, generate multiple solution approaches.

## Problem
{problem}

## Previous Analysis
{analysis}

## Hypothesis Instructions
Generate 3-5 distinct solution approaches. For each:
1. State the core hypothesis/approach
2. Explain why it might work
3. Identify potential failure modes
4. Estimate likelihood of success (high/medium/low)

## Output Format
For each hypothesis:
### Hypothesis N: [Name]
- Approach: [description]
- Rationale: [why this might work]
- Risks: [potential issues]
- Confidence: [high/medium/low]
"""

IMPLEMENTATION_PROMPT = """
You are an implementation expert. Execute the best solution approach.

## Problem
{problem}

## Selected Approach
{approach}

## Previous Attempts (for reference)
{previous_attempts}

## Implementation Instructions
1. Implement the solution step by step
2. Show your reasoning at each step
3. If generating code, ensure it is complete and runnable
4. Handle edge cases explicitly
5. Verify your solution addresses the original problem

## Output Format
Provide:
1. Step-by-step implementation
2. Final solution/code
3. Verification that it solves the problem
"""

VERIFICATION_PROMPT = """
You are a verification expert. Evaluate the solution against the problem requirements.

## Original Problem
{problem}

## Proposed Solution
{solution}

## Verification Instructions
Evaluate the solution on:
1. Correctness: Does it solve the problem correctly?
2. Completeness: Does it handle all cases?
3. Coherence: Is the reasoning sound?
4. Efficiency: Is it reasonably efficient?

Provide a score from 0-100 for each dimension and overall.

## Output Format
```json
{{
  "correctness": <0-100>,
  "completeness": <0-100>,
  "coherence": <0-100>,
  "efficiency": <0-100>,
  "overall": <0-100>,
  "issues": ["list of issues found"],
  "improvements": ["suggested improvements"]
}}
```
"""

REFINEMENT_PROMPT = """
You are a refinement expert. Improve the solution based on feedback.

## Original Problem
{problem}

## Current Solution
{solution}

## Feedback
{feedback}

## Previous Iterations
{history}

## Refinement Instructions
1. Address each issue in the feedback
2. Incorporate suggested improvements
3. Maintain what was working well
4. Verify the refined solution

## Output
Provide the improved solution with explanation of changes made.
"""


class SoftScorer:
    """
    Implements soft-scoring with partial credit.

    Borrowed from poetiq's _soft_score and _build_feedback functions.
    """

    def __init__(self, llm_call: Optional[LLMCallable] = None):
        self.llm_call = llm_call

    def score_solution(
        self,
        problem: str,
        solution: Solution,
        expected: Optional[str] = None
    ) -> SoftScore:
        """
        Score a solution with partial credit.

        Unlike binary pass/fail, this provides gradient signal for refinement.
        """
        if self.llm_call:
            return self._llm_score(problem, solution, expected)
        else:
            return self._heuristic_score(problem, solution, expected)

    def _llm_score(
        self,
        problem: str,
        solution: Solution,
        expected: Optional[str]
    ) -> SoftScore:
        """Use LLM for scoring (more accurate but slower)."""
        prompt = VERIFICATION_PROMPT.format(
            problem=problem,
            solution=solution.content
        )

        try:
            response = self.llm_call(prompt, {"temperature": 0.0})

            # Try to extract JSON scores
            import re
            json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if json_match:
                scores = json.loads(json_match.group())
            else:
                # Fallback to heuristic
                return self._heuristic_score(problem, solution, expected)

            # Normalize to [0, 1]
            total = scores.get('overall', 50) / 100
            return SoftScore(
                total=total,
                correctness=scores.get('correctness', 50) / 100,
                completeness=scores.get('completeness', 50) / 100,
                coherence=scores.get('coherence', 50) / 100,
                novelty=0.5,  # Not evaluated by default
                feedback=f"Issues: {scores.get('issues', [])}. Improvements: {scores.get('improvements', [])}",
                breakdown=scores
            )

        except Exception as e:
            logging.warning("LLM scoring failed: %s", e)
            return self._heuristic_score(problem, solution, expected)

    def _heuristic_score(
        self,
        problem: str,
        solution: Solution,
        expected: Optional[str]
    ) -> SoftScore:
        """Heuristic scoring when LLM not available."""
        content = solution.content.lower()

        # Basic heuristics
        length_score = min(1.0, len(content) / 500)  # Reasonable length
        structure_score = 0.5

        # Check for structured output
        if any(marker in content for marker in ['##', '```', '- ', '1.']):
            structure_score = 0.8

        # Check for reasoning indicators
        reasoning_keywords = ['because', 'therefore', 'since', 'if', 'then', 'consider']
        reasoning_score = min(1.0, sum(1 for k in reasoning_keywords if k in content) / 3)

        total = (length_score * 0.2 + structure_score * 0.3 + reasoning_score * 0.5)

        return SoftScore(
            total=total,
            correctness=0.5,  # Unknown without verification
            completeness=length_score,
            coherence=structure_score,
            novelty=0.5,
            feedback="Heuristic scoring used. Consider LLM verification for accuracy."
        )

    def build_feedback(
        self,
        solution: Solution,
        score: SoftScore
    ) -> str:
        """
        Build detailed feedback for refinement loop.

        Borrowed from poetiq's _build_feedback function.
        """
        lines = []

        lines.append(f"## Evaluation (Score: {score.total:.2f})")
        lines.append("")
        lines.append("### Scores:")
        lines.append(f"- Correctness: {score.correctness:.2f}")
        lines.append(f"- Completeness: {score.completeness:.2f}")
        lines.append(f"- Coherence: {score.coherence:.2f}")
        lines.append(f"- Novelty: {score.novelty:.2f}")
        lines.append("")

        if score.feedback:
            lines.append("### Feedback:")
            lines.append(score.feedback)
            lines.append("")

        if score.total < 0.7:
            lines.append("### Areas for Improvement:")
            if score.correctness < 0.7:
                lines.append("- Verify the solution produces correct results")
            if score.completeness < 0.7:
                lines.append("- Ensure all cases are handled")
            if score.coherence < 0.7:
                lines.append("- Improve reasoning clarity and structure")

        return "\n".join(lines)


class FeedbackInjector:
    """
    Injects feedback from previous solutions into prompts.

    Implements poetiq's in-context learning through feedback injection.
    """

    def __init__(self, selection_probability: float = 0.7):
        """
        Args:
            selection_probability: Probability of including each previous solution.
                                   Prevents overfitting to early solutions.
        """
        self.selection_probability = selection_probability
        self._rng = random.Random()

    def select_solutions(
        self,
        solutions: List[Solution],
        max_solutions: int = 3,
        improving_order: bool = True
    ) -> List[Solution]:
        """
        Probabilistically select solutions for feedback.

        Borrowed from poetiq's probabilistic selection pattern.
        """
        if not solutions:
            return []

        # Sort by score
        sorted_solutions = sorted(
            solutions,
            key=lambda s: s.score,
            reverse=not improving_order  # Worst first if improving_order
        )

        # Probabilistic selection
        selected = []
        for solution in sorted_solutions:
            if self._rng.random() < self.selection_probability:
                selected.append(solution)
                if len(selected) >= max_solutions:
                    break

        return selected

    def inject_feedback(
        self,
        prompt: str,
        solutions: List[Solution],
        placeholder: str = "{previous_attempts}"
    ) -> str:
        """
        Inject selected solutions as feedback into prompt.
        """
        if not solutions:
            feedback_text = "No previous attempts available."
        else:
            feedback_parts = []
            for i, sol in enumerate(solutions):
                feedback_parts.append(f"### Attempt {i + 1} (Score: {sol.score:.2f})")
                feedback_parts.append(sol.content[:500] + "..." if len(sol.content) > 500 else sol.content)
                if sol.feedback:
                    feedback_parts.append(f"Feedback: {sol.feedback}")
                feedback_parts.append("")

            feedback_text = "\n".join(feedback_parts)

        return prompt.replace(placeholder, feedback_text)


class VotingMechanism:
    """
    Diversity-first voting for ensemble consensus.

    Implements poetiq's voting mechanism that prioritizes diverse solutions.
    """

    @staticmethod
    def canonical_key(solution: Solution) -> str:
        """
        Generate canonical key for grouping similar solutions.

        Similar to poetiq's canonical_test_key.
        """
        # Hash the core content (normalized)
        content = solution.content.strip().lower()
        # Remove whitespace variations
        content = ' '.join(content.split())
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def vote(
        self,
        solutions: List[Solution],
        min_score: float = 0.5
    ) -> List[Solution]:
        """
        Vote on solutions with diversity-first ordering.

        Returns solutions ordered by: consensus > diversity > score.
        """
        if not solutions:
            return []

        # Group by canonical key
        buckets: Dict[str, List[Solution]] = {}
        for sol in solutions:
            key = self.canonical_key(sol)
            if key not in buckets:
                buckets[key] = []
            buckets[key].append(sol)

        # Separate passers (good enough) from failures
        passers = {k: v for k, v in buckets.items()
                   if any(s.score >= min_score for s in v)}
        failures = {k: v for k, v in buckets.items()
                    if k not in passers}

        # Sort passers by vote count (popularity)
        passer_groups = sorted(
            passers.items(),
            key=lambda kv: len(kv[1]),
            reverse=True
        )

        # Sort failures by best score in group
        failure_groups = sorted(
            failures.items(),
            key=lambda kv: max(s.score for s in kv[1]),
            reverse=True
        )

        # Build diversity-first output
        result = []

        # One per passer group (diversity)
        for key, group in passer_groups:
            best_in_group = max(group, key=lambda s: s.score)
            result.append(best_in_group)

        # One per failure group
        for key, group in failure_groups:
            best_in_group = max(group, key=lambda s: s.score)
            result.append(best_in_group)

        # Remaining passers
        for key, group in passer_groups:
            for sol in sorted(group, key=lambda s: s.score, reverse=True)[1:]:
                result.append(sol)

        return result


class EnsembleReasoner:
    """
    Multi-expert parallel reasoning with fluid intelligence patterns.

    Main orchestrator implementing:
    - Parallel expert execution
    - Hierarchical prompt cascade
    - Soft-scoring feedback loop
    - Diversity-first voting

    Inspired by poetiq-arc-agi-solver's solve_parallel_coding.py
    """

    def __init__(
        self,
        llm_call: LLMCallable,
        num_experts: int = 3,
        max_iterations: int = 5,
        selection_probability: float = 0.7,
        min_score_threshold: float = 0.8,
        timeout_seconds: float = 300.0
    ):
        """
        Initialize ensemble reasoner.

        Args:
            llm_call: Function to call LLM (prompt, config) -> response
            num_experts: Number of parallel experts
            max_iterations: Maximum refinement iterations per expert
            selection_probability: Probability of including past solutions in feedback
            min_score_threshold: Score threshold for early termination
            timeout_seconds: Maximum time for entire reasoning session
        """
        self.llm_call = llm_call
        self.num_experts = num_experts
        self.max_iterations = max_iterations
        self.selection_probability = selection_probability
        self.min_score_threshold = min_score_threshold
        self.timeout_seconds = timeout_seconds

        self.scorer = SoftScorer(llm_call)
        self.feedback_injector = FeedbackInjector(selection_probability)
        self.voting = VotingMechanism()

    def create_expert_configs(
        self,
        base_temperature: float = 0.7,
        base_seed: int = 42
    ) -> List[ExpertConfig]:
        """
        Create diverse expert configurations.

        Each expert has different seed/temperature for exploration diversity.
        """
        configs = []
        for i in range(self.num_experts):
            # Vary temperature and seed for diversity
            temp = base_temperature + (i * 0.1)  # 0.7, 0.8, 0.9, ...
            seed = base_seed + (i * 1000)

            config = ExpertConfig(
                id=f"expert_{i}",
                name=f"Expert {i + 1}",
                prompt_template=IMPLEMENTATION_PROMPT,
                temperature=min(temp, 1.0),
                seed=seed,
                reasoning_effort="high" if i == 0 else "medium"
            )
            configs.append(config)

        return configs

    def _run_single_expert(
        self,
        expert: ExpertConfig,
        problem: str,
        initial_analysis: str = "",
        approach: str = ""
    ) -> List[Solution]:
        """
        Run a single expert through refinement iterations.

        Implements poetiq's solve_coding pattern.
        """
        solutions = []

        for iteration in range(self.max_iterations):
            # Select previous solutions for feedback
            selected = self.feedback_injector.select_solutions(
                solutions,
                max_solutions=3,
                improving_order=True  # Show worst first (improving order)
            )

            # Build prompt with feedback
            prompt = self.feedback_injector.inject_feedback(
                expert.prompt_template.format(
                    problem=problem,
                    analysis=initial_analysis,
                    approach=approach or "Use your best judgment",
                    previous_attempts="{previous_attempts}"
                ),
                selected
            )

            # Call LLM
            try:
                response = self.llm_call(prompt, {
                    "temperature": expert.temperature,
                    "seed": expert.seed + iteration if expert.seed else None,
                    "max_tokens": expert.max_tokens,
                    "reasoning_effort": expert.reasoning_effort
                })
            except Exception as e:
                logging.warning("Expert %s failed on iteration %d: %s",
                               expert.id, iteration, e)
                continue

            # Create solution
            solution = Solution(
                id=f"{expert.id}_iter_{iteration}",
                content=response,
                phase=ReasoningPhase.IMPLEMENTATION,
                iteration=iteration + 1,
                expert_id=expert.id
            )

            # Score solution
            score = self.scorer.score_solution(problem, solution)
            solution.score = score.total
            solution.feedback = self.scorer.build_feedback(solution, score)

            solutions.append(solution)

            # Early termination if good enough
            if score.total >= self.min_score_threshold:
                logging.info("Expert %s found good solution at iteration %d (score: %.2f)",
                            expert.id, iteration, score.total)
                break

        return solutions

    def reason_parallel(
        self,
        problem: str,
        analysis: str = "",
        approach: str = ""
    ) -> ReasoningResult:
        """
        Run parallel expert reasoning.

        Implements poetiq's solve_parallel_coding pattern with:
        - Parallel expert execution
        - Diversity-first voting
        - Consensus detection
        """
        start_time = time.time()
        all_solutions: List[Solution] = []

        # Create expert configs
        experts = self.create_expert_configs()

        # Run experts in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.num_experts) as executor:
            futures = {
                executor.submit(
                    self._run_single_expert,
                    expert, problem, analysis, approach
                ): expert
                for expert in experts
            }

            for future in as_completed(futures, timeout=self.timeout_seconds):
                expert = futures[future]
                try:
                    solutions = future.result()
                    all_solutions.extend(solutions)
                    logging.info("Expert %s produced %d solutions",
                                expert.id, len(solutions))
                except Exception as e:
                    logging.warning("Expert %s failed: %s", expert.id, e)

        # Vote on solutions
        ranked_solutions = self.voting.vote(all_solutions, min_score=0.5)

        # Find best and consensus
        best_solution = max(all_solutions, key=lambda s: s.score) if all_solutions else None
        consensus_solution = ranked_solutions[0] if ranked_solutions else None

        elapsed = time.time() - start_time

        return ReasoningResult(
            task=problem,
            solutions=ranked_solutions,
            best_solution=best_solution,
            consensus_solution=consensus_solution,
            phase=ReasoningPhase.IMPLEMENTATION,
            iterations=sum(s.iteration for s in all_solutions) if all_solutions else 0,
            total_time=elapsed,
            token_usage=0,  # Would need to track from LLM calls
            metadata={
                "num_experts": self.num_experts,
                "total_solutions": len(all_solutions),
                "unique_solutions": len(set(self.voting.canonical_key(s) for s in all_solutions))
            }
        )

    def reason_hierarchical(
        self,
        problem: str
    ) -> ReasoningResult:
        """
        Run hierarchical reasoning through all phases.

        Implements the full fluid intelligence loop:
        1. Analysis - understand the problem
        2. Hypothesis - generate approaches
        3. Implementation - execute (parallel)
        4. Verification - score
        5. Refinement - improve based on feedback
        """
        start_time = time.time()

        # Phase 1: Analysis
        logging.info("Phase 1: Analysis")
        analysis_prompt = ANALYSIS_PROMPT.format(problem=problem)
        analysis = self.llm_call(analysis_prompt, {"temperature": 0.3})

        # Phase 2: Hypothesis Generation
        logging.info("Phase 2: Hypothesis Generation")
        hypothesis_prompt = HYPOTHESIS_PROMPT.format(
            problem=problem,
            analysis=analysis
        )
        hypotheses = self.llm_call(hypothesis_prompt, {"temperature": 0.7})

        # Phase 3: Parallel Implementation
        logging.info("Phase 3: Parallel Implementation")
        result = self.reason_parallel(problem, analysis, hypotheses)

        # Phase 4 & 5: Verification and Refinement (built into parallel loop)

        result.metadata['analysis'] = analysis[:500]  # Store truncated
        result.metadata['hypotheses'] = hypotheses[:500]
        result.total_time = time.time() - start_time

        return result


# ---- Convenience Functions ----

def create_ensemble_reasoner(
    llm_call: LLMCallable,
    num_experts: int = 3,
    max_iterations: int = 5
) -> EnsembleReasoner:
    """Create an ensemble reasoner with default settings."""
    return EnsembleReasoner(
        llm_call=llm_call,
        num_experts=num_experts,
        max_iterations=max_iterations
    )


def run_fluid_reasoning(
    problem: str,
    llm_call: LLMCallable,
    hierarchical: bool = True,
    num_experts: int = 3
) -> ReasoningResult:
    """
    Convenience function to run fluid intelligence reasoning.

    Args:
        problem: The problem to solve
        llm_call: LLM call function
        hierarchical: Whether to use hierarchical (full) or parallel (fast) mode
        num_experts: Number of parallel experts

    Returns:
        ReasoningResult with ranked solutions
    """
    reasoner = create_ensemble_reasoner(llm_call, num_experts)

    if hierarchical:
        return reasoner.reason_hierarchical(problem)
    else:
        return reasoner.reason_parallel(problem)
