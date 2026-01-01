"""
Soft Scoring System - Partial Credit for Solution Quality.

Inspired by Poetiq's scoring approach, this module provides:
- Soft scoring (0-1 range) for partial solutions
- Multi-dimensional evaluation criteria
- Feedback generation with actionable improvements
- Score aggregation and comparison

This enables gradual improvement tracking and better feedback loops.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ScoreResult:
    """Result of a scoring evaluation."""
    score: float  # 0-1 overall score
    success: bool  # Did it pass the success threshold?
    dimension_scores: Dict[str, float] = field(default_factory=dict)
    feedback: str = ""
    improvements: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoringCriteria:
    """Definition of a scoring dimension."""
    name: str
    weight: float = 1.0
    description: str = ""
    scorer: Optional[Callable[[str, Dict[str, Any]], float]] = None
    success_threshold: float = 0.8


class SoftScorer:
    """
    Multi-dimensional soft scoring system.

    Key features from Poetiq:
    - Partial credit (not just pass/fail)
    - Multiple evaluation dimensions
    - Weighted aggregation
    - Detailed feedback generation
    """

    def __init__(
        self,
        criteria: Optional[List[ScoringCriteria]] = None,
        success_threshold: float = 0.9,
    ):
        self.criteria = criteria or []
        self.success_threshold = success_threshold

    def add_criterion(
        self,
        name: str,
        scorer: Callable[[str, Dict[str, Any]], float],
        weight: float = 1.0,
        description: str = "",
        success_threshold: float = 0.8,
    ) -> None:
        """Add a scoring criterion."""
        self.criteria.append(ScoringCriteria(
            name=name,
            weight=weight,
            description=description,
            scorer=scorer,
            success_threshold=success_threshold,
        ))

    def score(
        self,
        solution: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ScoreResult:
        """
        Score a solution against all criteria.

        Returns a ScoreResult with:
        - Overall weighted score
        - Per-dimension scores
        - Success determination
        - Actionable feedback
        """
        context = context or {}
        dimension_scores = {}
        improvements = []
        feedback_parts = []

        total_weight = sum(c.weight for c in self.criteria)
        weighted_sum = 0.0

        for criterion in self.criteria:
            try:
                if criterion.scorer:
                    dim_score = criterion.scorer(solution, context)
                else:
                    dim_score = 0.5  # Default neutral score

                # Clamp to 0-1
                dim_score = max(0.0, min(1.0, dim_score))
                dimension_scores[criterion.name] = dim_score

                # Weighted contribution
                weighted_sum += dim_score * criterion.weight

                # Generate feedback
                if dim_score < criterion.success_threshold:
                    if criterion.description:
                        feedback_parts.append(
                            f"{criterion.name}: {dim_score:.2f} - {criterion.description}"
                        )
                    else:
                        feedback_parts.append(
                            f"{criterion.name}: {dim_score:.2f} - needs improvement"
                        )
                    improvements.append(f"Improve {criterion.name}")
                else:
                    feedback_parts.append(
                        f"{criterion.name}: {dim_score:.2f} - good"
                    )

            except Exception as e:
                logger.warning(f"Scorer {criterion.name} failed: {e}")
                dimension_scores[criterion.name] = 0.0

        # Calculate overall score
        if total_weight > 0:
            overall_score = weighted_sum / total_weight
        else:
            overall_score = 0.5

        success = overall_score >= self.success_threshold

        feedback = "\n".join(feedback_parts) if feedback_parts else "No feedback available."

        return ScoreResult(
            score=overall_score,
            success=success,
            dimension_scores=dimension_scores,
            feedback=feedback,
            improvements=improvements,
        )


# Pre-built scorers for common use cases

def code_syntax_scorer(solution: str, context: Dict[str, Any]) -> float:
    """Score code based on syntax validity."""
    try:
        compile(solution, "<string>", "exec")
        return 1.0
    except SyntaxError:
        return 0.0


def code_length_scorer(
    min_lines: int = 1,
    max_lines: int = 200,
) -> Callable[[str, Dict[str, Any]], float]:
    """Score based on code length (prefer reasonable length)."""
    def scorer(solution: str, context: Dict[str, Any]) -> float:
        lines = len(solution.strip().split("\n"))
        if lines < min_lines:
            return lines / min_lines
        if lines > max_lines:
            return max(0, 1 - (lines - max_lines) / max_lines)
        return 1.0
    return scorer


def keyword_presence_scorer(
    required_keywords: List[str],
    optional_keywords: Optional[List[str]] = None,
) -> Callable[[str, Dict[str, Any]], float]:
    """Score based on presence of expected keywords."""
    optional_keywords = optional_keywords or []

    def scorer(solution: str, context: Dict[str, Any]) -> float:
        solution_lower = solution.lower()

        # Required keywords are worth more
        required_found = sum(
            1 for kw in required_keywords
            if kw.lower() in solution_lower
        )
        required_score = required_found / len(required_keywords) if required_keywords else 1.0

        # Optional keywords add bonus
        if optional_keywords:
            optional_found = sum(
                1 for kw in optional_keywords
                if kw.lower() in solution_lower
            )
            optional_score = optional_found / len(optional_keywords)
            return 0.7 * required_score + 0.3 * optional_score

        return required_score

    return scorer


def output_match_scorer(
    expected_pattern: str,
    is_regex: bool = False,
) -> Callable[[str, Dict[str, Any]], float]:
    """Score based on matching expected output pattern."""
    def scorer(solution: str, context: Dict[str, Any]) -> float:
        if is_regex:
            if re.search(expected_pattern, solution):
                return 1.0
            return 0.0
        else:
            if expected_pattern in solution:
                return 1.0
            # Partial match based on common substrings
            matches = 0
            words = expected_pattern.split()
            for word in words:
                if word in solution:
                    matches += 1
            return matches / len(words) if words else 0.0
    return scorer


def array_similarity_scorer(
    expected_array: List[List[Any]],
) -> Callable[[str, Dict[str, Any]], float]:
    """Score based on similarity to expected array (for ARC-style tasks)."""
    def scorer(solution: str, context: Dict[str, Any]) -> float:
        try:
            # Try to parse solution as JSON array
            parsed = json.loads(solution)
            if not isinstance(parsed, list):
                return 0.0

            expected = np.array(expected_array)
            actual = np.array(parsed)

            # Shape mismatch
            if expected.shape != actual.shape:
                return 0.0

            # Pixel-level accuracy
            if expected.size == 0:
                return 1.0

            accuracy = float(np.mean(expected == actual))
            return accuracy

        except (json.JSONDecodeError, ValueError):
            return 0.0

    return scorer


def structural_scorer(
    required_elements: List[str],
) -> Callable[[str, Dict[str, Any]], float]:
    """Score based on presence of structural elements (classes, functions, etc.)."""
    def scorer(solution: str, context: Dict[str, Any]) -> float:
        found = 0
        for element in required_elements:
            pattern = None
            if element.startswith("def "):
                # Function definition
                func_name = element[4:].strip()
                pattern = rf"def\s+{re.escape(func_name)}\s*\("
            elif element.startswith("class "):
                # Class definition
                class_name = element[6:].strip()
                pattern = rf"class\s+{re.escape(class_name)}"
            elif element.startswith("import "):
                # Import statement
                module = element[7:].strip()
                pattern = rf"import\s+{re.escape(module)}"
            else:
                # Literal match
                pattern = re.escape(element)

            if pattern and re.search(pattern, solution):
                found += 1

        return found / len(required_elements) if required_elements else 1.0

    return scorer


class CodeExecutionScorer:
    """
    Score code by executing it and comparing outputs.

    Similar to Poetiq's sandbox evaluation:
    - Execute code in isolated environment
    - Compare outputs against expected
    - Track partial correctness
    """

    def __init__(
        self,
        timeout_seconds: float = 5.0,
        allowed_imports: Optional[List[str]] = None,
    ):
        self.timeout_seconds = timeout_seconds
        self.allowed_imports = allowed_imports or ["numpy", "json", "math", "re"]

    def score(
        self,
        code: str,
        test_cases: List[Dict[str, Any]],
    ) -> ScoreResult:
        """
        Execute code against test cases and score results.

        Each test case should have:
        - input: The input to pass to the function
        - expected: The expected output
        - (optional) function_name: Name of function to call
        """
        results = []
        feedback_parts = []

        for i, test_case in enumerate(test_cases):
            try:
                # Create isolated namespace
                namespace = {"__builtins__": {}}

                # Add allowed imports
                for module_name in self.allowed_imports:
                    try:
                        namespace[module_name] = __import__(module_name)
                    except ImportError:
                        pass

                # Execute the code
                exec(code, namespace)

                # Get the function to call
                func_name = test_case.get("function_name", "transform")
                if func_name not in namespace:
                    # Try to find any callable
                    callables = [
                        name for name, obj in namespace.items()
                        if callable(obj) and not name.startswith("_")
                    ]
                    if callables:
                        func_name = callables[0]
                    else:
                        results.append(0.0)
                        feedback_parts.append(f"Test {i+1}: No callable function found")
                        continue

                func = namespace[func_name]

                # Call the function
                input_val = test_case["input"]
                if isinstance(input_val, list):
                    input_val = np.array(input_val)

                result = func(input_val)

                # Convert result for comparison
                if hasattr(result, "tolist"):
                    result = result.tolist()

                expected = test_case["expected"]

                # Compare
                if result == expected:
                    results.append(1.0)
                    feedback_parts.append(f"Test {i+1}: Passed")
                else:
                    # Calculate partial score
                    try:
                        result_arr = np.array(result)
                        expected_arr = np.array(expected)
                        if result_arr.shape == expected_arr.shape:
                            partial = float(np.mean(result_arr == expected_arr))
                        else:
                            partial = 0.0
                    except:
                        partial = 0.0

                    results.append(partial)
                    feedback_parts.append(
                        f"Test {i+1}: Failed (score: {partial:.2f})"
                    )

            except Exception as e:
                results.append(0.0)
                feedback_parts.append(f"Test {i+1}: Error - {str(e)[:50]}")

        # Calculate overall score
        overall_score = float(np.mean(results)) if results else 0.0
        success = overall_score >= 0.99  # All tests must pass

        return ScoreResult(
            score=overall_score,
            success=success,
            dimension_scores={"execution": overall_score},
            feedback="\n".join(feedback_parts),
            improvements=[] if success else ["Fix failing test cases"],
        )


def create_composite_scorer(
    *scorers: Tuple[str, Callable[[str, Dict[str, Any]], float], float],
) -> SoftScorer:
    """
    Create a composite scorer from multiple scoring functions.

    Args:
        scorers: Tuples of (name, scorer_function, weight)
    """
    scorer = SoftScorer()
    for name, scorer_fn, weight in scorers:
        scorer.add_criterion(name, scorer_fn, weight)
    return scorer


def build_feedback(
    score_result: ScoreResult,
    solution: str,
    expected: Optional[str] = None,
) -> str:
    """
    Build detailed feedback from a score result.

    Inspired by Poetiq's _build_feedback function:
    - Shows what worked
    - Shows what didn't with specifics
    - Provides actionable improvements
    """
    lines = []

    # Overall status
    status = "PASS" if score_result.success else "FAIL"
    lines.append(f"Status: {status} (Score: {score_result.score:.2f})")
    lines.append("")

    # Per-dimension breakdown
    lines.append("Dimension Scores:")
    for dim_name, dim_score in score_result.dimension_scores.items():
        indicator = "+" if dim_score >= 0.8 else "-"
        lines.append(f"  {indicator} {dim_name}: {dim_score:.2f}")
    lines.append("")

    # Detailed feedback
    lines.append("Feedback:")
    lines.append(score_result.feedback)
    lines.append("")

    # Improvements
    if score_result.improvements:
        lines.append("Suggested Improvements:")
        for imp in score_result.improvements:
            lines.append(f"  - {imp}")

    # Diff if expected provided
    if expected and not score_result.success:
        lines.append("")
        lines.append("Expected vs. Actual Diff:")
        # Simple diff visualization
        exp_lines = expected.strip().split("\n")[:10]
        sol_lines = solution.strip().split("\n")[:10]
        for i, (e, s) in enumerate(zip(exp_lines, sol_lines)):
            if e != s:
                lines.append(f"  Line {i+1}: expected '{e[:50]}...' got '{s[:50]}...'")

    return "\n".join(lines)
