"""
ARC Evaluator Module - Scoring and feedback for ARC solutions.

Provides exact-match evaluation (required for ARC) plus partial credit
scoring for refinement loop feedback.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .grid import Grid, GridPair
from .dataset import ARCTask

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Result of evaluating a solution against an ARC task."""
    correct: bool
    score: float  # 0-1, partial credit
    feedback: str
    predictions: List[Grid]
    expected: List[Optional[Grid]]
    per_test_scores: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_correct(self) -> int:
        return sum(1 for s in self.per_test_scores if s == 1.0)

    @property
    def num_tests(self) -> int:
        return len(self.per_test_scores)


class ARCEvaluator:
    """
    Evaluator for ARC task solutions.

    Supports:
    - Exact match evaluation (competition standard)
    - Partial credit for refinement feedback
    - Detailed mismatch analysis
    """

    def __init__(self, allow_partial_credit: bool = True):
        self.allow_partial_credit = allow_partial_credit

    def evaluate(
        self,
        task: ARCTask,
        predictions: List[Grid],
    ) -> EvaluationResult:
        """
        Evaluate predictions against task test outputs.

        Args:
            task: The ARC task
            predictions: List of predicted output grids (one per test)

        Returns:
            EvaluationResult with scores and feedback
        """
        if len(predictions) != len(task.test):
            return EvaluationResult(
                correct=False,
                score=0.0,
                feedback=f"Wrong number of predictions: got {len(predictions)}, expected {len(task.test)}",
                predictions=predictions,
                expected=[p.output for p in task.test],
            )

        per_test_scores = []
        feedback_parts = []

        for i, (pred, test_pair) in enumerate(zip(predictions, task.test)):
            expected = test_pair.output

            if expected is None:
                # No ground truth available (competition mode)
                feedback_parts.append(f"Test {i+1}: No ground truth available")
                per_test_scores.append(0.0)  # Can't score
                continue

            # Check exact match
            if pred == expected:
                per_test_scores.append(1.0)
                feedback_parts.append(f"Test {i+1}: CORRECT")
            else:
                # Calculate partial score
                if self.allow_partial_credit:
                    partial_score = self._calculate_partial_score(pred, expected)
                    per_test_scores.append(partial_score)

                    # Generate detailed feedback
                    mismatch_feedback = self._generate_mismatch_feedback(pred, expected, i)
                    feedback_parts.append(mismatch_feedback)
                else:
                    per_test_scores.append(0.0)
                    feedback_parts.append(f"Test {i+1}: INCORRECT")

        # Overall score
        overall_score = sum(per_test_scores) / len(per_test_scores) if per_test_scores else 0.0
        all_correct = all(s == 1.0 for s in per_test_scores)

        return EvaluationResult(
            correct=all_correct,
            score=overall_score,
            feedback="\n".join(feedback_parts),
            predictions=predictions,
            expected=[p.output for p in task.test],
            per_test_scores=per_test_scores,
        )

    def evaluate_single(self, predicted: Grid, expected: Grid) -> Tuple[bool, float, str]:
        """Evaluate a single prediction against expected output."""
        if predicted == expected:
            return True, 1.0, "CORRECT: Exact match"

        partial_score = self._calculate_partial_score(predicted, expected)
        feedback = self._generate_mismatch_feedback(predicted, expected, 0)

        return False, partial_score, feedback

    def _calculate_partial_score(self, predicted: Grid, expected: Grid) -> float:
        """
        Calculate partial credit score.

        Scoring components:
        1. Shape match (40%)
        2. Pixel accuracy (40%)
        3. Color palette match (10%)
        4. Pattern similarity (10%)
        """
        scores = {}

        # 1. Shape match
        if predicted.shape == expected.shape:
            scores["shape"] = 1.0
        else:
            # Partial credit based on size similarity
            pred_size = predicted.height * predicted.width
            exp_size = expected.height * expected.width
            size_ratio = min(pred_size, exp_size) / max(pred_size, exp_size)

            # Also consider aspect ratio
            pred_aspect = predicted.height / predicted.width if predicted.width > 0 else 0
            exp_aspect = expected.height / expected.width if expected.width > 0 else 0
            aspect_diff = abs(pred_aspect - exp_aspect) / max(pred_aspect, exp_aspect, 1)

            scores["shape"] = size_ratio * (1 - min(1, aspect_diff))

        # 2. Pixel accuracy (only if shapes match or can be compared)
        if predicted.shape == expected.shape:
            scores["pixels"] = float(np.mean(predicted.data == expected.data))
        else:
            # Compare overlapping region
            min_h = min(predicted.height, expected.height)
            min_w = min(predicted.width, expected.width)
            if min_h > 0 and min_w > 0:
                pred_crop = predicted.data[:min_h, :min_w]
                exp_crop = expected.data[:min_h, :min_w]
                overlap_accuracy = float(np.mean(pred_crop == exp_crop))
                coverage = (min_h * min_w) / max(predicted.size, expected.size)
                scores["pixels"] = overlap_accuracy * coverage
            else:
                scores["pixels"] = 0.0

        # 3. Color palette match
        pred_colors = predicted.unique_colors
        exp_colors = expected.unique_colors
        if exp_colors:
            color_overlap = len(pred_colors & exp_colors) / len(exp_colors)
            extra_colors_penalty = max(0, len(pred_colors) - len(exp_colors)) / 10
            scores["colors"] = max(0, color_overlap - extra_colors_penalty)
        else:
            scores["colors"] = 1.0 if not pred_colors else 0.0

        # 4. Pattern similarity (histogram comparison)
        pred_counts = predicted.color_counts
        exp_counts = expected.color_counts

        total_exp = sum(exp_counts.values())
        total_pred = sum(pred_counts.values())

        if total_exp > 0 and total_pred > 0:
            hist_diff = 0
            for c in range(10):
                exp_ratio = exp_counts.get(c, 0) / total_exp
                pred_ratio = pred_counts.get(c, 0) / total_pred
                hist_diff += abs(exp_ratio - pred_ratio)
            scores["pattern"] = max(0, 1 - hist_diff / 2)
        else:
            scores["pattern"] = 0.5

        # Weighted combination
        weights = {"shape": 0.4, "pixels": 0.4, "colors": 0.1, "pattern": 0.1}
        total_score = sum(scores[k] * weights[k] for k in weights)

        return total_score

    def _generate_mismatch_feedback(
        self,
        predicted: Grid,
        expected: Grid,
        test_idx: int
    ) -> str:
        """Generate detailed feedback about prediction errors."""
        lines = [f"Test {test_idx + 1}: INCORRECT"]

        # Shape mismatch
        if predicted.shape != expected.shape:
            lines.append(
                f"  Shape mismatch: predicted {predicted.shape}, expected {expected.shape}"
            )

            # Suggest what might be wrong
            if predicted.height > expected.height or predicted.width > expected.width:
                lines.append("  -> Output is too large, consider cropping or different transformation")
            elif predicted.height < expected.height or predicted.width < expected.width:
                lines.append("  -> Output is too small, consider padding or scaling")

        # Color analysis
        pred_colors = predicted.unique_colors
        exp_colors = expected.unique_colors

        missing_colors = exp_colors - pred_colors
        extra_colors = pred_colors - exp_colors

        if missing_colors:
            lines.append(f"  Missing colors: {sorted(missing_colors)}")
        if extra_colors:
            lines.append(f"  Extra colors: {sorted(extra_colors)}")

        # Pixel-level analysis (if same shape)
        if predicted.shape == expected.shape:
            diff_mask = predicted.data != expected.data
            num_wrong = int(np.sum(diff_mask))
            total = predicted.size
            accuracy = 1 - num_wrong / total

            lines.append(f"  Pixel accuracy: {accuracy:.1%} ({num_wrong}/{total} wrong)")

            if num_wrong > 0 and num_wrong <= 10:
                # Show specific wrong pixels
                wrong_positions = list(zip(*np.where(diff_mask)))[:5]
                for r, c in wrong_positions:
                    lines.append(
                        f"    ({r},{c}): predicted {predicted[r,c]}, expected {expected[r,c]}"
                    )

            # Pattern analysis
            if num_wrong > total * 0.5:
                lines.append("  -> More than half the pixels are wrong, fundamental logic error")
            elif num_wrong > total * 0.2:
                lines.append("  -> Significant errors, check transformation logic")
            else:
                lines.append("  -> Close! Minor adjustments needed")

        return "\n".join(lines)

    def validate_on_training(
        self,
        task: ARCTask,
        transform_fn: Callable[[Grid], Grid]
    ) -> Tuple[bool, float, str]:
        """
        Validate a transformation function on training examples.

        Returns (all_correct, avg_score, feedback)
        """
        scores = []
        feedback_parts = []

        for i, pair in enumerate(task.train):
            try:
                prediction = transform_fn(pair.input)
                correct, score, fb = self.evaluate_single(prediction, pair.output)
                scores.append(score)
                feedback_parts.append(f"Train {i+1}: {fb}")
            except Exception as e:
                scores.append(0.0)
                feedback_parts.append(f"Train {i+1}: ERROR - {str(e)[:100]}")

        avg_score = sum(scores) / len(scores) if scores else 0.0
        all_correct = all(s == 1.0 for s in scores)

        return all_correct, avg_score, "\n".join(feedback_parts)


class CodeExecutionEvaluator:
    """
    Evaluate code-based solutions by executing them.

    Executes Python code in a sandbox and tests against training examples.
    """

    def __init__(
        self,
        timeout_seconds: float = 5.0,
        max_output_size: int = 10000,
    ):
        self.timeout_seconds = timeout_seconds
        self.max_output_size = max_output_size
        self.arc_evaluator = ARCEvaluator(allow_partial_credit=True)

    def evaluate_code(
        self,
        code: str,
        task: ARCTask,
        function_name: str = "transform"
    ) -> EvaluationResult:
        """
        Execute code and evaluate against task.

        Args:
            code: Python code defining a transform function
            task: The ARC task to evaluate against
            function_name: Name of the function to call

        Returns:
            EvaluationResult with execution results
        """
        # Parse and validate code
        try:
            # Create sandbox namespace
            namespace = self._create_sandbox()

            # Execute code to define function
            exec(code, namespace)

            if function_name not in namespace:
                # Try to find any suitable function
                candidates = [
                    name for name, obj in namespace.items()
                    if callable(obj) and not name.startswith("_")
                ]
                if candidates:
                    function_name = candidates[0]
                else:
                    return EvaluationResult(
                        correct=False,
                        score=0.0,
                        feedback=f"No callable function found. Define a function named '{function_name}'",
                        predictions=[],
                        expected=[p.output for p in task.test],
                    )

            transform_fn = namespace[function_name]

            # Test on training examples first
            train_correct, train_score, train_fb = self._test_on_training(
                transform_fn, task
            )

            if not train_correct and train_score < 0.5:
                return EvaluationResult(
                    correct=False,
                    score=train_score * 0.5,  # Penalize for failing training
                    feedback=f"Failed on training examples:\n{train_fb}",
                    predictions=[],
                    expected=[p.output for p in task.test],
                    metadata={"training_score": train_score}
                )

            # Run on test inputs
            predictions = []
            for test_pair in task.test:
                try:
                    result = transform_fn(test_pair.input.data)
                    if hasattr(result, "tolist"):
                        result = result.tolist()
                    predictions.append(Grid(result))
                except Exception as e:
                    # Return empty grid on error
                    predictions.append(Grid.empty(1, 1))

            # Evaluate predictions
            eval_result = self.arc_evaluator.evaluate(task, predictions)
            eval_result.metadata["training_score"] = train_score
            eval_result.metadata["training_feedback"] = train_fb

            return eval_result

        except SyntaxError as e:
            return EvaluationResult(
                correct=False,
                score=0.0,
                feedback=f"Syntax error in code: {e}",
                predictions=[],
                expected=[p.output for p in task.test],
            )
        except Exception as e:
            return EvaluationResult(
                correct=False,
                score=0.0,
                feedback=f"Execution error: {e}",
                predictions=[],
                expected=[p.output for p in task.test],
            )

    def _create_sandbox(self) -> Dict[str, Any]:
        """Create a restricted execution namespace."""
        import numpy as np

        # Allowed builtins
        safe_builtins = {
            "abs": abs,
            "all": all,
            "any": any,
            "bool": bool,
            "dict": dict,
            "enumerate": enumerate,
            "filter": filter,
            "float": float,
            "frozenset": frozenset,
            "int": int,
            "len": len,
            "list": list,
            "map": map,
            "max": max,
            "min": min,
            "print": print,
            "range": range,
            "reversed": reversed,
            "round": round,
            "set": set,
            "slice": slice,
            "sorted": sorted,
            "str": str,
            "sum": sum,
            "tuple": tuple,
            "zip": zip,
            "True": True,
            "False": False,
            "None": None,
        }

        return {
            "__builtins__": safe_builtins,
            "np": np,
            "numpy": np,
        }

    def _test_on_training(
        self,
        transform_fn: Callable,
        task: ARCTask
    ) -> Tuple[bool, float, str]:
        """Test transform function on training examples."""
        scores = []
        feedback_parts = []

        for i, pair in enumerate(task.train):
            try:
                result = transform_fn(pair.input.data)
                if hasattr(result, "tolist"):
                    result = result.tolist()
                prediction = Grid(result)

                correct, score, fb = self.arc_evaluator.evaluate_single(
                    prediction, pair.output
                )
                scores.append(score)
                feedback_parts.append(f"Train {i+1}: {'PASS' if correct else 'FAIL'} ({score:.2f})")

            except Exception as e:
                scores.append(0.0)
                feedback_parts.append(f"Train {i+1}: ERROR - {str(e)[:50]}")

        avg_score = sum(scores) / len(scores) if scores else 0.0
        all_correct = len(scores) > 0 and all(s == 1.0 for s in scores)

        return all_correct, avg_score, "\n".join(feedback_parts)


def parse_grid_from_response(response: str) -> Optional[Grid]:
    """
    Parse a grid from LLM response text.

    Handles various formats:
    - JSON array
    - Code block with array
    - Plain text grid
    """
    # Try JSON array
    json_patterns = [
        r"```json\s*([\[\{].*?[\]\}])\s*```",
        r"```\s*([\[\{].*?[\]\}])\s*```",
        r"(\[\[.*?\]\])",
    ]

    for pattern in json_patterns:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
                if isinstance(data, list) and len(data) > 0:
                    if isinstance(data[0], list):
                        return Grid(data)
                    elif isinstance(data[0], int):
                        # Single row
                        return Grid([data])
            except json.JSONDecodeError:
                continue

    # Try Python literal
    python_patterns = [
        r"```python\s*.*?=\s*(\[\[.*?\]\])\s*```",
        r"output\s*=\s*(\[\[.*?\]\])",
    ]

    for pattern in python_patterns:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            try:
                import ast
                data = ast.literal_eval(match.group(1))
                if isinstance(data, list):
                    return Grid(data)
            except (ValueError, SyntaxError):
                continue

    # Try plain text grid (digits separated by spaces/commas)
    lines = response.strip().split("\n")
    grid_lines = []

    for line in lines:
        # Extract digits
        digits = re.findall(r"\d", line)
        if digits and len(digits) > 0:
            grid_lines.append([int(d) for d in digits])

    if grid_lines:
        # Validate all rows have same length
        max_len = max(len(row) for row in grid_lines)
        if all(len(row) == max_len for row in grid_lines):
            return Grid(grid_lines)

    return None
