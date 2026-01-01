"""
In-Context Learning Adapter for GPT-5.2-Codex.

Replaces traditional fine-tuning/TTT with massive context utilization:
- 400K token context = ~50 full ARC tasks worth of examples
- No gradient updates needed
- Dynamic example selection and ordering
- Few-shot learning with optimal exemplars

This is the GPT-5.2 Codex equivalent of test-time training.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .grid import Grid
from .dataset import ARCTask
from .synthesis.augmentation import AugmentationPipeline

logger = logging.getLogger(__name__)


@dataclass
class ICLConfig:
    """Configuration for in-context learning."""

    # Example selection
    max_examples_in_context: int = 20  # How many solved tasks to include
    max_augmentations_per_task: int = 4  # Augmented versions of current task

    # Similarity-based retrieval
    use_similarity_retrieval: bool = True
    similarity_threshold: float = 0.5

    # Example ordering
    order_by_similarity: bool = True  # Most similar first
    order_by_difficulty: bool = False  # Easy to hard

    # Format optimization
    include_solution_traces: bool = True  # Include reasoning steps
    include_code_solutions: bool = True  # Include working code

    # Context budget (tokens)
    max_context_tokens: int = 150000  # Leave room for response


@dataclass
class SolvedExample:
    """A previously solved ARC task for ICL."""
    task_id: str
    examples: List[Tuple[NDArray, NDArray]]  # (input, output) pairs
    solution_code: str
    reasoning_trace: Optional[str] = None
    difficulty_score: float = 0.5
    concepts: List[str] = field(default_factory=list)


class InContextAdapter:
    """
    In-context learning adapter for GPT-5.2-Codex.

    Instead of fine-tuning, we:
    1. Build a library of solved examples
    2. Retrieve similar examples for new tasks
    3. Augment current task for implicit "training"
    4. Format everything into optimal context
    """

    def __init__(self, config: Optional[ICLConfig] = None):
        self.config = config or ICLConfig()
        self.solved_examples: List[SolvedExample] = []
        self.augmentation = AugmentationPipeline()

    def add_solved_example(
        self,
        task: ARCTask,
        solution_code: str,
        reasoning_trace: Optional[str] = None,
    ) -> None:
        """Add a successfully solved task to the example library."""
        examples = [(p.input.data, p.output.data) for p in task.train]

        # Detect concepts
        concepts = self._detect_concepts(examples)

        # Estimate difficulty
        difficulty = self._estimate_difficulty(examples)

        self.solved_examples.append(SolvedExample(
            task_id=task.id,
            examples=examples,
            solution_code=solution_code,
            reasoning_trace=reasoning_trace,
            difficulty_score=difficulty,
            concepts=concepts,
        ))

        logger.debug(f"Added solved example {task.id} with concepts: {concepts}")

    def build_context(self, task: ARCTask) -> str:
        """
        Build optimal context for solving a new task.

        Returns formatted context string with:
        1. Similar solved examples (with code)
        2. Augmented versions of current task
        3. Concept explanations
        """
        context_parts = []

        # 1. Add similar solved examples
        if self.solved_examples:
            similar = self._retrieve_similar(task)
            if similar:
                context_parts.append(self._format_solved_examples(similar))

        # 2. Add augmented task examples
        augmented_context = self._build_augmented_context(task)
        if augmented_context:
            context_parts.append(augmented_context)

        # 3. Add concept hints
        concepts = self._detect_concepts_for_task(task)
        if concepts:
            context_parts.append(self._format_concept_hints(concepts))

        return "\n\n".join(context_parts)

    def _retrieve_similar(
        self,
        task: ARCTask,
        top_k: int = 5,
    ) -> List[SolvedExample]:
        """Retrieve most similar solved examples."""
        if not self.solved_examples:
            return []

        # Compute similarity scores
        task_features = self._extract_features(task)

        scored = []
        for example in self.solved_examples:
            example_features = self._features_from_examples(example.examples)
            similarity = self._compute_similarity(task_features, example_features)

            if similarity >= self.config.similarity_threshold:
                scored.append((similarity, example))

        # Sort by similarity
        scored.sort(reverse=True, key=lambda x: x[0])

        return [ex for _, ex in scored[:top_k]]

    def _extract_features(self, task: ARCTask) -> Dict[str, Any]:
        """Extract features for similarity matching."""
        features = {
            "num_examples": len(task.train),
            "input_sizes": [],
            "output_sizes": [],
            "colors_used": set(),
            "shape_changes": [],
            "size_ratios": [],
        }

        for pair in task.train:
            inp, out = pair.input.data, pair.output.data

            features["input_sizes"].append(inp.shape)
            features["output_sizes"].append(out.shape)
            features["colors_used"].update(np.unique(inp).tolist())
            features["colors_used"].update(np.unique(out).tolist())

            # Shape change type
            if inp.shape == out.shape:
                features["shape_changes"].append("same")
            elif inp.shape[0] == out.shape[0]:
                features["shape_changes"].append("width_change")
            elif inp.shape[1] == out.shape[1]:
                features["shape_changes"].append("height_change")
            else:
                features["shape_changes"].append("both_change")

            # Size ratio
            ratio = out.size / max(inp.size, 1)
            features["size_ratios"].append(ratio)

        return features

    def _features_from_examples(
        self,
        examples: List[Tuple[NDArray, NDArray]]
    ) -> Dict[str, Any]:
        """Extract features from example list."""
        features = {
            "num_examples": len(examples),
            "input_sizes": [],
            "output_sizes": [],
            "colors_used": set(),
            "shape_changes": [],
            "size_ratios": [],
        }

        for inp, out in examples:
            features["input_sizes"].append(inp.shape)
            features["output_sizes"].append(out.shape)
            features["colors_used"].update(np.unique(inp).tolist())
            features["colors_used"].update(np.unique(out).tolist())

            if inp.shape == out.shape:
                features["shape_changes"].append("same")
            elif inp.shape[0] == out.shape[0]:
                features["shape_changes"].append("width_change")
            elif inp.shape[1] == out.shape[1]:
                features["shape_changes"].append("height_change")
            else:
                features["shape_changes"].append("both_change")

            ratio = out.size / max(inp.size, 1)
            features["size_ratios"].append(ratio)

        return features

    def _compute_similarity(
        self,
        f1: Dict[str, Any],
        f2: Dict[str, Any]
    ) -> float:
        """Compute similarity between two feature sets."""
        score = 0.0

        # Color overlap (Jaccard)
        c1, c2 = f1["colors_used"], f2["colors_used"]
        if c1 or c2:
            color_sim = len(c1 & c2) / len(c1 | c2)
            score += color_sim * 0.3

        # Shape change pattern match
        sc1, sc2 = f1["shape_changes"], f2["shape_changes"]
        if sc1 and sc2:
            matches = sum(1 for a, b in zip(sc1, sc2) if a == b)
            shape_sim = matches / max(len(sc1), len(sc2))
            score += shape_sim * 0.4

        # Size ratio similarity
        r1, r2 = f1["size_ratios"], f2["size_ratios"]
        if r1 and r2:
            avg1, avg2 = np.mean(r1), np.mean(r2)
            ratio_sim = 1.0 / (1.0 + abs(avg1 - avg2))
            score += ratio_sim * 0.3

        return score

    def _build_augmented_context(self, task: ARCTask) -> str:
        """Build context with augmented task examples."""
        base_examples = [(p.input.data, p.output.data) for p in task.train]

        # Generate augmentations
        augmented = self.augmentation.augment_batch(
            base_examples,
            multiplier=self.config.max_augmentations_per_task
        )

        # Format for context
        lines = ["## Additional Example Variations (same transformation rule):"]

        for i, (inp, out) in enumerate(augmented[:self.config.max_augmentations_per_task]):
            lines.append(f"\nVariation {i + 1}:")
            lines.append("Input:")
            lines.append(self._grid_to_str(inp))
            lines.append("Output:")
            lines.append(self._grid_to_str(out))

        return "\n".join(lines)

    def _format_solved_examples(self, examples: List[SolvedExample]) -> str:
        """Format solved examples for context."""
        lines = ["## Previously Solved Similar Tasks:"]
        lines.append("(Study these solutions for patterns that may apply)")

        for i, ex in enumerate(examples):
            lines.append(f"\n### Solved Task {i + 1}")

            # Show 1-2 examples
            for j, (inp, out) in enumerate(ex.examples[:2]):
                lines.append(f"Example {j + 1}:")
                lines.append("Input:")
                lines.append(self._grid_to_str(inp))
                lines.append("Output:")
                lines.append(self._grid_to_str(out))

            # Show solution code
            if self.config.include_code_solutions:
                lines.append("Solution:")
                lines.append("```python")
                lines.append(ex.solution_code)
                lines.append("```")

            # Show reasoning trace
            if self.config.include_solution_traces and ex.reasoning_trace:
                lines.append(f"Reasoning: {ex.reasoning_trace[:500]}...")

        return "\n".join(lines)

    def _format_concept_hints(self, concepts: List[str]) -> str:
        """Format concept hints for context."""
        if not concepts:
            return ""

        lines = ["## Detected Concepts:"]
        for concept in concepts:
            hint = self._get_concept_hint(concept)
            lines.append(f"- **{concept}**: {hint}")

        return "\n".join(lines)

    def _detect_concepts(
        self,
        examples: List[Tuple[NDArray, NDArray]]
    ) -> List[str]:
        """Detect transformation concepts in examples."""
        concepts = []

        for inp, out in examples:
            # Rotation detection
            for k in [1, 2, 3]:
                if np.array_equal(np.rot90(inp, k), out):
                    concepts.append(f"rotate_{k*90}")
                    break

            # Flip detection
            if np.array_equal(np.fliplr(inp), out):
                concepts.append("flip_horizontal")
            if np.array_equal(np.flipud(inp), out):
                concepts.append("flip_vertical")

            # Scaling detection
            if inp.shape != out.shape:
                h_ratio = out.shape[0] / inp.shape[0]
                w_ratio = out.shape[1] / inp.shape[1]
                if h_ratio == w_ratio and h_ratio == int(h_ratio):
                    concepts.append(f"scale_{int(h_ratio)}x")
                elif h_ratio > 1 or w_ratio > 1:
                    concepts.append("upscale")
                else:
                    concepts.append("downscale")

            # Color change detection
            in_colors = set(np.unique(inp))
            out_colors = set(np.unique(out))
            if in_colors != out_colors:
                if len(out_colors) < len(in_colors):
                    concepts.append("color_reduction")
                elif len(out_colors) > len(in_colors):
                    concepts.append("color_addition")
                else:
                    concepts.append("color_swap")

        return list(set(concepts))

    def _detect_concepts_for_task(self, task: ARCTask) -> List[str]:
        """Detect concepts for a task."""
        examples = [(p.input.data, p.output.data) for p in task.train]
        return self._detect_concepts(examples)

    def _estimate_difficulty(
        self,
        examples: List[Tuple[NDArray, NDArray]]
    ) -> float:
        """Estimate task difficulty 0-1."""
        scores = []

        for inp, out in examples:
            # Size complexity
            size_score = min(1.0, (inp.size + out.size) / 200)
            scores.append(size_score)

            # Color complexity
            colors = len(set(np.unique(inp)) | set(np.unique(out)))
            color_score = colors / 10
            scores.append(color_score)

            # Shape change complexity
            if inp.shape != out.shape:
                scores.append(0.7)
            else:
                # Pixel difference complexity
                diff = np.sum(inp != out) / inp.size
                scores.append(diff)

        return np.mean(scores) if scores else 0.5

    def _get_concept_hint(self, concept: str) -> str:
        """Get hint text for a concept."""
        hints = {
            "rotate_90": "Grid is rotated 90 degrees clockwise",
            "rotate_180": "Grid is rotated 180 degrees",
            "rotate_270": "Grid is rotated 270 degrees clockwise",
            "flip_horizontal": "Grid is flipped left-to-right",
            "flip_vertical": "Grid is flipped top-to-bottom",
            "scale_2x": "Each cell becomes 2x2 block",
            "scale_3x": "Each cell becomes 3x3 block",
            "upscale": "Output is larger than input",
            "downscale": "Output is smaller than input",
            "color_reduction": "Some colors are removed or merged",
            "color_addition": "New colors appear in output",
            "color_swap": "Colors are exchanged/remapped",
        }
        return hints.get(concept, "Pattern transformation")

    def _grid_to_str(self, grid: NDArray) -> str:
        """Convert grid to string."""
        return "\n".join(" ".join(str(int(x)) for x in row) for row in grid)

    def get_context_size_estimate(self, task: ARCTask) -> int:
        """Estimate token count for context."""
        # Rough estimate: 4 chars per token
        context = self.build_context(task)
        return len(context) // 4

    def save_library(self, path: str) -> None:
        """Save solved examples library."""
        import json

        data = []
        for ex in self.solved_examples:
            data.append({
                "task_id": ex.task_id,
                "examples": [
                    {"input": inp.tolist(), "output": out.tolist()}
                    for inp, out in ex.examples
                ],
                "solution_code": ex.solution_code,
                "reasoning_trace": ex.reasoning_trace,
                "difficulty_score": ex.difficulty_score,
                "concepts": ex.concepts,
            })

        with open(path, 'w') as f:
            json.dump(data, f)

    def load_library(self, path: str) -> None:
        """Load solved examples library."""
        import json

        with open(path) as f:
            data = json.load(f)

        self.solved_examples = []
        for item in data:
            examples = [
                (np.array(ex["input"]), np.array(ex["output"]))
                for ex in item["examples"]
            ]
            self.solved_examples.append(SolvedExample(
                task_id=item["task_id"],
                examples=examples,
                solution_code=item["solution_code"],
                reasoning_trace=item.get("reasoning_trace"),
                difficulty_score=item.get("difficulty_score", 0.5),
                concepts=item.get("concepts", []),
            ))
