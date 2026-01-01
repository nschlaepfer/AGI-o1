"""
ARC Dataset Module - Loading and managing ARC-AGI tasks.

Supports loading from:
- Official ARC-AGI GitHub repository
- Kaggle competition files
- Custom JSON files
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple
from urllib.request import urlopen
from urllib.error import URLError
import hashlib

from .grid import Grid, GridPair

logger = logging.getLogger(__name__)


# Official ARC-AGI repository URLs
ARC_REPO_BASE = "https://raw.githubusercontent.com/fchollet/ARC-AGI/master/data"
ARC_TRAINING_URL = f"{ARC_REPO_BASE}/training"
ARC_EVALUATION_URL = f"{ARC_REPO_BASE}/evaluation"


@dataclass
class ARCTask:
    """
    A single ARC-AGI task with training examples and test cases.

    Attributes:
        id: Unique task identifier
        train: List of input-output training pairs
        test: List of test inputs (outputs may be None for competition)
    """
    id: str
    train: List[GridPair]
    test: List[GridPair]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, task_id: str, data: Dict[str, Any]) -> "ARCTask":
        """Create task from ARC JSON format."""
        train_pairs = [
            GridPair.from_json(pair) for pair in data.get("train", [])
        ]

        test_pairs = []
        for pair in data.get("test", []):
            input_grid = Grid.from_json(pair["input"])
            output_grid = Grid.from_json(pair["output"]) if "output" in pair else None
            test_pairs.append(GridPair(input=input_grid, output=output_grid))

        return cls(
            id=task_id,
            train=train_pairs,
            test=test_pairs,
            metadata=data.get("metadata", {})
        )

    def to_json(self) -> Dict[str, Any]:
        """Convert task to ARC JSON format."""
        return {
            "train": [pair.to_json() for pair in self.train],
            "test": [
                {"input": pair.input.to_json(),
                 **({"output": pair.output.to_json()} if pair.output else {})}
                for pair in self.test
            ]
        }

    @property
    def num_train(self) -> int:
        return len(self.train)

    @property
    def num_test(self) -> int:
        return len(self.test)

    def get_input_shapes(self) -> List[Tuple[int, int]]:
        """Get shapes of all input grids."""
        shapes = []
        for pair in self.train:
            shapes.append(pair.input.shape)
        for pair in self.test:
            shapes.append(pair.input.shape)
        return shapes

    def get_output_shapes(self) -> List[Tuple[int, int]]:
        """Get shapes of all output grids (where available)."""
        shapes = []
        for pair in self.train:
            shapes.append(pair.output.shape)
        for pair in self.test:
            if pair.output is not None:
                shapes.append(pair.output.shape)
        return shapes

    def has_consistent_shapes(self) -> bool:
        """Check if all inputs/outputs have same shapes."""
        input_shapes = set(self.get_input_shapes())
        output_shapes = set(self.get_output_shapes())
        return len(input_shapes) == 1 and len(output_shapes) == 1

    def get_all_colors(self) -> set:
        """Get all colors used in task."""
        colors = set()
        for pair in self.train:
            colors.update(pair.input.unique_colors)
            colors.update(pair.output.unique_colors)
        for pair in self.test:
            colors.update(pair.input.unique_colors)
            if pair.output:
                colors.update(pair.output.unique_colors)
        return colors

    def augment_with_rotation(self, k: int) -> "ARCTask":
        """Create augmented task with rotation applied."""
        return ARCTask(
            id=f"{self.id}_rot{k*90}",
            train=[GridPair(p.input.rotate(k), p.output.rotate(k)) for p in self.train],
            test=[GridPair(p.input.rotate(k),
                          p.output.rotate(k) if p.output else None) for p in self.test],
            metadata={**self.metadata, "augmentation": f"rotate_{k*90}"}
        )

    def augment_with_flip(self, horizontal: bool = True) -> "ARCTask":
        """Create augmented task with flip applied."""
        flip_fn = lambda g: g.flip_horizontal() if horizontal else g.flip_vertical()
        flip_name = "flip_h" if horizontal else "flip_v"

        return ARCTask(
            id=f"{self.id}_{flip_name}",
            train=[GridPair(flip_fn(p.input), flip_fn(p.output)) for p in self.train],
            test=[GridPair(flip_fn(p.input),
                          flip_fn(p.output) if p.output else None) for p in self.test],
            metadata={**self.metadata, "augmentation": flip_name}
        )

    def get_all_augmentations(self) -> List["ARCTask"]:
        """Generate all 8 geometric augmentations."""
        augmented = [self]

        # Rotations
        for k in [1, 2, 3]:
            augmented.append(self.augment_with_rotation(k))

        # Flips
        augmented.append(self.augment_with_flip(horizontal=True))
        augmented.append(self.augment_with_flip(horizontal=False))

        # Rotation + flip combinations
        rotated_90 = self.augment_with_rotation(1)
        augmented.append(rotated_90.augment_with_flip(horizontal=True))
        augmented.append(rotated_90.augment_with_flip(horizontal=False))

        return augmented

    def format_for_prompt(self, include_test_output: bool = False) -> str:
        """Format task for LLM prompt."""
        lines = [f"Task ID: {self.id}", "", "Training Examples:"]

        for i, pair in enumerate(self.train, 1):
            lines.append(f"\nExample {i}:")
            lines.append("Input:")
            lines.append(pair.input.to_string())
            lines.append("Output:")
            lines.append(pair.output.to_string())

        lines.append("\nTest Input(s):")
        for i, pair in enumerate(self.test, 1):
            lines.append(f"\nTest {i}:")
            lines.append(pair.input.to_string())
            if include_test_output and pair.output:
                lines.append("Expected Output:")
                lines.append(pair.output.to_string())

        return "\n".join(lines)

    def format_as_json_prompt(self) -> str:
        """Format task as JSON for LLM prompt (more precise)."""
        data = {
            "train": [
                {"input": p.input.to_json(), "output": p.output.to_json()}
                for p in self.train
            ],
            "test": [{"input": p.input.to_json()} for p in self.test]
        }
        return json.dumps(data, indent=2)


class ARCDataset:
    """
    Manager for ARC-AGI task datasets.

    Supports loading from multiple sources and provides iteration,
    filtering, and caching capabilities.
    """

    def __init__(self, tasks: Optional[List[ARCTask]] = None):
        self._tasks: Dict[str, ARCTask] = {}
        if tasks:
            for task in tasks:
                self._tasks[task.id] = task

    def __len__(self) -> int:
        return len(self._tasks)

    def __iter__(self) -> Iterator[ARCTask]:
        return iter(self._tasks.values())

    def __getitem__(self, task_id: str) -> ARCTask:
        return self._tasks[task_id]

    def __contains__(self, task_id: str) -> bool:
        return task_id in self._tasks

    @property
    def task_ids(self) -> List[str]:
        return list(self._tasks.keys())

    def add_task(self, task: ARCTask) -> None:
        self._tasks[task.id] = task

    def get_task(self, task_id: str) -> Optional[ARCTask]:
        return self._tasks.get(task_id)

    def filter(self, predicate) -> "ARCDataset":
        """Create filtered dataset."""
        filtered_tasks = [t for t in self._tasks.values() if predicate(t)]
        return ARCDataset(filtered_tasks)

    def sample(self, n: int, seed: Optional[int] = None) -> "ARCDataset":
        """Random sample of tasks."""
        import random
        rng = random.Random(seed)
        sampled = rng.sample(list(self._tasks.values()), min(n, len(self._tasks)))
        return ARCDataset(sampled)

    @classmethod
    def from_directory(cls, path: Path | str) -> "ARCDataset":
        """Load all tasks from a directory of JSON files."""
        path = Path(path)
        dataset = cls()

        for json_file in path.glob("*.json"):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                task_id = json_file.stem
                task = ARCTask.from_json(task_id, data)
                dataset.add_task(task)
            except Exception as e:
                logger.warning(f"Failed to load {json_file}: {e}")

        logger.info(f"Loaded {len(dataset)} tasks from {path}")
        return dataset

    @classmethod
    def from_json_file(cls, path: Path | str) -> "ARCDataset":
        """Load dataset from single JSON file (Kaggle format)."""
        path = Path(path)
        dataset = cls()

        with open(path) as f:
            data = json.load(f)

        # Handle different formats
        if isinstance(data, dict):
            for task_id, task_data in data.items():
                task = ARCTask.from_json(task_id, task_data)
                dataset.add_task(task)
        elif isinstance(data, list):
            for i, task_data in enumerate(data):
                task_id = task_data.get("id", f"task_{i}")
                task = ARCTask.from_json(task_id, task_data)
                dataset.add_task(task)

        logger.info(f"Loaded {len(dataset)} tasks from {path}")
        return dataset

    @classmethod
    def download_training(cls, cache_dir: Optional[Path] = None) -> "ARCDataset":
        """Download official ARC-AGI training set."""
        return cls._download_from_github("training", cache_dir)

    @classmethod
    def download_evaluation(cls, cache_dir: Optional[Path] = None) -> "ARCDataset":
        """Download official ARC-AGI evaluation set."""
        return cls._download_from_github("evaluation", cache_dir)

    @classmethod
    def _download_from_github(cls, split: str, cache_dir: Optional[Path] = None) -> "ARCDataset":
        """Download tasks from GitHub repository."""
        # Try to use cached data first
        if cache_dir:
            cache_path = Path(cache_dir) / f"arc_{split}"
            if cache_path.exists():
                return cls.from_directory(cache_path)

        # Download index first (need to know task IDs)
        # For simplicity, we'll load a known list of task IDs
        # In production, you'd fetch the directory listing

        logger.info(f"Downloading ARC {split} set from GitHub...")

        # This is a simplified version - in practice you'd list the directory
        # For now, we'll try to load from local cache or raise an error
        raise NotImplementedError(
            f"Direct GitHub download not implemented. "
            f"Please download the ARC-AGI dataset manually from "
            f"https://github.com/fchollet/ARC-AGI and use from_directory()"
        )

    def save_to_directory(self, path: Path | str) -> None:
        """Save all tasks to a directory."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        for task in self._tasks.values():
            task_path = path / f"{task.id}.json"
            with open(task_path, "w") as f:
                json.dump(task.to_json(), f, indent=2)

        logger.info(f"Saved {len(self)} tasks to {path}")

    def get_statistics(self) -> Dict[str, Any]:
        """Compute dataset statistics."""
        if not self._tasks:
            return {}

        num_train_examples = []
        num_test_examples = []
        input_heights = []
        input_widths = []
        output_heights = []
        output_widths = []
        num_colors = []

        for task in self._tasks.values():
            num_train_examples.append(task.num_train)
            num_test_examples.append(task.num_test)

            for pair in task.train:
                input_heights.append(pair.input.height)
                input_widths.append(pair.input.width)
                output_heights.append(pair.output.height)
                output_widths.append(pair.output.width)

            num_colors.append(len(task.get_all_colors()))

        return {
            "num_tasks": len(self),
            "train_examples": {
                "mean": sum(num_train_examples) / len(num_train_examples),
                "min": min(num_train_examples),
                "max": max(num_train_examples),
            },
            "test_examples": {
                "mean": sum(num_test_examples) / len(num_test_examples),
                "min": min(num_test_examples),
                "max": max(num_test_examples),
            },
            "input_size": {
                "height_mean": sum(input_heights) / len(input_heights),
                "width_mean": sum(input_widths) / len(input_widths),
                "height_max": max(input_heights),
                "width_max": max(input_widths),
            },
            "output_size": {
                "height_mean": sum(output_heights) / len(output_heights),
                "width_mean": sum(output_widths) / len(output_widths),
                "height_max": max(output_heights),
                "width_max": max(output_widths),
            },
            "colors": {
                "mean": sum(num_colors) / len(num_colors),
                "min": min(num_colors),
                "max": max(num_colors),
            }
        }


def load_kaggle_data(
    challenges_path: Path | str,
    solutions_path: Optional[Path | str] = None
) -> ARCDataset:
    """
    Load Kaggle competition format data.

    Args:
        challenges_path: Path to challenges JSON file
        solutions_path: Optional path to solutions JSON file

    Returns:
        ARCDataset with all tasks
    """
    with open(challenges_path) as f:
        challenges = json.load(f)

    solutions = None
    if solutions_path:
        with open(solutions_path) as f:
            solutions = json.load(f)

    dataset = ARCDataset()

    for task_id, task_data in challenges.items():
        # Add solutions if available
        if solutions and task_id in solutions:
            for i, test_pair in enumerate(task_data.get("test", [])):
                if i < len(solutions[task_id]):
                    test_pair["output"] = solutions[task_id][i]

        task = ARCTask.from_json(task_id, task_data)
        dataset.add_task(task)

    return dataset
