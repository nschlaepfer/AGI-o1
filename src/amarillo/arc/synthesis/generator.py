"""
Synthetic Task Generator for ARC-AGI.

Based on NVARC's approach:
- Concept decomposition
- Staged puzzle generation
- Curriculum complexity
"""

from __future__ import annotations

import random
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum, auto

import numpy as np
from numpy.typing import NDArray

from ..grid import Grid, GridPair
from ..dataset import ARCTask
from ..dsl.primitives import get_all_primitives

logger = logging.getLogger(__name__)


class ConceptType(Enum):
    """Categories of ARC concepts."""
    GEOMETRIC = auto()  # Rotation, flip, translation
    COLOR = auto()  # Color mapping, filling
    OBJECT = auto()  # Object detection, manipulation
    PATTERN = auto()  # Repetition, symmetry, tiling
    COUNTING = auto()  # Count-based transformations
    CONDITIONAL = auto()  # If-then logic
    COMPOSITE = auto()  # Multi-step transformations


@dataclass
class Concept:
    """A learnable ARC concept."""
    name: str
    category: ConceptType
    generator: Callable[[], Tuple[Grid, Grid]]
    description: str = ""
    difficulty: int = 1  # 1-5
    prerequisite_concepts: List[str] = field(default_factory=list)


class SyntheticGenerator:
    """
    Generate synthetic ARC-like tasks.

    Strategies:
    1. Random grid + random transformation
    2. Concept-based generation
    3. Compositional tasks (combine concepts)
    4. Augmentation of existing tasks
    """

    def __init__(
        self,
        min_grid_size: int = 3,
        max_grid_size: int = 15,
        num_colors: int = 10,
        seed: Optional[int] = None,
    ):
        self.min_grid_size = min_grid_size
        self.max_grid_size = max_grid_size
        self.num_colors = num_colors
        self.rng = np.random.default_rng(seed)
        self._primitives = get_all_primitives()

    def random_grid(
        self,
        height: Optional[int] = None,
        width: Optional[int] = None,
        sparsity: float = 0.5,
        num_colors: Optional[int] = None,
    ) -> Grid:
        """Generate random grid."""
        height = height or self.rng.integers(self.min_grid_size, self.max_grid_size + 1)
        width = width or self.rng.integers(self.min_grid_size, self.max_grid_size + 1)
        num_colors = num_colors or self.rng.integers(2, self.num_colors + 1)

        # Create sparse random grid
        data = np.zeros((height, width), dtype=np.int8)

        num_pixels = int(height * width * (1 - sparsity))
        colors = self.rng.integers(1, num_colors, size=num_pixels)
        positions = self.rng.choice(height * width, size=num_pixels, replace=False)

        for pos, color in zip(positions, colors):
            r, c = pos // width, pos % width
            data[r, c] = color

        return Grid(data)

    def random_shape_grid(
        self,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_shapes: int = 3,
    ) -> Grid:
        """Generate grid with random geometric shapes."""
        height = height or self.rng.integers(8, 15)
        width = width or self.rng.integers(8, 15)

        data = np.zeros((height, width), dtype=np.int8)

        for _ in range(num_shapes):
            shape_type = self.rng.choice(["rectangle", "line", "cross", "dot"])
            color = self.rng.integers(1, 10)

            if shape_type == "rectangle":
                h = self.rng.integers(2, min(5, height - 1))
                w = self.rng.integers(2, min(5, width - 1))
                r = self.rng.integers(0, height - h)
                c = self.rng.integers(0, width - w)
                data[r:r+h, c:c+w] = color

            elif shape_type == "line":
                if self.rng.random() < 0.5:  # Horizontal
                    r = self.rng.integers(0, height)
                    c1 = self.rng.integers(0, width - 2)
                    c2 = self.rng.integers(c1 + 1, width)
                    data[r, c1:c2] = color
                else:  # Vertical
                    c = self.rng.integers(0, width)
                    r1 = self.rng.integers(0, height - 2)
                    r2 = self.rng.integers(r1 + 1, height)
                    data[r1:r2, c] = color

            elif shape_type == "cross":
                r = self.rng.integers(1, height - 1)
                c = self.rng.integers(1, width - 1)
                data[r, c] = color
                data[r-1, c] = color
                data[r+1, c] = color
                data[r, c-1] = color
                data[r, c+1] = color

            elif shape_type == "dot":
                r = self.rng.integers(0, height)
                c = self.rng.integers(0, width)
                data[r, c] = color

        return Grid(data)

    def generate_task(
        self,
        num_train: int = 3,
        num_test: int = 1,
        transformation: Optional[Callable[[Grid], Grid]] = None,
        difficulty: int = 1,
    ) -> ARCTask:
        """
        Generate a synthetic ARC task.

        Args:
            num_train: Number of training examples
            num_test: Number of test examples
            transformation: Optional specific transformation to use
            difficulty: Difficulty level (1-5)
        """
        if transformation is None:
            transformation = self._random_transformation(difficulty)

        train_pairs = []
        for _ in range(num_train):
            input_grid = self._generate_input_for_difficulty(difficulty)
            try:
                output_grid = transformation(input_grid)
                train_pairs.append(GridPair(input_grid, output_grid))
            except Exception:
                # Retry with new input
                continue

        # Ensure we have enough training examples
        while len(train_pairs) < num_train:
            input_grid = self._generate_input_for_difficulty(difficulty)
            try:
                output_grid = transformation(input_grid)
                train_pairs.append(GridPair(input_grid, output_grid))
            except Exception:
                continue

        test_pairs = []
        for _ in range(num_test):
            input_grid = self._generate_input_for_difficulty(difficulty)
            try:
                output_grid = transformation(input_grid)
                test_pairs.append(GridPair(input_grid, output_grid))
            except Exception:
                continue

        task_id = f"synth_{self.rng.integers(0, 1000000):06d}"

        return ARCTask(
            id=task_id,
            train=train_pairs,
            test=test_pairs,
            metadata={
                "synthetic": True,
                "difficulty": difficulty,
            }
        )

    def _generate_input_for_difficulty(self, difficulty: int) -> Grid:
        """Generate input grid appropriate for difficulty level."""
        if difficulty <= 2:
            # Simple: small grids, few shapes
            return self.random_shape_grid(
                height=self.rng.integers(4, 8),
                width=self.rng.integers(4, 8),
                num_shapes=self.rng.integers(1, 3)
            )
        elif difficulty <= 4:
            # Medium: larger grids, more shapes
            return self.random_shape_grid(
                height=self.rng.integers(6, 12),
                width=self.rng.integers(6, 12),
                num_shapes=self.rng.integers(2, 5)
            )
        else:
            # Hard: large grids, many shapes, patterns
            return self.random_shape_grid(
                height=self.rng.integers(10, 15),
                width=self.rng.integers(10, 15),
                num_shapes=self.rng.integers(3, 7)
            )

    def _random_transformation(self, difficulty: int) -> Callable[[Grid], Grid]:
        """Get random transformation for difficulty level."""
        simple_transforms = [
            lambda g: Grid(np.rot90(g.data, 1)),
            lambda g: Grid(np.rot90(g.data, 2)),
            lambda g: Grid(np.fliplr(g.data)),
            lambda g: Grid(np.flipud(g.data)),
            lambda g: Grid(g.data.T),
        ]

        medium_transforms = [
            lambda g: Grid(np.tile(g.data, (2, 1))),
            lambda g: Grid(np.tile(g.data, (1, 2))),
            lambda g: Grid(np.concatenate([g.data, np.fliplr(g.data)], axis=1)),
            lambda g: Grid(np.concatenate([g.data, np.flipud(g.data)], axis=0)),
            self._scale_transform(2),
        ]

        hard_transforms = [
            self._color_swap_transform(),
            self._crop_largest_transform(),
            self._compose_transforms([
                lambda g: Grid(np.rot90(g.data, 1)),
                lambda g: Grid(np.fliplr(g.data)),
            ]),
        ]

        if difficulty <= 2:
            return self.rng.choice(simple_transforms)
        elif difficulty <= 4:
            all_transforms = simple_transforms + medium_transforms
            return self.rng.choice(all_transforms)
        else:
            all_transforms = simple_transforms + medium_transforms + hard_transforms
            return self.rng.choice(all_transforms)

    def _scale_transform(self, factor: int) -> Callable[[Grid], Grid]:
        def transform(g: Grid) -> Grid:
            return Grid(np.kron(g.data, np.ones((factor, factor), dtype=np.int8)))
        return transform

    def _color_swap_transform(self) -> Callable[[Grid], Grid]:
        c1, c2 = self.rng.choice(range(1, 10), size=2, replace=False)
        def transform(g: Grid) -> Grid:
            data = g.data.copy()
            mask1 = g.data == c1
            mask2 = g.data == c2
            data[mask1] = c2
            data[mask2] = c1
            return Grid(data)
        return transform

    def _crop_largest_transform(self) -> Callable[[Grid], Grid]:
        def transform(g: Grid) -> Grid:
            # Crop to bounding box of non-zero pixels
            mask = g.data != 0
            if not mask.any():
                return g
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            return Grid(g.data[rows][:, cols])
        return transform

    def _compose_transforms(self, transforms: List[Callable]) -> Callable[[Grid], Grid]:
        def composed(g: Grid) -> Grid:
            result = g
            for t in transforms:
                result = t(result)
            return result
        return composed

    def generate_batch(
        self,
        num_tasks: int,
        difficulty_range: Tuple[int, int] = (1, 5),
    ) -> List[ARCTask]:
        """Generate batch of synthetic tasks."""
        tasks = []
        for _ in range(num_tasks):
            difficulty = self.rng.integers(difficulty_range[0], difficulty_range[1] + 1)
            task = self.generate_task(difficulty=difficulty)
            tasks.append(task)
        return tasks


class ConceptGenerator:
    """
    Generate tasks based on specific concepts.

    Concepts are fundamental ARC primitives that can be combined
    to create more complex tasks.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        self._concepts: Dict[str, Concept] = {}
        self._register_concepts()

    def _register_concepts(self) -> None:
        """Register all available concepts."""

        # Geometric concepts
        self._concepts["rotate_90"] = Concept(
            name="rotate_90",
            category=ConceptType.GEOMETRIC,
            generator=self._make_rotate_90_task,
            description="Rotate grid 90 degrees clockwise",
            difficulty=1,
        )

        self._concepts["rotate_180"] = Concept(
            name="rotate_180",
            category=ConceptType.GEOMETRIC,
            generator=self._make_rotate_180_task,
            description="Rotate grid 180 degrees",
            difficulty=1,
        )

        self._concepts["flip_horizontal"] = Concept(
            name="flip_horizontal",
            category=ConceptType.GEOMETRIC,
            generator=self._make_flip_h_task,
            description="Flip grid horizontally",
            difficulty=1,
        )

        self._concepts["flip_vertical"] = Concept(
            name="flip_vertical",
            category=ConceptType.GEOMETRIC,
            generator=self._make_flip_v_task,
            description="Flip grid vertically",
            difficulty=1,
        )

        # Pattern concepts
        self._concepts["mirror_horizontal"] = Concept(
            name="mirror_horizontal",
            category=ConceptType.PATTERN,
            generator=self._make_mirror_h_task,
            description="Mirror grid horizontally",
            difficulty=2,
        )

        self._concepts["tile_2x2"] = Concept(
            name="tile_2x2",
            category=ConceptType.PATTERN,
            generator=self._make_tile_task,
            description="Tile grid 2x2",
            difficulty=2,
        )

        self._concepts["scale_2x"] = Concept(
            name="scale_2x",
            category=ConceptType.PATTERN,
            generator=self._make_scale_task,
            description="Scale grid 2x",
            difficulty=2,
        )

        # Color concepts
        self._concepts["recolor"] = Concept(
            name="recolor",
            category=ConceptType.COLOR,
            generator=self._make_recolor_task,
            description="Replace one color with another",
            difficulty=1,
        )

        self._concepts["swap_colors"] = Concept(
            name="swap_colors",
            category=ConceptType.COLOR,
            generator=self._make_swap_colors_task,
            description="Swap two colors",
            difficulty=2,
        )

        # Object concepts
        self._concepts["crop_to_content"] = Concept(
            name="crop_to_content",
            category=ConceptType.OBJECT,
            generator=self._make_crop_content_task,
            description="Crop to bounding box of content",
            difficulty=2,
        )

        self._concepts["extract_object"] = Concept(
            name="extract_object",
            category=ConceptType.OBJECT,
            generator=self._make_extract_object_task,
            description="Extract largest connected component",
            difficulty=3,
        )

    def get_concept(self, name: str) -> Optional[Concept]:
        return self._concepts.get(name)

    def list_concepts(self, category: Optional[ConceptType] = None) -> List[str]:
        if category:
            return [n for n, c in self._concepts.items() if c.category == category]
        return list(self._concepts.keys())

    def generate_concept_task(
        self,
        concept_name: str,
        num_train: int = 3,
        num_test: int = 1,
    ) -> Optional[ARCTask]:
        """Generate task for specific concept."""
        concept = self.get_concept(concept_name)
        if not concept:
            return None

        train_pairs = []
        for _ in range(num_train):
            try:
                input_grid, output_grid = concept.generator()
                train_pairs.append(GridPair(input_grid, output_grid))
            except Exception:
                continue

        test_pairs = []
        for _ in range(num_test):
            try:
                input_grid, output_grid = concept.generator()
                test_pairs.append(GridPair(input_grid, output_grid))
            except Exception:
                continue

        if not train_pairs:
            return None

        return ARCTask(
            id=f"concept_{concept_name}_{self.rng.integers(0, 10000):04d}",
            train=train_pairs,
            test=test_pairs,
            metadata={
                "concept": concept_name,
                "category": concept.category.name,
                "difficulty": concept.difficulty,
            }
        )

    def generate_curriculum(
        self,
        max_difficulty: int = 5,
        tasks_per_concept: int = 5,
    ) -> List[ARCTask]:
        """Generate curriculum ordered by difficulty."""
        tasks = []

        # Sort concepts by difficulty
        sorted_concepts = sorted(
            self._concepts.values(),
            key=lambda c: c.difficulty
        )

        for concept in sorted_concepts:
            if concept.difficulty > max_difficulty:
                break

            for _ in range(tasks_per_concept):
                task = self.generate_concept_task(concept.name)
                if task:
                    tasks.append(task)

        return tasks

    # ========================================================================
    # Concept Task Generators
    # ========================================================================

    def _random_input(self, size_range: Tuple[int, int] = (4, 8)) -> Grid:
        """Generate random input grid."""
        h = self.rng.integers(size_range[0], size_range[1] + 1)
        w = self.rng.integers(size_range[0], size_range[1] + 1)

        data = np.zeros((h, w), dtype=np.int8)

        # Add some colored pixels
        num_pixels = self.rng.integers(3, h * w // 2)
        for _ in range(num_pixels):
            r = self.rng.integers(0, h)
            c = self.rng.integers(0, w)
            color = self.rng.integers(1, 10)
            data[r, c] = color

        return Grid(data)

    def _make_rotate_90_task(self) -> Tuple[Grid, Grid]:
        input_grid = self._random_input()
        output_grid = Grid(np.rot90(input_grid.data, -1))
        return input_grid, output_grid

    def _make_rotate_180_task(self) -> Tuple[Grid, Grid]:
        input_grid = self._random_input()
        output_grid = Grid(np.rot90(input_grid.data, 2))
        return input_grid, output_grid

    def _make_flip_h_task(self) -> Tuple[Grid, Grid]:
        input_grid = self._random_input()
        output_grid = Grid(np.fliplr(input_grid.data))
        return input_grid, output_grid

    def _make_flip_v_task(self) -> Tuple[Grid, Grid]:
        input_grid = self._random_input()
        output_grid = Grid(np.flipud(input_grid.data))
        return input_grid, output_grid

    def _make_mirror_h_task(self) -> Tuple[Grid, Grid]:
        input_grid = self._random_input((3, 6))
        mirrored = np.concatenate([input_grid.data, np.fliplr(input_grid.data)], axis=1)
        output_grid = Grid(mirrored)
        return input_grid, output_grid

    def _make_tile_task(self) -> Tuple[Grid, Grid]:
        input_grid = self._random_input((3, 5))
        tiled = np.tile(input_grid.data, (2, 2))
        output_grid = Grid(tiled)
        return input_grid, output_grid

    def _make_scale_task(self) -> Tuple[Grid, Grid]:
        input_grid = self._random_input((3, 6))
        scaled = np.kron(input_grid.data, np.ones((2, 2), dtype=np.int8))
        output_grid = Grid(scaled)
        return input_grid, output_grid

    def _make_recolor_task(self) -> Tuple[Grid, Grid]:
        input_grid = self._random_input()
        colors = list(input_grid.unique_colors - {0})
        if not colors:
            colors = [1]
        old_color = int(self.rng.choice(colors))
        new_color = self.rng.integers(1, 10)
        while new_color == old_color:
            new_color = self.rng.integers(1, 10)

        data = input_grid.data.copy()
        data[data == old_color] = new_color
        output_grid = Grid(data)
        return input_grid, output_grid

    def _make_swap_colors_task(self) -> Tuple[Grid, Grid]:
        input_grid = self._random_input()
        colors = list(input_grid.unique_colors - {0})
        if len(colors) < 2:
            # Add another color
            data = input_grid.data.copy()
            data[0, 0] = 1
            data[1, 1] = 2
            input_grid = Grid(data)
            colors = [1, 2]

        c1, c2 = self.rng.choice(colors, size=2, replace=False)

        data = input_grid.data.copy()
        mask1 = input_grid.data == c1
        mask2 = input_grid.data == c2
        data[mask1] = c2
        data[mask2] = c1
        output_grid = Grid(data)
        return input_grid, output_grid

    def _make_crop_content_task(self) -> Tuple[Grid, Grid]:
        # Create grid with content in center
        h, w = self.rng.integers(8, 12), self.rng.integers(8, 12)
        data = np.zeros((h, w), dtype=np.int8)

        # Put content in a smaller region
        ch, cw = self.rng.integers(2, 5), self.rng.integers(2, 5)
        cr, cc = self.rng.integers(1, h - ch - 1), self.rng.integers(1, w - cw - 1)

        for r in range(cr, cr + ch):
            for c in range(cc, cc + cw):
                if self.rng.random() > 0.3:
                    data[r, c] = self.rng.integers(1, 10)

        input_grid = Grid(data)

        # Crop to content
        mask = data != 0
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        cropped = data[rows][:, cols]
        output_grid = Grid(cropped)

        return input_grid, output_grid

    def _make_extract_object_task(self) -> Tuple[Grid, Grid]:
        from scipy import ndimage

        # Create grid with multiple objects
        h, w = self.rng.integers(8, 12), self.rng.integers(8, 12)
        data = np.zeros((h, w), dtype=np.int8)

        # Create 2-3 separate objects
        num_objects = self.rng.integers(2, 4)
        for _ in range(num_objects):
            oh, ow = self.rng.integers(2, 4), self.rng.integers(2, 4)
            or_, oc = self.rng.integers(0, h - oh), self.rng.integers(0, w - ow)
            color = self.rng.integers(1, 10)
            data[or_:or_+oh, oc:oc+ow] = color

        input_grid = Grid(data)

        # Extract largest object
        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        labeled, num_features = ndimage.label(data != 0, structure)

        if num_features == 0:
            return input_grid, input_grid

        # Find largest
        largest_label = 0
        largest_size = 0
        for i in range(1, num_features + 1):
            size = np.sum(labeled == i)
            if size > largest_size:
                largest_size = size
                largest_label = i

        # Extract
        mask = labeled == largest_label
        rows, cols = np.where(mask)
        min_r, max_r = rows.min(), rows.max()
        min_c, max_c = cols.min(), cols.max()

        cropped = data[min_r:max_r+1, min_c:max_c+1].copy()
        cropped[~mask[min_r:max_r+1, min_c:max_c+1]] = 0
        output_grid = Grid(cropped)

        return input_grid, output_grid
