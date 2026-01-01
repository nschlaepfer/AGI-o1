"""
Data Augmentation for ARC Tasks.

Implements augmentation strategies used in winning solutions:
- Geometric augmentations (rotation, flip)
- Color permutations
- Leave-one-out training set augmentation
- Scale variations
"""

from __future__ import annotations

import random
from typing import List, Optional, Tuple, Callable

import numpy as np
from numpy.typing import NDArray

from ..grid import Grid, GridPair
from ..dataset import ARCTask


def rotate_augmentation(grid: Grid, k: int = 1) -> Grid:
    """Rotate grid 90 degrees clockwise, k times."""
    return Grid(np.rot90(grid.data, -k))


def flip_h_augmentation(grid: Grid) -> Grid:
    """Flip horizontally."""
    return Grid(np.fliplr(grid.data))


def flip_v_augmentation(grid: Grid) -> Grid:
    """Flip vertically."""
    return Grid(np.flipud(grid.data))


def transpose_augmentation(grid: Grid) -> Grid:
    """Transpose grid."""
    return Grid(grid.data.T)


def color_permutation(grid: Grid, permutation: List[int]) -> Grid:
    """Apply color permutation (permutation[i] = new color for i)."""
    data = grid.data.copy()
    result = np.zeros_like(data)
    for old_color, new_color in enumerate(permutation):
        result[data == old_color] = new_color
    return Grid(result)


def random_color_permutation(
    grid: Grid,
    preserve_background: bool = True,
    rng: Optional[np.random.Generator] = None
) -> Grid:
    """Apply random color permutation."""
    rng = rng or np.random.default_rng()

    if preserve_background:
        perm = [0] + list(rng.permutation(range(1, 10)))
    else:
        perm = list(rng.permutation(range(10)))

    return color_permutation(grid, perm)


def augment_pair(
    pair: GridPair,
    transform: Callable[[Grid], Grid]
) -> GridPair:
    """Apply same transformation to both input and output."""
    return GridPair(
        input=transform(pair.input),
        output=transform(pair.output) if pair.output else None
    )


def augment_task(
    task: ARCTask,
    transform: Callable[[Grid], Grid],
    suffix: str = ""
) -> ARCTask:
    """Apply transformation to all examples in task."""
    return ARCTask(
        id=f"{task.id}{suffix}",
        train=[augment_pair(p, transform) for p in task.train],
        test=[augment_pair(p, transform) for p in task.test],
        metadata={**task.metadata, "augmentation": suffix}
    )


def generate_augmentations(
    task: ARCTask,
    include_rotations: bool = True,
    include_flips: bool = True,
    include_transpose: bool = False,
    include_color_perms: int = 0,  # Number of random color permutations
    rng: Optional[np.random.Generator] = None,
) -> List[ARCTask]:
    """
    Generate all augmented versions of a task.

    Returns list including original task.
    """
    rng = rng or np.random.default_rng()
    augmented = [task]

    # Rotations (90, 180, 270)
    if include_rotations:
        for k in [1, 2, 3]:
            aug = augment_task(
                task,
                lambda g, k=k: rotate_augmentation(g, k),
                f"_rot{k*90}"
            )
            augmented.append(aug)

    # Flips
    if include_flips:
        augmented.append(augment_task(task, flip_h_augmentation, "_fliph"))
        augmented.append(augment_task(task, flip_v_augmentation, "_flipv"))

    # Transpose
    if include_transpose:
        augmented.append(augment_task(task, transpose_augmentation, "_transpose"))

    # Color permutations
    for i in range(include_color_perms):
        perm = [0] + list(rng.permutation(range(1, 10)))
        aug = augment_task(
            task,
            lambda g, p=perm: color_permutation(g, p),
            f"_colorperm{i}"
        )
        augmented.append(aug)

    return augmented


def leave_one_out_augmentation(task: ARCTask) -> List[ARCTask]:
    """
    Generate leave-one-out training sets for test-time training.

    For each training example, creates a task where that example
    is moved to test and others remain in training.
    """
    augmented = []

    for i in range(len(task.train)):
        new_train = [p for j, p in enumerate(task.train) if j != i]
        new_test = [task.train[i]]

        aug_task = ARCTask(
            id=f"{task.id}_loo{i}",
            train=new_train,
            test=new_test,
            metadata={
                **task.metadata,
                "leave_one_out_index": i,
            }
        )
        augmented.append(aug_task)

    return augmented


def generate_ttt_dataset(
    task: ARCTask,
    num_augmentations: int = 8,
    include_loo: bool = True,
) -> List[Tuple[NDArray, NDArray]]:
    """
    Generate test-time training dataset for a task.

    Returns list of (input, output) array pairs for fine-tuning.
    """
    pairs = []

    # Original training pairs
    for p in task.train:
        pairs.append((p.input.data.copy(), p.output.data.copy()))

    # Geometric augmentations of each pair
    transforms = [
        lambda g: rotate_augmentation(g, 1),
        lambda g: rotate_augmentation(g, 2),
        lambda g: rotate_augmentation(g, 3),
        flip_h_augmentation,
        flip_v_augmentation,
    ]

    for p in task.train:
        for transform in transforms[:num_augmentations]:
            aug_input = transform(p.input)
            aug_output = transform(p.output)
            pairs.append((aug_input.data.copy(), aug_output.data.copy()))

    # Leave-one-out (use training examples as if they were test)
    if include_loo:
        loo_tasks = leave_one_out_augmentation(task)
        for loo_task in loo_tasks:
            # The "test" in LOO is actually from training
            for p in loo_task.test:
                pairs.append((p.input.data.copy(), p.output.data.copy()))

    return pairs


def apply_inverse_augmentation(
    grid: Grid,
    augmentation_id: str
) -> Grid:
    """Apply inverse of an augmentation to restore original orientation."""
    if "_rot90" in augmentation_id:
        return rotate_augmentation(grid, 3)  # Rotate back
    elif "_rot180" in augmentation_id:
        return rotate_augmentation(grid, 2)
    elif "_rot270" in augmentation_id:
        return rotate_augmentation(grid, 1)
    elif "_fliph" in augmentation_id:
        return flip_h_augmentation(grid)  # Flip is its own inverse
    elif "_flipv" in augmentation_id:
        return flip_v_augmentation(grid)
    elif "_transpose" in augmentation_id:
        return transpose_augmentation(grid)
    else:
        return grid


class AugmentationPipeline:
    """
    Configurable augmentation pipeline for training data.

    Supports:
    - Random selection of augmentations
    - Weighted sampling
    - Reproducible augmentation sequences
    """

    def __init__(
        self,
        use_rotations: bool = True,
        use_flips: bool = True,
        use_transpose: bool = False,
        use_color_perms: bool = False,
        color_perm_preserve_bg: bool = True,
        seed: Optional[int] = None,
    ):
        self.use_rotations = use_rotations
        self.use_flips = use_flips
        self.use_transpose = use_transpose
        self.use_color_perms = use_color_perms
        self.color_perm_preserve_bg = color_perm_preserve_bg
        self.rng = np.random.default_rng(seed)

        self._build_transforms()

    def _build_transforms(self) -> None:
        """Build list of available transforms."""
        self.transforms: List[Tuple[str, Callable[[Grid], Grid]]] = [
            ("identity", lambda g: g),
        ]

        if self.use_rotations:
            self.transforms.extend([
                ("rot90", lambda g: rotate_augmentation(g, 1)),
                ("rot180", lambda g: rotate_augmentation(g, 2)),
                ("rot270", lambda g: rotate_augmentation(g, 3)),
            ])

        if self.use_flips:
            self.transforms.extend([
                ("fliph", flip_h_augmentation),
                ("flipv", flip_v_augmentation),
            ])

        if self.use_transpose:
            self.transforms.append(("transpose", transpose_augmentation))

    def random_augmentation(self, grid: Grid) -> Tuple[str, Grid]:
        """Apply random augmentation and return (name, augmented_grid)."""
        name, transform = self.rng.choice(self.transforms)
        result = transform(grid)

        if self.use_color_perms and self.rng.random() < 0.3:
            result = random_color_permutation(
                result,
                preserve_background=self.color_perm_preserve_bg,
                rng=self.rng
            )
            name = f"{name}_colorperm"

        return name, result

    def augment_batch(
        self,
        pairs: List[Tuple[NDArray, NDArray]],
        multiplier: int = 4,
    ) -> List[Tuple[NDArray, NDArray]]:
        """Augment a batch of input-output pairs."""
        augmented = []

        for input_arr, output_arr in pairs:
            input_grid = Grid(input_arr)
            output_grid = Grid(output_arr)

            # Include original
            augmented.append((input_arr.copy(), output_arr.copy()))

            # Generate augmentations
            for _ in range(multiplier - 1):
                name, transform = self.rng.choice(self.transforms)

                aug_input = transform(input_grid)
                aug_output = transform(output_grid)

                if self.use_color_perms and self.rng.random() < 0.2:
                    perm = [0] + list(self.rng.permutation(range(1, 10)))
                    aug_input = color_permutation(aug_input, perm)
                    aug_output = color_permutation(aug_output, perm)

                augmented.append((aug_input.data.copy(), aug_output.data.copy()))

        return augmented
