"""
Grid Operations Module - Core data structure for ARC tasks.

Provides a comprehensive Grid class with all transformations needed
for ARC-AGI-2 pattern recognition and manipulation.
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Iterator
from copy import deepcopy

import numpy as np
from numpy.typing import NDArray


# ARC color palette (0-9)
ARC_COLORS = {
    0: "black",
    1: "blue",
    2: "red",
    3: "green",
    4: "yellow",
    5: "grey",
    6: "magenta",
    7: "orange",
    8: "azure",
    9: "maroon",
}


@dataclass
class BoundingBox:
    """Axis-aligned bounding box for grid regions."""
    min_row: int
    min_col: int
    max_row: int
    max_col: int

    @property
    def height(self) -> int:
        return self.max_row - self.min_row + 1

    @property
    def width(self) -> int:
        return self.max_col - self.min_col + 1

    @property
    def area(self) -> int:
        return self.height * self.width

    def contains(self, row: int, col: int) -> bool:
        return (self.min_row <= row <= self.max_row and
                self.min_col <= col <= self.max_col)


@dataclass
class GridObject:
    """A connected component or pattern within a grid."""
    pixels: List[Tuple[int, int]]  # (row, col) coordinates
    color: int
    bbox: BoundingBox

    @property
    def size(self) -> int:
        return len(self.pixels)

    def to_grid(self, background: int = 0) -> "Grid":
        """Convert object to its own grid."""
        data = np.full((self.bbox.height, self.bbox.width), background, dtype=np.int8)
        for r, c in self.pixels:
            data[r - self.bbox.min_row, c - self.bbox.min_col] = self.color
        return Grid(data)


class Grid:
    """
    Core grid data structure for ARC tasks.

    Wraps a 2D numpy array with values 0-9 (ARC colors) and provides
    all transformation primitives needed for pattern recognition.
    """

    def __init__(self, data: NDArray[np.int8] | List[List[int]]):
        if isinstance(data, list):
            self._data = np.array(data, dtype=np.int8)
        else:
            self._data = data.astype(np.int8)

        # Cache for expensive computations
        self._hash: Optional[str] = None
        self._objects_cache: Optional[List[GridObject]] = None

    @classmethod
    def from_json(cls, json_data: List[List[int]]) -> "Grid":
        """Create grid from ARC JSON format."""
        return cls(json_data)

    @classmethod
    def empty(cls, height: int, width: int, fill: int = 0) -> "Grid":
        """Create an empty grid with specified dimensions."""
        return cls(np.full((height, width), fill, dtype=np.int8))

    @classmethod
    def from_string(cls, s: str, color_map: Optional[Dict[str, int]] = None) -> "Grid":
        """Parse grid from string representation."""
        if color_map is None:
            color_map = {str(i): i for i in range(10)}
            color_map["."] = 0

        rows = []
        for line in s.strip().split("\n"):
            row = [color_map.get(c, 0) for c in line.strip()]
            if row:
                rows.append(row)
        return cls(rows)

    # ==================== Properties ====================

    @property
    def data(self) -> NDArray[np.int8]:
        return self._data

    @property
    def height(self) -> int:
        return self._data.shape[0]

    @property
    def width(self) -> int:
        return self._data.shape[1]

    @property
    def shape(self) -> Tuple[int, int]:
        return self._data.shape

    @property
    def size(self) -> int:
        return self._data.size

    @property
    def unique_colors(self) -> Set[int]:
        return set(np.unique(self._data).tolist())

    @property
    def color_counts(self) -> Dict[int, int]:
        unique, counts = np.unique(self._data, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))

    @property
    def background_color(self) -> int:
        """Most common color (assumed to be background)."""
        counts = self.color_counts
        return max(counts.keys(), key=lambda c: counts[c])

    def hash(self) -> str:
        """Compute stable hash for deduplication."""
        if self._hash is None:
            self._hash = hashlib.md5(self._data.tobytes()).hexdigest()
        return self._hash

    # ==================== Basic Operations ====================

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Grid):
            return False
        return np.array_equal(self._data, other._data)

    def __hash__(self) -> int:
        return hash(self.hash())

    def __getitem__(self, key) -> Any:
        return self._data[key]

    def __setitem__(self, key, value) -> None:
        self._data[key] = value
        self._invalidate_cache()

    def copy(self) -> "Grid":
        return Grid(self._data.copy())

    def to_json(self) -> List[List[int]]:
        """Convert to ARC JSON format."""
        return self._data.tolist()

    def to_string(self, color_map: Optional[Dict[int, str]] = None) -> str:
        """Convert to string representation."""
        if color_map is None:
            color_map = {i: str(i) for i in range(10)}
            color_map[0] = "."

        lines = []
        for row in self._data:
            lines.append("".join(color_map.get(int(c), "?") for c in row))
        return "\n".join(lines)

    def _invalidate_cache(self) -> None:
        self._hash = None
        self._objects_cache = None

    # ==================== Geometric Transformations ====================

    def rotate(self, k: int = 1) -> "Grid":
        """Rotate grid 90 degrees clockwise, k times."""
        return Grid(np.rot90(self._data, -k))

    def rotate_90(self) -> "Grid":
        return self.rotate(1)

    def rotate_180(self) -> "Grid":
        return self.rotate(2)

    def rotate_270(self) -> "Grid":
        return self.rotate(3)

    def flip_horizontal(self) -> "Grid":
        """Flip left-right."""
        return Grid(np.fliplr(self._data))

    def flip_vertical(self) -> "Grid":
        """Flip top-bottom."""
        return Grid(np.flipud(self._data))

    def transpose(self) -> "Grid":
        return Grid(self._data.T)

    def translate(self, dy: int, dx: int, wrap: bool = False, fill: int = 0) -> "Grid":
        """Translate grid by (dy, dx). Positive = down/right."""
        result = np.full_like(self._data, fill)

        # Source region
        src_y_start = max(0, -dy)
        src_y_end = min(self.height, self.height - dy)
        src_x_start = max(0, -dx)
        src_x_end = min(self.width, self.width - dx)

        # Destination region
        dst_y_start = max(0, dy)
        dst_y_end = min(self.height, self.height + dy)
        dst_x_start = max(0, dx)
        dst_x_end = min(self.width, self.width + dx)

        if wrap:
            result = np.roll(np.roll(self._data, dy, axis=0), dx, axis=1)
        else:
            result[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
                self._data[src_y_start:src_y_end, src_x_start:src_x_end]

        return Grid(result)

    def scale(self, factor: int) -> "Grid":
        """Scale grid by integer factor (each pixel becomes factor x factor)."""
        return Grid(np.kron(self._data, np.ones((factor, factor), dtype=np.int8)))

    def crop(self, bbox: BoundingBox) -> "Grid":
        """Extract region defined by bounding box."""
        return Grid(self._data[
            bbox.min_row:bbox.max_row + 1,
            bbox.min_col:bbox.max_col + 1
        ])

    def crop_to_content(self, background: int = 0) -> "Grid":
        """Remove background border, keeping only content."""
        mask = self._data != background
        if not mask.any():
            return Grid.empty(1, 1, background)

        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        return Grid(self._data[rows][:, cols])

    def pad(self, top: int = 0, bottom: int = 0, left: int = 0, right: int = 0,
            fill: int = 0) -> "Grid":
        """Add padding around grid."""
        return Grid(np.pad(
            self._data,
            ((top, bottom), (left, right)),
            mode="constant",
            constant_values=fill
        ))

    def resize(self, new_height: int, new_width: int, fill: int = 0) -> "Grid":
        """Resize grid to new dimensions (crop or pad as needed)."""
        result = np.full((new_height, new_width), fill, dtype=np.int8)

        copy_h = min(self.height, new_height)
        copy_w = min(self.width, new_width)

        result[:copy_h, :copy_w] = self._data[:copy_h, :copy_w]
        return Grid(result)

    def tile(self, rows: int, cols: int) -> "Grid":
        """Tile grid to create larger grid."""
        return Grid(np.tile(self._data, (rows, cols)))

    # ==================== Color Operations ====================

    def recolor(self, old_color: int, new_color: int) -> "Grid":
        """Replace one color with another."""
        result = self._data.copy()
        result[result == old_color] = new_color
        return Grid(result)

    def recolor_map(self, color_map: Dict[int, int]) -> "Grid":
        """Apply color mapping."""
        result = self._data.copy()
        for old, new in color_map.items():
            result[self._data == old] = new
        return Grid(result)

    def invert_colors(self, exclude: Optional[Set[int]] = None) -> "Grid":
        """Swap each color with its complement (9 - color)."""
        exclude = exclude or {0}  # Don't invert background by default
        result = self._data.copy()
        for c in range(10):
            if c not in exclude:
                result[self._data == c] = 9 - c
        return Grid(result)

    def fill_color(self, color: int, mask: Optional[NDArray[np.bool_]] = None) -> "Grid":
        """Fill with solid color (optionally using mask)."""
        result = self._data.copy()
        if mask is not None:
            result[mask] = color
        else:
            result.fill(color)
        return Grid(result)

    def color_mask(self, color: int) -> NDArray[np.bool_]:
        """Get boolean mask for specific color."""
        return self._data == color

    def most_common_color(self, exclude: Optional[Set[int]] = None) -> int:
        """Get most common color, optionally excluding some."""
        exclude = exclude or set()
        counts = self.color_counts
        for c in exclude:
            counts.pop(c, None)
        return max(counts.keys(), key=lambda c: counts[c]) if counts else 0

    # ==================== Pattern Operations ====================

    def find_objects(self, background: int = 0, connectivity: int = 4) -> List[GridObject]:
        """Find connected components (objects) in grid."""
        if self._objects_cache is not None:
            return self._objects_cache

        from scipy import ndimage

        objects = []
        mask = self._data != background

        if connectivity == 4:
            structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        else:  # 8-connectivity
            structure = np.ones((3, 3))

        labeled, num_features = ndimage.label(mask, structure=structure)

        for i in range(1, num_features + 1):
            obj_mask = labeled == i
            pixels = list(zip(*np.where(obj_mask)))

            if pixels:
                rows, cols = zip(*pixels)
                color = int(self._data[pixels[0][0], pixels[0][1]])

                bbox = BoundingBox(
                    min_row=min(rows),
                    min_col=min(cols),
                    max_row=max(rows),
                    max_col=max(cols)
                )

                objects.append(GridObject(
                    pixels=list(pixels),
                    color=color,
                    bbox=bbox
                ))

        self._objects_cache = objects
        return objects

    def find_patterns(self, pattern: "Grid", background: int = 0) -> List[Tuple[int, int]]:
        """Find all occurrences of a pattern in grid."""
        matches = []
        ph, pw = pattern.shape

        for r in range(self.height - ph + 1):
            for c in range(self.width - pw + 1):
                region = self._data[r:r+ph, c:c+pw]
                if np.array_equal(region, pattern._data):
                    matches.append((r, c))

        return matches

    def extract_subgrid(self, row: int, col: int, height: int, width: int) -> "Grid":
        """Extract a subgrid at specified position."""
        return Grid(self._data[row:row+height, col:col+width])

    def overlay(self, other: "Grid", row: int = 0, col: int = 0,
                transparent: Optional[int] = None) -> "Grid":
        """Overlay another grid on top at specified position."""
        result = self._data.copy()

        # Compute overlap region
        src_r = max(0, -row)
        src_c = max(0, -col)
        dst_r = max(0, row)
        dst_c = max(0, col)

        h = min(other.height - src_r, self.height - dst_r)
        w = min(other.width - src_c, self.width - dst_c)

        if h > 0 and w > 0:
            src_region = other._data[src_r:src_r+h, src_c:src_c+w]

            if transparent is not None:
                mask = src_region != transparent
                result[dst_r:dst_r+h, dst_c:dst_c+w][mask] = src_region[mask]
            else:
                result[dst_r:dst_r+h, dst_c:dst_c+w] = src_region

        return Grid(result)

    def flood_fill(self, row: int, col: int, new_color: int,
                   connectivity: int = 4) -> "Grid":
        """Flood fill starting from a position."""
        result = self._data.copy()
        old_color = result[row, col]

        if old_color == new_color:
            return Grid(result)

        stack = [(row, col)]

        if connectivity == 4:
            neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        else:
            neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                        (0, 1), (1, -1), (1, 0), (1, 1)]

        while stack:
            r, c = stack.pop()
            if result[r, c] != old_color:
                continue

            result[r, c] = new_color

            for dr, dc in neighbors:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.height and 0 <= nc < self.width:
                    if result[nr, nc] == old_color:
                        stack.append((nr, nc))

        return Grid(result)

    # ==================== Analysis Operations ====================

    def is_symmetric(self, axis: str = "horizontal") -> bool:
        """Check if grid has symmetry."""
        if axis == "horizontal":
            return np.array_equal(self._data, np.fliplr(self._data))
        elif axis == "vertical":
            return np.array_equal(self._data, np.flipud(self._data))
        elif axis == "diagonal":
            if self.height != self.width:
                return False
            return np.array_equal(self._data, self._data.T)
        return False

    def count_color(self, color: int) -> int:
        """Count pixels of specific color."""
        return int(np.sum(self._data == color))

    def similarity(self, other: "Grid") -> float:
        """Calculate pixel-wise similarity (0-1)."""
        if self.shape != other.shape:
            return 0.0
        return float(np.mean(self._data == other._data))

    def diff(self, other: "Grid") -> "Grid":
        """Create diff grid (non-zero where different)."""
        if self.shape != other.shape:
            raise ValueError("Grids must have same shape for diff")

        result = np.zeros_like(self._data)
        diff_mask = self._data != other._data
        result[diff_mask] = 1  # Mark differences
        return Grid(result)

    # ==================== Augmentation ====================

    def all_rotations(self) -> List["Grid"]:
        """Get all 4 rotation variants."""
        return [self, self.rotate_90(), self.rotate_180(), self.rotate_270()]

    def all_flips(self) -> List["Grid"]:
        """Get all flip variants."""
        return [self, self.flip_horizontal(), self.flip_vertical()]

    def all_transforms(self) -> List["Grid"]:
        """Get all 8 transformation variants (4 rotations x 2 flips)."""
        transforms = []
        for rotated in self.all_rotations():
            transforms.append(rotated)
            transforms.append(rotated.flip_horizontal())
        return transforms

    def random_color_permutation(self, rng: Optional[np.random.Generator] = None,
                                  preserve_background: bool = True) -> "Grid":
        """Apply random color permutation."""
        rng = rng or np.random.default_rng()

        colors = list(range(10))
        if preserve_background:
            # Keep 0 as 0
            other_colors = colors[1:]
            rng.shuffle(other_colors)
            perm = [0] + list(other_colors)
        else:
            rng.shuffle(colors)
            perm = colors

        color_map = {i: perm[i] for i in range(10)}
        return self.recolor_map(color_map)


@dataclass
class GridPair:
    """An input-output grid pair for ARC tasks."""
    input: Grid
    output: Grid

    def to_json(self) -> Dict[str, List[List[int]]]:
        return {
            "input": self.input.to_json(),
            "output": self.output.to_json()
        }

    @classmethod
    def from_json(cls, data: Dict[str, List[List[int]]]) -> "GridPair":
        return cls(
            input=Grid.from_json(data["input"]),
            output=Grid.from_json(data["output"])
        )

    def augment(self, transform_fn) -> "GridPair":
        """Apply same transformation to both input and output."""
        return GridPair(
            input=transform_fn(self.input),
            output=transform_fn(self.output)
        )
