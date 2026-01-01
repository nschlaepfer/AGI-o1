"""
DSL Primitives - Core operations for ARC grid transformations.

Based on analysis of winning ARC solutions (Icecuber, NVARC, ARChitects),
this module provides a comprehensive set of primitives covering:
- Geometric transformations
- Color operations
- Object manipulation
- Pattern matching
- Composition operations
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from enum import Enum, auto

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class PrimitiveType(Enum):
    """Categories of DSL primitives."""
    GEOMETRIC = auto()
    COLOR = auto()
    OBJECT = auto()
    PATTERN = auto()
    COMPOSITE = auto()
    CONTROL = auto()


@dataclass
class DSLPrimitive:
    """
    A single DSL primitive operation.

    Attributes:
        name: Unique identifier
        func: The actual function to execute
        input_types: Expected input types
        output_type: Output type
        params: Parameter specifications
        category: Type of primitive
        description: Human-readable description
        cost: Complexity cost for search (lower = simpler)
    """
    name: str
    func: Callable
    input_types: List[str]
    output_type: str
    params: List[Dict[str, Any]] = field(default_factory=list)
    category: PrimitiveType = PrimitiveType.GEOMETRIC
    description: str = ""
    cost: float = 1.0

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


@dataclass
class DSLProgram:
    """
    A program composed of DSL primitives.

    Programs are represented as trees where each node is either:
    - A primitive application
    - An input reference
    - A constant value
    """
    op: str  # Primitive name or "INPUT" or "CONST"
    args: List[Union["DSLProgram", Any]] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        if self.op == "INPUT":
            return "input"
        if self.op == "CONST":
            return f"const({self.params.get('value', '?')})"

        args_str = ", ".join(str(a) for a in self.args)
        params_str = ", ".join(f"{k}={v}" for k, v in self.params.items())

        if args_str and params_str:
            return f"{self.op}({args_str}, {params_str})"
        elif args_str:
            return f"{self.op}({args_str})"
        elif params_str:
            return f"{self.op}({params_str})"
        else:
            return f"{self.op}()"

    @property
    def depth(self) -> int:
        """Tree depth of the program."""
        if not self.args:
            return 1
        return 1 + max(
            (a.depth if isinstance(a, DSLProgram) else 0) for a in self.args
        )

    @property
    def size(self) -> int:
        """Number of nodes in program tree."""
        if not self.args:
            return 1
        return 1 + sum(
            (a.size if isinstance(a, DSLProgram) else 1) for a in self.args
        )


def execute_program(
    program: DSLProgram,
    input_grid: NDArray[np.int8],
    primitives: Dict[str, DSLPrimitive]
) -> NDArray[np.int8]:
    """
    Execute a DSL program on an input grid.

    Args:
        program: The program to execute
        input_grid: Input grid as numpy array
        primitives: Dictionary of available primitives

    Returns:
        Output grid as numpy array
    """
    if program.op == "INPUT":
        return input_grid.copy()

    if program.op == "CONST":
        return np.array(program.params["value"], dtype=np.int8)

    # Evaluate arguments recursively
    evaluated_args = []
    for arg in program.args:
        if isinstance(arg, DSLProgram):
            evaluated_args.append(execute_program(arg, input_grid, primitives))
        else:
            evaluated_args.append(arg)

    # Execute primitive
    if program.op not in primitives:
        raise ValueError(f"Unknown primitive: {program.op}")

    primitive = primitives[program.op]
    return primitive.func(*evaluated_args, **program.params)


# ============================================================================
# Geometric Primitives
# ============================================================================

def _rotate(grid: NDArray, k: int = 1) -> NDArray:
    """Rotate grid 90 degrees clockwise, k times."""
    return np.rot90(grid, -k)

def _flip_h(grid: NDArray) -> NDArray:
    """Flip horizontally."""
    return np.fliplr(grid)

def _flip_v(grid: NDArray) -> NDArray:
    """Flip vertically."""
    return np.flipud(grid)

def _transpose(grid: NDArray) -> NDArray:
    """Transpose grid."""
    return grid.T

def _translate(grid: NDArray, dy: int = 0, dx: int = 0, fill: int = 0) -> NDArray:
    """Translate grid by offset."""
    result = np.full_like(grid, fill)
    h, w = grid.shape

    src_y = slice(max(0, -dy), min(h, h - dy))
    src_x = slice(max(0, -dx), min(w, w - dx))
    dst_y = slice(max(0, dy), min(h, h + dy))
    dst_x = slice(max(0, dx), min(w, w + dx))

    result[dst_y, dst_x] = grid[src_y, src_x]
    return result

def _scale(grid: NDArray, factor: int = 2) -> NDArray:
    """Scale grid by integer factor."""
    return np.kron(grid, np.ones((factor, factor), dtype=np.int8))

def _crop(grid: NDArray, top: int = 0, left: int = 0, height: int = -1, width: int = -1) -> NDArray:
    """Crop grid to region."""
    if height == -1:
        height = grid.shape[0] - top
    if width == -1:
        width = grid.shape[1] - left
    return grid[top:top+height, left:left+width].copy()

def _crop_to_content(grid: NDArray, background: int = 0) -> NDArray:
    """Crop to bounding box of non-background pixels."""
    mask = grid != background
    if not mask.any():
        return grid[:1, :1].copy()

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    return grid[rows][:, cols].copy()

def _pad(grid: NDArray, top: int = 0, bottom: int = 0, left: int = 0, right: int = 0, fill: int = 0) -> NDArray:
    """Pad grid with fill value."""
    return np.pad(grid, ((top, bottom), (left, right)), mode='constant', constant_values=fill)

def _tile(grid: NDArray, rows: int = 2, cols: int = 2) -> NDArray:
    """Tile grid to create larger grid."""
    return np.tile(grid, (rows, cols))

def _resize(grid: NDArray, height: int, width: int, fill: int = 0) -> NDArray:
    """Resize grid to specific dimensions."""
    result = np.full((height, width), fill, dtype=np.int8)
    h, w = min(grid.shape[0], height), min(grid.shape[1], width)
    result[:h, :w] = grid[:h, :w]
    return result


# ============================================================================
# Color Primitives
# ============================================================================

def _recolor(grid: NDArray, old_color: int, new_color: int) -> NDArray:
    """Replace one color with another."""
    result = grid.copy()
    result[result == old_color] = new_color
    return result

def _recolor_map(grid: NDArray, color_map: Dict[int, int]) -> NDArray:
    """Apply color mapping."""
    result = grid.copy()
    for old, new in color_map.items():
        result[grid == old] = new
    return result

def _fill_color(grid: NDArray, color: int) -> NDArray:
    """Fill entire grid with color."""
    return np.full_like(grid, color)

def _color_mask(grid: NDArray, color: int) -> NDArray:
    """Create binary mask for color (1 where color, 0 elsewhere)."""
    return (grid == color).astype(np.int8)

def _apply_mask(grid: NDArray, mask: NDArray, color: int) -> NDArray:
    """Apply color where mask is non-zero."""
    result = grid.copy()
    result[mask != 0] = color
    return result

def _most_common_color(grid: NDArray, exclude_zero: bool = True) -> int:
    """Find most common color."""
    unique, counts = np.unique(grid, return_counts=True)
    if exclude_zero:
        mask = unique != 0
        unique, counts = unique[mask], counts[mask]
    if len(unique) == 0:
        return 0
    return int(unique[np.argmax(counts)])

def _least_common_color(grid: NDArray, exclude_zero: bool = True) -> int:
    """Find least common color."""
    unique, counts = np.unique(grid, return_counts=True)
    if exclude_zero:
        mask = unique != 0
        unique, counts = unique[mask], counts[mask]
    if len(unique) == 0:
        return 0
    return int(unique[np.argmin(counts)])

def _count_colors(grid: NDArray) -> int:
    """Count number of unique colors."""
    return len(np.unique(grid))

def _swap_colors(grid: NDArray, color1: int, color2: int) -> NDArray:
    """Swap two colors."""
    result = grid.copy()
    mask1 = grid == color1
    mask2 = grid == color2
    result[mask1] = color2
    result[mask2] = color1
    return result


# ============================================================================
# Object/Pattern Primitives
# ============================================================================

def _find_objects(grid: NDArray, background: int = 0, connectivity: int = 4) -> List[NDArray]:
    """Find connected components."""
    from scipy import ndimage

    mask = grid != background

    if connectivity == 4:
        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    else:
        structure = np.ones((3, 3))

    labeled, num_features = ndimage.label(mask, structure=structure)

    objects = []
    for i in range(1, num_features + 1):
        obj_mask = labeled == i
        # Extract bounding box
        rows, cols = np.where(obj_mask)
        if len(rows) == 0:
            continue
        min_r, max_r = rows.min(), rows.max()
        min_c, max_c = cols.min(), cols.max()

        obj = np.zeros((max_r - min_r + 1, max_c - min_c + 1), dtype=np.int8)
        for r, c in zip(rows, cols):
            obj[r - min_r, c - min_c] = grid[r, c]
        objects.append(obj)

    return objects

def _extract_object(grid: NDArray, index: int = 0, background: int = 0) -> NDArray:
    """Extract nth object from grid."""
    objects = _find_objects(grid, background)
    if index < len(objects):
        return objects[index]
    return grid.copy()

def _largest_object(grid: NDArray, background: int = 0) -> NDArray:
    """Extract largest connected component."""
    objects = _find_objects(grid, background)
    if not objects:
        return grid.copy()
    return max(objects, key=lambda o: o.size)

def _smallest_object(grid: NDArray, background: int = 0) -> NDArray:
    """Extract smallest connected component."""
    objects = _find_objects(grid, background)
    if not objects:
        return grid.copy()
    return min(objects, key=lambda o: o.size)

def _overlay(grid1: NDArray, grid2: NDArray, row: int = 0, col: int = 0, transparent: int = 0) -> NDArray:
    """Overlay grid2 on grid1 at position."""
    result = grid1.copy()
    h, w = grid2.shape
    gh, gw = grid1.shape

    # Clip to bounds
    src_r = max(0, -row)
    src_c = max(0, -col)
    dst_r = max(0, row)
    dst_c = max(0, col)

    copy_h = min(h - src_r, gh - dst_r)
    copy_w = min(w - src_c, gw - dst_c)

    if copy_h > 0 and copy_w > 0:
        src = grid2[src_r:src_r+copy_h, src_c:src_c+copy_w]
        if transparent >= 0:
            mask = src != transparent
            result[dst_r:dst_r+copy_h, dst_c:dst_c+copy_w][mask] = src[mask]
        else:
            result[dst_r:dst_r+copy_h, dst_c:dst_c+copy_w] = src

    return result

def _flood_fill(grid: NDArray, row: int, col: int, new_color: int, connectivity: int = 4) -> NDArray:
    """Flood fill from position."""
    result = grid.copy()
    old_color = grid[row, col]

    if old_color == new_color:
        return result

    h, w = grid.shape
    stack = [(row, col)]

    if connectivity == 4:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    while stack:
        r, c = stack.pop()
        if result[r, c] != old_color:
            continue
        result[r, c] = new_color

        for dr, dc in neighbors:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and result[nr, nc] == old_color:
                stack.append((nr, nc))

    return result

def _fill_enclosed(grid: NDArray, boundary_color: int, fill_color: int, background: int = 0) -> NDArray:
    """Fill regions enclosed by boundary color."""
    result = grid.copy()
    h, w = grid.shape

    # Find all regions that can reach the edge (not enclosed)
    reachable = np.zeros((h, w), dtype=bool)
    stack = []

    # Start from edges
    for i in range(h):
        if grid[i, 0] != boundary_color:
            stack.append((i, 0))
        if grid[i, w-1] != boundary_color:
            stack.append((i, w-1))
    for j in range(w):
        if grid[0, j] != boundary_color:
            stack.append((0, j))
        if grid[h-1, j] != boundary_color:
            stack.append((h-1, j))

    while stack:
        r, c = stack.pop()
        if reachable[r, c] or grid[r, c] == boundary_color:
            continue
        reachable[r, c] = True

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                stack.append((nr, nc))

    # Fill unreachable non-boundary regions
    result[(~reachable) & (grid != boundary_color)] = fill_color
    return result


# ============================================================================
# Pattern Matching Primitives
# ============================================================================

def _find_pattern(grid: NDArray, pattern: NDArray) -> List[Tuple[int, int]]:
    """Find all occurrences of pattern in grid."""
    matches = []
    ph, pw = pattern.shape
    gh, gw = grid.shape

    for r in range(gh - ph + 1):
        for c in range(gw - pw + 1):
            if np.array_equal(grid[r:r+ph, c:c+pw], pattern):
                matches.append((r, c))

    return matches

def _replace_pattern(grid: NDArray, old_pattern: NDArray, new_pattern: NDArray) -> NDArray:
    """Replace all occurrences of pattern."""
    result = grid.copy()
    positions = _find_pattern(grid, old_pattern)

    ph, pw = new_pattern.shape
    for r, c in positions:
        if r + ph <= result.shape[0] and c + pw <= result.shape[1]:
            result[r:r+ph, c:c+pw] = new_pattern

    return result

def _mirror_pattern(grid: NDArray, axis: str = "horizontal") -> NDArray:
    """Create mirrored pattern."""
    if axis == "horizontal":
        return np.concatenate([grid, np.fliplr(grid)], axis=1)
    elif axis == "vertical":
        return np.concatenate([grid, np.flipud(grid)], axis=0)
    elif axis == "both":
        top = np.concatenate([grid, np.fliplr(grid)], axis=1)
        return np.concatenate([top, np.flipud(top)], axis=0)
    return grid.copy()

def _repeat_pattern(grid: NDArray, direction: str = "right", count: int = 2) -> NDArray:
    """Repeat pattern in direction."""
    if direction == "right":
        return np.tile(grid, (1, count))
    elif direction == "down":
        return np.tile(grid, (count, 1))
    elif direction == "both":
        return np.tile(grid, (count, count))
    return grid.copy()


# ============================================================================
# Composite/Control Primitives
# ============================================================================

def _identity(grid: NDArray) -> NDArray:
    """Return unchanged grid."""
    return grid.copy()

def _compose(grid: NDArray, *operations: Callable) -> NDArray:
    """Apply sequence of operations."""
    result = grid.copy()
    for op in operations:
        result = op(result)
    return result

def _conditional(grid: NDArray, condition: Callable, if_true: Callable, if_false: Callable) -> NDArray:
    """Conditional execution."""
    if condition(grid):
        return if_true(grid)
    return if_false(grid)

def _per_object(grid: NDArray, operation: Callable, background: int = 0) -> NDArray:
    """Apply operation to each object separately, then recombine."""
    objects = _find_objects(grid, background)
    if not objects:
        return grid.copy()

    # Find bounding boxes
    from scipy import ndimage
    mask = grid != background
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    labeled, num_features = ndimage.label(mask, structure)

    result = np.full_like(grid, background)

    for i in range(1, num_features + 1):
        obj_mask = labeled == i
        rows, cols = np.where(obj_mask)
        if len(rows) == 0:
            continue

        min_r, max_r = rows.min(), rows.max()
        min_c, max_c = cols.min(), cols.max()

        # Extract, transform, overlay
        obj = grid[min_r:max_r+1, min_c:max_c+1].copy()
        obj[~obj_mask[min_r:max_r+1, min_c:max_c+1]] = background

        transformed = operation(obj)
        result = _overlay(result, transformed, min_r, min_c, transparent=background)

    return result


# ============================================================================
# Registry of all primitives
# ============================================================================

def get_all_primitives() -> Dict[str, DSLPrimitive]:
    """Get dictionary of all DSL primitives."""
    primitives = {}

    # Geometric
    primitives["rotate90"] = DSLPrimitive(
        "rotate90", lambda g: _rotate(g, 1), ["grid"], "grid",
        category=PrimitiveType.GEOMETRIC, description="Rotate 90 degrees clockwise", cost=1.0
    )
    primitives["rotate180"] = DSLPrimitive(
        "rotate180", lambda g: _rotate(g, 2), ["grid"], "grid",
        category=PrimitiveType.GEOMETRIC, description="Rotate 180 degrees", cost=1.0
    )
    primitives["rotate270"] = DSLPrimitive(
        "rotate270", lambda g: _rotate(g, 3), ["grid"], "grid",
        category=PrimitiveType.GEOMETRIC, description="Rotate 270 degrees clockwise", cost=1.0
    )
    primitives["flip_h"] = DSLPrimitive(
        "flip_h", _flip_h, ["grid"], "grid",
        category=PrimitiveType.GEOMETRIC, description="Flip horizontally", cost=1.0
    )
    primitives["flip_v"] = DSLPrimitive(
        "flip_v", _flip_v, ["grid"], "grid",
        category=PrimitiveType.GEOMETRIC, description="Flip vertically", cost=1.0
    )
    primitives["transpose"] = DSLPrimitive(
        "transpose", _transpose, ["grid"], "grid",
        category=PrimitiveType.GEOMETRIC, description="Transpose grid", cost=1.0
    )
    primitives["translate"] = DSLPrimitive(
        "translate", _translate, ["grid"], "grid",
        params=[{"name": "dy", "type": "int"}, {"name": "dx", "type": "int"}, {"name": "fill", "type": "int", "default": 0}],
        category=PrimitiveType.GEOMETRIC, description="Translate by offset", cost=1.5
    )
    primitives["scale"] = DSLPrimitive(
        "scale", _scale, ["grid"], "grid",
        params=[{"name": "factor", "type": "int", "default": 2}],
        category=PrimitiveType.GEOMETRIC, description="Scale by factor", cost=1.5
    )
    primitives["crop"] = DSLPrimitive(
        "crop", _crop, ["grid"], "grid",
        params=[{"name": "top", "type": "int"}, {"name": "left", "type": "int"},
                {"name": "height", "type": "int"}, {"name": "width", "type": "int"}],
        category=PrimitiveType.GEOMETRIC, description="Crop to region", cost=2.0
    )
    primitives["crop_to_content"] = DSLPrimitive(
        "crop_to_content", _crop_to_content, ["grid"], "grid",
        params=[{"name": "background", "type": "int", "default": 0}],
        category=PrimitiveType.GEOMETRIC, description="Crop to content bounds", cost=1.5
    )
    primitives["pad"] = DSLPrimitive(
        "pad", _pad, ["grid"], "grid",
        params=[{"name": "top", "type": "int"}, {"name": "bottom", "type": "int"},
                {"name": "left", "type": "int"}, {"name": "right", "type": "int"},
                {"name": "fill", "type": "int", "default": 0}],
        category=PrimitiveType.GEOMETRIC, description="Add padding", cost=1.5
    )
    primitives["tile"] = DSLPrimitive(
        "tile", _tile, ["grid"], "grid",
        params=[{"name": "rows", "type": "int", "default": 2}, {"name": "cols", "type": "int", "default": 2}],
        category=PrimitiveType.GEOMETRIC, description="Tile grid", cost=1.5
    )
    primitives["resize"] = DSLPrimitive(
        "resize", _resize, ["grid"], "grid",
        params=[{"name": "height", "type": "int"}, {"name": "width", "type": "int"},
                {"name": "fill", "type": "int", "default": 0}],
        category=PrimitiveType.GEOMETRIC, description="Resize to dimensions", cost=2.0
    )

    # Color
    primitives["recolor"] = DSLPrimitive(
        "recolor", _recolor, ["grid"], "grid",
        params=[{"name": "old_color", "type": "int"}, {"name": "new_color", "type": "int"}],
        category=PrimitiveType.COLOR, description="Replace color", cost=1.0
    )
    primitives["fill_color"] = DSLPrimitive(
        "fill_color", _fill_color, ["grid"], "grid",
        params=[{"name": "color", "type": "int"}],
        category=PrimitiveType.COLOR, description="Fill with color", cost=1.0
    )
    primitives["color_mask"] = DSLPrimitive(
        "color_mask", _color_mask, ["grid"], "grid",
        params=[{"name": "color", "type": "int"}],
        category=PrimitiveType.COLOR, description="Create mask for color", cost=1.0
    )
    primitives["apply_mask"] = DSLPrimitive(
        "apply_mask", _apply_mask, ["grid", "mask"], "grid",
        params=[{"name": "color", "type": "int"}],
        category=PrimitiveType.COLOR, description="Apply color through mask", cost=1.5
    )
    primitives["swap_colors"] = DSLPrimitive(
        "swap_colors", _swap_colors, ["grid"], "grid",
        params=[{"name": "color1", "type": "int"}, {"name": "color2", "type": "int"}],
        category=PrimitiveType.COLOR, description="Swap two colors", cost=1.0
    )

    # Object
    primitives["extract_object"] = DSLPrimitive(
        "extract_object", _extract_object, ["grid"], "grid",
        params=[{"name": "index", "type": "int", "default": 0}, {"name": "background", "type": "int", "default": 0}],
        category=PrimitiveType.OBJECT, description="Extract nth object", cost=2.0
    )
    primitives["largest_object"] = DSLPrimitive(
        "largest_object", _largest_object, ["grid"], "grid",
        params=[{"name": "background", "type": "int", "default": 0}],
        category=PrimitiveType.OBJECT, description="Extract largest object", cost=2.0
    )
    primitives["smallest_object"] = DSLPrimitive(
        "smallest_object", _smallest_object, ["grid"], "grid",
        params=[{"name": "background", "type": "int", "default": 0}],
        category=PrimitiveType.OBJECT, description="Extract smallest object", cost=2.0
    )
    primitives["overlay"] = DSLPrimitive(
        "overlay", _overlay, ["grid", "grid"], "grid",
        params=[{"name": "row", "type": "int", "default": 0}, {"name": "col", "type": "int", "default": 0},
                {"name": "transparent", "type": "int", "default": 0}],
        category=PrimitiveType.OBJECT, description="Overlay grids", cost=1.5
    )
    primitives["flood_fill"] = DSLPrimitive(
        "flood_fill", _flood_fill, ["grid"], "grid",
        params=[{"name": "row", "type": "int"}, {"name": "col", "type": "int"},
                {"name": "new_color", "type": "int"}, {"name": "connectivity", "type": "int", "default": 4}],
        category=PrimitiveType.OBJECT, description="Flood fill from position", cost=2.0
    )
    primitives["fill_enclosed"] = DSLPrimitive(
        "fill_enclosed", _fill_enclosed, ["grid"], "grid",
        params=[{"name": "boundary_color", "type": "int"}, {"name": "fill_color", "type": "int"},
                {"name": "background", "type": "int", "default": 0}],
        category=PrimitiveType.OBJECT, description="Fill enclosed regions", cost=2.5
    )

    # Pattern
    primitives["mirror_pattern"] = DSLPrimitive(
        "mirror_pattern", _mirror_pattern, ["grid"], "grid",
        params=[{"name": "axis", "type": "str", "default": "horizontal"}],
        category=PrimitiveType.PATTERN, description="Mirror to create symmetric pattern", cost=1.5
    )
    primitives["repeat_pattern"] = DSLPrimitive(
        "repeat_pattern", _repeat_pattern, ["grid"], "grid",
        params=[{"name": "direction", "type": "str", "default": "right"}, {"name": "count", "type": "int", "default": 2}],
        category=PrimitiveType.PATTERN, description="Repeat pattern", cost=1.5
    )

    # Control
    primitives["identity"] = DSLPrimitive(
        "identity", _identity, ["grid"], "grid",
        category=PrimitiveType.CONTROL, description="Return unchanged", cost=0.5
    )

    return primitives
