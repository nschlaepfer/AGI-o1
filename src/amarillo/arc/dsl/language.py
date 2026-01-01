"""
ARC DSL Language Definition - Grammar and construction utilities.

Provides:
- DSL construction with type checking
- Program serialization/deserialization
- Standard DSL configurations
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from numpy.typing import NDArray

from .primitives import DSLPrimitive, DSLProgram, execute_program, get_all_primitives, PrimitiveType

logger = logging.getLogger(__name__)


@dataclass
class DSLConfig:
    """Configuration for DSL instance."""
    max_program_depth: int = 6
    max_program_size: int = 20
    allowed_categories: Set[PrimitiveType] = field(default_factory=lambda: set(PrimitiveType))
    excluded_primitives: Set[str] = field(default_factory=set)
    param_ranges: Dict[str, Tuple[Any, Any]] = field(default_factory=dict)


class ARCDSL:
    """
    Domain-Specific Language for ARC grid transformations.

    Provides:
    - Primitive registry with type information
    - Program construction and validation
    - Execution with error handling
    - Program enumeration for search
    """

    def __init__(self, config: Optional[DSLConfig] = None):
        self.config = config or DSLConfig()
        self._primitives: Dict[str, DSLPrimitive] = {}
        self._load_primitives()

    def _load_primitives(self) -> None:
        """Load primitives based on config."""
        all_primitives = get_all_primitives()

        for name, primitive in all_primitives.items():
            # Filter by category
            if self.config.allowed_categories and primitive.category not in self.config.allowed_categories:
                continue

            # Filter excluded
            if name in self.config.excluded_primitives:
                continue

            self._primitives[name] = primitive

        logger.info(f"Loaded {len(self._primitives)} DSL primitives")

    @property
    def primitives(self) -> Dict[str, DSLPrimitive]:
        return self._primitives

    @property
    def primitive_names(self) -> List[str]:
        return list(self._primitives.keys())

    def get_primitive(self, name: str) -> Optional[DSLPrimitive]:
        return self._primitives.get(name)

    def has_primitive(self, name: str) -> bool:
        return name in self._primitives

    # ========================================================================
    # Program Construction
    # ========================================================================

    def input_node(self) -> DSLProgram:
        """Create input reference node."""
        return DSLProgram(op="INPUT")

    def const_node(self, value: Any) -> DSLProgram:
        """Create constant value node."""
        return DSLProgram(op="CONST", params={"value": value})

    def apply(self, primitive_name: str, *args: DSLProgram, **kwargs) -> DSLProgram:
        """Create primitive application node."""
        if not self.has_primitive(primitive_name):
            raise ValueError(f"Unknown primitive: {primitive_name}")

        return DSLProgram(
            op=primitive_name,
            args=list(args),
            params=kwargs
        )

    def compose(self, *programs: DSLProgram) -> DSLProgram:
        """Compose programs sequentially (left to right)."""
        if not programs:
            return self.input_node()

        result = programs[0]
        for program in programs[1:]:
            # Replace INPUT in program with result
            result = self._substitute_input(program, result)

        return result

    def _substitute_input(self, program: DSLProgram, replacement: DSLProgram) -> DSLProgram:
        """Replace INPUT nodes with replacement."""
        if program.op == "INPUT":
            return replacement

        new_args = []
        for arg in program.args:
            if isinstance(arg, DSLProgram):
                new_args.append(self._substitute_input(arg, replacement))
            else:
                new_args.append(arg)

        return DSLProgram(op=program.op, args=new_args, params=program.params.copy())

    # ========================================================================
    # Program Validation
    # ========================================================================

    def validate(self, program: DSLProgram) -> Tuple[bool, str]:
        """Validate program structure."""
        # Check depth
        if program.depth > self.config.max_program_depth:
            return False, f"Program depth {program.depth} exceeds max {self.config.max_program_depth}"

        # Check size
        if program.size > self.config.max_program_size:
            return False, f"Program size {program.size} exceeds max {self.config.max_program_size}"

        # Check primitives exist
        def check_node(node: DSLProgram) -> Tuple[bool, str]:
            if node.op in ("INPUT", "CONST"):
                return True, ""

            if not self.has_primitive(node.op):
                return False, f"Unknown primitive: {node.op}"

            for arg in node.args:
                if isinstance(arg, DSLProgram):
                    valid, msg = check_node(arg)
                    if not valid:
                        return False, msg

            return True, ""

        return check_node(program)

    # ========================================================================
    # Program Execution
    # ========================================================================

    def execute(self, program: DSLProgram, input_grid: NDArray[np.int8]) -> NDArray[np.int8]:
        """Execute program on input grid."""
        valid, msg = self.validate(program)
        if not valid:
            raise ValueError(f"Invalid program: {msg}")

        return execute_program(program, input_grid, self._primitives)

    def safe_execute(self, program: DSLProgram, input_grid: NDArray[np.int8]) -> Optional[NDArray[np.int8]]:
        """Execute with error handling, returns None on failure."""
        try:
            return self.execute(program, input_grid)
        except Exception as e:
            logger.debug(f"Execution failed: {e}")
            return None

    def test_on_examples(
        self,
        program: DSLProgram,
        examples: List[Tuple[NDArray, NDArray]]
    ) -> Tuple[bool, float]:
        """
        Test program on input/output examples.

        Returns (all_correct, accuracy).
        """
        if not examples:
            return False, 0.0

        correct = 0
        for input_grid, expected_output in examples:
            try:
                output = self.execute(program, input_grid)
                if np.array_equal(output, expected_output):
                    correct += 1
            except Exception:
                pass

        accuracy = correct / len(examples)
        return correct == len(examples), accuracy

    # ========================================================================
    # Serialization
    # ========================================================================

    def serialize(self, program: DSLProgram) -> str:
        """Serialize program to JSON string."""
        def to_dict(node: DSLProgram) -> Dict:
            result = {"op": node.op}
            if node.args:
                result["args"] = [
                    to_dict(a) if isinstance(a, DSLProgram) else a
                    for a in node.args
                ]
            if node.params:
                # Convert numpy types to Python types
                params = {}
                for k, v in node.params.items():
                    if hasattr(v, "tolist"):
                        params[k] = v.tolist()
                    else:
                        params[k] = v
                result["params"] = params
            return result

        return json.dumps(to_dict(program))

    def deserialize(self, s: str) -> DSLProgram:
        """Deserialize program from JSON string."""
        def from_dict(d: Dict) -> DSLProgram:
            args = []
            if "args" in d:
                for a in d["args"]:
                    if isinstance(a, dict):
                        args.append(from_dict(a))
                    else:
                        args.append(a)

            return DSLProgram(
                op=d["op"],
                args=args,
                params=d.get("params", {})
            )

        return from_dict(json.loads(s))

    # ========================================================================
    # Enumeration (for search)
    # ========================================================================

    def enumerate_programs(
        self,
        max_depth: int = 3,
        param_values: Optional[Dict[str, List[Any]]] = None
    ) -> List[DSLProgram]:
        """
        Enumerate all programs up to given depth.

        This is exponential - use sparingly!
        """
        param_values = param_values or self._default_param_values()

        programs = []

        def enumerate_at_depth(depth: int) -> List[DSLProgram]:
            if depth == 0:
                return [self.input_node()]

            result = []
            subprograms = enumerate_at_depth(depth - 1)

            for prim_name, prim in self._primitives.items():
                if len(prim.input_types) == 1:
                    # Unary primitive
                    for sub in subprograms:
                        # Generate parameter combinations
                        for params in self._enumerate_params(prim, param_values):
                            prog = DSLProgram(op=prim_name, args=[sub], params=params)
                            if prog.size <= self.config.max_program_size:
                                result.append(prog)

            return result

        for d in range(1, max_depth + 1):
            programs.extend(enumerate_at_depth(d))

        return programs

    def _default_param_values(self) -> Dict[str, List[Any]]:
        """Default parameter value ranges."""
        return {
            "int": [-2, -1, 0, 1, 2, 3],
            "color": list(range(10)),
            "factor": [2, 3, 4],
            "axis": ["horizontal", "vertical", "both"],
            "direction": ["right", "down", "both"],
            "connectivity": [4, 8],
        }

    def _enumerate_params(
        self,
        primitive: DSLPrimitive,
        param_values: Dict[str, List[Any]]
    ) -> List[Dict[str, Any]]:
        """Enumerate parameter combinations for a primitive."""
        if not primitive.params:
            return [{}]

        # Get values for each parameter
        param_options = []
        param_names = []

        for param in primitive.params:
            name = param["name"]
            ptype = param.get("type", "int")

            if name in param_values:
                values = param_values[name]
            elif ptype in param_values:
                values = param_values[ptype]
            elif "default" in param:
                values = [param["default"]]
            else:
                values = [0]

            param_names.append(name)
            param_options.append(values)

        # Generate combinations
        import itertools
        combinations = []
        for combo in itertools.product(*param_options):
            combinations.append(dict(zip(param_names, combo)))

        return combinations

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def describe_primitive(self, name: str) -> str:
        """Get human-readable description of primitive."""
        prim = self.get_primitive(name)
        if not prim:
            return f"Unknown primitive: {name}"

        params_str = ""
        if prim.params:
            params = [f"{p['name']}: {p.get('type', 'any')}" for p in prim.params]
            params_str = f"({', '.join(params)})"

        return f"{name}{params_str}: {prim.description}"

    def describe_program(self, program: DSLProgram) -> str:
        """Get human-readable description of program."""
        return str(program)

    def get_primitives_by_category(self, category: PrimitiveType) -> List[str]:
        """Get primitive names by category."""
        return [
            name for name, prim in self._primitives.items()
            if prim.category == category
        ]

    def program_cost(self, program: DSLProgram) -> float:
        """Calculate total cost of program."""
        if program.op == "INPUT":
            return 0.0
        if program.op == "CONST":
            return 0.5

        prim = self.get_primitive(program.op)
        cost = prim.cost if prim else 1.0

        for arg in program.args:
            if isinstance(arg, DSLProgram):
                cost += self.program_cost(arg)

        return cost


def create_standard_dsl() -> ARCDSL:
    """Create DSL with standard configuration."""
    return ARCDSL(DSLConfig())


def create_minimal_dsl() -> ARCDSL:
    """Create DSL with only basic operations."""
    return ARCDSL(DSLConfig(
        allowed_categories={PrimitiveType.GEOMETRIC, PrimitiveType.COLOR},
        max_program_depth=4,
        max_program_size=10,
    ))


def create_object_dsl() -> ARCDSL:
    """Create DSL focused on object manipulation."""
    return ARCDSL(DSLConfig(
        allowed_categories={PrimitiveType.OBJECT, PrimitiveType.PATTERN, PrimitiveType.CONTROL},
        max_program_depth=5,
    ))
