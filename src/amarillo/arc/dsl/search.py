"""
Program Search Algorithms for ARC DSL.

Implements multiple search strategies:
- Beam Search: Fast, breadth-first with pruning
- Genetic Search: Evolutionary approach with crossover/mutation
- Neural-Guided Search: Use LLM to guide program construction
- Hybrid Search: Combine multiple strategies
"""

from __future__ import annotations

import heapq
import logging
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
from numpy.typing import NDArray

from .primitives import DSLProgram, get_all_primitives
from .language import ARCDSL

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Result from program search."""
    program: Optional[DSLProgram]
    success: bool
    score: float
    programs_evaluated: int
    elapsed_seconds: float
    all_candidates: List[Tuple[float, DSLProgram]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ProgramSearcher(ABC):
    """Abstract base class for program search algorithms."""

    def __init__(self, dsl: ARCDSL):
        self.dsl = dsl

    @abstractmethod
    def search(
        self,
        examples: List[Tuple[NDArray, NDArray]],
        max_time_seconds: float = 60.0,
        max_programs: int = 10000,
    ) -> SearchResult:
        """
        Search for program that solves all examples.

        Args:
            examples: List of (input, output) array pairs
            max_time_seconds: Time budget
            max_programs: Maximum programs to evaluate

        Returns:
            SearchResult with best program found
        """
        pass

    def evaluate_program(
        self,
        program: DSLProgram,
        examples: List[Tuple[NDArray, NDArray]]
    ) -> float:
        """Evaluate program on examples, return accuracy."""
        if not examples:
            return 0.0

        correct = 0
        for input_arr, expected in examples:
            try:
                output = self.dsl.execute(program, input_arr)
                if np.array_equal(output, expected):
                    correct += 1
            except Exception:
                pass

        return correct / len(examples)


class BeamSearch(ProgramSearcher):
    """
    Beam search for program synthesis.

    Maintains top-k programs at each depth level,
    expanding with primitives and pruning by score.
    """

    def __init__(
        self,
        dsl: ARCDSL,
        beam_width: int = 100,
        max_depth: int = 5,
        use_cost_penalty: bool = True,
    ):
        super().__init__(dsl)
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.use_cost_penalty = use_cost_penalty

    def search(
        self,
        examples: List[Tuple[NDArray, NDArray]],
        max_time_seconds: float = 60.0,
        max_programs: int = 10000,
    ) -> SearchResult:
        start_time = time.time()
        programs_evaluated = 0
        best_score = -1.0
        best_program = None
        all_candidates = []

        # Initialize beam with input node
        beam: List[Tuple[float, DSLProgram]] = [(0.0, self.dsl.input_node())]

        for depth in range(self.max_depth):
            # Check time
            if time.time() - start_time > max_time_seconds:
                break

            new_beam = []

            for _, program in beam:
                # Expand with all primitives
                for prim_name, prim in self.dsl.primitives.items():
                    if len(prim.input_types) != 1:
                        continue  # Only unary for now

                    # Generate parameter combinations
                    param_combos = self._get_param_combos(prim)

                    for params in param_combos:
                        if programs_evaluated >= max_programs:
                            break

                        new_prog = DSLProgram(
                            op=prim_name,
                            args=[program],
                            params=params
                        )

                        # Validate
                        valid, _ = self.dsl.validate(new_prog)
                        if not valid:
                            continue

                        # Evaluate
                        score = self.evaluate_program(new_prog, examples)
                        programs_evaluated += 1

                        # Apply cost penalty
                        if self.use_cost_penalty:
                            cost = self.dsl.program_cost(new_prog)
                            adjusted_score = score - 0.01 * cost
                        else:
                            adjusted_score = score

                        # Track
                        all_candidates.append((score, new_prog))

                        if score > best_score:
                            best_score = score
                            best_program = new_prog

                            if score == 1.0:
                                return SearchResult(
                                    program=best_program,
                                    success=True,
                                    score=1.0,
                                    programs_evaluated=programs_evaluated,
                                    elapsed_seconds=time.time() - start_time,
                                    all_candidates=sorted(all_candidates, reverse=True)[:100],
                                )

                        new_beam.append((adjusted_score, new_prog))

                    if programs_evaluated >= max_programs:
                        break

                if programs_evaluated >= max_programs:
                    break

            # Prune to beam width
            new_beam.sort(reverse=True)
            beam = new_beam[:self.beam_width]

            if not beam:
                break

            logger.debug(f"Depth {depth + 1}: beam size {len(beam)}, best score {best_score:.3f}")

        return SearchResult(
            program=best_program,
            success=best_score == 1.0,
            score=best_score,
            programs_evaluated=programs_evaluated,
            elapsed_seconds=time.time() - start_time,
            all_candidates=sorted(all_candidates, reverse=True)[:100],
        )

    def _get_param_combos(self, prim) -> List[Dict[str, Any]]:
        """Get parameter combinations for primitive."""
        if not prim.params:
            return [{}]

        # Use common values
        param_values = {
            "int": [0, 1, 2, -1],
            "color": [0, 1, 2, 3, 4, 5],
            "factor": [2, 3],
            "axis": ["horizontal", "vertical"],
            "direction": ["right", "down"],
            "connectivity": [4],
            "background": [0],
        }

        import itertools
        param_options = []
        param_names = []

        for p in prim.params:
            name = p["name"]
            ptype = p.get("type", "int")

            if name in param_values:
                values = param_values[name]
            elif ptype in param_values:
                values = param_values[ptype]
            elif "default" in p:
                values = [p["default"]]
            else:
                values = [0]

            param_names.append(name)
            param_options.append(values[:4])  # Limit options

        combos = []
        for combo in itertools.product(*param_options):
            combos.append(dict(zip(param_names, combo)))

        return combos[:20]  # Limit total combinations


class GeneticSearch(ProgramSearcher):
    """
    Genetic algorithm for program synthesis.

    Uses:
    - Tournament selection
    - Subtree crossover
    - Point mutation
    - Population diversity maintenance
    """

    def __init__(
        self,
        dsl: ARCDSL,
        population_size: int = 100,
        elite_size: int = 10,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.7,
        tournament_size: int = 5,
    ):
        super().__init__(dsl)
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size

    def search(
        self,
        examples: List[Tuple[NDArray, NDArray]],
        max_time_seconds: float = 60.0,
        max_programs: int = 10000,
    ) -> SearchResult:
        start_time = time.time()
        programs_evaluated = 0
        best_score = -1.0
        best_program = None
        all_candidates = []

        # Initialize population
        population = self._initialize_population()
        programs_evaluated += len(population)

        # Evaluate initial population
        fitness = []
        for prog in population:
            score = self.evaluate_program(prog, examples)
            fitness.append(score)
            all_candidates.append((score, prog))

            if score > best_score:
                best_score = score
                best_program = prog

            if score == 1.0:
                return SearchResult(
                    program=prog,
                    success=True,
                    score=1.0,
                    programs_evaluated=programs_evaluated,
                    elapsed_seconds=time.time() - start_time,
                    all_candidates=sorted(all_candidates, reverse=True)[:100],
                )

        generation = 0

        while (time.time() - start_time < max_time_seconds and
               programs_evaluated < max_programs):

            generation += 1

            # Create next generation
            new_population = []

            # Elitism
            elite_indices = sorted(range(len(fitness)), key=lambda i: fitness[i], reverse=True)[:self.elite_size]
            for i in elite_indices:
                new_population.append(population[i])

            # Fill rest with crossover and mutation
            while len(new_population) < self.population_size:
                if random.random() < self.crossover_rate:
                    # Crossover
                    parent1 = self._tournament_select(population, fitness)
                    parent2 = self._tournament_select(population, fitness)
                    child = self._crossover(parent1, parent2)
                else:
                    # Clone
                    child = self._tournament_select(population, fitness)

                # Mutation
                if random.random() < self.mutation_rate:
                    child = self._mutate(child)

                # Validate
                valid, _ = self.dsl.validate(child)
                if valid:
                    new_population.append(child)

            population = new_population

            # Evaluate
            fitness = []
            for prog in population:
                score = self.evaluate_program(prog, examples)
                fitness.append(score)
                programs_evaluated += 1
                all_candidates.append((score, prog))

                if score > best_score:
                    best_score = score
                    best_program = prog

                if score == 1.0:
                    return SearchResult(
                        program=prog,
                        success=True,
                        score=1.0,
                        programs_evaluated=programs_evaluated,
                        elapsed_seconds=time.time() - start_time,
                        all_candidates=sorted(all_candidates, reverse=True)[:100],
                        metadata={"generations": generation},
                    )

            logger.debug(f"Generation {generation}: best {best_score:.3f}, avg {np.mean(fitness):.3f}")

        return SearchResult(
            program=best_program,
            success=best_score == 1.0,
            score=best_score,
            programs_evaluated=programs_evaluated,
            elapsed_seconds=time.time() - start_time,
            all_candidates=sorted(all_candidates, reverse=True)[:100],
            metadata={"generations": generation},
        )

    def _initialize_population(self) -> List[DSLProgram]:
        """Create initial random population."""
        population = []

        for _ in range(self.population_size):
            prog = self._random_program(max_depth=3)
            population.append(prog)

        return population

    def _random_program(self, max_depth: int = 3) -> DSLProgram:
        """Generate random program."""
        if max_depth <= 1 or random.random() < 0.3:
            return self.dsl.input_node()

        # Choose random primitive
        prim_names = list(self.dsl.primitives.keys())
        prim_name = random.choice(prim_names)
        prim = self.dsl.get_primitive(prim_name)

        # Generate random parameters
        params = {}
        for p in prim.params:
            ptype = p.get("type", "int")
            if ptype == "int":
                params[p["name"]] = random.randint(-2, 5)
            elif ptype == "color":
                params[p["name"]] = random.randint(0, 9)
            elif ptype == "str":
                options = p.get("options", ["horizontal", "vertical"])
                params[p["name"]] = random.choice(options)
            elif "default" in p:
                params[p["name"]] = p["default"]

        # Recursive for arguments
        args = []
        for _ in range(len(prim.input_types)):
            args.append(self._random_program(max_depth - 1))

        return DSLProgram(op=prim_name, args=args, params=params)

    def _tournament_select(
        self,
        population: List[DSLProgram],
        fitness: List[float]
    ) -> DSLProgram:
        """Tournament selection."""
        indices = random.sample(range(len(population)), min(self.tournament_size, len(population)))
        best_idx = max(indices, key=lambda i: fitness[i])
        return population[best_idx]

    def _crossover(self, parent1: DSLProgram, parent2: DSLProgram) -> DSLProgram:
        """Subtree crossover."""
        # For simplicity, just swap a random subtree
        if parent1.op == "INPUT":
            return parent2
        if parent2.op == "INPUT":
            return parent1

        # Pick random crossover point in parent1
        if parent1.args and random.random() < 0.5:
            new_args = list(parent1.args)
            idx = random.randint(0, len(new_args) - 1)
            new_args[idx] = parent2 if random.random() < 0.3 else self._random_subtree(parent2)
            return DSLProgram(op=parent1.op, args=new_args, params=parent1.params.copy())

        return parent1

    def _random_subtree(self, program: DSLProgram) -> DSLProgram:
        """Extract random subtree."""
        if program.op == "INPUT" or not program.args:
            return program

        if random.random() < 0.5:
            return program

        idx = random.randint(0, len(program.args) - 1)
        arg = program.args[idx]
        if isinstance(arg, DSLProgram):
            return self._random_subtree(arg)
        return program

    def _mutate(self, program: DSLProgram) -> DSLProgram:
        """Point mutation."""
        if program.op == "INPUT":
            # Replace with random primitive
            return self._random_program(max_depth=2)

        if program.op == "CONST":
            return program

        # Mutate parameters
        if program.params and random.random() < 0.5:
            new_params = program.params.copy()
            key = random.choice(list(new_params.keys()))
            val = new_params[key]
            if isinstance(val, int):
                new_params[key] = val + random.randint(-2, 2)
            return DSLProgram(op=program.op, args=program.args, params=new_params)

        # Mutate primitive
        if random.random() < 0.3:
            prim_names = list(self.dsl.primitives.keys())
            new_op = random.choice(prim_names)
            new_prim = self.dsl.get_primitive(new_op)

            # Keep compatible args
            if len(new_prim.input_types) == len(program.args):
                return DSLProgram(op=new_op, args=program.args, params={})

        # Mutate random arg
        if program.args:
            new_args = list(program.args)
            idx = random.randint(0, len(new_args) - 1)
            if isinstance(new_args[idx], DSLProgram):
                new_args[idx] = self._mutate(new_args[idx])
            return DSLProgram(op=program.op, args=new_args, params=program.params)

        return program


class NeuralGuidedSearch(ProgramSearcher):
    """
    Neural-guided program search using LLM.

    Uses LLM to:
    - Suggest which primitives to try
    - Provide parameter values
    - Guide search direction based on feedback
    """

    def __init__(
        self,
        dsl: ARCDSL,
        llm_client,
        model: str = "gpt-4o-mini",
        beam_width: int = 20,
    ):
        super().__init__(dsl)
        self.llm_client = llm_client
        self.model = model
        self.beam_width = beam_width

    def search(
        self,
        examples: List[Tuple[NDArray, NDArray]],
        max_time_seconds: float = 60.0,
        max_programs: int = 1000,
    ) -> SearchResult:
        start_time = time.time()
        programs_evaluated = 0
        best_score = -1.0
        best_program = None
        all_candidates = []

        # Format examples for LLM
        examples_str = self._format_examples(examples)

        # Get initial suggestions from LLM
        suggestions = self._get_suggestions(examples_str, [])

        # Parse and evaluate suggestions
        for suggestion in suggestions:
            if programs_evaluated >= max_programs:
                break
            if time.time() - start_time > max_time_seconds:
                break

            program = self._parse_suggestion(suggestion)
            if program is None:
                continue

            valid, _ = self.dsl.validate(program)
            if not valid:
                continue

            score = self.evaluate_program(program, examples)
            programs_evaluated += 1
            all_candidates.append((score, program))

            if score > best_score:
                best_score = score
                best_program = program

            if score == 1.0:
                return SearchResult(
                    program=program,
                    success=True,
                    score=1.0,
                    programs_evaluated=programs_evaluated,
                    elapsed_seconds=time.time() - start_time,
                    all_candidates=sorted(all_candidates, reverse=True)[:100],
                )

        # Iterative refinement
        iteration = 0
        while (time.time() - start_time < max_time_seconds and
               programs_evaluated < max_programs and
               iteration < 5):

            iteration += 1

            # Get more suggestions based on best so far
            history = [(s, str(p)) for s, p in sorted(all_candidates, reverse=True)[:5]]
            suggestions = self._get_suggestions(examples_str, history)

            for suggestion in suggestions:
                if programs_evaluated >= max_programs:
                    break

                program = self._parse_suggestion(suggestion)
                if program is None:
                    continue

                valid, _ = self.dsl.validate(program)
                if not valid:
                    continue

                score = self.evaluate_program(program, examples)
                programs_evaluated += 1
                all_candidates.append((score, program))

                if score > best_score:
                    best_score = score
                    best_program = program

                if score == 1.0:
                    return SearchResult(
                        program=program,
                        success=True,
                        score=1.0,
                        programs_evaluated=programs_evaluated,
                        elapsed_seconds=time.time() - start_time,
                        all_candidates=sorted(all_candidates, reverse=True)[:100],
                    )

        return SearchResult(
            program=best_program,
            success=best_score == 1.0,
            score=best_score,
            programs_evaluated=programs_evaluated,
            elapsed_seconds=time.time() - start_time,
            all_candidates=sorted(all_candidates, reverse=True)[:100],
        )

    def _format_examples(self, examples: List[Tuple[NDArray, NDArray]]) -> str:
        """Format examples for LLM."""
        lines = ["Input/Output Examples:"]
        for i, (inp, out) in enumerate(examples):
            lines.append(f"\nExample {i+1}:")
            lines.append(f"Input ({inp.shape}):")
            lines.append(str(inp.tolist()))
            lines.append(f"Output ({out.shape}):")
            lines.append(str(out.tolist()))
        return "\n".join(lines)

    def _get_suggestions(
        self,
        examples_str: str,
        history: List[Tuple[float, str]]
    ) -> List[str]:
        """Get program suggestions from LLM."""
        primitives_desc = "\n".join([
            f"- {name}: {self.dsl.describe_primitive(name)}"
            for name in self.dsl.primitive_names
        ])

        history_str = ""
        if history:
            history_str = "\n\nPrevious attempts:\n"
            for score, prog_str in history:
                history_str += f"Score {score:.2f}: {prog_str}\n"

        prompt = f"""You are a program synthesis expert. Given input/output examples, suggest DSL programs that transform inputs to outputs.

Available primitives:
{primitives_desc}

{examples_str}
{history_str}

Suggest 5 different programs that might solve this transformation.
Format each as: PROGRAM: <program_expression>

Example formats:
PROGRAM: rotate90(input)
PROGRAM: recolor(flip_h(input), old_color=1, new_color=2)
PROGRAM: crop_to_content(scale(input, factor=2))

Focus on the transformation pattern and use appropriate primitives.
"""

        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1024,
            )
            content = response.choices[0].message.content or ""

            # Extract programs
            import re
            matches = re.findall(r"PROGRAM:\s*(.+?)(?:\n|$)", content)
            return matches

        except Exception as e:
            logger.warning(f"LLM call failed: {e}")
            return []

    def _parse_suggestion(self, suggestion: str) -> Optional[DSLProgram]:
        """Parse program from suggestion string."""
        try:
            suggestion = suggestion.strip()

            # Simple recursive parser
            def parse_expr(s: str) -> Tuple[Optional[DSLProgram], str]:
                s = s.strip()

                if s.startswith("input"):
                    rest = s[5:].strip()
                    return DSLProgram(op="INPUT"), rest

                # Find function name
                match = re.match(r"(\w+)\s*\(", s)
                if not match:
                    return None, s

                func_name = match.group(1)
                rest = s[match.end():]

                # Parse arguments
                args = []
                params = {}

                while rest and not rest.startswith(")"):
                    rest = rest.strip()
                    if rest.startswith(","):
                        rest = rest[1:].strip()

                    # Check for named parameter
                    param_match = re.match(r"(\w+)\s*=\s*", rest)
                    if param_match:
                        param_name = param_match.group(1)
                        rest = rest[param_match.end():]

                        # Parse value
                        val_match = re.match(r"(['\"]?)([^'\"(),]+)\1", rest)
                        if val_match:
                            val = val_match.group(2)
                            rest = rest[val_match.end():]

                            # Convert type
                            try:
                                val = int(val)
                            except ValueError:
                                pass

                            params[param_name] = val
                    else:
                        # Positional argument (subexpression)
                        sub_prog, rest = parse_expr(rest)
                        if sub_prog:
                            args.append(sub_prog)

                    rest = rest.strip()
                    if rest.startswith(","):
                        rest = rest[1:]

                # Skip closing paren
                if rest.startswith(")"):
                    rest = rest[1:]

                return DSLProgram(op=func_name, args=args, params=params), rest

            program, _ = parse_expr(suggestion)
            return program

        except Exception as e:
            logger.debug(f"Parse failed for '{suggestion}': {e}")
            return None


class HybridSearch(ProgramSearcher):
    """
    Hybrid search combining multiple strategies.

    Runs different searchers in parallel/sequence and
    combines results.
    """

    def __init__(
        self,
        dsl: ARCDSL,
        searchers: Optional[List[ProgramSearcher]] = None,
    ):
        super().__init__(dsl)
        self.searchers = searchers or [
            BeamSearch(dsl, beam_width=50),
            GeneticSearch(dsl, population_size=50),
        ]

    def search(
        self,
        examples: List[Tuple[NDArray, NDArray]],
        max_time_seconds: float = 60.0,
        max_programs: int = 10000,
    ) -> SearchResult:
        start_time = time.time()
        total_evaluated = 0
        best_score = -1.0
        best_program = None
        all_candidates = []

        # Allocate time/budget to each searcher
        time_per_searcher = max_time_seconds / len(self.searchers)
        programs_per_searcher = max_programs // len(self.searchers)

        for searcher in self.searchers:
            remaining_time = max_time_seconds - (time.time() - start_time)
            if remaining_time <= 0:
                break

            result = searcher.search(
                examples,
                max_time_seconds=min(time_per_searcher, remaining_time),
                max_programs=programs_per_searcher,
            )

            total_evaluated += result.programs_evaluated
            all_candidates.extend(result.all_candidates)

            if result.score > best_score:
                best_score = result.score
                best_program = result.program

            if result.success:
                return SearchResult(
                    program=best_program,
                    success=True,
                    score=1.0,
                    programs_evaluated=total_evaluated,
                    elapsed_seconds=time.time() - start_time,
                    all_candidates=sorted(all_candidates, reverse=True)[:100],
                    metadata={"winning_searcher": type(searcher).__name__},
                )

        return SearchResult(
            program=best_program,
            success=best_score == 1.0,
            score=best_score,
            programs_evaluated=total_evaluated,
            elapsed_seconds=time.time() - start_time,
            all_candidates=sorted(all_candidates, reverse=True)[:100],
        )
