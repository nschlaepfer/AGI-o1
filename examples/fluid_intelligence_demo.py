"""
Fluid Intelligence Demo - Combining New Modules with AGI-o1.

This example demonstrates how to use the new modules:
1. FluidReasoner for iterative refinement
2. WorkspaceManager for session state persistence
3. SoftScorer for partial credit evaluation

Together, these enable "fluid intelligence" - the ability to:
- Learn from mistakes within a single session
- Track and persist state across interactions
- Measure partial progress toward goals
"""

import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from openai import OpenAI

from fluid_reasoning import (
    FluidReasoner,
    MultiExpertReasoner,
    ExpertConfig,
    create_simple_evaluator,
)
from workspace_manager import (
    WorkspaceRegistry,
    create_workspace,
    get_workspace,
    get_or_create_workspace,
)
from scoring import (
    SoftScorer,
    code_syntax_scorer,
    keyword_presence_scorer,
    ScoreResult,
    build_feedback,
)

# Load environment
load_dotenv()


def demo_fluid_reasoning():
    """
    Demonstrate fluid reasoning with iterative refinement.

    This shows how the system:
    1. Makes an initial attempt
    2. Gets feedback on what's wrong
    3. Uses feedback to improve
    4. Iterates until success or max iterations
    """
    print("\n" + "="*60)
    print("DEMO 1: Fluid Reasoning with Iterative Refinement")
    print("="*60 + "\n")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Define evaluation criteria
    required_keywords = ["def transform", "np.array", "return"]

    def evaluate_code_solution(solution: str, context: dict) -> tuple:
        """Evaluate a code solution."""
        # Check syntax
        try:
            compile(solution, "<string>", "exec")
            syntax_ok = True
        except SyntaxError as e:
            return False, 0.0, f"Syntax error: {e}"

        # Check for required elements
        keywords_found = sum(1 for kw in required_keywords if kw.lower() in solution.lower())
        keyword_score = keywords_found / len(required_keywords)

        # Check code length (prefer reasonable length)
        lines = len(solution.strip().split("\n"))
        length_score = 1.0 if 5 <= lines <= 50 else 0.5

        # Overall score
        score = 0.6 * keyword_score + 0.2 * length_score + 0.2 * (1.0 if syntax_ok else 0.0)

        success = score >= 0.9

        feedback_parts = []
        if keyword_score < 1.0:
            missing = [kw for kw in required_keywords if kw.lower() not in solution.lower()]
            feedback_parts.append(f"Missing elements: {', '.join(missing)}")
        if not syntax_ok:
            feedback_parts.append("Fix syntax errors")
        if success:
            feedback_parts.append("All checks passed!")

        feedback = " | ".join(feedback_parts) if feedback_parts else "Evaluation complete."

        return success, score, feedback

    # Configure the reasoner
    config = ExpertConfig(
        solver_prompt="""You are an expert Python programmer solving ARC-AGI-style grid transformation tasks.

**Task:**
$$task$$

Write a Python function called `transform` that takes a numpy array as input and returns the transformed array.

Requirements:
1. Import numpy as np
2. Define: def transform(grid: np.ndarray) -> np.ndarray
3. Return a numpy array
4. Include a docstring explaining the transformation

Example format:
```python
import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    \"\"\"Description of transformation.\"\"\"
    # Your implementation
    return result
```
""",
        feedback_prompt="""**Previous Attempts:**
The following are your previous attempts with feedback. Learn from these to improve.

$$feedback$$

Study what went wrong and fix ALL issues in your next attempt.
""",
        model_id="gpt-4o-mini",  # Use a cheaper model for demo
        max_iterations=5,
        temperature=0.7,
        reasoning_effort="",  # Not using reasoning model for demo
    )

    reasoner = FluidReasoner(
        client=client,
        evaluate_fn=evaluate_code_solution,
        config=config,
    )

    # Define a task
    task = """
    Create a transform function that rotates a 2D grid 90 degrees clockwise.

    Example:
    Input:
    [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]]

    Output:
    [[7, 4, 1],
     [8, 5, 2],
     [9, 6, 3]]
    """

    print(f"Task: {task.strip()}")
    print("\nStarting fluid reasoning loop...")
    print("-" * 40)

    # Run the solver
    result = asyncio.run(reasoner.solve(task))

    print(f"\nResult:")
    print(f"  Success: {result.success}")
    print(f"  Best Score: {result.best_score:.2f}")
    print(f"  Iterations: {result.iteration_count}")
    print(f"  Total Solutions Tried: {len(result.solutions)}")

    print("\n" + "-" * 40)
    print("Evolution of Solutions:")
    for i, sol in enumerate(result.solutions):
        print(f"\n  Iteration {sol.iteration}:")
        print(f"    Score: {sol.score:.2f}")
        print(f"    Feedback: {sol.feedback[:100]}...")

    print("\n" + "-" * 40)
    print("Final Solution:")
    print(result.final_output[:500] + "..." if len(result.final_output) > 500 else result.final_output)


def demo_workspace_manager():
    """
    Demonstrate workspace management with persistence.

    This shows how the system:
    1. Creates persistent workspaces for tasks
    2. Tracks messages and metrics
    3. Saves and restores state
    """
    print("\n" + "="*60)
    print("DEMO 2: Workspace Management with Persistence")
    print("="*60 + "\n")

    # Create a workspace registry
    registry = WorkspaceRegistry(
        storage_dir=Path(__file__).parent / "demo_workspaces",
        auto_save=True,
    )

    # Create a workspace for a task
    workspace = registry.create(
        task="Implement a sorting algorithm",
        workspace_id="demo_sort_task",
        context={"language": "python", "complexity": "medium"},
    )

    print(f"Created workspace: {workspace.id}")
    print(f"Task: {workspace.task}")

    # Simulate interaction
    workspace.add_message("user", "Please implement quicksort in Python")
    workspace.add_message("assistant", "Here's a quicksort implementation...")
    workspace.add_tool_call(
        tool_name="deep_reasoning",
        arguments={"query": "optimal quicksort pivot selection"},
        result="For optimal pivot selection, use median-of-three...",
        success=True,
    )
    workspace.update_metrics(prompt_tokens=150, completion_tokens=200, iterations=1)

    print(f"\nWorkspace state:")
    print(f"  Messages: {len(workspace.history)}")
    print(f"  Tool calls: {workspace.metrics.tool_calls}")
    print(f"  Total tokens: {workspace.metrics.total_tokens}")

    # Get context summary
    print(f"\nContext summary:")
    print(workspace.get_context_summary())

    # Simulate detach/reattach (persistence)
    registry.detach("demo_sort_task")
    print("\nDetached from workspace (saved to disk)")

    # Simulate new session - restore from disk
    restored_workspace = registry.get_or_create(
        workspace_id="demo_sort_task",
        task="",  # Will be loaded from disk
    )
    print(f"\nRestored workspace: {restored_workspace.id}")
    print(f"  Restored messages: {len(restored_workspace.history)}")
    print(f"  Restored tool calls: {restored_workspace.metrics.tool_calls}")

    # List all workspaces
    print("\nAll workspaces:")
    for ws_info in registry.list_workspaces():
        print(f"  - {ws_info['id']}: {ws_info['status']} ({ws_info['messages']} messages)")

    # Cleanup
    registry.dispose("demo_sort_task")
    print("\nDisposed demo workspace")


def demo_soft_scoring():
    """
    Demonstrate soft scoring for partial credit.

    This shows how the system:
    1. Evaluates solutions on multiple dimensions
    2. Provides partial credit (not just pass/fail)
    3. Generates actionable feedback
    """
    print("\n" + "="*60)
    print("DEMO 3: Soft Scoring with Partial Credit")
    print("="*60 + "\n")

    # Create a multi-dimensional scorer
    scorer = SoftScorer(success_threshold=0.8)

    # Add scoring criteria
    scorer.add_criterion(
        name="syntax",
        scorer=code_syntax_scorer,
        weight=1.0,
        description="Python syntax must be valid",
    )

    scorer.add_criterion(
        name="structure",
        scorer=keyword_presence_scorer(
            required_keywords=["def", "return"],
            optional_keywords=["class", "import"],
        ),
        weight=0.8,
        description="Must define functions with return statements",
    )

    scorer.add_criterion(
        name="documentation",
        scorer=keyword_presence_scorer(
            required_keywords=['"""', "Args:", "Returns:"],
        ),
        weight=0.5,
        description="Should include docstrings with Args and Returns",
    )

    # Test solutions with varying quality
    solutions = [
        # Good solution
        '''
import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """Rotate grid 90 degrees clockwise.

    Args:
        grid: Input 2D numpy array

    Returns:
        Rotated 2D numpy array
    """
    return np.rot90(grid, k=-1)
''',
        # Partial solution (missing docstring)
        '''
import numpy as np

def transform(grid):
    return np.rot90(grid, k=-1)
''',
        # Poor solution (syntax error)
        '''
def transform(grid)  # Missing colon
    return grid
''',
    ]

    for i, solution in enumerate(solutions):
        print(f"\nSolution {i+1}:")
        print("-" * 40)
        print(solution.strip()[:200])
        print("-" * 40)

        result = scorer.score(solution, {})

        print(f"\nScore: {result.score:.2f} ({'PASS' if result.success else 'FAIL'})")
        print(f"\nDimension Breakdown:")
        for dim, score in result.dimension_scores.items():
            indicator = "+" if score >= 0.8 else "-"
            print(f"  {indicator} {dim}: {score:.2f}")

        if result.improvements:
            print(f"\nImprovements needed:")
            for imp in result.improvements:
                print(f"  - {imp}")

        print()


def demo_integrated_pipeline():
    """
    Demonstrate the full integrated pipeline.

    This combines all modules:
    1. Create a workspace for the task
    2. Use soft scoring for evaluation
    3. Apply fluid reasoning for iterative improvement
    4. Track everything in the workspace
    """
    print("\n" + "="*60)
    print("DEMO 4: Integrated Fluid Intelligence Pipeline")
    print("="*60 + "\n")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # 1. Create workspace for the task
    workspace = create_workspace(
        task="Implement array rotation",
        workspace_id="demo_integrated",
        context={"type": "coding", "difficulty": "easy"},
    )

    print(f"Created workspace: {workspace.id}")

    # 2. Set up soft scorer
    scorer = SoftScorer(success_threshold=0.85)
    scorer.add_criterion("syntax", code_syntax_scorer, weight=1.0)
    scorer.add_criterion(
        "structure",
        keyword_presence_scorer(["def transform", "return"]),
        weight=0.8,
    )

    def evaluate_with_scoring(solution: str, context: dict) -> tuple:
        """Evaluation function using soft scorer."""
        result = scorer.score(solution, context)
        workspace.add_message(
            "evaluation",
            f"Score: {result.score:.2f}",
            metadata={"dimension_scores": result.dimension_scores},
        )
        return result.success, result.score, result.feedback

    # 3. Configure fluid reasoner
    config = ExpertConfig(
        solver_prompt="""Write a Python function to $$task$$

Use this format:
```python
def transform(input_data):
    # Your implementation
    return result
```
""",
        feedback_prompt="Previous attempts:\n$$feedback$$\n\nImprove based on feedback.",
        model_id="gpt-4o-mini",
        max_iterations=3,
    )

    reasoner = FluidReasoner(client, evaluate_with_scoring, config)

    # 4. Run the pipeline
    print("\nRunning fluid intelligence pipeline...")
    result = asyncio.run(reasoner.solve("rotate a 2D list 90 degrees clockwise"))

    # 5. Update workspace with results
    workspace.add_message(
        "result",
        result.final_output,
        metadata={
            "success": result.success,
            "score": result.best_score,
            "iterations": result.iteration_count,
        },
    )
    workspace.update_metrics(
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        iterations=result.iteration_count,
    )

    if result.success:
        workspace.mark_complete("success")
    else:
        workspace.mark_complete("partial")

    # 6. Report
    print(f"\nPipeline Results:")
    print(f"  Success: {result.success}")
    print(f"  Final Score: {result.best_score:.2f}")
    print(f"  Iterations: {result.iteration_count}")
    print(f"\nWorkspace Summary:")
    print(workspace.get_context_summary())

    # Cleanup
    from workspace_manager import get_registry
    get_registry().dispose("demo_integrated")


if __name__ == "__main__":
    print("\n" + "#"*60)
    print("# FLUID INTELLIGENCE DEMONSTRATION")
    print("#"*60)

    # Run demos
    try:
        # Demo 1: Basic fluid reasoning
        # demo_fluid_reasoning()  # Uncomment if you have API key

        # Demo 2: Workspace management
        demo_workspace_manager()

        # Demo 3: Soft scoring
        demo_soft_scoring()

        # Demo 4: Integrated pipeline
        # demo_integrated_pipeline()  # Uncomment if you have API key

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "#"*60)
    print("# DEMONSTRATION COMPLETE")
    print("#"*60 + "\n")
