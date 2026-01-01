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
    WorkspaceManager,
    get_workspace_manager,
    TaskStatus,
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
    2. Tracks reasoning steps and solutions
    3. Saves and restores state via checkpoints
    """
    print("\n" + "="*60)
    print("DEMO 2: Workspace Management with Persistence")
    print("="*60 + "\n")

    # Create a workspace manager with custom path
    manager = WorkspaceManager(
        workspace_path=Path(__file__).parent / "demo_workspaces",
        persist=True,
    )

    # Create a task with isolated workspace
    task = manager.create_task(
        title="Implement a sorting algorithm",
        description="Implement quicksort in Python",
        metadata={"language": "python", "complexity": "medium"},
    )

    print(f"Created task: {task.id}")
    print(f"Title: {task.title}")

    # Simulate reasoning steps
    manager.add_reasoning_step(
        task.id,
        step_type="user_request",
        content="Please implement quicksort in Python",
    )
    manager.add_reasoning_step(
        task.id,
        step_type="assistant_response",
        content="Here's a quicksort implementation...",
    )
    manager.add_reasoning_step(
        task.id,
        step_type="tool_call",
        content="For optimal pivot selection, use median-of-three...",
        metadata={
            "tool_name": "deep_reasoning",
            "arguments": {"query": "optimal quicksort pivot selection"},
            "success": True,
        },
    )
    manager.update_token_budget(350)  # 150 + 200

    # Get updated task
    task = manager.get_task(task.id)
    print(f"\nTask state:")
    print(f"  Reasoning steps: {len(task.reasoning_history)}")
    print(f"  Status: {task.status.value}")

    # Get session summary
    print(f"\nSession summary:")
    print(manager.get_session_summary())

    # Save checkpoint (persistence)
    checkpoint_path = manager.save_checkpoint("demo_checkpoint")
    print(f"\nSaved checkpoint: {checkpoint_path}")

    # Simulate adding a solution
    manager.add_solution(
        task.id,
        solution={"code": "def quicksort(arr): ..."},
        score=0.85,
    )

    # List all tasks
    print("\nAll tasks:")
    for t in manager.list_tasks():
        print(f"  - {t.id[:8]}...: {t.title} ({t.status.value}, score: {t.score:.2f})")

    # List checkpoints
    print(f"\nAvailable checkpoints: {manager.list_checkpoints()}")

    # Cleanup
    manager.delete_task(task.id)
    print("\nDeleted demo task")


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
    1. Create a task in the workspace manager
    2. Use soft scoring for evaluation
    3. Apply fluid reasoning for iterative improvement
    4. Track everything via reasoning steps and solutions
    """
    print("\n" + "="*60)
    print("DEMO 4: Integrated Fluid Intelligence Pipeline")
    print("="*60 + "\n")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # 1. Create task in workspace manager
    manager = get_workspace_manager()
    task = manager.create_task(
        title="Implement array rotation",
        description="Rotate a 2D list 90 degrees clockwise",
        metadata={"type": "coding", "difficulty": "easy"},
    )
    task_id = task.id

    print(f"Created task: {task.id}")

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
        manager.add_reasoning_step(
            task_id,
            step_type="evaluation",
            content=f"Score: {result.score:.2f}",
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

    # 5. Update task with results
    manager.add_solution(
        task_id,
        solution={"code": result.final_output},
        score=result.best_score,
    )
    manager.add_reasoning_step(
        task_id,
        step_type="result",
        content=result.final_output,
        metadata={
            "success": result.success,
            "iterations": result.iteration_count,
        },
    )
    manager.update_token_budget(result.prompt_tokens + result.completion_tokens)

    # Update task status
    if result.success:
        manager.update_task(task_id, status=TaskStatus.COMPLETED)
    else:
        manager.update_task(task_id, status=TaskStatus.FAILED)

    # 6. Report
    print(f"\nPipeline Results:")
    print(f"  Success: {result.success}")
    print(f"  Final Score: {result.best_score:.2f}")
    print(f"  Iterations: {result.iteration_count}")
    print(f"\nWorkspace Summary:")
    print(manager.get_session_summary())

    # Cleanup
    manager.delete_task(task_id)


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
