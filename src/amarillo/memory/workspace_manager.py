"""
Workspace Manager for AGI-o1 - Session and Task State Management.

Inspired by emdash's architecture:
- Multi-layer state: persistent (JSON) + real-time (memory) + activity tracking
- Workspace isolation for parallel task execution
- Activity monitoring with hysteresis (debounced busy/idle states)
- Event-driven state updates

Key patterns borrowed from emdash:
- WorktreeService.ts: Task isolation via separate workspaces
- activityStore.ts: Real-time activity monitoring with timers
- taskTerminalsStore.ts: Per-task state with snapshot-based updates
- DatabaseService.ts: Persistent state with migrations
"""

import json
import logging
import os
import time
import threading
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set
from uuid import uuid4


class TaskStatus(Enum):
    """Task lifecycle states."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    WAITING = "waiting"  # Waiting for user input or external event
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ActivityState(Enum):
    """Agent activity states (from emdash activityStore pattern)."""
    IDLE = "idle"
    BUSY = "busy"
    THINKING = "thinking"
    NEUTRAL = "neutral"


@dataclass
class TaskContext:
    """
    Context for a single task/subtask.

    Borrowed from emdash's Tasks schema with extensions for AGI-o1.
    """
    id: str
    title: str
    description: str = ""
    status: TaskStatus = TaskStatus.PENDING
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    parent_id: Optional[str] = None
    workspace_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Reasoning state
    reasoning_history: List[Dict[str, Any]] = field(default_factory=list)
    solutions: List[Dict[str, Any]] = field(default_factory=list)
    best_solution: Optional[Dict[str, Any]] = None
    score: float = 0.0

    # Activity tracking (from emdash activityStore pattern)
    activity_state: ActivityState = ActivityState.IDLE
    last_activity: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['status'] = self.status.value
        data['activity_state'] = self.activity_state.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskContext':
        """Create from dictionary."""
        data = dict(data)
        if isinstance(data.get('status'), str):
            data['status'] = TaskStatus(data['status'])
        if isinstance(data.get('activity_state'), str):
            data['activity_state'] = ActivityState(data['activity_state'])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class WorkspaceSnapshot:
    """
    Snapshot of workspace state for external store pattern.

    Borrowed from emdash's useSyncExternalStore pattern.
    """
    tasks: List[TaskContext]
    active_task_id: Optional[str]
    global_context: Dict[str, Any]
    token_budget_used: int
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            'tasks': [t.to_dict() for t in self.tasks],
            'active_task_id': self.active_task_id,
            'global_context': self.global_context,
            'token_budget_used': self.token_budget_used,
            'timestamp': self.timestamp
        }


class ActivityMonitor:
    """
    Real-time activity monitoring with hysteresis.

    Borrowed from emdash's activityStore.ts:
    - Two-phase clearing: immediate busy, debounced idle
    - Minimum hold time to prevent UI flicker
    - Multi-listener support
    """

    BUSY_HOLD_MS = 300  # Minimum time to show busy state
    CLEAR_BUSY_MS = 500  # Debounce delay for clearing busy

    def __init__(self):
        self._state: Dict[str, ActivityState] = {}
        self._timers: Dict[str, Optional[threading.Timer]] = {}
        self._last_busy_time: Dict[str, float] = {}
        self._listeners: Dict[str, Set[Callable[[ActivityState], None]]] = {}
        self._lock = threading.Lock()

    def set_activity(self, task_id: str, state: ActivityState) -> None:
        """Set activity state with hysteresis handling."""
        with self._lock:
            current = self._state.get(task_id, ActivityState.IDLE)

            # Cancel any pending timer
            timer = self._timers.get(task_id)
            if timer:
                timer.cancel()
                self._timers[task_id] = None

            if state == ActivityState.BUSY:
                # Immediate transition to busy
                self._state[task_id] = state
                self._last_busy_time[task_id] = time.time()
                self._notify_listeners(task_id, state)

            elif state == ActivityState.IDLE and current == ActivityState.BUSY:
                # Debounced transition to idle
                elapsed = (time.time() - self._last_busy_time.get(task_id, 0)) * 1000
                delay = max(0, self.CLEAR_BUSY_MS - elapsed) / 1000

                def clear_busy():
                    with self._lock:
                        self._state[task_id] = ActivityState.IDLE
                        self._notify_listeners(task_id, ActivityState.IDLE)

                self._timers[task_id] = threading.Timer(delay, clear_busy)
                self._timers[task_id].start()

            else:
                self._state[task_id] = state
                self._notify_listeners(task_id, state)

    def get_activity(self, task_id: str) -> ActivityState:
        """Get current activity state for a task."""
        return self._state.get(task_id, ActivityState.IDLE)

    def subscribe(self, task_id: str, callback: Callable[[ActivityState], None]) -> Callable[[], None]:
        """
        Subscribe to activity state changes.

        Returns unsubscribe function (emdash pattern).
        """
        with self._lock:
            if task_id not in self._listeners:
                self._listeners[task_id] = set()
            self._listeners[task_id].add(callback)

        def unsubscribe():
            with self._lock:
                if task_id in self._listeners:
                    self._listeners[task_id].discard(callback)

        return unsubscribe

    def _notify_listeners(self, task_id: str, state: ActivityState) -> None:
        """Notify all listeners of state change."""
        listeners = self._listeners.get(task_id, set())
        for callback in listeners:
            try:
                callback(state)
            except Exception as e:
                logging.warning("Activity listener error: %s", e)


class WorkspaceManager:
    """
    Manages workspace state, tasks, and persistence.

    Implements multi-layer state management:
    1. Persistent layer (JSON file storage)
    2. Real-time layer (in-memory state)
    3. Activity layer (monitoring with hysteresis)

    Key patterns from emdash:
    - WorktreeService: Task isolation
    - DatabaseService: Persistent state with structure
    - activityStore: Real-time monitoring
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        workspace_path: Optional[Path] = None,
        persist: bool = True
    ):
        """
        Initialize workspace manager.

        Args:
            workspace_path: Base path for workspace data
            persist: Whether to persist state to disk
        """
        self.workspace_path = workspace_path or Path.cwd() / ".workspace"
        self.persist = persist

        # State layers
        self._tasks: Dict[str, TaskContext] = {}
        self._active_task_id: Optional[str] = None
        self._global_context: Dict[str, Any] = {}
        self._token_budget_used: int = 0

        # Activity monitoring
        self._activity_monitor = ActivityMonitor()

        # Listeners for external store pattern
        self._snapshot_listeners: Set[Callable[[WorkspaceSnapshot], None]] = set()

        # Thread safety
        self._lock = threading.RLock()

        # Initialize workspace
        if persist:
            self._ensure_workspace_dirs()
            self._load_state()

    def _ensure_workspace_dirs(self) -> None:
        """Create workspace directories if needed."""
        self.workspace_path.mkdir(parents=True, exist_ok=True)
        (self.workspace_path / "tasks").mkdir(exist_ok=True)
        (self.workspace_path / "snapshots").mkdir(exist_ok=True)

    def _state_file(self) -> Path:
        """Get path to main state file."""
        return self.workspace_path / "state.json"

    def _load_state(self) -> None:
        """Load persisted state from disk."""
        state_file = self._state_file()
        if not state_file.exists():
            return

        try:
            with open(state_file, 'r') as f:
                data = json.load(f)

            self._active_task_id = data.get('active_task_id')
            self._global_context = data.get('global_context', {})
            self._token_budget_used = data.get('token_budget_used', 0)

            for task_data in data.get('tasks', []):
                task = TaskContext.from_dict(task_data)
                self._tasks[task.id] = task

            logging.info("Loaded workspace state: %d tasks", len(self._tasks))

        except Exception as e:
            logging.warning("Failed to load workspace state: %s", e)

    def _save_state(self) -> None:
        """Persist current state to disk."""
        if not self.persist:
            return

        try:
            snapshot = self.get_snapshot()
            with open(self._state_file(), 'w') as f:
                json.dump(snapshot.to_dict(), f, indent=2)

        except Exception as e:
            logging.warning("Failed to save workspace state: %s", e)

    def get_snapshot(self) -> WorkspaceSnapshot:
        """
        Get current workspace snapshot.

        Implements external store snapshot pattern from emdash.
        """
        with self._lock:
            return WorkspaceSnapshot(
                tasks=list(self._tasks.values()),
                active_task_id=self._active_task_id,
                global_context=dict(self._global_context),
                token_budget_used=self._token_budget_used,
                timestamp=datetime.utcnow().isoformat()
            )

    def subscribe(self, callback: Callable[[WorkspaceSnapshot], None]) -> Callable[[], None]:
        """
        Subscribe to workspace state changes.

        Returns unsubscribe function (emdash useSyncExternalStore pattern).
        """
        self._snapshot_listeners.add(callback)

        def unsubscribe():
            self._snapshot_listeners.discard(callback)

        return unsubscribe

    def _notify_change(self) -> None:
        """Notify listeners of state change."""
        snapshot = self.get_snapshot()
        for callback in self._snapshot_listeners:
            try:
                callback(snapshot)
            except Exception as e:
                logging.warning("Snapshot listener error: %s", e)

        self._save_state()

    # ---- Task Management ----

    def create_task(
        self,
        title: str,
        description: str = "",
        parent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> TaskContext:
        """
        Create a new task with isolated workspace.

        Inspired by emdash's WorktreeService - each task gets isolation.
        """
        with self._lock:
            task_id = str(uuid4())

            # Create task-specific workspace (like emdash worktrees)
            task_workspace = self.workspace_path / "tasks" / task_id
            task_workspace.mkdir(parents=True, exist_ok=True)

            task = TaskContext(
                id=task_id,
                title=title,
                description=description,
                parent_id=parent_id,
                workspace_path=str(task_workspace),
                metadata=metadata or {}
            )

            self._tasks[task_id] = task

            # Auto-activate if no active task
            if not self._active_task_id:
                self._active_task_id = task_id

            self._notify_change()
            logging.info("Created task '%s' with workspace: %s", title, task_workspace)

            return task

    def get_task(self, task_id: str) -> Optional[TaskContext]:
        """Get task by ID."""
        return self._tasks.get(task_id)

    def get_active_task(self) -> Optional[TaskContext]:
        """Get currently active task."""
        if self._active_task_id:
            return self._tasks.get(self._active_task_id)
        return None

    def set_active_task(self, task_id: str) -> bool:
        """Set the active task."""
        with self._lock:
            if task_id not in self._tasks:
                return False
            self._active_task_id = task_id
            self._notify_change()
            return True

    def update_task(
        self,
        task_id: str,
        status: Optional[TaskStatus] = None,
        score: Optional[float] = None,
        metadata_update: Optional[Dict[str, Any]] = None
    ) -> Optional[TaskContext]:
        """Update task properties."""
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return None

            if status:
                task.status = status
            if score is not None:
                task.score = score
            if metadata_update:
                task.metadata.update(metadata_update)

            task.updated_at = datetime.utcnow().isoformat()
            self._notify_change()

            return task

    def add_solution(
        self,
        task_id: str,
        solution: Dict[str, Any],
        score: float
    ) -> Optional[TaskContext]:
        """
        Add a solution attempt to a task.

        Implements best-result tracking from poetiq-arc-agi-solver.
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return None

            solution_entry = {
                'solution': solution,
                'score': score,
                'timestamp': datetime.utcnow().isoformat(),
                'iteration': len(task.solutions) + 1
            }
            task.solutions.append(solution_entry)

            # Update best solution if this is better
            if not task.best_solution or score > task.score:
                task.best_solution = solution_entry
                task.score = score

            task.updated_at = datetime.utcnow().isoformat()
            self._notify_change()

            return task

    def add_reasoning_step(
        self,
        task_id: str,
        step_type: str,
        content: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[TaskContext]:
        """Add a reasoning step to task history."""
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return None

            step = {
                'type': step_type,
                'content': content,
                'metadata': metadata or {},
                'timestamp': datetime.utcnow().isoformat()
            }
            task.reasoning_history.append(step)
            task.updated_at = datetime.utcnow().isoformat()

            self._notify_change()
            return task

    def list_tasks(
        self,
        status: Optional[TaskStatus] = None,
        parent_id: Optional[str] = None
    ) -> List[TaskContext]:
        """List tasks with optional filtering."""
        with self._lock:
            tasks = list(self._tasks.values())

            if status:
                tasks = [t for t in tasks if t.status == status]
            if parent_id is not None:
                tasks = [t for t in tasks if t.parent_id == parent_id]

            return sorted(tasks, key=lambda t: t.created_at, reverse=True)

    def delete_task(self, task_id: str) -> bool:
        """Delete a task and its workspace."""
        with self._lock:
            task = self._tasks.pop(task_id, None)
            if not task:
                return False

            # Clean up workspace
            if task.workspace_path:
                import shutil
                try:
                    shutil.rmtree(task.workspace_path)
                except Exception as e:
                    logging.warning("Failed to cleanup task workspace: %s", e)

            # Update active task if needed
            if self._active_task_id == task_id:
                remaining = list(self._tasks.keys())
                self._active_task_id = remaining[0] if remaining else None

            self._notify_change()
            return True

    # ---- Activity Tracking ----

    def set_task_activity(self, task_id: str, state: ActivityState) -> None:
        """Update task activity state with hysteresis."""
        self._activity_monitor.set_activity(task_id, state)

        task = self._tasks.get(task_id)
        if task:
            task.activity_state = state
            task.last_activity = datetime.utcnow().isoformat()

    def get_task_activity(self, task_id: str) -> ActivityState:
        """Get current task activity state."""
        return self._activity_monitor.get_activity(task_id)

    def subscribe_activity(
        self,
        task_id: str,
        callback: Callable[[ActivityState], None]
    ) -> Callable[[], None]:
        """Subscribe to task activity changes."""
        return self._activity_monitor.subscribe(task_id, callback)

    # ---- Global Context ----

    def set_context(self, key: str, value: Any) -> None:
        """Set global context value."""
        with self._lock:
            self._global_context[key] = value
            self._notify_change()

    def get_context(self, key: str, default: Any = None) -> Any:
        """Get global context value."""
        return self._global_context.get(key, default)

    def update_token_budget(self, tokens: int) -> None:
        """Update token budget usage."""
        with self._lock:
            self._token_budget_used += tokens
            self._notify_change()

    # ---- Checkpoint/Restore ----

    def save_checkpoint(self, name: Optional[str] = None) -> Path:
        """
        Save a named checkpoint of current state.

        Inspired by emdash's checkpoint pattern for session recovery.
        """
        checkpoint_name = name or f"checkpoint_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        checkpoint_path = self.workspace_path / "snapshots" / f"{checkpoint_name}.json"

        snapshot = self.get_snapshot()
        with open(checkpoint_path, 'w') as f:
            json.dump(snapshot.to_dict(), f, indent=2)

        logging.info("Saved checkpoint: %s", checkpoint_path)
        return checkpoint_path

    def restore_checkpoint(self, name: str) -> bool:
        """Restore from a named checkpoint."""
        checkpoint_path = self.workspace_path / "snapshots" / f"{name}.json"

        if not checkpoint_path.exists():
            logging.warning("Checkpoint not found: %s", name)
            return False

        try:
            with open(checkpoint_path, 'r') as f:
                data = json.load(f)

            with self._lock:
                self._tasks.clear()
                for task_data in data.get('tasks', []):
                    task = TaskContext.from_dict(task_data)
                    self._tasks[task.id] = task

                self._active_task_id = data.get('active_task_id')
                self._global_context = data.get('global_context', {})
                self._token_budget_used = data.get('token_budget_used', 0)

            self._notify_change()
            logging.info("Restored checkpoint: %s", name)
            return True

        except Exception as e:
            logging.error("Failed to restore checkpoint: %s", e)
            return False

    def list_checkpoints(self) -> List[str]:
        """List available checkpoints."""
        snapshots_dir = self.workspace_path / "snapshots"
        if not snapshots_dir.exists():
            return []
        return [p.stem for p in snapshots_dir.glob("*.json")]

    # ---- Session Summary ----

    def get_session_summary(self) -> str:
        """Get human-readable session summary."""
        snapshot = self.get_snapshot()

        lines = [
            f"=== Workspace Session Summary ===",
            f"Tasks: {len(snapshot.tasks)}",
            f"Active Task: {snapshot.active_task_id or 'None'}",
            f"Token Budget Used: {snapshot.token_budget_used:,}",
            ""
        ]

        for task in snapshot.tasks[:5]:  # Show top 5
            status_icon = {
                TaskStatus.PENDING: "⏳",
                TaskStatus.IN_PROGRESS: "🔄",
                TaskStatus.COMPLETED: "✅",
                TaskStatus.FAILED: "❌",
                TaskStatus.PAUSED: "⏸️",
                TaskStatus.WAITING: "⏸️",
                TaskStatus.CANCELLED: "🚫"
            }.get(task.status, "❓")

            active = " [ACTIVE]" if task.id == snapshot.active_task_id else ""
            lines.append(f"  {status_icon} {task.title} (score: {task.score:.2f}){active}")

        if len(snapshot.tasks) > 5:
            lines.append(f"  ... and {len(snapshot.tasks) - 5} more tasks")

        return "\n".join(lines)


# ---- Global Instance ----

_workspace_manager: Optional[WorkspaceManager] = None


def get_workspace_manager(workspace_path: Optional[Path] = None) -> WorkspaceManager:
    """Get or create the global workspace manager instance."""
    global _workspace_manager
    if _workspace_manager is None:
        _workspace_manager = WorkspaceManager(workspace_path)
    return _workspace_manager


def reset_workspace_manager(workspace_path: Optional[Path] = None) -> WorkspaceManager:
    """Reset and recreate the workspace manager."""
    global _workspace_manager
    _workspace_manager = WorkspaceManager(workspace_path)
    return _workspace_manager
