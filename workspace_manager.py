"""
Workspace Manager - Session State Management with Persistence.

Inspired by Emdash's SessionRegistry and TerminalSessionManager, this module provides:
- Registry pattern for workspace management
- State snapshot and restoration
- Activity tracking and event listeners
- Workspace lifecycle management (create, attach, detach, dispose)

This enables persistent, resumable agent sessions with proper state management.
"""

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_SNAPSHOT_INTERVAL_SECONDS = 120  # 2 minutes
DEFAULT_MAX_HISTORY_ITEMS = 100
WORKSPACE_DIR = Path(__file__).parent / "workspaces"


@dataclass
class WorkspaceSnapshot:
    """Snapshot of workspace state for persistence."""
    version: int = 1
    created_at: str = ""
    workspace_id: str = ""
    task: str = ""
    status: str = "active"
    history: List[Dict[str, Any]] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkspaceMetrics:
    """Metrics tracking for a workspace."""
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tool_calls: int = 0
    iterations: int = 0
    start_time: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    snapshot_count: int = 0


class Workspace:
    """
    Represents a single agent workspace/session.

    Inspired by TerminalSessionManager:
    - Manages lifecycle (create, attach, detach, dispose)
    - Tracks activity and metrics
    - Supports snapshot and restore
    - Emits events for state changes
    """

    def __init__(
        self,
        workspace_id: str,
        task: str,
        context: Optional[Dict[str, Any]] = None,
        max_history: int = DEFAULT_MAX_HISTORY_ITEMS,
    ):
        self.id = workspace_id
        self.task = task
        self.context = context or {}
        self.max_history = max_history

        self.history: List[Dict[str, Any]] = []
        self.metrics = WorkspaceMetrics()
        self.status = "active"

        self._disposed = False
        self._attached = False

        # Event listeners
        self._activity_listeners: Set[Callable[[], None]] = set()
        self._complete_listeners: Set[Callable[[str], None]] = set()
        self._error_listeners: Set[Callable[[str], None]] = set()

        # Snapshot timer
        self._last_snapshot_at: Optional[float] = None
        self._last_snapshot_reason: Optional[str] = None

    def attach(self) -> None:
        """Attach to this workspace (make it active)."""
        if self._disposed:
            raise RuntimeError(f"Workspace {self.id} is disposed")
        self._attached = True
        self._emit_activity()
        logger.info(f"Attached to workspace {self.id}")

    def detach(self) -> None:
        """Detach from this workspace."""
        if self._attached:
            self._attached = False
            self._capture_snapshot("detach")
            logger.info(f"Detached from workspace {self.id}")

    def dispose(self) -> None:
        """Dispose of this workspace and clean up resources."""
        if self._disposed:
            return
        self._disposed = True
        self.detach()
        self._capture_snapshot("dispose")
        self._activity_listeners.clear()
        self._complete_listeners.clear()
        self._error_listeners.clear()
        logger.info(f"Disposed workspace {self.id}")

    def add_message(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a message to the workspace history."""
        if self._disposed:
            raise RuntimeError(f"Workspace {self.id} is disposed")

        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }
        self.history.append(message)

        # Trim history if needed
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

        self.metrics.last_activity = time.time()
        self._emit_activity()

    def add_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        result: str,
        success: bool = True,
    ) -> None:
        """Add a tool call record to history."""
        self.add_message(
            role="tool",
            content=result,
            metadata={
                "tool_name": tool_name,
                "arguments": arguments,
                "success": success,
            },
        )
        self.metrics.tool_calls += 1

    def update_metrics(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        iterations: int = 0,
    ) -> None:
        """Update workspace metrics."""
        self.metrics.prompt_tokens += prompt_tokens
        self.metrics.completion_tokens += completion_tokens
        self.metrics.total_tokens += prompt_tokens + completion_tokens
        self.metrics.iterations += iterations
        self.metrics.last_activity = time.time()

    def mark_complete(self, outcome: str = "success") -> None:
        """Mark the workspace as complete."""
        self.status = "completed"
        self._emit_complete(outcome)
        self._capture_snapshot("complete")

    def mark_error(self, error: str) -> None:
        """Mark the workspace as having an error."""
        self.status = "error"
        self._emit_error(error)
        self._capture_snapshot("error")

    def get_context_summary(self) -> str:
        """Get a summary of the workspace context for prompts."""
        lines = [f"Workspace: {self.id}", f"Task: {self.task}"]

        if self.context:
            lines.append("Context:")
            for key, value in self.context.items():
                lines.append(f"  {key}: {value}")

        lines.append(f"Status: {self.status}")
        lines.append(f"Messages: {len(self.history)}")
        lines.append(f"Tool calls: {self.metrics.tool_calls}")

        return "\n".join(lines)

    def get_recent_history(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get the most recent history items."""
        return self.history[-n:]

    def to_snapshot(self) -> WorkspaceSnapshot:
        """Create a snapshot of the workspace state."""
        return WorkspaceSnapshot(
            version=1,
            created_at=datetime.now().isoformat(),
            workspace_id=self.id,
            task=self.task,
            status=self.status,
            history=list(self.history),
            context=dict(self.context),
            metrics={
                "total_tokens": self.metrics.total_tokens,
                "prompt_tokens": self.metrics.prompt_tokens,
                "completion_tokens": self.metrics.completion_tokens,
                "tool_calls": self.metrics.tool_calls,
                "iterations": self.metrics.iterations,
                "start_time": self.metrics.start_time,
                "last_activity": self.metrics.last_activity,
                "snapshot_count": self.metrics.snapshot_count,
            },
        )

    @classmethod
    def from_snapshot(cls, snapshot: WorkspaceSnapshot) -> "Workspace":
        """Restore a workspace from a snapshot."""
        workspace = cls(
            workspace_id=snapshot.workspace_id,
            task=snapshot.task,
            context=snapshot.context,
        )
        workspace.status = snapshot.status
        workspace.history = list(snapshot.history)

        # Restore metrics
        if snapshot.metrics:
            workspace.metrics.total_tokens = snapshot.metrics.get("total_tokens", 0)
            workspace.metrics.prompt_tokens = snapshot.metrics.get("prompt_tokens", 0)
            workspace.metrics.completion_tokens = snapshot.metrics.get("completion_tokens", 0)
            workspace.metrics.tool_calls = snapshot.metrics.get("tool_calls", 0)
            workspace.metrics.iterations = snapshot.metrics.get("iterations", 0)
            workspace.metrics.start_time = snapshot.metrics.get("start_time", time.time())
            workspace.metrics.last_activity = snapshot.metrics.get("last_activity", time.time())
            workspace.metrics.snapshot_count = snapshot.metrics.get("snapshot_count", 0)

        return workspace

    def _capture_snapshot(self, reason: str) -> None:
        """Capture and persist a snapshot."""
        self.metrics.snapshot_count += 1
        self._last_snapshot_at = time.time()
        self._last_snapshot_reason = reason
        # Note: Actual persistence is handled by WorkspaceRegistry

    def _emit_activity(self) -> None:
        """Emit activity event to listeners."""
        for listener in self._activity_listeners:
            try:
                listener()
            except Exception as e:
                logger.warning(f"Activity listener error: {e}")

    def _emit_complete(self, outcome: str) -> None:
        """Emit completion event to listeners."""
        for listener in self._complete_listeners:
            try:
                listener(outcome)
            except Exception as e:
                logger.warning(f"Complete listener error: {e}")

    def _emit_error(self, error: str) -> None:
        """Emit error event to listeners."""
        for listener in self._error_listeners:
            try:
                listener(error)
            except Exception as e:
                logger.warning(f"Error listener error: {e}")

    # Listener registration methods

    def on_activity(self, callback: Callable[[], None]) -> Callable[[], None]:
        """Register an activity listener. Returns unsubscribe function."""
        self._activity_listeners.add(callback)
        return lambda: self._activity_listeners.discard(callback)

    def on_complete(self, callback: Callable[[str], None]) -> Callable[[], None]:
        """Register a completion listener. Returns unsubscribe function."""
        self._complete_listeners.add(callback)
        return lambda: self._complete_listeners.discard(callback)

    def on_error(self, callback: Callable[[str], None]) -> Callable[[], None]:
        """Register an error listener. Returns unsubscribe function."""
        self._error_listeners.add(callback)
        return lambda: self._error_listeners.discard(callback)


class WorkspaceRegistry:
    """
    Registry for managing multiple workspaces.

    Inspired by Emdash's SessionRegistry:
    - getOrCreate pattern for workspace access
    - Lifecycle management (attach, detach, dispose)
    - Persistence via snapshots
    - Global operations (disposeAll)
    """

    def __init__(
        self,
        storage_dir: Optional[Path] = None,
        auto_save: bool = True,
    ):
        self.storage_dir = storage_dir or WORKSPACE_DIR
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.auto_save = auto_save

        self._workspaces: Dict[str, Workspace] = {}
        self._lock = Lock()

    def create(
        self,
        task: str,
        workspace_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Workspace:
        """Create a new workspace."""
        with self._lock:
            workspace_id = workspace_id or f"ws_{uuid.uuid4().hex[:8]}"

            if workspace_id in self._workspaces:
                raise ValueError(f"Workspace {workspace_id} already exists")

            workspace = Workspace(
                workspace_id=workspace_id,
                task=task,
                context=context,
            )
            self._workspaces[workspace_id] = workspace

            if self.auto_save:
                self._save_workspace(workspace)

            logger.info(f"Created workspace {workspace_id}")
            return workspace

    def get(self, workspace_id: str) -> Optional[Workspace]:
        """Get a workspace by ID."""
        return self._workspaces.get(workspace_id)

    def get_or_create(
        self,
        workspace_id: str,
        task: str = "",
        context: Optional[Dict[str, Any]] = None,
    ) -> Workspace:
        """Get an existing workspace or create a new one."""
        with self._lock:
            if workspace_id in self._workspaces:
                return self._workspaces[workspace_id]

            # Try to restore from disk
            restored = self._load_workspace(workspace_id)
            if restored:
                self._workspaces[workspace_id] = restored
                return restored

            # Create new
            workspace = Workspace(
                workspace_id=workspace_id,
                task=task,
                context=context,
            )
            self._workspaces[workspace_id] = workspace

            if self.auto_save:
                self._save_workspace(workspace)

            return workspace

    def attach(self, workspace_id: str) -> Optional[Workspace]:
        """Attach to a workspace."""
        workspace = self.get(workspace_id)
        if workspace:
            workspace.attach()
        return workspace

    def detach(self, workspace_id: str) -> None:
        """Detach from a workspace."""
        workspace = self.get(workspace_id)
        if workspace:
            workspace.detach()
            if self.auto_save:
                self._save_workspace(workspace)

    def dispose(self, workspace_id: str) -> None:
        """Dispose of a workspace."""
        with self._lock:
            workspace = self._workspaces.pop(workspace_id, None)
            if workspace:
                workspace.dispose()
                if self.auto_save:
                    self._save_workspace(workspace)

    def dispose_all(self) -> None:
        """Dispose of all workspaces."""
        with self._lock:
            for workspace_id in list(self._workspaces.keys()):
                workspace = self._workspaces.pop(workspace_id, None)
                if workspace:
                    workspace.dispose()
                    if self.auto_save:
                        self._save_workspace(workspace)

    def list_workspaces(self) -> List[Dict[str, Any]]:
        """List all workspaces with basic info."""
        result = []
        for ws in self._workspaces.values():
            result.append({
                "id": ws.id,
                "task": ws.task,
                "status": ws.status,
                "messages": len(ws.history),
                "last_activity": ws.metrics.last_activity,
            })
        return result

    def list_saved_workspaces(self) -> List[str]:
        """List all saved workspace IDs from disk."""
        if not self.storage_dir.exists():
            return []
        return [f.stem for f in self.storage_dir.glob("*.json")]

    def save_all(self) -> None:
        """Save all workspaces to disk."""
        for workspace in self._workspaces.values():
            self._save_workspace(workspace)

    def _save_workspace(self, workspace: Workspace) -> None:
        """Save a workspace to disk."""
        try:
            snapshot = workspace.to_snapshot()
            file_path = self.storage_dir / f"{workspace.id}.json"

            with open(file_path, "w") as f:
                json.dump({
                    "version": snapshot.version,
                    "created_at": snapshot.created_at,
                    "workspace_id": snapshot.workspace_id,
                    "task": snapshot.task,
                    "status": snapshot.status,
                    "history": snapshot.history,
                    "context": snapshot.context,
                    "metrics": snapshot.metrics,
                }, f, indent=2)

            logger.debug(f"Saved workspace {workspace.id}")
        except Exception as e:
            logger.error(f"Failed to save workspace {workspace.id}: {e}")

    def _load_workspace(self, workspace_id: str) -> Optional[Workspace]:
        """Load a workspace from disk."""
        try:
            file_path = self.storage_dir / f"{workspace_id}.json"
            if not file_path.exists():
                return None

            with open(file_path, "r") as f:
                data = json.load(f)

            snapshot = WorkspaceSnapshot(
                version=data.get("version", 1),
                created_at=data.get("created_at", ""),
                workspace_id=data.get("workspace_id", workspace_id),
                task=data.get("task", ""),
                status=data.get("status", "active"),
                history=data.get("history", []),
                context=data.get("context", {}),
                metrics=data.get("metrics", {}),
            )

            workspace = Workspace.from_snapshot(snapshot)
            logger.info(f"Loaded workspace {workspace_id} from disk")
            return workspace

        except Exception as e:
            logger.error(f"Failed to load workspace {workspace_id}: {e}")
            return None


# Global registry instance
_registry: Optional[WorkspaceRegistry] = None


def get_registry() -> WorkspaceRegistry:
    """Get the global workspace registry."""
    global _registry
    if _registry is None:
        _registry = WorkspaceRegistry()
    return _registry


def create_workspace(
    task: str,
    workspace_id: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Workspace:
    """Create a new workspace in the global registry."""
    return get_registry().create(task, workspace_id, context)


def get_workspace(workspace_id: str) -> Optional[Workspace]:
    """Get a workspace from the global registry."""
    return get_registry().get(workspace_id)


def get_or_create_workspace(
    workspace_id: str,
    task: str = "",
    context: Optional[Dict[str, Any]] = None,
) -> Workspace:
    """Get or create a workspace in the global registry."""
    return get_registry().get_or_create(workspace_id, task, context)
