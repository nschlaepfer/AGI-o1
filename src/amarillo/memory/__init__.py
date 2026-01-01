"""Memory management modules for Amarillo."""

from .chat_history import ChatHistory, SYSTEM_PROMPT
from .reasoning_bank import ReasoningBank, MemoryItem
from .notes import NotesManager
from .workspace_manager import (
    WorkspaceManager,
    TaskContext,
    TaskStatus,
    ActivityState,
    WorkspaceSnapshot,
    ActivityMonitor,
    get_workspace_manager,
    reset_workspace_manager,
)

__all__ = [
    # Chat History
    "ChatHistory",
    "SYSTEM_PROMPT",
    # Reasoning Bank
    "ReasoningBank",
    "MemoryItem",
    # Notes
    "NotesManager",
    # Workspace Manager
    "WorkspaceManager",
    "TaskContext",
    "TaskStatus",
    "ActivityState",
    "WorkspaceSnapshot",
    "ActivityMonitor",
    "get_workspace_manager",
    "reset_workspace_manager",
]
