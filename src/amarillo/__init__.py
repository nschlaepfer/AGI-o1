"""
Amarillo - Advanced AI Assistant with GPT-5.2 and Codex Integration.

A modular AI assistant featuring:
- GPT-5.2-Codex for advanced agentic coding (400K context, 128K output)
- Multi-level reasoning with xhigh effort for complex challenges
- ReasoningBank memory system for learning from past interactions
- Memory-aware Test-Time Scaling (MaTTS) for multi-pass reasoning
- Notes system for persistent knowledge management
- Insight Capsule generation for executive summaries
"""

__version__ = "0.1.0"
__author__ = "AGI-o1 Team"

from amarillo.core import config, REPO_ROOT
from amarillo.memory import ChatHistory, ReasoningBank, NotesManager, WorkspaceManager
from amarillo.reasoning import FluidReasoner, EnsembleReasoner
from amarillo.tools import ToolExecutor, get_tools_schema

__all__ = [
    "__version__",
    "__author__",
    "config",
    "REPO_ROOT",
    "ChatHistory",
    "ReasoningBank",
    "NotesManager",
    "WorkspaceManager",
    "FluidReasoner",
    "EnsembleReasoner",
    "ToolExecutor",
    "get_tools_schema",
]
