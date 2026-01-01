"""
Constants for AGI-o1.

Centralizes magic numbers, default values, and string constants.
"""
from typing import Final

# Chat history defaults
DEFAULT_MAX_MESSAGES: Final[int] = 50
DEFAULT_DIALOGUE_TURNS: Final[int] = 4

# Pagination defaults
DEFAULT_PAGE_SIZE: Final[int] = 10
DEFAULT_PAGE: Final[int] = 1

# API request limits
MAX_CHAT_ITERATIONS: Final[int] = 3
MAX_OUTPUT_TOKENS_CODE: Final[int] = 4096
MAX_OUTPUT_TOKENS_REASONING: Final[int] = 800
MAX_OUTPUT_TOKENS_INSIGHT: Final[int] = 900
MAX_OUTPUT_TOKENS_EVALUATION: Final[int] = 400
MAX_OUTPUT_TOKENS_AGGREGATOR: Final[int] = 600

# MaTTS (Memory-aware Test-Time Scaling) limits
MIN_MTTS_PASSES: Final[int] = 1
MAX_MTTS_PASSES: Final[int] = 6
DEFAULT_MTTS_PASSES: Final[int] = 3

# Task session limits
DEFAULT_MAX_TASK_TURNS: Final[int] = 10
MAX_SESSION_MESSAGES: Final[int] = 200

# ReasoningBank retrieval defaults
DEFAULT_MEMORY_TOP_K: Final[int] = 3
DEFAULT_MEMORY_TOP_K_MTTS: Final[int] = 3

# Note system markers
NOTES_MARKER: Final[str] = "## Notes Repository"
NOTE_HEADING_PREFIX: Final[str] = "### "

# Slugify settings
MAX_SLUG_LENGTH: Final[int] = 60

# Temperature settings for different operations
TEMP_MTTS_PASS: Final[float] = 0.4
TEMP_MTTS_AGGREGATION: Final[float] = 0.2
TEMP_MTTS_SEQUENTIAL: Final[float] = 0.3
TEMP_EVALUATION: Final[float] = 0.0
TEMP_DISTILLATION: Final[float] = 0.2

# Top-p settings
TOP_P_MTTS: Final[float] = 0.9

# Confidence thresholds
MIN_CONFIDENCE_THRESHOLD: Final[float] = 0.4

# Outcome labels
OUTCOME_SUCCESS: Final[str] = "success"
OUTCOME_FAILURE: Final[str] = "failure"
OUTCOME_MIXED: Final[str] = "mixed"
OUTCOME_UNKNOWN: Final[str] = "unknown"

# Valid reasoning effort levels
REASONING_EFFORTS: Final[tuple] = ("none", "low", "medium", "high", "xhigh")

# Session commands
SESSION_EXIT_COMMANDS: Final[frozenset] = frozenset({":exit", ":cancel"})
SESSION_SUCCESS_COMMANDS: Final[frozenset] = frozenset({":success", ":done"})
SESSION_FAIL_COMMANDS: Final[frozenset] = frozenset({":fail", ":failure"})
SESSION_MIXED_COMMANDS: Final[frozenset] = frozenset({":mixed"})
SESSION_CONFIRM_COMMANDS: Final[frozenset] = frozenset({":confirm"})
SESSION_RESET_COMMANDS: Final[frozenset] = frozenset({":reset", ":undo"})
SESSION_SHOW_COMMANDS: Final[frozenset] = frozenset({":show"})
