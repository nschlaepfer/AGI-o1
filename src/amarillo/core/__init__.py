"""Core configuration, constants, and utilities for Amarillo."""

from .config import config, load_config, AppConfig, ModelConfig, LogConfig, ReasoningBankConfig, REPO_ROOT
from .constants import (
    DEFAULT_MAX_MESSAGES,
    DEFAULT_DIALOGUE_TURNS,
    DEFAULT_PAGE_SIZE,
    DEFAULT_PAGE,
    MAX_CHAT_ITERATIONS,
    MAX_OUTPUT_TOKENS_CODE,
    MAX_OUTPUT_TOKENS_REASONING,
    REASONING_EFFORTS,
    OUTCOME_SUCCESS,
    OUTCOME_FAILURE,
    OUTCOME_MIXED,
    OUTCOME_UNKNOWN,
    SESSION_EXIT_COMMANDS,
    SESSION_SUCCESS_COMMANDS,
    SESSION_FAIL_COMMANDS,
    SESSION_MIXED_COMMANDS,
)
from .utils import (
    slugify,
    format_timestamp,
    safe_json_loads,
    render_message_content,
    format_message_for_display,
    write_log_file,
    PromptLogger,
    format_memory_listing,
)

__all__ = [
    # Config
    "config",
    "load_config",
    "AppConfig",
    "ModelConfig",
    "LogConfig",
    "ReasoningBankConfig",
    "REPO_ROOT",
    # Constants
    "DEFAULT_MAX_MESSAGES",
    "DEFAULT_DIALOGUE_TURNS",
    "DEFAULT_PAGE_SIZE",
    "DEFAULT_PAGE",
    "MAX_CHAT_ITERATIONS",
    "MAX_OUTPUT_TOKENS_CODE",
    "MAX_OUTPUT_TOKENS_REASONING",
    "REASONING_EFFORTS",
    "OUTCOME_SUCCESS",
    "OUTCOME_FAILURE",
    "OUTCOME_MIXED",
    "OUTCOME_UNKNOWN",
    "SESSION_EXIT_COMMANDS",
    "SESSION_SUCCESS_COMMANDS",
    "SESSION_FAIL_COMMANDS",
    "SESSION_MIXED_COMMANDS",
    # Utils
    "slugify",
    "format_timestamp",
    "safe_json_loads",
    "render_message_content",
    "format_message_for_display",
    "write_log_file",
    "PromptLogger",
    "format_memory_listing",
]
