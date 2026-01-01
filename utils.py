"""
Utility functions for AGI-o1.

Contains helper functions for logging, text processing, and common operations.
"""
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from constants import MAX_SLUG_LENGTH


def slugify(value: str, max_length: int = MAX_SLUG_LENGTH) -> str:
    """
    Convert a string to a URL-safe slug.
    
    Args:
        value: The string to slugify
        max_length: Maximum length of the resulting slug
    
    Returns:
        A lowercase, hyphen-separated slug
    """
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = value.strip("-")
    if not value:
        value = "insight"
    return value[:max_length]


def format_timestamp(fmt: str = "%Y%m%d_%H%M%S") -> str:
    """Return current timestamp in the specified format."""
    return datetime.now().strftime(fmt)


def safe_json_loads(text: str, default: Any = None) -> Any:
    """
    Safely parse JSON with a default fallback.
    
    Args:
        text: JSON string to parse
        default: Value to return if parsing fails
    
    Returns:
        Parsed JSON or default value
    """
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return default


def render_message_content(msg: Dict[str, Any]) -> str:
    """
    Extract and render displayable content from a message dict.
    
    Handles various message formats including tool calls and function calls.
    
    Args:
        msg: Message dictionary with role, content, and optional tool_calls
    
    Returns:
        Human-readable string representation of the message content
    """
    content = msg.get("content", "")
    
    # Handle list content (multimodal messages)
    if isinstance(content, list):
        content = json.dumps(content)
    
    # If content is empty, check for tool/function calls
    if not content:
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            call_names = ", ".join(
                tc.get("function", {}).get("name", "unknown")
                for tc in tool_calls
            )
            content = f"[tool call -> {call_names}]"
        elif msg.get("function_call"):
            fn = msg["function_call"]
            content = f"[function call -> {fn.get('name', 'unknown')}]"
    
    return content


def format_message_for_display(msg: Dict[str, Any], include_system: bool = False) -> Optional[str]:
    """
    Format a message dictionary for display in dialogue transcripts.
    
    Args:
        msg: Message dictionary
        include_system: Whether to include system messages
    
    Returns:
        Formatted string or None if message should be skipped
    """
    role = msg.get("role", "")
    if role == "system" and not include_system:
        return None
    
    name = msg.get("name")
    prefix = f"{role}" + (f"({name})" if name else "")
    content = render_message_content(msg)
    
    return f"{prefix}: {content}"


def write_log_file(
    directory: Path,
    prefix: str,
    content: str,
    extension: str = ".log"
) -> Path:
    """
    Write content to a timestamped log file.
    
    Args:
        directory: Directory to write the file in
        prefix: Filename prefix
        content: Content to write
        extension: File extension (default: .log)
    
    Returns:
        Path to the created file
    """
    filename = f"{prefix}_{format_timestamp()}{extension}"
    filepath = directory / filename
    filepath.write_text(content)
    return filepath


class PromptLogger:
    """Conditional logging for prompts and responses."""
    
    def __init__(self, log_prompts: bool = True, log_responses: bool = False):
        self.log_prompts = log_prompts
        self.log_responses = log_responses
    
    def log_prompt(self, label: str, payload: str) -> None:
        """Log a prompt if prompt logging is enabled."""
        if self.log_prompts:
            logging.debug("[PROMPT][%s] %s", label, payload)
    
    def log_response(self, label: str, payload: str) -> None:
        """Log a response if response logging is enabled."""
        if self.log_responses:
            logging.debug("[RESPONSE][%s] %s", label, payload)


def format_memory_listing(retrieved: List[Dict[str, Any]]) -> str:
    """
    Format retrieved ReasoningBank memories for display.
    
    Args:
        retrieved: List of retrieved memory items with scores
    
    Returns:
        Formatted string listing all memories
    """
    if not retrieved:
        return "No ReasoningBank memories available."
    
    lines = []
    for entry in retrieved:
        memory = entry["item"]
        score = entry.get("score")
        score_text = f" (score={score:.2f})" if score is not None else ""
        lines.append(f"- {memory.title}{score_text} [{memory.outcome}]")
        lines.append(f"  {memory.description}")
    
    return "\n".join(lines)
