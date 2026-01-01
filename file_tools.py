"""
File Tools for AGI-o1 - Text Editing with Context Window Management.

Implements OpenAI GPT-5-Codex recommended tools:
- read_file: Read files in chunks with line offset/limit for context management
- list_dir: List directory contents
- glob_search: Search for files using glob patterns
- apply_patch: Apply diff-based edits (35% lower failure rate per OpenAI research)
- write_file: Write/create files
- search_file: Search within file content

Best practices implemented:
- Chunked reading (400-800 tokens per chunk with 20% overlap)
- Token budget tracking for context window anxiety management
- Structured diffs for reliable edits
- Line number metadata for precise navigation

References:
- https://openai.com/index/introducing-gpt-5-2-codex/
- https://cookbook.openai.com/examples/gpt-5-codex_prompting_guide
"""

import fnmatch
import glob
import logging
import os
import re
import difflib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Default chunk sizes based on OpenAI research
DEFAULT_CHUNK_LINES = 100  # ~400-600 tokens for typical code
MAX_CHUNK_LINES = 500  # Upper bound for larger reads
OVERLAP_LINES = 20  # 20% overlap for context continuity
APPROX_TOKENS_PER_LINE = 5  # Rough estimate for token budgeting


@dataclass
class FileChunk:
    """Represents a chunk of file content with metadata."""
    content: str
    start_line: int
    end_line: int
    total_lines: int
    file_path: str
    has_more: bool
    estimated_tokens: int


@dataclass
class TokenBudget:
    """Track token usage for context window management."""
    used: int = 0
    limit: int = 100000  # Conservative default for GPT-5.2-Codex (400K actual)
    warning_threshold: float = 0.75

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)

    @property
    def usage_percent(self) -> float:
        return (self.used / self.limit) * 100 if self.limit > 0 else 0

    @property
    def should_warn(self) -> bool:
        return self.usage_percent >= (self.warning_threshold * 100)

    def consume(self, tokens: int) -> None:
        self.used += tokens

    def summary(self) -> str:
        status = "WARNING: High usage" if self.should_warn else "OK"
        return f"[Context: {self.used:,}/{self.limit:,} tokens ({self.usage_percent:.1f}%) - {status}]"


# Global token budget tracker
_token_budget = TokenBudget()


def get_token_budget() -> TokenBudget:
    """Get the global token budget tracker."""
    return _token_budget


def reset_token_budget(limit: int = 100000) -> None:
    """Reset the token budget for a new session."""
    global _token_budget
    _token_budget = TokenBudget(limit=limit)


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for text (rough approximation).

    Uses a simple heuristic based on word count and special characters.
    For precise counting, use tiktoken library.
    """
    # Rough estimate: ~0.75 tokens per word, plus extras for punctuation/code
    words = len(text.split())
    specials = len(re.findall(r'[{}\[\]();:,.<>!=+\-*/&|^~]', text))
    return int(words * 0.75 + specials * 0.5 + len(text.split('\n')) * 0.1)


def read_file(
    file_path: str,
    offset: int = 0,
    limit: Optional[int] = None,
    show_line_numbers: bool = True,
    context_lines: int = 0
) -> str:
    """
    Read a file with chunked access for context window management.

    Args:
        file_path: Absolute or relative path to the file
        offset: Line number to start from (0-indexed)
        limit: Maximum number of lines to read (default: 100 for chunks)
        show_line_numbers: Include line numbers in output (default: True)
        context_lines: Extra lines before offset for context (default: 0)

    Returns:
        Formatted file content with metadata

    Example output:
        ```
        [File: /path/to/file.py | Lines 50-150 of 500 | ~320 tokens]
        50: def process_data(self):
        51:     for item in self.items:
        ...
        ```
    """
    try:
        path = Path(file_path).resolve()

        if not path.exists():
            return f"Error: File not found: {file_path}"

        if not path.is_file():
            return f"Error: Path is not a file: {file_path}"

        # Read all lines
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            all_lines = f.readlines()

        total_lines = len(all_lines)

        # Apply defaults and bounds
        if limit is None:
            limit = DEFAULT_CHUNK_LINES

        # Adjust offset for context
        actual_offset = max(0, offset - context_lines)

        # Calculate slice bounds
        start = actual_offset
        end = min(start + limit + context_lines, total_lines)

        # Extract chunk
        chunk_lines = all_lines[start:end]

        # Format output
        output_lines = []

        # Header with metadata
        has_more = end < total_lines
        chunk_text = ''.join(chunk_lines)
        tokens = estimate_tokens(chunk_text)

        header = f"[File: {path} | Lines {start + 1}-{end} of {total_lines}"
        if has_more:
            header += f" | ~{tokens} tokens | MORE CONTENT BELOW"
        else:
            header += f" | ~{tokens} tokens | END OF FILE"
        header += "]"
        output_lines.append(header)
        output_lines.append("")

        # Content with line numbers
        for i, line in enumerate(chunk_lines):
            line_no = start + i + 1  # 1-indexed for display
            line = line.rstrip('\n\r')
            if show_line_numbers:
                output_lines.append(f"{line_no:>6}: {line}")
            else:
                output_lines.append(line)

        # Footer hint for chunked reading
        if has_more:
            output_lines.append("")
            remaining = total_lines - end
            output_lines.append(f"[... {remaining} more lines. Use offset={end} to continue]")

        result = '\n'.join(output_lines)

        # Track token usage
        _token_budget.consume(tokens)

        logging.info(
            "Read file %s: lines %d-%d of %d (~%d tokens)",
            path, start + 1, end, total_lines, tokens
        )

        return result

    except PermissionError:
        return f"Error: Permission denied reading: {file_path}"
    except Exception as e:
        logging.exception("Error reading file %s: %s", file_path, e)
        return f"Error reading file: {str(e)}"


def read_file_range(
    file_path: str,
    start_line: int,
    end_line: int,
    show_line_numbers: bool = True
) -> str:
    """
    Read a specific range of lines from a file.

    Args:
        file_path: Path to the file
        start_line: Starting line (1-indexed, inclusive)
        end_line: Ending line (1-indexed, inclusive)
        show_line_numbers: Include line numbers in output

    Returns:
        Formatted content for the specified range
    """
    # Convert to 0-indexed offset and calculate limit
    offset = max(0, start_line - 1)
    limit = max(1, end_line - start_line + 1)
    return read_file(file_path, offset=offset, limit=limit, show_line_numbers=show_line_numbers)


def list_dir(
    path: str = ".",
    pattern: Optional[str] = None,
    show_hidden: bool = False,
    recursive: bool = False,
    max_entries: int = 100
) -> str:
    """
    List directory contents with optional filtering.

    Args:
        path: Directory path to list
        pattern: Optional glob pattern to filter (e.g., "*.py")
        show_hidden: Include hidden files/directories
        recursive: List recursively
        max_entries: Maximum entries to return

    Returns:
        Formatted directory listing with file types and sizes
    """
    try:
        dir_path = Path(path).resolve()

        if not dir_path.exists():
            return f"Error: Directory not found: {path}"

        if not dir_path.is_dir():
            return f"Error: Not a directory: {path}"

        entries = []

        if recursive:
            iterator = dir_path.rglob(pattern or "*")
        else:
            iterator = dir_path.glob(pattern or "*")

        for entry in iterator:
            if not show_hidden and entry.name.startswith('.'):
                continue

            try:
                stat = entry.stat()
                size = stat.st_size

                if entry.is_dir():
                    entry_type = "[DIR]"
                    size_str = "-"
                elif entry.is_symlink():
                    entry_type = "[LINK]"
                    size_str = "-"
                else:
                    entry_type = "[FILE]"
                    if size < 1024:
                        size_str = f"{size}B"
                    elif size < 1024 * 1024:
                        size_str = f"{size // 1024}K"
                    else:
                        size_str = f"{size // (1024 * 1024)}M"

                rel_path = entry.relative_to(dir_path) if recursive else entry.name
                entries.append((entry_type, str(rel_path), size_str))

            except (PermissionError, OSError):
                continue

            if len(entries) >= max_entries:
                break

        if not entries:
            return f"Directory is empty or no matches: {path}"

        # Sort: directories first, then files, alphabetically
        entries.sort(key=lambda x: (x[0] != "[DIR]", x[1].lower()))

        # Format output
        output = [f"[Directory: {dir_path}]", ""]

        for entry_type, name, size in entries:
            output.append(f"  {entry_type:6} {size:>8}  {name}")

        if len(entries) >= max_entries:
            output.append(f"\n[Truncated at {max_entries} entries]")

        return '\n'.join(output)

    except Exception as e:
        logging.exception("Error listing directory %s: %s", path, e)
        return f"Error listing directory: {str(e)}"


def glob_search(
    pattern: str,
    base_path: str = ".",
    max_results: int = 50
) -> str:
    """
    Search for files using glob patterns.

    Args:
        pattern: Glob pattern (e.g., "**/*.py", "src/**/*.ts")
        base_path: Base directory for search
        max_results: Maximum results to return

    Returns:
        List of matching file paths
    """
    try:
        base = Path(base_path).resolve()

        if not base.exists():
            return f"Error: Base path not found: {base_path}"

        matches = []
        for match in base.glob(pattern):
            matches.append(str(match))
            if len(matches) >= max_results:
                break

        if not matches:
            return f"No files matching pattern: {pattern}"

        output = [f"[Glob search: {pattern} in {base}]", ""]
        output.extend(f"  {m}" for m in sorted(matches))

        if len(matches) >= max_results:
            output.append(f"\n[Truncated at {max_results} results]")

        return '\n'.join(output)

    except Exception as e:
        logging.exception("Error in glob search: %s", e)
        return f"Error in glob search: {str(e)}"


def search_in_file(
    file_path: str,
    pattern: str,
    context_lines: int = 2,
    max_matches: int = 20,
    regex: bool = False
) -> str:
    """
    Search for a pattern within a file and return matching lines with context.

    Args:
        file_path: Path to the file to search
        pattern: Search pattern (string or regex if regex=True)
        context_lines: Lines of context before/after match
        max_matches: Maximum matches to return
        regex: Treat pattern as regex

    Returns:
        Formatted search results with line numbers and context
    """
    try:
        path = Path(file_path).resolve()

        if not path.exists():
            return f"Error: File not found: {file_path}"

        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()

        matches = []

        if regex:
            try:
                compiled = re.compile(pattern, re.IGNORECASE)
            except re.error as e:
                return f"Invalid regex pattern: {e}"

        for i, line in enumerate(lines):
            line_no = i + 1

            if regex:
                found = compiled.search(line)
            else:
                found = pattern.lower() in line.lower()

            if found:
                # Get context
                start = max(0, i - context_lines)
                end = min(len(lines), i + context_lines + 1)

                context = []
                for j in range(start, end):
                    prefix = ">>>" if j == i else "   "
                    context.append(f"{lines[j].rstrip()}")

                matches.append({
                    'line': line_no,
                    'content': line.rstrip(),
                    'context': context,
                    'context_start': start + 1
                })

                if len(matches) >= max_matches:
                    break

        if not matches:
            return f"No matches for '{pattern}' in {path.name}"

        output = [f"[Search: '{pattern}' in {path}]", f"[{len(matches)} matches found]", ""]

        for m in matches:
            output.append(f"Line {m['line']}: {m['content']}")
            if context_lines > 0:
                output.append(f"  Context (lines {m['context_start']}-{m['context_start'] + len(m['context']) - 1}):")
                for ctx_line in m['context']:
                    output.append(f"    {ctx_line}")
                output.append("")

        return '\n'.join(output)

    except Exception as e:
        logging.exception("Error searching file %s: %s", file_path, e)
        return f"Error searching file: {str(e)}"


def apply_patch(
    file_path: str,
    old_content: str,
    new_content: str,
    description: Optional[str] = None
) -> str:
    """
    Apply a diff-based edit to a file using search-and-replace.

    This is the OpenAI-recommended approach for reliable code editing,
    with 35% lower failure rates compared to JSON-based edits.

    Args:
        file_path: Path to the file to edit
        old_content: Exact content to find and replace
        new_content: Content to replace with
        description: Optional description of the change

    Returns:
        Success/failure message with diff preview

    Example:
        apply_patch(
            "/path/to/file.py",
            old_content="def old_function():\n    pass",
            new_content="def new_function():\n    return True"
        )
    """
    try:
        path = Path(file_path).resolve()

        if not path.exists():
            return f"Error: File not found: {file_path}"

        with open(path, 'r', encoding='utf-8') as f:
            original = f.read()

        # Normalize line endings
        old_normalized = old_content.replace('\r\n', '\n')
        original_normalized = original.replace('\r\n', '\n')

        if old_normalized not in original_normalized:
            # Try to find similar content
            lines_old = old_normalized.strip().split('\n')
            lines_orig = original_normalized.split('\n')

            # Find closest match using difflib
            matches = difflib.get_close_matches(
                old_normalized.strip(),
                ['\n'.join(lines_orig[i:i+len(lines_old)])
                 for i in range(len(lines_orig) - len(lines_old) + 1)],
                n=1,
                cutoff=0.6
            )

            if matches:
                suggestion = f"\n\nDid you mean this content?\n```\n{matches[0]}\n```"
            else:
                suggestion = "\n\nThe old_content was not found in the file. Check for exact match including whitespace."

            return f"Error: Content to replace not found in {path.name}{suggestion}"

        # Count occurrences
        count = original_normalized.count(old_normalized)
        if count > 1:
            return (
                f"Error: Found {count} occurrences of the content in {path.name}. "
                "Provide more context to make the match unique, or use replace_all parameter."
            )

        # Apply replacement
        new_file_content = original_normalized.replace(old_normalized, new_content.replace('\r\n', '\n'))

        # Generate diff for preview
        diff = list(difflib.unified_diff(
            original_normalized.split('\n'),
            new_file_content.split('\n'),
            fromfile=f"a/{path.name}",
            tofile=f"b/{path.name}",
            lineterm=''
        ))

        # Write the file
        with open(path, 'w', encoding='utf-8') as f:
            f.write(new_file_content)

        # Format output
        desc_text = f" - {description}" if description else ""
        output = [f"[Patch applied to {path}{desc_text}]", ""]

        if diff:
            output.append("Diff preview:")
            output.append("```diff")
            output.extend(diff[:30])  # Limit diff preview
            if len(diff) > 30:
                output.append(f"... ({len(diff) - 30} more lines)")
            output.append("```")

        logging.info("Applied patch to %s: %s", path, description or "no description")

        return '\n'.join(output)

    except Exception as e:
        logging.exception("Error applying patch to %s: %s", file_path, e)
        return f"Error applying patch: {str(e)}"


def apply_patch_multiple(
    file_path: str,
    replacements: List[Tuple[str, str]],
    description: Optional[str] = None
) -> str:
    """
    Apply multiple search-and-replace operations to a file atomically.

    Args:
        file_path: Path to the file
        replacements: List of (old_content, new_content) tuples
        description: Optional description

    Returns:
        Success/failure message
    """
    try:
        path = Path(file_path).resolve()

        if not path.exists():
            return f"Error: File not found: {file_path}"

        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()

        original = content
        applied = 0
        errors = []

        for i, (old, new) in enumerate(replacements):
            old_norm = old.replace('\r\n', '\n')
            if old_norm in content:
                content = content.replace(old_norm, new.replace('\r\n', '\n'), 1)
                applied += 1
            else:
                errors.append(f"Replacement {i + 1}: Content not found")

        if applied == 0:
            return f"Error: No replacements applied. Issues:\n" + '\n'.join(errors)

        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)

        result = f"[Applied {applied}/{len(replacements)} replacements to {path}]"
        if errors:
            result += f"\nWarnings:\n" + '\n'.join(errors)

        return result

    except Exception as e:
        logging.exception("Error in multi-patch: %s", e)
        return f"Error: {str(e)}"


def write_file(
    file_path: str,
    content: str,
    create_dirs: bool = True
) -> str:
    """
    Write content to a file, creating directories if needed.

    Args:
        file_path: Path for the new/existing file
        content: Content to write
        create_dirs: Create parent directories if they don't exist

    Returns:
        Success/failure message
    """
    try:
        path = Path(file_path).resolve()

        if create_dirs:
            path.parent.mkdir(parents=True, exist_ok=True)

        existed = path.exists()

        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)

        lines = content.count('\n') + 1
        tokens = estimate_tokens(content)

        action = "Updated" if existed else "Created"
        logging.info("%s file %s: %d lines, ~%d tokens", action, path, lines, tokens)

        return f"[{action} file: {path}]\n[{lines} lines, ~{tokens} tokens]"

    except Exception as e:
        logging.exception("Error writing file %s: %s", file_path, e)
        return f"Error writing file: {str(e)}"


def insert_at_line(
    file_path: str,
    line_number: int,
    content: str,
    after: bool = True
) -> str:
    """
    Insert content at a specific line number.

    Args:
        file_path: Path to the file
        line_number: Line number (1-indexed) for insertion
        content: Content to insert
        after: If True, insert after the line; if False, insert before

    Returns:
        Success/failure message
    """
    try:
        path = Path(file_path).resolve()

        if not path.exists():
            return f"Error: File not found: {file_path}"

        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        if line_number < 1 or line_number > len(lines) + 1:
            return f"Error: Line {line_number} out of range (file has {len(lines)} lines)"

        # Convert to 0-indexed
        idx = line_number - 1
        if after:
            idx += 1

        # Ensure content ends with newline
        if not content.endswith('\n'):
            content += '\n'

        # Insert content
        new_content_lines = content.split('\n')
        for i, new_line in enumerate(new_content_lines[:-1]):  # Skip empty last element
            lines.insert(idx + i, new_line + '\n')

        with open(path, 'w', encoding='utf-8') as f:
            f.writelines(lines)

        position = "after" if after else "before"
        logging.info("Inserted %d lines %s line %d in %s",
                    len(new_content_lines) - 1, position, line_number, path)

        return f"[Inserted content {position} line {line_number} in {path.name}]"

    except Exception as e:
        logging.exception("Error inserting at line: %s", e)
        return f"Error: {str(e)}"


def delete_lines(
    file_path: str,
    start_line: int,
    end_line: int
) -> str:
    """
    Delete a range of lines from a file.

    Args:
        file_path: Path to the file
        start_line: First line to delete (1-indexed, inclusive)
        end_line: Last line to delete (1-indexed, inclusive)

    Returns:
        Success/failure message with deleted content preview
    """
    try:
        path = Path(file_path).resolve()

        if not path.exists():
            return f"Error: File not found: {file_path}"

        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        if start_line < 1 or end_line > len(lines) or start_line > end_line:
            return f"Error: Invalid line range {start_line}-{end_line} (file has {len(lines)} lines)"

        # Extract deleted content for preview
        deleted = lines[start_line - 1:end_line]
        deleted_preview = ''.join(deleted[:5])
        if len(deleted) > 5:
            deleted_preview += f"... ({len(deleted) - 5} more lines)"

        # Remove lines
        new_lines = lines[:start_line - 1] + lines[end_line:]

        with open(path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)

        logging.info("Deleted lines %d-%d from %s", start_line, end_line, path)

        return f"[Deleted lines {start_line}-{end_line} from {path.name}]\nDeleted content:\n```\n{deleted_preview}\n```"

    except Exception as e:
        logging.exception("Error deleting lines: %s", e)
        return f"Error: {str(e)}"


# Function schemas for OpenAI tool calling
def get_file_tool_schemas() -> List[Dict[str, Any]]:
    """
    Get the file tool schemas for GPT-5-Codex.

    Returns:
        List of function specification dictionaries
    """
    return [
        {
            "name": "read_file",
            "description": (
                "Read a file with chunked access for context window management. "
                "Use offset/limit to read specific portions. Shows line numbers and token estimates. "
                "ALWAYS use this before editing a file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file to read"
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Line number to start from (0-indexed). Default: 0",
                        "default": 0
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum lines to read. Default: 100. Use smaller for large files.",
                        "default": 100
                    },
                    "show_line_numbers": {
                        "type": "boolean",
                        "description": "Include line numbers in output. Default: true",
                        "default": True
                    }
                },
                "required": ["file_path"]
            }
        },
        {
            "name": "read_file_range",
            "description": (
                "Read a specific range of lines from a file (1-indexed). "
                "Convenient wrapper for reading exact line ranges."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file"
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "First line to read (1-indexed)"
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Last line to read (1-indexed)"
                    }
                },
                "required": ["file_path", "start_line", "end_line"]
            }
        },
        {
            "name": "list_dir",
            "description": (
                "List directory contents with file types and sizes. "
                "Supports glob patterns and recursive listing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path. Default: current directory",
                        "default": "."
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern to filter (e.g., '*.py')"
                    },
                    "show_hidden": {
                        "type": "boolean",
                        "description": "Include hidden files",
                        "default": False
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "List recursively",
                        "default": False
                    }
                },
                "required": []
            }
        },
        {
            "name": "glob_search",
            "description": (
                "Search for files using glob patterns like '**/*.py' or 'src/**/*.ts'. "
                "Useful for finding files across a project."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern (e.g., '**/*.py', 'src/**/*.ts')"
                    },
                    "base_path": {
                        "type": "string",
                        "description": "Base directory for search",
                        "default": "."
                    }
                },
                "required": ["pattern"]
            }
        },
        {
            "name": "search_in_file",
            "description": (
                "Search for a pattern within a file. Returns matching lines with context. "
                "Supports both plain text and regex patterns."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file to search"
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Search pattern"
                    },
                    "context_lines": {
                        "type": "integer",
                        "description": "Lines of context around matches",
                        "default": 2
                    },
                    "regex": {
                        "type": "boolean",
                        "description": "Treat pattern as regex",
                        "default": False
                    }
                },
                "required": ["file_path", "pattern"]
            }
        },
        {
            "name": "apply_patch",
            "description": (
                "Apply a diff-based edit using search-and-replace. "
                "PREFERRED method for editing files (35% lower failure rate). "
                "Provide the exact old content to find and the new content to replace it with. "
                "Include enough context to make the match unique."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file to edit"
                    },
                    "old_content": {
                        "type": "string",
                        "description": "Exact content to find and replace (include sufficient context)"
                    },
                    "new_content": {
                        "type": "string",
                        "description": "New content to replace with"
                    },
                    "description": {
                        "type": "string",
                        "description": "Optional description of the change"
                    }
                },
                "required": ["file_path", "old_content", "new_content"]
            }
        },
        {
            "name": "write_file",
            "description": (
                "Write content to a file. Creates parent directories if needed. "
                "Use for creating new files. For editing existing files, prefer apply_patch."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path for the file"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write"
                    },
                    "create_dirs": {
                        "type": "boolean",
                        "description": "Create parent directories if needed",
                        "default": True
                    }
                },
                "required": ["file_path", "content"]
            }
        },
        {
            "name": "insert_at_line",
            "description": (
                "Insert content at a specific line number. "
                "Can insert before or after the specified line."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file"
                    },
                    "line_number": {
                        "type": "integer",
                        "description": "Line number (1-indexed)"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to insert"
                    },
                    "after": {
                        "type": "boolean",
                        "description": "Insert after the line (default: true)",
                        "default": True
                    }
                },
                "required": ["file_path", "line_number", "content"]
            }
        },
        {
            "name": "delete_lines",
            "description": (
                "Delete a range of lines from a file. "
                "Shows a preview of deleted content."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file"
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "First line to delete (1-indexed)"
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Last line to delete (1-indexed)"
                    }
                },
                "required": ["file_path", "start_line", "end_line"]
            }
        },
        {
            "name": "get_context_budget",
            "description": (
                "Get the current context window token budget status. "
                "Use this to check how much context has been used and plan accordingly."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    ]


def get_file_tools_schema() -> List[Dict[str, Any]]:
    """Convert file tool schemas to the tools format expected by newer models."""
    return [{"type": "function", "function": spec} for spec in get_file_tool_schemas()]


def execute_file_tool(function_name: str, function_args: Dict[str, Any]) -> str:
    """
    Execute a file tool and return the result.

    Args:
        function_name: Name of the tool to execute
        function_args: Arguments for the tool

    Returns:
        String result from tool execution
    """
    try:
        if function_name == "read_file":
            return read_file(
                file_path=function_args["file_path"],
                offset=function_args.get("offset", 0),
                limit=function_args.get("limit"),
                show_line_numbers=function_args.get("show_line_numbers", True)
            )

        if function_name == "read_file_range":
            return read_file_range(
                file_path=function_args["file_path"],
                start_line=function_args["start_line"],
                end_line=function_args["end_line"]
            )

        if function_name == "list_dir":
            return list_dir(
                path=function_args.get("path", "."),
                pattern=function_args.get("pattern"),
                show_hidden=function_args.get("show_hidden", False),
                recursive=function_args.get("recursive", False)
            )

        if function_name == "glob_search":
            return glob_search(
                pattern=function_args["pattern"],
                base_path=function_args.get("base_path", ".")
            )

        if function_name == "search_in_file":
            return search_in_file(
                file_path=function_args["file_path"],
                pattern=function_args["pattern"],
                context_lines=function_args.get("context_lines", 2),
                regex=function_args.get("regex", False)
            )

        if function_name == "apply_patch":
            return apply_patch(
                file_path=function_args["file_path"],
                old_content=function_args["old_content"],
                new_content=function_args["new_content"],
                description=function_args.get("description")
            )

        if function_name == "write_file":
            return write_file(
                file_path=function_args["file_path"],
                content=function_args["content"],
                create_dirs=function_args.get("create_dirs", True)
            )

        if function_name == "insert_at_line":
            return insert_at_line(
                file_path=function_args["file_path"],
                line_number=function_args["line_number"],
                content=function_args["content"],
                after=function_args.get("after", True)
            )

        if function_name == "delete_lines":
            return delete_lines(
                file_path=function_args["file_path"],
                start_line=function_args["start_line"],
                end_line=function_args["end_line"]
            )

        if function_name == "get_context_budget":
            return _token_budget.summary()

        return f"Unknown file tool: {function_name}"

    except KeyError as missing:
        logging.error("Missing argument '%s' for %s", missing, function_name)
        return f"Missing required argument '{missing}' for {function_name}"
    except Exception as e:
        logging.exception("Error executing file tool %s: %s", function_name, e)
        return f"Error: {str(e)}"


# Convenience function to check if a tool name is a file tool
FILE_TOOL_NAMES = {
    "read_file", "read_file_range", "list_dir", "glob_search",
    "search_in_file", "apply_patch", "write_file", "insert_at_line",
    "delete_lines", "get_context_budget"
}


def is_file_tool(name: str) -> bool:
    """Check if a tool name is a file tool."""
    return name in FILE_TOOL_NAMES
