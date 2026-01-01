"""
AGI-o1: Advanced AI Assistant with GPT-5.2 and Codex Integration.

A modular AI assistant featuring:
- GPT-5.2-Codex for advanced agentic coding (400K context, 128K output)
- Multi-level reasoning with xhigh effort for complex challenges
- ReasoningBank memory system for learning from past interactions
- Memory-aware Test-Time Scaling (MaTTS) for multi-pass reasoning
- Notes system for persistent knowledge management
- Insight Capsule generation for executive summaries

Usage:
    python AGI-o1.py

Configuration:
    See .env.example and CODEX_SETUP.md for configuration options.
"""
from openai import OpenAI
import json
import logging
import os
import re
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from dotenv import load_dotenv

from reasoning_bank import ReasoningBank
from file_tools import (
    read_file, read_file_range, list_dir, glob_search, search_in_file,
    apply_patch, write_file, insert_at_line, delete_lines,
    get_file_tool_schemas, execute_file_tool, is_file_tool,
    get_token_budget, reset_token_budget
)
from workspace_manager import (
    WorkspaceManager, TaskContext, TaskStatus, ActivityState,
    get_workspace_manager, reset_workspace_manager
)
from ensemble_reasoning import (
    EnsembleReasoner, ReasoningResult, Solution, ReasoningPhase,
    run_fluid_reasoning, create_ensemble_reasoner
)

# Resolve repository root and load environment variables
REPO_ROOT = Path(__file__).resolve().parent
load_dotenv(REPO_ROOT / ".env")


def _bool_from_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_api_key(env: str, key_map: Dict[str, Optional[str]]) -> Optional[str]:
    if env in key_map and key_map[env]:
        return key_map[env]
    # Fallback to legacy single-key layout if present
    return os.getenv("OPENAI_API_KEY")


OPENAI_ENVIRONMENT = os.getenv("OPENAI_ENVIRONMENT", "sandbox").strip().lower()
API_KEYS: Dict[str, Optional[str]] = {
    "sandbox": os.getenv("OPENAI_API_KEY_SANDBOX"),
    "staging": os.getenv("OPENAI_API_KEY_STAGING"),
    "production": os.getenv("OPENAI_API_KEY_PRODUCTION"),
}
OPENAI_API_KEY = _resolve_api_key(OPENAI_ENVIRONMENT, API_KEYS)
if not OPENAI_API_KEY:
    raise ValueError(
        f"No OpenAI API key configured for environment '{OPENAI_ENVIRONMENT}'. "
        "Populate `.env` or secret manager with the correct OPENAI_API_KEY_* variable."
    )

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# GPT-5.2 and Codex model configuration
# Available models: gpt-5.2, gpt-5.2-codex (most advanced agentic coding)
# GPT-5.1-codex-max (long-horizon agentic tasks), gpt-5.1-codex-mini (cost-effective)
MODEL_ASSISTANT = os.getenv("OPENAI_MODEL_ASSISTANT", os.getenv("OPENAI_MODEL_REASONING", "gpt-5.2"))
MODEL_REASONING = os.getenv("OPENAI_MODEL_REASONING", "gpt-5.2")
# Reasoning effort levels: none, low, medium, high, xhigh (GPT-5.2 supports xhigh for hardest tasks)
MODEL_REASONING_EFFORT = os.getenv("OPENAI_MODEL_REASONING_EFFORT", "high")
MODEL_CODE = os.getenv("OPENAI_MODEL_CODE", "gpt-5.2-codex")
MODEL_CODE_MAX = os.getenv("OPENAI_MODEL_CODE_MAX", "gpt-5.1-codex-max")
MODEL_REALTIME = os.getenv("OPENAI_MODEL_REALTIME", "gpt-realtime")
MODEL_REALTIME_MINI = os.getenv("OPENAI_MODEL_REALTIME_MINI", "gpt-realtime-mini")
MODEL_EMBEDDING = os.getenv("OPENAI_MODEL_EMBEDDING", "text-embedding-3-large")
# Enable auto-compaction for long context (GPT-5.2 supports 400K context window)
MODEL_COMPACTION = _bool_from_env("OPENAI_MODEL_COMPACTION", True)

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_SAMPLE_PROMPTS = _bool_from_env("LOG_SAMPLE_PROMPTS", True)
LOG_SAMPLE_RESPONSES = _bool_from_env("LOG_SAMPLE_RESPONSES", False)

# Create necessary directories if they don't exist
log_dir = str(REPO_ROOT / "logs")
o1_responses_dir = str(REPO_ROOT / "o1_responses")
insight_capsule_dir = str(REPO_ROOT / "insight_capsules")
paper_path = REPO_ROOT / "docs" / "paper.txt"
notes_marker = "## Notes Repository"
note_heading_prefix = "### "
os.makedirs(log_dir, exist_ok=True)
os.makedirs(o1_responses_dir, exist_ok=True)
os.makedirs(insight_capsule_dir, exist_ok=True)

REASONING_BANK_ENABLED = _bool_from_env("REASONING_BANK_ENABLED", True)
REASONING_BANK_PATH = Path(
    os.getenv("REASONING_BANK_PATH", REPO_ROOT / "docs" / "reasoning_bank.json")
)
REASONING_BANK_MAX_ITEMS = int(os.getenv("REASONING_BANK_MAX_ITEMS", "500"))

# Set up logging
log_filename = os.path.join(log_dir, f"chat_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    filename=log_filename,
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.info(
    "OpenAI environment '%s' initialized with assistant=%s, reasoning=%s (effort=%s), "
    "code=%s, code_max=%s, realtime=%s, realtime_mini=%s, compaction=%s",
    OPENAI_ENVIRONMENT,
    MODEL_ASSISTANT,
    MODEL_REASONING,
    MODEL_REASONING_EFFORT,
    MODEL_CODE,
    MODEL_CODE_MAX,
    MODEL_REALTIME,
    MODEL_REALTIME_MINI,
    MODEL_COMPACTION,
)

reasoning_bank: Optional[ReasoningBank] = None
if REASONING_BANK_ENABLED:
    try:
        reasoning_bank = ReasoningBank(
            client=client,
            storage_path=REASONING_BANK_PATH,
            embedding_model=MODEL_EMBEDDING,
            reasoning_model=MODEL_REASONING,
            max_items=REASONING_BANK_MAX_ITEMS,
        )
        logging.info(
            "ReasoningBank enabled with storage at %s",
            REASONING_BANK_PATH,
        )
    except Exception as exc:
        logging.warning("Failed to initialize ReasoningBank: %s", exc)
        reasoning_bank = None


def _log_prompt(label: str, payload: str) -> None:
    if LOG_SAMPLE_PROMPTS:
        logging.debug("[PROMPT][%s] %s", label, payload)


def _log_response(label: str, payload: str) -> None:
    if LOG_SAMPLE_RESPONSES:
        logging.debug("[RESPONSE][%s] %s", label, payload)


def _slugify(value: str, max_length: int = 60) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = value.strip("-")
    if not value:
        value = "insight"
    return value[:max_length]


def _ensure_notes_section() -> None:
    paper_path.parent.mkdir(parents=True, exist_ok=True)
    if not paper_path.exists():
        paper_path.write_text("")
    text = paper_path.read_text()
    if notes_marker not in text:
        with paper_path.open("a") as handle:
            handle.write("\n\n" + notes_marker + "\n\n")


def _load_notes():
    _ensure_notes_section()
    text = paper_path.read_text()
    marker_index = text.find(notes_marker)
    before = text[:marker_index].rstrip("\n")
    after = text[marker_index + len(notes_marker):]
    lines = after.splitlines()
    notes = {}
    order = []
    current_key = None
    current_lines = []
    for line in lines:
        if line.startswith(note_heading_prefix):
            if current_key is not None:
                notes[current_key] = "\n".join(current_lines).strip()
                order.append(current_key)
            current_key = line[len(note_heading_prefix):].strip()
            current_lines = []
        else:
            if current_key is not None:
                current_lines.append(line)
    if current_key is not None:
        notes[current_key] = "\n".join(current_lines).strip()
        order.append(current_key)
    return before, notes, order


def _write_notes(before_section: str, notes_map: dict, order: list) -> None:
    lines = []
    if before_section:
        lines.append(before_section.rstrip("\n"))
    lines.append(notes_marker)
    for key in order:
        if key not in notes_map:
            continue
        lines.append("")
        lines.append(f"{note_heading_prefix}{key}")
        content = notes_map[key]
        if content:
            lines.append(content)
    final_text = "\n".join(lines)
    if not final_text.endswith("\n"):
        final_text += "\n"
    paper_path.write_text(final_text)


def _format_memory_listing(retrieved: List[Dict[str, Any]]) -> str:
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

class ChatHistory:
    def __init__(self, max_messages=50):
        self.messages = deque(maxlen=max_messages)
        self.system_message = {
            "role": "system",
            "content": (
                "You are AGI-o1, an advanced AI coding agent powered by GPT-5.2-Codex with specialized capabilities. "
                "Your primary objective is to help users with complex coding tasks, file editing, and knowledge work.\n\n"

                "## CORE PRINCIPLES\n"
                "1. ALWAYS read files before editing them. Never guess at file contents.\n"
                "2. Use chunked reading for large files to manage context window. Check token budget regularly.\n"
                "3. Before calling a tool, explain WHY you are calling it (tool preambles improve accuracy).\n"
                "4. Prefer 'apply_patch' for editing files (35% lower failure rate than other methods).\n\n"

                "## FILE OPERATIONS (Context-Aware)\n"
                "- 'read_file': Read files in chunks with offset/limit. Use for context management.\n"
                "- 'read_file_range': Read specific line ranges (1-indexed).\n"
                "- 'list_dir': List directory contents with filtering.\n"
                "- 'glob_search': Find files using patterns like '**/*.py'.\n"
                "- 'search_in_file': Search within files with context.\n"
                "- 'apply_patch': PREFERRED edit method - search-and-replace with exact matching.\n"
                "- 'write_file': Create new files only. Use apply_patch for edits.\n"
                "- 'insert_at_line': Insert content at specific line numbers.\n"
                "- 'delete_lines': Remove line ranges from files.\n"
                "- 'get_context_budget': Check token usage to prevent context overflow.\n\n"

                "## CONTEXT WINDOW MANAGEMENT\n"
                "- Default chunk size: 100 lines (~400-600 tokens).\n"
                "- For large files: Read in chunks, use offset to paginate.\n"
                "- Check 'get_context_budget' before reading many files.\n"
                "- If context is getting full, summarize and focus on relevant sections.\n\n"

                "## REASONING & CODE GENERATION\n"
                "- 'deep_reasoning': For complex analysis requiring extended thinking.\n"
                "- 'coder': Generate code with GPT-5.2-Codex (400K context, 128K output).\n"
                "- 'coder_xhigh': Maximum reasoning effort for hardest challenges.\n"
                "- Use reasoning_effort='xhigh' for complex problems.\n\n"

                "## MEMORY MANAGEMENT\n"
                "- 'save_note': Store important information for future reference.\n"
                "- 'edit_note': Update existing notes.\n"
                "- 'view_note': Recall stored information.\n"
                "- 'search_notes': Find relevant data across notes.\n"
                "- 'list_notes': Review available notes.\n\n"

                "## TOOL CALLING BEST PRACTICES (Per OpenAI Research)\n"
                "1. Always explain your reasoning before calling a tool.\n"
                "2. Read files before editing - never assume content.\n"
                "3. Use 'apply_patch' with sufficient context for unique matches.\n"
                "4. For coding tasks, call 'coder' with clear, specific requirements.\n"
                "5. Check context budget periodically during long sessions.\n\n"

                "Make independent decisions. Prioritize code quality and reliability over speed."
            )
        }

    def add_message(self, role, content, name=None):
        message = {"role": role, "content": content}
        if name:
            message["name"] = name
        self.messages.append(message)

    def add_raw_message(self, message: Dict[str, Any]) -> None:
        """
        Append a pre-constructed message dictionary to the history.
        """
        self.messages.append(dict(message))

    def add_tool_message(self, tool_call_id: str, content: Any, name: Optional[str] = None):
        """
        Append a tool (function) response using the modern tool role format.
        """
        if isinstance(content, str):
            rendered_content = content
        else:
            try:
                rendered_content = json.dumps(content)
            except TypeError:
                rendered_content = str(content)
        message = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": rendered_content,
        }
        if name:
            message["name"] = name
        self.messages.append(message)

    def get_messages(self, memory_prompt: Optional[str] = None):
        messages = [dict(self.system_message)]
        if memory_prompt:
            messages.append({"role": "system", "content": memory_prompt})
        messages.extend(list(self.messages))
        return messages

    def recent_dialogue(self, turns: int = 4) -> str:
        relevant = list(self.messages)[-turns * 2 :]
        lines: List[str] = []
        for msg in relevant:
            role = msg.get("role", "")
            if role == "system":
                continue
            name = msg.get("name")
            prefix = f"{role}" + (f"({name})" if name else "")
            content = msg.get("content", "")
            if isinstance(content, list):
                content = json.dumps(content)
            if not content:
                if msg.get("tool_calls"):
                    call_names = ", ".join(
                        tc.get("function", {}).get("name", "unknown")
                        for tc in msg.get("tool_calls", [])
                    )
                    content = f"[tool call -> {call_names}]"
                elif msg.get("function_call"):
                    fn = msg["function_call"]
                    content = f"[function call -> {fn.get('name', 'unknown')}]"
            lines.append(f"{prefix}: {content}")
        return "\n".join(lines)

    def render_dialogue(self) -> str:
        lines: List[str] = []
        for msg in self.messages:
            role = msg.get("role", "")
            if role == "system":
                continue
            name = msg.get("name")
            prefix = f"{role}" + (f"({name})" if name else "")
            content = msg.get("content", "")
            if isinstance(content, list):
                content = json.dumps(content)
            if not content:
                if msg.get("tool_calls"):
                    call_names = ", ".join(
                        tc.get("function", {}).get("name", "unknown")
                        for tc in msg.get("tool_calls", [])
                    )
                    content = f"[tool call -> {call_names}]"
                elif msg.get("function_call"):
                    fn = msg["function_call"]
                    content = f"[function call -> {fn.get('name', 'unknown')}]"
            lines.append(f"{prefix}: {content}")
        return "\n".join(lines)

    def last_assistant_message(self) -> Optional[str]:
        for msg in reversed(self.messages):
            if msg.get("role") == "assistant":
                return msg.get("content")
        return None

    def edit_message(self, index, new_content):
        if 0 <= index < len(self.messages):
            self.messages[index]["content"] = new_content
            logging.info("Edited message at index %d.", index)

    def remove_message(self, index):
        if 0 <= index < len(self.messages):
            removed = self.messages[index]
            del self.messages[index]
            logging.info("Removed message at index %d: %s", index, removed)

chat_history = ChatHistory()

# Define available functions for the assistant
def get_available_functions():
    # Start with file tools (priority for coding tasks per OpenAI recommendations)
    file_tools = get_file_tool_schemas()

    # Core agent functions
    core_functions = [
        {
            "name": "deep_reasoning",
            "description": "Utilize advanced cognitive processing to perform deep reasoning and complex analysis, suitable for intricate decision-making, logical problem-solving, programming challenges, and addressing advanced STEM questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The complex question or problem to analyze"
                    }
                },
                "required": ["query"]
            }
        },
        {
            "name": "retrieve_knowledge",
            "description": "Retrieve factual information and knowledge using OpenAI's Retrieval tool.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query for information retrieval"
                    }
                },
                "required": ["query"]
            }
        },
        {
            "name": "save_note",
            "description": "Save a note into docs/paper.txt under a specified category and filename.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "The category under which to save the note"
                    },
                    "filename": {
                        "type": "string",
                        "description": "The name of the file to save the note"
                    },
                    "content": {
                        "type": "string",
                        "description": "The content of the note"
                    }
                },
                "required": ["category", "filename", "content"]
            }
        },
        {
            "name": "edit_note",
            "description": "Edit the content of an existing note inside docs/paper.txt.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "The category of the note to edit"
                    },
                    "filename": {
                        "type": "string",
                        "description": "The name of the note to edit"
                    },
                    "new_content": {
                        "type": "string",
                        "description": "The new content to replace the existing note"
                    }
                },
                "required": ["category", "filename", "new_content"]
            }
        },
        {
            "name": "list_notes",
            "description": "List all notes tracked within docs/paper.txt, optionally filtered by category and page.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Optional category to filter notes"
                    },
                    "page": {
                        "type": "integer",
                        "description": "Optional page number for pagination"
                    }
                },
                "required": []
            }
        },
        {
            "name": "view_note",
            "description": "Display the content of a specific note stored in docs/paper.txt.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "The category of the note to view"
                    },
                    "filename": {
                        "type": "string",
                        "description": "The name of the note to view"
                    }
                },
                "required": ["category", "filename"]
            }
        },
        {
            "name": "delete_note",
            "description": "Delete a note from docs/paper.txt within a specified category.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "The category of the note to delete"
                    },
                    "filename": {
                        "type": "string",
                        "description": "The name of the note to delete"
                    }
                },
                "required": ["category", "filename"]
            }
        },
        {
            "name": "search_notes",
            "description": "Search for a specific query within all notes saved in docs/paper.txt.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query string"
                    }
                },
                "required": ["query"]
            }
        },
        {
            "name": "fetch_weather",
            "description": "Retrieve the current weather information for a specified location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The location (city and state/country) for weather information"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Optional temperature unit"
                    }
                },
                "required": ["location"]
            }
        },
        {
            "name": "assist_user",
            "description": "Provide general assistance, interact with the user, and offer explanations using the configured assistant model.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The user's request or question"
                    }
                },
                "required": ["query"]
            }
        },
        {
            "name": "coder",
            "description": "Generate and run Python code using GPT-5.2-Codex (most advanced agentic coding model). Supports 400K context, 128K output, native compaction for long-running tasks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The coding task or question"
                    },
                    "use_max_model": {
                        "type": "boolean",
                        "description": "Use GPT-5.1-Codex-Max for long-horizon agentic tasks (default: false)",
                        "default": False
                    },
                    "reasoning_effort": {
                        "type": "string",
                        "enum": ["none", "low", "medium", "high", "xhigh"],
                        "description": "Reasoning effort level. Use 'xhigh' for the hardest coding challenges (default: high)"
                    }
                },
                "required": ["query"]
            }
        },
        {
            "name": "coder_xhigh",
            "description": "Generate code using GPT-5.2-Codex with xhigh reasoning effort for the most complex engineering challenges requiring extended thinking.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The complex coding task requiring maximum reasoning"
                    }
                },
                "required": ["query"]
            }
        },
        {
            "name": "fluid_reasoning",
            "description": (
                "Execute fluid intelligence reasoning with multi-expert ensemble. "
                "Uses parallel experts with different seeds/temperatures, hierarchical prompt cascade, "
                "soft-scoring with partial credit, and diversity-first voting. "
                "Best for novel problems requiring creative problem-solving."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "The problem or task to solve with fluid reasoning"
                    },
                    "num_experts": {
                        "type": "integer",
                        "description": "Number of parallel experts (default: 3)",
                        "default": 3
                    },
                    "hierarchical": {
                        "type": "boolean",
                        "description": "Use full hierarchical reasoning (analysis->hypothesis->implementation). Default: true",
                        "default": True
                    }
                },
                "required": ["task"]
            }
        },
        {
            "name": "workspace_status",
            "description": "Get current workspace status including active tasks, token budget, and session summary.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        },
    ]

    # Combine file tools (priority) with core functions
    # File tools come first as per OpenAI recommendation for coding agents
    return file_tools + core_functions


def get_available_tools():
    """
    Convert function specifications into the tools schema expected by newer models.
    """
    return [{"type": "function", "function": spec} for spec in get_available_functions()]


def _execute_tool(function_name: str, function_args: Dict[str, Any]) -> str:
    """
    Dispatch a function/tool call and return a textual result.
    """
    try:
        # Check if it's a file tool first (priority for coding tasks)
        if is_file_tool(function_name):
            return execute_file_tool(function_name, function_args)

        if function_name == "deep_reasoning":
            return deep_reasoning(function_args["query"])
        if function_name == "retrieve_knowledge":
            return retrieve_knowledge(function_args["query"])
        if function_name == "save_note":
            save_note(function_args["category"], function_args["filename"], function_args["content"])
            return f"Saved note to docs/paper.txt as {function_args['category']}/{function_args['filename']}."
        if function_name == "edit_note":
            return edit_note(function_args["category"], function_args["filename"], function_args["new_content"])
        if function_name == "list_notes":
            return list_notes(function_args.get("category"), function_args.get("page", 1))
        if function_name == "view_note":
            return view_note(function_args["category"], function_args["filename"])
        if function_name == "delete_note":
            return delete_note(function_args["category"], function_args["filename"])
        if function_name == "search_notes":
            return search_notes(function_args["query"])
        if function_name == "fetch_weather":
            return fetch_weather(function_args["location"], function_args.get("unit", "celsius"))
        if function_name == "assist_user":
            return assist_user(function_args["query"])
        if function_name == "coder":
            return coder(
                function_args["query"],
                use_max_model=function_args.get("use_max_model", False),
                reasoning_effort=function_args.get("reasoning_effort"),
            )
        if function_name == "coder_xhigh":
            return coder(function_args["query"], use_max_model=False, reasoning_effort="xhigh")
        if function_name == "fluid_reasoning":
            return fluid_reasoning(
                function_args["task"],
                num_experts=function_args.get("num_experts", 3),
                hierarchical=function_args.get("hierarchical", True)
            )
        if function_name == "workspace_status":
            workspace = get_workspace_manager(REPO_ROOT / ".workspace")
            return workspace.get_session_summary()
        logging.warning("Unknown function call requested: %s", function_name)
        return f"Unknown function: {function_name}"
    except KeyError as missing:
        logging.error("Missing argument '%s' for function '%s'. Args: %s", missing, function_name, function_args)
        return f"Missing argument '{missing}' for function '{function_name}'."
    except Exception as exc:
        logging.exception("Error executing function '%s': %s", function_name, exc)
        return f"Error while executing '{function_name}': {exc}"


def gpt4o_chat(user_input, is_initial_response=False, chat: ChatHistory = None):
    if chat is None:
        chat = chat_history
    logging.info("User input: %s", user_input)
    chat.add_message("user", user_input)

    retrieved_memory = []
    memory_prompt = None
    if reasoning_bank and not is_initial_response:
        retrieved_memory = reasoning_bank.retrieve(user_input, top_k=3)
        memory_prompt = reasoning_bank.format_memory_prompt(retrieved_memory)
        if memory_prompt:
            logging.debug("Injected ReasoningBank memory into prompt.")

    max_iterations = 3  # Limit the number of iterations
    for iteration in range(max_iterations):
        model_label = MODEL_ASSISTANT
        print(f"Calling {model_label} for response... (Iteration {iteration + 1})")
        _log_prompt("assistant_chat", user_input)
        response = client.chat.completions.create(
            model=MODEL_ASSISTANT,
            messages=chat.get_messages(memory_prompt=memory_prompt),
            tools=get_available_tools(),
            tool_choice="auto"
        )

        message = response.choices[0].message

        tool_calls = getattr(message, "tool_calls", None) or []
        if tool_calls:
            structured_message = {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": getattr(tc, "type", "function"),
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            }
            chat.add_raw_message(structured_message)
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                print(f"Function call detected: {function_name}")
                try:
                    function_args = json.loads(tool_call.function.arguments or "{}")
                except json.JSONDecodeError:
                    logging.warning("Failed to decode arguments for %s: %s", function_name, tool_call.function.arguments)
                    function_args = {}
                logging.info("Function call detected: %s with args %s", function_name, function_args)
                tool_result = _execute_tool(function_name, function_args)
                chat.add_tool_message(tool_call.id, tool_result, name=function_name)
            continue

        if message.function_call:
            structured_message = {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": getattr(message.function_call, "id", f"legacy_{uuid4().hex}"),
                        "type": "function",
                        "function": {
                            "name": message.function_call.name,
                            "arguments": message.function_call.arguments,
                        },
                    }
                ],
            }
            chat.add_raw_message(structured_message)
            function_name = message.function_call.name
            print(f"Function call detected: {function_name}")
            try:
                function_args = json.loads(message.function_call.arguments or "{}")
            except json.JSONDecodeError:
                logging.warning("Failed to decode arguments for %s: %s", function_name, message.function_call.arguments)
                function_args = {}
            logging.info("Function call detected: %s with args %s", function_name, function_args)
            tool_call_id = getattr(message.function_call, "id", None) or f"legacy_{uuid4().hex}"
            tool_result = _execute_tool(function_name, function_args)
            chat.add_tool_message(tool_call_id, tool_result, name=function_name)
            continue

        if message.content:
            chat.add_message("assistant", message.content)

            if not is_initial_response and iteration < max_iterations - 1:
                # Check for missed function calls or unsaved information
                check_response = client.chat.completions.create(
                    model=MODEL_ASSISTANT,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant tasked with checking if any function calls were missed or if any information was not saved in the previous response. If you find any missed actions, please call the appropriate function. Pay special attention to saving important information using the save_note function."},
                        {"role": "user", "content": user_input},
                        {"role": "assistant", "content": message.content},
                        {"role": "user", "content": "Check if any function calls were missed or if any information was not saved. If there's any important information that should be saved for future reference, use the save_note function to store it."}
                    ],
                    tools=get_available_tools(),
                    tool_choice="auto"
                )

                check_message = check_response.choices[0].message

                check_tool_calls = getattr(check_message, "tool_calls", None) or []
                if check_tool_calls:
                    print("Missed function call or unsaved information detected. Processing...")
                    structured_message = {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": getattr(tc, "type", "function"),
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in check_tool_calls
                        ],
                    }
                    chat.add_raw_message(structured_message)

                    for tool_call in check_tool_calls:
                        function_name = tool_call.function.name
                        try:
                            function_args = json.loads(tool_call.function.arguments or "{}")
                        except json.JSONDecodeError:
                            logging.warning("Failed to decode arguments for %s during missed call handling: %s", function_name, tool_call.function.arguments)
                            function_args = {}
                        tool_result = _execute_tool(function_name, function_args)
                        chat.add_tool_message(tool_call.id, tool_result, name=function_name)

                    continue  # Continue the loop to process any additional missed function calls

                if check_message.function_call:
                    print("Missed function call or unsaved information detected. Processing...")
                    structured_message = {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": getattr(check_message.function_call, "id", f"legacy_{uuid4().hex}"),
                                "type": "function",
                                "function": {
                                    "name": check_message.function_call.name,
                                    "arguments": check_message.function_call.arguments,
                                },
                            }
                        ],
                    }
                    chat.add_raw_message(structured_message)

                    function_name = check_message.function_call.name
                    try:
                        function_args = json.loads(check_message.function_call.arguments or "{}")
                    except json.JSONDecodeError:
                        logging.warning("Failed to decode arguments for %s during legacy missed call handling: %s", function_name, check_message.function_call.arguments)
                        function_args = {}
                    tool_call_id = getattr(check_message.function_call, "id", None) or f"legacy_{uuid4().hex}"
                    tool_result = _execute_tool(function_name, function_args)
                    chat.add_tool_message(tool_call_id, tool_result, name=function_name)

                    continue  # Continue the loop to process any additional missed function calls

            _log_response("assistant_chat", message.content)
            if reasoning_bank and not is_initial_response:
                try:
                    transcript = chat.recent_dialogue(turns=6)
                    memory_ids = [item["item"].id for item in retrieved_memory if item.get("item")]
                    evaluation = evaluate_interaction(
                        task=user_input,
                        agent_output=message.content,
                        transcript=transcript,
                    )
                    reasoning_bank.process_interaction(
                        task=user_input,
                        agent_output=message.content,
                        transcript=transcript,
                        metadata={
                            "source": "assistant_chat",
                            "memory_ids": memory_ids,
                            "evaluation": evaluation,
                        },
                        outcome=evaluation.get("outcome"),
                    )
                    if evaluation:
                        print("\n[Evaluation]\n" + json.dumps(evaluation, indent=2) + "\n")
                except Exception as exc:
                    logging.warning("ReasoningBank update error: %s", exc)
            return message.content  # If no missed calls or initial response, return the original response

        logging.warning("Assistant response has no content and no function call")
        return "I'm sorry, I didn't understand that."

    # If we've reached the maximum number of iterations, return the last response
    return "I apologize, but I'm having trouble processing your request. Could you please rephrase or provide more information?"

def deep_reasoning(query, reasoning_effort: Optional[str] = None):
    """
    Utilize GPT-5.2 advanced cognitive processing for deep reasoning.

    Supports reasoning effort levels:
    - none: No extended thinking
    - low: Quick reasoning
    - medium: Balanced (recommended for daily use)
    - high: Extended thinking for complex tasks
    - xhigh: Maximum reasoning for the hardest challenges (GPT-5.2 exclusive)

    Args:
        query: The complex question or problem to analyze
        reasoning_effort: Override the default reasoning effort level
    """
    prompt = f"Research query: {query}"
    _log_prompt("deep_reasoning", prompt)
    response_text = None
    effort = reasoning_effort or MODEL_REASONING_EFFORT

    try:
        request_args = {
            "model": MODEL_REASONING,
            "input": prompt,
        }
        if effort:
            request_args["reasoning"] = {"effort": effort}
        # Enable compaction for long context handling with GPT-5.2
        if MODEL_COMPACTION:
            request_args["store"] = True

        response = client.responses.create(**request_args)
        response_text = response.output_text
    except Exception as exc:
        logging.warning(
            "Primary reasoning model call failed (%s). Falling back to assistant model.",
            exc,
        )
        response = client.chat.completions.create(
            model=MODEL_ASSISTANT,
            messages=[{"role": "user", "content": prompt}],
        )
        response_text = response.choices[0].message.content

    logging.info("Reasoning flow completed with %s (effort=%s)", MODEL_REASONING, effort)
    _log_response("deep_reasoning", response_text or "")

    # Log response to a separate file for auditing
    o1_log_filename = os.path.join(
        o1_responses_dir, f"reasoning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    with open(o1_log_filename, "w") as f:
        f.write(f"Query: {query}\nModel: {MODEL_REASONING}\nEffort: {effort}\n\nResponse:\n{response_text}")

    return response_text

def retrieve_knowledge(query):
    """
    Function to manage information using OpenAI's Retrieval tool.
    """
    try:
        # Create an Assistant with the Retrieval tool
        assistant = client.beta.assistants.create(
            name="Librarian",
            instructions="You are a helpful librarian. Use the Retrieval tool to find and provide information based on the user's query.",
            tools=[{"type": "retrieval"}],
            model=MODEL_ASSISTANT
        )

        # Create a Thread for this query
        thread = client.beta.threads.create()

        # Add the user's message to the thread
        client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=query
        )

        # Run the Assistant
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant.id
        )

        # Wait for the run to complete
        while run.status != "completed":
            run = client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id
            )

        # Retrieve the Assistant's response
        messages = client.beta.threads.messages.list(thread_id=thread.id)

        # Get the last assistant message
        for message in reversed(messages):
            if message.role == "assistant":
                return message.content[0].text.value

        return "No response from librarian."

    except Exception as e:
        logging.error("Error in librarian function: %s", e)
        return f"An error occurred while retrieving information: {str(e)}"

def save_note(category, filename, content):
    """
    Save content to the paper notes repository under the specified category/filename.
    """
    key = f"{category}/{filename}"
    before, notes_map, order = _load_notes()
    if key not in order:
        order.append(key)
    notes_map[key] = content.strip()
    _write_notes(before, notes_map, order)
    logging.info("Saved paper note: %s", key)

def edit_note(category, filename, new_content):
    """
    Edit content of an existing note; creates it if missing.
    """
    key = f"{category}/{filename}"
    before, notes_map, order = _load_notes()
    created = key not in notes_map
    if created:
        order.append(key)
    notes_map[key] = new_content.strip()
    _write_notes(before, notes_map, order)
    logging.info("Edited paper note: %s (created=%s)", key, created)
    if created:
        return f"Created and saved new note '{key}'."
    return f"Updated the content of '{key}'."

def list_notes(category=None, page=1, page_size=10):
    """
    List all notes stored in the paper repository, optionally filtered by category.
    """
    _, notes_map, order = _load_notes()
    files = []
    for key in order:
        if "/" in key:
            cat, _ = key.split("/", 1)
        else:
            cat = ""
        if category and cat != category:
            continue
        files.append(key)

    if not files:
        if category:
            return f"No notes found for category '{category}'."
        return "No notes stored in docs/paper.txt."

    total_files = len(files)
    total_pages = (total_files + page_size - 1) // page_size
    if page < 1 or page > total_pages:
        return f"Invalid page number. There are {total_pages} pages available."

    start = (page - 1) * page_size
    end = start + page_size
    paginated_files = files[start:end]
    file_list = "\n".join(paginated_files)
    logging.info("Listed paper notes for page %s.", page)
    return f"Available notes (Page {page}/{total_pages}):\n{file_list}"

def view_note(category, filename):
    """
    View the content of a note from the paper repository.
    """
    key = f"{category}/{filename}"
    _, notes_map, _ = _load_notes()
    if key not in notes_map:
        logging.warning("Requested note %s does not exist.", key)
        return f"Note '{key}' does not exist."
    content = notes_map[key]
    logging.info("Viewed paper note: %s", key)
    return f"Content of '{key}':\n\n{content}"

def delete_note(category, filename):
    """
    Delete a note from the paper repository.
    """
    key = f"{category}/{filename}"
    before, notes_map, order = _load_notes()
    if key not in notes_map:
        logging.warning("Attempted to delete missing note: %s", key)
        return f"Note '{key}' does not exist."
    notes_map.pop(key)
    order = [k for k in order if k != key]
    _write_notes(before, notes_map, order)
    logging.info("Deleted paper note: %s", key)
    return f"Deleted note '{key}'."

def search_notes(query):
    """
    Search for a query string within all notes stored in the paper repository.
    """
    _, notes_map, order = _load_notes()
    matching = []
    q_lower = query.lower()
    for key in order:
        content = notes_map.get(key, "")
        if q_lower in content.lower() or q_lower in key.lower():
            matching.append(key)
    if not matching:
        return f"No notes contain the query '{query}'."
    file_list = "\n".join(matching)
    logging.info("Search notes for query '%s' -> %d matches.", query, len(matching))
    return f"Notes containing '{query}':\n{file_list}"

def fetch_weather(location, unit='celsius'):
    """
    Get the current weather in a given location.
    This is a placeholder implementation. You should integrate with a real weather API.
    """
    try:
        # Placeholder response
        weather_info = f"The current weather in {location} is sunny with a temperature of 25 degrees {unit}."
        logging.info("Retrieved weather information for %s: %s", location, weather_info)
        return weather_info
    except Exception as e:
        logging.error("Error retrieving weather information: %s", e)
        return f"An error occurred while retrieving weather information: {str(e)}"

def assist_user(query):
    _log_prompt("assist_user", query)
    response = client.chat.completions.create(
        model=MODEL_ASSISTANT,
        messages=[
            {"role": "user", "content": query}
        ],
    )
    logging.info("Assistant interaction completed")
    content = response.choices[0].message.content
    _log_response("assist_user", content)
    return content

def read_all_scratch_pad_files():
    """
    Backwards-compatible helper that now reads all notes from docs/paper.txt.
    """
    _, notes_map, order = _load_notes()
    all_content = []
    for key in order:
        if "/" in key:
            category, filename = key.split("/", 1)
        else:
            category, filename = "", key
        content = notes_map.get(key, "")
        all_content.append(f"Category: {category}, File: {filename}\n{content}\n")
    return "\n".join(all_content)


def memory_aware_mtts(task: str, passes: int = 3):
    """
    Execute Memory-aware Test-Time Scaling (MaTTS) by running multiple reasoning passes
    and aggregating the outcomes into a final response.
    """
    passes = max(1, min(passes, 6))
    retrieved_memory = []
    memory_prompt = None
    if reasoning_bank:
        retrieved_memory = reasoning_bank.retrieve(task, top_k=3)
        memory_prompt = reasoning_bank.format_memory_prompt(retrieved_memory)

    memory_context = memory_prompt or "No prior ReasoningBank items retrieved."
    trajectories = []

    for idx in range(passes):
        iteration_prompt = (
            "You are performing a Memory-aware Test-Time Scaling reasoning pass.\n"
            f"Task:\n{task}\n\n"
            f"Retrieved memory hints:\n{memory_context}\n\n"
            f"Pass {idx + 1} of {passes} — provide your reasoning steps and conclude with 'FINAL ANSWER:' on the last line."
        )
        request_args = {
            "model": MODEL_REASONING,
            "input": iteration_prompt,
            "max_output_tokens": 800,
            "temperature": 0.4,
            "top_p": 0.9,
        }
        if MODEL_REASONING_EFFORT:
            request_args["reasoning"] = {"effort": MODEL_REASONING_EFFORT}
        response = client.responses.create(**request_args)
        trajectories.append(
            {
                "index": idx + 1,
                "content": response.output_text.strip(),
            }
        )

    aggregation_prompt = [
        "You are a ReasoningBank aggregator. Combine the reasoning passes below into a single, concise answer.",
        "Highlight consensus, note divergences, and provide the final recommended solution.",
        "",
    ]
    for trajectory in trajectories:
        aggregation_prompt.append(f"Pass {trajectory['index']}:\n{trajectory['content']}\n")
    aggregation_prompt.append("Return the final answer preceded by 'FINAL ANSWER:' on its own line.")

    aggregator_response = client.responses.create(
        model=MODEL_REASONING,
        input="\n".join(aggregation_prompt),
        max_output_tokens=600,
        temperature=0.2,
    )
    final_answer = aggregator_response.output_text.strip()

    evaluation = evaluate_interaction(
        task=task,
        agent_output=final_answer,
        transcript="\n\n".join([f"Pass {traj['index']}:\n{traj['content']}" for traj in trajectories]),
    )

    if reasoning_bank:
        transcript = "\n\n".join(
            [f"Pass {traj['index']}:\n{traj['content']}" for traj in trajectories]
        )
        try:
            reasoning_bank.process_interaction(
                task=task,
                agent_output=final_answer,
                transcript=transcript,
                metadata={
                    "source": "mtts",
                    "passes": passes,
                    "memory_ids": [item["item"].id for item in retrieved_memory if item.get("item")],
                    "evaluation": evaluation,
                },
                outcome=evaluation.get("outcome"),
            )
        except Exception as exc:
            logging.warning("ReasoningBank MTTS update failed: %s", exc)

    return final_answer, trajectories, retrieved_memory, evaluation


def evaluate_interaction(task: str, agent_output: str, transcript: str) -> Dict[str, Any]:
    """
    Evaluate the interaction outcome (success/failure) and capture rationale for ReasoningBank.
    """
    prompt = {
        "role": "system",
        "content": (
            "You are an impartial evaluator. Review the agent's answer relative to the task and classify the outcome."
        ),
    }
    user_payload = {
        "role": "user",
        "content": (
            "Task:\n"
            f"{task}\n\n"
            "Agent Answer:\n"
            f"{agent_output}\n\n"
            "Recent Transcript:\n"
            f"{transcript}\n\n"
            "Respond in JSON with fields outcome (success, failure, mixed, unknown), confidence (0-1), "
            "rationale (string), and improvements (array of strings)."
        ),
    }
    try:
        response = client.responses.create(
            model=MODEL_REASONING,
            input=[prompt, user_payload],
            max_output_tokens=400,
            temperature=0.0,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "Evaluation",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "outcome": {
                                "type": "string",
                                "enum": ["success", "failure", "mixed", "unknown"],
                            },
                            "confidence": {"type": "number"},
                            "rationale": {"type": "string"},
                            "improvements": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["outcome", "confidence", "rationale", "improvements"],
                        "additionalProperties": False,
                    },
                },
            },
        )
        return json.loads(response.output_text)
    except Exception as exc:
        logging.warning("Evaluation failed: %s", exc)
        return {
            "outcome": "unknown",
            "confidence": 0.0,
            "rationale": "Evaluation failed.",
            "improvements": [],
        }


def memory_aware_mtts_sequential(task: str, passes: int = 3):
    """
    Sequential MaTTS: iteratively refine a single trajectory with self-reflection notes.
    """
    passes = max(1, min(passes, 6))
    retrieved_memory = []
    memory_prompt = None
    if reasoning_bank:
        retrieved_memory = reasoning_bank.retrieve(task, top_k=3)
        memory_prompt = reasoning_bank.format_memory_prompt(retrieved_memory)

    memory_context = memory_prompt or "No prior ReasoningBank items retrieved."
    refinements = []
    previous_answer = ""

    for idx in range(passes):
        iteration_prompt = (
            "You are performing sequential self-refinement with ReasoningBank guidance.\n"
            f"Task:\n{task}\n\n"
            f"Retrieved memory hints:\n{memory_context}\n\n"
            "Previous attempt:\n"
            f"{previous_answer or '[none]'}\n\n"
            f"Pass {idx + 1} of {passes}. Analyze weaknesses, outline corrections, and finish with 'FINAL ANSWER:' line."
        )
        request_args = {
            "model": MODEL_REASONING,
            "input": iteration_prompt,
            "max_output_tokens": 700,
            "temperature": 0.3,
            "top_p": 0.9,
        }
        if MODEL_REASONING_EFFORT:
            request_args["reasoning"] = {"effort": MODEL_REASONING_EFFORT}
        response = client.responses.create(**request_args)
        content = response.output_text.strip()
        refinements.append({"index": idx + 1, "content": content})
        previous_answer = content

    final_answer = previous_answer

    evaluation = evaluate_interaction(
        task=task,
        agent_output=final_answer,
        transcript="\n\n".join([f"Pass {r['index']}:\n{r['content']}" for r in refinements]),
    )

    if reasoning_bank:
        transcript = "\n\n".join([f"Pass {r['index']}:\n{r['content']}" for r in refinements])
        try:
            reasoning_bank.process_interaction(
                task=task,
                agent_output=final_answer,
                transcript=transcript,
                metadata={
                    "source": "mtts_sequential",
                    "passes": passes,
                    "memory_ids": [item["item"].id for item in retrieved_memory if item.get("item")],
                    "evaluation": evaluation,
                },
                outcome=evaluation.get("outcome"),
            )
        except Exception as exc:
            logging.warning("ReasoningBank sequential update failed: %s", exc)

    return final_answer, refinements, retrieved_memory, evaluation


def run_task_session(task: str, max_turns: int = 10):
    """
    Interactive sub-loop for tackling a specific task with its own chat history.
    """
    session_history = ChatHistory(max_messages=200)
    print(f"--- Starting task session: {task} ---")
    print("Enter messages to interact with the agent.")
    print("Commands: :success, :fail, :mixed, :confirm, :reset, :exit, :show (view transcript)")
    outcome = "unknown"
    turn = 0
    pending_outcome: Optional[str] = None

    while turn < max_turns:
        user_msg = input(f"{task}> ").strip()
        if not user_msg:
            continue
        low = user_msg.lower()
        if low in {":exit", ":cancel"}:
            print("Session cancelled.")
            outcome = "unknown"
            break
        if low in {":success", ":done"}:
            pending_outcome = "success"
            print("Pending outcome set to SUCCESS. Type :confirm to finalize or continue the conversation.")
            continue
        if low in {":fail", ":failure"}:
            pending_outcome = "failure"
            print("Pending outcome set to FAILURE. Type :confirm to finalize or continue the conversation.")
            continue
        if low in {":mixed"}:
            pending_outcome = "mixed"
            print("Pending outcome set to MIXED. Type :confirm to finalize or continue the conversation.")
            continue
        if low == ":confirm":
            if pending_outcome:
                outcome = pending_outcome
                print(f"Session marked as {outcome.upper()}.")
                break
            else:
                print("No pending outcome. Use :success / :fail / :mixed before confirming.")
                continue
        if low in {":reset", ":undo"}:
            pending_outcome = None
            print("Pending outcome cleared.")
            continue
        if low == ":show":
            print("\n--- Session Transcript ---")
            print(session_history.render_dialogue() or "[empty]")
            print("--------------------------\n")
            continue

        response = gpt4o_chat(user_msg, chat=session_history)
        print("\nAI:", response, "\n")
        turn += 1

    dialogue = session_history.render_dialogue()
    final_answer = session_history.last_assistant_message() or ""
    evaluation = None
    if dialogue.strip():
        evaluation = evaluate_interaction(task, final_answer, dialogue)
        if outcome != "unknown":
            evaluation["outcome"] = outcome

        if reasoning_bank:
            try:
                reasoning_bank.process_interaction(
                    task=task,
                    agent_output=final_answer,
                    transcript=dialogue,
                    metadata={
                        "source": "task_session",
                        "turns": turn,
                        "evaluation": evaluation,
                    },
                    outcome=evaluation.get("outcome") if evaluation else outcome,
                )
            except Exception as exc:
                logging.warning("ReasoningBank session update failed: %s", exc)

    if evaluation:
        print("\n[Session evaluation]")
        print(json.dumps(evaluation, indent=2))
    print(f"--- Task session '{task}' ended ---\n")

def coder(query, use_max_model: bool = False, reasoning_effort: Optional[str] = None):
    """
    Function to generate and run Python code using the configured coding model.

    GPT-5.2-Codex is the most advanced agentic coding model with:
    - 400K context window, 128K max output tokens
    - Native compaction for long-running tasks
    - Strong performance on refactors, migrations, and security tasks

    Args:
        query: The coding task or question
        use_max_model: If True, uses gpt-5.1-codex-max for long-horizon agentic tasks
        reasoning_effort: Override reasoning effort (none, low, medium, high, xhigh)
    """
    prompt = (
        "Generate and run Python code for the following task. "
        "Only output the running Python code and its output. No explanations or additional text.\n"
        f"Task: {query}"
    )
    _log_prompt("coder", prompt)

    model_to_use = MODEL_CODE_MAX if use_max_model else MODEL_CODE
    effort = reasoning_effort or MODEL_REASONING_EFFORT

    try:
        request_args = {
            "model": model_to_use,
            "input": prompt,
            "max_output_tokens": 4096,  # GPT-5.2-Codex supports up to 128K output
        }
        # Add reasoning effort for Codex models
        if effort:
            request_args["reasoning"] = {"effort": effort}
        # Enable compaction for long context handling
        if MODEL_COMPACTION:
            request_args["store"] = True  # Enable context store for compaction

        response = client.responses.create(**request_args)
        model_response = response.output_text
    except Exception as primary_exc:
        logging.warning(
            "Primary coding model %s failed (%s). Falling back to assistant model.",
            model_to_use,
            primary_exc,
        )
        try:
            response = client.chat.completions.create(
                model=MODEL_ASSISTANT,
                messages=[
                    {
                        "role": "user",
                        "content": prompt + "\nRemember: only output executable code and execution result.",
                    }
                ],
            )
            model_response = response.choices[0].message.content
        except Exception as fallback_exc:
            logging.error(
                "Fallback assistant model failed for coder task (%s).",
                fallback_exc,
            )
            return f"An error occurred while generating or running code: {fallback_exc}"

    logging.info("Code generation completed with %s model (effort=%s)", model_to_use, effort)
    o1_log_filename = os.path.join(
        o1_responses_dir, f"code_response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    with open(o1_log_filename, "w") as f:
        f.write(f"Query: {query}\nModel: {model_to_use}\nEffort: {effort}\n\nResponse:\n{model_response}")

    _log_response("coder", model_response or "")
    return model_response


def _llm_call_adapter(prompt: str, config: Dict[str, Any]) -> str:
    """
    Adapter to make OpenAI client compatible with ensemble_reasoning.

    Used by EnsembleReasoner for parallel expert execution.
    """
    temperature = config.get("temperature", 0.7)
    max_tokens = config.get("max_tokens", 4096)
    reasoning_effort = config.get("reasoning_effort", MODEL_REASONING_EFFORT)

    try:
        # Try responses API first (for reasoning models)
        request_args = {
            "model": MODEL_REASONING,
            "input": prompt,
            "max_output_tokens": max_tokens,
            "temperature": temperature,
        }
        if reasoning_effort:
            request_args["reasoning"] = {"effort": reasoning_effort}

        response = client.responses.create(**request_args)
        return response.output_text

    except Exception as e:
        logging.warning("Responses API failed, falling back to chat: %s", e)
        # Fallback to chat completions
        response = client.chat.completions.create(
            model=MODEL_ASSISTANT,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""


def fluid_reasoning(
    task: str,
    num_experts: int = 3,
    max_iterations: int = 5,
    hierarchical: bool = True
) -> str:
    """
    Execute fluid intelligence reasoning with multi-expert ensemble.

    Implements patterns from poetiq-arc-agi-solver:
    - Parallel expert execution with different seeds/temperatures
    - Hierarchical prompt cascade (analysis → hypothesis → implementation)
    - Soft-scoring with partial credit
    - Diversity-first voting for consensus

    Args:
        task: The problem or task to solve
        num_experts: Number of parallel experts (default: 3)
        max_iterations: Max refinement iterations per expert (default: 5)
        hierarchical: Use full hierarchical reasoning (default: True)

    Returns:
        The best solution found by the ensemble
    """
    logging.info("Starting fluid reasoning with %d experts (hierarchical=%s)",
                num_experts, hierarchical)

    # Get workspace manager for state tracking
    workspace = get_workspace_manager(REPO_ROOT / ".workspace")

    # Create task in workspace
    task_context = workspace.create_task(
        title=f"Fluid Reasoning: {task[:50]}...",
        description=task,
        metadata={"type": "fluid_reasoning", "num_experts": num_experts}
    )

    workspace.set_task_activity(task_context.id, ActivityState.BUSY)

    try:
        # Run ensemble reasoning
        result = run_fluid_reasoning(
            problem=task,
            llm_call=_llm_call_adapter,
            hierarchical=hierarchical,
            num_experts=num_experts
        )

        # Store solutions in workspace
        for sol in result.solutions[:5]:  # Store top 5
            workspace.add_solution(
                task_context.id,
                {"content": sol.content, "phase": sol.phase.value},
                sol.score
            )

        # Update task status
        if result.best_solution and result.best_solution.score >= 0.7:
            workspace.update_task(task_context.id, status=TaskStatus.COMPLETED)
        else:
            workspace.update_task(task_context.id, status=TaskStatus.IN_PROGRESS)

        # Format output
        output_lines = [
            f"## Fluid Reasoning Results",
            f"",
            f"**Task:** {task[:100]}...",
            f"**Experts:** {num_experts}",
            f"**Mode:** {'Hierarchical' if hierarchical else 'Parallel'}",
            f"**Time:** {result.total_time:.2f}s",
            f"**Solutions Found:** {len(result.solutions)}",
            f"**Unique Solutions:** {result.metadata.get('unique_solutions', 0)}",
            f"",
        ]

        if result.best_solution:
            output_lines.append(f"### Best Solution (Score: {result.best_solution.score:.2f})")
            output_lines.append("")
            output_lines.append(result.best_solution.content)

        if result.consensus_solution and result.consensus_solution != result.best_solution:
            output_lines.append("")
            output_lines.append(f"### Consensus Solution (Score: {result.consensus_solution.score:.2f})")
            output_lines.append("")
            output_lines.append(result.consensus_solution.content[:500] + "...")

        # Log to file
        log_filename = os.path.join(
            o1_responses_dir,
            f"fluid_reasoning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        with open(log_filename, "w") as f:
            f.write(f"Task: {task}\n")
            f.write(f"Experts: {num_experts}\n")
            f.write(f"Mode: {'Hierarchical' if hierarchical else 'Parallel'}\n")
            f.write(f"Time: {result.total_time:.2f}s\n\n")
            f.write("--- Solutions ---\n")
            for i, sol in enumerate(result.solutions):
                f.write(f"\n[{i+1}] Score: {sol.score:.2f}\n{sol.content}\n")

        logging.info("Fluid reasoning completed: %d solutions, best score %.2f",
                    len(result.solutions),
                    result.best_solution.score if result.best_solution else 0)

        return "\n".join(output_lines)

    except Exception as e:
        logging.exception("Fluid reasoning failed: %s", e)
        workspace.update_task(task_context.id, status=TaskStatus.FAILED)
        return f"Fluid reasoning failed: {str(e)}"

    finally:
        workspace.set_task_activity(task_context.id, ActivityState.IDLE)


def generate_insight_capsule(topic: str, context: str, metadata: Optional[Dict[str, str]] = None):
    """
    Create a structured insight capsule using the reasoning model and persist it to disk.
    """
    if not topic:
        raise ValueError("Topic is required to generate an insight capsule.")

    context_for_prompt = context.strip() or "No additional context provided."
    capsule_schema = {
        "type": "object",
        "properties": {
            "topic": {"type": "string"},
            "summary": {"type": "string"},
            "primary_drivers": {"type": "array", "items": {"type": "string"}},
            "blockers": {"type": "array", "items": {"type": "string"}},
            "recommended_actions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string"},
                        "owner_hint": {"type": "string"},
                        "expected_impact": {"type": "string"},
                    },
                    "required": ["description"],
                    "additionalProperties": False,
                },
            },
            "metrics": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "current": {"type": "string"},
                        "target": {"type": "string"},
                    },
                    "required": ["name", "current"],
                    "additionalProperties": False,
                },
            },
            "confidence": {"type": "string"},
            "review_horizon": {"type": "string"},
        },
        "required": ["topic", "summary", "primary_drivers", "recommended_actions", "confidence"],
        "additionalProperties": False,
    }

    schema_description = json.dumps(capsule_schema, indent=2)
    prompt = (
        "You are producing an executive Insight Capsule summarizing a complex situation.\n"
        "Respond with JSON only, matching the provided schema precisely. Do not include prose outside JSON.\n"
        f"Schema:\n{schema_description}\n"
        f"\nTopic: {topic}\n\nContext:\n{context_for_prompt}\n"
    )

    _log_prompt("insight_capsule", prompt)

    request_args = {
        "model": MODEL_REASONING,
        "input": prompt,
        "max_output_tokens": 900,
    }
    if MODEL_REASONING_EFFORT:
        request_args["reasoning"] = {"effort": MODEL_REASONING_EFFORT}

    try:
        response = client.responses.create(**request_args)
        response_dict = response.model_dump()
    except Exception as exc:
        logging.error("Failed to generate insight capsule: %s", exc)
        raise

    capsule_data = None
    for item in response_dict.get("output", []):
        for block in item.get("content", []):
            if block.get("type") in {"json", "json_schema"} and "json" in block:
                capsule_data = block["json"]
                break
        if capsule_data:
            break

    if capsule_data is None:
        try:
            capsule_data = json.loads(response.output_text)
        except Exception as parse_exc:
            logging.error("Unable to parse insight capsule JSON: %s", parse_exc)
            raise ValueError("Insight capsule response did not contain valid JSON.")

    capsule_data.setdefault("topic", topic)
    capsule_data["generated_at"] = datetime.utcnow().isoformat() + "Z"
    capsule_data["context_excerpt"] = context_for_prompt[:1200]
    if metadata:
        capsule_data["metadata"] = metadata

    filename = f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{_slugify(topic)}.json"
    file_path = os.path.join(insight_capsule_dir, filename)

    with open(file_path, "w") as f:
        json.dump(capsule_data, f, indent=2)

    logging.info("Insight capsule generated for topic '%s' -> %s", topic, file_path)
    _log_response("insight_capsule", json.dumps(capsule_data))

    return {"file_path": file_path, "capsule": capsule_data}

# Main interaction loop
def main():
    # Reset token budget for new session
    reset_token_budget(limit=100000)  # Conservative limit for context tracking

    print("=" * 80)
    print("  AGI-o1 System | Powered by GPT-5.2-Codex")
    print("  Advanced AI Coding Agent with Context Window Management")
    print("=" * 80)
    print("\nType 'exit', 'quit', or 'bye' to end the conversation.\n")

    print("FILE OPERATIONS (New!):")
    print("  /read <path> [offset] [limit]              - Read file with chunked access")
    print("  /ls [path] [pattern]                       - List directory contents")
    print("  /find <glob_pattern>                       - Search for files (e.g., **/*.py)")
    print("  /search <path> <pattern>                   - Search within a file")
    print("  /context                                   - Show token budget status")
    print("")
    print("NOTES & MEMORY:")
    print("  /save_note <category> <filename> <content> - Save a note to docs/paper.txt")
    print("  /edit_note <category> <filename> <content> - Edit an existing note")
    print("  /list_notes [category] [page]              - List notes, optionally paginated")
    print("  /view_note <category> <filename>           - View the content of a note")
    print("  /delete_note <category> <filename>         - Delete a note")
    print("  /search_notes <query>                      - Search within all notes")
    print("")
    print("FLUID INTELLIGENCE (New!):")
    print("  /fluid <task>                              - Multi-expert ensemble reasoning")
    print("  /fluid_fast <task>                         - Fast parallel reasoning (no hierarchy)")
    print("  /workspace                                 - Show workspace/session status")
    print("  /tasks                                     - List all tasks in workspace")
    print("  /checkpoint [name]                         - Save workspace checkpoint")
    print("")
    print("REASONING & SESSIONS:")
    print("  /rb_stats                                  - Show ReasoningBank metadata")
    print("  /rb_query <query>                          - Retrieve relevant ReasoningBank items")
    print("  /rb_recent                                 - Show recent ReasoningBank items")
    print("  /mtts <passes> :: <task>                   - Run MaTTS multi-pass reasoning")
    print("  /mtts_seq <passes> :: <task>               - Run sequential MaTTS self-refinement")
    print("  /session <task>                            - Start an interactive task sub-loop")
    print("  /insight <topic> :: <context>              - Generate an Insight Capsule")
    print("")
    print("CHAT MANAGEMENT:")
    print("  /edit <index> <new_content>                - Edit a message at a specific index")
    print("  /remove <index>                            - Remove a message at a specific index")
    print("-" * 80)

    # Read all stored notes
    scratch_pad_content = read_all_scratch_pad_files()

    # Start the conversation by summarizing stored notes
    initial_query = f"""Here is the content of all stored notes:

{scratch_pad_content}

Please summarize this information and suggest a topic or question we could discuss based on it. If there are no files or the content is empty, please mention that and suggest a general topic to discuss and ask for their name. This is the first interaction. Greet the user this way. Make this a two sentence response. You are like alfred to batman. You are the intelligent agent that helps the user with their requests and questions. You are also a personal assistant to the user."""

    response = gpt4o_chat(initial_query, is_initial_response=True)
    print("\nAI:", response)
    print("\n" + "-" * 80 + "\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ['exit', 'quit', 'bye']:
            logging.info("User ended the conversation")
            print("Goodbye!")
            break

        # FILE OPERATION COMMANDS (New!)
        elif user_input.startswith("/read "):
            # Parse read command: /read <path> [offset] [limit]
            parts = user_input.split()
            if len(parts) >= 2:
                file_path = parts[1]
                offset = int(parts[2]) if len(parts) > 2 else 0
                limit = int(parts[3]) if len(parts) > 3 else None
                result = read_file(file_path, offset=offset, limit=limit)
                print(result)
            else:
                print("Usage: /read <path> [offset] [limit]")
            continue

        elif user_input.startswith("/ls"):
            # Parse ls command: /ls [path] [pattern]
            parts = user_input.split()
            path = parts[1] if len(parts) > 1 else "."
            pattern = parts[2] if len(parts) > 2 else None
            result = list_dir(path, pattern=pattern)
            print(result)
            continue

        elif user_input.startswith("/find "):
            # Parse find command: /find <glob_pattern>
            pattern = user_input[6:].strip()
            if pattern:
                result = glob_search(pattern)
                print(result)
            else:
                print("Usage: /find <glob_pattern> (e.g., **/*.py)")
            continue

        elif user_input.startswith("/search "):
            # Parse search command: /search <path> <pattern>
            parts = user_input.split(maxsplit=2)
            if len(parts) == 3:
                file_path = parts[1]
                pattern = parts[2]
                result = search_in_file(file_path, pattern)
                print(result)
            else:
                print("Usage: /search <path> <pattern>")
            continue

        elif user_input == "/context":
            # Show token budget status
            budget = get_token_budget()
            print(budget.summary())
            continue

        # FLUID INTELLIGENCE COMMANDS (New!)
        elif user_input.startswith("/fluid "):
            # Fluid reasoning with multi-expert ensemble
            task = user_input[7:].strip()
            if task:
                print("Starting fluid reasoning with 3 experts (hierarchical mode)...")
                result = fluid_reasoning(task, num_experts=3, hierarchical=True)
                print(result)
            else:
                print("Usage: /fluid <task>")
            continue

        elif user_input.startswith("/fluid_fast "):
            # Fast parallel reasoning without hierarchy
            task = user_input[12:].strip()
            if task:
                print("Starting fast fluid reasoning with 3 experts (parallel mode)...")
                result = fluid_reasoning(task, num_experts=3, hierarchical=False)
                print(result)
            else:
                print("Usage: /fluid_fast <task>")
            continue

        elif user_input == "/workspace":
            # Show workspace status
            workspace = get_workspace_manager(REPO_ROOT / ".workspace")
            print(workspace.get_session_summary())
            continue

        elif user_input == "/tasks":
            # List all tasks in workspace
            workspace = get_workspace_manager(REPO_ROOT / ".workspace")
            tasks = workspace.list_tasks()
            if not tasks:
                print("No tasks in workspace.")
            else:
                print(f"Tasks in workspace ({len(tasks)}):")
                for task in tasks:
                    status_icon = {
                        TaskStatus.PENDING: "⏳",
                        TaskStatus.IN_PROGRESS: "🔄",
                        TaskStatus.COMPLETED: "✅",
                        TaskStatus.FAILED: "❌",
                        TaskStatus.PAUSED: "⏸️",
                        TaskStatus.WAITING: "⏸️",
                        TaskStatus.CANCELLED: "🚫"
                    }.get(task.status, "❓")
                    active = " [ACTIVE]" if task.id == workspace._active_task_id else ""
                    print(f"  {status_icon} {task.title} (score: {task.score:.2f}){active}")
            continue

        elif user_input.startswith("/checkpoint"):
            # Save workspace checkpoint
            parts = user_input.split(maxsplit=1)
            name = parts[1] if len(parts) > 1 else None
            workspace = get_workspace_manager(REPO_ROOT / ".workspace")
            checkpoint_path = workspace.save_checkpoint(name)
            print(f"Checkpoint saved: {checkpoint_path}")
            continue

        elif user_input.startswith("/edit "):
            # Parse edit command: /edit <index> <new_content>
            parts = user_input.split(maxsplit=2)
            if len(parts) == 3:
                try:
                    index = int(parts[1])
                    new_content = parts[2]
                    chat_history.edit_message(index, new_content)
                    print(f"Message at index {index} edited.")
                except ValueError:
                    print("Invalid edit command. Use: /edit <index> <new_content>")
            else:
                print("Invalid edit command format. Use: /edit <index> <new_content>")
            continue
        elif user_input.startswith("/remove "):
            # Parse remove command: /remove <index>
            parts = user_input.split(maxsplit=1)
            if len(parts) == 2:
                try:
                    index = int(parts[1])
                    chat_history.remove_message(index)
                    print(f"Message at index {index} removed.")
                except ValueError:
                    print("Invalid remove command. Use: /remove <index>")
            else:
                print("Invalid remove command format. Use: /remove <index>")
            continue
        elif user_input.startswith("/save_note "):
            # Parse save command: /save_note <category> <filename> <content>
            parts = user_input.split(maxsplit=3)
            if len(parts) == 4:
                category = parts[1]
                filename = parts[2]
                content = parts[3]
                save_note(category, filename, content)
                print(f"Content saved to docs/paper.txt as '{category}/{filename}'.")
            else:
                print("Invalid save_note command format. Use: /save_note <category> <filename> <content>")
            continue
        elif user_input.startswith("/edit_note "):
            # Parse edit command: /edit_note <category> <filename> <new_content>
            parts = user_input.split(maxsplit=4)
            if len(parts) == 4:
                category = parts[1]
                filename = parts[2]
                new_content = parts[3]
                result = edit_note(category, filename, new_content)
                print(result)
            else:
                print("Invalid edit_note command format. Use: /edit_note <category> <filename> <new_content>")
            continue
        elif user_input.startswith("/insight "):
            payload = user_input[len("/insight "):].strip()
            if not payload:
                print("Usage: /insight <topic> :: <context (optional)>")
                continue

            topic_part = payload
            context_part = ""
            if "::" in payload:
                topic_part, context_part = [segment.strip() for segment in payload.split("::", 1)]

            if not topic_part:
                print("Insight command requires a topic before '::'.")
                continue

            if not context_part:
                # Default to recent user and assistant exchanges as context
                recent_messages = list(chat_history.messages)[-6:]
                context_chunks = []
                for message in recent_messages:
                    role = message.get("role", "user")
                    content = message.get("content", "")
                    context_chunks.append(f"{role}: {content}")
                context_part = "\n".join(context_chunks)

            try:
                capsule_result = generate_insight_capsule(
                    topic_part,
                    context_part,
                    metadata={
                        "requested_by": "cli",
                        "environment": OPENAI_ENVIRONMENT,
                    },
                )
                print(f"Insight capsule saved to: {capsule_result['file_path']}")
                print(json.dumps(capsule_result["capsule"], indent=2))
            except Exception as exc:
                logging.error("Insight capsule generation failed: %s", exc)
                print(f"Failed to generate insight capsule: {exc}")
            continue
        elif user_input.startswith("/list_notes"):
            # Parse list command: /list_notes [category] [page]
            parts = user_input.split(maxsplit=3)
            category = None
            page = 1
            if len(parts) >= 2:
                category = parts[1]
            if len(parts) == 3:
                try:
                    page = int(parts[2])
                except ValueError:
                    print("Invalid page number. It must be an integer.")
                    continue
            files = list_notes(category, page)
            print(files)
            continue
        elif user_input.startswith("/view_note "):
            # Parse view command: /view_note <category> <filename>
            parts = user_input.split(maxsplit=2)
            if len(parts) == 3:
                category = parts[1]
                filename = parts[2]
                content = view_note(category, filename)
                print(content)
            else:
                print("Invalid view_note command format. Use: /view_note <category> <filename>")
            continue
        elif user_input.startswith("/delete_note "):
            # Parse delete command: /delete_note <category> <filename>
            parts = user_input.split(maxsplit=2)
            if len(parts) == 3:
                category = parts[1]
                filename = parts[2]
                result = delete_note(category, filename)
                print(result)
            else:
                print("Invalid delete_note command format. Use: /delete_note <category> <filename>")
            continue
        elif user_input.startswith("/search_notes "):
            # Parse search command: /search_notes <query>
            parts = user_input.split(maxsplit=1)
            if len(parts) == 2:
                query = parts[1]
                results = search_notes(query)
                print(results)
            else:
                print("Invalid search_notes command format. Use: /search_notes <query>")
            continue
        elif user_input.startswith("/rb_stats"):
            if reasoning_bank:
                print(json.dumps(reasoning_bank.stats(), indent=2))
            else:
                print("ReasoningBank is disabled.")
            continue
        elif user_input.startswith("/rb_recent"):
            if reasoning_bank:
                retrieved = reasoning_bank.retrieve(query=None, top_k=5)
                print(_format_memory_listing(retrieved))
            else:
                print("ReasoningBank is disabled.")
            continue
        elif user_input.startswith("/rb_query"):
            if not reasoning_bank:
                print("ReasoningBank is disabled.")
                continue
            parts = user_input.split(maxsplit=1)
            if len(parts) == 1 or not parts[1].strip():
                print("Usage: /rb_query <query>")
                continue
            query = parts[1].strip()
            retrieved = reasoning_bank.retrieve(query=query, top_k=5)
            print(_format_memory_listing(retrieved))
            continue
        elif user_input.startswith("/mtts "):
            payload = user_input[len("/mtts "):].strip()
            if "::" in payload:
                passes_part, task_part = [segment.strip() for segment in payload.split("::", 1)]
            else:
                passes_part, task_part = payload, ""
            if not task_part:
                # If no explicit delimiter, treat entire payload as task with default passes
                task_part = passes_part
                passes_value = 3
            else:
                try:
                    passes_value = int(passes_part)
                except ValueError:
                    print("Invalid passes value. Use /mtts <passes> :: <task>")
                    continue
            if not task_part:
                print("Usage: /mtts <passes> :: <task>")
                continue
            final_answer, trajectories, retrieved_memories, evaluation = memory_aware_mtts(task_part, passes=passes_value)
            chat_history.add_message("user", f"[MaTTS task] {task_part}")
            chat_history.add_message("assistant", final_answer)
            if retrieved_memories:
                print("Retrieved ReasoningBank items:")
                print(_format_memory_listing(retrieved_memories))
                print("")
            print("MaTTS trajectories:")
            for traj in trajectories:
                print(f"-- Pass {traj['index']} --\n{traj['content']}\n")
            print("Final synthesized answer:\n", final_answer)
            if evaluation:
                print("\nEvaluation:")
                print(json.dumps(evaluation, indent=2))
            continue
        elif user_input.startswith("/mtts_seq "):
            payload = user_input[len("/mtts_seq "):].strip()
            if "::" in payload:
                passes_part, task_part = [segment.strip() for segment in payload.split("::", 1)]
            else:
                passes_part, task_part = payload, ""
            if not task_part:
                task_part = passes_part
                passes_value = 3
            else:
                try:
                    passes_value = int(passes_part)
                except ValueError:
                    print("Invalid passes value. Use /mtts_seq <passes> :: <task>")
                    continue
            if not task_part:
                print("Usage: /mtts_seq <passes> :: <task>")
                continue
            final_answer, refinements, retrieved_memories, evaluation = memory_aware_mtts_sequential(
                task_part, passes=passes_value
            )
            chat_history.add_message("user", f"[MaTTS sequential task] {task_part}")
            chat_history.add_message("assistant", final_answer)
            if retrieved_memories:
                print("Retrieved ReasoningBank items:")
                print(_format_memory_listing(retrieved_memories))
                print("")
            print("Sequential refinements:")
            for step in refinements:
                print(f"-- Pass {step['index']} --\n{step['content']}\n")
            print("Final sequential answer:\n", final_answer)
            if evaluation:
                print("\nEvaluation:")
                print(json.dumps(evaluation, indent=2))
            continue
        elif user_input.startswith("/session "):
            session_task = user_input[len("/session "):].strip()
            if not session_task:
                print("Usage: /session <task>")
                continue
            run_task_session(session_task)
            continue
        # Add more custom commands as needed

        logging.info("Processing user input...")
        response = gpt4o_chat(user_input)
        logging.info("AI response: %s", response)
        print("\nAI:", response)
        print("\n" + "-" * 80 + "\n")

    logging.info("Conversation ended")


if __name__ == "__main__":
    main()
