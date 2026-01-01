"""
Tool definitions and execution for AGI-o1.

Contains function schemas for the AI assistant and tool execution dispatch.
"""
import json
import logging
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

from constants import REASONING_EFFORTS


# Type alias for tool function
ToolFunction = Callable[..., str]


def get_function_schemas() -> List[Dict[str, Any]]:
    """
    Get the function/tool schemas for the AI assistant.
    
    Returns:
        List of function specification dictionaries
    """
    return [
        {
            "name": "deep_reasoning",
            "description": (
                "Utilize GPT-5.2 advanced cognitive processing for deep reasoning. "
                "Supports xhigh reasoning effort for the hardest challenges."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The complex question or problem to analyze"
                    },
                    "reasoning_effort": {
                        "type": "string",
                        "enum": list(REASONING_EFFORTS),
                        "description": "Reasoning effort level override"
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
            "description": "Provide general assistance, interact with the user, and offer explanations.",
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
            "description": (
                "Generate and run Python code using GPT-5.2-Codex. "
                "Supports 400K context, 128K output, native compaction for long-running tasks."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The coding task or question"
                    },
                    "use_max_model": {
                        "type": "boolean",
                        "description": "Use GPT-5.1-Codex-Max for long-horizon agentic tasks",
                        "default": False
                    },
                    "reasoning_effort": {
                        "type": "string",
                        "enum": list(REASONING_EFFORTS),
                        "description": "Reasoning effort level. Use 'xhigh' for hardest challenges"
                    }
                },
                "required": ["query"]
            }
        },
        {
            "name": "coder_xhigh",
            "description": (
                "Generate code using GPT-5.2-Codex with xhigh reasoning effort "
                "for the most complex engineering challenges."
            ),
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
    ]


def get_tools_schema() -> List[Dict[str, Any]]:
    """
    Convert function specifications into the tools schema expected by newer models.
    
    Returns:
        List of tool specification dictionaries
    """
    return [{"type": "function", "function": spec} for spec in get_function_schemas()]


class ToolExecutor:
    """
    Executes tool/function calls and returns results.
    
    Provides a registry-based dispatch mechanism for tool execution.
    """
    
    def __init__(self):
        self._registry: Dict[str, ToolFunction] = {}
    
    def register(self, name: str, func: ToolFunction) -> None:
        """
        Register a tool function.
        
        Args:
            name: Tool name matching the schema
            func: Function to execute for this tool
        """
        self._registry[name] = func
    
    def execute(self, function_name: str, function_args: Dict[str, Any]) -> str:
        """
        Execute a tool and return the result.
        
        Args:
            function_name: Name of the tool to execute
            function_args: Arguments for the tool
        
        Returns:
            String result from the tool execution
        """
        try:
            if function_name not in self._registry:
                logging.warning("Unknown function call requested: %s", function_name)
                return f"Unknown function: {function_name}"
            
            func = self._registry[function_name]
            return func(**function_args)
            
        except KeyError as missing:
            logging.error(
                "Missing argument '%s' for function '%s'. Args: %s",
                missing, function_name, function_args
            )
            return f"Missing argument '{missing}' for function '{function_name}'."
        except TypeError as type_err:
            logging.error(
                "Type error executing '%s': %s. Args: %s",
                function_name, type_err, function_args
            )
            return f"Invalid arguments for '{function_name}': {type_err}"
        except Exception as exc:
            logging.exception("Error executing function '%s': %s", function_name, exc)
            return f"Error while executing '{function_name}': {exc}"


def parse_tool_call_arguments(arguments: Optional[str]) -> Dict[str, Any]:
    """
    Safely parse tool call arguments from JSON string.
    
    Args:
        arguments: JSON string of arguments, or None
    
    Returns:
        Parsed dictionary or empty dict on failure
    """
    if not arguments:
        return {}
    try:
        return json.loads(arguments)
    except json.JSONDecodeError as e:
        logging.warning("Failed to decode tool arguments: %s", e)
        return {}


def build_tool_call_message(tool_calls: List[Any]) -> Dict[str, Any]:
    """
    Build a structured assistant message from tool calls.
    
    Args:
        tool_calls: List of tool call objects from API response
    
    Returns:
        Properly structured message dictionary for chat history
    """
    return {
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


def build_legacy_function_call_message(function_call: Any) -> Dict[str, Any]:
    """
    Build a structured message from a legacy function call.
    
    Args:
        function_call: Legacy function call object
    
    Returns:
        Properly structured message dictionary for chat history
    """
    call_id = getattr(function_call, "id", None) or f"legacy_{uuid4().hex}"
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": function_call.name,
                    "arguments": function_call.arguments,
                },
            }
        ],
    }
