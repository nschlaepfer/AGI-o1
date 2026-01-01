"""
Chat history management for AGI-o1.

Provides the ChatHistory class for managing conversation state.
"""
import json
import logging
from collections import deque
from typing import Any, Dict, List, Optional

from amarillo.core.constants import DEFAULT_MAX_MESSAGES, DEFAULT_DIALOGUE_TURNS
from amarillo.core.utils import format_message_for_display, render_message_content


# System prompt for the AI assistant
SYSTEM_PROMPT = """You are an advanced AI assistant functioning like a brain with specialized regions. \
Your primary objective is to provide high-quality, thoughtful responses. Key instructions:

1. For ANY task requiring deep thinking, complex reasoning, or that a human would need to contemplate, \
ALWAYS use the 'deep_reasoning' function. This includes but is not limited to:
   - Decision-making and problem-solving
   - Logical reasoning and analysis
   - Programming and technical tasks
   - Complex STEM questions
   - Creative thinking and ideation
   - Ethical considerations
   - Strategic planning
   - Any task that a human would need to contemplate for a long time to decide on an answer.
   - Any task that requires you to think about what you are doing or thinking.
   - Any code related task.

2. Analyze user queries thoroughly to determine if they require deep thinking.
3. Use 'retrieve_knowledge' for factual information retrieval when deep analysis isn't needed.
4. Use 'assist_user' for general interaction, writing assistance, and simple explanations.
5. Manage your memory proactively using note functions:
   - 'save_note' to store important information
   - 'edit_note' to update existing notes
   - 'view_note' to recall stored information
   - 'search_notes' to find relevant data
   - 'list_notes' to review notes
6. Use 'fetch_weather' for weather-related queries.
7. Always incorporate function results into your final response.
8. Provide clear, concise, and accurate information.
9. Continuously improve your knowledge by managing information in the notes. \
Acquire as much information as possible. Actively do this on your OWN.

Make independent decisions, prioritizing the use of 'deep_reasoning' for any non-trivial task. \
Your goal is to leverage your advanced capabilities to provide thoughtful, well-reasoned responses."""


class ChatHistory:
    """
    Manages conversation history with bounded message storage.
    
    Provides methods for adding, editing, and retrieving messages,
    as well as rendering conversation transcripts.
    """
    
    def __init__(self, max_messages: int = DEFAULT_MAX_MESSAGES):
        """
        Initialize chat history with a maximum message limit.
        
        Args:
            max_messages: Maximum number of messages to retain
        """
        self.messages: deque = deque(maxlen=max_messages)
        self.system_message: Dict[str, str] = {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }

    def add_message(
        self,
        role: str,
        content: str,
        name: Optional[str] = None
    ) -> None:
        """
        Add a simple text message to the history.
        
        Args:
            role: Message role (user, assistant, system)
            content: Message content
            name: Optional name for the message author
        """
        message: Dict[str, Any] = {"role": role, "content": content}
        if name:
            message["name"] = name
        self.messages.append(message)

    def add_raw_message(self, message: Dict[str, Any]) -> None:
        """
        Append a pre-constructed message dictionary to the history.
        
        Args:
            message: Complete message dictionary
        """
        self.messages.append(dict(message))

    def add_tool_message(
        self,
        tool_call_id: str,
        content: Any,
        name: Optional[str] = None
    ) -> None:
        """
        Append a tool (function) response using the modern tool role format.
        
        Args:
            tool_call_id: ID of the tool call this responds to
            content: Result content (will be JSON-serialized if not a string)
            name: Optional function name
        """
        if isinstance(content, str):
            rendered_content = content
        else:
            try:
                rendered_content = json.dumps(content)
            except TypeError:
                rendered_content = str(content)
        
        message: Dict[str, Any] = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": rendered_content,
        }
        if name:
            message["name"] = name
        self.messages.append(message)

    def get_messages(self, memory_prompt: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all messages including system message for API calls.
        
        Args:
            memory_prompt: Optional memory context to inject
        
        Returns:
            List of message dictionaries ready for API call
        """
        messages = [dict(self.system_message)]
        if memory_prompt:
            messages.append({"role": "system", "content": memory_prompt})
        messages.extend(list(self.messages))
        return messages

    def _render_messages(
        self,
        messages: List[Dict[str, Any]],
        include_system: bool = False
    ) -> str:
        """
        Internal method to render a list of messages as a transcript.
        
        Args:
            messages: List of messages to render
            include_system: Whether to include system messages
        
        Returns:
            Formatted transcript string
        """
        lines: List[str] = []
        for msg in messages:
            formatted = format_message_for_display(msg, include_system=include_system)
            if formatted:
                lines.append(formatted)
        return "\n".join(lines)

    def recent_dialogue(self, turns: int = DEFAULT_DIALOGUE_TURNS) -> str:
        """
        Get a transcript of recent dialogue turns.
        
        Args:
            turns: Number of conversation turns to include
        
        Returns:
            Formatted transcript of recent messages
        """
        # Each turn typically includes user + assistant messages
        relevant = list(self.messages)[-turns * 2:]
        return self._render_messages(relevant)

    def render_dialogue(self, include_system: bool = False) -> str:
        """
        Render the full conversation as a transcript.
        
        Args:
            include_system: Whether to include system messages
        
        Returns:
            Formatted transcript of all messages
        """
        return self._render_messages(list(self.messages), include_system=include_system)

    def last_assistant_message(self) -> Optional[str]:
        """
        Get the content of the last assistant message.
        
        Returns:
            Content string or None if no assistant message found
        """
        for msg in reversed(self.messages):
            if msg.get("role") == "assistant":
                return msg.get("content")
        return None

    def edit_message(self, index: int, new_content: str) -> bool:
        """
        Edit a message at the specified index.
        
        Args:
            index: Position of the message to edit
            new_content: New content for the message
        
        Returns:
            True if successful, False if index out of range
        """
        if 0 <= index < len(self.messages):
            self.messages[index]["content"] = new_content
            logging.info("Edited message at index %d.", index)
            return True
        logging.warning("Failed to edit message: index %d out of range.", index)
        return False

    def remove_message(self, index: int) -> bool:
        """
        Remove a message at the specified index.
        
        Args:
            index: Position of the message to remove
        
        Returns:
            True if successful, False if index out of range
        """
        if 0 <= index < len(self.messages):
            removed = self.messages[index]
            del self.messages[index]
            logging.info("Removed message at index %d: %s", index, removed)
            return True
        logging.warning("Failed to remove message: index %d out of range.", index)
        return False

    def clear(self) -> None:
        """Clear all messages from history."""
        self.messages.clear()
        logging.info("Chat history cleared.")

    def __len__(self) -> int:
        """Return the number of messages in history."""
        return len(self.messages)
