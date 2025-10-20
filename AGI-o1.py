from openai import OpenAI
import os
import json
from dotenv import load_dotenv
import logging
from datetime import datetime
from collections import deque
from typing import Dict, Optional, List, Any, Tuple
from pathlib import Path
import re

from reasoning_bank import ReasoningBank

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

MODEL_ASSISTANT = os.getenv("OPENAI_MODEL_ASSISTANT", os.getenv("OPENAI_MODEL_REASONING", "gpt-5"))
MODEL_REASONING = os.getenv("OPENAI_MODEL_REASONING", "gpt-5")
MODEL_REASONING_EFFORT = os.getenv("OPENAI_MODEL_REASONING_EFFORT", "high")
MODEL_CODE = os.getenv("OPENAI_MODEL_CODE", "gpt-5-codex")
MODEL_REALTIME = os.getenv("OPENAI_MODEL_REALTIME", "gpt-realtime")
MODEL_REALTIME_MINI = os.getenv("OPENAI_MODEL_REALTIME_MINI", "gpt-realtime-mini")
MODEL_EMBEDDING = os.getenv("OPENAI_MODEL_EMBEDDING", "text-embedding-3-small")

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
    "code=%s, realtime=%s, realtime_mini=%s",
    OPENAI_ENVIRONMENT,
    MODEL_ASSISTANT,
    MODEL_REASONING,
    MODEL_REASONING_EFFORT,
    MODEL_CODE,
    MODEL_REALTIME,
    MODEL_REALTIME_MINI,
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
                "You are an advanced AI assistant functioning like a brain with specialized regions. "
                "Your primary objective is to provide high-quality, thoughtful responses. Key instructions:"
                "\n1. For ANY task requiring deep thinking, complex reasoning, or that a human would need to contemplate, "
                "ALWAYS use the 'deep_reasoning' function. This includes but is not limited to:"
                "\n   - Decision-making and problem-solving"
                "\n   - Logical reasoning and analysis"
                "\n   - Programming and technical tasks"
                "\n   - Complex STEM questions"
                "\n   - Creative thinking and ideation"
                "\n   - Ethical considerations"
                "\n   - Strategic planning"
                "\n   - Any task that a human would need to contemplate for a long time to decide on an answer."
                "\n   - Any task that requires you to think about what you are doing or thinking."
                "\n   - Any code related task. "
                "\n2. Analyze user queries thoroughly to determine if they require deep thinking."
                "\n3. Use 'retrieve_knowledge' for factual information retrieval when deep analysis isn't needed."
                "\n4. Use 'assist_user' for general interaction, writing assistance, and simple explanations."
                "\n5. Manage your memory proactively using note functions:"
                "\n   - 'save_note' to store important information"
                "\n   - 'edit_note' to update existing notes"
                "\n   - 'view_note' to recall stored information"
                "\n   - 'search_notes' to find relevant data"
                "\n   - 'list_notes' to review notes"
                "\n6. Use 'fetch_weather' for weather-related queries."
                "\n7. Always incorporate function results into your final response."
                "\n8. Provide clear, concise, and accurate information."
                "\n9. Continuously improve your knowledge by managing information in the notes. Acquire as much information as possible. Actively do this on your OWN."
                "\nMake independent decisions, prioritizing the use of 'deep_reasoning' for any non-trivial task. "
                "Your goal is to leverage your advanced capabilities to provide thoughtful, well-reasoned responses."
            )
        }

    def add_message(self, role, content, name=None):
        message = {"role": role, "content": content}
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
            logging.info(f"Edited message at index {index}.")

    def remove_message(self, index):
        if 0 <= index < len(self.messages):
            removed = self.messages[index]
            del self.messages[index]
            logging.info(f"Removed message at index {index}: {removed}")

chat_history = ChatHistory()

# Define available functions for the assistant
def get_available_functions():
    return [
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
            "description": "Generate and run Python code using the configured coding model (e.g., GPT-5-Codex). Only outputs the running code and its output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The coding task or question"
                    }
                },
                "required": ["query"]
            }
        },
    ]

def gpt4o_chat(user_input, is_initial_response=False, chat: ChatHistory = None):
    if chat is None:
        chat = chat_history
    logging.info(f"User input: {user_input}")
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
            functions=get_available_functions(),
            function_call="auto"
        )

        message = response.choices[0].message

        if message.function_call:
            function_name = message.function_call.name
            print(f"Function call detected: {function_name}")
            function_args = json.loads(message.function_call.arguments)
            logging.info(f"Function call detected: {function_name} with args {function_args}")

            # Handle function calls
            if function_name == "deep_reasoning":
                research_result = deep_reasoning(function_args['query'])
                chat.add_message("function", research_result, name=function_name)
            elif function_name == "retrieve_knowledge":
                librarian_result = retrieve_knowledge(function_args['query'])
                chat.add_message("function", librarian_result, name=function_name)
            elif function_name == "save_note":
                save_note(function_args['category'], function_args['filename'], function_args['content'])
                chat.add_message("function", f"Saved note to docs/paper.txt as {function_args['category']}/{function_args['filename']}.", name=function_name)
            elif function_name == "edit_note":
                edit_note_result = edit_note(function_args['category'], function_args['filename'], function_args['new_content'])
                chat.add_message("function", edit_note_result, name=function_name)
            elif function_name == "list_notes":
                files = list_notes(function_args.get('category'), function_args.get('page', 1))
                chat.add_message("function", files, name=function_name)
            elif function_name == "view_note":
                content = view_note(function_args['category'], function_args['filename'])
                chat.add_message("function", content, name=function_name)
            elif function_name == "delete_note":
                result = delete_note(function_args['category'], function_args['filename'])
                chat.add_message("function", result, name=function_name)
            elif function_name == "search_notes":
                search_results = search_notes(function_args['query'])
                chat.add_message("function", search_results, name=function_name)
            elif function_name == "fetch_weather":
                weather_result = fetch_weather(function_args['location'], function_args.get('unit', 'celsius'))
                chat.add_message("function", weather_result, name=function_name)
            elif function_name == "assist_user":
                interaction_result = assist_user(function_args['query'])
                chat.add_message("function", interaction_result, name=function_name)
            elif function_name == "coder":
                code_result = coder(function_args['query'])
                chat.add_message("function", code_result, name=function_name)
            else:
                logging.warning(f"Unknown function call: {function_name}")
                chat.add_message("function", f"Unknown function: {function_name}", name=function_name)
        else:
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
                        functions=get_available_functions(),
                        function_call="auto"
                    )

                    check_message = check_response.choices[0].message

                    if check_message.function_call:
                        print("Missed function call or unsaved information detected. Processing...")
                        chat.add_message("system", "Missed function call or unsaved information detected. Processing...")

                        function_name = check_message.function_call.name
                        function_args = json.loads(check_message.function_call.arguments)

                        if function_name == "save_note":
                            save_note(function_args['category'], function_args['filename'], function_args['content'])
                            chat.add_message("function", f"Saved note to docs/paper.txt as {function_args['category']}/{function_args['filename']}.", name=function_name)
                        else:
                            # Handle other function calls as needed
                            chat.add_message("function", f"Processed missed function call: {function_name}", name=function_name)

                        continue  # Continue the loop to process any additional missed function calls

                if message.content:
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
            else:
                logging.warning("Assistant response has no content and no function call")
                return "I'm sorry, I didn't understand that."

    # If we've reached the maximum number of iterations, return the last response
    return "I apologize, but I'm having trouble processing your request. Could you please rephrase or provide more information?"

def deep_reasoning(query):
    prompt = f"Research query: {query}"
    _log_prompt("deep_reasoning", prompt)
    response_text = None

    try:
        reasoning_kwargs = (
            {"effort": MODEL_REASONING_EFFORT}
            if MODEL_REASONING_EFFORT
            else None
        )
        response = client.responses.create(
            model=MODEL_REASONING,
            input=prompt,
            **({"reasoning": reasoning_kwargs} if reasoning_kwargs else {}),
        )
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

    logging.info("Reasoning flow completed")
    _log_response("deep_reasoning", response_text or "")

    # Log response to a separate file for auditing
    o1_log_filename = os.path.join(
        o1_responses_dir, f"reasoning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    with open(o1_log_filename, "w") as f:
        f.write(f"Query: {query}\n\nResponse:\n{response_text}")

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
        logging.error(f"Error in librarian function: {str(e)}")
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
        logging.info(f"Retrieved weather information for {location}: {weather_info}")
        return weather_info
    except Exception as e:
        logging.error(f"Error retrieving weather information: {str(e)}")
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

def coder(query):
    """
    Function to generate and run Python code using the configured coding model.
    """
    prompt = (
        "Generate and run Python code for the following task. "
        "Only output the running Python code and its output. No explanations or additional text.\n"
        f"Task: {query}"
    )
    _log_prompt("coder", prompt)

    try:
        response = client.responses.create(
            model=MODEL_CODE,
            input=prompt,
            max_output_tokens=1200,
        )
        model_response = response.output_text
    except Exception as primary_exc:
        logging.warning(
            "Primary coding model %s failed (%s). Falling back to assistant model.",
            MODEL_CODE,
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

    logging.info("Code generation completed with primary/fallback model")
    o1_log_filename = os.path.join(
        o1_responses_dir, f"code_response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    with open(o1_log_filename, "w") as f:
        f.write(f"Query: {query}\n\nResponse:\n{model_response}")

    _log_response("coder", model_response or "")
    return model_response


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
    print("Welcome to the AGI-o1 System. Type 'exit', 'quit', or 'bye' to end the conversation.")
    print("Available Commands:")
    print("  /edit <index> <new_content>                - Edit a message at a specific index.")
    print("  /remove <index>                            - Remove a message at a specific index.")
    print("  /save_note <category> <filename> <content> - Save a note to docs/paper.txt.")
    print("  /edit_note <category> <filename> <new_content> - Edit an existing note.")
    print("  /list_notes [category] [page]              - List notes, optionally within a category and paginated.")
    print("  /view_note <category> <filename>           - View the content of a note.")
    print("  /delete_note <category> <filename>         - Delete a note.")
    print("  /search_notes <query>                      - Search for a query string within all notes.")
    print("  /rb_stats                                  - Show ReasoningBank metadata.")
    print("  /rb_query <query>                          - Retrieve relevant ReasoningBank items.")
    print("  /rb_recent                                 - Show recent ReasoningBank items.")
    print("  /mtts <passes> :: <task>                   - Run MaTTS multi-pass reasoning (default passes=3).")
    print("  /mtts_seq <passes> :: <task>               - Run sequential MaTTS self-refinement.")
    print("  /session <task>                            - Start an interactive task sub-loop.")
    print("  /insight <topic> :: <context>              - Generate an Insight Capsule (context optional).")
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
        logging.info(f"AI response: {response}")
        print("\nAI:", response)
        print("\n" + "-" * 80 + "\n")

    logging.info("Conversation ended")

if __name__ == "__main__":
    main()

# Additional Function Call Examples

# Example 1: Getting Current Weather
def example_weather():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            }
        }
    ]
    messages = [{"role": "user", "content": "What's the weather like in Boston today?"}]
    completion = client.chat.completions.create(
        model=MODEL_ASSISTANT,
        messages=messages,
        functions=tools,
        function_call="auto"
    )

    print(completion)

# Example 2: Handling Images (Functionality to be implemented later)
def example_image_handling():
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                }
            },
        ],
    }]
    response = client.chat.completions.create(
        model=MODEL_ASSISTANT,
        messages=messages,
        max_tokens=300,
    )

    print(response.choices[0])

# Uncomment the following lines to run the examples
# example_weather()
# example_image_handling()
