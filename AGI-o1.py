from openai import OpenAI
import os
import json
from dotenv import load_dotenv
import logging
from datetime import datetime
from collections import deque

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Create necessary directories if they don't exist
log_dir = "logs"
o1_responses_dir = "o1_responses"
scratch_pad_dir = "scratch_pad"  # Directory for scratch pad files
os.makedirs(log_dir, exist_ok=True)
os.makedirs(o1_responses_dir, exist_ok=True)
os.makedirs(scratch_pad_dir, exist_ok=True)

# Set up logging
log_filename = os.path.join(log_dir, f"chat_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(filename=log_filename, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

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

    def get_messages(self):
        return [self.system_message] + list(self.messages)

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
            "description": "Save a note to the scratch pad under a specified category and filename.",
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
            "description": "Edit the content of an existing note in the scratch pad within a specified category.",
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
            "description": "List all notes in the scratch pad, optionally filtered by category and page.",
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
            "description": "Display the content of a specific note from the scratch pad within a specified category.",
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
            "description": "Delete a note from the scratch pad within a specified category.",
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
            "description": "Search for a specific query within all notes in the scratch pad.",
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
            "description": "Provide general assistance, interact with the user, and offer explanations using GPT-4o.",
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
    ]

def gpt4o_chat(user_input, is_initial_response=False):
    logging.info(f"User input: {user_input}")
    chat_history.add_message("user", user_input)
    
    while True:
        print("Calling GPT-4o for response...")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=chat_history.get_messages(),
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
                chat_history.add_message("function", research_result, name=function_name)
            elif function_name == "retrieve_knowledge":
                librarian_result = retrieve_knowledge(function_args['query'])
                chat_history.add_message("function", librarian_result, name=function_name)
            elif function_name == "save_note":
                save_note(function_args['category'], function_args['filename'], function_args['content'])
                chat_history.add_message("function", f"Saved note to {function_args['category']}/{function_args['filename']}.txt.", name=function_name)
            elif function_name == "edit_note":
                edit_note_result = edit_note(function_args['category'], function_args['filename'], function_args['new_content'])
                chat_history.add_message("function", edit_note_result, name=function_name)
            elif function_name == "list_notes":
                files = list_notes(function_args.get('category'), function_args.get('page', 1))
                chat_history.add_message("function", files, name=function_name)
            elif function_name == "view_note":
                content = view_note(function_args['category'], function_args['filename'])
                chat_history.add_message("function", content, name=function_name)
            elif function_name == "delete_note":
                result = delete_note(function_args['category'], function_args['filename'])
                chat_history.add_message("function", result, name=function_name)
            elif function_name == "search_notes":
                search_results = search_notes(function_args['query'])
                chat_history.add_message("function", search_results, name=function_name)
            elif function_name == "fetch_weather":
                weather_result = fetch_weather(function_args['location'], function_args.get('unit', 'celsius'))
                chat_history.add_message("function", weather_result, name=function_name)
            elif function_name == "assist_user":
                interaction_result = assist_user(function_args['query'])
                chat_history.add_message("function", interaction_result, name=function_name)
            else:
                logging.warning(f"Unknown function call: {function_name}")
                chat_history.add_message("function", f"Unknown function: {function_name}", name=function_name)
        else:
            if message.content:
                chat_history.add_message("assistant", message.content)
                
                if not is_initial_response:
                    # Check for missed function calls or unsaved information
                    check_response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant tasked with checking if any function calls were missed or if any information was not saved in the previous response. If you find any missed actions, please call the appropriate function."},
                            {"role": "user", "content": f"Previous response: {message.content}\n\nCheck if any function calls were missed or if any information was not saved."}
                        ],
                        functions=get_available_functions(),
                        function_call="auto"
                    )
                    
                    check_message = check_response.choices[0].message
                    
                    if check_message.function_call:
                        print("Missed function call detected. Processing...")
                        chat_history.add_message("system", "Missed function call detected. Processing...")
                        continue  # Continue the loop to process the missed function call
                
                return message.content  # If no missed calls or initial response, return the original response
            else:
                logging.warning("Assistant response has no content and no function call")
                return "I'm sorry, I didn't understand that."

def deep_reasoning(query):
    response = client.chat.completions.create(
        model="o1-preview",
        messages=[
            {"role": "user", "content": f"Research query: {query}"}
        ],
    )
    logging.info("o1 research completed")

    # Log O1 response to a separate file
    o1_response = response.choices[0].message.content
    o1_log_filename = os.path.join(o1_responses_dir, f"o1_response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    with open(o1_log_filename, 'w') as f:
        f.write(f"Query: {query}\n\nResponse:\n{o1_response}")

    return o1_response

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
            model="gpt-4o"
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
    Save content to a scratch pad file within a specified category.
    """
    try:
        category_path = os.path.join(scratch_pad_dir, category)
        os.makedirs(category_path, exist_ok=True)
        file_path = os.path.join(category_path, f"{filename}.txt")
        with open(file_path, 'w') as f:
            f.write(content)
        logging.info(f"Saved scratch pad file: {file_path}")
    except Exception as e:
        logging.error(f"Error saving scratch pad file {filename} in category {category}: {str(e)}")

def edit_note(category, filename, new_content):
    """
    Edit content of an existing scratch pad file within a specified category.
    If the file doesn't exist, create it.
    """
    try:
        category_path = os.path.join(scratch_pad_dir, category)
        os.makedirs(category_path, exist_ok=True)
        file_path = os.path.join(category_path, f"{filename}.txt")
        
        # Check if the file exists
        file_existed = os.path.exists(file_path)
        
        # Write the new content (this will create the file if it doesn't exist)
        with open(file_path, 'w') as f:
            f.write(new_content)
        
        if file_existed:
            logging.info(f"Edited existing scratch pad file: {file_path}")
            return f"Edited existing scratch pad file: {category}/{filename}.txt"
        else:
            logging.info(f"Created new scratch pad file: {file_path}")
            return f"Created new scratch pad file: {category}/{filename}.txt"
    except Exception as e:
        error_msg = f"Error editing/creating scratch pad file {filename} in category {category}: {str(e)}"
        logging.error(error_msg)
        return error_msg

def list_notes(category=None, page=1, page_size=10):
    try:
        if not os.path.exists(scratch_pad_dir):
            return f"Scratch pad directory does not exist: {scratch_pad_dir}"
        
        files = []
        if category:
            category_path = os.path.join(scratch_pad_dir, category)
            if not os.path.exists(category_path):
                return f"Category '{category}' does not exist."
            for f in os.listdir(category_path):
                if os.path.isfile(os.path.join(category_path, f)):
                    files.append(f"{category}/{f}")
        else:
            for cat in os.listdir(scratch_pad_dir):
                cat_path = os.path.join(scratch_pad_dir, cat)
                if os.path.isdir(cat_path):
                    for f in os.listdir(cat_path):
                        if os.path.isfile(os.path.join(cat_path, f)):
                            files.append(f"{cat}/{f}")

        if not files:
            return f"No scratch pad files available in directory: {scratch_pad_dir}"

        # Implement pagination
        total_files = len(files)
        total_pages = (total_files + page_size - 1) // page_size
        if page < 1 or page > total_pages:
            return f"Invalid page number. There are {total_pages} pages available."

        start = (page - 1) * page_size
        end = start + page_size
        paginated_files = files[start:end]
        file_list = "\n".join(paginated_files)
        logging.info(f"Listed scratch pad files for page {page}.")

        return f"Available scratch pad files (Page {page}/{total_pages}):\n{file_list}"
    except Exception as e:
        logging.error(f"Error listing scratch pad files: {str(e)}")
        return f"An error occurred while listing scratch pad files: {str(e)}\nScratch pad directory: {scratch_pad_dir}"

def view_note(category, filename):
    """
    View the content of a scratch pad file within a specified category.
    """
    try:
        file_path = os.path.join(scratch_pad_dir, category, f"{filename}.txt")
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                content = f.read()
            logging.info(f"Viewed scratch pad file: {file_path}")
            return f"Content of '{category}/{filename}.txt':\n\n{content}"
        else:
            logging.warning(f"Scratch pad file {file_path} does not exist.")
            return f"Scratch pad file '{category}/{filename}.txt' does not exist."
    except Exception as e:
        logging.error(f"Error viewing scratch pad file {filename} in category {category}: {str(e)}")
        return f"An error occurred while viewing the scratch pad file: {str(e)}"

def delete_note(category, filename):
    """
    Delete a scratch pad file within a specified category.
    """
    try:
        file_path = os.path.join(scratch_pad_dir, category, f"{filename}.txt")
        if os.path.exists(file_path):
            os.remove(file_path)
            logging.info(f"Deleted scratch pad file: {file_path}")
            return f"Deleted scratch pad file '{category}/{filename}.txt'."
        else:
            logging.warning(f"Scratch pad file {file_path} does not exist.")
            return f"Scratch pad file '{category}/{filename}.txt' does not exist."
    except Exception as e:
        logging.error(f"Error deleting scratch pad file {filename} in category {category}: {str(e)}")
        return f"An error occurred while deleting the scratch pad file: {str(e)}"

def search_notes(query):
    """
    Search for a query string within all scratch pad files.
    Returns a list of files that contain the query string.
    """
    try:
        matching_files = []
        for category in os.listdir(scratch_pad_dir):
            category_path = os.path.join(scratch_pad_dir, category)
            if os.path.isdir(category_path):
                for f in os.listdir(category_path):
                    file_path = os.path.join(category_path, f)
                    if os.path.isfile(file_path):
                        with open(file_path, 'r') as file:
                            content = file.read()
                            if query.lower() in content.lower():
                                matching_files.append(f"{category}/{f}")
        if not matching_files:
            return f"No scratch pad files contain the query '{query}'."
        file_list = "\n".join(matching_files)
        logging.info(f"Searched scratch pad files for query '{query}'. Found {len(matching_files)} matches.")
        return f"Files containing '{query}':\n{file_list}"
    except Exception as e:
        logging.error(f"Error searching scratch pad files: {str(e)}")
        return f"An error occurred while searching scratch pad files: {str(e)}"

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
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": query}
        ],
    )
    logging.info("gpt4o interaction completed")
    return response.choices[0].message.content

def read_all_scratch_pad_files():
    all_content = []
    for category in os.listdir(scratch_pad_dir):
        category_path = os.path.join(scratch_pad_dir, category)
        if os.path.isdir(category_path):
            for filename in os.listdir(category_path):
                file_path = os.path.join(category_path, filename)
                if os.path.isfile(file_path):
                    with open(file_path, 'r') as f:
                        content = f.read()
                        all_content.append(f"Category: {category}, File: {filename}\n{content}\n")
    return "\n".join(all_content)

# Main interaction loop
def main():
    print("Welcome to the AGI-o1 System. Type 'exit', 'quit', or 'bye' to end the conversation.")
    print("Available Commands:")
    print("  /edit <index> <new_content>                - Edit a message at a specific index.")
    print("  /remove <index>                            - Remove a message at a specific index.")
    print("  /save_note <category> <filename> <content> - Save a note to the scratch pad.")
    print("  /edit_note <category> <filename> <new_content> - Edit an existing note.")
    print("  /list_notes [category] [page]              - List notes, optionally within a category and paginated.")
    print("  /view_note <category> <filename>           - View the content of a note.")
    print("  /delete_note <category> <filename>         - Delete a note.")
    print("  /search_notes <query>                      - Search for a query string within all notes.")
    print("-" * 80)

    # Read all scratch pad files
    scratch_pad_content = read_all_scratch_pad_files()

    # Start the conversation by summarizing scratch pad files
    initial_query = f"""Here is the content of all scratch pad files:

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
                print(f"Content saved to scratch pad as '{category}/{filename}.txt'.")
            else:
                print("Invalid save_note command format. Use: /save_note <category> <filename> <content>")
            continue
        elif user_input.startswith("/edit_note "):
            # Parse edit scratch pad command: /edit_note <category> <filename> <new_content>
            parts = user_input.split(maxsplit=4)
            if len(parts) == 4:
                category = parts[1]
                filename = parts[2]
                new_content = parts[3]
                edit_note(category, filename, new_content)
                print(f"Scratch pad '{category}/{filename}.txt' edited.")
            else:
                print("Invalid edit_note command format. Use: /edit_note <category> <filename> <new_content>")
            continue
        elif user_input.startswith("/list_notes"):
            # Parse list_scratch_pad command: /list_notes [category] [page]
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
            # Parse view scratch pad command: /view_note <category> <filename>
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
            # Parse delete scratch pad command: /delete_note <category> <filename>
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
            # Parse search scratch pad command: /search_notes <query>
            parts = user_input.split(maxsplit=1)
            if len(parts) == 2:
                query = parts[1]
                results = search_notes(query)
                print(results)
            else:
                print("Invalid search_notes command format. Use: /search_notes <query>")
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
        model="gpt-4o",
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
        model="gpt-4o",
        messages=messages,
        max_tokens=300,
    )

    print(response.choices[0])

# Uncomment the following lines to run the examples
# example_weather()
# example_image_handling()