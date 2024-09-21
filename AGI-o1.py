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
                "You are an advanced AI assistant with access to various functions. Your role is to:"
                "\n1. Analyze user queries thoroughly."
                "\n2. Autonomously decide which function, if any, is most appropriate to use."
                "\n3. For complex STEM questions, prefer the 'o1_research' function."
                "\n4. Use 'librarian' for general information retrieval."
                "\n5. Utilize scratch pad functions proactively for your own benefit:"
                "\n   - Use 'save_scratch_pad' to store important information for future reference."
                "\n   - Use 'edit_scratch_pad' to update existing notes as needed."
                "\n   - Use 'view_scratch_pad' to recall stored information."
                "\n   - Use 'search_scratch_pad' to find relevant stored data."
                "\n   - Use 'list_scratch_pad_files' to review available notes."
                "\n   - Organize information into appropriate categories for easy retrieval."
                "\n6. Apply 'get_current_weather' for weather-related queries."
                "\n7. If no function is needed, respond directly using your knowledge."
                "\n8. Always incorporate function results into your final response."
                "\n9. Provide clear, concise, and accurate information."
                "\n10. Continuously improve your knowledge base by storing and updating information in the scratch pad."
                "\nMake decisions independently and use the most suitable approach for each query, "
                "leveraging the scratch pad system to enhance your capabilities over time."
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
            "name": "o1_research",
            "description": "Perform complex reasoning for STEM (Science, Technology, Engineering, Mathematics) tasks that relate to the user's query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The STEM question to be researched"
                    }
                },
                "required": ["query"]
            }
        },
        {
            "name": "librarian",
            "description": "Retrieve information using OpenAI's Retrieval tool.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query to search for information"
                    }
                },
                "required": ["query"]
            }
        },
        {
            "name": "save_scratch_pad",
            "description": "Save content to a scratch pad file within a specified category.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "The category under which to save the file."
                    },
                    "filename": {
                        "type": "string",
                        "description": "The name of the file to save the content."
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to save in the file."
                    }
                },
                "required": ["category", "filename", "content"]
            }
        },
        {
            "name": "edit_scratch_pad",
            "description": "Edit content of an existing scratch pad file within a specified category.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "The category of the file to edit."
                    },
                    "filename": {
                        "type": "string",
                        "description": "The name of the file to edit."
                    },
                    "new_content": {
                        "type": "string",
                        "description": "The new content to replace the existing content."
                    }
                },
                "required": ["category", "filename", "new_content"]
            }
        },
        {
            "name": "list_scratch_pad_files",
            "description": "List all available scratch pad files, optionally within a specific category and paginated.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "The category to list files from. If omitted, lists from all categories."
                    },
                    "page": {
                        "type": "integer",
                        "description": "The page number to display."
                    }
                },
                "required": []
            }
        },
        {
            "name": "view_scratch_pad",
            "description": "View the content of a scratch pad file within a specified category.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "The category of the file to view."
                    },
                    "filename": {
                        "type": "string",
                        "description": "The name of the file to view."
                    }
                },
                "required": ["category", "filename"]
            }
        },
        {
            "name": "delete_scratch_pad",
            "description": "Delete a scratch pad file within a specified category.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "The category of the file to delete."
                    },
                    "filename": {
                        "type": "string",
                        "description": "The name of the file to delete."
                    }
                },
                "required": ["category", "filename"]
            }
        },
        {
            "name": "search_scratch_pad",
            "description": "Search for a query string within all scratch pad files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The string to search for within scratch pad files."
                    }
                },
                "required": ["query"]
            }
        },
        {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Unit for temperature."
                    }
                },
                "required": ["location"]
            }
        },
        # Placeholder for image handling functions
        # {
        #     "name": "handle_image",
        #     "description": "Process an image and provide a description.",
        #     "parameters": {
        #         "type": "object",
        #         "properties": {
        #             "image_url": {
        #                 "type": "string",
        #                 "description": "URL of the image to process."
        #             }
        #         },
        #         "required": ["image_url"]
        #     }
        # }
    ]

def gpt4o_chat(user_input):
    logging.info(f"User input: {user_input}")
    chat_history.add_message("user", user_input)
    print("Calling GPT-4o for initial response...")
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=chat_history.get_messages(),
        functions=get_available_functions(),
        function_call="auto"
    )

    message = response.choices[0].message

    if message.content:
        print(f"Initial response: {message.content[:50]}...")
    else:
        print("Model chose to use a function call instead of responding directly.")

    if message.function_call:
        function_name = message.function_call.name
        print(f"Function call: {function_name}")
        function_args = json.loads(message.function_call.arguments)
        logging.info(f"Function call detected: {function_name} with args {function_args}")

        # Handle function calls
        if function_name == "o1_research":
            research_result = o1_research(function_args['query'])
            # Add function result to chat history with the 'name' parameter
            chat_history.add_message("function", research_result, name=function_name)
        elif function_name == "librarian":
            librarian_result = librarian(function_args['query'])
            chat_history.add_message("function", librarian_result, name=function_name)
        elif function_name == "save_scratch_pad":
            save_scratch_pad(function_args['category'], function_args['filename'], function_args['content'])
            chat_history.add_message("function", f"Saved content to {function_args['category']}/{function_args['filename']}.txt.", name=function_name)
        elif function_name == "edit_scratch_pad":
            edit_scratch_pad(function_args['category'], function_args['filename'], function_args['new_content'])
            chat_history.add_message("function", f"Edited content of {function_args['category']}/{function_args['filename']}.txt.", name=function_name)
        elif function_name == "list_scratch_pad_files":
            files = list_scratch_pad_files(function_args.get('category'), function_args.get('page'))
            chat_history.add_message("function", files, name=function_name)
        elif function_name == "view_scratch_pad":
            content = view_scratch_pad(function_args['category'], function_args['filename'])
            chat_history.add_message("function", content, name=function_name)
        elif function_name == "delete_scratch_pad":
            result = delete_scratch_pad(function_args['category'], function_args['filename'])
            chat_history.add_message("function", result, name=function_name)
        elif function_name == "search_scratch_pad":
            search_results = search_scratch_pad(function_args['query'])
            chat_history.add_message("function", search_results, name=function_name)
        elif function_name == "get_current_weather":
            weather_result = get_current_weather(function_args['location'], function_args.get('unit', 'celsius'))
            chat_history.add_message("function", weather_result, name=function_name)
        # elif function_name == "handle_image":
            # Handle image processing here
        else:
            logging.warning(f"Unknown function call: {function_name}")
            chat_history.add_message("function", f"Unknown function: {function_name}", name=function_name)

        # Call GPT-4o again with the function response
        logging.info("Calling GPT-4o for final response with function results...")
        print("Calling GPT-4o for final response with function results...")
        final_response = client.chat.completions.create(
            model="gpt-4o",
            messages=chat_history.get_messages()
        )
        final_message = final_response.choices[0].message
        print(f"Final response type: {final_message.content[:50]}...")
        chat_history.add_message("assistant", final_message.content)
        return final_message.content
    else:
        logging.info("No function call detected, returning initial response")
        print("No function call detected, returning initial response")
        chat_history.add_message("assistant", message.content)
        return message.content

def o1_research(query):
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

def librarian(query):
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

def save_scratch_pad(category, filename, content):
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

def edit_scratch_pad(category, filename, new_content):
    """
    Edit content of an existing scratch pad file within a specified category.
    """
    try:
        file_path = os.path.join(scratch_pad_dir, category, f"{filename}.txt")
        if os.path.exists(file_path):
            with open(file_path, 'w') as f:
                f.write(new_content)
            logging.info(f"Edited scratch pad file: {file_path}")
        else:
            logging.warning(f"Scratch pad file {file_path} does not exist.")
    except Exception as e:
        logging.error(f"Error editing scratch pad file {filename} in category {category}: {str(e)}")

def list_scratch_pad_files(category=None, page=1, page_size=10):
    """
    List all available scratch pad files, optionally within a specific category and paginated.
    """
    try:
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
            return "No scratch pad files available."

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
        return f"An error occurred while listing scratch pad files: {str(e)}"

def view_scratch_pad(category, filename):
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

def delete_scratch_pad(category, filename):
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

def search_scratch_pad(query):
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

def get_current_weather(location, unit='celsius'):
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

# Placeholder for image handling functionality
# def handle_image(image_url):
#     """
#     Process an image and provide a description.
#     """
#     try:
#         # Implement image processing here
#         description = "Image processing functionality is not yet implemented."
#         logging.info(f"Processed image from URL: {image_url}")
#         return description
#     except Exception as e:
#         logging.error(f"Error processing image: {str(e)}")
#         return f"An error occurred while processing the image: {str(e)}"

# Main interaction loop
def main():
    print("Welcome to the Enhanced ChatGPT System. Type 'exit', 'quit', or 'bye' to end the conversation.")
    print("Available Commands:")
    print("  /edit <index> <new_content>                        - Edit a message at a specific index.")
    print("  /remove <index>                                    - Remove a message at a specific index.")
    print("  /save <category> <filename> <content>              - Save content to a scratch pad file within a category.")
    print("  /edit_scratch <category> <filename> <new_content>  - Edit an existing scratch pad file within a category.")
    print("  /list_scratch_pad [category] [page]                - List scratch pad files, optionally within a category and paginated.")
    print("  /view_scratch <category> <filename>                - View the content of a scratch pad file within a category.")
    print("  /delete_scratch <category> <filename>              - Delete a scratch pad file within a category.")
    print("  /search_scratch <query>                            - Search for a query string within all scratch pad files.")
    print("-" * 80)

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
        elif user_input.startswith("/save "):
            # Parse save command: /save <category> <filename> <content>
            parts = user_input.split(maxsplit=3)
            if len(parts) == 4:
                category = parts[1]
                filename = parts[2]
                content = parts[3]
                save_scratch_pad(category, filename, content)
                print(f"Content saved to scratch pad as '{category}/{filename}.txt'.")
            else:
                print("Invalid save command format. Use: /save <category> <filename> <content>")
            continue
        elif user_input.startswith("/edit_scratch "):
            # Parse edit scratch pad command: /edit_scratch <category> <filename> <new_content>
            parts = user_input.split(maxsplit=4)
            if len(parts) == 4:
                category = parts[1]
                filename = parts[2]
                new_content = parts[3]
                edit_scratch_pad(category, filename, new_content)
                print(f"Scratch pad '{category}/{filename}.txt' edited.")
            else:
                print("Invalid edit_scratch command format. Use: /edit_scratch <category> <filename> <new_content>")
            continue
        elif user_input.startswith("/list_scratch_pad"):
            # Parse list_scratch_pad command: /list_scratch_pad [category] [page]
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
            files = list_scratch_pad_files(category, page)
            print(files)
            continue
        elif user_input.startswith("/view_scratch "):
            # Parse view scratch pad command: /view_scratch <category> <filename>
            parts = user_input.split(maxsplit=2)
            if len(parts) == 3:
                category = parts[1]
                filename = parts[2]
                content = view_scratch_pad(category, filename)
                print(content)
            else:
                print("Invalid view_scratch command format. Use: /view_scratch <category> <filename>")
            continue
        elif user_input.startswith("/delete_scratch "):
            # Parse delete scratch pad command: /delete_scratch <category> <filename>
            parts = user_input.split(maxsplit=2)
            if len(parts) == 3:
                category = parts[1]
                filename = parts[2]
                result = delete_scratch_pad(category, filename)
                print(result)
            else:
                print("Invalid delete_scratch command format. Use: /delete_scratch <category> <filename>")
            continue
        elif user_input.startswith("/search_scratch "):
            # Parse search scratch pad command: /search_scratch <query>
            parts = user_input.split(maxsplit=1)
            if len(parts) == 2:
                query = parts[1]
                results = search_scratch_pad(query)
                print(results)
            else:
                print("Invalid search_scratch command format. Use: /search_scratch <query>")
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