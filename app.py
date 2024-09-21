from openai import OpenAI
import os
import json
from dotenv import load_dotenv
import logging
from datetime import datetime
from collections import deque

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Create log directories if they don't exist
log_dir = "logs"
o1_responses_dir = "o1_responses"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(o1_responses_dir, exist_ok=True)

# Set up logging
log_filename = os.path.join(log_dir, f"chat_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(filename=log_filename, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class ChatHistory:
    def __init__(self, max_messages=50):
        self.messages = deque(maxlen=max_messages)
        self.system_message = {"role": "system", "content": "You are an AI assistant. For complex math/logic/reasoning/coding questions, always use the 'o1_research' function as this can provide a more accurate response. Ensure your final response incorporates the o1 research results."}

    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})

    def get_messages(self):
        return [self.system_message] + list(self.messages)

    def edit_message(self, index, new_content):
        if 0 <= index < len(self.messages):
            self.messages[index]["content"] = new_content

    def remove_message(self, index):
        if 0 <= index < len(self.messages):
            del self.messages[index]

chat_history = ChatHistory()

def gpt4o_chat(user_input):
    logging.info(f"User input: {user_input}")
    chat_history.add_message("user", user_input)
    print("Calling GPT-4o for initial response...")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=chat_history.get_messages(),
        functions=[
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
            }
        ],
        function_call="auto"
    )

    message = response.choices[0].message

    if message.content:
        print(f"Initial response: {message.content[:50]}...")
    else:
        print("Model chose to use a function call instead of responding directly.")

    if message.function_call:
        print(f"Function call: {message.function_call.name}")
        # ... rest of the function call handling ...
    else:
        logging.info("No function call detected, returning initial response")
        print("No function call detected, returning initial response")
        chat_history.add_message("assistant", message.content)
        return message.content

    if message.function_call and message.function_call.name == "o1_research":
        logging.info("Function call detected: o1_research")
        function_args = json.loads(message.function_call.arguments)
        logging.info(f"Performing o1 research with query: {function_args['query']}")
        print(f"Performing o1 research with query: {function_args['query']}")
        research_result = o1_research(function_args['query'])
        
        logging.info("Calling GPT-4o for final response with o1 research results...")
        print("Calling GPT-4o for final response with o1 research results...")
        final_response = client.chat.completions.create(
            model="gpt-4o",
            messages=chat_history.get_messages() + [
                {"role": "function", "name": "o1_research", "content": research_result}
            ]
        )
        final_message = final_response.choices[0].message
        print(f"Final response type: {final_message.content[:50]}...")
        chat_history.add_message("assistant", final_message.content)
        return final_message.content
    elif message.function_call and message.function_call.name == "librarian":
        logging.info("Function call detected: librarian")
        function_args = json.loads(message.function_call.arguments)
        logging.info(f"Performing librarian search with query: {function_args['query']}")
        print(f"Performing librarian search with query: {function_args['query']}")
        librarian_result = librarian(function_args['query'])
        
        logging.info("Calling GPT-4o for final response with librarian results...")
        print("Calling GPT-4o for final response with librarian results...")
        final_response = client.chat.completions.create(
            model="gpt-4o",
            messages=chat_history.get_messages() + [
                {"role": "function", "name": "librarian", "content": librarian_result}
            ]
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
        for message in messages:
            if message.role == "assistant":
                return message.content[0].text.value

    except Exception as e:
        logging.error(f"Error in librarian function: {str(e)}")
        return f"An error occurred while retrieving information: {str(e)}"

# Main interaction loop
while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit', 'bye']:
        logging.info("User ended the conversation")
        break
    elif user_input.startswith("/edit"):
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
        continue
    elif user_input.startswith("/remove"):
        # Parse remove command: /remove <index>
        parts = user_input.split()
        if len(parts) == 2:
            try:
                index = int(parts[1])
                chat_history.remove_message(index)
                print(f"Message at index {index} removed.")
            except ValueError:
                print("Invalid remove command. Use: /remove <index>")
        continue
    
    logging.info("Processing user input...")
    response = gpt4o_chat(user_input)
    logging.info(f"AI response: {response}")
    print("\nAI:", response)
    print("\n" + "-"*50 + "\n")

logging.info("Conversation ended")