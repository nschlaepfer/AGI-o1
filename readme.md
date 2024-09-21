# AGI-o1

AGI-o1 is an experimental AI assistant designed to enhance complex reasoning capabilities by leveraging OpenAI's GPT-4o model with function calling features. It intelligently decides when to delegate tasks to specialized functions like `o1_research` for deep STEM queries or `librarian` for retrieval augmented generation (RAG) using OpenAI's Retrieval tool. This system aims to provide more accurate and contextually relevant responses, especially for complex math, logic, reasoning, and coding questions.

> **Note:** This project is a work in progress and not fully implemented yet. Contributions and feedback are welcome!

## Table of Contents

- [Features](#features)
- [How It Works](#how-it-works)
  - [Function Calling](#function-calling)
  - [Chat History Management](#chat-history-management)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the Assistant](#running-the-assistant)
  - [Editing and Removing Messages](#editing-and-removing-messages)
- [Code Structure](#code-structure)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Features

- **Intelligent Function Calling:** Automatically decides when to invoke specialized functions (`o1_research` or `librarian`) based on the user's query.
- **Enhanced STEM Capabilities:** Utilizes the `o1_research` function to perform complex reasoning for STEM-related questions.
- **Retrieval Augmented Generation (RAG):** Plans to integrate the `librarian` function to retrieve information using OpenAI's Retrieval tool.
- **Chat History Management:** Maintains a history of the conversation with options to edit or remove messages.
- **Logging and Debugging:** Comprehensive logging for tracking user inputs, AI responses, and function calls.
- **Extensible Design:** Modular functions and classes allow for easy expansion and integration of new features.

## How It Works

### Function Calling

AGI-o1 leverages OpenAI's function calling capabilities to extend the AI assistant's functionality. When the AI model determines that a user's query requires specialized handling, it can choose to call one of the following functions:

- **`o1_research` Function:**
  - **Purpose:** Performs deep research and complex reasoning for STEM-related queries.
  - **How It Works:** Sends the query to the `o1-preview` model and processes the response.
- **`librarian` Function:**
  - **Purpose:** Retrieves information using OpenAI's Retrieval tool (RAG).
  - **How It Works:** Creates an assistant with retrieval capabilities and fetches information relevant to the query.
  - **Status:** Not fully implemented yet.

### Chat History Management

The `ChatHistory` class manages the conversation history between the user and the assistant:

- **Message Storage:** Uses a deque to store up to 50 messages.
- **System Message:** Initializes with a system prompt guiding the assistant's behavior.
- **Editing Messages:** Supports editing messages by index.
- **Removing Messages:** Allows removal of messages by index.

## Installation

### Prerequisites

- Python 3.7 or higher
- An OpenAI API key with access to GPT-4o and other required models
- [pip](https://pip.pypa.io/en/stable/installation/) package manager
- [virtualenv](https://virtualenv.pypa.io/en/latest/installation.html) (optional but recommended)

### Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/nschlaepfer/AGI-o1.git
   cd AGI-o1
   ```

2. **Create a Virtual Environment (Optional but Recommended)**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**

   - Create a `.env` file in the project root directory.
   - Add your OpenAI API key to the `.env` file:

     ```bash
     OPENAI_API_KEY=your_openai_api_key_here
     ```

5. **Create Necessary Directories**

   The script will automatically create `logs` and `o1_responses` directories if they don't exist.

## Usage

### Running the Assistant

Execute the main script to start the AI assistant:

```bash
python agi_o1.py
```

### Interacting with the Assistant

- **Normal Queries:** Simply type your message and press Enter.

  ```plaintext
  You: How do I solve a quadratic equation?
  ```

- **Ending the Conversation:** Type `exit`, `quit`, or `bye`.

  ```plaintext
  You: exit
  ```

### Editing and Removing Messages

- **Edit a Message:**

  Use the `/edit` command followed by the message index and the new content.

  ```plaintext
  You: /edit 2 Can you explain the Pythagorean theorem?
  ```

- **Remove a Message:**

  Use the `/remove` command followed by the message index.

  ```plaintext
  You: /remove 3
  ```

> **Note:** The index refers to the position of the message in the conversation history.

## Code Structure

- **`agi_o1.py`:** The main script that runs the AI assistant.
- **Classes:**
  - **`ChatHistory`:** Manages the conversation history.
- **Functions:**
  - **`gpt4o_chat(user_input)`:** Handles user input and AI responses, including function calls.
  - **`o1_research(query)`:** Performs complex reasoning using the `o1-preview` model.
  - **`librarian(query)`:** (Planned) Retrieves information using OpenAI's Retrieval tool.
- **Directories:**
  - **`logs/`:** Stores log files of the conversation.
  - **`o1_responses/`:** Stores responses from the `o1_research` function.

## Future Improvements

- **Implement `librarian` Functionality:**
  - Complete the integration with OpenAI's Retrieval tool for RAG.
- **Error Handling:**
  - Improve exception handling, especially in the `librarian` function.
- **User Interface:**
  - Develop a graphical user interface (GUI) for a better user experience.
- **Extensibility:**
  - Add more specialized functions as needed.
- **Testing:**
  - Write unit tests for all functions and classes.
- **Documentation:**
  - Enhance code comments and add docstrings for all functions and classes.

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the Repository**

   Click on the 'Fork' button at the top right corner of the repository page.

2. **Clone Your Fork**

   ```bash
   git clone https://github.com/nschlaepfer/AGI-o1.git
   cd AGI-o1
   ```

3. **Create a New Branch**

   ```bash
   git checkout -b feature/your_feature_name
   ```

4. **Make Your Changes**

5. **Commit Your Changes**

   ```bash
   git commit -am 'Add some feature'
   ```

6. **Push to the Branch**

   ```bash
   git push origin feature/your_feature_name
   ```

7. **Create a Pull Request**

   Go to the original repository and click on 'New Pull Request'.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

- **OpenAI:** For providing the GPT-4o model and API.
- **Contributors:** Thank you to everyone who has contributed to this project.
- **Community:** Thanks to the AI and open-source communities for inspiration and support.

---

Feel free to reach out if you have any questions or need assistance getting started!