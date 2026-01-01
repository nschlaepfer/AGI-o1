# AGI-o1

Advanced AI assistant powered by GPT-5.2-Codex with 400K context, 128K output, and persistent memory.

![Architecture](AGI-o1.png)

## Quick Start

```bash
git clone https://github.com/nschlaepfer/AGI-o1.git && cd AGI-o1
pip install openai python-dotenv
cp .env.example .env  # Add your OPENAI_API_KEY
python agi_o1.py
```

**Requirements:** Python 3.10+, OpenAI API key, ChatGPT Plus/Pro/Business/Edu/Enterprise subscription

## Features

| Feature | Description |
|---------|-------------|
| GPT-5.2-Codex | 56.4% SWE-Bench Pro, 64.0% Terminal-Bench 2.0 |
| 400K Context | Handle massive codebases |
| ReasoningBank | Persistent memory across sessions |
| Multi-level Reasoning | none/low/medium/high/xhigh effort levels |
| Function Calling | OpenAI tools API integration |
| Scratch Pad | Organized note storage system |

## Configuration

Edit `.env`:

```env
OPENAI_API_KEY_SANDBOX=sk-...
OPENAI_MODEL_CODE=gpt-5.2-codex
OPENAI_MODEL_REASONING_EFFORT=high
OPENAI_MODEL_COMPACTION=true
```

## Commands

| Command | Description |
|---------|-------------|
| `exit` / `quit` | End session |
| `/save <cat> <file> <content>` | Save to scratch pad |
| `/view_scratch <cat> <file>` | View scratch pad file |
| `/list_scratch_pad [cat]` | List scratch pad files |
| `/edit_scratch <cat> <file> <content>` | Edit scratch pad file |
| `/delete_scratch <cat> <file>` | Delete scratch pad file |
| `/search_scratch <query>` | Search scratch pad |
| `/edit <idx> <content>` | Edit chat message |
| `/remove <idx>` | Remove chat message |

## Built-in Functions

- **o1_research** - Deep STEM reasoning
- **librarian** - Information retrieval
- **get_current_weather** - Weather lookup
- **Scratch pad operations** - save, edit, list, view, delete, search

## Project Structure

```
AGI-o1/
├── agi_o1.py          # Main script
├── logs/              # Chat logs
├── o1_responses/      # Deep reasoning outputs
├── scratch_pad/       # User notes
├── docs/              # paper.txt, reasoning_bank.json
└── insight_capsules/  # Generated summaries
```

## License

MIT - [@nschlaepfer](https://github.com/nschlaepfer)
