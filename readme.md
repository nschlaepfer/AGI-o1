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

## Fluid Intelligence Modules (New)

Inspired by [Poetiq's ARC-AGI solver](https://poetiq.ai) and [Emdash](https://emdash.sh), these modules enable adaptive reasoning:

### FluidReasoning (`fluid_reasoning.py`)
Iterative refinement with feedback accumulation:
- Generate hypothesis, evaluate, build feedback, iterate
- Solution pool with probabilistic selection
- Multi-expert voting for ensemble reasoning
- Early termination on success

```python
from fluid_reasoning import FluidReasoner, ExpertConfig

reasoner = FluidReasoner(client, evaluate_fn, config)
result = await reasoner.solve(task)
# Result includes: final_output, solutions, best_score, iteration_count
```

### WorkspaceManager (`workspace_manager.py`)
Session state management with persistence:
- Registry pattern (getOrCreate)
- Automatic snapshot/restore
- Activity tracking and event listeners
- Workspace lifecycle (attach/detach/dispose)

```python
from workspace_manager import create_workspace, get_or_create_workspace

workspace = create_workspace("Implement sorting", workspace_id="task_001")
workspace.add_message("user", "Please implement quicksort")
workspace.mark_complete("success")
```

### SoftScoring (`scoring.py`)
Partial credit evaluation system:
- Multi-dimensional scoring criteria
- Weighted aggregation
- Actionable feedback generation
- Pre-built scorers (syntax, keywords, execution)

```python
from scoring import SoftScorer, code_syntax_scorer

scorer = SoftScorer(success_threshold=0.8)
scorer.add_criterion("syntax", code_syntax_scorer, weight=1.0)
result = scorer.score(solution, context)
# Result includes: score, success, dimension_scores, feedback
```

### Demo
```bash
python examples/fluid_intelligence_demo.py
```

## Project Structure

```
AGI-o1/
├── AGI-o1.py            # Main script
├── fluid_reasoning.py   # Iterative refinement module
├── workspace_manager.py # Session state management
├── scoring.py           # Soft scoring system
├── reasoning_bank.py    # Persistent memory
├── chat_history.py      # Conversation management
├── tools.py             # Function schemas
├── notes.py             # Notes management
├── config.py            # Configuration
├── examples/            # Demo scripts
│   └── fluid_intelligence_demo.py
├── logs/                # Chat logs
├── o1_responses/        # Deep reasoning outputs
├── workspaces/          # Persistent workspace state
├── docs/                # paper.txt, reasoning_bank.json
└── insight_capsules/    # Generated summaries
```

## License

MIT - [@nschlaepfer](https://github.com/nschlaepfer)
