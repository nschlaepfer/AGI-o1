# Amarillo

Advanced AI assistant powered by GPT-5.2-Codex with 400K context, 128K output, and persistent memory.

![Architecture](AGI-o1.png)

## Quick Start

```bash
git clone https://github.com/nschlaepfer/amarillo.git && cd amarillo
pip install -e .
cp .env.example .env  # Add your OPENAI_API_KEY
amarillo
```

Or run directly:
```bash
python -m amarillo.main
```

**Requirements:** Python 3.10+, OpenAI API key

## Features

| Feature | Description |
|---------|-------------|
| GPT-5.2-Codex | 56.4% SWE-Bench Pro, 64.0% Terminal-Bench 2.0 |
| 400K Context | Handle massive codebases |
| ReasoningBank | Persistent memory across sessions |
| Multi-level Reasoning | none/low/medium/high/xhigh effort levels |
| Function Calling | OpenAI tools API integration |
| Scratch Pad | Organized note storage system |

## Installation

### Development Install
```bash
pip install -e ".[dev]"
```

### Production Install
```bash
pip install .
```

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

- **deep_reasoning** - Deep STEM reasoning with configurable effort
- **retrieve_knowledge** - Information retrieval
- **coder** - Code generation with GPT-5.2-Codex
- **Scratch pad operations** - save, edit, list, view, delete, search

## Fluid Intelligence Modules

Inspired by [Poetiq's ARC-AGI solver](https://poetiq.ai) and [Emdash](https://emdash.sh), these modules enable adaptive reasoning:

### FluidReasoning
Iterative refinement with feedback accumulation:
- Generate hypothesis, evaluate, build feedback, iterate
- Solution pool with probabilistic selection
- Multi-expert voting for ensemble reasoning
- Early termination on success

```python
from amarillo.reasoning import FluidReasoner, ExpertConfig

reasoner = FluidReasoner(client, evaluate_fn, config)
result = await reasoner.solve(task)
# Result includes: final_output, solutions, best_score, iteration_count
```

### WorkspaceManager
Session state management with persistence:
- Registry pattern (getOrCreate)
- Automatic snapshot/restore
- Activity tracking and event listeners
- Workspace lifecycle (attach/detach/dispose)

```python
from amarillo.memory import WorkspaceManager, get_workspace_manager

manager = get_workspace_manager()
task = manager.create_task("Implement sorting")
task.add_reasoning_step("analysis", "Breaking down the problem...")
```

### SoftScoring
Partial credit evaluation system:
- Multi-dimensional scoring criteria
- Weighted aggregation
- Actionable feedback generation
- Pre-built scorers (syntax, keywords, execution)

```python
from amarillo.reasoning import SoftScorer, code_syntax_scorer

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
amarillo/
├── src/amarillo/           # Main package
│   ├── __init__.py         # Package exports
│   ├── main.py             # CLI entry point
│   ├── core/               # Configuration & utilities
│   │   ├── config.py       # Environment & settings
│   │   ├── constants.py    # Magic numbers & defaults
│   │   └── utils.py        # Helper functions
│   ├── memory/             # State management
│   │   ├── chat_history.py # Conversation tracking
│   │   ├── reasoning_bank.py # Persistent memory
│   │   ├── notes.py        # Notes system
│   │   └── workspace_manager.py # Session state
│   ├── reasoning/          # Intelligence modules
│   │   ├── fluid_reasoning.py # Iterative refinement
│   │   ├── ensemble_reasoning.py # Multi-expert voting
│   │   └── scoring.py      # Soft scoring system
│   └── tools/              # Tool definitions
│       ├── tools.py        # Function schemas
│       └── file_tools.py   # File operations
├── tests/                  # Test suite
├── examples/               # Demo scripts
├── docs/                   # Documentation & data
├── scripts/                # Utility scripts
├── pyproject.toml          # Package configuration
├── requirements.txt        # Dependencies
└── .env.example            # Environment template
```

## Development

### Running Tests
```bash
pytest
```

### Code Formatting
```bash
black src/ tests/
ruff check src/ tests/
```

## License

MIT - [@nschlaepfer](https://github.com/nschlaepfer)
