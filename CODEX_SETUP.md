# Codex SDK Setup Guide

This guide covers setting up OpenAI's Codex SDK and integrating GPT-5.2-Codex with AGI-o1.

## Prerequisites

- **ChatGPT Plus, Pro, Business, Edu, or Enterprise subscription** (includes Codex)
- **OpenAI API key** with credits for API access
- Python 3.10+
- Node.js 18+ (for Codex CLI/SDK)

## Quick Start

### 1. Python Setup (Recommended)

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment
cp .env.example .env
# Edit .env with your API keys
```

### 2. Configure Models

Edit your `.env` file:

```bash
# Use GPT-5.2 for reasoning (supports xhigh effort)
OPENAI_MODEL_REASONING=gpt-5.2
OPENAI_MODEL_REASONING_EFFORT=high  # or xhigh for hardest tasks

# Use GPT-5.2-Codex for coding (most advanced agentic model)
OPENAI_MODEL_CODE=gpt-5.2-codex

# Use GPT-5.1-Codex-Max for long-horizon tasks
OPENAI_MODEL_CODE_MAX=gpt-5.1-codex-max

# Enable compaction for 400K context
OPENAI_MODEL_COMPACTION=true
```

### 3. Run AGI-o1

```bash
python AGI-o1.py
```

## Available Models

| Model | Use Case | Context | Output |
|-------|----------|---------|--------|
| `gpt-5.2` | General reasoning, analysis | 400K | 128K |
| `gpt-5.2-codex` | Advanced agentic coding | 400K | 128K |
| `gpt-5.1-codex-max` | Long-horizon engineering | 400K | 128K |
| `gpt-5.1-codex-mini` | Cost-effective coding | 400K | 128K |

## Reasoning Effort Levels

| Level | Description | Use Case |
|-------|-------------|----------|
| `none` | No extended thinking | Simple queries |
| `low` | Quick reasoning | Basic tasks |
| `medium` | Balanced (default) | Daily driver |
| `high` | Extended thinking | Complex tasks |
| `xhigh` | Maximum reasoning | Hardest challenges |

## Codex CLI (Optional)

For terminal-based Codex:

```bash
# Install globally
npm i -g @openai/codex
# Or with Homebrew
brew install --cask codex

# Run
codex
```

## Codex SDK for Node.js (Advanced)

For programmatic control of Codex agents:

```bash
npm install @openai/codex-sdk
```

Example usage:

```javascript
import { CodexSDK } from '@openai/codex-sdk';

const codex = new CodexSDK({
  apiKey: process.env.OPENAI_API_KEY
});

const session = await codex.createSession({
  model: 'gpt-5.2-codex',
  tools: ['bash', 'computer']
});

const result = await session.send('Refactor the authentication module');
```

## Agents SDK + MCP Integration

Use Codex as an MCP server for the Agents SDK:

```bash
pip install openai-agents
```

```python
from agents import Agent
from agents.mcp import MCPServerStdio

# Create MCP server wrapping Codex CLI
mcp_server = MCPServerStdio(
    command=["codex", "--non-interactive"]
)

# Use with Agents SDK
agent = Agent(
    model="gpt-5.2",
    tools=[mcp_server]
)
```

## Pricing (API)

| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|-------|----------------------|------------------------|
| GPT-5.2 | $1.75 | $14.00 |
| GPT-5.2-Codex | $1.75 | $14.00 |
| GPT-5.1-Codex-Max | ~$1.50 | ~$12.00 |

## Resources

- [Codex Documentation](https://developers.openai.com/codex/)
- [Codex SDK Reference](https://developers.openai.com/codex/sdk/)
- [GPT-5.2 Guide](https://platform.openai.com/docs/guides/latest-model)
- [Agents SDK](https://developers.openai.com/codex/guides/agents-sdk/)
- [OpenAI Cookbook](https://cookbook.openai.com/examples/gpt-5/gpt-5-1-codex-max_prompting_guide)
