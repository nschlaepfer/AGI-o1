# Context editing

> Automatically manage conversation context as it grows with context editing.

<Note>
  Context editing is currently in beta with support for tool result clearing. To enable it, use the beta header `context-management-2025-06-27` in your API requests. Additional context editing strategies will be added in future releases.

  Please reach out through our [feedback form](https://forms.gle/YXC2EKGMhjN1c4L88) to share your feedback on this feature.
</Note>

## How it works

The `clear_tool_uses_20250919` strategy clears tool results when conversation context grows beyond your configured threshold. When activated, the API automatically clears the oldest tool results in chronological order, replacing them with placeholder text to let Claude know the tool result was removed. By default, only tool results are cleared. You can optionally clear both tool results and tool calls (the tool use parameters) by setting `clear_tool_inputs` to true.

Context editing invalidates cached prompt prefixes because clearing content modifies the prompt structure, breaking the match requirement for cache hits. To account for this, we recommend clearing enough tokens to make the cache invalidation worthwhile. Use the `clear_at_least` parameter to ensure a minimum number of tokens is cleared each time. When using [prompt caching](/en/docs/build-with-claude/prompt-caching) with context editing, you'll incur cache write costs each time content is cleared, but subsequent requests can reuse the newly cached prefix.

## Supported models

Context editing is available on:

* Claude Opus 4.1 (`claude-opus-4-1-20250805`)
* Claude Opus 4 (`claude-opus-4-20250514`)
* Claude Sonnet 4.5 (`claude-sonnet-4-5-20250929`)
* Claude Sonnet 4 (`claude-sonnet-4-20250514`)

## Basic usage

The simplest way to enable context editing is to specify only the strategy type, as all other [configuration options](#configuration-options) will use their default values:

<CodeGroup>
  ```bash cURL
  curl https://api.anthropic.com/v1/messages \
      --header "x-api-key: $ANTHROPIC_API_KEY" \
      --header "anthropic-version: 2023-06-01" \
      --header "content-type: application/json" \
      --header "anthropic-beta: context-management-2025-06-27" \
      --data '{
          "model": "claude-sonnet-4-5",
          "max_tokens": 4096,
          "messages": [
              {
                  "role": "user",
                  "content": "Search for recent developments in AI"
              }
          ],
          "tools": [
              {
                  "type": "web_search_20250305",
                  "name": "web_search"
              }
          ],
          "context_management": {
              "edits": [
                  {"type": "clear_tool_uses_20250919"}
              ]
          }
      }'
  ```

  ```python Python
  response = client.beta.messages.create(
      model="claude-sonnet-4-5",
      max_tokens=4096,
      messages=[
          {
              "role": "user",
              "content": "Search for recent developments in AI"
          }
      ],
      tools=[
          {
              "type": "web_search_20250305",
              "name": "web_search"
          }
      ],
      betas=["context-management-2025-06-27"],
      context_management={
          "edits": [
              {"type": "clear_tool_uses_20250919"}
          ]
      }
  )
  ```

  ```typescript TypeScript
  import Anthropic from '@anthropic-ai/sdk';

  const anthropic = new Anthropic({
    apiKey: process.env.ANTHROPIC_API_KEY,
  });

  const response = await anthropic.beta.messages.create({
    model: "claude-sonnet-4-5",
    max_tokens: 4096,
    messages: [
      {
        role: "user",
        content: "Search for recent developments in AI"
      }
    ],
    tools: [
      {
        type: "web_search_20250305",
        name: "web_search"
      }
    ],
    context_management: {
      edits: [
        { type: "clear_tool_uses_20250919" }
      ]
    },
    betas: ["context-management-2025-06-27"]
  });
  ```
</CodeGroup>

## Advanced configuration

You can customize the context editing behavior with additional parameters:

<CodeGroup>
  ```bash cURL
  curl https://api.anthropic.com/v1/messages \
      --header "x-api-key: $ANTHROPIC_API_KEY" \
      --header "anthropic-version: 2023-06-01" \
      --header "content-type: application/json" \
      --header "anthropic-beta: context-management-2025-06-27" \
      --data '{
          "model": "claude-sonnet-4-5",
          "max_tokens": 4096,
          "messages": [
              {
                  "role": "user",
                  "content": "Create a simple command line calculator app using Python"
              }
          ],
          "tools": [
              {
                  "type": "text_editor_20250728",
                  "name": "str_replace_based_edit_tool",
                  "max_characters": 10000
              },
              {
                  "type": "web_search_20250305",
                  "name": "web_search",
                  "max_uses": 3
              }
          ],
          "context_management": {
              "edits": [
                  {
                      "type": "clear_tool_uses_20250919",
                      "trigger": {
                          "type": "input_tokens",
                          "value": 30000
                      },
                      "keep": {
                          "type": "tool_uses",
                          "value": 3
                      },
                      "clear_at_least": {
                          "type": "input_tokens",
                          "value": 5000
                      },
                      "exclude_tools": ["web_search"]
                  }
              ]
          }
      }'
  ```

  ```python Python
  response = client.beta.messages.create(
      model="claude-sonnet-4-5",
      max_tokens=4096,
      messages=[
          {
              "role": "user",
              "content": "Create a simple command line calculator app using Python"
          }
      ],
      tools=[
          {
              "type": "text_editor_20250728",
              "name": "str_replace_based_edit_tool",
              "max_characters": 10000
          },
          {
              "type": "web_search_20250305",
              "name": "web_search",
              "max_uses": 3
          }
      ],
      betas=["context-management-2025-06-27"],
      context_management={
          "edits": [
              {
                  "type": "clear_tool_uses_20250919",
                  # Trigger clearing when threshold is exceeded
                  "trigger": {
                      "type": "input_tokens",
                      "value": 30000
                  },
                  # Number of tool uses to keep after clearing
                  "keep": {
                      "type": "tool_uses",
                      "value": 3
                  },
                  # Optional: Clear at least this many tokens
                  "clear_at_least": {
                      "type": "input_tokens",
                      "value": 5000
                  },
                  # Exclude these tools from being cleared
                  "exclude_tools": ["web_search"]
              }
          ]
      }
  )
  ```

  ```typescript TypeScript
  import Anthropic from '@anthropic-ai/sdk';

  const anthropic = new Anthropic({
    apiKey: process.env.ANTHROPIC_API_KEY,
  });

  const response = await anthropic.beta.messages.create({
    model: "claude-sonnet-4-5",
    max_tokens: 4096,
    messages: [
      {
        role: "user",
        content: "Create a simple command line calculator app using Python"
      }
    ],
    tools: [
      {
        type: "text_editor_20250728",
        name: "str_replace_based_edit_tool",
        max_characters: 10000
      },
      {
        type: "web_search_20250305",
        name: "web_search",
        max_uses: 3
      }
    ],
    betas: ["context-management-2025-06-27"],
    context_management: {
      edits: [
        {
          type: "clear_tool_uses_20250919",
          // Trigger clearing when threshold is exceeded
          trigger: {
            type: "input_tokens",
            value: 30000
          },
          // Number of tool uses to keep after clearing
          keep: {
            type: "tool_uses",
            value: 3
          },
          // Optional: Clear at least this many tokens
          clear_at_least: {
            type: "input_tokens",
            value: 5000
          },
          // Exclude these tools from being cleared
          exclude_tools: ["web_search"]
        }
      ]
    }
  });
  ```
</CodeGroup>

## Configuration options

| Configuration option | Default              | Description                                                                                                                                                                                                                                           |
| -------------------- | -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `trigger`            | 100,000 input tokens | Defines when the context editing strategy activates. Once the prompt exceeds this threshold, clearing will begin. You can specify this value in either `input_tokens` or `tool_uses`.                                                                 |
| `keep`               | 3 tool uses          | Defines how many recent tool use/result pairs to keep after clearing occurs. The API removes the oldest tool interactions first, preserving the most recent ones.                                                                                     |
| `clear_at_least`     | None                 | Ensures a minimum number of tokens is cleared each time the strategy activates. If the API can't clear at least the specified amount, the strategy will not be applied. This helps determine if context clearing is worth breaking your prompt cache. |
| `exclude_tools`      | None                 | List of tool names whose tool uses and results should never be cleared. Useful for preserving important context.                                                                                                                                      |
| `clear_tool_inputs`  | `false`              | Controls whether the tool call parameters are cleared along with the tool results. By default, only the tool results are cleared while keeping Claude's original tool calls visible.                                                                  |

## Response format

You can see which context edits were applied to your request using the `context_management` response field, along with helpful statistics about the content and input tokens cleared.

```json Response
{
    "id": "msg_013Zva2CMHLNnXjNJJKqJ2EF",
    "type": "message",
    "role": "assistant",
    "content": [...],
    "usage": {...},
    "context_management": {
        "applied_edits": [
            {
                "type": "clear_tool_uses_20250919",
                "cleared_tool_uses": 8,
                "cleared_input_tokens": 50000
            }
        ]
    }
}
```

For streaming responses, the context edits will be included in the final `message_delta` event:

```json Streaming Response
{
    "type": "message_delta",
    "delta": {
        "stop_reason": "end_turn",
        "stop_sequence": null
    },
    "usage": {
        "output_tokens": 1024
    },
    "context_management": {
        "applied_edits": [...]
    }
}
```

## Token counting

The [/v1/messages/count\_tokens](/en/docs/build-with-claude/token-counting) endpoint supports context management, allowing you to preview how many tokens your prompt will use after context editing is applied.

<CodeGroup>
  ```bash cURL
  curl https://api.anthropic.com/v1/messages/count_tokens \
      --header "x-api-key: $ANTHROPIC_API_KEY" \
      --header "anthropic-version: 2023-06-01" \
      --header "content-type: application/json" \
      --header "anthropic-beta: context-management-2025-06-27" \
      --data '{
          "model": "claude-sonnet-4-5",
          "messages": [
              {
                  "role": "user",
                  "content": "Continue our conversation..."
              }
          ],
          "tools": [...],
          "context_management": {
              "edits": [
                  {
                      "type": "clear_tool_uses_20250919",
                      "trigger": {
                          "type": "input_tokens",
                          "value": 30000
                      },
                      "keep": {
                          "type": "tool_uses",
                          "value": 5
                      }
                  }
              ]
          }
      }'
  ```

  ```python Python
  response = client.beta.messages.count_tokens(
      model="claude-sonnet-4-5",
      messages=[
          {
              "role": "user",
              "content": "Continue our conversation..."
          }
      ],
      tools=[...],  # Your tool definitions
      betas=["context-management-2025-06-27"],
      context_management={
          "edits": [
              {
                  "type": "clear_tool_uses_20250919",
                  "trigger": {
                      "type": "input_tokens",
                      "value": 30000
                  },
                  "keep": {
                      "type": "tool_uses",
                      "value": 5
                  }
              }
          ]
      }
  )

  print(f"Original tokens: {response.context_management['original_input_tokens']}")
  print(f"After clearing: {response.input_tokens}")
  print(f"Savings: {response.context_management['original_input_tokens'] - response.input_tokens} tokens")
  ```

  ```typescript TypeScript
  import Anthropic from '@anthropic-ai/sdk';

  const anthropic = new Anthropic({
    apiKey: process.env.ANTHROPIC_API_KEY,
  });

  const response = await anthropic.beta.messages.countTokens({
    model: "claude-sonnet-4-5",
    messages: [
      {
        role: "user",
        content: "Continue our conversation..."
      }
    ],
    tools: [...],  // Your tool definitions
    betas: ["context-management-2025-06-27"],
    context_management: {
      edits: [
        {
          type: "clear_tool_uses_20250919",
          trigger: {
            type: "input_tokens",
            value: 30000
          },
          keep: {
            type: "tool_uses",
            value: 5
          }
        }
      ]
    }
  });

  console.log(`Original tokens: ${response.context_management?.original_input_tokens}`);
  console.log(`After clearing: ${response.input_tokens}`);
  console.log(`Savings: ${(response.context_management?.original_input_tokens || 0) - response.input_tokens} tokens`);
  ```
</CodeGroup>

```json Response
{
    "input_tokens": 25000,
    "context_management": {
        "original_input_tokens": 70000
    }
}
```

The response shows both the final token count after context management is applied (`input_tokens`) and the original token count before any clearing occurred (`original_input_tokens`).
