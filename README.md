# claude-agent-kit

A minimal, production-quality **agentic tool-use loop** built from scratch on the
[Anthropic Messages API](https://docs.anthropic.com/en/docs/build-with-claude).
No frameworks, no magic — just the loop every agent runtime (Claude Code
included) is built on, in ~200 lines of readable Python.

Give it a prompt and some tools; it calls the model, executes every
`tool_use` block the model requests (parallel calls included), feeds the
results back, and stops when the model answers with text — or raises
`IterationLimitError` if it loops past the iteration guard.

## Install

```bash
pip install claude-agent-kit
# or from source:
git clone https://github.com/venkatk1801/claude-agent-kit
cd claude-agent-kit && pip install -e ".[dev]"
```

Set your API key:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

## Quickstart

```python
import anthropic
from agent_kit import builtin_tools, run_agent

client = anthropic.Anthropic()
tools = builtin_tools("/path/to/sandbox")   # read_file, write_file, list_dir, calculator

result = run_agent(
    client,
    prompt="List the files here and summarize what this project does.",
    tools=tools,
    max_iterations=10,
    on_event=lambda event, payload: print(event, payload.get("name", "")),
)

print(result.final_text)
print(f"{result.iterations} iterations, {result.tool_calls} tool calls")
print(f"estimated cost: ${result.estimated_cost_usd:.4f}")
```

Custom tools are just a dataclass:

```python
from agent_kit import Tool

weather = Tool(
    name="get_weather",
    description="Current weather for a city.",
    input_schema={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
    handler=lambda args: f"Sunny, 72°F in {args['city']}",
)

result = run_agent(client, "What's the weather in Austin?", tools=[weather])
```

## Architecture

```
                    ┌─────────────────────────────────────┐
                    │            run_agent()              │
                    │                                     │
  user prompt ──►   │  ┌───────────────────────────────┐  │
                    │  │ 1. POST /v1/messages          │  │
                    │  │    (system, history, tools)   │  │
                    │  └──────────────┬────────────────┘  │
                    │                 │ response.content  │
                    │        ┌────────┴────────┐          │
                    │        │ text only?      │          │
                    │        │  yes → RETURN   │          │
                    │        │  no  ↓          │          │
                    │  ┌─────┴───────────────┐ │          │
                    │  │ 2. execute EVERY    │ │          │
                    │  │    tool_use block   │ │          │
                    │  │    (parallel-safe)  │ │          │
                    │  └─────┬───────────────┘ │          │
                    │        │ tool_result × N │          │
                    │  ┌─────┴───────────────┐ │          │
                    │  │ 3. append as ONE    │ │          │
                    │  │    user turn, loop  │◄┘          │
                    │  └─────────────────────┘           │
                    │                                     │
                    │ guard: max_iterations →             │
                    │        IterationLimitError          │
                    └─────────────────────────────────────┘
```

Key design decisions:

- **Verbatim history** — the assistant turn is appended exactly as the API
  returned it (including `tool_use` blocks), and all `tool_result` blocks go
  back as a *single* user turn. That's the contract the API expects.
- **Parallel tool calls** — one turn can contain N `tool_use` blocks; all are
  executed and all N results are returned together. One broken tool reports an
  `"Error: ..."` result instead of crashing the loop.
- **Hard iteration guard** — `max_iterations` (default 10) bounds API spend;
  exceeding it raises `IterationLimitError` rather than looping forever.

## Cost tracking

Every run accumulates real token usage and estimates spend from a per-model
price table (`MODEL_PRICES` in `agent_kit/loop.py`):

```python
result = run_agent(client, "2 + 2?", tools=[calculator])

u = result.usage
print(u.input_tokens, u.output_tokens)          # raw totals
print(u.cache_read_input_tokens)                # served from prompt cache
print(f"${result.estimated_cost_usd:.4f}")       # USD estimate

# or compute manually for any accumulated usage:
from agent_kit import estimate_cost
estimate_cost(u, model="claude-opus-4-1")
```

Supported models: `claude-sonnet-4-5` ($3/$15 per MTok in/out),
`claude-opus-4-1` ($15/$75). Unknown models raise `ValueError` so a missing
price entry can never silently under-report cost.

## Prompt caching

Agentic loops re-send the system prompt and tool definitions on *every* turn —
exactly the stable prefix prompt caching is designed for. Mark the system
prompt and tool definitions cacheable:

```python
from agent_kit import cacheable, cache_report, run_agent

SYSTEM = cacheable("You are a precise research assistant...")  # → cached blocks

result = run_agent(
    client,
    prompt="...",
    system=SYSTEM,
    tools=tools,
    cache_tools=True,      # also marks the serialized tool definitions
)

print(cache_report([result.usage]))
# {'cache_read_input_tokens': 12540, 'cache_creation_input_tokens': 13020, ...}
```

Cache reads are billed at 10% of the input price, so on a 10-turn run the
savings are substantial. See `examples/research_agent.py` for a runnable demo.

## Built-in tools

| Tool | Description |
| ---- | ----------- |
| `read_file` | Read a text file (with truncation limit) |
| `write_file` | Write text, creating parent dirs as needed |
| `list_dir` | List a directory's entries |
| `calculator` | Safe arithmetic evaluator (AST-based, no `eval`) |

File tools are **sandboxed** to a working directory you choose — `../` escapes
raise `PathTraversalError`. The calculator parses with `ast` and only allows
numbers, arithmetic operators, and a whitelist of `math` functions.

## Tests

```bash
pip install -e ".[dev]"
pytest          # 29 tests, all mocked — no API key, no network
```

## License

MIT © 2026 VenkataRamana Kandi
