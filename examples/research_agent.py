"""Research agent demo.

Asks Claude a question and lets it use the built-in tools (file access +
calculator) to answer. Requires ``ANTHROPIC_API_KEY`` in the environment
(or a ``.env`` file in the project root).

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/research_agent.py "How many Python files are in this project?"
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import anthropic  # noqa: E402

from agent_kit import (  # noqa: E402
    builtin_tools,
    cache_report,
    cacheable,
    run_agent,
)

# Mark the system prompt cacheable: it is identical on every turn, so
# Anthropic serves it from the prompt cache instead of re-processing it.
SYSTEM = cacheable(
    "You are a precise research assistant. Use the tools available to gather "
    "facts before answering. When you have enough information, answer the "
    "user's question concisely and cite what you found."
)


def main() -> None:
    load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Error: set ANTHROPIC_API_KEY in your environment or a .env file.")

    question = " ".join(sys.argv[1:]) or (
        "List the files in the working directory, then tell me how many "
        "Python source files there are. Use the calculator for any arithmetic."
    )

    workdir = Path(__file__).resolve().parent.parent
    tools = builtin_tools(workdir)
    client = anthropic.Anthropic()

    state = {"result": None}

    def on_event(event: str, payload: dict) -> None:
        if event == "tool_call":
            print(f"  🔧 {payload['name']}({payload['input']})")

    print(f"❓ {question}\n")
    result = run_agent(
        client,
        prompt=question,
        system=SYSTEM,
        tools=tools,
        cache_tools=True,  # also cache the tool definitions
        max_iterations=10,
        on_event=on_event,
    )
    state["result"] = result

    print(f"\n💬 {result.final_text}")
    print(
        f"\n📊 {result.iterations} iteration(s), {result.tool_calls} tool call(s) | "
        f"tokens in={result.usage.input_tokens} out={result.usage.output_tokens} | "
        f"est. cost ${result.estimated_cost_usd:.4f}"
    )
    print(f"   cache: {cache_report([result.usage])}")


if __name__ == "__main__":
    main()
