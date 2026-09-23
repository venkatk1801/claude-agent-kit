"""The agentic tool-use loop.

``run_agent()`` drives a model through repeated turns: send the conversation
to the Messages API, execute every ``tool_use`` block the model requests
(parallel calls in one turn included), feed the ``tool_result`` blocks back,
and stop when the model answers with text only — or raise
``IterationLimitError`` if it keeps reaching for tools past ``max_iterations``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from agent_kit.caching import cacheable
from agent_kit.tools import Tool, ToolNotFoundError


class IterationLimitError(RuntimeError):
    """Raised when the agent still wants tools after ``max_iterations`` turns."""


#: Per-million-token prices in USD. ``cache_write`` = 1.25x input price per
#: Anthropic's pricing model; cache reads cost 10% of the input price.
MODEL_PRICES: dict[str, dict[str, float]] = {
    "claude-sonnet-4-5": {
        "input": 3.0,
        "output": 15.0,
        "cache_write": 3.75,
        "cache_read": 0.30,
    },
    "claude-opus-4-1": {
        "input": 15.0,
        "output": 75.0,
        "cache_write": 18.75,
        "cache_read": 1.50,
    },
}

DEFAULT_MODEL = "claude-sonnet-4-5"


@dataclass
class TokenUsage:
    """Accumulated token usage across a run."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    def add(self, usage: Any) -> None:
        """Fold one API response's usage into the running total."""
        self.input_tokens += getattr(usage, "input_tokens", 0) or 0
        self.output_tokens += getattr(usage, "output_tokens", 0) or 0
        self.cache_creation_input_tokens += (
            getattr(usage, "cache_creation_input_tokens", 0) or 0
        )
        self.cache_read_input_tokens += (
            getattr(usage, "cache_read_input_tokens", 0) or 0
        )


def estimate_cost(usage: TokenUsage, model: str = DEFAULT_MODEL) -> float:
    """Estimated USD cost for accumulated usage on *model*."""
    prices = MODEL_PRICES[model]
    return (
        usage.input_tokens / 1e6 * prices["input"]
        + usage.output_tokens / 1e6 * prices["output"]
        + usage.cache_creation_input_tokens / 1e6 * prices["cache_write"]
        + usage.cache_read_input_tokens / 1e6 * prices["cache_read"]
    )


@dataclass
class RunResult:
    """Everything a run produced."""

    final_text: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    iterations: int = 0
    tool_calls: int = 0
    usage: TokenUsage = field(default_factory=TokenUsage)
    model: str = DEFAULT_MODEL

    @property
    def estimated_cost_usd(self) -> float:
        return estimate_cost(self.usage, self.model)


def _extract_text(content: list[Any]) -> str:
    parts = []
    for block in content:
        if isinstance(block, dict):
            kind, text = block.get("type"), block.get("text")
        else:
            kind, text = getattr(block, "type", None), getattr(block, "text", None)
        if kind == "text":
            parts.append(text or "")
    return "".join(parts)


def run_agent(
    client: Any,
    prompt: str,
    system: Any = "You are a helpful AI assistant.",
    tools: list[Tool] | None = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 4096,
    max_iterations: int = 10,
    on_event: Callable[[str, dict[str, Any]], None] | None = None,
    cache_tools: bool = False,
) -> RunResult:
    """Run an agentic tool-use loop until the model answers with text only.

    Args:
        client: An ``anthropic.Anthropic`` client (or a compatible mock).
        prompt: The user's request.
        system: System prompt — a string, or a list of blocks (e.g. from
            :func:`agent_kit.cacheable`) for prompt caching.
        tools: List of :class:`agent_kit.Tool` the model may call.
        model: Model id; must exist in the price table for cost estimates.
        max_tokens: Max output tokens per API call.
        max_iterations: Hard cap on API round-trips. Exceeding it raises
            :class:`IterationLimitError`.
        on_event: Optional callback ``(event_name, payload)`` receiving
            ``"message"``, ``"tool_call"``, ``"tool_result"``, and ``"done"``
            events (useful for streaming-style progress UIs).
        cache_tools: Add the prompt-caching marker to the serialized tool
            definitions so repeated turns reuse the cached prefix.

    Returns:
        A :class:`RunResult` with the final text, full message history,
        iteration/tool counts, and accumulated usage + cost.
    """
    if model not in MODEL_PRICES:
        raise ValueError(f"No price table entry for model {model!r}.")
    tools = tools or []
    tool_map = {t.name: t for t in tools}
    api_tools = [t.to_api_dict() for t in tools]
    if cache_tools:
        api_tools = [{**d, **cacheable()} for d in api_tools]

    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    usage = TokenUsage()
    total_tool_calls = 0

    def emit(event: str, payload: dict[str, Any]) -> None:
        if on_event:
            on_event(event, payload)

    for iteration in range(1, max_iterations + 1):
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
            tools=api_tools or None,
        )
        usage.add(response.usage)
        emit("message", {"iteration": iteration, "response": response})

        tool_uses = [b for b in response.content if getattr(b, "type", None) == "tool_use"]
        assistant_text = _extract_text(response.content)

        # Append the assistant's turn verbatim — tool_use blocks included.
        messages.append({"role": "assistant", "content": response.content})

        if not tool_uses:
            emit("done", {"final_text": assistant_text})
            return RunResult(
                final_text=assistant_text,
                messages=messages,
                iterations=iteration,
                tool_calls=total_tool_calls,
                usage=usage,
                model=model,
            )

        # Execute every requested tool (parallel calls in one turn included)
        # and return all results in a single user turn.
        results: list[dict[str, Any]] = []
        for block in tool_uses:
            total_tool_calls += 1
            tool = tool_map.get(block.name)
            emit("tool_call", {"name": block.name, "input": block.input})
            if tool is None:
                result_text = f"Error: unknown tool {block.name!r}."
            else:
                try:
                    result_text = tool.call(block.input or {})
                except Exception as exc:  # never let one tool crash the loop
                    result_text = f"Error: {exc}"
            emit("tool_result", {"name": block.name, "result": result_text})
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_text,
                }
            )
        messages.append({"role": "user", "content": results})

    raise IterationLimitError(
        f"Agent did not finish within {max_iterations} iterations."
    )
