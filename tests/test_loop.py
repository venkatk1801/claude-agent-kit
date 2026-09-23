"""Tests for the agentic loop. The Anthropic client is fully mocked —
no network calls and no API key needed."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agent_kit import (  # noqa: E402
    IterationLimitError,
    TokenUsage,
    estimate_cost,
    run_agent,
)
from agent_kit.loop import MODEL_PRICES  # noqa: E402
from agent_kit.tools import Tool  # noqa: E402


def text_block(text: str):
    return SimpleNamespace(type="text", text=text)


def tool_block(tool_id: str, name: str, tool_input: dict):
    return SimpleNamespace(type="tool_use", id=tool_id, name=name, input=tool_input)


def usage(**kwargs):
    defaults = {
        "input_tokens": 100,
        "output_tokens": 50,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


class FakeMessages:
    """Returns scripted responses in order, like an actor following a script."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        if not self._responses:
            raise AssertionError("More API calls than scripted responses.")
        return self._responses.pop(0)


def make_message(content, **usage_kwargs):
    return SimpleNamespace(content=content, usage=usage(**usage_kwargs))


def echo_tool():
    return Tool(
        name="echo",
        description="Echo the input back.",
        input_schema={"type": "object", "properties": {"msg": {"type": "string"}}},
        handler=lambda args: f"echo:{args['msg']}",
    )


def failing_tool():
    def boom(_args):
        raise RuntimeError("kaboom")

    return Tool(
        name="boom",
        description="Always fails.",
        input_schema={"type": "object"},
        handler=boom,
    )


def test_multi_turn_tool_loop():
    """Turn 1 requests a tool, turn 2 answers with text. Usage accumulates."""
    client = SimpleNamespace(
        messages=FakeMessages(
            [
                make_message([tool_block("t1", "echo", {"msg": "hi"})]),
                make_message([text_block("The tool said: echo:hi")]),
            ]
        )
    )
    result = run_agent(client, "Say hi via the tool.", tools=[echo_tool()])
    assert result.final_text == "The tool said: echo:hi"
    assert result.iterations == 2
    assert result.tool_calls == 1
    assert result.usage.input_tokens == 200
    assert result.usage.output_tokens == 100
    # tool_result blocks were fed back in a single user turn
    last_user = result.messages[-2]
    assert last_user["role"] == "user"
    assert last_user["content"][0]["type"] == "tool_result"
    assert last_user["content"][0]["tool_use_id"] == "t1"
    assert "echo:hi" in last_user["content"][0]["content"]


def test_parallel_tool_calls_in_one_turn():
    """Two tool_use blocks in one assistant turn are both executed."""
    client = SimpleNamespace(
        messages=FakeMessages(
            [
                make_message(
                    [
                        tool_block("t1", "echo", {"msg": "a"}),
                        tool_block("t2", "echo", {"msg": "b"}),
                    ]
                ),
                make_message([text_block("done")]),
            ]
        )
    )
    seen = []
    result = run_agent(
        client,
        "Echo a and b.",
        tools=[echo_tool()],
        on_event=lambda e, p: seen.append(e),
    )
    assert result.final_text == "done"
    assert result.tool_calls == 2
    assert result.iterations == 2
    results_turn = result.messages[-2]
    ids = [b["tool_use_id"] for b in results_turn["content"]]
    assert ids == ["t1", "t2"]
    assert seen.count("tool_call") == 2
    assert seen.count("tool_result") == 2


def test_unknown_tool_returns_error_not_crash():
    client = SimpleNamespace(
        messages=FakeMessages(
            [
                make_message([tool_block("t9", "nope", {})]),
                make_message([text_block("recovered")]),
            ]
        )
    )
    result = run_agent(client, "go", tools=[echo_tool()])
    assert result.final_text == "recovered"
    assert "unknown tool" in result.messages[2]["content"][0]["content"]


def test_failing_tool_handler_does_not_crash_loop():
    client = SimpleNamespace(
        messages=FakeMessages(
            [
                make_message([tool_block("t1", "boom", {})]),
                make_message([text_block("survived")]),
            ]
        )
    )
    result = run_agent(client, "go", tools=[failing_tool()])
    assert result.final_text == "survived"
    assert "kaboom" in result.messages[2]["content"][0]["content"]


def test_iteration_limit_raises():
    """A model that never stops using tools hits the guard."""
    scripted = [make_message([tool_block(f"t{i}", "echo", {"msg": "x"})]) for i in range(20)]
    client = SimpleNamespace(messages=FakeMessages(scripted))
    with pytest.raises(IterationLimitError):
        run_agent(client, "loop forever", tools=[echo_tool()], max_iterations=3)
    assert len(client.messages.requests) == 3


def test_estimated_cost_math():
    usage_obj = TokenUsage(
        input_tokens=1_000_000,
        output_tokens=500_000,
        cache_creation_input_tokens=1_000_000,
        cache_read_input_tokens=1_000_000,
    )
    p = MODEL_PRICES["claude-sonnet-4-5"]
    expected = p["input"] + 0.5 * p["output"] + p["cache_write"] + p["cache_read"]
    assert estimate_cost(usage_obj, "claude-sonnet-4-5") == pytest.approx(expected)
    # opus is priced higher
    assert estimate_cost(usage_obj, "claude-opus-4-1") > expected


def test_unknown_model_rejected():
    client = SimpleNamespace(messages=FakeMessages([]))
    with pytest.raises(ValueError):
        run_agent(client, "hi", model="claude-nonexistent-9")


def test_system_and_tools_passed_through():
    client = SimpleNamespace(
        messages=FakeMessages([make_message([text_block("ok")])])
    )
    tools = [echo_tool()]
    run_agent(client, "hi", system="SYS", tools=tools, model="claude-sonnet-4-5")
    req = client.messages.requests[0]
    assert req["system"] == "SYS"
    assert req["model"] == "claude-sonnet-4-5"
    assert req["tools"][0]["name"] == "echo"
    assert "input_schema" in req["tools"][0]
