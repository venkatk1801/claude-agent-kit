"""Prompt caching helpers.

Anthropic's prompt caching lets you mark large, stable parts of a request —
the system prompt and the tool definitions — as cacheable. Cached prefixes are
then billed at a deep discount and skip re-processing on subsequent turns of
an agentic loop.

Usage::

    from agent_kit import cacheable, run_agent

    run_agent(
        client,
        prompt="...",
        system=cacheable("You are a helpful assistant."),
        tools=[{**t.to_api_dict(), **cacheable()} for t in tools],
    )

The ``cache_report()`` helper summarizes ``cache_read_input_tokens`` and
``cache_creation_input_tokens`` from accumulated usage so you can see what
caching saved you.
"""

from __future__ import annotations

from typing import Any, Iterable


def cacheable(text_or_block: str | dict[str, Any] | None = None, **kwargs: Any) -> Any:
    """Mark a block as cacheable by adding the Anthropic cache-control marker.

    - ``cacheable("...text...")`` → ``[{"type": "text", "text": ..., "cache_control": {"type": "ephemeral"}}]``
    - ``cacheable({"name": ..., "input_schema": ...})`` → the same dict with a
      ``cache_control`` key added (use for tool definitions).
    - ``cacheable()`` → ``{"cache_control": {"type": "ephemeral"}}`` so it can be
      splatted into an existing dict: ``{**tool_def, **cacheable()}``.
    """
    marker = {"cache_control": {"type": "ephemeral"}}
    if text_or_block is None:
        return {**marker, **kwargs}
    if isinstance(text_or_block, str):
        return [{"type": "text", "text": text_or_block, **marker}]
    if isinstance(text_or_block, dict):
        return {**text_or_block, **marker, **kwargs}
    raise TypeError(f"Cannot mark {type(text_or_block).__name__} as cacheable.")


def cache_report(usage_events: Iterable[Any]) -> dict[str, int]:
    """Summarize prompt-cache statistics across usage snapshots.

    Accepts anything exposing ``cache_read_input_tokens`` /
    ``cache_creation_input_tokens`` attributes (e.g. Anthropic usage objects
    or :class:`agent_kit.TokenUsage`).
    """
    reads = 0
    writes = 0
    for u in usage_events:
        reads += getattr(u, "cache_read_input_tokens", 0) or 0
        writes += getattr(u, "cache_creation_input_tokens", 0) or 0
    return {
        "cache_read_input_tokens": reads,
        "cache_creation_input_tokens": writes,
        "cache_hits": reads,  # alias: tokens served from cache
    }
