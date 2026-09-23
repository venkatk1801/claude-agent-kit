"""Public API for claude-agent-kit."""

from agent_kit.caching import cacheable, cache_report
from agent_kit.loop import (
    IterationLimitError,
    RunResult,
    TokenUsage,
    estimate_cost,
    run_agent,
)
from agent_kit.tools import (
    Tool,
    builtin_tools,
    calculator,
    list_dir,
    read_file,
    write_file,
)

__all__ = [
    "Tool",
    "RunResult",
    "TokenUsage",
    "IterationLimitError",
    "run_agent",
    "estimate_cost",
    "cacheable",
    "cache_report",
    "builtin_tools",
    "read_file",
    "write_file",
    "list_dir",
    "calculator",
]

__version__ = "0.1.0"
