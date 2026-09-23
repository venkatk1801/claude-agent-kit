"""Tool definitions and safe built-in tools.

Every tool is a small dataclass carrying its name, human description, JSON
input schema (for the model's ``tools`` parameter), and a plain Python
callable ``handler`` that receives the parsed JSON input dict and returns a
string result.

File tools are sandboxed: every resolved path must stay inside the tool's
working directory. Path traversal attempts raise ``PathTraversalError``.
"""

from __future__ import annotations

import ast
import math
import operator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


class PathTraversalError(ValueError):
    """Raised when a tool call tries to escape its sandbox directory."""


class ToolNotFoundError(KeyError):
    """Raised when the model requests a tool that was never registered."""


@dataclass
class Tool:
    """A callable tool exposed to the model."""

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], str]

    def to_api_dict(self) -> dict[str, Any]:
        """Serialize to the shape the Messages API expects in ``tools=[...]``."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    def call(self, tool_input: dict[str, Any]) -> str:
        """Invoke the handler with the model's parsed JSON input."""
        try:
            return str(self.handler(tool_input))
        except PathTraversalError:
            raise
        except Exception as exc:  # handlers report errors as strings, not crashes
            return f"Error: {exc}"


def _sandboxed(path_str: str, workdir: Path) -> Path:
    """Resolve *path_str* and guarantee it stays inside *workdir*."""
    workdir = workdir.resolve()
    candidate = (workdir / path_str).resolve()
    if candidate != workdir and workdir not in candidate.parents:
        raise PathTraversalError(
            f"Refusing path outside working directory: {path_str!r}"
        )
    return candidate


def make_read_file(workdir: Path | str) -> Tool:
    workdir = Path(workdir)

    def handler(args: dict[str, Any]) -> str:
        path = _sandboxed(args["path"], workdir)
        if not path.is_file():
            return f"Error: {args['path']!r} is not a file."
        max_chars = int(args.get("max_chars", 50_000))
        text = path.read_text(encoding="utf-8", errors="replace")
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n...[truncated at {max_chars} chars]"
        return text

    return Tool(
        name="read_file",
        description=(
            "Read a text file relative to the working directory and return "
            "its contents."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path of the file to read."},
                "max_chars": {
                    "type": "integer",
                    "description": "Maximum characters to return (default 50000).",
                },
            },
            "required": ["path"],
        },
        handler=handler,
    )


def make_write_file(workdir: Path | str) -> Tool:
    workdir = Path(workdir)

    def handler(args: dict[str, Any]) -> str:
        path = _sandboxed(args["path"], workdir)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(args["content"], encoding="utf-8")
        return f"Wrote {len(args['content'])} chars to {args['path']!r}."

    return Tool(
        name="write_file",
        description=(
            "Write text content to a file relative to the working directory. "
            "Creates parent directories as needed, overwrites existing files."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path of the file to write."},
                "content": {"type": "string", "description": "Text content to write."},
            },
            "required": ["path", "content"],
        },
        handler=handler,
    )


def make_list_dir(workdir: Path | str) -> Tool:
    workdir = Path(workdir)

    def handler(args: dict[str, Any]) -> str:
        path = _sandboxed(args.get("path", "."), workdir)
        if not path.is_dir():
            return f"Error: {args.get('path', '.')!r} is not a directory."
        entries = sorted(
            (p.name + ("/" if p.is_dir() else "")) for p in path.iterdir()
        )
        return "\n".join(entries) if entries else "(empty directory)"

    return Tool(
        name="list_dir",
        description="List files and directories relative to the working directory.",
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative directory path (default: the working directory).",
                }
            },
        },
        handler=handler,
    )


# --- Safe arithmetic evaluator -------------------------------------------

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_ALLOWED_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}

_ALLOWED_NAMES = {
    name: getattr(math, name)
    for name in (
        "pi", "e", "tau", "inf", "nan",
        "sqrt", "sin", "cos", "tan", "log", "log10", "exp",
        "floor", "ceil", "fabs", "factorial", "pow", "gcd",
    )
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return _ALLOWED_BINOPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return _ALLOWED_UNARYOPS[type(node.op)](_safe_eval(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id not in _ALLOWED_NAMES or node.keywords:
            raise ValueError(f"Function not allowed: {getattr(node.func, 'id', '?')}")
        fn = _ALLOWED_NAMES[node.func.id]
        return fn(*(_safe_eval(a) for a in node.args))
    if isinstance(node, ast.Name) and node.id in _ALLOWED_NAMES:
        value = _ALLOWED_NAMES[node.id]
        if isinstance(value, (int, float)):
            return value
    raise ValueError("Only arithmetic expressions are allowed.")


def calculator_tool() -> Tool:
    def handler(args: dict[str, Any]) -> str:
        expr = args["expression"]
        try:
            result = _safe_eval(ast.parse(expr, mode="eval"))
        except Exception as exc:
            return f"Error: invalid expression ({exc})"
        return str(result)

    return Tool(
        name="calculator",
        description=(
            "Evaluate an arithmetic expression and return the result. Supports "
            "+, -, *, /, //, %, **, parentheses, and math functions like "
            "sqrt, sin, log, pi."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Arithmetic expression to evaluate, e.g. 'sqrt(16) + 2**3'.",
                }
            },
            "required": ["expression"],
        },
        handler=handler,
    )


def builtin_tools(workdir: Path | str) -> list[Tool]:
    """Return the four built-in tools, sandboxed to *workdir*."""
    return [
        make_read_file(workdir),
        make_write_file(workdir),
        make_list_dir(workdir),
        calculator_tool(),
    ]


# Convenience module-level aliases that sandbox to the current directory.
read_file = make_read_file(Path.cwd())
write_file = make_write_file(Path.cwd())
list_dir = make_list_dir(Path.cwd())
calculator = calculator_tool()
