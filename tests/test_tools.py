"""Tests for the built-in tools: sandboxing and the safe calculator."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agent_kit.tools import (  # noqa: E402
    PathTraversalError,
    builtin_tools,
    calculator_tool,
    make_list_dir,
    make_read_file,
    make_write_file,
)


@pytest.fixture()
def workdir(tmp_path):
    (tmp_path / "notes.txt").write_text("hello world")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "inner.txt").write_text("inner")
    return tmp_path


def test_read_file(workdir):
    tool = make_read_file(workdir)
    assert tool.call({"path": "notes.txt"}) == "hello world"
    assert tool.call({"path": "sub/inner.txt"}) == "inner"


def test_write_file_roundtrip(workdir):
    tool = make_write_file(workdir)
    assert "Wrote" in tool.call({"path": "new/deep.txt", "content": "data"})
    assert (workdir / "new" / "deep.txt").read_text() == "data"


def test_list_dir(workdir):
    tool = make_list_dir(workdir)
    listing = tool.call({"path": "."})
    assert "notes.txt" in listing
    assert "sub/" in listing


@pytest.mark.parametrize("bad", ["../evil.txt", "../../etc/passwd", "sub/../../x"])
def test_read_file_rejects_traversal(workdir, bad):
    with pytest.raises(PathTraversalError):
        make_read_file(workdir).call({"path": bad})


@pytest.mark.parametrize("bad", ["../evil.txt", "a/../../b.txt"])
def test_write_file_rejects_traversal(workdir, bad):
    with pytest.raises(PathTraversalError):
        make_write_file(workdir).call({"path": bad, "content": "x"})
    assert not (workdir.parent / "evil.txt").exists()


def test_list_dir_rejects_traversal(workdir):
    with pytest.raises(PathTraversalError):
        make_list_dir(workdir).call({"path": ".."})


def test_missing_file_is_error_string_not_crash(workdir):
    assert "Error" in make_read_file(workdir).call({"path": "nope.txt"})


@pytest.mark.parametrize(
    "expr, expected",
    [
        ("2 + 3 * 4", "14"),
        ("(10 - 4) / 3", "2.0"),
        ("2 ** 10", "1024"),
        ("sqrt(16) + 1", "5.0"),
        ("pi * 2", str(2 * 3.141592653589793)),
    ],
)
def test_calculator_arithmetic(expr, expected):
    assert calculator_tool().call({"expression": expr}) == expected


@pytest.mark.parametrize(
    "evil",
    [
        "__import__('os').system('echo pwned')",
        "open('/etc/passwd').read()",
        "[x for x in range(3)]",
        "lambda: 1",
        "1; import os",
    ],
)
def test_calculator_rejects_code_injection(evil):
    result = calculator_tool().call({"expression": evil})
    assert result.startswith("Error")


def test_builtin_tools_registers_four(workdir):
    tools = builtin_tools(workdir)
    assert {t.name for t in tools} == {"read_file", "write_file", "list_dir", "calculator"}
    for t in tools:
        d = t.to_api_dict()
        assert set(d) == {"name", "description", "input_schema"}
