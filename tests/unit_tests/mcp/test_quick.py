"""Unit tests for the one-liner MCP server exposure (genai_tk.mcp.quick)."""

from unittest.mock import patch

from langchain_core.tools import tool
from mcp.server.fastmcp import FastMCP

from genai_tk.mcp.quick import build_server, expose

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def plain_add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def plain_greet(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {name}!"


@tool
def lc_multiply(x: int, y: int) -> int:
    """Multiply two numbers."""
    return x * y


# ---------------------------------------------------------------------------
# build_server tests
# ---------------------------------------------------------------------------


class TestBuildServer:
    def test_build_with_plain_functions(self):
        server = build_server("test", tools=[plain_add, plain_greet])
        assert isinstance(server, FastMCP)

    def test_build_with_langchain_tool(self):
        server = build_server("test", tools=[lc_multiply])
        assert isinstance(server, FastMCP)

    def test_build_with_mixed_tools(self):
        server = build_server("test", tools=[plain_add, lc_multiply])
        assert isinstance(server, FastMCP)

    def test_build_empty_tools(self):
        server = build_server("test", tools=[])
        assert isinstance(server, FastMCP)

    def test_build_with_description(self):
        server = build_server("test", tools=[plain_add], description="A test server")
        assert isinstance(server, FastMCP)


# ---------------------------------------------------------------------------
# expose tests (mock server.run to avoid blocking)
# ---------------------------------------------------------------------------


class TestExpose:
    @patch.object(FastMCP, "run")
    def test_expose_plain_function(self, mock_run):
        expose("test", tools=[plain_add])
        mock_run.assert_called_once_with(transport="stdio")

    @patch.object(FastMCP, "run")
    def test_expose_langchain_tool(self, mock_run):
        expose("test", tools=[lc_multiply])
        mock_run.assert_called_once_with(transport="stdio")

    @patch.object(FastMCP, "run")
    def test_expose_mixed(self, mock_run):
        expose("test", tools=[plain_add, lc_multiply, plain_greet])
        mock_run.assert_called_once_with(transport="stdio")

    @patch.object(FastMCP, "run")
    def test_expose_custom_transport(self, mock_run):
        expose("test", tools=[plain_add], transport="sse")
        mock_run.assert_called_once_with(transport="sse")

    @patch.object(FastMCP, "run")
    def test_expose_skips_non_callable(self, mock_run):
        expose("test", tools=[plain_add, "not_a_tool", 42])  # type: ignore[list-item]
        mock_run.assert_called_once()


# ---------------------------------------------------------------------------
# Import from package
# ---------------------------------------------------------------------------


class TestPackageImport:
    def test_import_expose(self):
        from genai_tk.mcp import expose as expose_fn

        assert expose_fn is expose

    def test_import_build_server(self):
        from genai_tk.mcp import build_server as bs

        assert bs is build_server
