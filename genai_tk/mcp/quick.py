"""One-liner MCP server exposure for genai-tk.

Create and run MCP servers from Python functions and LangChain tools
without writing any YAML configuration.

Example:
    ```python
    from genai_tk.mcp import expose


    def greet(name: str) -> str:
        return f"Hello, {name}!"


    expose("greeter", tools=[greet])
    ```
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

from langchain_core.tools import BaseTool
from loguru import logger
from mcp.server.fastmcp import FastMCP

from genai_tk.mcp.tool_adapter import register_tools


def expose(
    name: str,
    tools: list[BaseTool | Callable[..., Any]],
    *,
    description: str | None = None,
    transport: str = "stdio",
    host: str = "0.0.0.0",
    port: int = 8000,
) -> None:
    """Create and run an MCP server from a list of tools or functions.

    Accepts any mix of LangChain ``BaseTool`` instances and plain Python
    callables. Plain functions are auto-wrapped as MCP tools using their
    name, docstring, and type annotations.

    Args:
        name: Server name (shown to MCP clients).
        tools: List of LangChain ``BaseTool`` instances or plain callables.
        description: Optional server description.
        transport: MCP transport protocol: ``"stdio"`` (default), ``"sse"``,
            or ``"streamable-http"``.
        host: Bind address for HTTP transports (default ``"0.0.0.0"``).
        port: Port for HTTP transports (default ``8000``).

    Example:
        ```python
        from langchain_core.tools import tool
        from genai_tk.mcp import expose

        @tool
        def add(a: int, b: int) -> int:
            \"\"\"Add two numbers.\"\"\"
            return a + b

        def multiply(x: int, y: int) -> int:
            \"\"\"Multiply two numbers.\"\"\"
            return x * y

        # Mix of BaseTool and plain function
        expose("calculator", tools=[add, multiply])
        ```
    """
    server = FastMCP(name, instructions=description)

    lc_tools: list[BaseTool] = []
    plain_funcs: list[Callable[..., Any]] = []

    for t in tools:
        if isinstance(t, BaseTool):
            lc_tools.append(t)
        elif callable(t):
            plain_funcs.append(t)
        else:
            logger.warning(f"Skipping non-callable, non-BaseTool item: {t!r}")

    # Register LangChain tools via existing adapter
    if lc_tools:
        register_tools(server, lc_tools)
        logger.info(f"[{name}] registered {len(lc_tools)} LangChain tool(s)")

    # Register plain functions directly on FastMCP
    for func in plain_funcs:
        tool_name = func.__name__
        tool_desc = inspect.getdoc(func) or f"Tool: {tool_name}"
        server.add_tool(func, name=tool_name, description=tool_desc)
        logger.info(f"[{name}] registered function tool: {tool_name!r}")

    total = len(lc_tools) + len(plain_funcs)
    logger.info(f"Starting MCP server '{name}' ({total} tool(s)) over {transport}")
    server.run(transport=transport)  # type: ignore[arg-type]


def build_server(
    name: str,
    tools: list[BaseTool | Callable[..., Any]],
    *,
    description: str | None = None,
) -> FastMCP:
    """Build an MCP server without running it.

    Useful for testing or when you need to customize the server before
    starting it.

    Args:
        name: Server name.
        tools: List of LangChain ``BaseTool`` instances or plain callables.
        description: Optional server description.

    Returns:
        Configured ``FastMCP`` instance (not yet running).

    Example:
        ```python
        from genai_tk.mcp import build_server

        server = build_server("test", tools=[my_func])
        # Inspect server.list_tools(), add more tools, then server.run()
        ```
    """
    server = FastMCP(name, instructions=description)

    lc_tools: list[BaseTool] = []
    plain_funcs: list[Callable[..., Any]] = []

    for t in tools:
        if isinstance(t, BaseTool):
            lc_tools.append(t)
        elif callable(t):
            plain_funcs.append(t)

    if lc_tools:
        register_tools(server, lc_tools)

    for func in plain_funcs:
        tool_name = func.__name__
        tool_desc = inspect.getdoc(func) or f"Tool: {tool_name}"
        server.add_tool(func, name=tool_name, description=tool_desc)

    return server
