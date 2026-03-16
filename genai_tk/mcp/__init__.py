"""MCP server exposure package for genai-tk.

This package allows exposing LangChain tools and agents as MCP servers,
configurable via YAML, launchable via CLI or standalone scripts.

Quick API:
    ```python
    from genai_tk.mcp import expose


    def greet(name: str) -> str:
        return f"Hello, {name}!"


    expose("greeter", tools=[greet])
    ```

YAML-based API:
    ```bash
    # Serve a configured MCP server over stdio
    uv run cli mcp serve --name search

    # List available servers
    uv run cli mcp list

    # Generate a standalone script
    uv run cli mcp generate --name search --output server_search.py
    ```
"""

from genai_tk.mcp.quick import build_server, expose

__all__ = ["expose", "build_server"]
