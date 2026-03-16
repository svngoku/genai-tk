"""Cloud sandbox code execution tools for AI applications.

This module provides LangChain tools for executing code in cloud sandbox environments.
Supported providers include E2B Code Interpreter, Daytona, and Blaxel sandboxes.
Each factory function returns ready-to-use LangChain tools with proper authentication
and resource cleanup.
"""

import os

from langchain_core.tools import BaseTool, tool
from loguru import logger


def e2b_code_interpreter_tool() -> BaseTool:
    """Create a LangChain tool for executing Python code in an E2B sandbox.

    Returns a tool that spins up an E2B Code Interpreter sandbox, executes
    the provided Python code, and returns the result. Requires the
    ``E2B_API_KEY`` environment variable to be set.

    Example:
        ```python
        from genai_tk.tools.langchain.sandbox_tools import e2b_code_interpreter_tool

        tool = e2b_code_interpreter_tool()
        result = tool.invoke({"code": "print(2 + 2)"})
        print(result)
        ```
    """
    try:
        from e2b_code_interpreter import Sandbox
    except ImportError as ex:
        raise ImportError("e2b-code-interpreter package is required. Install with: uv add e2b-code-interpreter") from ex

    api_key = os.environ.get("E2B_API_KEY")
    if not api_key:
        raise ValueError("E2B_API_KEY environment variable is required. Get your key at https://e2b.dev")

    @tool
    def e2b_execute_code(code: str) -> str:
        """Execute Python code in an E2B cloud sandbox and return the result.

        Args:
            code: Python code to execute in the sandbox.
        """
        try:
            with Sandbox.create() as sandbox:
                execution = sandbox.run_code(code)

                if execution.error:
                    logger.error(f"E2B execution error: {execution.error}")
                    return f"Execution error: {execution.error}"

                return execution.text or "Code executed successfully (no output)."
        except Exception as e:
            logger.error(f"E2B sandbox error: {e}")
            return f"Sandbox error: {e}"

    return e2b_execute_code


def daytona_code_execution_tool() -> BaseTool:
    """Create a LangChain tool for executing code in a Daytona sandbox.

    Returns a tool that creates a Daytona sandbox, runs the provided code,
    and cleans up after execution. Requires the ``DAYTONA_API_KEY``
    environment variable to be set.

    Example:
        ```python
        from genai_tk.tools.langchain.sandbox_tools import daytona_code_execution_tool

        tool = daytona_code_execution_tool()
        result = tool.invoke({"code": "print('Hello from Daytona!')"})
        print(result)
        ```
    """
    try:
        from daytona import Daytona, DaytonaConfig
    except ImportError as ex:
        raise ImportError("daytona package is required. Install with: uv add daytona") from ex

    api_key = os.environ.get("DAYTONA_API_KEY")
    if not api_key:
        raise ValueError("DAYTONA_API_KEY environment variable is required. Get your key at https://daytona.io")

    config = DaytonaConfig(api_key=api_key)
    daytona = Daytona(config)

    @tool
    def daytona_execute_code(code: str) -> str:
        """Execute code in a Daytona cloud sandbox and return the result.

        Args:
            code: Code to execute in the sandbox.
        """
        sandbox = None
        try:
            sandbox = daytona.create()
            response = sandbox.process.code_run(code)

            if response.exit_code != 0:
                logger.error(f"Daytona execution failed with exit code {response.exit_code}")
                return f"Execution error (exit code {response.exit_code}): {response.result}"

            return response.result or "Code executed successfully (no output)."
        except Exception as e:
            logger.error(f"Daytona sandbox error: {e}")
            return f"Sandbox error: {e}"
        finally:
            if sandbox:
                try:
                    daytona.delete(sandbox)
                except Exception as e:
                    logger.warning(f"Failed to clean up Daytona sandbox: {e}")

    return daytona_execute_code


async def blaxel_sandbox_tools(sandbox_name: str) -> list[BaseTool]:
    """Create LangChain tools from a Blaxel sandbox MCP server.

    Returns a list of LangChain tools exposed by the specified Blaxel sandbox.
    Requires the ``BL_API_KEY`` and ``BL_WORKSPACE`` environment variables to be set.

    Args:
        sandbox_name: Name of the Blaxel sandbox to connect to.

    Example:
        ```python
        import asyncio
        from genai_tk.tools.langchain.sandbox_tools import blaxel_sandbox_tools

        tools = asyncio.run(blaxel_sandbox_tools("my-sandbox"))
        for t in tools:
            print(t.name)
        ```
    """
    try:
        from blaxel.langgraph import bl_tools
    except ImportError as ex:
        raise ImportError("blaxel package is required. Install with: uv add blaxel") from ex

    if not os.environ.get("BL_API_KEY"):
        raise ValueError("BL_API_KEY environment variable is required. Get your key at https://blaxel.ai")
    if not os.environ.get("BL_WORKSPACE"):
        raise ValueError("BL_WORKSPACE environment variable is required.")

    logger.info(f"Loading Blaxel sandbox tools for '{sandbox_name}'")
    tools = await bl_tools([f"sandbox/{sandbox_name}"])
    return tools
