"""High-level Agent API for genai-tk.

Provides a simple, production-friendly interface for creating and running agents
without manually wiring LLM factories, checkpointers, middleware, and profiles.

Example:
    ```python
    from genai_tk import Agent

    # From a YAML profile
    agent = Agent("Research")
    result = agent.run("Summarize recent AI news")

    # Ad-hoc (no profile needed)
    agent = Agent(llm="gpt_41mini@openai", tools=["tavily"])
    result = agent.run("What happened today in tech?")

    # Async
    result = await agent.arun("Explain quantum computing")
    ```
"""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr


class Agent(BaseModel):
    """High-level agent that wraps the LangChain agent factory.

    Accepts either a YAML profile name or inline configuration. Sensible
    defaults are applied so that a minimal ``Agent()`` call produces a
    working agent using the default LLM and profile from config.

    Args:
        profile: Name of an agent profile defined in ``langchain.yaml``.
            When provided, the profile's tools, middleware, system prompt,
            and LLM are loaded automatically.
        llm: LLM identifier override (e.g. ``"gpt_41mini@openai"``).
            Takes precedence over the profile's LLM setting.
        tools: Additional tool names (MCP server names) or ``BaseTool``
            instances to attach to the agent.
        system_prompt: System prompt override. Takes precedence over
            the profile's system prompt.
        agent_type: Agent engine type (``"react"``, ``"deep"``, or
            ``"custom"``). Only used for ad-hoc agents (no profile).
        chat_mode: If True, enable memory checkpointer for multi-turn
            conversations. Defaults to False for single-shot usage.
        details: When True, show verbose tool-call panels in the console.
        thread_id: Conversation thread identifier for multi-turn chat.
            Defaults to ``"default"``.
    """

    profile: str | None = Field(None, description="YAML profile name to load")
    llm: str | None = Field(None, description="LLM identifier override")
    tools: list[str | BaseTool] = Field(default_factory=list, description="MCP server names or BaseTool instances")
    system_prompt: str | None = Field(None, description="System prompt override")
    agent_type: str = Field("react", description="Agent engine type: react, deep, or custom")
    chat_mode: bool = Field(False, description="Enable memory checkpointer for multi-turn")
    details: bool = Field(False, description="Show verbose tool-call panels")
    thread_id: str = Field("default", description="Conversation thread ID")

    _compiled_agent: Any = PrivateAttr(default=None)
    _backend: Any = PrivateAttr(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def _ensure_agent(self) -> Any:
        """Lazily build and cache the compiled LangGraph agent."""
        if self._compiled_agent is not None:
            return self._compiled_agent

        from genai_tk.agents.langchain.config import (
            AgentProfileConfig,
            CheckpointerConfig,
            MiddlewareConfig,
            load_unified_config,
            resolve_profile,
        )
        from genai_tk.agents.langchain.factory import create_langchain_agent

        # Separate BaseTool instances from MCP server name strings
        extra_tool_objects: list[BaseTool] = []
        mcp_server_names: list[str] = []
        for t in self.tools:
            if isinstance(t, BaseTool):
                extra_tool_objects.append(t)
            elif isinstance(t, str):
                mcp_server_names.append(t)

        if self.profile:
            # Profile-based: load from YAML and merge overrides
            config = load_unified_config()
            resolved = resolve_profile(config, self.profile)

            # Apply system_prompt override
            if self.system_prompt is not None:
                resolved = resolved.model_copy(update={"system_prompt": self.system_prompt})

            agent = await create_langchain_agent(
                resolved,
                llm_override=self.llm,
                extra_tools=extra_tool_objects or None,
                extra_mcp_servers=mcp_server_names or None,
                force_memory_checkpointer=self.chat_mode,
                details=self.details,
            )
        else:
            # Ad-hoc: build a minimal profile on the fly
            default_middleware = MiddlewareConfig(
                **{"class": "genai_tk.agents.langchain.middleware.rich_middleware:RichToolCallMiddleware"}
            )
            ad_hoc_profile = AgentProfileConfig(
                name="adhoc",
                type=self.agent_type,  # type: ignore[arg-type]
                llm=self.llm,
                system_prompt=self.system_prompt,
                mcp_servers=mcp_server_names,
                middlewares=[default_middleware],
                checkpointer=CheckpointerConfig(type="memory") if self.chat_mode else CheckpointerConfig(type="none"),
            )

            agent = await create_langchain_agent(
                ad_hoc_profile,
                extra_tools=extra_tool_objects or None,
                force_memory_checkpointer=self.chat_mode,
                details=self.details,
            )

        self._compiled_agent = agent
        self._backend = getattr(agent, "_backend", None)
        logger.info(f"Agent initialized (profile={self.profile or 'adhoc'}, llm={self.llm or 'default'})")
        return agent

    async def arun(self, query: str) -> str:
        """Run the agent asynchronously and return the final text response.

        Args:
            query: The user query to execute.

        Returns:
            The assistant's final text response.

        Example:
            ```python
            agent = Agent("Research")
            answer = await agent.arun("What is quantum computing?")
            print(answer)
            ```
        """
        agent = await self._ensure_agent()
        config = {"configurable": {"thread_id": self.thread_id}}
        result = await agent.ainvoke({"messages": [HumanMessage(content=query)]}, config)
        return _extract_text(result)

    def run(self, query: str) -> str:
        """Run the agent synchronously and return the final text response.

        Convenience wrapper around ``arun()`` for scripts and notebooks.

        Args:
            query: The user query to execute.

        Returns:
            The assistant's final text response.

        Example:
            ```python
            agent = Agent(llm="gpt_41mini@openai")
            answer = agent.run("Tell me a joke")
            print(answer)
            ```
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import nest_asyncio

            nest_asyncio.apply()
            return loop.run_until_complete(self.arun(query))

        return asyncio.run(self.arun(query))

    async def ainvoke_raw(self, query: str) -> dict[str, Any]:
        """Run the agent and return the full LangGraph state dict.

        Useful when you need access to intermediate messages, tool calls,
        or other state beyond the final text response.

        Args:
            query: The user query to execute.

        Returns:
            The full LangGraph result dictionary.
        """
        agent = await self._ensure_agent()
        config = {"configurable": {"thread_id": self.thread_id}}
        return await agent.ainvoke({"messages": [HumanMessage(content=query)]}, config)

    async def cleanup(self) -> None:
        """Stop backend resources (if any) and reset the agent.

        Call this when you're done with a deep agent that uses a sandbox backend.
        """
        if self._backend is not None and hasattr(self._backend, "stop"):
            await self._backend.stop()
        self._compiled_agent = None
        self._backend = None


def _extract_text(result: Any) -> str:
    """Extract the final assistant text from a LangGraph result."""
    if isinstance(result, dict) and "messages" in result:
        messages = result["messages"]
        if messages:
            final = messages[-1]
            content = getattr(final, "content", str(final))
            if isinstance(content, list):
                return "\n".join(str(block) for block in content)
            return str(content)
    if isinstance(result, AIMessage):
        content = result.content
        if isinstance(content, list):
            return "\n".join(str(block) for block in content)
        return str(content)
    return str(result)
