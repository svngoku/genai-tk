"""Unit tests for the high-level Agent API (genai_tk.Agent)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from genai_tk.agent import Agent, _extract_text

# ---------------------------------------------------------------------------
# _extract_text helper
# ---------------------------------------------------------------------------


class TestExtractText:
    def test_extract_from_langgraph_dict(self):
        result = {"messages": [HumanMessage(content="hi"), AIMessage(content="hello back")]}
        assert _extract_text(result) == "hello back"

    def test_extract_from_empty_messages(self):
        result = {"messages": []}
        assert _extract_text(result) == str({"messages": []})

    def test_extract_from_ai_message(self):
        result = AIMessage(content="direct message")
        assert _extract_text(result) == "direct message"

    def test_extract_from_list_content(self):
        result = {"messages": [AIMessage(content=["part1", "part2"])]}
        assert _extract_text(result) == "part1\npart2"

    def test_extract_from_ai_message_list_content(self):
        result = AIMessage(content=["chunk1", "chunk2"])
        assert _extract_text(result) == "chunk1\nchunk2"

    def test_extract_from_plain_string(self):
        assert _extract_text("raw string") == "raw string"


# ---------------------------------------------------------------------------
# Agent construction
# ---------------------------------------------------------------------------


class TestAgentConstruction:
    def test_default_agent(self):
        agent = Agent()
        assert agent.profile is None
        assert agent.llm is None
        assert agent.tools == []
        assert agent.agent_type == "react"
        assert agent.chat_mode is False
        assert agent.thread_id == "default"

    def test_profile_agent(self):
        agent = Agent(profile="Research")
        assert agent.profile == "Research"

    def test_adhoc_agent(self):
        agent = Agent(llm="gpt_41mini@openai", system_prompt="Be concise")
        assert agent.llm == "gpt_41mini@openai"
        assert agent.system_prompt == "Be concise"

    def test_with_string_tools(self):
        agent = Agent(tools=["tavily", "filesystem"])
        assert len(agent.tools) == 2

    def test_chat_mode(self):
        agent = Agent(chat_mode=True, thread_id="thread-42")
        assert agent.chat_mode is True
        assert agent.thread_id == "thread-42"


# ---------------------------------------------------------------------------
# Agent.arun — patch at the source modules where imports happen
# ---------------------------------------------------------------------------

FACTORY_PATH = "genai_tk.agents.langchain.factory.create_langchain_agent"
CONFIG_LOAD_PATH = "genai_tk.agents.langchain.config.load_unified_config"
CONFIG_RESOLVE_PATH = "genai_tk.agents.langchain.config.resolve_profile"


class TestAgentArun:
    @pytest.mark.asyncio
    async def test_arun_with_profile(self):
        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {"messages": [AIMessage(content="result from profile")]}

        with (
            patch(CONFIG_LOAD_PATH) as mock_load,
            patch(CONFIG_RESOLVE_PATH) as mock_resolve,
            patch(FACTORY_PATH, new_callable=AsyncMock) as mock_create,
        ):
            mock_load.return_value = MagicMock()
            mock_resolve.return_value = MagicMock()
            mock_create.return_value = mock_agent

            agent = Agent(profile="Research")
            result = await agent.arun("test query")

            assert result == "result from profile"
            mock_create.assert_awaited_once()
            mock_agent.ainvoke.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_arun_adhoc(self):
        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {"messages": [AIMessage(content="adhoc result")]}

        with patch(FACTORY_PATH, new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_agent

            agent = Agent(llm="parrot_local@fake")
            result = await agent.arun("hello")

            assert result == "adhoc result"
            mock_create.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_arun_caches_agent(self):
        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {"messages": [AIMessage(content="cached")]}

        with patch(FACTORY_PATH, new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_agent

            agent = Agent(llm="parrot_local@fake")
            await agent.arun("first")
            await agent.arun("second")

            # Factory called only once — agent is cached
            mock_create.assert_awaited_once()
            assert mock_agent.ainvoke.await_count == 2


# ---------------------------------------------------------------------------
# Agent.ainvoke_raw
# ---------------------------------------------------------------------------


class TestAgentAinvokeRaw:
    @pytest.mark.asyncio
    async def test_returns_full_state(self):
        full_state = {"messages": [AIMessage(content="raw")], "extra": "data"}
        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = full_state

        with patch(FACTORY_PATH, new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_agent

            agent = Agent(llm="parrot_local@fake")
            result = await agent.ainvoke_raw("test")

            assert result == full_state
            assert result["extra"] == "data"


# ---------------------------------------------------------------------------
# Agent.cleanup
# ---------------------------------------------------------------------------


class TestAgentCleanup:
    @pytest.mark.asyncio
    async def test_cleanup_stops_backend(self):
        mock_backend = AsyncMock()
        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {"messages": [AIMessage(content="ok")]}
        mock_agent._backend = mock_backend

        with patch(FACTORY_PATH, new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_agent

            agent = Agent(llm="parrot_local@fake")
            await agent.arun("init")
            await agent.cleanup()

            mock_backend.stop.assert_awaited_once()
            assert agent._compiled_agent is None

    @pytest.mark.asyncio
    async def test_cleanup_without_backend(self):
        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {"messages": [AIMessage(content="ok")]}

        with patch(FACTORY_PATH, new_callable=AsyncMock) as mock_create:
            # No _backend attribute
            del mock_agent._backend
            mock_create.return_value = mock_agent

            agent = Agent(llm="parrot_local@fake")
            await agent.arun("init")
            await agent.cleanup()  # Should not raise


# ---------------------------------------------------------------------------
# Import from top-level package
# ---------------------------------------------------------------------------


class TestTopLevelImport:
    def test_import_from_genai_tk(self):
        from genai_tk import Agent as AgentFromPkg

        assert AgentFromPkg is Agent
