# GenAI Toolkit (`genai-tk`)

A comprehensive toolkit for building AI applications with LangChain, LangGraph, and modern AI frameworks.

## Overview

GenAI Toolkit provides reusable components, agents, and utilities for building sophisticated AI applications. It focuses on:

- **Multi-Agent Workflows** - Build complex agent interactions with LangGraph
- **RAG (Retrieval Augmented Generation)** - Full RAG pipeline support with multiple vector stores and retrievers
- **Framework-Specific Tools** - Extensive tool ecosystems for LangChain and SmolAgents
- **Type Safety** - Pydantic-based structured data handling with dynamic models
- **Enhanced Configuration** - Flexible, hierarchical config system with directory auto-discovery
- **Provider Agnostic** - Support for OpenAI, Anthropic, local models, and 100+ providers via LiteLLM
- **Robust Error Handling** - Graceful handling of optional dependencies and missing configurations
- **Developer Experience** - Works from any project directory, comprehensive CLI with framework-specific shells
- **Modular Architecture** - Clean separation between core, extra, tools, and utilities
- **Data Processing** - Built-in OCR, anonymization, and image analysis capabilities

## ✨ Recent Enhancements

- **🚀 High-level Agent API**: `from genai_tk import Agent` — create and run agents in 3 lines, with profile-based or ad-hoc configuration
- **📦 Modular Dependencies**: Core install is ~20 packages; heavy deps moved to 10 optional extras (`[providers]`, `[agents]`, `[rag]`, `[all]`, etc.)
- **🔌 One-liner MCP Servers**: `from genai_tk.mcp import expose` — turn any Python function or LangChain tool into an MCP server instantly
- **🧹 Unified LLM API**: Single clean `get_llm()` function — deprecated parameters and duplicate `get_llm_unified()` removed
- **🤖 Unified LangChain Agents**: Single `cli agents langchain` command with YAML profiles for `react`, `deep`, and `custom` agent types
- **📋 YAML-Driven Agent Config**: Middleware, checkpointer, LLM, tools, and MCP servers all configurable per profile with global defaults
- **📍 Flexible Configuration Discovery**: Automatically finds config files by searching parent directories
- **📁 Work from Anywhere**: Run commands from notebooks, subdirectories, or any project location
- **⚙️ Dynamic Path Resolution**: Paths automatically adjust based on project structure
- **🔄 Environment Switching**: Easy switching between development, testing, production configurations
- **🔐 Optional Dependencies**: Graceful handling of missing packages (e.g., `langchain_postgres`)

## Installation

```bash
# Install core only (LangChain + OpenAI — lightweight, ~20 deps)
uv pip install git+https://github.com/tclatos/genai-tk@main

# Install with specific extras
uv pip install "genai-tk[providers] @ git+https://github.com/tclatos/genai-tk@main"  # All LLM providers
uv pip install "genai-tk[agents,mcp] @ git+https://github.com/tclatos/genai-tk@main" # Agent frameworks + MCP
uv pip install "genai-tk[rag] @ git+https://github.com/tclatos/genai-tk@main"         # ChromaDB, BM25, chunking

# Install everything
uv pip install "genai-tk[all] @ git+https://github.com/tclatos/genai-tk@main"

# Development installation (all extras included)
git clone https://github.com/tclatos/genai-tk.git
cd genai-tk
uv sync --all-extras
```

**Available extras:** `providers`, `langchain-extra`, `agents`, `mcp`, `rag`, `graph`, `nlp`, `search`, `docs`, `workflow`, `all`

## Quick Start

### 🚀 High-level Agent API (recommended)

The fastest way to get a production-ready agent running:

```python
from genai_tk import Agent

# From a YAML profile — loads LLM, tools, middleware from config
agent = Agent("Research")
result = agent.run("Summarize recent AI news")
print(result)

# Ad-hoc — no config file needed
agent = Agent(llm="gpt_41mini@openai", tools=["tavily"])
result = agent.run("What happened in tech today?")

# Async support
result = await agent.arun("Explain quantum computing")

# Multi-turn chat
agent = Agent("Research", chat_mode=True)
agent.run("What is LangGraph?")
agent.run("How does it compare to CrewAI?")  # remembers context
```

### 🔌 One-liner MCP Server

Expose any function or LangChain tool as an MCP server for Claude Desktop, Cursor, or Amp:

```python
from genai_tk.mcp import expose

def greet(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {name}!"

def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

expose("my-tools", tools=[greet, add])  # starts stdio MCP server
```

Works with LangChain tools too:

```python
from langchain_core.tools import tool
from genai_tk.mcp import expose

@tool
def search_docs(query: str) -> str:
    """Search documentation."""
    return f"Results for: {query}"

expose("doc-search", tools=[search_docs])
```

### Core API

```python
from genai_tk.core.llm_factory import get_llm

# Create LLM — uses default from config
llm = get_llm()

# Specify a model
llm = get_llm(llm="gpt_41mini@openai", streaming=True)

# Use in a chain
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
chain = prompt | get_llm()
result = chain.invoke({"topic": "AI"})
```

### CLI Usage

```bash
# List all configured agent profiles
cli agents langchain --list

# Single-shot query with default profile
cli agents langchain "What's the weather like today?"

# Select a specific profile
cli agents langchain -p filesystem "List Python files in src/"

# Interactive chat mode
cli agents langchain -p Coding --chat

# Override engine type and LLM at runtime
cli agents langchain -p Research --type react --llm gpt_41mini@openai "Quick summary"

# SmolAgents shell
cli agents smol

# RAG operations
cli rag create-index my_data/
cli rag query "What are the main features?"
```

### Configuration Management

```python
from genai_tk.utils.config_mngr import global_config

# Works from notebooks/, demos/, or any subdirectory
config = global_config()

# Get configuration values
default_model = config.get('llm.models.default')
project_path = config.get('paths.project')

# Switch environments
config.select_config('production')
```

## Package Structure

```
genai_tk/
├── __init__.py              # Package root — exports Agent
├── agent.py                 # ⭐ High-level Agent API (Agent class)
├── agents/                  # Agent implementations
│   ├── langchain/          # Unified LangChain agent (react | deep | custom)
│   │   ├── config.py       # Pydantic config models + loader
│   │   ├── factory.py      # Unified agent factory
│   │   ├── agent.py        # Shell & direct runner
│   │   ├── commands.py     # CLI command registration
│   │   └── middleware/     # Rich console middleware
│   ├── deepagent/          # DeepAgents integration
│   ├── deer_flow/          # Deer-flow integration
│   └── smolagents/         # SmolAgents implementation
├── core/                    # Core AI components
│   ├── llm_factory.py      # LLM creation and management (get_llm)
│   ├── embeddings_factory.py # Embeddings models
│   ├── embeddings_store.py # Vector databases
│   ├── cache.py            # Caching utilities
│   ├── chain_registry.py   # Chain registration system
│   ├── mcp_client.py       # Model Context Protocol client
│   └── ...
├── mcp/                     # MCP server tools
│   ├── quick.py            # ⭐ One-liner expose() and build_server()
│   ├── server_builder.py   # YAML-based MCP server builder
│   ├── tool_adapter.py     # LangChain → FastMCP adapter
│   └── ...
├── extra/                   # Extended AI capabilities
│   ├── graphs/             # Agent graphs (ReAct, SQL, etc.)
│   ├── loaders/            # Data loaders (OCR, etc.)
│   ├── retrievers/         # Retrieval components (BM25, etc.)
│   ├── rag/                # RAG pipelines
│   └── ...
├── main/                    # CLI and command interface
│   └── cli.py              # Main CLI entry point
├── tools/                   # Framework-specific tools
│   ├── langchain/          # LangChain-compatible tools
│   └── smolagents/         # SmolAgents-compatible tools
└── utils/                   # Utilities and helpers
    ├── config_mngr.py      # Configuration management
    └── ...
```

## Key Components

### Top-level API

- **`Agent`** (`genai_tk.agent`) - High-level agent with sync/async `.run()` / `.arun()` — wraps the full agent factory
- **`expose()`** (`genai_tk.mcp`) - One-liner MCP server from functions or LangChain tools
- **`get_llm()`** (`genai_tk.core.llm_factory`) - Unified LLM creation with fuzzy model resolution

### Core (`genai_tk.core`)

- **LLM Factory** - Creates Language Models from multiple providers with fuzzy model resolution
- **Embeddings Factory** - Provides embeddings for semantic search
- **Embeddings Store** - Vector database management for RAG (Chroma, InMemory, PgVector)
- **MCP Client** - Model Context Protocol client integration
- **Cache** - Intelligent caching system for AI responses
- **Chain Registry** - Centralized chain registration and discovery

### MCP (`genai_tk.mcp`)

- **Quick API** - `expose()` and `build_server()` for instant MCP server creation
- **Server Builder** - YAML-driven MCP server configuration and startup
- **Tool Adapter** - Automatic LangChain BaseTool → MCP tool wrapping

### Agents (`genai_tk.agents`)

- **LangChain** - Unified agent factory (react, deep, custom) with YAML profiles
- **DeepAgents** - Deep agent integration with planning, skills, and backends
- **SmolAgents** - SmolAgents framework integration
- **Deer-flow** - ByteDance Deer-flow agent integration

### Extra (`genai_tk.extra`)

- **Agent Graphs** - ReAct, SQL, and structured output agents
- **Data Loaders** - OCR (Mistral), document processing
- **Retrievers** - BM25S and other retrieval components
- **RAG Pipelines** - End-to-end RAG workflows with Prefect orchestration

## Supported AI Providers

- **OpenAI** - GPT models, embeddings, tools
- **Anthropic** - Claude models (via OpenRouter)
- **Local Models** - Ollama, VLLM, local inference
- **DeepSeek** - DeepSeek models and reasoning  
- **Mistral** - Mistral AI models
- **Groq** - Fast inference endpoints
- **LiteLLM** - 100+ LLM providers unified API

## Agent Frameworks

### LangChain Agents (`cli agents langchain`)

All LangChain-based agents are configured through a single `config/basic/agents/langchain.yaml` file and launched via one command.

**Profile types:**
- `react` — standard ReAct agent via LangChain `create_agent`
- `deep` — advanced multi-step agent via DeepAgents `create_deep_agent` with skills, planning, and file system access
- `custom` — functional ReAct agent built from scratch with LangGraph

**CLI:**
```bash
# List profiles
cli agents langchain --list

# Single-shot (default profile)
cli agents langchain "Research quantum computing trends"

# Select profile + interactive chat
cli agents langchain -p Coding --chat

# Override type and LLM without editing YAML
cli agents langchain -p Research --type react --llm gpt_41mini@openai "Quick answer"
```

**YAML profile example:**
```yaml
langchain_agents:
  defaults:
    type: react
    middlewares:
      - class: genai_tk.agents.langchain.middleware.rich_middleware:RichToolCallMiddleware
    checkpointer:
      type: none

  default_profile: "Research"

  profiles:
    - name: "filesystem"
      type: react
      mcp_servers: [filesystem]

    - name: "Research"
      type: deep
      llm: "gpt_41@openai"
      mcp_servers: [tavily-mcp]
      skill_directories: ["${paths.project}/skills"]
      middlewares:
        - class: deepagents.middleware.summarization:SummarizationMiddleware
          model: "gpt-4.1@openrouter"
```

### Deer-flow Integration

GenAI Toolkit integrates with [Deer-flow](https://github.com/bytedance/deer-flow), ByteDance's LangGraph-based agent system with advanced features like subagents, sandboxed execution, and skill libraries.

**Setup**:
```bash
# Quick setup with automated script
bash scripts/setup_deerflow.sh

# Or manual setup:
export DEER_FLOW_PATH=/path/to/deer-flow
git clone https://github.com/bytedance/deer-flow.git $DEER_FLOW_PATH
uv pip install -e $DEER_FLOW_PATH/backend
```

**Usage**:
```bash
# List available profiles
cli agents deerflow --list

# Run in chat mode
cli agents deerflow --chat

# Single-shot query
cli agents deerflow "Research the latest AI developments"
```

See [docs/Deer_Flow_Integration.md](docs/Deer_Flow_Integration.md) for full documentation.

## Configuration

**🚀 Enhanced Configuration System**

GenAI Toolkit features a **flexible, hierarchical configuration system** with these key improvements:

- **📁 Parent Directory Search**: Automatically finds configuration files by searching up the directory tree
- **🎯 Works from Any Directory**: Run commands from notebooks, subdirectories, or anywhere in your project
- **⚙️ Dynamic Path Resolution**: Paths automatically adjust based on project location
- **🔄 Environment Overrides**: Switch between development, testing, and production configurations
- **🔐 Optional Dependencies**: Graceful handling of missing optional dependencies

**Configuration Structure**:
```yaml
# config/app_conf.yaml - Main configuration
default_config: ${oc.env:BLUEPRINT_CONFIG,basic}

paths:
  project: ${oc.env:PWD}  # Auto-detected project root
  config: ${paths.project}/config
  data_root: ${oc.env:HOME}

# config/basic/providers/llm.yaml - LLM configurations  
llm:
  models:
    default: gpt_4_openai
    gpt_4_openai:
      provider: openai
      model: gpt-4-turbo
    gpt_4_groq:
      provider: groq
      model: llama-3.1-70b-versatile

# config/basic/providers/embeddings.yaml - Embedding configurations
embeddings:
  models:
    default: text_small_openai
    text_small_openai:
      provider: openai
      model: text-embedding-3-small

# config/basic/agents/langchain.yaml - Unified agent profiles
langchain_agents:
  defaults:
    type: react
    checkpointer: {type: none}
  default_profile: "Research"
  profiles:
    - name: "Research"
      type: deep
      llm: "gpt_41@openai"
```

**Environment Variables** (loaded from `.env` in project root or parents):
```bash
# API Keys
OPENAI_API_KEY=your-key-here
GROQ_API_KEY=your-groq-key
ANTHROPIC_API_KEY=your-anthropic-key

# Configuration Selection
BLUEPRINT_CONFIG=development  # Switch configuration environments
```

**Usage from Any Directory**:
```python
from genai_tk.utils.config_mngr import global_config

# Works from notebooks/, demos/, or any subdirectory!
config = global_config()
model_id = config.get('llm.models.default')
```

## Development

```bash
# Install development dependencies
make install-dev

# Format code  
make fmt

# Run linting
make lint

# Run tests
make test

# Run all checks
make check
```

## Testing

```bash
# Unit tests only
make test-unit

# Integration tests only  
make test-integration

# Run specific test
uv run pytest tests/unit_tests/core/test_llm_factory.py::test_basic_call -v

# Run tests by pattern
uv run pytest tests/unit_tests/ -k "test_name_pattern" -v
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes following the coding standards
4. Add tests for new functionality
5. Run `make check` to ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Related Projects

- **GenAI Blueprint** (`genai_bp`) - Application framework built on genai-tk
- **LangChain** - Core LLM application framework
- **LangGraph** - Multi-agent workflow engine

## Support

- Documentation: [Agents.md](Agents.md)
- Issues: [GitHub Issues](https://github.com/tclatos/genai-tk/issues)
- Discussions: [GitHub Discussions](https://github.com/tclatos/genai-tk/discussions)