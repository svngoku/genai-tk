"""GenAI Toolkit - Core AI components for building applications."""

__version__ = "0.1.0"

# Import main components to make them easily accessible
try:
    from . import core, extra, utils  # noqa: F401
except ImportError:
    # Handle case where dependencies aren't installed
    pass


# High-level API — lazy import to avoid pulling heavy deps at module load
def __getattr__(name: str):
    if name == "Agent":
        from genai_tk.agent import Agent

        return Agent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
