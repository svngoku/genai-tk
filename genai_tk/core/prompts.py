"""Prompt utilities and wrapper functions.

This module provides utilities for creating and managing LangChain prompts,
with support for system messages, user inputs, and additional message types.

Key Features:
- Automatic whitespace normalization
- System message support
- Flexible message composition
- Integration with LangChain's prompt templates

Example:
    >>> # Create simple prompt
    >>> prompt = def_prompt(system="You are a helpful assistant", user="Tell me a joke")

    >>> # Create prompt with additional messages
    >>> prompt = def_prompt(
    ...     system="You are a math tutor",
    ...     user="Solve this problem: {problem}",
    ...     other_msg={"placeholder": "{scratchpad}"},
    ... )
"""

from datetime import datetime
from textwrap import dedent

from langchain_core.prompts import (
    BasePromptTemplate,
    ChatPromptTemplate,
)

DEFAULT_SYSTEM_PROMPT = ""


def datetime_context() -> str:
    """Return a single line with the current date and time.

    Intended to be prepended to any agent's system prompt so the model
    always knows the current time regardless of the framework.

    The format is deliberately human-friendly to avoid the LLM copying it
    verbatim into tool parameters that expect a specific machine format
    (e.g. Tavily's ``start_date`` requires ``YYYY-MM-DD``).

    Example:
        ```python
        prompt = f"{datetime_context()}\\n\\nYou are a helpful assistant."
        ```
    """
    now_local = datetime.now().astimezone()
    return (
        f"Current date: {now_local.strftime('%A, %B %d, %Y')}. "
        f"Current time: {now_local.strftime('%I:%M %p %Z')}. "
        "Do not pass this datetime directly as a parameter to any tool; "
        "use only the format each tool requires."
    )


def with_datetime_context(system_prompt: str | None) -> str:
    """Prepend ``datetime_context()`` to an existing system prompt.

    Args:
        system_prompt: Existing prompt text, or None.

    Returns:
        The datetime line followed by the original prompt (if any).
    """
    dt = datetime_context()
    if system_prompt:
        return f"{dt}\n\n{system_prompt}"
    return dt


def dedent_ws(text: str) -> str:
    r"""'detent' function replacement to remove any common leading whitespace from every line in `text`.

    It address 'dedent' choice to not consider tabs and space as equivalent, by replacing tabs by 4 whitespace,
    so "   hello" and "\\thello" are considered to have common leading whitespace.

    It preserves relative indentation after removing the common leading whitespace.
    """
    text = text.replace("\t", "    ")
    text = dedent(text)
    return text


def def_prompt(system: str | None = None, user: str = "", other_msg: dict | None = None) -> BasePromptTemplate:
    """Small wrapper around 'ChatPromptTemplate.from_messages" with just a user  and optional system prompt and other messages.
    Common leading whitespace and tags are removed from the system and user strings.

    Example:
    .. code-block:: python
        prompt = def_prompt(
            system="You are an helpful agent", user="bla bla", other_msg={"placeholder": "{agent_scratchpad}"}
        )

    """
    if other_msg is None:
        other_msg = {}
    messages: list = []
    if system:
        messages.append(("system", dedent_ws(system)))
    messages.append(("user", dedent_ws(user)))
    other = list(other_msg.items())
    messages.extend(other)
    return ChatPromptTemplate.from_messages(messages)


def dict_input_message(user: str, system: str | None = None) -> dict[str, list[tuple]]:
    """
    Return an input message as dict, in the form : {"messages": [("user", query)]},  typically for use as input of a CompiledGraph.
    """
    msg = [("user", dedent_ws(user))]
    if system:
        msg += [("system", dedent_ws(system))]
    return {"messages": msg}


def list_input_message(user: str, system: str | None = None) -> list[dict[str, str]]:
    """
    Return an input message in the form: [{"role": "user", "content": query}]
    """
    msg = [{"role": "user", "content": dedent_ws(user)}]
    if system:
        msg += [{"role": "system", "content": dedent_ws(system)}]
    return msg
