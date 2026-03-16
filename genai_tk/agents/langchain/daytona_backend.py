"""LangChain deepagents BackendProtocol backed by Daytona sandboxes.

Uses the ``daytona-sdk`` Python package to create and manage Daytona sandboxes
for secure code execution.  The sandbox lifecycle is managed via ``start()`` /
``stop()`` (or the async context manager).  All file and shell operations are
routed through ``sandbox.process.exec()``.

``DaytonaBackend`` implements ``SandboxBackendProtocol`` (which extends
``BackendProtocol``) from deepagents, providing:

- ``execute_tool`` — low-level named-tool dispatch (bash, ls, read_file, write_file, str_replace)
- ``aexecute`` — shell command → ``ExecuteResponse``
- ``als_info`` / ``aread`` / ``awrite`` / ``aedit`` — file operations → typed results
- ``agrep_raw`` / ``aglob_info`` — search operations
- ``aupload_files`` / ``adownload_files`` — bulk file I/O

Example:
    ```python
    from genai_tk.agents.langchain.daytona_backend import DaytonaBackend, DaytonaBackendConfig

    config = DaytonaBackendConfig(api_key="your-api-key")
    async with DaytonaBackend(config=config) as backend:
        result = await backend.execute_tool("bash", {"command": "echo hello"})
        print(result.output)

        resp = await backend.aexecute("ls /tmp")
        print(resp.output)

        files = await backend.als_info("/home/user")
        print([f["path"] for f in files])
    ```
"""

from __future__ import annotations

import os
import shlex
import uuid
from typing import Any

from deepagents.backends.protocol import (
    EditResult,
    ExecuteResponse,
    FileDownloadResponse,
    FileInfo,
    FileUploadResponse,
    GrepMatch,
    SandboxBackendProtocol,
    WriteResult,
)
from loguru import logger
from pydantic import BaseModel, Field, PrivateAttr

from genai_tk.agents.langchain.sandbox_backend import SandboxToolResult

_SUPPORTED_TOOLS = frozenset({"bash", "ls", "read_file", "write_file", "str_replace"})


class DaytonaBackendConfig(BaseModel):
    """Configuration for ``DaytonaBackend``."""

    api_key: str | None = None
    language: str = "python"
    work_dir: str = "/home/user"


class DaytonaBackend(SandboxBackendProtocol, BaseModel):
    """deepagents ``SandboxBackendProtocol`` backed by Daytona sandboxes.

    Manages the Daytona sandbox lifecycle: creates a sandbox on ``start()``,
    exposes the full ``BackendProtocol`` interface plus the low-level
    ``execute_tool`` dispatch, and deletes the sandbox on ``stop()``.

    All shell and file operations are implemented via ``sandbox.process.exec()``.
    """

    config: DaytonaBackendConfig = Field(default_factory=DaytonaBackendConfig)

    _daytona_client: Any = PrivateAttr(default=None)
    _sandbox: Any = PrivateAttr(default=None)
    _instance_id: str = PrivateAttr(default_factory=lambda: uuid.uuid4().hex[:12])

    model_config = {"arbitrary_types_allowed": True}

    @property
    def id(self) -> str:
        """Unique identifier: sandbox ID when running, otherwise a random hex."""
        if self._sandbox is not None:
            return str(getattr(self._sandbox, "id", self._instance_id))
        return self._instance_id

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Create a Daytona client and sandbox."""
        try:
            from daytona import CreateSandboxBaseParams, Daytona, DaytonaConfig  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError("daytona package is required: uv add daytona") from exc

        api_key = self.config.api_key or os.environ.get("DAYTONA_API_KEY")
        if not api_key:
            raise ValueError("Daytona API key must be provided via config.api_key or DAYTONA_API_KEY env var")

        logger.debug("Creating Daytona client")
        self._daytona_client = Daytona(DaytonaConfig(api_key=api_key))

        logger.debug(f"Creating Daytona sandbox (language={self.config.language})")
        self._sandbox = self._daytona_client.create(
            CreateSandboxBaseParams(language=self.config.language),
        )
        logger.info(f"Daytona sandbox ready: {self.id}")

    async def stop(self) -> None:
        """Delete the Daytona sandbox and clean up."""
        if self._sandbox is not None and self._daytona_client is not None:
            sandbox_id = self.id
            logger.debug(f"Deleting Daytona sandbox {sandbox_id}")
            try:
                self._daytona_client.delete(self._sandbox)
            except Exception as exc:
                logger.warning(f"Failed to delete sandbox {sandbox_id}: {exc}")
            self._sandbox = None
        self._daytona_client = None
        logger.info("DaytonaBackend stopped")

    async def __aenter__(self) -> DaytonaBackend:
        await self.start()
        return self

    async def __aexit__(self, *_args: object) -> None:
        await self.stop()

    # ------------------------------------------------------------------
    # BackendProtocol interface
    # ------------------------------------------------------------------

    def list_tools(self) -> list[str]:
        """Return the tool names supported by this backend."""
        return sorted(_SUPPORTED_TOOLS)

    async def execute_tool(self, tool_name: str, tool_input: dict) -> SandboxToolResult:
        """Execute a named tool inside the Daytona sandbox.

        Args:
            tool_name: One of ``bash``, ``ls``, ``read_file``, ``write_file``, ``str_replace``.
            tool_input: Tool-specific parameters.

        Returns:
            ``SandboxToolResult`` with ``output``, ``exit_code``, and optional ``error``.
        """
        if self._sandbox is None:
            raise RuntimeError("Backend not started — use 'async with DaytonaBackend()' or call start() first")
        if tool_name not in _SUPPORTED_TOOLS:
            raise ValueError(f"Unsupported tool '{tool_name}'. Available: {sorted(_SUPPORTED_TOOLS)}")

        try:
            match tool_name:
                case "bash":
                    return await self._run_bash(tool_input)
                case "ls":
                    return await self._run_ls(tool_input)
                case "read_file":
                    return await self._run_read_file(tool_input)
                case "write_file":
                    return await self._run_write_file(tool_input)
                case "str_replace":
                    return await self._run_str_replace(tool_input)
                case _:  # pragma: no cover
                    raise ValueError(f"Unhandled tool: {tool_name}")
        except Exception as exc:
            logger.error(f"Tool '{tool_name}' raised: {exc}")
            return SandboxToolResult(tool_name=tool_name, output="", exit_code=1, error=str(exc))

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    def _exec(self, command: str) -> Any:
        """Run a command in the sandbox via ``sandbox.process.exec()``."""
        assert self._sandbox is not None
        return self._sandbox.process.exec(command)

    async def _run_bash(self, tool_input: dict) -> SandboxToolResult:
        command: str = tool_input["command"]
        result = self._exec(command)
        output = getattr(result, "output", "") or ""
        exit_code = getattr(result, "exit_code", 0) or 0
        return SandboxToolResult(tool_name="bash", output=output, exit_code=exit_code)

    async def _run_ls(self, tool_input: dict) -> SandboxToolResult:
        path: str = tool_input.get("path", self.config.work_dir)
        result = self._exec(f"ls -la {shlex.quote(path)}")
        output = getattr(result, "output", "") or ""
        return SandboxToolResult(tool_name="ls", output=output)

    async def _run_read_file(self, tool_input: dict) -> SandboxToolResult:
        file_path: str = tool_input["path"]
        result = self._exec(f"cat {shlex.quote(file_path)}")
        output = getattr(result, "output", "") or ""
        exit_code = getattr(result, "exit_code", 0) or 0
        return SandboxToolResult(tool_name="read_file", output=output, exit_code=exit_code)

    async def _run_write_file(self, tool_input: dict) -> SandboxToolResult:
        file_path: str = tool_input["path"]
        content: str = tool_input["content"]
        cmd = f"mkdir -p $(dirname {shlex.quote(file_path)}) && cat > {shlex.quote(file_path)} << 'GENAI_EOF'\n{content}\nGENAI_EOF"
        result = self._exec(cmd)
        exit_code = getattr(result, "exit_code", 0) or 0
        if exit_code != 0:
            error_output = getattr(result, "output", "") or ""
            return SandboxToolResult(tool_name="write_file", output="", exit_code=exit_code, error=error_output)
        return SandboxToolResult(tool_name="write_file", output=f"Written: {file_path}")

    async def _run_str_replace(self, tool_input: dict) -> SandboxToolResult:
        file_path: str = tool_input["path"]
        old_str: str = tool_input["old_str"]
        new_str: str = tool_input["new_str"]

        read_result = self._exec(f"cat {shlex.quote(file_path)}")
        original = getattr(read_result, "output", "") or ""

        count = original.count(old_str)
        if count == 0:
            return SandboxToolResult(
                tool_name="str_replace",
                output="",
                exit_code=1,
                error=f"String not found in {file_path}",
            )

        updated = original.replace(old_str, new_str)
        write_cmd = f"cat > {shlex.quote(file_path)} << 'GENAI_EOF'\n{updated}\nGENAI_EOF"
        self._exec(write_cmd)
        return SandboxToolResult(
            tool_name="str_replace",
            output=f"Replaced {count}x in: {file_path}",
        )

    # ------------------------------------------------------------------
    # SandboxBackendProtocol — aexecute
    # ------------------------------------------------------------------

    async def aexecute(  # noqa: ASYNC109
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        """Execute a shell command and return a structured ``ExecuteResponse``.

        Args:
            command: Shell command to run inside the sandbox.
            timeout: Not enforced by this backend (ignored).

        Returns:
            ``ExecuteResponse`` with combined output and exit code.
        """
        result = await self._run_bash({"command": command})
        return ExecuteResponse(output=result.output, exit_code=result.exit_code)

    # ------------------------------------------------------------------
    # BackendProtocol — file operations
    # ------------------------------------------------------------------

    async def als_info(self, path: str) -> list[FileInfo]:
        """List directory contents with metadata.

        Args:
            path: Absolute path to the directory.

        Returns:
            List of ``FileInfo`` dicts with ``path`` and optional ``size``.
        """
        result = self._exec(f"ls -la {shlex.quote(path)}")
        output = getattr(result, "output", "") or ""
        infos: list[FileInfo] = []
        for line in output.splitlines():
            parts = line.split()
            if len(parts) >= 9 and not line.startswith("total"):
                file_name = " ".join(parts[8:])
                if file_name in (".", ".."):
                    continue
                full_path = f"{path.rstrip('/')}/{file_name}"
                info: FileInfo = {"path": full_path}
                try:
                    info["size"] = int(parts[4])
                except (ValueError, IndexError):
                    pass
                infos.append(info)
        return infos

    async def aread(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:
        """Read a file with optional line-based pagination.

        Lines are 1-indexed in the returned text.  Each returned line is
        prefixed with its absolute line number followed by a colon and space.

        Args:
            file_path: Absolute path to the file.
            offset: Zero-based line index to start from (default: 0).
            limit: Maximum number of lines to return (default: 2000).

        Returns:
            Formatted string with line numbers, or an error message prefixed
            with ``Error:``.
        """
        result = self._exec(f"cat {shlex.quote(file_path)}")
        exit_code = getattr(result, "exit_code", 0) or 0
        if exit_code != 0:
            error_output = getattr(result, "output", "") or ""
            return f"Error: {error_output.strip() or 'Failed to read file'}"
        content = getattr(result, "output", "") or ""
        lines = content.splitlines(keepends=True)
        page = lines[offset : offset + limit]
        return "".join(f"{offset + i + 1}: {line}" for i, line in enumerate(page))

    async def awrite(self, file_path: str, content: str) -> WriteResult:
        """Write content to a new file; returns an error if the file already exists.

        Args:
            file_path: Absolute destination path.
            content: Text content to write.

        Returns:
            ``WriteResult`` with ``path`` on success or ``error`` on failure.
        """
        check = self._exec(f"test -e {shlex.quote(file_path)} && echo EXISTS || echo ABSENT")
        check_output = getattr(check, "output", "") or ""
        if "EXISTS" in check_output:
            return WriteResult(error=f"File already exists: {file_path}")
        try:
            cmd = (
                f"mkdir -p $(dirname {shlex.quote(file_path)}) && "
                f"cat > {shlex.quote(file_path)} << 'GENAI_EOF'\n{content}\nGENAI_EOF"
            )
            result = self._exec(cmd)
            exit_code = getattr(result, "exit_code", 0) or 0
            if exit_code != 0:
                error_output = getattr(result, "output", "") or ""
                return WriteResult(error=error_output or f"Failed to write {file_path}")
            return WriteResult(path=file_path)
        except Exception as exc:
            return WriteResult(error=str(exc))

    async def aedit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """Replace ``old_string`` with ``new_string`` in an existing file.

        Reads the file into memory, performs the Python string replacement,
        then writes it back.

        Args:
            file_path: Absolute path to the file to edit.
            old_string: Exact text to search for.
            new_string: Replacement text.
            replace_all: When ``True`` replace all occurrences; when ``False``
                (default) replace only the first occurrence.

        Returns:
            ``EditResult`` with ``path`` and ``occurrences`` on success, or
            ``error`` when the string is not found or the file cannot be read.
        """
        read_result = self._exec(f"cat {shlex.quote(file_path)}")
        exit_code = getattr(read_result, "exit_code", 0) or 0
        if exit_code != 0:
            error_output = getattr(read_result, "output", "") or ""
            return EditResult(error=f"Cannot read {file_path}: {error_output}")

        original = getattr(read_result, "output", "") or ""
        count = original.count(old_string)
        if count == 0:
            return EditResult(error=f"String not found in {file_path}")

        if replace_all:
            updated = original.replace(old_string, new_string)
            occurrences = count
        else:
            updated = original.replace(old_string, new_string, 1)
            occurrences = 1

        write_cmd = f"cat > {shlex.quote(file_path)} << 'GENAI_EOF'\n{updated}\nGENAI_EOF"
        write_result = self._exec(write_cmd)
        write_exit = getattr(write_result, "exit_code", 0) or 0
        if write_exit != 0:
            error_output = getattr(write_result, "output", "") or ""
            return EditResult(error=f"Cannot write {file_path}: {error_output}")

        return EditResult(path=file_path, occurrences=occurrences)

    async def agrep_raw(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> list[GrepMatch] | str:
        """Search for a literal text pattern in files using ``grep``.

        Args:
            pattern: Literal string to search for (exact substring match).
            path: Directory to search in; defaults to ``work_dir``.
            glob: Optional filename glob pattern to restrict the search.

        Returns:
            List of ``GrepMatch`` dicts on success, or an error string on
            grep failure (exit code > 1).
        """
        search_path = path or self.config.work_dir
        cmd = f"grep -rna {shlex.quote(pattern)} {shlex.quote(search_path)}"
        if glob:
            cmd += f" --include={shlex.quote(glob)}"
        result = self._exec(cmd)
        exit_code = getattr(result, "exit_code", 0) or 0
        output = getattr(result, "output", "") or ""
        if exit_code > 1:
            return f"grep error: {output.strip()}"
        matches: list[GrepMatch] = []
        for line in output.splitlines():
            parts = line.split(":", 2)
            if len(parts) == 3:
                try:
                    matches.append(GrepMatch(path=parts[0], line=int(parts[1]), text=parts[2]))
                except ValueError:
                    pass
        return matches

    async def aglob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        """Find files matching a glob pattern.

        Uses Python's ``glob.glob`` inside the sandbox so that ``**`` recursive
        patterns are fully supported.

        Args:
            pattern: Glob pattern with wildcards.
            path: Base directory to search from (default: ``/``).

        Returns:
            List of ``FileInfo`` dicts with the matched absolute paths.
        """
        py_cmd = (
            "import glob, os, sys; "
            f"results = glob.glob({pattern!r}, root_dir={path!r}, recursive=True); "
            f"[print(os.path.join({path!r}, r)) for r in sorted(results)]"
        )
        cmd = f"python3 -c {shlex.quote(py_cmd)}"
        result = self._exec(cmd)
        output = getattr(result, "output", "") or ""
        infos: list[FileInfo] = []
        for line in output.splitlines():
            line = line.strip()
            if line:
                infos.append(FileInfo(path=line))
        return infos

    async def aupload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload multiple files into the sandbox.

        Args:
            files: List of ``(path, content)`` tuples where content is UTF-8
                encoded bytes.

        Returns:
            List of ``FileUploadResponse`` objects in the same order as input.
        """
        responses: list[FileUploadResponse] = []
        for file_path, content in files:
            try:
                text = content.decode("utf-8")
                cmd = (
                    f"mkdir -p $(dirname {shlex.quote(file_path)}) && "
                    f"cat > {shlex.quote(file_path)} << 'GENAI_EOF'\n{text}\nGENAI_EOF"
                )
                result = self._exec(cmd)
                exit_code = getattr(result, "exit_code", 0) or 0
                if exit_code != 0:
                    raise RuntimeError(getattr(result, "output", "") or "write failed")
                responses.append(FileUploadResponse(path=file_path))
            except Exception as exc:
                logger.warning(f"upload_files failed for {file_path}: {exc}")
                responses.append(FileUploadResponse(path=file_path, error="permission_denied"))
        return responses

    async def adownload_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download multiple files from the sandbox.

        Args:
            paths: Absolute file paths to download.

        Returns:
            List of ``FileDownloadResponse`` objects in the same order as input.
        """
        responses: list[FileDownloadResponse] = []
        for file_path in paths:
            try:
                result = self._exec(f"cat {shlex.quote(file_path)}")
                exit_code = getattr(result, "exit_code", 0) or 0
                if exit_code != 0:
                    raise RuntimeError(getattr(result, "output", "") or "read failed")
                output = getattr(result, "output", "") or ""
                responses.append(FileDownloadResponse(path=file_path, content=output.encode("utf-8")))
            except Exception as exc:
                logger.warning(f"download_files failed for {file_path}: {exc}")
                responses.append(FileDownloadResponse(path=file_path, error="file_not_found"))
        return responses
