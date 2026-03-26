"""External tool management: dependency checking and safe subprocess execution."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


class ToolError(RuntimeError):
    """Raised when an external tool is missing or fails."""
    pass


def check_tool(name: str, min_version: str | None = None) -> str:
    """Check that an external tool is available on PATH.

    Returns the full path to the tool.
    Raises ToolError if not found.
    """
    path = shutil.which(name)
    if path is None:
        raise ToolError(
            f"Required tool '{name}' not found on PATH. "
            f"Please install it and ensure it is accessible."
        )
    return path


def check_all_tools(tools: list[str]) -> dict[str, str]:
    """Check all required tools, report all missing at once."""
    found = {}
    missing = []
    for tool in tools:
        path = shutil.which(tool)
        if path:
            found[tool] = path
        else:
            missing.append(tool)

    if missing:
        raise ToolError(
            f"Missing required tools: {', '.join(missing)}. "
            f"Please install them and ensure they are on PATH."
        )
    return found


def run_cmd(
    cmd: list[str],
    description: str = "",
    check: bool = True,
    timeout: int | None = 600,
    capture: bool = True,
) -> subprocess.CompletedProcess:
    """Run a subprocess with proper error handling and logging.

    Args:
        cmd: Command and arguments
        description: Human-readable description for error messages
        check: Raise on non-zero exit code
        timeout: Timeout in seconds (None for no timeout)
        capture: Capture stdout/stderr

    Returns:
        CompletedProcess result

    Raises:
        ToolError: If the command fails or times out
    """
    desc = description or cmd[0]
    logger.debug("Running: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            check=check,
            capture_output=capture,
            text=capture,
            timeout=timeout,
        )
        return result
    except FileNotFoundError:
        raise ToolError(
            f"{desc}: command '{cmd[0]}' not found. "
            f"Please install it and ensure it is on PATH."
        )
    except subprocess.CalledProcessError as e:
        stderr = e.stderr[:2000] if e.stderr else "(no stderr)"
        raise ToolError(
            f"{desc} failed (exit code {e.returncode}):\n{stderr}"
        )
    except subprocess.TimeoutExpired:
        raise ToolError(
            f"{desc} timed out after {timeout}s. "
            f"Try increasing --threads or reducing input size."
        )


def run_pipeline(
    cmds: list[list[str]],
    output_path: Path | None = None,
    description: str = "",
    timeout: int = 600,
) -> None:
    """Run a pipeline of commands (cmd1 | cmd2 | ... > output).

    Args:
        cmds: List of commands to pipe together
        output_path: File to write final stdout to (None for discard)
        description: Human-readable description for errors
        timeout: Timeout in seconds
    """
    desc = description or f"pipeline ({' | '.join(c[0] for c in cmds)})"

    try:
        procs = []
        for i, cmd in enumerate(cmds):
            stdin = procs[-1].stdout if procs else None
            if i == len(cmds) - 1 and output_path:
                stdout = open(output_path, "wb")
            elif i < len(cmds) - 1:
                stdout = subprocess.PIPE
            else:
                stdout = subprocess.PIPE

            proc = subprocess.Popen(
                cmd, stdin=stdin, stdout=stdout, stderr=subprocess.PIPE,
            )
            # Close upstream stdout so it can receive SIGPIPE
            if procs:
                procs[-1].stdout.close()
            procs.append(proc)

        # Wait for all processes
        for proc in procs:
            proc.wait(timeout=timeout)

        # Close output file if opened
        if output_path and hasattr(procs[-1].stdout, 'close'):
            pass  # stdout was the file, already closed by Popen

        # Check exit codes
        for i, proc in enumerate(procs):
            if proc.returncode != 0:
                stderr = proc.stderr.read().decode() if proc.stderr else ""
                raise ToolError(
                    f"{desc}: step {i+1} ({cmds[i][0]}) failed "
                    f"(exit code {proc.returncode}):\n{stderr[:2000]}"
                )

    except FileNotFoundError as e:
        raise ToolError(f"{desc}: {e}")
    except subprocess.TimeoutExpired:
        for proc in procs:
            proc.kill()
        raise ToolError(f"{desc} timed out after {timeout}s")
