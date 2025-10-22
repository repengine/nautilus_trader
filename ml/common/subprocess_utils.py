"""Typed helpers for safe subprocess execution."""

from __future__ import annotations

import inspect
import logging
import os
import shlex
import shutil
import subprocess  # nosec: B404 - centralized wrapper enforces safe subprocess usage without shell
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SubprocessExecutionError(RuntimeError):
    """Raised when a subprocess fails or cannot be invoked safely."""

    command: tuple[str, ...]
    returncode: int
    stdout: str | bytes | None
    stderr: str | bytes | None

    def __post_init__(self) -> None:
        if not self.command:
            raise ValueError("command must not be empty")

    def __str__(self) -> str:
        cmd_display = shlex.join(self.command)
        return f"SubprocessExecutionError(command={cmd_display}, returncode={self.returncode})"


def _resolve_command(cmd: Sequence[str]) -> list[str]:
    """
    Resolve the executable path and validate the command.
    """
    if not cmd:
        raise ValueError("Command sequence is empty")

    resolved = list(cmd)
    executable = resolved[0]
    if os.path.isabs(executable) or executable.startswith("."):
        return resolved
    located = shutil.which(executable)
    if located is None:
        raise FileNotFoundError(f"Executable '{executable}' not found on PATH")
    return resolved


def run_command(
    cmd: Sequence[str],
    *,
    timeout: float | None = None,
    env: Mapping[str, str] | None = None,
    cwd: str | os.PathLike[str] | None = None,
    capture_output: bool = False,
    text: bool = True,
    check: bool = True,
    log: logging.Logger | None = None,
    stdout: Any | None = None,
    stderr: Any | None = None,
    merge_stderr: bool = False,
    **kwargs: Any,
) -> subprocess.CompletedProcess[str] | subprocess.CompletedProcess[bytes]:
    """
    Execute a subprocess with consistent validation and logging.
    """
    if "shell" in kwargs:
        raise ValueError("shell=True is not permitted; provide an explicit command sequence instead")
    effective_log = log or logger
    try:
        resolved_cmd = _resolve_command(cmd)
    except (FileNotFoundError, OSError) as exc:
        effective_log.error(
            "subprocess_executable_missing command=%s",
            shlex.join(cmd),
            exc_info=True,
        )
        raise SubprocessExecutionError(
            command=tuple(cmd),
            returncode=-1,
            stdout=None,
            stderr=str(exc),
        ) from exc
    effective_log.debug(
        "subprocess.run command=%s timeout=%s cwd=%s capture_output=%s check=%s",
        shlex.join(resolved_cmd),
        timeout,
        cwd,
        capture_output,
        check,
    )
    try:
        run_kwargs: dict[str, Any] = {
            "check": check,
            "text": text,
        }
        if timeout is not None:
            run_kwargs["timeout"] = timeout
        if env is not None:
            run_kwargs["env"] = dict(env)
        if cwd is not None:
            run_kwargs["cwd"] = cwd
        if capture_output:
            run_kwargs["capture_output"] = True
        if stdout is not None:
            run_kwargs["stdout"] = stdout
        if merge_stderr:
            run_kwargs["stderr"] = subprocess.STDOUT
        elif stderr is not None:
            run_kwargs["stderr"] = stderr

        if "shell" in kwargs:
            # Guard against shell=True slipping through when subprocess signature inspection changes.
            raise ValueError("shell parameter is not supported; commands must be provided as sequences")
        run_kwargs.update(kwargs)

        try:
            sig = inspect.signature(subprocess.run)
        except (ValueError, TypeError):
            filtered_kwargs = run_kwargs
        else:
            has_var_kw = any(
                param.kind is inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()
            )
            if has_var_kw:
                filtered_kwargs = run_kwargs
            else:
                allowed_keys = {
                    name
                    for name, param in sig.parameters.items()
                    if param.kind
                    in (
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        inspect.Parameter.KEYWORD_ONLY,
                    )
                }
                filtered_kwargs = {key: value for key, value in run_kwargs.items() if key in allowed_keys}

        completed = subprocess.run(  # nosec: B603 - command validated and shell usage explicitly prohibited
            resolved_cmd,
            **filtered_kwargs,
        )
        return completed
    except subprocess.CalledProcessError as exc:
        effective_log.error(
            "subprocess_failed command=%s returncode=%s",
            shlex.join(resolved_cmd),
            exc.returncode,
            exc_info=True,
        )
        raise SubprocessExecutionError(
            command=tuple(resolved_cmd),
            returncode=exc.returncode,
            stdout=exc.stdout,
            stderr=exc.stderr,
        ) from exc
    except (FileNotFoundError, OSError) as exc:
        effective_log.error(
            "subprocess_executable_missing command=%s",
            shlex.join(resolved_cmd),
            exc_info=True,
        )
        raise SubprocessExecutionError(
            command=tuple(resolved_cmd),
            returncode=-1,
            stdout=None,
            stderr=str(exc),
        ) from exc
