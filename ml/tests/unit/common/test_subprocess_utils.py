from __future__ import annotations

import sys

import pytest

from ml.common.subprocess_utils import SubprocessExecutionError
from ml.common.subprocess_utils import run_command


def test_run_command_capture_output() -> None:
    proc = run_command(
        [sys.executable, "-c", "print('ok')"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert (proc.stdout or "").strip() == "ok"


def test_run_command_failure_without_check() -> None:
    proc = run_command(
        [sys.executable, "-c", "import sys; sys.exit(5)"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 5
    assert (proc.stdout or "") == ""


def test_run_command_failure_raises_with_check() -> None:
    with pytest.raises(SubprocessExecutionError) as exc_info:
        run_command([sys.executable, "-c", "import sys; sys.exit(3)"])
    err = exc_info.value
    assert err.returncode == 3
    assert len(err.command) >= 2


def test_run_command_missing_binary() -> None:
    with pytest.raises(SubprocessExecutionError) as exc_info:
        run_command(["command-that-does-not-exist-12345"])
    err = exc_info.value
    assert err.returncode == -1
    assert "command-that-does-not-exist-12345" in err.command[0]
