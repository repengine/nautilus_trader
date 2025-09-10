from __future__ import annotations

from pathlib import Path

from ml.cli import observability as obs_cli


def test_cli_status_no_worker(capsys: object) -> None:
    code = obs_cli.main(["status"])  # no worker started in this process
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert code == 0
    assert "async_worker_running=False" in captured.out
    assert "queue_size=0" in captured.out


def test_cli_start_async_runs_and_stops(tmp_path: Path) -> None:
    # Run async mode for a very short duration with a temp base path
    code = obs_cli.main(
        [
            "start",
            "--async",
            "--duration",
            "0.05",
            "--base-path",
            str(tmp_path),
        ],
    )
    assert code == 0
