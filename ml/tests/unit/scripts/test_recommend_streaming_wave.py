from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from ml.common.subprocess_utils import SubprocessExecutionError
from ml.scripts import recommend_streaming_wave as script
from ml.training.event_driven.wave_planner import WaveBounds
from ml.training.event_driven.wave_planner import WaveRecommendation


def _recommendation() -> WaveRecommendation:
    current = WaveBounds(
        shard_row_budget=120_000,
        max_total_rows=120_000,
        max_total_sequences=90_000,
        max_shards=32,
    )
    proposed = WaveBounds(
        shard_row_budget=150_000,
        max_total_rows=150_000,
        max_total_sequences=108_000,
        max_shards=40,
    )
    return WaveRecommendation(
        current=current,
        proposed=proposed,
        notes=(),
        warnings=(),
    )


def test_main_runs_validate_wave_when_requested(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    commands: list[tuple[str, ...]] = []

    def fake_run_command(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        commands.append(tuple(cmd))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(script, "run_command", fake_run_command)
    monkeypatch.setattr(script, "summarize_manifests", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(script, "recommend_next_wave", lambda *_args, **_kwargs: _recommendation())

    exit_code = script.main(
        [
            "--manifest-dir",
            str(tmp_path),
            "--run-validate-wave",
            "--validate-wave-args",
            "--manifest-limit 3",
        ],
    )

    assert exit_code == 0
    assert commands == [
        (
            "poetry",
            "run",
            "python",
            "-m",
            "ml.scripts.validate_wave",
            "--manifest-dir",
            str(tmp_path),
            "--manifest-limit",
            "3",
        ),
    ]


def test_main_propagates_validate_wave_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run_command(_cmd: list[str]) -> None:
        raise SubprocessExecutionError(
            command=("poetry",),
            returncode=4,
            stdout=None,
            stderr=None,
        )

    monkeypatch.setattr(script, "run_command", fake_run_command)
    monkeypatch.setattr(script, "summarize_manifests", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(script, "recommend_next_wave", lambda *_args, **_kwargs: _recommendation())

    exit_code = script.main(
        [
            "--manifest-dir",
            str(tmp_path),
            "--run-validate-wave",
        ],
    )

    assert exit_code == 4
