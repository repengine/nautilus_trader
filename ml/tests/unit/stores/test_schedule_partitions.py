from __future__ import annotations

from dataclasses import dataclass
import sys
from typing import Any

import pytest

from ml.stores import schedule_partitions


@dataclass(frozen=True)
class DummyCandidates:
    """
    Minimal candidate container for connection selection.
    """

    urls: list[str]


class StubLogger:
    """
    Logger stub that records warnings for assertions.
    """

    def __init__(self) -> None:
        self.warnings: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def info(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def debug(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def warning(self, *args: Any, **kwargs: Any) -> None:
        self.warnings.append((args, kwargs))

    def error(self, *_args: Any, **_kwargs: Any) -> None:
        return None


class StubPartitionManager:
    """
    PartitionManager stub used for CLI flow tests.
    """

    last_instance: StubPartitionManager | None = None
    raise_on_run: bool = False

    def __init__(
        self,
        *,
        connection_string: str,
        months_ahead: int,
        retention_months: int,
        logger: Any,
    ) -> None:
        self.connection_string = connection_string
        self.months_ahead = months_ahead
        self.retention_months = retention_months
        self.logger = logger
        self.tables = ["ml_feature_values"]
        self.stats_calls = 0
        self.run_calls = 0
        self.__class__.last_instance = self

    def get_partition_stats(self) -> dict[str, list[dict[str, Any]]]:
        self.stats_calls += 1
        return {
            "ml_feature_values": [
                {
                    "name": "ml_feature_values_2024_01",
                    "size_bytes": 0,
                    "size": "0B",
                },
            ],
        }

    def run_maintenance(self) -> dict[str, int]:
        self.run_calls += 1
        if self.__class__.raise_on_run:
            raise RuntimeError("boom")
        return {"created": 1, "removed": 0}


def test_setup_logging_returns_named_logger(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    setup_logging should configure logging and return the expected logger name.
    """
    recorded: dict[str, str] = {}

    def fake_configure_logging(*, level: str) -> None:
        recorded["level"] = level

    monkeypatch.setattr(schedule_partitions, "configure_logging", fake_configure_logging)

    logger = schedule_partitions.setup_logging(verbose=True)

    assert recorded["level"] == "DEBUG"
    assert logger.name == "partition_scheduler"


def test_main_exits_when_no_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    main should exit early if no DB connection candidates are found.
    """
    monkeypatch.setattr(
        schedule_partitions,
        "collect_postgres_candidates",
        lambda *_args, **_kwargs: DummyCandidates(urls=[]),
    )
    monkeypatch.setattr(
        schedule_partitions,
        "setup_logging",
        lambda *_args, **_kwargs: StubLogger(),
    )
    monkeypatch.setattr(schedule_partitions, "PartitionManager", StubPartitionManager)
    monkeypatch.setattr(sys, "argv", ["schedule_partitions.py"])

    with pytest.raises(SystemExit) as exc:
        schedule_partitions.main()

    assert "No PostgreSQL connection candidates found" in str(exc.value)


def test_main_stats_only_skips_maintenance(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Stats-only mode should query stats and skip maintenance.
    """
    monkeypatch.setattr(
        schedule_partitions,
        "collect_postgres_candidates",
        lambda *_args, **_kwargs: DummyCandidates(urls=["postgresql://db"]),
    )
    monkeypatch.setattr(
        schedule_partitions,
        "select_first_working_connection",
        lambda urls: urls[0],
    )
    monkeypatch.setattr(
        schedule_partitions,
        "setup_logging",
        lambda *_args, **_kwargs: StubLogger(),
    )
    monkeypatch.setattr(schedule_partitions, "PartitionManager", StubPartitionManager)
    monkeypatch.setattr(sys, "argv", ["schedule_partitions.py", "--stats-only"])

    schedule_partitions.main()

    manager = StubPartitionManager.last_instance
    assert manager is not None
    assert manager.stats_calls == 1
    assert manager.run_calls == 0


def test_main_dry_run_skips_maintenance(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Dry-run mode should inspect stats without executing maintenance.
    """
    monkeypatch.setattr(
        schedule_partitions,
        "collect_postgres_candidates",
        lambda *_args, **_kwargs: DummyCandidates(urls=["postgresql://db"]),
    )
    monkeypatch.setattr(
        schedule_partitions,
        "select_first_working_connection",
        lambda urls: urls[0],
    )
    monkeypatch.setattr(
        schedule_partitions,
        "setup_logging",
        lambda *_args, **_kwargs: StubLogger(),
    )
    monkeypatch.setattr(schedule_partitions, "PartitionManager", StubPartitionManager)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "schedule_partitions.py",
            "--dry-run",
            "--months-ahead",
            "2",
            "--retention-months",
            "5",
        ],
    )

    schedule_partitions.main()

    manager = StubPartitionManager.last_instance
    assert manager is not None
    assert manager.months_ahead == 2
    assert manager.retention_months == 5
    assert manager.stats_calls == 1
    assert manager.run_calls == 0


def test_main_uses_first_candidate_when_probe_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    If the connectivity probe fails, main should fall back to the first candidate.
    """
    logger = StubLogger()

    def _raise_probe(_urls: list[str]) -> str:
        raise RuntimeError("probe failed")

    monkeypatch.setattr(
        schedule_partitions,
        "collect_postgres_candidates",
        lambda *_args, **_kwargs: DummyCandidates(
            urls=["postgresql://db1", "postgresql://db2"],
        ),
    )
    monkeypatch.setattr(
        schedule_partitions,
        "select_first_working_connection",
        _raise_probe,
    )
    monkeypatch.setattr(
        schedule_partitions,
        "setup_logging",
        lambda *_args, **_kwargs: logger,
    )
    monkeypatch.setattr(schedule_partitions, "PartitionManager", StubPartitionManager)
    monkeypatch.setattr(sys, "argv", ["schedule_partitions.py", "--stats-only"])

    schedule_partitions.main()

    manager = StubPartitionManager.last_instance
    assert manager is not None
    assert manager.connection_string == "postgresql://db1"
    assert len(logger.warnings) == 1


def test_main_daemon_mode_stops_on_keyboard_interrupt(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Daemon mode should exit cleanly on KeyboardInterrupt.
    """
    monkeypatch.setattr(
        schedule_partitions,
        "collect_postgres_candidates",
        lambda *_args, **_kwargs: DummyCandidates(urls=["postgresql://db"]),
    )
    monkeypatch.setattr(
        schedule_partitions,
        "select_first_working_connection",
        lambda urls: urls[0],
    )
    monkeypatch.setattr(
        schedule_partitions,
        "setup_logging",
        lambda *_args, **_kwargs: StubLogger(),
    )
    monkeypatch.setattr(schedule_partitions, "PartitionManager", StubPartitionManager)

    def _raise_keyboard_interrupt(*_args: Any, **_kwargs: Any) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(schedule_partitions.time, "sleep", _raise_keyboard_interrupt)
    monkeypatch.setattr(
        sys,
        "argv",
        ["schedule_partitions.py", "--daemon", "--interval", "1"],
    )

    schedule_partitions.main()

    manager = StubPartitionManager.last_instance
    assert manager is not None
    assert manager.run_calls == 1


def test_main_exits_nonzero_when_maintenance_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    A failing maintenance run should exit with a non-zero code.
    """
    StubPartitionManager.raise_on_run = True

    monkeypatch.setattr(
        schedule_partitions,
        "collect_postgres_candidates",
        lambda *_args, **_kwargs: DummyCandidates(urls=["postgresql://db"]),
    )
    monkeypatch.setattr(
        schedule_partitions,
        "select_first_working_connection",
        lambda urls: urls[0],
    )
    monkeypatch.setattr(
        schedule_partitions,
        "setup_logging",
        lambda *_args, **_kwargs: StubLogger(),
    )
    monkeypatch.setattr(schedule_partitions, "PartitionManager", StubPartitionManager)
    monkeypatch.setattr(sys, "argv", ["schedule_partitions.py"])

    with pytest.raises(SystemExit) as exc:
        schedule_partitions.main()

    assert exc.value.code == 1

    StubPartitionManager.raise_on_run = False
