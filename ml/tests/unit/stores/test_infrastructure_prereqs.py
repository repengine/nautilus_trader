from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import pytest

from ml.stores import infrastructure as infra


@dataclass
class FakeResult:
    """
    Simple result wrapper for scalar queries.
    """

    value: Any

    def scalar(self) -> Any:
        return self.value


class FakeConn:
    """
    Connection stub that returns queued scalar values.
    """

    def __init__(self, scalars: list[Any]) -> None:
        self._scalars = list(scalars)
        self.executed: list[str] = []

    def execute(self, stmt: Any, _params: Any | None = None) -> FakeResult:
        self.executed.append(str(stmt))
        value = self._scalars.pop(0) if self._scalars else None
        return FakeResult(value)


class FakeContext:
    """
    Context manager wrapper for FakeConn.
    """

    def __init__(self, conn: FakeConn) -> None:
        self._conn = conn

    def __enter__(self) -> FakeConn:
        return self._conn

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object | None,
    ) -> bool:
        return False


class FakeEngine:
    """
    Engine stub that yields the provided connection.
    """

    def __init__(self, conn: FakeConn) -> None:
        self._conn = conn

    def begin(self) -> FakeContext:
        return FakeContext(self._conn)


def test_check_db_prereqs_all_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    When functions and partitions exist, the preflight should be OK.
    """
    scalars = [True] * (len(infra.REQUIRED_FUNCTIONS) + len(infra.PARTITIONED_TABLES))
    conn = FakeConn(scalars)
    engine = FakeEngine(conn)

    monkeypatch.setattr(infra, "get_or_create_engine", lambda *_args, **_kwargs: engine)
    monkeypatch.setattr(infra, "_ensure_helper_functions", lambda _conn: None)

    summary = infra.check_db_prereqs("postgresql://test")

    assert summary["ok"] is True
    for fn in infra.REQUIRED_FUNCTIONS:
        assert summary[f"fn:{fn}"] is True

    today = date.today()
    suffix = f"_{today.year:04d}_{today.month:02d}"
    for table in infra.PARTITIONED_TABLES:
        assert summary[f"partition:{table}{suffix}"] is True


def test_check_db_prereqs_missing_partitions_triggers_remediation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Missing partitions should invoke PartitionManager remediation.
    """

    class StubPartitionManager:
        last_instance: StubPartitionManager | None = None

        def __init__(self, connection_string: str) -> None:
            self.connection_string = connection_string
            self.ensure_calls: list[str] = []
            self.future_calls = 0
            self.__class__.last_instance = self

        def ensure_current_partition(self, table: str) -> bool:
            self.ensure_calls.append(table)
            return True

        def create_future_partitions(self) -> int:
            self.future_calls += 1
            return 0

    scalars = [
        True,
        True,
        False,
        False,
        False,
        True,
        True,
        True,
    ]
    conn = FakeConn(scalars)
    engine = FakeEngine(conn)

    monkeypatch.setattr(infra, "get_or_create_engine", lambda *_args, **_kwargs: engine)
    monkeypatch.setattr(infra, "_ensure_helper_functions", lambda _conn: None)
    monkeypatch.setattr(infra, "PartitionManager", StubPartitionManager)

    summary = infra.check_db_prereqs("postgresql://test")

    assert summary["ok"] is False
    assert StubPartitionManager.last_instance is not None
    assert StubPartitionManager.last_instance.future_calls == 1
    assert set(StubPartitionManager.last_instance.ensure_calls) == set(infra.PARTITIONED_TABLES)

    today = date.today()
    suffix = f"_{today.year:04d}_{today.month:02d}"
    for table in infra.PARTITIONED_TABLES:
        assert summary[f"partition:{table}{suffix}"] is True


def test_run_partition_maintenance_invokes_manager(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    run_partition_maintenance should delegate to the manager.
    """

    class StubManager:
        last_instance: StubManager | None = None

        def __init__(self, connection_string: str, months_ahead: int, retention_months: int) -> None:
            self.connection_string = connection_string
            self.months_ahead = months_ahead
            self.retention_months = retention_months
            self.run_calls = 0
            self.__class__.last_instance = self

        def run_maintenance(self) -> None:
            self.run_calls += 1

    monkeypatch.setattr(infra, "PartitionManager", StubManager)

    infra.run_partition_maintenance("postgresql://test", months_ahead=2, retention_months=7)

    assert StubManager.last_instance is not None
    assert StubManager.last_instance.connection_string == "postgresql://test"
    assert StubManager.last_instance.months_ahead == 2
    assert StubManager.last_instance.retention_months == 7
    assert StubManager.last_instance.run_calls == 1


def test_partition_manager_run_maintenance_aggregates(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    run_maintenance should call subroutines and return totals.
    """
    monkeypatch.setattr(infra, "get_or_create_engine", lambda *_args, **_kwargs: object())

    manager = infra.PartitionManager("postgresql://test")

    calls: list[str] = []

    def _ensure(table: str) -> bool:
        calls.append(table)
        return True

    manager.ensure_current_partition = _ensure
    manager.create_future_partitions = lambda: 5
    manager.cleanup_old_partitions = lambda: 2

    result = manager.run_maintenance()

    assert calls == manager.tables
    assert result["created"] == 5
    assert result["removed"] == 2
    assert "timestamp" in result
