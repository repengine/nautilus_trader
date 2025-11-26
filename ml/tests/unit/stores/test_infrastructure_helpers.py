"""
Tests for store infrastructure helpers.
"""

from __future__ import annotations

import pytest

from ml.stores import infrastructure as infra


def test_ensure_partition_helpers_invokes_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    The helper refresh should call into the deployment routine with an open connection.
    """

    observed: dict[str, object] = {}

    class _Context:
        def __init__(self) -> None:
            self.connection = object()

        def __enter__(self) -> object:
            observed["conn_entered"] = self.connection
            return self.connection

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: object | None,
        ) -> bool:
            return False

    class _Engine:
        def begin(self) -> _Context:
            return _Context()

    def _fake_get_or_create_engine(dsn: str) -> _Engine:
        assert dsn == "postgresql://test"
        return _Engine()

    def _fake_helper(conn: object) -> None:
        observed["conn"] = conn

    monkeypatch.setattr(infra, "get_or_create_engine", _fake_get_or_create_engine)
    monkeypatch.setattr(infra, "_ensure_helper_functions", _fake_helper)

    infra.ensure_partition_helpers("postgresql://test")

    assert observed["conn"] is observed["conn_entered"]


def test_ensure_partition_helpers_raises_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Deploy failures should bubble up so orchestrator bootstrap can fail fast.
    """

    class _Context:
        def __enter__(self) -> object:
            return object()

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: object | None,
        ) -> bool:
            return False

    class _Engine:
        def begin(self) -> _Context:
            return _Context()

    def _fake_get_or_create_engine(dsn: str) -> _Engine:
        assert dsn == "postgresql://test"
        return _Engine()

    def _fake_helper(_: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(infra, "get_or_create_engine", _fake_get_or_create_engine)
    monkeypatch.setattr(infra, "_ensure_helper_functions", _fake_helper)

    with pytest.raises(RuntimeError, match="boom"):
        infra.ensure_partition_helpers("postgresql://test")
