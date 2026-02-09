from __future__ import annotations

from datetime import UTC
from datetime import datetime
from unittest.mock import MagicMock
from unittest.mock import patch

import ml.common.timestamps as timestamps
import ml.observability.migrations as migrations
import pytest


class _ScalarResult:
    def __init__(self, value: object) -> None:
        self._value = value

    def scalar_one(self) -> object:
        return self._value


def _postgres_conn_with_scalars(values: list[object]) -> MagicMock:
    iterator = iter(values)
    conn = MagicMock()
    preparer = MagicMock()
    preparer.quote.side_effect = lambda ident: f'"{ident}"'
    conn.dialect.identifier_preparer = preparer

    def _execute(_stmt: object, _params: object = None) -> _ScalarResult:
        try:
            value = next(iterator)
        except StopIteration:
            value = 0
        return _ScalarResult(value)

    conn.execute.side_effect = _execute
    return conn


def _engine_for_conn(conn: MagicMock, *, dialect_name: str = "postgresql") -> MagicMock:
    engine = MagicMock()
    engine.dialect.name = dialect_name
    begin_ctx = MagicMock()
    begin_ctx.__enter__.return_value = conn
    begin_ctx.__exit__.return_value = False
    engine.begin.return_value = begin_ctx
    return engine


def _patch_fixed_now(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FixedDateTime(datetime):
        @classmethod
        def now(cls, _tz: object = None) -> _FixedDateTime:
            return cls(2025, 1, 15, tzinfo=UTC)

    monkeypatch.setattr(migrations, "datetime", _FixedDateTime)


def _identity_sanitize(value: int, *, context: str) -> int:
    del context
    return value


def test_month_bounds_returns_start_end_month() -> None:
    dt = datetime(2025, 3, 15, tzinfo=UTC)
    start, end = migrations._month_bounds(dt)
    assert start == datetime(2025, 3, 1, tzinfo=UTC)
    assert end == datetime(2025, 4, 1, tzinfo=UTC)


def test_month_bounds_rolls_over_december() -> None:
    dt = datetime(2025, 12, 31, tzinfo=UTC)
    start, end = migrations._month_bounds(dt)
    assert start == datetime(2025, 12, 1, tzinfo=UTC)
    assert end == datetime(2026, 1, 1, tzinfo=UTC)


def test_create_index_raises_for_unsupported_method() -> None:
    conn = MagicMock()
    conn.dialect.identifier_preparer.quote.side_effect = lambda ident: f'"{ident}"'
    with patch.object(conn, "execute") as execute:
        try:
            migrations._create_index(
                conn,
                "idx_bad",
                "obs_metrics",
                ("timestamp",),
                using="unknown",
            )
        except ValueError as exc:
            assert "Unsupported index method 'unknown'" in str(exc)
        else:
            raise AssertionError("Expected ValueError")
        execute.assert_not_called()


def test_create_index_executes_with_quoted_columns() -> None:
    conn = MagicMock()
    conn.dialect.identifier_preparer.quote.side_effect = lambda ident: f'"{ident}"'
    migrations._create_index(
        conn,
        "obs_metrics_timestamp_brin",
        "obs_metrics",
        ("timestamp",),
        using="brin",
    )
    _, params = conn.execute.call_args.args
    assert params["using_clause"] == " USING BRIN"
    assert params["columns_expr"] == '"timestamp"'


def test_create_partitioned_parent_executes_statement() -> None:
    conn = MagicMock()
    migrations._create_partitioned_parent(conn, "obs_metrics", "timestamp")
    assert conn.execute.call_count == 1


def test_drop_table_cascade_executes_statement() -> None:
    conn = MagicMock()
    migrations._drop_table_cascade(conn, "obs_metrics")
    _, params = conn.execute.call_args.args
    assert params["table_name"] == "obs_metrics"


def test_create_partition_executes_statement() -> None:
    conn = MagicMock()
    migrations._create_partition(
        conn,
        "obs_metrics",
        "obs_metrics_2025_01",
        100,
        200,
    )
    _, params = conn.execute.call_args.args
    assert params["partition_name"] == "obs_metrics_2025_01"
    assert params["start_bound"] == "100"
    assert params["end_bound"] == "200"


def test_apply_observability_indices_noop_for_non_postgres() -> None:
    engine = _engine_for_conn(MagicMock(), dialect_name="sqlite")
    migrations.apply_observability_indices(engine)
    engine.begin.assert_not_called()


def test_apply_observability_indices_dispatches_expected_indexes() -> None:
    conn = MagicMock()
    engine = _engine_for_conn(conn)
    with patch("ml.observability.migrations._create_index") as create_index:
        migrations.apply_observability_indices(engine)

    assert create_index.call_count == len(migrations.OBS_TABLES) + 2
    created_names = [call.args[1] for call in create_index.call_args_list]
    for table, ts_col in migrations.OBS_TABLES.items():
        assert f"{table}_{ts_col}_brin" in created_names
    assert "obs_event_correlation_instrument_ts_idx" in created_names
    assert "obs_metrics_name_ts_idx" in created_names


def test_ensure_monthly_partitions_noop_for_non_postgres() -> None:
    engine = _engine_for_conn(MagicMock(), dialect_name="sqlite")
    migrations.ensure_monthly_partitions(engine, "obs_metrics", "timestamp")
    engine.begin.assert_not_called()


def test_ensure_monthly_partitions_creates_parent_when_table_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _postgres_conn_with_scalars([False])
    engine = _engine_for_conn(conn)
    _patch_fixed_now(monkeypatch)
    monkeypatch.setattr(timestamps, "sanitize_timestamp_ns", _identity_sanitize)

    with (
        patch("ml.observability.migrations._create_partitioned_parent") as create_parent,
        patch("ml.observability.migrations._create_partition") as create_partition,
        patch("ml.observability.migrations._create_index") as create_index,
    ):
        migrations.ensure_monthly_partitions(engine, "obs_metrics", "timestamp")

    create_parent.assert_called_once_with(conn, "obs_metrics", "timestamp")
    assert create_partition.call_count == 2
    assert create_index.call_count == 2
    partition_names = [call.args[2] for call in create_partition.call_args_list]
    assert partition_names == ["obs_metrics_2025_01", "obs_metrics_2025_02"]


def test_ensure_monthly_partitions_drops_empty_non_partitioned_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _postgres_conn_with_scalars([True, False, 0])
    engine = _engine_for_conn(conn)
    _patch_fixed_now(monkeypatch)
    monkeypatch.setattr(timestamps, "sanitize_timestamp_ns", _identity_sanitize)

    with (
        patch("ml.observability.migrations._drop_table_cascade") as drop_table,
        patch("ml.observability.migrations._create_partitioned_parent") as create_parent,
        patch("ml.observability.migrations._create_partition") as create_partition,
        patch("ml.observability.migrations._create_index") as create_index,
    ):
        migrations.ensure_monthly_partitions(engine, "obs_metrics", "timestamp")

    drop_table.assert_called_once_with(conn, "obs_metrics")
    create_parent.assert_called_once_with(conn, "obs_metrics", "timestamp")
    assert create_partition.call_count == 2
    assert create_index.call_count == 2


def test_ensure_monthly_partitions_keeps_nonempty_non_partitioned_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _postgres_conn_with_scalars([True, False, 5])
    engine = _engine_for_conn(conn)
    _patch_fixed_now(monkeypatch)

    with (
        patch("ml.observability.migrations._drop_table_cascade") as drop_table,
        patch("ml.observability.migrations._create_partitioned_parent") as create_parent,
        patch("ml.observability.migrations._create_partition") as create_partition,
        patch("ml.observability.migrations._create_index") as create_index,
    ):
        migrations.ensure_monthly_partitions(engine, "obs_metrics", "timestamp")

    drop_table.assert_not_called()
    create_parent.assert_not_called()
    create_partition.assert_not_called()
    create_index.assert_not_called()


def test_ensure_monthly_partitions_uses_existing_partitioned_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _postgres_conn_with_scalars([True, True])
    engine = _engine_for_conn(conn)
    _patch_fixed_now(monkeypatch)
    monkeypatch.setattr(timestamps, "sanitize_timestamp_ns", _identity_sanitize)

    with (
        patch("ml.observability.migrations._drop_table_cascade") as drop_table,
        patch("ml.observability.migrations._create_partitioned_parent") as create_parent,
        patch("ml.observability.migrations._create_partition") as create_partition,
        patch("ml.observability.migrations._create_index") as create_index,
    ):
        migrations.ensure_monthly_partitions(engine, "obs_metrics", "timestamp")

    drop_table.assert_not_called()
    create_parent.assert_not_called()
    assert create_partition.call_count == 2
    assert create_index.call_count == 2


def test_apply_observability_monthly_partitions_noop_for_non_postgres() -> None:
    engine = _engine_for_conn(MagicMock(), dialect_name="sqlite")
    migrations.apply_observability_monthly_partitions(engine)
    engine.begin.assert_not_called()


def test_apply_observability_monthly_partitions_dispatches_tables() -> None:
    engine = _engine_for_conn(MagicMock(), dialect_name="postgresql")
    with patch("ml.observability.migrations.ensure_monthly_partitions") as ensure:
        migrations.apply_observability_monthly_partitions(engine)
    assert ensure.call_count == len(migrations.OBS_TABLES)
    dispatched = {(call.args[1], call.args[2]) for call in ensure.call_args_list}
    assert dispatched == set(migrations.OBS_TABLES.items())
