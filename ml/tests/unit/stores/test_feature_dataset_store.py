from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from ml.stores import feature_dataset_store as fds


def _make_store(monkeypatch: pytest.MonkeyPatch) -> fds.FeatureDatasetStore:
    dummy_engine = object()
    monkeypatch.setattr(fds, "get_or_create_engine", lambda *_args, **_kwargs: dummy_engine)
    return fds.FeatureDatasetStore("postgresql://test")


def test_frame_to_records_supports_common_inputs() -> None:
    """
    Frame coercion should accept lists, pandas, and polars-like inputs.
    """

    class _PolarsLike:
        def __init__(self, rows: list[dict[str, Any]]) -> None:
            self._rows = rows

        def to_dicts(self) -> list[dict[str, Any]]:
            return self._rows

    rows = [{"a": 1}, {"a": 2}]

    assert fds.FeatureDatasetStore._frame_to_records(rows) == rows

    df = pd.DataFrame(rows)
    assert fds.FeatureDatasetStore._frame_to_records(df) == rows

    polars_like = _PolarsLike(rows)
    assert fds.FeatureDatasetStore._frame_to_records(polars_like) == rows

    with pytest.raises(TypeError):
        fds.FeatureDatasetStore._frame_to_records(object())


def test_write_macro_releases_filters_and_coerces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Macro release ingestion should drop invalid rows and coerce values.
    """
    store = _make_store(monkeypatch)
    monkeypatch.setattr(fds, "sanitize_timestamp_ns", lambda value, context: 123)

    captured: dict[str, Any] = {}

    def _capture_bulk_upsert(*, table: Any, records: list[dict[str, Any]], conflict_cols: Any) -> int:
        captured["table"] = table.name
        captured["records"] = records
        captured["conflict_cols"] = tuple(conflict_cols)
        return len(records)

    monkeypatch.setattr(store, "_bulk_upsert", _capture_bulk_upsert)

    frame = [
        {
            "series_id": "GDP",
            "observation_ts": "100",
            "release_ts": "200",
            "release_end_ts": "300",
            "value": "1.5",
            "source": None,
            "run_id": 7,
        },
        {"series_id": "BAD", "observation_ts": None, "release_ts": None},
    ]

    count = store.write_macro_releases(frame)

    assert count == 1
    assert captured["table"] == "macro_release_calendar"
    assert captured["conflict_cols"] == (
        "series_id",
        "observation_ts",
        "release_ts",
        "ts_event",
    )

    record = captured["records"][0]
    assert record["series_id"] == "GDP"
    assert record["observation_ts"] == 100
    assert record["release_ts"] == 200
    assert record["release_end_ts"] == 300
    assert record["value"] == 1.5
    assert record["ts_event"] == 200
    assert record["ts_init"] == 123
    assert record["source"] == ""
    assert record["run_id"] == "7"


def test_write_events_calendar_normalizes_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Events ingestion should coerce metadata and default instrument IDs.
    """
    store = _make_store(monkeypatch)
    monkeypatch.setattr(fds, "sanitize_timestamp_ns", lambda value, context: 555)

    captured: dict[str, Any] = {}

    def _capture_bulk_upsert(*, table: Any, records: list[dict[str, Any]], conflict_cols: Any) -> int:
        captured["table"] = table.name
        captured["records"] = records
        return len(records)

    monkeypatch.setattr(store, "_bulk_upsert", _capture_bulk_upsert)

    frame = [
        {
            "event_timestamp": "100",
            "event_type": "FOMC",
            "name": "Meeting",
            "instrument_id": None,
            "importance": "high",
            "source": "fed",
            "metadata": '{"foo": 1}',
        },
        {"event_timestamp": None, "event_type": "", "name": "skip"},
    ]

    count = store.write_events_calendar(frame)

    assert count == 1
    assert captured["table"] == "events_calendar"

    record = captured["records"][0]
    assert record["instrument_id"] == ""
    assert record["metadata"] == {"foo": 1}
    assert record["ts_event"] == 100
    assert record["ts_init"] == 555


def test_write_micro_features_filters_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Time-series feature ingestion should drop unknown columns.
    """
    store = _make_store(monkeypatch)
    monkeypatch.setattr(fds, "sanitize_timestamp_ns", lambda value, context: 999)

    captured: dict[str, Any] = {}

    def _capture_bulk_upsert(*, table: Any, records: list[dict[str, Any]], conflict_cols: Any) -> int:
        captured["table"] = table.name
        captured["records"] = records
        return len(records)

    monkeypatch.setattr(store, "_bulk_upsert", _capture_bulk_upsert)

    frame = [
        {
            "timestamp": "123",
            "instrument_id": "EUR/USD.SIM",
            "midprice": 1.2,
            "spread_bps": 0.1,
            "extra_col": 5,
        },
        {"timestamp": None, "instrument_id": "EUR/USD.SIM"},
    ]

    count = store.write_micro_features(frame)

    assert count == 1
    assert captured["table"] == "microstructure_minute"

    record = captured["records"][0]
    assert record["instrument_id"] == "EUR/USD.SIM"
    assert record["timestamp"] == 123
    assert record["ts_event"] == 123
    assert record["ts_init"] == 999
    assert record["midprice"] == 1.2
    assert record["spread_bps"] == 0.1
    assert "extra_col" not in record


def test_write_l2_features_includes_depth_columns_and_conflicts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    L2 ingestion should keep depth columns and track conflict keys.
    """
    store = _make_store(monkeypatch)
    monkeypatch.setattr(fds, "sanitize_timestamp_ns", lambda value, context: 321)

    captured: dict[str, Any] = {}

    def _capture_bulk_upsert(*, table: Any, records: list[dict[str, Any]], conflict_cols: Any) -> int:
        captured["table"] = table.name
        captured["records"] = records
        captured["conflict_cols"] = tuple(conflict_cols)
        return len(records)

    monkeypatch.setattr(store, "_bulk_upsert", _capture_bulk_upsert)

    frame = [
        {
            "timestamp": "123",
            "instrument_id": "SPY.XNAS",
            "midprice": 100.1,
            "depth_imbalance_top1": 0.1,
            "dwp_bps_top3": 0.2,
            "bid_slope_top5": 0.3,
            "ask_slope_top10": 0.4,
            "extra_col": 9,
        },
        {"timestamp": None, "instrument_id": "SPY.XNAS"},
    ]

    count = store.write_l2_features(frame)

    assert count == 1
    assert captured["table"] == "l2_minute"
    assert captured["conflict_cols"] == ("instrument_id", "timestamp", "ts_event")

    record = captured["records"][0]
    assert record["instrument_id"] == "SPY.XNAS"
    assert record["timestamp"] == 123
    assert record["ts_event"] == 123
    assert record["ts_init"] == 321
    assert record["depth_imbalance_top1"] == 0.1
    assert record["dwp_bps_top3"] == 0.2
    assert record["bid_slope_top5"] == 0.3
    assert record["ask_slope_top10"] == 0.4
    assert "extra_col" not in record


def test_bulk_upsert_excludes_conflict_columns(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Bulk upsert should avoid updating conflict key columns.
    """
    store = _make_store(monkeypatch)

    class _Conn:
        def __init__(self) -> None:
            self.executed: list[object] = []

        def execute(self, stmt: object) -> None:
            self.executed.append(stmt)

    class _Begin:
        def __init__(self, conn: _Conn) -> None:
            self._conn = conn

        def __enter__(self) -> _Conn:
            return self._conn

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    class _Engine:
        def __init__(self) -> None:
            self.conn = _Conn()

        def begin(self) -> _Begin:
            return _Begin(self.conn)

    store._engine = _Engine()

    class _InsertStub:
        def __init__(self, table: Any) -> None:
            self.table = table
            self.excluded = type(
                "Excluded",
                (),
                {col.name: f"excluded.{col.name}" for col in table.columns},
            )()
            self.records: list[dict[str, Any]] | None = None
            self.index_elements: list[Any] | None = None
            self.set_: dict[str, object] | None = None

        def values(self, records: list[dict[str, Any]]):
            self.records = records
            return self

        def on_conflict_do_update(self, *, index_elements: list[Any], set_: dict[str, object]):
            self.index_elements = index_elements
            self.set_ = set_
            return self

    holder: dict[str, _InsertStub] = {}

    def _pg_insert(table: Any) -> _InsertStub:
        stub = _InsertStub(table)
        holder["stub"] = stub
        return stub

    monkeypatch.setattr(fds, "pg_insert", _pg_insert)

    table = fds.Table(
        "l2_minute",
        fds.MetaData(),
        fds.Column("instrument_id", fds.VARCHAR(32), primary_key=True),
        fds.Column("timestamp", fds.BIGINT, primary_key=True),
        fds.Column("ts_event", fds.BIGINT, primary_key=True),
        fds.Column("midprice", fds.DOUBLE_PRECISION),
    )

    records = [
        {"instrument_id": "SPY", "timestamp": 1, "ts_event": 1, "midprice": 10.0},
    ]
    count = store._bulk_upsert(
        table=table,
        records=records,
        conflict_cols=("instrument_id", "timestamp", "ts_event"),
    )

    assert count == 1
    stub = holder["stub"]
    assert [col.name for col in (stub.index_elements or [])] == [
        "instrument_id",
        "timestamp",
        "ts_event",
    ]
    assert stub.set_ == {"midprice": "excluded.midprice"}
