from __future__ import annotations

import sys
from datetime import UTC
from datetime import datetime
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from ml.data.ingest.databento_adapter import DatabentoAPIClient


class _PolicyStub:
    def __init__(
        self,
        *,
        filtered_symbols: list[str] | None = None,
        clamp_start: datetime | None = None,
        clamp_end: datetime | None = None,
    ) -> None:
        self._filtered_symbols = filtered_symbols
        self._clamp_start = clamp_start
        self._clamp_end = clamp_end
        self.validated: list[tuple[str, str]] = []

    def validate_dataset_schema(self, *, dataset: str, schema: str) -> None:
        self.validated.append((dataset, schema))

    def filter_symbols(self, symbols: list[str]) -> list[str]:
        return list(symbols) if self._filtered_symbols is None else list(self._filtered_symbols)

    def clamp_range(
        self,
        start: datetime,
        end: datetime,
        *,
        dataset: str | None = None,
        schema: str | None = None,
    ) -> tuple[datetime, datetime]:
        del dataset, schema
        return self._clamp_start or start, self._clamp_end or end


class _FakeResult:
    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame

    def to_df(self) -> pd.DataFrame:
        return self._frame.copy()


class _TimeseriesStub:
    def __init__(self, result: Any) -> None:
        self._result = result
        self.calls: list[dict[str, Any]] = []

    def get_range(self, **kwargs: Any) -> Any:
        self.calls.append(dict(kwargs))
        return self._result


def _build_client(
    monkeypatch: pytest.MonkeyPatch,
    *,
    result: Any,
    metadata: Any = None,
    symbology: Any = None,
) -> tuple[DatabentoAPIClient, _TimeseriesStub]:
    timeseries = _TimeseriesStub(result)
    fake_historical = SimpleNamespace(
        timeseries=timeseries,
        metadata={"status": "ok"} if metadata is None else metadata,
        symbology=symbology,
    )
    fake_module = SimpleNamespace(Historical=lambda _api_key: fake_historical)
    monkeypatch.setitem(sys.modules, "databento", fake_module)
    client = DatabentoAPIClient(api_key="test-key")
    return client, timeseries


def test_post_init_raises_runtime_error_on_invalid_databento_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "databento", SimpleNamespace())
    with pytest.raises(RuntimeError):
        DatabentoAPIClient(api_key="bad")


def test_get_data_returns_empty_for_empty_symbols(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _ = _build_client(monkeypatch, result=[])
    frame = client.get_data(
        dataset="XNAS.ITCH",
        symbols=[],
        schema="trades",
        start="2025-01-01T00:00:00",
        end="2025-01-01T01:00:00",
    )
    assert frame.empty


def test_get_data_returns_empty_when_policy_filters_all_symbols(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _PolicyStub(filtered_symbols=[])
    client, _ = _build_client(monkeypatch, result=[])
    monkeypatch.setattr("ml.data.ingest.databento_adapter.schema_spec_for", lambda _schema: None)
    monkeypatch.setattr(
        "ml.data.ingest.databento_adapter.DatabentoCoveragePolicy.from_env",
        lambda: policy,
    )
    frame = client.get_data(
        dataset="XNAS.ITCH",
        symbols=["AAPL"],
        schema="trades",
        start="2025-01-01T00:00:00",
        end="2025-01-01T01:00:00",
    )
    assert frame.empty
    assert policy.validated == [("XNAS.ITCH", "trades")]


def test_get_data_maps_quote_and_trade_columns_and_sets_ts_init(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = [
        {
            "ts": "2025-01-01T00:00:00Z",
            "bid_px": 100.0,
            "ask_px": 100.5,
            "bid_sz": 12,
            "ask_sz": 15,
            "price": 100.25,
        },
    ]
    policy = _PolicyStub()
    client, timeseries = _build_client(monkeypatch, result=payload)
    monkeypatch.setattr("ml.data.ingest.databento_adapter.schema_spec_for", lambda _schema: None)
    monkeypatch.setattr(
        "ml.data.ingest.databento_adapter.DatabentoCoveragePolicy.from_env",
        lambda: policy,
    )

    frame = client.get_data(
        dataset="XNAS.ITCH",
        symbols=["AAPL"],
        schema="tbbo-trades",
        start="2025-01-01T00:00:00",
        end="2025-01-01T01:00:00",
    )

    assert len(timeseries.calls) == 1
    call = timeseries.calls[0]
    assert call["symbols"] == "AAPL"
    assert call["start"].endswith("+00:00")
    assert call["end"].endswith("+00:00")
    assert frame["bid"].iloc[0] == pytest.approx(100.0)
    assert frame["ask"].iloc[0] == pytest.approx(100.5)
    assert frame["bid_size"].iloc[0] == 12
    assert frame["ask_size"].iloc[0] == 15
    assert frame["last"].iloc[0] == pytest.approx(100.25)
    assert frame["trade_count"].iloc[0] == 1
    assert pd.api.types.is_integer_dtype(frame["ts_event"])
    assert pd.api.types.is_integer_dtype(frame["ts_init"])


def test_get_data_normalizes_datetime_index_and_non_integer_ts_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = pd.DatetimeIndex(
        [datetime(2025, 1, 1, tzinfo=UTC), datetime(2025, 1, 1, 0, 1, tzinfo=UTC)],
        name="ts",
    )
    frame_with_index = pd.DataFrame({"price": [1.0, 1.1]}, index=index)
    policy = _PolicyStub()
    client, _ = _build_client(monkeypatch, result=_FakeResult(frame_with_index))
    monkeypatch.setattr("ml.data.ingest.databento_adapter.schema_spec_for", lambda _schema: None)
    monkeypatch.setattr(
        "ml.data.ingest.databento_adapter.DatabentoCoveragePolicy.from_env",
        lambda: policy,
    )

    normalized = client.get_data(
        dataset="XNAS.ITCH",
        symbols=["AAPL"],
        schema="quotes",
        start="2025-01-01T00:00:00",
        end="2025-01-01T01:00:00",
    )
    assert pd.api.types.is_integer_dtype(normalized["ts_event"])
    assert pd.api.types.is_integer_dtype(normalized["ts_init"])

    payload = pd.DataFrame(
        {
            "ts_event": ["1704067200000000000"],
            "ts_init": ["1704067200000000001"],
        },
    )
    numeric_client, _ = _build_client(monkeypatch, result=_FakeResult(payload))
    monkeypatch.setattr(
        "ml.data.ingest.databento_adapter.DatabentoCoveragePolicy.from_env",
        lambda: _PolicyStub(),
    )
    numeric = numeric_client.get_data(
        dataset="XNAS.ITCH",
        symbols=["MSFT"],
        schema="trades",
        start="2025-01-01T00:00:00",
        end="2025-01-01T01:00:00",
    )
    assert pd.api.types.is_integer_dtype(numeric["ts_event"])
    assert pd.api.types.is_integer_dtype(numeric["ts_init"])


def test_metadata_and_symbology_properties_expose_underlying_clients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = {"meta": "value"}
    symbology = {"sym": "client"}
    client, _ = _build_client(
        monkeypatch,
        result=[],
        metadata=metadata,
        symbology=symbology,
    )
    assert client.metadata_client is metadata
    assert client.symbology_client is symbology


def test_get_data_handles_datetime_ts_column_and_invalid_timestamp_conversions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("ml.data.ingest.databento_adapter.schema_spec_for", lambda _schema: None)
    monkeypatch.setattr(
        "ml.data.ingest.databento_adapter.DatabentoCoveragePolicy.from_env",
        lambda: _PolicyStub(),
    )

    datetime_payload = pd.DataFrame(
        {
            "ts": pd.to_datetime(["2025-01-01T00:00:00Z"], utc=True),
            "price": [10.0],
        },
    )
    datetime_client, _ = _build_client(monkeypatch, result=_FakeResult(datetime_payload))
    datetime_frame = datetime_client.get_data(
        dataset="XNAS.ITCH",
        symbols=["NVDA"],
        schema="trades",
        start=datetime(2025, 1, 1, tzinfo=UTC),
        end=datetime(2025, 1, 1, 1, tzinfo=UTC),
    )
    assert pd.api.types.is_integer_dtype(datetime_frame["ts_event"])

    invalid_ts_payload = pd.DataFrame({"ts": [object()], "price": [11.0]})
    invalid_ts_client, _ = _build_client(monkeypatch, result=_FakeResult(invalid_ts_payload))
    invalid_ts_frame = invalid_ts_client.get_data(
        dataset="XNAS.ITCH",
        symbols=["NVDA"],
        schema="trades",
        start=datetime(2025, 1, 1, tzinfo=UTC),
        end=datetime(2025, 1, 1, 1, tzinfo=UTC),
    )
    assert "ts_event" not in invalid_ts_frame.columns

    invalid_event_payload = pd.DataFrame({"ts_event": ["bad-value"], "ts_init": ["bad-init"]})
    invalid_event_client, _ = _build_client(monkeypatch, result=_FakeResult(invalid_event_payload))
    invalid_event_frame = invalid_event_client.get_data(
        dataset="XNAS.ITCH",
        symbols=["NVDA"],
        schema="trades",
        start=datetime(2025, 1, 1, tzinfo=UTC),
        end=datetime(2025, 1, 1, 1, tzinfo=UTC),
    )
    assert invalid_event_frame["ts_event"].iloc[0] == "bad-value"
    assert invalid_event_frame["ts_init"].iloc[0] == "bad-init"


def test_get_data_converts_ts_init_datetime_when_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = pd.DataFrame(
        {
            "ts_event": [1704067200000000000],
            "ts_init": ["2025-01-01T00:00:00Z"],
        },
    )
    monkeypatch.setattr("ml.data.ingest.databento_adapter.schema_spec_for", lambda _schema: None)
    monkeypatch.setattr(
        "ml.data.ingest.databento_adapter.DatabentoCoveragePolicy.from_env",
        lambda: _PolicyStub(),
    )
    client, _ = _build_client(monkeypatch, result=_FakeResult(payload))
    converted = client.get_data(
        dataset="XNAS.ITCH",
        symbols=["AAPL"],
        schema="trades",
        start=datetime(2025, 1, 1, tzinfo=UTC),
        end=datetime(2025, 1, 1, 1, tzinfo=UTC),
    )
    assert pd.api.types.is_integer_dtype(converted["ts_init"])
