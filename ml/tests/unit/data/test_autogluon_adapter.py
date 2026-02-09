from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import cast

import pandas as pd
import pytest

from ml.config.autogluon import AutoGluonDataConfig
from ml.data import autogluon_adapter as adapter


@dataclass(frozen=True)
class _ConfigWrapper:
    data_config: AutoGluonDataConfig

    def get_data_config(self) -> AutoGluonDataConfig:
        return self.data_config


@pytest.mark.skipif(not adapter.HAS_POLARS or adapter.pl is None, reason="Polars not available")
def test_compute_forward_return_raises_for_missing_columns() -> None:
    pl_mod = adapter.pl
    assert pl_mod is not None
    frame = pl_mod.DataFrame({"instrument_id": ["SPY"], "ts_event": [1]})

    with pytest.raises(ValueError, match="Price column"):
        adapter.compute_forward_return(frame, price_col="close")


@pytest.mark.skipif(not adapter.HAS_POLARS or adapter.pl is None, reason="Polars not available")
def test_validate_nautilus_dataset_reports_timestamp_type_errors() -> None:
    pl_mod = adapter.pl
    assert pl_mod is not None
    config = AutoGluonDataConfig()
    frame = pl_mod.DataFrame(
        {
            "instrument_id": ["SPY"],
            "ts_event": ["not-an-int"],
            "forward_return": [0.1],
        },
    )

    errors = adapter.validate_nautilus_dataset(frame, config)

    assert any("must be integer" in message for message in errors)


def test_canonicalize_timestamp_column_keeps_input_for_non_ts_event() -> None:
    frame = pd.DataFrame({"timestamp": [1], "instrument_id": ["SPY"]})

    normalized = adapter.canonicalize_timestamp_column(frame, timestamp_column="timestamp")

    assert normalized is frame


def test_canonicalize_timestamp_column_handles_pandas_columns() -> None:
    frame_dual = pd.DataFrame(
        {
            "instrument_id": ["SPY"],
            "ts_event": [1],
            "timestamp": [1],
        },
    )
    frame_alias = pd.DataFrame({"instrument_id": ["SPY"], "timestamp": [1]})

    dropped = adapter.canonicalize_timestamp_column(frame_dual, timestamp_column="ts_event")
    renamed = adapter.canonicalize_timestamp_column(frame_alias, timestamp_column="ts_event")

    assert "timestamp" not in dropped.columns
    assert "ts_event" in dropped.columns
    assert "ts_event" in renamed.columns
    assert "timestamp" not in renamed.columns


@pytest.mark.skipif(not adapter.HAS_POLARS or adapter.pl is None, reason="Polars not available")
def test_canonicalize_timestamp_column_handles_polars_columns() -> None:
    pl_mod = adapter.pl
    assert pl_mod is not None
    frame_dual = pl_mod.DataFrame(
        {
            "instrument_id": ["SPY"],
            "ts_event": [1],
            "timestamp": [1],
        },
    )
    frame_alias = pl_mod.DataFrame({"instrument_id": ["SPY"], "timestamp": [1]})

    dropped = adapter.canonicalize_timestamp_column(frame_dual, timestamp_column="ts_event")
    renamed = adapter.canonicalize_timestamp_column(frame_alias, timestamp_column="ts_event")

    assert "timestamp" not in dropped.columns
    assert "ts_event" in dropped.columns
    assert "ts_event" in renamed.columns
    assert "timestamp" not in renamed.columns


def test_canonicalize_timestamp_column_returns_passthrough_for_non_dataframe_types() -> None:
    class _FrameLike:
        columns = ("ts_event", "timestamp")

    frame_like = _FrameLike()

    result = adapter.canonicalize_timestamp_column(frame_like, timestamp_column="ts_event")

    assert result is frame_like


def test_extract_covariates_resolves_and_deduplicates_columns() -> None:
    frame = pd.DataFrame(
        {
            "instrument_id": ["SPY"],
            "ts_event": [1],
            "forward_return": [0.1],
            "known": [1.0],
            "hour": [9],
            "past_only": [2.0],
            "sector": ["equity"],
        },
    )
    config = AutoGluonDataConfig(
        known_covariates=("known", "hour_sin", "hour_cos"),
        past_covariates=("past_only",),
        static_features=("sector",),
    )

    covariates = adapter.extract_covariates(frame, config)

    assert covariates["known"] == ["known", "hour"]
    assert covariates["past"] == ["past_only"]
    assert covariates["static"] == ["sector"]


def test_validate_nautilus_dataset_reports_missing_covariates_and_static_features() -> None:
    frame = pd.DataFrame(
        {
            "instrument_id": ["SPY"],
            "ts_event": [1],
            "forward_return": [0.1],
        },
    )
    config = AutoGluonDataConfig(
        known_covariates=("hour_sin",),
        past_covariates=("ret_1",),
        static_features=("sector",),
    )

    errors = adapter.validate_nautilus_dataset(frame, config)

    assert "Missing known covariate column: hour_sin" in errors
    assert "Missing past covariate column: ret_1" in errors
    assert "Missing static feature column: sector" in errors


@pytest.mark.skipif(not adapter.HAS_POLARS or adapter.pl is None, reason="Polars not available")
def test_convert_to_timeseries_pandas_from_polars_drops_nan_targets() -> None:
    pl_mod = adapter.pl
    assert pl_mod is not None
    frame = pl_mod.DataFrame(
        {
            "instrument_id": [2, 1, 1],
            "ts_event": [3_000_000_000, 2_000_000_000, 1_000_000_000],
            "forward_return": [0.3, None, 0.1],
            "hour": [11, 10, 9],
            "sector": ["eq", "eq", "eq"],
        },
    )
    config = AutoGluonDataConfig(
        known_covariates=("hour",),
        static_features=("sector",),
    )

    converted = adapter.convert_to_timeseries_pandas(frame, config)

    assert list(converted.columns)[:3] == ["item_id", "timestamp", "target"]
    assert converted["item_id"].dtype == object
    assert converted["item_id"].tolist() == ["1", "2"]
    assert str(converted["timestamp"].dtype).startswith("datetime64")
    assert len(converted) == 2


def test_convert_to_timeseries_pandas_parses_strings_and_strips_timezone() -> None:
    frame = pd.DataFrame(
        {
            "instrument_id": [10, 10],
            "ts_event": ["2024-01-02T09:31:00Z", "2024-01-02T09:30:00Z"],
            "target": [0.2, 0.1],
        },
    )
    config = AutoGluonDataConfig(target_column="target")

    converted = adapter.convert_to_timeseries_pandas(frame, config)

    assert converted["item_id"].tolist() == ["10", "10"]
    assert str(converted["timestamp"].dtype).startswith("datetime64")
    assert converted["timestamp"].dt.tz is None
    assert converted["timestamp"].is_monotonic_increasing


@pytest.mark.skipif(not adapter.HAS_POLARS or adapter.pl is None, reason="Polars not available")
def test_convert_to_timeseries_pandas_raises_when_validation_fails() -> None:
    pl_mod = adapter.pl
    assert pl_mod is not None
    frame = pl_mod.DataFrame({"instrument_id": ["SPY"], "ts_event": [1]})

    with pytest.raises(ValueError, match="Dataset validation failed"):
        adapter.convert_to_timeseries_pandas(frame, AutoGluonDataConfig())


def test_convert_to_timeseries_pandas_raises_when_pandas_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dependency_calls: list[list[str]] = []

    def _record(deps: list[str]) -> None:
        dependency_calls.append(deps)

    monkeypatch.setattr(adapter, "HAS_PANDAS", False)
    monkeypatch.setattr(adapter, "pd", None)
    monkeypatch.setattr(adapter, "check_ml_dependencies", _record)

    with pytest.raises(ImportError, match="Pandas not available"):
        adapter.convert_to_timeseries_pandas(pd.DataFrame({"x": [1]}), AutoGluonDataConfig())

    assert dependency_calls == [["pandas"]]


@pytest.mark.skipif(not adapter.HAS_POLARS or adapter.pl is None, reason="Polars not available")
def test_compute_forward_return_raises_for_missing_item_and_timestamp_columns() -> None:
    pl_mod = adapter.pl
    assert pl_mod is not None
    missing_item = pl_mod.DataFrame({"ts_event": [1], "close": [100.0]})
    missing_ts = pl_mod.DataFrame({"instrument_id": ["SPY"], "close": [100.0]})

    with pytest.raises(ValueError, match="Item id column"):
        adapter.compute_forward_return(missing_item, price_col="close")

    with pytest.raises(ValueError, match="Timestamp column"):
        adapter.compute_forward_return(missing_ts, price_col="close")


def test_compute_forward_return_checks_polars_dependency_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dependency_calls: list[list[str]] = []

    def _record(deps: list[str]) -> None:
        dependency_calls.append(deps)

    monkeypatch.setattr(adapter, "HAS_POLARS", False)
    monkeypatch.setattr(adapter, "pl", None)
    monkeypatch.setattr(adapter, "check_ml_dependencies", _record)

    with pytest.raises(ImportError, match="Polars not available"):
        adapter.compute_forward_return(cast(Any, pd.DataFrame({"x": [1]})))

    assert dependency_calls == [["polars"]]


def test_convert_to_timeseries_dataframe_uses_timeseries_constructor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    converted = pd.DataFrame({"item_id": ["SPY"], "timestamp": [pd.Timestamp("2024-01-01")], "target": [0.1]})
    calls: list[dict[str, Any]] = []

    class _FakeTSDF:
        @staticmethod
        def from_data_frame(df: pd.DataFrame, *, id_column: str, timestamp_column: str) -> dict[str, Any]:
            calls.append(
                {
                    "df": df,
                    "id_column": id_column,
                    "timestamp_column": timestamp_column,
                },
            )
            return {"ok": True}

    monkeypatch.setattr(adapter, "TimeSeriesDataFrame", _FakeTSDF)
    monkeypatch.setattr(adapter, "convert_to_timeseries_pandas", lambda *_args, **_kwargs: converted)

    result = adapter.convert_to_timeseries_dataframe(pd.DataFrame({"x": [1]}), AutoGluonDataConfig())

    assert result == {"ok": True}
    assert calls[0]["id_column"] == "item_id"
    assert calls[0]["timestamp_column"] == "timestamp"


def test_convert_to_timeseries_dataframe_raises_when_pandas_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dependency_calls: list[list[str]] = []

    def _record(deps: list[str]) -> None:
        dependency_calls.append(deps)

    monkeypatch.setattr(adapter, "HAS_PANDAS", False)
    monkeypatch.setattr(adapter, "pd", None)
    monkeypatch.setattr(adapter, "check_ml_dependencies", _record)

    with pytest.raises(ImportError, match="Pandas not available"):
        adapter.convert_to_timeseries_dataframe(pd.DataFrame({"x": [1]}), AutoGluonDataConfig())

    assert dependency_calls == [["pandas"]]


def test_convert_to_timeseries_pandas_uses_get_data_config_when_available() -> None:
    converted_input = pd.DataFrame(
        {
            "instrument_id": ["SPY"],
            "ts_event": [1_000_000_000],
            "forward_return": [0.2],
        },
    )
    wrapped = _ConfigWrapper(data_config=AutoGluonDataConfig())

    result = adapter.convert_to_timeseries_pandas(converted_input, wrapped)

    assert list(result.columns) == ["item_id", "timestamp", "target"]


def test_convert_to_timeseries_dataframe_checks_autogluon_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dependency_calls: list[list[str]] = []

    def _record(deps: list[str]) -> None:
        dependency_calls.append(deps)

    class _FakeTSDF:
        @staticmethod
        def from_data_frame(df: pd.DataFrame, *, id_column: str, timestamp_column: str) -> pd.DataFrame:
            return df[[id_column, timestamp_column]]

    converted = pd.DataFrame({"item_id": ["SPY"], "timestamp": [pd.Timestamp("2024-01-01")], "target": [0.1]})

    monkeypatch.setattr(adapter, "HAS_AUTOGLUON", False)
    monkeypatch.setattr(adapter, "check_ml_dependencies", _record)
    monkeypatch.setattr(adapter, "TimeSeriesDataFrame", _FakeTSDF)
    monkeypatch.setattr(adapter, "convert_to_timeseries_pandas", lambda *_args, **_kwargs: converted)

    result = adapter.convert_to_timeseries_dataframe(pd.DataFrame({"x": [1]}), AutoGluonDataConfig())

    assert dependency_calls == [["autogluon"]]
    assert list(result.columns) == ["item_id", "timestamp"]
