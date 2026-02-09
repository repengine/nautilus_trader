from __future__ import annotations

import builtins
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from ml.config.scheduler_config import DatabentoConfig
from ml.config.scheduler_config import SchedulerConfig
from ml.data import collection_coordinator as coordinator_module
from ml.data.collection_coordinator import CollectionCoordinator


@dataclass(slots=True)
class _DataItem:
    ts_event: int


class _MetricSpy:
    def __init__(self) -> None:
        self.label_calls: list[dict[str, object]] = []
        self.inc_calls: list[float] = []
        self.observe_calls: list[float] = []
        self.set_calls: list[float] = []

    def labels(self, **kwargs: object) -> _MetricSpy:
        self.label_calls.append(kwargs)
        return self

    def inc(self, value: float = 1.0) -> None:
        self.inc_calls.append(float(value))

    def observe(self, value: float) -> None:
        self.observe_calls.append(float(value))

    def set(self, value: float) -> None:
        self.set_calls.append(float(value))


def _build_config(
    tmp_path: Path,
    *,
    use_temporary_files: bool = True,
    api_key: str | None = "test-key",
    schema: str = "ohlcv-1m",
    max_retries: int = 2,
) -> SchedulerConfig:
    databento = DatabentoConfig(
        dataset="DBEQ.BASIC",
        schema=schema,
        stype_in="raw_symbol",
        use_temporary_files=use_temporary_files,
        temp_data_dir=str(tmp_path / "temp_dbn"),
        price_precision=2,
        api_key=api_key,
    )
    return SchedulerConfig(
        symbols=["SPY.XNAS"],
        retention_days=30,
        databento=databento,
        max_retries=max_retries,
        retry_delay_seconds=0.0,
    )


def _build_coordinator(
    tmp_path: Path,
    *,
    config: SchedulerConfig | None = None,
    data_registry: Any | None = None,
) -> tuple[CollectionCoordinator, MagicMock, MagicMock, MagicMock]:
    cfg = config or _build_config(tmp_path)
    catalog = MagicMock()
    databento_loader = MagicMock()
    registry_integrator = MagicMock()
    coordinator = CollectionCoordinator(
        catalog=catalog,
        config=cfg,
        databento_loader=databento_loader,
        registry_integrator=registry_integrator,
        data_registry=data_registry,
    )
    return coordinator, catalog, databento_loader, registry_integrator


def test_collect_latest_data_requires_api_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _build_config(tmp_path, api_key=None)
    coordinator, _, _, _ = _build_coordinator(tmp_path, config=config)
    monkeypatch.delenv("DATABENTO_API_KEY", raising=False)

    with pytest.raises(ValueError, match="DATABENTO_API_KEY"):
        coordinator.collect_latest_data(
            symbols=["SPY.XNAS"],
            target_date=datetime(2024, 1, 3, 0, 0, 0),
            run_id="run-1",
        )


def test_collect_latest_data_raises_import_error_when_databento_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator, _, _, _ = _build_coordinator(tmp_path)
    monkeypatch.setenv("DATABENTO_API_KEY", "env-key")

    original_import = builtins.__import__

    def _import(name: str, *args: object, **kwargs: object) -> Any:
        if name == "databento":
            raise ImportError("databento missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _import)

    with pytest.raises(ImportError):
        coordinator.collect_latest_data(
            symbols=["SPY.XNAS"],
            target_date=datetime(2024, 1, 3, 0, 0, 0),
            run_id="run-2",
        )


def test_collect_latest_data_counts_results_and_cleans_temp_dir(tmp_path: Path) -> None:
    config = _build_config(tmp_path, use_temporary_files=True)
    coordinator, _, _, _ = _build_coordinator(tmp_path, config=config)
    temp_dir = Path(config.databento.temp_data_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    coordinator.collect_symbol_data = MagicMock(side_effect=[True, False])

    fake_db_module = SimpleNamespace(Historical=lambda _key: object())
    with pytest.MonkeyPatch.context() as patch_ctx:
        patch_ctx.setitem(sys.modules, "databento", fake_db_module)
        collected, failed = coordinator.collect_latest_data(
            symbols=["SPY.XNAS", "QQQ.XNAS"],
            target_date=datetime(2024, 1, 3, 0, 0, 0),
            run_id="run-3",
        )

    assert (collected, failed) == (1, 1)
    assert temp_dir.exists() is False


def test_collect_latest_data_high_failure_rate_increments_rate_limit_metric(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _build_config(tmp_path, use_temporary_files=False)
    coordinator, _, _, _ = _build_coordinator(tmp_path, config=config)
    rate_limit_metric = _MetricSpy()
    monkeypatch.setattr(coordinator_module, "api_rate_limit_hits", rate_limit_metric)

    coordinator.collect_symbol_data = MagicMock(return_value=False)
    fake_db_module = SimpleNamespace(Historical=lambda _key: object())
    with pytest.MonkeyPatch.context() as patch_ctx:
        patch_ctx.setitem(sys.modules, "databento", fake_db_module)
        collected, failed = coordinator.collect_latest_data(
            symbols=["SPY.XNAS", "QQQ.XNAS", "IWM.XNAS", "AAPL.XNAS"],
            target_date=datetime(2024, 1, 3, 0, 0, 0),
            run_id="run-4",
        )

    assert (collected, failed) == (0, 4)
    assert rate_limit_metric.inc_calls == [1.0]


def test_collect_symbol_data_rejects_invalid_symbol_format(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator, _, _, _ = _build_coordinator(tmp_path)
    error_metric = _MetricSpy()
    monkeypatch.setattr(coordinator_module, "data_collection_errors_total", error_metric)

    success = coordinator.collect_symbol_data(
        client=MagicMock(),
        symbol="SPY",
        start_date=datetime(2024, 1, 3, 0, 0, 0),
        end_date=datetime(2024, 1, 3, 23, 59, 59),
        target_date=datetime(2024, 1, 3, 0, 0, 0),
        temp_data_dir=tmp_path,
        run_id="run-5",
    )

    assert success is False
    assert error_metric.label_calls[0]["error_type"] == "invalid_symbol_format"


def test_collect_symbol_data_direct_processing_path_returns_false(tmp_path: Path) -> None:
    config = _build_config(tmp_path, use_temporary_files=False)
    coordinator, _, _, _ = _build_coordinator(tmp_path, config=config)
    client = MagicMock()
    client.timeseries.get_range.return_value = MagicMock()

    success = coordinator.collect_symbol_data(
        client=client,
        symbol="SPY.XNAS",
        start_date=datetime(2024, 1, 3, 0, 0, 0),
        end_date=datetime(2024, 1, 3, 23, 59, 59),
        target_date=datetime(2024, 1, 3, 0, 0, 0),
        temp_data_dir=tmp_path,
        run_id="run-6",
    )

    assert success is False
    client.timeseries.get_range.assert_called_once()


def test_collect_symbol_data_temp_file_path_short_circuits_for_mocks(tmp_path: Path) -> None:
    coordinator, _, _, _ = _build_coordinator(tmp_path)

    response = MagicMock()

    def _to_file(path: str) -> None:
        Path(path).write_bytes(b"dbn")

    response.to_file.side_effect = _to_file
    client = MagicMock()
    client.timeseries.get_range.return_value = response
    coordinator._load_from_dbn_file = MagicMock(return_value=[MagicMock()])

    success = coordinator.collect_symbol_data(
        client=client,
        symbol="SPY.XNAS",
        start_date=datetime(2024, 1, 3, 0, 0, 0),
        end_date=datetime(2024, 1, 3, 23, 59, 59),
        target_date=datetime(2024, 1, 3, 0, 0, 0),
        temp_data_dir=tmp_path,
        run_id="run-7",
    )

    assert success is True


def test_collect_symbol_data_returns_false_when_no_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    coordinator, _, _, _ = _build_coordinator(tmp_path)
    no_data_metric = _MetricSpy()
    monkeypatch.setattr(coordinator_module, "data_collection_errors_total", no_data_metric)

    response = MagicMock()

    def _to_file(path: str) -> None:
        Path(path).write_bytes(b"dbn")

    response.to_file.side_effect = _to_file
    client = MagicMock()
    client.timeseries.get_range.return_value = response
    coordinator._load_from_dbn_file = MagicMock(return_value=[])

    success = coordinator.collect_symbol_data(
        client=client,
        symbol="SPY.XNAS",
        start_date=datetime(2024, 1, 3, 0, 0, 0),
        end_date=datetime(2024, 1, 3, 23, 59, 59),
        target_date=datetime(2024, 1, 3, 0, 0, 0),
        temp_data_dir=tmp_path,
        run_id="run-8",
    )

    assert success is False
    assert no_data_metric.label_calls[0]["error_type"] == "no_data"


def test_collect_symbol_data_retries_and_classifies_rate_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _build_config(tmp_path, max_retries=2)
    coordinator, _, _, _ = _build_coordinator(tmp_path, config=config)
    error_metric = _MetricSpy()
    rate_metric = _MetricSpy()
    monkeypatch.setattr(coordinator_module, "data_collection_errors_total", error_metric)
    monkeypatch.setattr(coordinator_module, "api_rate_limit_hits", rate_metric)

    client = MagicMock()
    client.timeseries.get_range.side_effect = RuntimeError("rate limit exceeded")

    success = coordinator.collect_symbol_data(
        client=client,
        symbol="SPY.XNAS",
        start_date=datetime(2024, 1, 3, 0, 0, 0),
        end_date=datetime(2024, 1, 3, 23, 59, 59),
        target_date=datetime(2024, 1, 3, 0, 0, 0),
        temp_data_dir=tmp_path,
        run_id="run-9",
    )

    assert success is False
    assert client.timeseries.get_range.call_count == 2
    assert all(call["error_type"] == "rate_limit" for call in error_metric.label_calls)
    assert len(rate_metric.inc_calls) >= 2


def test_collect_symbol_data_writes_and_registers_for_real_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator, _, _, _ = _build_coordinator(tmp_path)
    collected_metric = _MetricSpy()
    latency_metric = _MetricSpy()
    staleness_metric = _MetricSpy()
    monkeypatch.setattr(coordinator_module, "data_collected_total", collected_metric)
    monkeypatch.setattr(coordinator_module, "data_collection_latency", latency_metric)
    monkeypatch.setattr(coordinator_module, "data_staleness_seconds", staleness_metric)

    response = MagicMock()
    response.to_file.side_effect = lambda path: Path(path).write_bytes(b"dbn")
    client = MagicMock()
    client.timeseries.get_range.return_value = response
    coordinator._load_from_dbn_file = MagicMock(return_value=[_DataItem(ts_event=100)])
    coordinator._write_and_register = MagicMock(return_value=True)

    success = coordinator.collect_symbol_data(
        client=client,
        symbol="SPY.XNAS",
        start_date=datetime(2024, 1, 3, 0, 0, 0),
        end_date=datetime(2024, 1, 3, 23, 59, 59),
        target_date=datetime(2024, 1, 3, 0, 0, 0),
        temp_data_dir=tmp_path,
        run_id="run-9b",
    )

    assert success is True
    coordinator._write_and_register.assert_called_once()
    assert collected_metric.inc_calls == [1.0]
    assert len(latency_metric.observe_calls) == 1
    assert len(staleness_metric.set_calls) == 1


def test_collect_symbol_data_returns_false_when_write_register_fails(tmp_path: Path) -> None:
    coordinator, _, _, _ = _build_coordinator(tmp_path)
    response = MagicMock()
    response.to_file.side_effect = lambda path: Path(path).write_bytes(b"dbn")
    client = MagicMock()
    client.timeseries.get_range.return_value = response
    coordinator._load_from_dbn_file = MagicMock(return_value=[_DataItem(ts_event=100)])
    coordinator._write_and_register = MagicMock(return_value=False)

    success = coordinator.collect_symbol_data(
        client=client,
        symbol="SPY.XNAS",
        start_date=datetime(2024, 1, 3, 0, 0, 0),
        end_date=datetime(2024, 1, 3, 23, 59, 59),
        target_date=datetime(2024, 1, 3, 0, 0, 0),
        temp_data_dir=tmp_path,
        run_id="run-9c",
    )

    assert success is False


@pytest.mark.parametrize(
    ("message", "expected_status", "expected_error_type"),
    [
        ("forbidden", "401", "auth"),
        ("unexpected failure", "500", "unknown"),
    ],
)
def test_collect_symbol_data_classifies_non_rate_limit_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    message: str,
    expected_status: str,
    expected_error_type: str,
) -> None:
    config = _build_config(tmp_path, max_retries=1)
    coordinator, _, _, _ = _build_coordinator(tmp_path, config=config)
    request_metric = _MetricSpy()
    error_metric = _MetricSpy()
    monkeypatch.setattr(coordinator_module, "api_request_total", request_metric)
    monkeypatch.setattr(coordinator_module, "data_collection_errors_total", error_metric)

    client = MagicMock()
    client.timeseries.get_range.side_effect = RuntimeError(message)

    success = coordinator.collect_symbol_data(
        client=client,
        symbol="SPY.XNAS",
        start_date=datetime(2024, 1, 3, 0, 0, 0),
        end_date=datetime(2024, 1, 3, 23, 59, 59),
        target_date=datetime(2024, 1, 3, 0, 0, 0),
        temp_data_dir=tmp_path,
        run_id="run-9d",
    )

    assert success is False
    if expected_status in {"401", "500"}:
        assert request_metric.label_calls[0]["status_code"] == expected_status
    assert error_metric.label_calls[0]["error_type"] == expected_error_type


def test_collect_symbol_data_returns_false_when_no_attempts_configured(tmp_path: Path) -> None:
    config = _build_config(tmp_path, max_retries=0)
    coordinator, _, _, _ = _build_coordinator(tmp_path, config=config)

    success = coordinator.collect_symbol_data(
        client=MagicMock(),
        symbol="SPY.XNAS",
        start_date=datetime(2024, 1, 3, 0, 0, 0),
        end_date=datetime(2024, 1, 3, 23, 59, 59),
        target_date=datetime(2024, 1, 3, 0, 0, 0),
        temp_data_dir=tmp_path,
        run_id="run-9e",
    )

    assert success is False


def test_load_from_dbn_file_maps_venue_and_schema_flags(tmp_path: Path) -> None:
    config = _build_config(tmp_path, schema="ohlcv-1m")
    coordinator, _, loader, _ = _build_coordinator(tmp_path, config=config)
    loader.from_dbn_file.return_value = ["ok"]

    result = coordinator._load_from_dbn_file(
        file_path=tmp_path / "data.dbn",
        symbol_code="SPY",
        venue="XNAS",
    )

    assert result == ["ok"]
    kwargs = loader.from_dbn_file.call_args.kwargs
    assert str(kwargs["instrument_id"]) == "SPY.NASDAQ"
    assert kwargs["bars_timestamp_on_close"] is True
    assert kwargs["include_trades"] is False


def test_load_from_dbn_file_enables_trades_flag_for_trade_schema(tmp_path: Path) -> None:
    config = _build_config(tmp_path, schema="trades")
    coordinator, _, loader, _ = _build_coordinator(tmp_path, config=config)
    loader.from_dbn_file.return_value = []

    coordinator._load_from_dbn_file(
        file_path=tmp_path / "data.dbn",
        symbol_code="ES",
        venue="GLBX",
    )

    kwargs = loader.from_dbn_file.call_args.kwargs
    assert kwargs["bars_timestamp_on_close"] is False
    assert kwargs["include_trades"] is True


def test_write_and_register_success_without_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    coordinator, catalog, _, _ = _build_coordinator(tmp_path, data_registry=None)
    write_ops_metric = _MetricSpy()
    write_latency_metric = _MetricSpy()
    monkeypatch.setattr(coordinator_module, "catalog_write_operations_total", write_ops_metric)
    monkeypatch.setattr(coordinator_module, "catalog_write_latency", write_latency_metric)

    success = coordinator._write_and_register(
        data=[_DataItem(ts_event=10), _DataItem(ts_event=20)],
        symbol_code="SPY",
        venue="XNAS",
        target_date=datetime(2024, 1, 3, 0, 0, 0),
        run_id="run-10",
    )

    assert success is True
    catalog.write_data.assert_called_once()
    assert write_ops_metric.label_calls[0]["status"] == "success"
    assert len(write_latency_metric.observe_calls) == 1


def test_write_and_register_success_with_registry_emits_events(tmp_path: Path) -> None:
    data_registry = MagicMock()
    coordinator, _catalog, _loader, registry_integrator = _build_coordinator(
        tmp_path,
        data_registry=data_registry,
    )

    success = coordinator._write_and_register(
        data=[_DataItem(ts_event=10), _DataItem(ts_event=40)],
        symbol_code="SPY",
        venue="XNAS",
        target_date=datetime(2024, 1, 3, 0, 0, 0),
        run_id="run-11",
    )

    assert success is True
    registry_integrator.ensure_dataset_registered.assert_called_once()
    data_registry.emit_event.assert_called_once()
    data_registry.update_watermark.assert_called_once()


def test_write_and_register_success_when_registry_event_emit_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_registry = MagicMock()
    data_registry.emit_event.side_effect = RuntimeError("emit boom")
    coordinator, _catalog, _loader, registry_integrator = _build_coordinator(
        tmp_path,
        data_registry=data_registry,
    )
    data_events_metric = _MetricSpy()

    fake_metrics_module = SimpleNamespace(data_events_total=data_events_metric)
    with pytest.MonkeyPatch.context() as patch_ctx:
        patch_ctx.setitem(sys.modules, "ml.common.metrics", fake_metrics_module)
        success = coordinator._write_and_register(
            data=[_DataItem(ts_event=10), _DataItem(ts_event=40)],
            symbol_code="SPY",
            venue="XNAS",
            target_date=datetime(2024, 1, 3, 0, 0, 0),
            run_id="run-11b",
        )

    assert success is True
    registry_integrator.ensure_dataset_registered.assert_called_once()
    assert data_registry.emit_event.called
    assert data_events_metric.inc_calls == [1.0]


def test_write_and_register_returns_false_on_catalog_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_registry = MagicMock()
    coordinator, catalog, _, _ = _build_coordinator(tmp_path, data_registry=data_registry)
    catalog.write_data.side_effect = RuntimeError("catalog boom")
    write_ops_metric = _MetricSpy()
    monkeypatch.setattr(coordinator_module, "catalog_write_operations_total", write_ops_metric)

    success = coordinator._write_and_register(
        data=[_DataItem(ts_event=10)],
        symbol_code="SPY",
        venue="XNAS",
        target_date=datetime(2024, 1, 3, 0, 0, 0),
        run_id="run-12",
    )

    assert success is False
    assert write_ops_metric.label_calls[0]["status"] == "failure"
    assert data_registry.emit_event.called


def test_write_and_register_returns_false_on_catalog_failure_without_registry(tmp_path: Path) -> None:
    coordinator, catalog, _, _ = _build_coordinator(tmp_path, data_registry=None)
    catalog.write_data.side_effect = RuntimeError("catalog boom")

    success = coordinator._write_and_register(
        data=[_DataItem(ts_event=10)],
        symbol_code="SPY",
        venue="XNAS",
        target_date=datetime(2024, 1, 3, 0, 0, 0),
        run_id="run-13",
    )

    assert success is False


def test_write_and_register_handles_failure_event_emission_errors(tmp_path: Path) -> None:
    data_registry = MagicMock()
    data_registry.emit_event.side_effect = RuntimeError("emit failure")
    coordinator, catalog, _, _ = _build_coordinator(tmp_path, data_registry=data_registry)
    catalog.write_data.side_effect = RuntimeError("catalog boom")

    success = coordinator._write_and_register(
        data=[_DataItem(ts_event=10)],
        symbol_code="SPY",
        venue="XNAS",
        target_date=datetime(2024, 1, 3, 0, 0, 0),
        run_id="run-14",
    )

    assert success is False
