from __future__ import annotations

import contextlib
import os
import sys
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from typing import Callable
from unittest.mock import MagicMock

import pytest

import ml._imports as imports_module
import ml.data.scheduler as scheduler_module
from ml.config.scheduler_config import DatabentoConfig
from ml.config.scheduler_config import SchedulerConfig
from ml.data.coverage.manager import BucketSpec
from ml.data.ingest.market_bindings import ResolvedMarketBinding
from ml.data.scheduler import DataScheduler
from ml.registry.dataclasses import DatasetType
from ml.registry.dataclasses import StorageKind
from ml.tests.utils.stubs import RegistryTestStub


pytestmark = pytest.mark.usefixtures(
    "isolated_prometheus_registry",
    "mock_tracing_backend",
    "isolated_orchestrator_env",
)


@dataclass(slots=True)
class _MetricStub:
    labels_calls: list[dict[str, object]]
    inc_calls: int
    observe_calls: list[float]
    set_calls: list[object]

    def __init__(self) -> None:
        self.labels_calls = []
        self.inc_calls = 0
        self.observe_calls = []
        self.set_calls = []

    def labels(self, **kwargs: object) -> _MetricStub:
        self.labels_calls.append(dict(kwargs))
        return self

    def inc(self, *_: object, **__: object) -> None:
        self.inc_calls += 1

    def observe(self, value: float) -> None:
        self.observe_calls.append(float(value))

    def set(self, value: object) -> None:
        self.set_calls.append(value)


@dataclass(slots=True)
class _DataItem:
    ts_event: int


@dataclass(slots=True)
class _ResponseStub:
    payload: bytes = b"dbn"

    def to_file(self, path: str) -> None:
        Path(path).write_bytes(self.payload)


@dataclass(slots=True)
class _TimeseriesStub:
    calls: int = 0
    error: Exception | None = None

    def get_range(
        self,
        *,
        dataset: str,
        symbols: list[str],
        schema: str,
        start: datetime,
        end: datetime,
        stype_in: str,
    ) -> _ResponseStub:
        del dataset, symbols, schema, start, end, stype_in
        self.calls += 1
        if self.error is not None:
            raise self.error
        return _ResponseStub()


@dataclass(slots=True)
class _FeatureStoreStub:
    calls: list[str]

    def __init__(self) -> None:
        self.calls = []

    def compute_and_store_historical(
        self,
        *,
        instrument_id: str,
        start: datetime,
        end: datetime,
        force_recompute: bool,
    ) -> int:
        del start, end, force_recompute
        self.calls.append(instrument_id)
        if "NVDA" in instrument_id:
            raise RuntimeError("store failed")
        return 3


@dataclass(slots=True)
class _CatalogStub:
    write_error: Exception | None = None
    write_calls: int = 0

    def write_data(self, _data: object) -> None:
        self.write_calls += 1
        if self.write_error is not None:
            raise self.write_error

    def query(
        self,
        *,
        data_cls: object,
        identifiers: list[str],
        start: int,
        end: int,
    ) -> list[object]:
        del data_cls, start, end
        ident = identifiers[0]
        if "AAPL" in ident:
            return [object(), object()]
        if "MSFT" in ident:
            return []
        if "NVDA" in ident:
            return [object()]
        return [object()]


def _patch_scheduler_metrics(monkeypatch: pytest.MonkeyPatch) -> dict[str, _MetricStub]:
    names = (
        "data_collection_errors_total",
        "catalog_write_operations_total",
        "data_collected_total",
        "data_collection_latency",
        "data_staleness_seconds",
        "api_request_total",
        "api_rate_limit_hits",
        "catalog_write_latency",
        "feature_store_operations_total",
        "feature_computation_latency",
        "feature_store_latency",
        "feature_computation_errors_total",
        "active_feature_tasks",
        "active_collection_tasks",
        "pipeline_runs_total",
        "pipeline_stage_latency",
        "data_retention_cleanup_total",
    )
    patched: dict[str, _MetricStub] = {}
    for name in names:
        stub = _MetricStub()
        monkeypatch.setattr(scheduler_module, name, stub)
        patched[name] = stub
    return patched


def _make_scheduler(config: SchedulerConfig) -> DataScheduler:
    scheduler = object.__new__(DataScheduler)
    scheduler.config = config
    scheduler.catalog = _CatalogStub()
    scheduler.feature_engineer = None
    scheduler._data_registry = None
    scheduler._feature_store = None
    scheduler._feature_store_connection = None
    scheduler._current_run_id = "run_test"
    scheduler._databento_loader = MagicMock()
    scheduler._init_mgr = SimpleNamespace(start_metrics_server=lambda _port: None)
    scheduler._retention_mgr = None
    scheduler._use_orchestrator = False
    scheduler._dual_write = False
    scheduler._dual_write_dataset_types = {
        DatasetType.BARS: True,
        DatasetType.TRADES: True,
        DatasetType.TBBO: True,
        DatasetType.MBP1: True,
        DatasetType.MBP10: True,
        DatasetType.MBO: True,
    }
    scheduler._dataset_type_identifier_templates = dict(scheduler_module.DATASET_TYPE_IDENTIFIER_DEFAULTS)
    scheduler.enabled = True
    return scheduler


def test_noop_metric_methods_are_safe() -> None:
    metric = scheduler_module._NoOpMetric()
    assert metric.labels(scope="x") is metric
    assert metric.inc() is None
    assert metric.observe() is None


def test_scheduler_init_exercises_optional_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    config = SchedulerConfig(
        symbols=["AAPL.XNAS"],
        feature_store_enabled=True,
        databento=DatabentoConfig(api_key="k"),
    )
    metric_ports: list[int] = []
    init_calls: list[bool] = []
    fs_calls: list[bool] = []

    monkeypatch.setattr(scheduler_module, "HAS_PROMETHEUS", True)
    monkeypatch.setattr(scheduler_module, "validate_dataset_type_templates", lambda mapping: mapping)
    monkeypatch.setattr(DataScheduler, "_init_data_registry", lambda self: init_calls.append(True))
    monkeypatch.setattr(DataScheduler, "_initialize_feature_store", lambda self: fs_calls.append(True))
    monkeypatch.setattr(DataScheduler, "_start_metrics_server", lambda self, port: metric_ports.append(port))

    scheduler = DataScheduler(
        catalog=MagicMock(),
        config=config,
        collector=MagicMock(),
        feature_engineer=SimpleNamespace(config=object()),
        start_metrics_server=True,
        metrics_port=9001,
        dual_write=True,
        dual_write_dataset_types={DatasetType.BARS: False},
        dataset_type_identifier_templates={DatasetType.BARS: "bars_{symbol}_{venue}"},
    )

    assert init_calls == [True]
    assert fs_calls == [True]
    assert metric_ports == [9001]
    assert scheduler._dual_write_enabled_for(DatasetType.BARS) is False
    assert scheduler._dual_write_enabled_for(DatasetType.TRADES) is True


def test_init_data_registry_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    scheduler = _make_scheduler(SchedulerConfig())

    registry_obj = object()
    scheduler._registry_integrator = SimpleNamespace(initialize_registry=lambda connection: registry_obj)
    scheduler._feature_store_connection = "postgresql://user:pass@host/db"
    DataScheduler._init_data_registry(scheduler)
    assert scheduler._data_registry is registry_obj

    scheduler._registry_integrator = SimpleNamespace(initialize_registry=lambda connection: None)

    captured: dict[str, object] = {}

    def _registry_ctor(*, registry_path: Path, persistence_config: object) -> object:
        captured["registry_path"] = registry_path
        captured["persistence_config"] = persistence_config
        return object()

    monkeypatch.setattr(scheduler_module, "DataRegistry", _registry_ctor)

    DataScheduler._init_data_registry(scheduler)
    assert captured["registry_path"] == Path("/tmp/ml_registry")

    scheduler._feature_store_connection = None
    monkeypatch.setattr(scheduler_module.Path, "home", lambda: tmp_path)
    DataScheduler._init_data_registry(scheduler)
    assert isinstance(captured["registry_path"], Path)

    monkeypatch.setattr(
        scheduler_module,
        "DataRegistry",
        lambda **_: (_ for _ in ()).throw(RuntimeError("db down")),
    )
    DataScheduler._init_data_registry(scheduler)
    assert scheduler._data_registry is None


def test_initialize_feature_store_success_and_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    scheduler = _make_scheduler(
        SchedulerConfig(
            feature_store_enabled=True,
            databento=DatabentoConfig(api_key="k"),
        ),
    )
    scheduler.feature_engineer = SimpleNamespace(config={"x": 1})
    scheduler._feature_store_connection = None

    check_calls: list[list[str]] = []
    monkeypatch.setattr(imports_module, "HAS_POLARS", False)
    monkeypatch.setattr(imports_module, "check_ml_dependencies", lambda deps: check_calls.append(list(deps)))

    created: list[str] = []

    class _FeatureStore:
        def __init__(self, *, connection_string: str, feature_config: object) -> None:
            del feature_config
            created.append(connection_string)

    monkeypatch.setattr("ml.stores.feature_store.FeatureStore", _FeatureStore)

    DataScheduler._initialize_feature_store(scheduler)
    assert check_calls == [["polars"]]
    assert created
    assert scheduler._feature_store is not None

    monkeypatch.setattr(
        "ml.stores.feature_store.FeatureStore",
        lambda **_: (_ for _ in ()).throw(RuntimeError("fail")),
    )
    DataScheduler._initialize_feature_store(scheduler)
    assert scheduler._feature_store is None


def test_start_metrics_server_success_and_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    scheduler = _make_scheduler(SchedulerConfig())
    fallback_server = object()
    scheduler._init_mgr = SimpleNamespace(start_metrics_server=lambda _port: fallback_server)

    class _MonitoringConfig:
        def __init__(self, *, enabled: bool, metrics_port: int) -> None:
            self.enabled = enabled
            self.metrics_port = metrics_port

    class _MetricsServer:
        def __init__(self, *, config: _MonitoringConfig) -> None:
            self.config = config
            self.started = False

        def start(self) -> None:
            self.started = True

    monkeypatch.setitem(sys.modules, "ml.monitoring._config", SimpleNamespace(MonitoringConfig=_MonitoringConfig))
    monkeypatch.setitem(sys.modules, "ml.monitoring.server", SimpleNamespace(MetricsServer=_MetricsServer))

    DataScheduler._start_metrics_server(scheduler, 9123)
    assert scheduler._metrics_server is not None

    class _BrokenServer:
        def __init__(self, *, config: _MonitoringConfig) -> None:
            del config
            raise RuntimeError("boom")

    monkeypatch.setitem(sys.modules, "ml.monitoring.server", SimpleNamespace(MetricsServer=_BrokenServer))
    DataScheduler._start_metrics_server(scheduler, 9124)
    assert scheduler._metrics_server is fallback_server


def test_run_daily_update_success_and_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    metrics = _patch_scheduler_metrics(monkeypatch)
    scheduler = _make_scheduler(SchedulerConfig())
    scheduler.feature_engineer = object()

    calls: list[str] = []
    scheduler._collect_latest_data = lambda: calls.append("collect")
    scheduler._collect_via_orchestrator = lambda: calls.append("collect_orch")
    scheduler._compute_features = lambda: calls.append("features")
    scheduler._clean_old_data = lambda: calls.append("cleanup")

    monkeypatch.setattr(scheduler_module, "track_pipeline_stage", lambda _stage: contextlib.nullcontext())

    DataScheduler.run_daily_update(scheduler)
    assert calls == ["collect", "features", "cleanup"]
    assert metrics["pipeline_runs_total"].labels_calls[-1]["status"] == "success"

    scheduler._collect_latest_data = lambda: (_ for _ in ()).throw(RuntimeError("collect failed"))
    with pytest.raises(RuntimeError, match="collect failed"):
        DataScheduler.run_daily_update(scheduler)
    assert metrics["pipeline_runs_total"].labels_calls[-1]["status"] == "failure"


def test_collect_latest_data_branches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    metrics = _patch_scheduler_metrics(monkeypatch)
    config = SchedulerConfig(
        symbols=["AAPL.XNAS", "MSFT.XNAS", "NVDA.XNAS"],
        databento=DatabentoConfig(
            api_key="test-key",
            use_temporary_files=True,
            temp_data_dir=str(tmp_path / "tmp_dbn"),
        ),
    )
    scheduler = _make_scheduler(config)
    scheduler._get_previous_trading_day = lambda: datetime(2025, 1, 10, tzinfo=UTC)

    monkeypatch.setitem(sys.modules, "databento", SimpleNamespace(Historical=lambda _key: object()))

    collect_calls: list[str] = []

    def _collect_symbol_data(**kwargs: object) -> bool:
        collect_calls.append(str(kwargs["symbol"]))
        return False

    scheduler._collect_symbol_data = _collect_symbol_data

    DataScheduler._collect_latest_data(scheduler)

    assert collect_calls == ["AAPL.XNAS", "MSFT.XNAS", "NVDA.XNAS"]
    assert metrics["active_collection_tasks"].set_calls[-1] == 0
    assert metrics["api_rate_limit_hits"].inc_calls == 1
    assert not (tmp_path / "tmp_dbn").exists()

    missing_key_scheduler = _make_scheduler(
        SchedulerConfig(symbols=["AAPL.XNAS"], databento=DatabentoConfig(api_key=None)),
    )
    missing_key_scheduler._get_previous_trading_day = lambda: datetime(2025, 1, 10, tzinfo=UTC)
    monkeypatch.delenv("DATABENTO_API_KEY", raising=False)
    with pytest.raises(ValueError):
        DataScheduler._collect_latest_data(missing_key_scheduler)


def test_collect_symbol_data_core_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    metrics = _patch_scheduler_metrics(monkeypatch)
    scheduler = _make_scheduler(
        SchedulerConfig(
            symbols=["SPY.XNAS"],
            max_retries=2,
            retry_delay_seconds=0.0,
            databento=DatabentoConfig(api_key="k", use_temporary_files=True, schema="ohlcv-1m"),
        ),
    )

    start = datetime(2025, 1, 10, tzinfo=UTC)
    end = datetime(2025, 1, 10, 23, 59, tzinfo=UTC)

    invalid = DataScheduler._collect_symbol_data(
        scheduler,
        client=SimpleNamespace(timeseries=_TimeseriesStub()),
        symbol="BAD",
        start_date=start,
        end_date=end,
        target_date=start,
        temp_data_dir=tmp_path,
    )
    assert invalid is False

    direct_scheduler = _make_scheduler(
        SchedulerConfig(
            symbols=["SPY.XNAS"],
            databento=DatabentoConfig(api_key="k", use_temporary_files=False),
        ),
    )
    direct = DataScheduler._collect_symbol_data(
        direct_scheduler,
        client=SimpleNamespace(timeseries=_TimeseriesStub()),
        symbol="SPY.XNAS",
        start_date=start,
        end_date=end,
        target_date=start,
        temp_data_dir=None,
    )
    assert direct is False

    scheduler._load_from_dbn_file = lambda *_args, **_kwargs: [MagicMock()]
    mocked = DataScheduler._collect_symbol_data(
        scheduler,
        client=SimpleNamespace(timeseries=_TimeseriesStub()),
        symbol="SPY.XNAS",
        start_date=start,
        end_date=end,
        target_date=start,
        temp_data_dir=tmp_path,
    )
    assert mocked is True

    scheduler._load_from_dbn_file = lambda *_args, **_kwargs: []
    no_data = DataScheduler._collect_symbol_data(
        scheduler,
        client=SimpleNamespace(timeseries=_TimeseriesStub()),
        symbol="SPY.XNAS",
        start_date=start,
        end_date=end,
        target_date=start,
        temp_data_dir=tmp_path,
    )
    assert no_data is False
    assert metrics["data_collection_errors_total"].labels_calls[-1]["error_type"] == "no_data"


def test_collect_symbol_data_registry_and_error_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    metrics = _patch_scheduler_metrics(monkeypatch)
    scheduler = _make_scheduler(
        SchedulerConfig(
            symbols=["AAPL.XNAS"],
            max_retries=1,
            databento=DatabentoConfig(api_key="k", use_temporary_files=True, schema="ohlcv-1m"),
        ),
    )
    scheduler._data_registry = RegistryTestStub()
    scheduler._ensure_dataset_registered = lambda **_kwargs: None
    scheduler._load_from_dbn_file = lambda *_args, **_kwargs: [_DataItem(100), _DataItem(200)]
    scheduler_module.data_events_total = _MetricStub()

    start = datetime(2025, 1, 10, tzinfo=UTC)
    end = datetime(2025, 1, 10, 23, 59, tzinfo=UTC)

    ok = DataScheduler._collect_symbol_data(
        scheduler,
        client=SimpleNamespace(timeseries=_TimeseriesStub()),
        symbol="AAPL.XNAS",
        start_date=start,
        end_date=end,
        target_date=start,
        temp_data_dir=tmp_path,
    )
    assert ok is True
    assert metrics["catalog_write_operations_total"].labels_calls[-1]["status"] == "success"

    class _EmitFailRegistry(RegistryTestStub):
        def emit_event(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError("event failed")

    scheduler._data_registry = _EmitFailRegistry()
    still_ok = DataScheduler._collect_symbol_data(
        scheduler,
        client=SimpleNamespace(timeseries=_TimeseriesStub()),
        symbol="AAPL.XNAS",
        start_date=start,
        end_date=end,
        target_date=start,
        temp_data_dir=tmp_path,
    )
    assert still_ok is True

    catalog = _CatalogStub(write_error=RuntimeError("catalog failed"))
    scheduler.catalog = catalog
    scheduler._data_registry = RegistryTestStub()
    failed = DataScheduler._collect_symbol_data(
        scheduler,
        client=SimpleNamespace(timeseries=_TimeseriesStub()),
        symbol="AAPL.XNAS",
        start_date=start,
        end_date=end,
        target_date=start,
        temp_data_dir=tmp_path,
    )
    assert failed is False

    scheduler._data_registry = _EmitFailRegistry()
    failed_again = DataScheduler._collect_symbol_data(
        scheduler,
        client=SimpleNamespace(timeseries=_TimeseriesStub()),
        symbol="AAPL.XNAS",
        start_date=start,
        end_date=end,
        target_date=start,
        temp_data_dir=tmp_path,
    )
    assert failed_again is False


@pytest.mark.parametrize(
    ("message", "expected_error_type", "expected_status_code"),
    [
        ("rate limit exceeded", "rate_limit", None),
        ("connection timeout", "connection", None),
        ("unauthorized", "auth", "401"),
        ("unexpected failure", "unknown", "500"),
    ],
)
def test_collect_symbol_data_error_classification(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    message: str,
    expected_error_type: str,
    expected_status_code: str | None,
) -> None:
    metrics = _patch_scheduler_metrics(monkeypatch)
    scheduler = _make_scheduler(
        SchedulerConfig(
            symbols=["AAPL.XNAS"],
            max_retries=1,
            retry_delay_seconds=0.0,
            databento=DatabentoConfig(api_key="k", use_temporary_files=True),
        ),
    )
    monkeypatch.setattr(scheduler_module.time, "sleep", lambda _seconds: None)

    result = DataScheduler._collect_symbol_data(
        scheduler,
        client=SimpleNamespace(timeseries=_TimeseriesStub(error=RuntimeError(message))),
        symbol="AAPL.XNAS",
        start_date=datetime(2025, 1, 10, tzinfo=UTC),
        end_date=datetime(2025, 1, 10, 23, 59, tzinfo=UTC),
        target_date=datetime(2025, 1, 10, tzinfo=UTC),
        temp_data_dir=tmp_path,
    )

    assert result is False
    assert metrics["data_collection_errors_total"].labels_calls[-1]["error_type"] == expected_error_type
    if expected_status_code is not None:
        assert metrics["api_request_total"].labels_calls[-1]["status_code"] == expected_status_code


def test_load_from_dbn_file_maps_venues() -> None:
    scheduler = _make_scheduler(
        SchedulerConfig(databento=DatabentoConfig(api_key="k", schema="trades")),
    )
    calls: dict[str, object] = {}

    def _from_dbn_file(**kwargs: object) -> list[object]:
        calls.update(kwargs)
        return [object()]

    scheduler._databento_loader = SimpleNamespace(from_dbn_file=_from_dbn_file)
    data = DataScheduler._load_from_dbn_file(scheduler, Path("/tmp/x.dbn"), "SPY", "XNAS")

    assert len(data) == 1
    assert "NASDAQ" in str(calls["instrument_id"])
    assert calls["include_trades"] is True


def test_build_orchestrator_and_collect_via_orchestrator(monkeypatch: pytest.MonkeyPatch) -> None:
    scheduler = _make_scheduler(
        SchedulerConfig(
            symbols=["AAPL.XNAS", "MSFT.XNAS"],
            databento=DatabentoConfig(api_key="api", dataset="EQUS.MINI", schema="ohlcv-1m"),
        ),
    )
    scheduler._feature_store_connection = "postgresql://user:pass@host/db"
    scheduler._data_registry = RegistryTestStub()

    monkeypatch.setattr(scheduler_module, "SqlCoverageProvider", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(scheduler_module, "SqlMarketDataWriter", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(scheduler_module, "DatabentoAPIClient", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(scheduler_module, "DatabentoIngestor", lambda **kwargs: SimpleNamespace(**kwargs))

    class _Orchestrator:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            self.backfill_calls: list[dict[str, object]] = []
            self.binding_calls: list[dict[str, object]] = []

        def backfill_gaps(self, **kwargs: object) -> None:
            self.backfill_calls.append(dict(kwargs))

        def backfill_binding(self, **kwargs: object) -> None:
            self.binding_calls.append(dict(kwargs))

        @staticmethod
        def resolve_market_bindings(**_kwargs: object) -> tuple[ResolvedMarketBinding, ...]:
            return ()

    monkeypatch.setattr(scheduler_module, "IngestionOrchestrator", _Orchestrator)

    orchestrator, lookback = DataScheduler._build_orchestrator(scheduler)
    assert lookback == 1
    assert isinstance(orchestrator, _Orchestrator)

    scheduler._build_orchestrator = lambda: (orchestrator, 2)
    DataScheduler._collect_via_orchestrator(scheduler)
    assert len(orchestrator.backfill_calls) == 2

    binding = ResolvedMarketBinding(
        binding_id="binding-1",
        symbol="AAPL",
        instrument_ids=("AAPL.XNAS",),
        dataset_id="EQUS.MINI",
        descriptor_id="EQUS.MINI",
        schema="ohlcv-1m",
        storage_kind=StorageKind.POSTGRES,
        license_start=None,
        license_end=None,
        start=None,
        end=None,
        source="descriptor",
    )
    scheduler.config = SchedulerConfig(
        symbols=["AAPL.XNAS"],
        market_dataset_id="EQUS.MINI",
        databento=DatabentoConfig(api_key="api", dataset="EQUS.MINI", schema="ohlcv-1m"),
    )

    monkeypatch.setattr(
        scheduler_module.IngestionOrchestrator,
        "resolve_market_bindings",
        staticmethod(lambda **_kwargs: (binding, binding)),
    )

    DataScheduler._collect_via_orchestrator(scheduler)
    assert len(orchestrator.binding_calls) == 1


def test_ensure_dataset_registered_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    scheduler = _make_scheduler(SchedulerConfig())
    DataScheduler._ensure_dataset_registered(
        scheduler,
        dataset_id="bars_aapl_xnas",
        dataset_type_label="bars",
        location="/tmp/catalog",
    )

    class _Registry:
        def __init__(self, fail_get: bool, fail_register: bool) -> None:
            self.fail_get = fail_get
            self.fail_register = fail_register
            self.registered: int = 0

        def get_manifest(self, _dataset_id: str) -> object:
            if self.fail_get:
                raise RuntimeError("missing")
            return object()

        def register_dataset(self, _manifest: object) -> None:
            self.registered += 1
            if self.fail_register:
                raise RuntimeError("register failed")

    scheduler._data_registry = _Registry(fail_get=False, fail_register=False)
    DataScheduler._ensure_dataset_registered(
        scheduler,
        dataset_id="bars_aapl_xnas",
        dataset_type_label="bars",
        location="/tmp/catalog",
    )
    assert scheduler._data_registry.registered == 0

    scheduler._data_registry = _Registry(fail_get=True, fail_register=False)
    DataScheduler._ensure_dataset_registered(
        scheduler,
        dataset_id="bars_aapl_xnas",
        dataset_type_label="bars",
        location="/tmp/catalog",
    )
    assert scheduler._data_registry.registered == 1

    scheduler._data_registry = _Registry(fail_get=True, fail_register=True)
    DataScheduler._ensure_dataset_registered(
        scheduler,
        dataset_id="bars_aapl_xnas",
        dataset_type_label="bars",
        location="/tmp/catalog",
    )
    assert scheduler._data_registry.registered == 1


def test_build_orchestrator_validation_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    scheduler = _make_scheduler(SchedulerConfig(databento=DatabentoConfig(api_key=None)))
    monkeypatch.delenv("DATABENTO_API_KEY", raising=False)
    with pytest.raises(ValueError, match="DATABENTO_API_KEY required"):
        DataScheduler._build_orchestrator(scheduler)

    scheduler = _make_scheduler(
        SchedulerConfig(databento=DatabentoConfig(api_key="k", dataset="EQUS.MINI", schema="ohlcv-1m")),
    )
    scheduler._feature_store_connection = None
    monkeypatch.delenv("DB_CONNECTION", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("NAUTILUS_DB_CONNECTION", raising=False)
    with pytest.raises(ValueError, match="DB connection required"):
        DataScheduler._build_orchestrator(scheduler)

    scheduler._feature_store_connection = "postgresql://user:pass@host/db"
    monkeypatch.setattr(scheduler_module, "SqlCoverageProvider", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(scheduler_module, "SqlMarketDataWriter", lambda **kwargs: SimpleNamespace(**kwargs))
    scheduler._data_registry = None
    with pytest.raises(RuntimeError, match="DataRegistry not initialized"):
        DataScheduler._build_orchestrator(scheduler)


def test_build_orchestrator_dual_write_domain_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    scheduler = _make_scheduler(
        SchedulerConfig(
            symbols=["SPY.XNAS"],
            databento=DatabentoConfig(api_key="api", dataset="EQUS.MINI", schema="ohlcv-1m"),
        ),
    )
    scheduler._feature_store_connection = "postgresql://user:pass@host/db"
    scheduler._data_registry = RegistryTestStub()
    scheduler._dual_write = True
    scheduler._dual_write_dataset_types[DatasetType.BARS] = True

    monkeypatch.setattr(scheduler_module, "SqlCoverageProvider", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(scheduler_module, "SqlMarketDataWriter", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(scheduler_module, "DatabentoAPIClient", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(scheduler_module, "DatabentoIngestor", lambda **kwargs: SimpleNamespace(**kwargs))

    class _RawWriter:
        def __init__(self, catalog: object, dataset_type_identifier_templates: object) -> None:
            del catalog, dataset_type_identifier_templates

    class _FilteredWriter:
        def __init__(self, writer: object, enabled: object) -> None:
            self.writer = writer
            self.enabled = enabled

        def is_enabled(self, dataset_type: DatasetType) -> bool:
            enabled = getattr(self, "enabled")
            if isinstance(enabled, dict):
                return bool(enabled.get(dataset_type, True))
            return True

    class _Orchestrator:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    monkeypatch.setattr(scheduler_module, "ParquetCatalogRawWriter", _RawWriter)
    monkeypatch.setattr(scheduler_module, "FilteredRawWriter", _FilteredWriter)
    monkeypatch.setattr(scheduler_module, "IngestionOrchestrator", _Orchestrator)

    class _Historical:
        def __init__(self, _key: str) -> None:
            self.timeseries = SimpleNamespace(get_range=lambda **_kwargs: _ResponseStub())

    monkeypatch.setitem(sys.modules, "databento", SimpleNamespace(Historical=_Historical))

    import nautilus_trader.adapters.databento.loaders as loaders_module

    class _Loader:
        def from_dbn_file(self, **_kwargs: object) -> list[object]:
            return [object()]

    monkeypatch.setattr(loaders_module, "DatabentoDataLoader", _Loader)

    orchestrator, _lookback = DataScheduler._build_orchestrator(scheduler)
    domain_loader = orchestrator.kwargs["domain_loader"]
    assert domain_loader is not None

    loaded = domain_loader.load(
        dataset_id="EQUS.MINI",
        schema="ohlcv-1m",
        instrument_id="SPY.XNAS",
        start_ns=1_700_000_000_000_000_000,
        end_ns=1_700_000_100_000_000_000,
    )
    assert loaded

    scheduler._dual_write_dataset_types[DatasetType.BARS] = False
    skipped = domain_loader.load(
        dataset_id="EQUS.MINI",
        schema="ohlcv-1m",
        instrument_id="SPY.XNAS",
        start_ns=1_700_000_000_000_000_000,
        end_ns=1_700_000_100_000_000_000,
    )
    assert skipped == []


def test_compute_features_and_cleanup_and_misc(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_scheduler_metrics(monkeypatch)

    scheduler = _make_scheduler(
        SchedulerConfig(
            symbols=["BAD", "AAPL.XNAS", "MSFT.XNAS", "NVDA.XNAS"],
            feature_store_enabled=True,
            databento=DatabentoConfig(api_key="k"),
        ),
    )
    scheduler.feature_engineer = object()
    scheduler.catalog = _CatalogStub()
    scheduler._feature_store = _FeatureStoreStub()
    scheduler._get_previous_trading_day = lambda: datetime(2025, 1, 10, tzinfo=UTC)

    monkeypatch.setattr(imports_module, "HAS_POLARS", True)
    monkeypatch.setattr(imports_module, "check_ml_dependencies", lambda _deps: None)

    DataScheduler._compute_features(scheduler)

    failing = _make_scheduler(
        SchedulerConfig(
            symbols=["AAPL.XNAS"],
            feature_store_enabled=True,
            databento=DatabentoConfig(api_key="k"),
        ),
    )
    failing.feature_engineer = object()
    failing._feature_store = _FeatureStoreStub()
    failing._get_previous_trading_day = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    with pytest.raises(RuntimeError, match="boom"):
        DataScheduler._compute_features(failing)

    clean_scheduler = _make_scheduler(SchedulerConfig(retention_days=10))
    DataScheduler._clean_old_data(clean_scheduler)

    class _BrokenMetric(_MetricStub):
        def inc(self, *_: object, **__: object) -> None:
            raise RuntimeError("metric failure")

    monkeypatch.setattr(scheduler_module, "data_retention_cleanup_total", _BrokenMetric())
    with pytest.raises(RuntimeError):
        DataScheduler._clean_old_data(clean_scheduler)


def test_compute_features_early_return_and_polars_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    scheduler = _make_scheduler(SchedulerConfig(feature_store_enabled=False))
    scheduler.feature_engineer = object()
    DataScheduler._compute_features(scheduler)

    scheduler = _make_scheduler(SchedulerConfig(feature_store_enabled=True))
    scheduler.feature_engineer = None
    DataScheduler._compute_features(scheduler)

    scheduler = _make_scheduler(SchedulerConfig(feature_store_enabled=True))
    scheduler.feature_engineer = object()
    scheduler._feature_store = None
    scheduler._initialize_feature_store = lambda: None
    DataScheduler._compute_features(scheduler)

    scheduler = _make_scheduler(
        SchedulerConfig(
            feature_store_enabled=True,
            symbols=["AAPL.XNAS"],
            databento=DatabentoConfig(api_key="k"),
        ),
    )
    scheduler.feature_engineer = object()
    scheduler._feature_store = _FeatureStoreStub()
    scheduler.catalog = _CatalogStub()
    scheduler._get_previous_trading_day = lambda: datetime(2025, 1, 10, tzinfo=UTC)
    monkeypatch.setattr(imports_module, "HAS_POLARS", False)
    called: list[list[str]] = []
    monkeypatch.setattr(imports_module, "check_ml_dependencies", lambda deps: called.append(list(deps)))
    DataScheduler._compute_features(scheduler)
    assert called == [["polars"]]


def test_run_daily_update_orchestrator_and_collect_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_scheduler_metrics(monkeypatch)
    scheduler = _make_scheduler(SchedulerConfig())
    scheduler._use_orchestrator = True
    scheduler.feature_engineer = None
    call_order: list[str] = []
    scheduler._collect_via_orchestrator = lambda: call_order.append("orch")
    scheduler._clean_old_data = lambda: call_order.append("cleanup")
    monkeypatch.setattr(scheduler_module, "track_pipeline_stage", lambda _stage: contextlib.nullcontext())
    DataScheduler.run_daily_update(scheduler)
    assert call_order == ["orch", "cleanup"]

    import builtins

    scheduler = _make_scheduler(SchedulerConfig(databento=DatabentoConfig(api_key="k")))
    scheduler._get_previous_trading_day = lambda: datetime(2025, 1, 10, tzinfo=UTC)
    original_import = builtins.__import__

    def _raise_databento(name: str, *args: object, **kwargs: object) -> object:
        if name == "databento":
            raise ImportError("missing databento")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _raise_databento)
    with pytest.raises(ImportError):
        DataScheduler._collect_latest_data(scheduler)


def test_scheduler_status_stop_schedule_and_main(monkeypatch: pytest.MonkeyPatch) -> None:
    scheduler = _make_scheduler(SchedulerConfig(collection_time="03:30", retention_days=11))
    scheduler.catalog = SimpleNamespace(path="/tmp/catalog")
    scheduler.feature_engineer = object()

    status = DataScheduler.get_status(scheduler)
    assert status["enabled"] is True
    assert status["collection_time"] == "03:30"

    class _Server:
        def __init__(self) -> None:
            self.stop_calls = 0

        def stop(self) -> None:
            self.stop_calls += 1

    server = _Server()
    scheduler._metrics_server = server
    DataScheduler.stop(scheduler)
    assert scheduler.enabled is False
    assert server.stop_calls == 1

    scheduler.enabled = True

    class _BrokenServer:
        def stop(self) -> None:
            raise RuntimeError("boom")

    scheduler._metrics_server = _BrokenServer()
    DataScheduler.stop(scheduler)
    assert scheduler.enabled is False

    DataScheduler.schedule_updates(scheduler)
    DataScheduler.schedule_updates(scheduler, cron_expression="*/5 * * * *")

    created: dict[str, object] = {}

    class _SchedulerStub:
        def __init__(self, *, catalog: object, config: object) -> None:
            created["catalog"] = catalog
            created["config"] = config

        def get_status(self) -> dict[str, object]:
            return {"ok": True}

    monkeypatch.setattr(scheduler_module, "ParquetDataCatalog", lambda path: SimpleNamespace(path=path))
    monkeypatch.setattr(scheduler_module, "DataScheduler", _SchedulerStub)
    monkeypatch.setattr(scheduler_module, "logger", SimpleNamespace(info=lambda *args, **kwargs: None))

    scheduler_module.main()
    assert isinstance(created["catalog"], SimpleNamespace)


def test_run_targeted_update_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    scheduler = _make_scheduler(
        SchedulerConfig(
            symbols=["AAPL.XNAS"],
            databento=DatabentoConfig(api_key="k"),
        ),
    )
    bucket = BucketSpec(
        dataset_id="EQUS.MINI",
        schema="ohlcv-1m",
        instrument_id="AAPL.XNAS",
        bucket_start_ns=int(datetime(2025, 1, 10, tzinfo=UTC).timestamp() * 1_000_000_000),
    )

    original_import = __import__

    def _raise_databento(name: str, *args: object, **kwargs: object) -> object:
        if name == "databento":
            raise ImportError("missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _raise_databento)
    with pytest.raises(ImportError):
        DataScheduler.run_targeted_update(scheduler, [bucket])


def test_misc_scheduler_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    scheduler = _make_scheduler(
        SchedulerConfig(
            symbols=["AAPL.XNAS"],
            feature_store_enabled=True,
            feature_store_connection="postgresql://user:pass@host/db",
            databento=DatabentoConfig(api_key="k"),
        ),
    )
    scheduler._dual_write = False
    assert scheduler._dual_write_enabled_for(DatasetType.BARS) is False
    assert scheduler.data_registry is None
    DataScheduler.run_targeted_update(scheduler, [])

    scheduler.feature_engineer = None
    scheduler._feature_store = None
    monkeypatch.setattr(imports_module, "HAS_POLARS", True)
    monkeypatch.setattr(imports_module, "check_ml_dependencies", lambda _deps: None)
    DataScheduler._initialize_feature_store(scheduler)
    assert scheduler._feature_store is not None
