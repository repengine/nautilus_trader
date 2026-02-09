"""
Unit tests for IngestionCoordinator (facade-only).
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from typing import cast
from unittest.mock import MagicMock

import pytest

import ml.orchestration.ingestion_coordinator as ingestion_coordinator_module
from ml.config.market_data import MarketDatasetInput
from ml.config.scheduler_config import SchedulerConfig
from ml.data.ingest.orchestrator import BackfillWindowList
from ml.data.ingest.subscription import SubscriptionPolicy as CoveragePolicy
from ml.data.vintage import VintagePolicy
from ml.orchestration.config_types import AutoFillUniverseConfig
from ml.orchestration.config_types import DatasetBuildConfig
from ml.orchestration.config_types import EarningsCoordinatorConfig
from ml.orchestration.config_types import MacroIngestionConfig
from ml.orchestration.config_types import PreIngestionOptions
from ml.orchestration.ingestion_coordinator import IngestBackfillRuntimeConfig
from ml.orchestration.ingestion_coordinator import IngestionCoordinator
from ml.orchestration.ingestion_coordinator import run_ingest_backfill
from ml.registry.dataclasses import DatasetType
from ml.registry.dataclasses import StorageKind
from ml.tests.utils.targets import build_default_target_semantics_payload


@pytest.fixture
def coordinator() -> IngestionCoordinator:
    """Create an IngestionCoordinator with minimal dependencies."""
    return IngestionCoordinator(
        coverage=MagicMock(),
        writer=MagicMock(),
        registry=MagicMock(),
        ingestor=MagicMock(),
    )


def test_coordinate_ingestion_sums_rows(coordinator: IngestionCoordinator) -> None:
    """coordinate_ingestion aggregates rows across instruments."""
    coordinator.backfill = MagicMock(
        side_effect=[
            BackfillWindowList(persisted=(), requested=(), frames_written=1, rows_written=10),
            BackfillWindowList(persisted=(), requested=(), frames_written=1, rows_written=5),
        ],
    )

    result = coordinator.coordinate_ingestion(
        dataset_id="databento.ohlcv-1s",
        schema="ohlcv-1s",
        instrument_ids=["SPY.XNAS", "QQQ.XNAS"],
        lookback_days=5,
    )

    assert result["fallback_level"] == "primary"
    assert result["rows_written"] == 15


def test_coordinate_ingestion_falls_back_on_error(coordinator: IngestionCoordinator) -> None:
    """coordinate_ingestion uses fallback when primary backfill fails."""
    coordinator.backfill = MagicMock(side_effect=RuntimeError("boom"))
    coordinator._handle_ingestion_fallback = MagicMock(
        return_value={"rows_written": 0, "fallback_level": "cached"},
    )

    result = coordinator.coordinate_ingestion(
        dataset_id="databento.ohlcv-1s",
        schema="ohlcv-1s",
        instrument_ids=["SPY.XNAS"],
        lookback_days=5,
    )

    assert result["fallback_level"] == "cached"
    coordinator._handle_ingestion_fallback.assert_called_once()


def test_backfill_delegates_to_orchestrator(coordinator: IngestionCoordinator) -> None:
    """backfill delegates to the ingestion orchestrator."""
    orchestrator = MagicMock()
    expected = BackfillWindowList(persisted=(), requested=(), frames_written=0, rows_written=0)
    orchestrator.backfill_gaps.return_value = expected
    coordinator._create_ingestion_orchestrator = MagicMock(return_value=orchestrator)

    result = coordinator.backfill(
        dataset_id="databento.ohlcv-1s",
        schema="ohlcv-1s",
        instrument_id="SPY.XNAS",
        lookback_days=5,
    )

    assert result is expected
    orchestrator.backfill_gaps.assert_called_once()


def test_backfill_binding_delegates_to_orchestrator(coordinator: IngestionCoordinator) -> None:
    """backfill_binding delegates to the ingestion orchestrator."""
    orchestrator = MagicMock()
    orchestrator.backfill_binding.return_value = {"SPY.XNAS": BackfillWindowList(persisted=(), requested=(), frames_written=0, rows_written=0)}
    coordinator._create_ingestion_orchestrator = MagicMock(return_value=orchestrator)

    binding = MagicMock()
    result = coordinator.backfill_binding(binding=binding, lookback_days=5)

    assert "SPY.XNAS" in result
    orchestrator.backfill_binding.assert_called_once_with(binding=binding, lookback_days=5)


def test_handle_ingestion_fallback_primary_uses_backfill(coordinator: IngestionCoordinator) -> None:
    """PRIMARY fallback uses component backfill."""
    coordinator.backfill = MagicMock(
        return_value=BackfillWindowList(persisted=(), requested=(), frames_written=1, rows_written=4),
    )

    result = coordinator._handle_ingestion_fallback(
        dataset_id="databento.ohlcv-1s",
        schema="ohlcv-1s",
        instrument_ids=["SPY.XNAS"],
        lookback_days=5,
        level="primary",
    )

    assert result["fallback_level"] == "primary"
    assert result["rows_written"] == 4


def test_handle_ingestion_fallback_cached_uses_backfill_coverage(
    coordinator: IngestionCoordinator,
) -> None:
    """CACHED fallback uses coverage gaps as a signal."""
    coordinator.backfill_coverage = MagicMock(return_value=[(1, 2), (3, 4)])

    result = coordinator._handle_ingestion_fallback(
        dataset_id="databento.ohlcv-1s",
        schema="ohlcv-1s",
        instrument_ids=["SPY.XNAS"],
        lookback_days=5,
        level="cached",
    )

    assert result["fallback_level"] == "cached"
    assert result["rows_written"] == 2


def test_run_ingest_backfill_executes_plan_and_saves_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _StubCoverage:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    class _StubWriter:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    class _StubRegistry:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    class _StubIngestor:
        def __init__(self, client: object) -> None:
            self.client = client

    class _StubOrchestrator:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    saved_state: list[tuple[str, object]] = []
    state_obj = object()
    execute_calls: list[dict[str, object]] = []
    emitted: list[str] = []

    monkeypatch.setattr("ml.stores.providers.SqlCoverageProvider", _StubCoverage)
    monkeypatch.setattr("ml.stores.providers.SqlMarketDataWriter", _StubWriter)
    monkeypatch.setattr("ml.registry.data_registry.DataRegistry", _StubRegistry)
    monkeypatch.setattr(ingestion_coordinator_module, "DatabentoIngestor", _StubIngestor)
    monkeypatch.setattr(ingestion_coordinator_module, "IngestionOrchestrator", _StubOrchestrator)
    monkeypatch.setattr(
        "ml.data.ingest.state.load_state",
        lambda _path: state_obj,
    )
    monkeypatch.setattr(
        "ml.data.ingest.state.save_state",
        lambda path, state: saved_state.append((str(path), state)),
    )
    monkeypatch.setattr(
        ingestion_coordinator_module,
        "execute_backfill_plan",
        lambda **kwargs: (
            execute_calls.append(dict(kwargs))
            or SimpleNamespace(total_windows=4, processed_bindings=())
        ),
    )

    config = IngestBackfillRuntimeConfig(
        db="postgresql://user:pass@localhost:5432/ml",
        dataset_id="EQUS.MINI",
        schema="ohlcv-1m",
        instruments=("SPY.XNAS",),
        lookback_days=3,
        table_name="market_data",
        state_path="checkpoints/test_state.json",
        coverage_mode="sql",
        write_mode="sql",
        client_mode="noop",
    )

    result = run_ingest_backfill(config, emit=emitted.append)

    assert result.total_windows_planned == 4
    assert result.state_saved is True
    assert saved_state == [("checkpoints/test_state.json", state_obj)]
    assert execute_calls and execute_calls[0]["state"] is state_obj
    assert emitted[-2:] == [
        "State saved to checkpoints/test_state.json",
        "Total windows planned: 4",
    ]


def test_run_ingest_backfill_dry_run_skips_state_save(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _StubCoverage:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    class _StubWriter:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    class _StubRegistry:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    class _StubIngestor:
        def __init__(self, client: object) -> None:
            self.client = client

    class _StubOrchestrator:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    save_calls: list[tuple[str, object]] = []
    execute_calls: list[dict[str, object]] = []

    monkeypatch.setattr("ml.stores.providers.SqlCoverageProvider", _StubCoverage)
    monkeypatch.setattr("ml.stores.providers.SqlMarketDataWriter", _StubWriter)
    monkeypatch.setattr("ml.registry.data_registry.DataRegistry", _StubRegistry)
    monkeypatch.setattr(ingestion_coordinator_module, "DatabentoIngestor", _StubIngestor)
    monkeypatch.setattr(ingestion_coordinator_module, "IngestionOrchestrator", _StubOrchestrator)
    monkeypatch.setattr("ml.data.ingest.state.load_state", lambda _path: object())
    monkeypatch.setattr(
        "ml.data.ingest.state.save_state",
        lambda path, state: save_calls.append((str(path), state)),
    )
    monkeypatch.setattr(
        ingestion_coordinator_module,
        "execute_backfill_plan",
        lambda **kwargs: (
            execute_calls.append(dict(kwargs))
            or SimpleNamespace(total_windows=2, processed_bindings=())
        ),
    )

    config = IngestBackfillRuntimeConfig(
        db="postgresql://user:pass@localhost:5432/ml",
        dataset_id="EQUS.MINI",
        schema="ohlcv-1m",
        instruments=("QQQ.XNAS",),
        lookback_days=1,
        state_path="checkpoints/test_state.json",
        coverage_mode="sql",
        write_mode="sql",
        client_mode="noop",
        dry_run=True,
    )

    result = run_ingest_backfill(config, emit=None)

    assert result.total_windows_planned == 2
    assert result.state_saved is False
    assert save_calls == []
    assert execute_calls and execute_calls[0]["state"] is None


def _build_runtime_config(**overrides: object) -> IngestBackfillRuntimeConfig:
    config_kwargs: dict[str, object] = {
        "db": "postgresql://user:pass@localhost:5432/ml",
        "dataset_id": "EQUS.MINI",
        "schema": "ohlcv-1m",
        "instruments": ("SPY.XNAS",),
        "lookback_days": 2,
        "catalog_path": "data/catalog",
        "client_mode": "catalog",
    }
    config_kwargs.update(overrides)
    return IngestBackfillRuntimeConfig(**config_kwargs)


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"dataset_id": " "}, "dataset_id must be non-empty"),
        ({"schema": " "}, "schema must be non-empty"),
        ({"lookback_days": 0}, "lookback_days must be >= 1"),
        ({"instruments": ()}, "No instruments provided"),
        ({"table_name": " "}, "table_name must be non-empty"),
        ({"state_path": " "}, "state_path must be non-empty"),
        ({"coverage_mode": "invalid"}, "coverage_mode must be one of"),
        ({"write_mode": "invalid"}, "write_mode must be one of"),
        ({"client_mode": "invalid"}, "client_mode must be one of"),
        ({"coverage_mode": "catalog", "catalog_path": None}, "--catalog-path is required for catalog coverage"),
        (
            {"also_write_catalog": True, "catalog_path": None, "client_mode": "noop"},
            "--also-write-catalog requires --catalog-path",
        ),
    ],
)
def test_runtime_config_validation_errors(override: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _build_runtime_config(**override)


def test_run_ingest_backfill_validates_runtime_requirements(monkeypatch: pytest.MonkeyPatch) -> None:
    class _StubCoverage:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    class _StubWriter:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    class _StubRegistry:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    monkeypatch.setattr("ml.stores.providers.SqlCoverageProvider", _StubCoverage)
    monkeypatch.setattr("ml.stores.providers.CatalogCoverageProvider", _StubCoverage)
    monkeypatch.setattr("ml.stores.providers.SqlMarketDataWriter", _StubWriter)
    monkeypatch.setattr("ml.registry.data_registry.DataRegistry", _StubRegistry)

    with pytest.raises(ValueError, match="--db is required for SQL coverage"):
        run_ingest_backfill(_build_runtime_config(db=None, coverage_mode="sql", client_mode="noop"))

    with pytest.raises(ValueError, match="parquet write-mode not implemented"):
        run_ingest_backfill(_build_runtime_config(write_mode="parquet", client_mode="noop"))

    monkeypatch.delenv("DATABENTO_API_KEY", raising=False)
    with pytest.raises(ValueError, match="--api-key"):
        run_ingest_backfill(
            _build_runtime_config(client_mode="databento", api_key=None, coverage_mode="sql"),
        )


def test_run_ingest_backfill_noop_client_path(monkeypatch: pytest.MonkeyPatch) -> None:
    class _StubCoverage:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    class _StubWriter:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    class _StubRegistry:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    class _StubIngestor:
        def __init__(self, client: object) -> None:
            self.client = client

    class _StubOrchestrator:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    observed_empties: list[bool] = []

    monkeypatch.setattr("ml.stores.providers.SqlCoverageProvider", _StubCoverage)
    monkeypatch.setattr("ml.stores.providers.SqlMarketDataWriter", _StubWriter)
    monkeypatch.setattr("ml.registry.data_registry.DataRegistry", _StubRegistry)
    monkeypatch.setattr(ingestion_coordinator_module, "DatabentoIngestor", _StubIngestor)
    monkeypatch.setattr(ingestion_coordinator_module, "IngestionOrchestrator", _StubOrchestrator)
    monkeypatch.setattr("ml.data.ingest.state.load_state", lambda _path: object())
    monkeypatch.setattr("ml.data.ingest.state.save_state", lambda _path, _state: None)

    def _execute_plan(**kwargs: object) -> SimpleNamespace:
        orchestrator = kwargs["orchestrator"]
        ingestor = orchestrator.kwargs["ingestor"]  # type: ignore[index]
        dataframe = ingestor.client.get_data(  # type: ignore[attr-defined]
            dataset="EQUS.MINI",
            symbols=["SPY"],
            schema="ohlcv-1m",
            start="2025-01-01",
            end="2025-01-02",
        )
        observed_empties.append(bool(getattr(dataframe, "empty", False)))
        return SimpleNamespace(total_windows=1, processed_bindings=())

    monkeypatch.setattr(ingestion_coordinator_module, "execute_backfill_plan", _execute_plan)

    result = run_ingest_backfill(
        _build_runtime_config(client_mode="noop", coverage_mode="sql"),
        emit=None,
    )

    assert result.total_windows_planned == 1
    assert observed_empties == [True]


def test_run_pre_ingestion_uses_scheduler(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: dict[str, object] = {}

    class _Catalog:
        def __init__(self, path: str) -> None:
            calls["catalog_path"] = path

    class _Scheduler:
        def __init__(self, **kwargs: object) -> None:
            calls["scheduler_kwargs"] = kwargs

        def run_daily_update(self) -> None:
            calls["ran"] = True

    monkeypatch.setattr("nautilus_trader.persistence.catalog.parquet.ParquetDataCatalog", _Catalog)
    monkeypatch.setattr("ml.data.scheduler.DataScheduler", _Scheduler)

    test_coordinator = IngestionCoordinator(
        coverage=MagicMock(),
        writer=MagicMock(),
        registry=MagicMock(),
        ingestor=MagicMock(),
    )
    options = PreIngestionOptions(use_orchestrator=True, dual_write=True, metrics_port=9999)
    test_coordinator.run_pre_ingestion(
        catalog_path=tmp_path / "catalog",
        scheduler_cfg=SchedulerConfig(),
        options=options,
    )

    assert calls["catalog_path"] == str(tmp_path / "catalog")
    scheduler_kwargs = calls["scheduler_kwargs"]
    assert isinstance(scheduler_kwargs, dict)
    assert scheduler_kwargs["dual_write"] is True  # type: ignore[index]
    assert scheduler_kwargs["metrics_port"] == 9999  # type: ignore[index]
    assert calls["ran"] is True


def test_create_ingestion_orchestrator_requires_dependencies() -> None:
    with pytest.raises(RuntimeError, match="Ingestor is not configured"):
        IngestionCoordinator(coverage=MagicMock(), writer=MagicMock(), registry=MagicMock())._create_ingestion_orchestrator()

    with pytest.raises(RuntimeError, match="Coverage provider is required"):
        IngestionCoordinator(writer=MagicMock(), registry=MagicMock(), ingestor=MagicMock())._create_ingestion_orchestrator()

    with pytest.raises(RuntimeError, match="Market data writer is required"):
        IngestionCoordinator(coverage=MagicMock(), registry=MagicMock(), ingestor=MagicMock())._create_ingestion_orchestrator()

    with pytest.raises(RuntimeError, match="Data registry is required"):
        IngestionCoordinator(coverage=MagicMock(), writer=MagicMock(), ingestor=MagicMock())._create_ingestion_orchestrator()


def test_resolve_policy_and_auto_fill_dispatch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class _PolicyOverride(CoveragePolicy):
        @classmethod
        def from_env(cls) -> CoveragePolicy:
            return cls(max_days=5)

    class _FailPolicy(CoveragePolicy):
        @classmethod
        def from_env(cls) -> CoveragePolicy:
            raise RuntimeError("boom")

    monkeypatch.setitem(
        sys.modules,
        "ml.orchestration.pipeline_orchestrator",
        SimpleNamespace(CoveragePolicy=_PolicyOverride),
    )
    resolved = IngestionCoordinator._resolve_coverage_policy()
    assert isinstance(resolved, CoveragePolicy)
    assert resolved.max_days == 5

    monkeypatch.setitem(
        sys.modules,
        "ml.orchestration.pipeline_orchestrator",
        SimpleNamespace(CoveragePolicy=_FailPolicy),
    )
    fallback = IngestionCoordinator._resolve_coverage_policy()
    assert isinstance(fallback, CoveragePolicy)

    dataset_cfg = DatasetBuildConfig(
        data_dir=str(tmp_path / "data"),
        symbols="SPY",
        out_dir=str(tmp_path / "out"),
        target_semantics=build_default_target_semantics_payload(),
        vintage_policy=VintagePolicy.REAL_TIME,
    )
    auto_fill_cfg = AutoFillUniverseConfig(
        enabled=True,
        dataset_id="EQUS.MINI",
        include_bars=True,
        include_tbbo=True,
        include_trades=True,
        include_l2=True,
        include_l3=True,
        instrument_ids=("SPY.XNAS",),
    )

    test_coordinator = IngestionCoordinator(
        coverage=MagicMock(),
        writer=MagicMock(),
        registry=MagicMock(),
        ingestor=MagicMock(),
    )
    dispatched: list[tuple[str, str, int]] = []
    test_coordinator._auto_fill_schema = lambda **kwargs: dispatched.append(  # type: ignore[method-assign]
        (str(kwargs["schema"]), str(kwargs["instrument_id"]), int(kwargs["lookback_days"])),
    )
    test_coordinator._auto_fill_l2 = lambda **kwargs: dispatched.append(("l2", "SPY.XNAS", 1))  # type: ignore[method-assign]
    test_coordinator._auto_fill_l3 = lambda **kwargs: dispatched.append(("l3", "SPY.XNAS", 1))  # type: ignore[method-assign]

    monkeypatch.setattr(ingestion_coordinator_module, "get_max_lookback_days", lambda _dataset, _policy: 3)
    test_coordinator.auto_fill_universe(
        dataset_cfg,
        auto_fill_cfg,
        resolve_instrument_ids_fn=lambda _cfg, _override: ("SPY.XNAS",),
    )

    assert ("ohlcv-1m", "SPY.XNAS", 3) in dispatched
    assert ("quotes", "SPY.XNAS", 3) in dispatched
    assert ("trades", "SPY.XNAS", 3) in dispatched
    assert ("l2", "SPY.XNAS", 1) in dispatched
    assert ("l3", "SPY.XNAS", 1) in dispatched


def test_remaining_gaps_registration_checkpoint_and_state(tmp_path: Path) -> None:
    test_coordinator = IngestionCoordinator()
    assert test_coordinator._remaining_coverage_gaps(
        dataset_id="EQUS.MINI",
        schema="bars",
        instrument_id="SPY.XNAS",
        lookback_days=2,
    ) == []

    test_coordinator.coverage = SimpleNamespace(
        read_bucket_coverage=lambda **_kwargs: set(),
    )
    gaps = test_coordinator._remaining_coverage_gaps(
        dataset_id="EQUS.MINI",
        schema="bars",
        instrument_id="SPY.XNAS",
        lookback_days=1,
    )
    assert isinstance(gaps, list)

    checkpoint = tmp_path / "checkpoint.json"
    test_coordinator._create_ingestion_checkpoint(
        checkpoint_path=checkpoint,
        rows_written=12,
        current_instrument_index=1,
        progress=0.5,
    )
    restored = test_coordinator._restore_from_checkpoint(checkpoint_path=checkpoint)
    assert restored["rows_written"] == 12
    assert restored["current_instrument_index"] == 1

    checkpoint.write_text("{broken-json", encoding="utf-8")
    fallback = test_coordinator._restore_from_checkpoint(checkpoint_path=checkpoint)
    assert fallback["rows_written"] == 0

    registry = MagicMock()
    registry.get_manifest.side_effect = RuntimeError("missing")
    test_coordinator.registry = registry
    test_coordinator.write_mode_tokens = ("sql",)
    test_coordinator._ensure_dataset_registered(
        dataset_id="EQUS.MINI",
        dataset_type=DatasetType.BARS,
        location=str(tmp_path),
    )
    registry.register_dataset.assert_called_once()

    state_before = test_coordinator._get_ingestion_state()
    assert state_before["last_ts_ns_by_instrument"] == {}
    test_coordinator._update_ingestion_state(
        rows_written=5,
        current_instrument="SPY.XNAS",
        ts_ns=12345,
    )
    state_after = test_coordinator._get_ingestion_state()
    assert cast(dict[str, int], state_after["last_ts_ns_by_instrument"])["SPY.XNAS"] == 12345


def test_validate_emit_events_and_fallback_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_coordinator = IngestionCoordinator()

    is_valid, errors = test_coordinator._validate_ingestion_data(data=None, instrument_id="SPY.XNAS")
    assert is_valid is False
    assert errors == ["Data is None"]

    class _Validator:
        def __init__(self, data_registry: object) -> None:
            self.data_registry = data_registry

        def preflight_check(
            self,
            *,
            dataset_id: str,
            data: object,
            strict: bool,
        ) -> tuple[bool, str | None, dict[str, object]]:
            del dataset_id, data, strict
            return True, None, {}

    monkeypatch.setattr("ml.stores.common.schema_validator.SchemaValidatorComponent", _Validator)
    test_coordinator._data_registry = object()
    is_valid, errors = test_coordinator._validate_ingestion_data(
        data={"ts_event": [1, None]},
        instrument_id="SPY.XNAS",
    )
    assert is_valid is False
    assert "Missing required keys: ['instrument_id']" in errors
    assert "ts_event contains null values" in errors

    test_coordinator._message_bus = None
    test_coordinator._emit_ingestion_event(
        event_type="ingestion_started",
        dataset_id="EQUS.MINI",
        rows_written=0,
    )

    published: list[tuple[str, dict[str, object]]] = []
    test_coordinator._message_bus = SimpleNamespace(publish=lambda topic, payload: published.append((topic, payload)))
    monkeypatch.setattr("ml.common.message_topics.build_topic_for_stage", lambda *_args, **_kwargs: "ml.topic")
    monkeypatch.setattr(
        "ml.config.bus.MessageBusConfig.from_env",
        lambda: SimpleNamespace(scheme="events", topic_prefix="ml"),
    )
    test_coordinator._emit_ingestion_event(
        event_type="ingestion_completed",
        dataset_id="EQUS.MINI",
        rows_written=25,
        instrument_id="SPY.XNAS",
        status="partial",
    )
    assert published and published[0][0] == "ml.topic"
    assert published[0][1]["status"] == "partial"
    assert published[0][1]["instrument_id"] == "SPY.XNAS"

    parquet_dir = Path("data/tier1")
    parquet_dir.mkdir(parents=True, exist_ok=True)
    parquet_file = parquet_dir / "SPY_test.parquet"
    parquet_file.write_text("x", encoding="utf-8")
    try:
        fallback = test_coordinator._handle_ingestion_fallback(
            dataset_id="EQUS.MINI",
            schema="bars",
            instrument_ids=["SPY.XNAS"],
            lookback_days=1,
            level="file",
        )
        assert fallback["fallback_level"] == "file"
        assert fallback["rows_written"] == 1
    finally:
        if parquet_file.exists():
            parquet_file.unlink()

    test_coordinator._data_store = None
    assert (
        test_coordinator.ingest_earnings_data(
            symbol="AAPL",
            start_date="2025-01-01",
            end_date="2025-01-02",
        )
        == 0
    )

    class _MacroResult:
        fred_refreshed = True
        alfred_refreshed = False
        fred_error = None
        alfred_error = None

    monkeypatch.setattr("ml.data.ingest.macro_refresh.ensure_macro_ready", lambda **_kwargs: _MacroResult())
    assert (
        test_coordinator.ingest_from_fred(
            series_ids=["DGS10"],
            start_date="2025-01-01",
            end_date="2025-01-31",
        )
        == 1
    )

    monkeypatch.setattr(
        "ml.data.ingest.macro_refresh.ensure_macro_ready",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert (
        test_coordinator.ingest_from_fred(
            series_ids=["DGS10"],
            start_date="2025-01-01",
            end_date="2025-01-31",
        )
        == 0
    )


class _MetricHandle:
    def __init__(
        self,
        *,
        increments: list[dict[str, str]],
        observations: list[tuple[dict[str, str], float]],
        labels: dict[str, str],
    ) -> None:
        self._increments = increments
        self._observations = observations
        self._labels = labels

    def inc(self) -> None:
        self._increments.append(dict(self._labels))

    def observe(self, value: float) -> None:
        self._observations.append((dict(self._labels), value))


class _MetricProbe:
    def __init__(self) -> None:
        self.increments: list[dict[str, str]] = []
        self.observations: list[tuple[dict[str, str], float]] = []

    def labels(self, **labels: str) -> _MetricHandle:
        return _MetricHandle(
            increments=self.increments,
            observations=self.observations,
            labels=labels,
        )


class _MetricBundle:
    def __init__(self) -> None:
        self.operations_total = _MetricProbe()
        self.latency_seconds = _MetricProbe()


def _build_dataset_config(
    tmp_path: Path,
    *,
    market_dataset_id: str | None = None,
) -> DatasetBuildConfig:
    return DatasetBuildConfig(
        data_dir=str(tmp_path / "data"),
        symbols="SPY",
        out_dir=str(tmp_path / "out"),
        target_semantics=build_default_target_semantics_payload(),
        vintage_policy=VintagePolicy.REAL_TIME,
        market_dataset_id=market_dataset_id,
    )


def test_run_ingest_backfill_catalog_modes_and_catalog_writer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _CatalogCoverage:
        instances: list[_CatalogCoverage] = []

        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            _CatalogCoverage.instances.append(self)

    class _StubWriter:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    class _StubRegistry:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    class _StubIngestor:
        def __init__(self, client: object) -> None:
            self.client = client

    orchestrator_kwargs: list[dict[str, object]] = []

    class _StubOrchestrator:
        def __init__(self, **kwargs: object) -> None:
            orchestrator_kwargs.append(dict(kwargs))

    class _StubCatalog:
        def __init__(self, path: str) -> None:
            self.path = path

    class _StubRawWriter:
        def __init__(self, catalog: object) -> None:
            self.catalog = catalog

    monkeypatch.setattr("ml.stores.providers.CatalogCoverageProvider", _CatalogCoverage)
    monkeypatch.setattr("ml.stores.providers.SqlMarketDataWriter", _StubWriter)
    monkeypatch.setattr("ml.registry.data_registry.DataRegistry", _StubRegistry)
    monkeypatch.setattr(ingestion_coordinator_module, "DatabentoIngestor", _StubIngestor)
    monkeypatch.setattr(ingestion_coordinator_module, "IngestionOrchestrator", _StubOrchestrator)
    monkeypatch.setattr("nautilus_trader.persistence.catalog.parquet.ParquetDataCatalog", _StubCatalog)
    monkeypatch.setattr("ml.stores.io_raw.ParquetCatalogRawWriter", _StubRawWriter)
    monkeypatch.setattr("ml.data.ingest.state.load_state", lambda _path: {"state": "ok"})
    monkeypatch.setattr("ml.data.ingest.state.save_state", lambda _path, _state: None)
    monkeypatch.setattr(
        ingestion_coordinator_module,
        "execute_backfill_plan",
        lambda **_kwargs: SimpleNamespace(total_windows=7, processed_bindings=()),
    )

    catalog_path = tmp_path / "catalog"
    config = IngestBackfillRuntimeConfig(
        db="postgresql://user:pass@localhost:5432/ml",
        dataset_id="EQUS.MINI",
        schema="ohlcv-1m",
        instruments=("SPY.XNAS",),
        lookback_days=2,
        catalog_path=str(catalog_path),
        coverage_mode="catalog",
        client_mode="catalog",
        also_write_catalog=True,
    )

    result = run_ingest_backfill(config)

    assert result.total_windows_planned == 7
    assert _CatalogCoverage.instances
    assert _CatalogCoverage.instances[0].kwargs["catalog_path"] == str(catalog_path)
    assert orchestrator_kwargs
    assert isinstance(orchestrator_kwargs[0]["raw_writer"], _StubRawWriter)


def test_run_ingest_backfill_databento_service_init_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _StubCoverage:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    class _StubWriter:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    class _StubRegistry:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    monkeypatch.setattr("ml.stores.providers.SqlCoverageProvider", _StubCoverage)
    monkeypatch.setattr("ml.stores.providers.SqlMarketDataWriter", _StubWriter)
    monkeypatch.setattr("ml.registry.data_registry.DataRegistry", _StubRegistry)

    def _raise_service_error() -> object:
        raise RuntimeError("service boom")

    monkeypatch.setattr("ml.data.ingest.api.ensure_service", _raise_service_error)

    config = IngestBackfillRuntimeConfig(
        db="postgresql://user:pass@localhost:5432/ml",
        dataset_id="EQUS.MINI",
        schema="ohlcv-1m",
        instruments=("SPY.XNAS",),
        lookback_days=1,
        coverage_mode="sql",
        client_mode="databento",
        api_key="test-key",
    )

    with pytest.raises(RuntimeError, match="Failed to initialise Databento ingestion service"):
        run_ingest_backfill(config)


def test_create_ingestion_orchestrator_uses_pipeline_override_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created_kwargs: list[dict[str, object]] = []

    class _OverrideOrchestrator:
        def __init__(self, **kwargs: object) -> None:
            created_kwargs.append(dict(kwargs))

    monkeypatch.setattr(
        "ml.orchestration.pipeline_orchestrator.IngestionOrchestrator",
        _OverrideOrchestrator,
    )

    test_coordinator = IngestionCoordinator(
        coverage=MagicMock(),
        writer=MagicMock(),
        registry=MagicMock(),
        ingestor=MagicMock(),
        service=MagicMock(),
    )
    orchestrator = test_coordinator._create_ingestion_orchestrator()

    assert isinstance(orchestrator, _OverrideOrchestrator)
    assert created_kwargs
    assert created_kwargs[0]["service"] is test_coordinator.service


def test_auto_fill_universe_skips_disabled_or_empty_resolution(tmp_path: Path) -> None:
    test_coordinator = IngestionCoordinator(
        coverage=MagicMock(),
        writer=MagicMock(),
        registry=MagicMock(),
        ingestor=MagicMock(),
    )
    dispatched: list[str] = []
    test_coordinator._auto_fill_schema = lambda **_kwargs: dispatched.append("schema")  # type: ignore[method-assign]
    test_coordinator._auto_fill_l2 = lambda **_kwargs: dispatched.append("l2")  # type: ignore[method-assign]
    test_coordinator._auto_fill_l3 = lambda **_kwargs: dispatched.append("l3")  # type: ignore[method-assign]

    test_coordinator.auto_fill_universe(
        _build_dataset_config(tmp_path),
        AutoFillUniverseConfig(enabled=False),
        resolve_instrument_ids_fn=lambda _cfg, _override: ("SPY.XNAS",),
    )
    assert dispatched == []

    test_coordinator.auto_fill_universe(
        _build_dataset_config(tmp_path),
        AutoFillUniverseConfig(enabled=True, include_l2=False, include_l3=False),
        resolve_instrument_ids_fn=lambda _cfg, _override: (),
    )
    assert dispatched == []


def test_auto_fill_schema_skips_non_positive_and_missing_ingestion_clients(
    tmp_path: Path,
) -> None:
    test_coordinator = IngestionCoordinator(
        coverage=MagicMock(),
        writer=MagicMock(),
        registry=MagicMock(),
        ingestor=MagicMock(),
    )
    metrics = _MetricBundle()
    test_coordinator._auto_fill_schema(
        dataset_id="EQUS.MINI",
        schema="bars",
        instrument_id="SPY.XNAS",
        lookback_days=0,
        metrics=metrics,  # type: ignore[arg-type]
        dataset_cfg=_build_dataset_config(tmp_path),
        processed_bindings=set(),
    )

    test_coordinator.ingestor = None
    test_coordinator.service = None
    test_coordinator._auto_fill_schema(
        dataset_id="EQUS.MINI",
        schema="bars",
        instrument_id="SPY.XNAS",
        lookback_days=2,
        metrics=metrics,  # type: ignore[arg-type]
        dataset_cfg=_build_dataset_config(tmp_path),
        processed_bindings=set(),
    )

    assert len(metrics.operations_total.increments) == 2
    assert all(item["status"] == "skipped" for item in metrics.operations_total.increments)


def test_auto_fill_schema_binding_discovery_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    test_coordinator = IngestionCoordinator(
        coverage=MagicMock(),
        writer=MagicMock(),
        registry=MagicMock(),
        ingestor=MagicMock(),
    )
    test_coordinator._ensure_dataset_registered = lambda **_kwargs: None  # type: ignore[method-assign]
    test_coordinator._remaining_coverage_gaps = lambda **_kwargs: []  # type: ignore[method-assign]
    metrics = _MetricBundle()

    primary_binding = SimpleNamespace(
        binding_id="primary-binding",
        dataset_id="EQUS.MINI",
        schema="bars",
        source="configured",
        storage_kind=StorageKind.PARQUET,
    )
    fallback_binding = SimpleNamespace(
        binding_id="fallback-binding",
        dataset_id="EQUS.MINI",
        schema="bars",
        source="discovered",
        storage_kind=StorageKind.PARQUET,
    )

    monkeypatch.setattr(
        ingestion_coordinator_module.IngestionOrchestrator,
        "resolve_market_bindings",
        lambda **_kwargs: (primary_binding,),
    )
    test_coordinator.discovery_client = SimpleNamespace(
        discover_binding_for_symbol=lambda **_kwargs: fallback_binding,
    )
    test_coordinator.backfill_binding = MagicMock(
        side_effect=[
            {
                "SPY.XNAS": BackfillWindowList(
                    persisted=(),
                    requested=((1, 2),),
                    frames_written=0,
                    rows_written=0,
                ),
            },
            {
                "SPY.XNAS": BackfillWindowList(
                    persisted=((1, 2),),
                    requested=((1, 2),),
                    frames_written=1,
                    rows_written=5,
                ),
            },
        ],
    )

    processed_bindings: set[tuple[str, str]] = set()
    test_coordinator._auto_fill_schema(
        dataset_id="EQUS.MINI",
        schema="bars",
        instrument_id="SPY.XNAS",
        lookback_days=1,
        metrics=metrics,  # type: ignore[arg-type]
        dataset_cfg=_build_dataset_config(tmp_path, market_dataset_id="EQUS.MINI"),
        processed_bindings=processed_bindings,
    )

    assert cast(MagicMock, test_coordinator.backfill_binding).call_count == 2
    assert ("fallback-binding", "bars") in processed_bindings
    assert metrics.operations_total.increments[-1]["status"] == "success"


def test_auto_fill_schema_binding_aggregate_and_partial_status(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    test_coordinator = IngestionCoordinator(
        coverage=MagicMock(),
        writer=MagicMock(),
        registry=MagicMock(),
        ingestor=MagicMock(),
    )
    test_coordinator._ensure_dataset_registered = lambda **_kwargs: None  # type: ignore[method-assign]
    test_coordinator._remaining_coverage_gaps = lambda **_kwargs: [(1, 2)]  # type: ignore[method-assign]
    metrics = _MetricBundle()

    binding = SimpleNamespace(
        binding_id="binding-1",
        dataset_id="EQUS.MINI",
        schema="bars",
        source="configured",
        storage_kind=StorageKind.PARQUET,
    )
    monkeypatch.setattr(
        ingestion_coordinator_module.IngestionOrchestrator,
        "resolve_market_bindings",
        lambda **_kwargs: (binding,),
    )
    test_coordinator.backfill_binding = MagicMock(
        return_value={
            "QQQ.XNAS": BackfillWindowList(
                persisted=((1, 2),),
                requested=((1, 2),),
                frames_written=1,
                rows_written=3,
            ),
        },
    )

    test_coordinator._auto_fill_schema(
        dataset_id="EQUS.MINI",
        schema="bars",
        instrument_id="SPY.XNAS",
        lookback_days=2,
        metrics=metrics,  # type: ignore[arg-type]
        dataset_cfg=_build_dataset_config(tmp_path, market_dataset_id="EQUS.MINI"),
        processed_bindings=set(),
    )

    assert metrics.operations_total.increments[-1]["status"] == "partial"


def test_resolve_populate_l2_default_and_override(monkeypatch: pytest.MonkeyPatch) -> None:
    def _override(config: object) -> object:
        return config

    monkeypatch.setitem(
        sys.modules,
        "ml.orchestration.pipeline_orchestrator",
        SimpleNamespace(populate_l2_efficient=_override),
    )
    assert IngestionCoordinator._resolve_populate_l2() is _override

    monkeypatch.setitem(
        sys.modules,
        "ml.orchestration.pipeline_orchestrator",
        SimpleNamespace(),
    )
    assert IngestionCoordinator._resolve_populate_l2() is ingestion_coordinator_module.populate_l2_efficient


def test_auto_fill_l2_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    test_coordinator = IngestionCoordinator()
    test_coordinator._ensure_dataset_registered = lambda **_kwargs: None  # type: ignore[method-assign]
    metrics = _MetricBundle()
    dataset_cfg = _build_dataset_config(tmp_path)
    auto_fill_cfg = AutoFillUniverseConfig(enabled=True, include_l2=True, l2_progress_file=str(tmp_path / "progress" / "l2.json"))

    test_coordinator._auto_fill_l2(
        dataset_cfg=dataset_cfg,
        auto_fill_cfg=auto_fill_cfg,
        instruments=("",),
        metrics=metrics,  # type: ignore[arg-type]
        policy=CoveragePolicy(max_days=5),
    )
    assert metrics.operations_total.increments[-1]["status"] == "skipped"

    monkeypatch.setattr(ingestion_coordinator_module, "get_max_lookback_days", lambda _dataset, _policy: 4)
    captured_configs: list[object] = []

    def _populate(config: object) -> object:
        captured_configs.append(config)
        return config

    test_coordinator._resolve_populate_l2 = lambda: _populate  # type: ignore[method-assign]
    test_coordinator._auto_fill_l2(
        dataset_cfg=dataset_cfg,
        auto_fill_cfg=auto_fill_cfg,
        instruments=("SPY.XNAS", "SPY.XNAS", "qqq"),
        metrics=metrics,  # type: ignore[arg-type]
        policy=CoveragePolicy(max_days=5),
    )

    assert captured_configs
    config_obj = captured_configs[0]
    assert getattr(config_obj, "symbols") == ("SPY", "QQQ")
    assert getattr(config_obj, "days") == 4

    def _raise_populate(_config: object) -> object:
        raise RuntimeError("l2 boom")

    test_coordinator._resolve_populate_l2 = lambda: _raise_populate  # type: ignore[method-assign]
    test_coordinator._auto_fill_l2(
        dataset_cfg=dataset_cfg,
        auto_fill_cfg=auto_fill_cfg,
        instruments=("SPY.XNAS",),
        metrics=metrics,  # type: ignore[arg-type]
        policy=CoveragePolicy(max_days=5),
    )
    assert metrics.operations_total.increments[-1]["status"] == "error"


def test_auto_fill_l3_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    test_coordinator = IngestionCoordinator()
    test_coordinator._ensure_dataset_registered = lambda **_kwargs: None  # type: ignore[method-assign]
    metrics = _MetricBundle()
    dataset_cfg = _build_dataset_config(tmp_path)
    auto_fill_cfg = AutoFillUniverseConfig(enabled=True, include_l3=True)
    policy = CoveragePolicy(max_days=5)

    def _raise_import(_module: str) -> object:
        raise ImportError("missing")

    monkeypatch.setattr(ingestion_coordinator_module.importlib, "import_module", _raise_import)
    test_coordinator._auto_fill_l3(
        dataset_cfg=dataset_cfg,
        auto_fill_cfg=auto_fill_cfg,
        instruments=("SPY.XNAS",),
        metrics=metrics,  # type: ignore[arg-type]
        policy=policy,
    )
    assert metrics.operations_total.increments[-1]["status"] == "skipped"

    monkeypatch.setattr(
        ingestion_coordinator_module.importlib,
        "import_module",
        lambda _module: SimpleNamespace(PopulateL3TaskConfig=None, populate_l3_efficient=None),
    )
    test_coordinator._auto_fill_l3(
        dataset_cfg=dataset_cfg,
        auto_fill_cfg=auto_fill_cfg,
        instruments=("SPY.XNAS",),
        metrics=metrics,  # type: ignore[arg-type]
        policy=policy,
    )
    assert metrics.operations_total.increments[-1]["status"] == "skipped"

    class _L3Config:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    captured: list[_L3Config] = []

    def _populate_ok(config: _L3Config) -> object:
        captured.append(config)
        return None

    monkeypatch.setattr(
        ingestion_coordinator_module.importlib,
        "import_module",
        lambda _module: SimpleNamespace(
            PopulateL3TaskConfig=_L3Config,
            populate_l3_efficient=_populate_ok,
        ),
    )
    test_coordinator._auto_fill_l3(
        dataset_cfg=dataset_cfg,
        auto_fill_cfg=auto_fill_cfg,
        instruments=("SPY.XNAS",),
        metrics=metrics,  # type: ignore[arg-type]
        policy=policy,
    )
    assert captured
    assert metrics.operations_total.increments[-1]["status"] == "success"

    def _populate_error(_config: _L3Config) -> object:
        raise RuntimeError("l3 boom")

    monkeypatch.setattr(
        ingestion_coordinator_module.importlib,
        "import_module",
        lambda _module: SimpleNamespace(
            PopulateL3TaskConfig=_L3Config,
            populate_l3_efficient=_populate_error,
        ),
    )
    test_coordinator._auto_fill_l3(
        dataset_cfg=dataset_cfg,
        auto_fill_cfg=auto_fill_cfg,
        instruments=("SPY.XNAS",),
        metrics=metrics,  # type: ignore[arg-type]
        policy=policy,
    )
    assert metrics.operations_total.increments[-1]["status"] == "error"


def test_coordinate_ingestion_empty_and_policy_zero_uses_input_lookback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_coordinator = IngestionCoordinator()
    result = test_coordinator.coordinate_ingestion(
        dataset_id="EQUS.MINI",
        schema="bars",
        instrument_ids=[],
        lookback_days=3,
    )
    assert result["fallback_level"] == "dummy"
    assert result["rows_written"] == 0

    monkeypatch.setattr(ingestion_coordinator_module, "get_max_lookback_days", lambda _dataset, _policy: 0)
    test_coordinator.backfill = MagicMock(
        return_value=BackfillWindowList(persisted=(), requested=(), frames_written=1, rows_written=2),
    )
    test_coordinator.coordinate_ingestion(
        dataset_id="EQUS.MINI",
        schema="bars",
        instrument_ids=["SPY.XNAS"],
        lookback_days=9,
    )
    cast(MagicMock, test_coordinator.backfill).assert_called_once_with(
        dataset_id="EQUS.MINI",
        schema="bars",
        instrument_id="SPY.XNAS",
        lookback_days=9,
    )


def test_ingest_wrapper_methods_and_state_skip_update(coordinator: IngestionCoordinator) -> None:
    expected = BackfillWindowList(persisted=(), requested=(), frames_written=1, rows_written=3)
    coordinator.backfill = MagicMock(return_value=expected)

    result = coordinator.ingest_from_databento(
        dataset_id="EQUS.MINI",
        schema="bars",
        instrument_id="SPY.XNAS",
        lookback_days=1,
    )
    assert result is expected
    assert coordinator.ingest_from_yahoo(symbol="SPY", start_date="2025-01-01", end_date="2025-01-02") == 0

    before = coordinator._get_ingestion_state()
    coordinator._update_ingestion_state(rows_written=1, current_instrument="SPY.XNAS", ts_ns=None)
    after = coordinator._get_ingestion_state()
    assert before == after


def test_ingest_from_fred_uses_config_series_and_error_summary_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, object]] = []

    class _ResultWithErrors:
        fred_refreshed = False
        alfred_refreshed = True
        fred_error = RuntimeError("fred failed")
        alfred_error = None

    def _ensure_macro_ready(**kwargs: object) -> _ResultWithErrors:
        captured.append(dict(kwargs))
        return _ResultWithErrors()

    monkeypatch.setattr("ml.data.ingest.macro_refresh.ensure_macro_ready", _ensure_macro_ready)

    test_coordinator = IngestionCoordinator(
        macro_config=MacroIngestionConfig(
            fred_path="fred.json",
            vintage_dir="vintage",
            series_ids=("DGS10",),
        ),
    )
    result = test_coordinator.ingest_from_fred(
        series_ids=[],
        start_date="2025-01-01",
        end_date="2025-02-01",
    )

    assert result == 1
    assert captured
    assert captured[0]["series_ids"] == ("DGS10",)


def test_ingest_earnings_data_success_and_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    service_calls: list[dict[str, object]] = []

    class _ServiceResult:
        actuals_written = 2
        estimates_written = 3
        duration_seconds = 0.5
        failures: tuple[str, ...] = ()

    class _Service:
        def __init__(self, *, config: object, writer: object) -> None:
            service_calls.append({"config": config, "writer": writer})

        def run(self) -> _ServiceResult:
            return _ServiceResult()

    monkeypatch.setattr(
        "ml.features.earnings.ingestion.service.EarningsIngestionService",
        _Service,
    )
    monkeypatch.setattr(
        "ml.config.earnings_ingestion.DEFAULT_SKIP_ACTUALS_TICKERS",
        ("SPY",),
    )

    test_coordinator = IngestionCoordinator(
        data_store=MagicMock(),
        earnings_config=EarningsCoordinatorConfig(skip_tickers=("QQQ",)),
    )
    assert (
        test_coordinator.ingest_earnings_data(
            symbol="aapl",
            start_date="2025-01-01",
            end_date="2025-01-31",
        )
        == 5
    )
    assert service_calls

    class _FailingService:
        def __init__(self, *, config: object, writer: object) -> None:
            del config, writer

        def run(self) -> _ServiceResult:
            raise RuntimeError("earnings boom")

    monkeypatch.setattr(
        "ml.features.earnings.ingestion.service.EarningsIngestionService",
        _FailingService,
    )
    assert (
        test_coordinator.ingest_earnings_data(
            symbol="aapl",
            start_date="2025-01-01",
            end_date="2025-01-31",
        )
        == 0
    )


def test_handle_ingestion_fallback_unknown_level_and_error_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_coordinator = IngestionCoordinator()

    def _raise_metric_error(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("metric boom")

    monkeypatch.setattr("ml.common.metrics_bootstrap.get_counter", _raise_metric_error)
    result = test_coordinator._handle_ingestion_fallback(
        dataset_id="EQUS.MINI",
        schema="bars",
        instrument_ids=["SPY.XNAS"],
        lookback_days=2,
        level="unknown",
    )
    assert result["fallback_level"] == "dummy"
    assert "Fallback activated" in cast(str, result["error"])

    test_coordinator.backfill = MagicMock(side_effect=RuntimeError("primary failed"))
    test_coordinator.backfill_coverage = MagicMock(side_effect=RuntimeError("cached failed"))
    monkeypatch.setattr("pathlib.Path.exists", lambda _self: False)

    result = test_coordinator._handle_ingestion_fallback(
        dataset_id="EQUS.MINI",
        schema="bars",
        instrument_ids=["SPY.XNAS"],
        lookback_days=2,
        level="primary",
    )
    assert result["fallback_level"] == "dummy"
    assert result["error"] == "cached failed"


def test_checkpoint_write_failure_and_restore_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    test_coordinator = IngestionCoordinator()

    def _write_error(self: Path, _content: str, encoding: str | None = None) -> int:
        del self, _content, encoding
        raise OSError("write failure")

    monkeypatch.setattr(Path, "write_text", _write_error)
    test_coordinator._create_ingestion_checkpoint(
        checkpoint_path=tmp_path / "checkpoint.json",
        rows_written=1,
        current_instrument_index=0,
        progress=0.1,
    )

    restored = test_coordinator._restore_from_checkpoint(
        checkpoint_path=tmp_path / "does-not-exist.json",
    )
    assert restored["rows_written"] == 0
    assert restored["current_instrument_index"] == 0


def test_validate_ingestion_data_dataframe_import_and_exception_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _BoolSeries:
        def __init__(self, value: bool) -> None:
            self._value = value

        def any(self) -> bool:
            return self._value

    class _Series:
        def isna(self) -> _BoolSeries:
            return _BoolSeries(True)

        def min(self) -> int:
            return 5

        def max(self) -> int:
            return 3

    class _Frame:
        columns = ("ts_event",)

        def __getitem__(self, key: str) -> _Series:
            del key
            return _Series()

    class _Validator:
        def __init__(self, data_registry: object) -> None:
            self.data_registry = data_registry

        def preflight_check(
            self,
            *,
            dataset_id: str,
            data: object,
            strict: bool,
        ) -> tuple[bool, str | None, dict[str, object]]:
            del dataset_id, data, strict
            return False, "invalid payload", {}

    test_coordinator = IngestionCoordinator()
    test_coordinator._data_registry = object()
    monkeypatch.setattr("ml.stores.common.schema_validator.SchemaValidatorComponent", _Validator)

    is_valid, errors = test_coordinator._validate_ingestion_data(
        data=_Frame(),
        instrument_id="SPY.XNAS",
    )
    assert is_valid is False
    assert "Missing required columns: ['instrument_id']" in errors
    assert "ts_event contains null values" in errors
    assert "ts_event values are not monotonic" in errors
    assert "Preflight check failed: invalid payload" in errors

    class _ExplodingValidator:
        def __init__(self, data_registry: object) -> None:
            self.data_registry = data_registry

        def preflight_check(
            self,
            *,
            dataset_id: str,
            data: object,
            strict: bool,
        ) -> tuple[bool, str | None, dict[str, object]]:
            del dataset_id, data, strict
            raise RuntimeError("boom")

    monkeypatch.setattr("ml.stores.common.schema_validator.SchemaValidatorComponent", _ExplodingValidator)
    is_valid, errors = test_coordinator._validate_ingestion_data(
        data=_Frame(),
        instrument_id="SPY.XNAS",
    )
    assert is_valid is False
    assert "Missing required columns: ['instrument_id']" in errors

    test_coordinator._data_registry = None
    is_valid, errors = test_coordinator._validate_ingestion_data(
        data=[],
        instrument_id="SPY.XNAS",
    )
    assert is_valid is False
    assert errors == ["Data is empty"]

    class _RaisingValidator:
        def __init__(self, data_registry: object) -> None:
            del data_registry
            raise RuntimeError("validator init failed")

    test_coordinator._data_registry = object()
    monkeypatch.setattr("ml.stores.common.schema_validator.SchemaValidatorComponent", _RaisingValidator)
    is_valid, errors = test_coordinator._validate_ingestion_data(
        data=[],
        instrument_id="SPY.XNAS",
    )
    assert is_valid is False
    assert errors == ["Data is empty"]


def test_emit_ingestion_event_without_publish_and_with_publish_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_coordinator = IngestionCoordinator()
    test_coordinator._message_bus = SimpleNamespace()
    test_coordinator._emit_ingestion_event(
        event_type="ingestion_completed",
        dataset_id="EQUS.MINI",
        rows_written=1,
    )

    monkeypatch.setattr("ml.common.message_topics.build_topic_for_stage", lambda *_args, **_kwargs: "ml.topic")
    monkeypatch.setattr(
        "ml.config.bus.MessageBusConfig.from_env",
        lambda: SimpleNamespace(scheme="events", topic_prefix="ml"),
    )

    def _raise_publish(_topic: str, _payload: dict[str, object]) -> None:
        raise RuntimeError("publish failed")

    test_coordinator._message_bus = SimpleNamespace(publish=_raise_publish)
    test_coordinator._emit_ingestion_event(
        event_type="ingestion_completed",
        dataset_id="EQUS.MINI",
        rows_written=2,
    )


def test_auto_fill_schema_skips_duplicate_binding_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    test_coordinator = IngestionCoordinator(
        coverage=MagicMock(),
        writer=MagicMock(),
        registry=MagicMock(),
        ingestor=MagicMock(),
    )
    metrics = _MetricBundle()
    binding = SimpleNamespace(
        binding_id="binding-duplicate",
        dataset_id="EQUS.MINI",
        schema="bars",
        source="configured",
        storage_kind=StorageKind.PARQUET,
    )
    monkeypatch.setattr(
        ingestion_coordinator_module.IngestionOrchestrator,
        "resolve_market_bindings",
        lambda **_kwargs: (binding,),
    )

    processed_bindings = {("binding-duplicate", "bars")}
    test_coordinator._auto_fill_schema(
        dataset_id="EQUS.MINI",
        schema="bars",
        instrument_id="SPY.XNAS",
        lookback_days=1,
        metrics=metrics,  # type: ignore[arg-type]
        dataset_cfg=_build_dataset_config(
            tmp_path,
            market_dataset_id="EQUS.MINI",
        ),
        processed_bindings=processed_bindings,
    )
    assert metrics.operations_total.increments[-1]["status"] == "skipped"


def test_run_ingest_backfill_catalog_client_requires_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _StubSqlCoverage:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    class _StubWriter:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    class _StubRegistry:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    monkeypatch.setattr("ml.stores.providers.SqlCoverageProvider", _StubSqlCoverage)
    monkeypatch.setattr("ml.stores.providers.SqlMarketDataWriter", _StubWriter)
    monkeypatch.setattr("ml.registry.data_registry.DataRegistry", _StubRegistry)

    config = IngestBackfillRuntimeConfig(
        db="postgresql://user:pass@localhost:5432/ml",
        dataset_id="EQUS.MINI",
        schema="ohlcv-1m",
        instruments=("SPY.XNAS",),
        lookback_days=2,
        coverage_mode="sql",
        client_mode="catalog",
        catalog_path="data/catalog",
    )
    object.__setattr__(config, "catalog_path", None)

    with pytest.raises(ValueError, match="--catalog-path is required for client-mode catalog"):
        run_ingest_backfill(config)


def test_auto_fill_schema_warns_when_market_inputs_resolve_empty(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    test_coordinator = IngestionCoordinator(
        coverage=MagicMock(),
        writer=MagicMock(),
        registry=MagicMock(),
        ingestor=MagicMock(),
    )
    test_coordinator._ensure_dataset_registered = lambda **_kwargs: None  # type: ignore[method-assign]
    test_coordinator._remaining_coverage_gaps = lambda **_kwargs: []  # type: ignore[method-assign]
    test_coordinator.backfill = MagicMock(
        return_value=BackfillWindowList(
            persisted=((1, 2),),
            requested=((1, 2),),
            frames_written=1,
            rows_written=1,
        ),
    )
    metrics = _MetricBundle()

    monkeypatch.setattr(
        ingestion_coordinator_module.IngestionOrchestrator,
        "resolve_market_bindings",
        lambda **_kwargs: (),
    )
    dataset_cfg = DatasetBuildConfig(
        data_dir=str(tmp_path / "data"),
        symbols="SPY",
        out_dir=str(tmp_path / "out"),
        target_semantics=build_default_target_semantics_payload(),
        vintage_policy=VintagePolicy.REAL_TIME,
        market_inputs=(MarketDatasetInput(descriptor_id="EQUS.MINI"),),
    )

    test_coordinator._auto_fill_schema(
        dataset_id="EQUS.MINI",
        schema="bars",
        instrument_id="SPY.XNAS",
        lookback_days=1,
        metrics=metrics,  # type: ignore[arg-type]
        dataset_cfg=dataset_cfg,
        processed_bindings=set(),
    )

    assert metrics.operations_total.increments[-1]["status"] == "success"
