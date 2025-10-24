from __future__ import annotations

import os

import pytest

if os.getenv("ML_ENABLE_COMPONENT_FACADES", "0") != "1":
    pytest.skip("component orchestrator tests disabled", allow_module_level=True)

import argparse
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from typing import Callable
from typing import Sequence
from typing import cast

import json
import numpy as np
import pandas as pd
import pytest

from ml.config.coverage import CoveragePolicy
from ml.config.market_data import MarketDatasetInput
from ml.config.market_data import MarketFeedDescriptor
from ml.data import DatasetMetadata
from ml.data import MarketBindingMetadata
from ml.config.market_data import MarketFeedDescriptorSet
from ml.data.ingest.market_bindings import ResolvedMarketBinding
from ml.data.ingest.orchestrator import BackfillWindowList
from ml.dashboard.services.pipelines_service import PipelineIntegrationService
from ml.data.ingest.service import IngestionError

# Reuse typed configuration structures from config_types to mirror runtime usage.
from ml.orchestration.config_loader import IngestionStageConfig
from ml.orchestration.config_loader import OrchestratorRunConfig
from ml.orchestration.config_loader import Stage
from ml.orchestration.config_loader import TrainingStageConfig
from ml.orchestration.config_types import AutoFillUniverseConfig
from ml.orchestration.config_types import DatasetBuildConfig
from ml.orchestration.config_types import HPOConfig
from ml.orchestration.config_types import IntegrationConfig
from ml.orchestration.config_types import OrchestratorConfig
from ml.orchestration.config_types import StudentDistillConfig
from ml.orchestration.config_types import TeacherTrainConfig
import ml.orchestration.pipeline_orchestrator as pipeline_orchestrator
from ml.orchestration.pipeline_orchestrator import _apply_default_market_inputs
from ml.orchestration.pipeline_orchestrator import _AutoFillMetrics
from ml.orchestration.pipeline_orchestrator import MLPipelineOrchestrator
from ml.orchestration.pipeline_orchestrator import _build_auto_fill_config_from_args
from ml.orchestration.pipeline_orchestrator import _parse_market_inputs_json
from ml.orchestration.pipeline_orchestrator import _resolve_write_mode_tokens
from ml.orchestration.pipeline_orchestrator import parse_args
from ml.registry.dataclasses import DataContract, DatasetManifest, DatasetType, StorageKind
from ml.data.vintage import VintagePolicy
from ml.stores.providers import DAY_NS


@dataclass(slots=True)
class _DiscoveryPayload:
    dataset_id: str
    schema: str
    coverage_start_ns: int
    coverage_end_ns: int
    storage_kind: StorageKind | None = None
    cost_usd: float | None = None
    symbol: str = ""
    requested_symbol: str = ""
    available_start_ns: int | None = None
    available_end_ns: int | None = None

    @property
    def coverage_span_ns(self) -> int:
        return max(self.coverage_end_ns - self.coverage_start_ns, 0)


@dataclass
class _Coverage:
    def read_bucket_coverage(
        self,
        *,
        dataset_id: str,
        schema: str,
        instrument_id: str,
        start_ns: int,
        end_ns: int,
    ) -> set[int]:
        return set()


@dataclass
class _Writer:
    def write(self, *, dataset_id: str, schema: str, instrument_id: str, df: pd.DataFrame) -> int:
        return len(df.index) if df is not None and not df.empty else 0


class _StageRecorder:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.last_config: OrchestratorConfig | None = None

    def run(self, cfg: OrchestratorConfig) -> int:  # pragma: no cover - simple stub
        self.calls.append("run")
        self.last_config = cfg
        return 0

    def run_training_only(self, cfg: OrchestratorConfig) -> int:  # pragma: no cover - simple stub
        self.calls.append("train")
        self.last_config = cfg
        return 0


@dataclass
class _CoverageWithAvailability:
    available: dict[tuple[str, str, str], set[int]]

    def read_bucket_coverage(
        self,
        *,
        dataset_id: str,
        schema: str,
        instrument_id: str,
        start_ns: int,
        end_ns: int,
    ) -> set[int]:
        return self.available.get((dataset_id, schema, instrument_id), set())


@dataclass
class _Registry:
    def emit_event(self, **kwargs: Any) -> None:  # pragma: no cover - stub
        return None

    def update_watermark(self, **kwargs: Any) -> None:  # pragma: no cover - stub
        return None


@dataclass
class _DiscoveryService:
    result: _DiscoveryPayload | None
    calls: list[tuple[str, str]] = field(default_factory=list)

    def discover_symbol_dataset(
        self,
        *,
        symbol: str,
        schema: str,
        start_ns: int,
        end_ns: int,
    ) -> _DiscoveryPayload | None:
        self.calls.append((symbol, schema))
        payload = self.result
        if payload is None:
            return None
        if not payload.symbol:
            payload.symbol = symbol
        if not payload.requested_symbol:
            payload.requested_symbol = symbol
        if payload.available_start_ns is None:
            payload.available_start_ns = start_ns
        if payload.available_end_ns is None:
            payload.available_end_ns = end_ns
        return payload

    def ingest(
        self, request: object, *, on_chunk: Callable[[object], None] | None = None
    ) -> list[object]:
        return []


@dataclass
class _ServiceStub:
    allowed_dataset: str
    calls: list[tuple[str, str]] = field(default_factory=list)

    def get_available_range_ns(
        self,
        *,
        dataset: str,
        schema: str | None = None,
    ) -> tuple[int | None, int | None]:
        self.calls.append((dataset, schema or ""))
        if dataset != self.allowed_dataset:
            raise IngestionError("dataset not permitted")
        return (0, None)

    def estimate_cost_usd(
        self,
        *,
        dataset: str,
        schema: str,
        symbols: Sequence[str],
        start: datetime,
        end: datetime,
    ) -> float:
        del symbols, start, end
        if dataset != self.allowed_dataset:
            raise IngestionError("dataset not permitted")
        return 0.0


@dataclass
class _IngestionOrchStub:
    resolved_bindings: tuple[ResolvedMarketBinding, ...] = ()
    binding_exception: Exception | None = None
    coverage_exception: Exception | None = None
    manual_exception: Exception | None = None
    coverage_windows: list[tuple[int, int]] = field(default_factory=list)
    manual_rows_written: int = 0
    auto_fill_calls: int = 0
    binding_calls: int = 0
    coverage_calls: int = 0
    manual_calls: int = 0

    def _auto_fill_universe(
        self,
        _dataset_cfg: DatasetBuildConfig,
        _auto_fill_cfg: AutoFillUniverseConfig,
    ) -> None:
        self.auto_fill_calls += 1

    def _resolve_market_inputs(
        self,
        _cfg: DatasetBuildConfig,
    ) -> tuple[tuple[MarketDatasetInput, ...] | None, tuple[ResolvedMarketBinding, ...]]:
        return None, self.resolved_bindings

    def backfill_binding(
        self,
        *,
        binding: ResolvedMarketBinding,
        lookback_days: int,
    ) -> dict[str, BackfillWindowList]:
        del lookback_days
        self.binding_calls += 1
        if self.binding_exception is not None:
            raise self.binding_exception
        return {
            binding.symbol: BackfillWindowList(
                rows_written=self.manual_rows_written,
            ),
        }

    def backfill_coverage(
        self,
        *,
        dataset_id: str,
        schema: str,
        instrument_id: str,
        policy: CoveragePolicy | None = None,
    ) -> list[tuple[int, int]]:
        del dataset_id, schema, instrument_id, policy
        self.coverage_calls += 1
        if self.coverage_exception is not None:
            raise self.coverage_exception
        return list(self.coverage_windows)

    def backfill(
        self,
        *,
        dataset_id: str,
        schema: str,
        instrument_id: str,
        lookback_days: int,
    ) -> BackfillWindowList:
        del dataset_id, schema, instrument_id, lookback_days
        self.manual_calls += 1
        if self.manual_exception is not None:
            raise self.manual_exception
        return BackfillWindowList(rows_written=self.manual_rows_written)


@dataclass
class _OrchestratorRecorder:
    run_calls: list[OrchestratorConfig] = field(default_factory=list)
    run_training_calls: list[OrchestratorConfig] = field(default_factory=list)
    ingestor: object = field(default_factory=object)
    service: object = field(default_factory=object)

    def run(self, cfg: OrchestratorConfig) -> int:
        self.run_calls.append(cfg)
        return 0

    def run_training_only(self, cfg: OrchestratorConfig) -> int:
        self.run_training_calls.append(cfg)
        return 0


def _write_dataset_metadata_file(out_dir: Path) -> None:
    metadata = {
        "dataset_id": "tft_dataset",
        "vintage_policy": "real_time",
        "vintage_cutoff": None,
        "build_ts": "2025-01-01T00:00:00",
        "ts_event_start": "2025-01-01T00:00:00",
        "ts_event_end": "2025-01-01T00:00:01",
        "overall_window": ["2025-01-01T00:00:00", "2025-01-01T00:00:01"],
        "train_window": None,
        "validation_window": None,
        "test_window": None,
        "macro_observation_counts": {},
    }
    (out_dir / "dataset_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


@dataclass
class _CapturingRegistry:
    manifests: dict[str, DatasetManifest]
    register_calls: int = 0
    updates: list[dict[str, Any]] = field(default_factory=list)

    def __init__(self) -> None:
        self.manifests = {}
        self.register_calls = 0
        self.updates = []

    def emit_event(self, **kwargs: Any) -> None:  # pragma: no cover - stub
        return None

    def update_watermark(self, **kwargs: Any) -> None:  # pragma: no cover - stub
        return None

    def get_manifest(self, dataset_id: str) -> DatasetManifest:
        try:
            return self.manifests[dataset_id]
        except KeyError as exc:  # pragma: no cover - defensive path
            raise ValueError(dataset_id) from exc

    def register_dataset(self, manifest: DatasetManifest) -> str:
        self.register_calls += 1
        self.manifests[manifest.dataset_id] = manifest
        return manifest.dataset_id

    def get_contract(self, dataset_id: str) -> DataContract:
        del dataset_id
        return cast(DataContract, object())

    def update_manifest(self, dataset_id: str, changes: dict[str, Any]) -> None:
        self.updates.append({"dataset_id": dataset_id, "changes": changes})


@dataclass
class _Ingestor:
    def ingest_time_window(self, **kwargs: Any) -> pd.DataFrame:
        return pd.DataFrame({"ts_event": [1, 2]})


class _CliWrapper:
    def __init__(self, fn: Callable[[list[str] | None], int]) -> None:
        self._fn = fn

    def __call__(self, argv: list[str] | None = None) -> int:
        return self._fn(argv)


def _ok(_: list[str] | None = None) -> int:
    return 0


def test_pipeline_orchestrator_runs_all_phases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    coverage = _Coverage()
    writer = _Writer()
    registry = _Registry()
    ingestor = _Ingestor()

    # record calls
    called: dict[str, int] = {"build": 0, "hpo": 0, "train": 0, "distill": 0}
    feature_registry_dir = tmp_path / "features"
    feature_registry_dir.mkdir(parents=True, exist_ok=True)
    model_registry_dir = tmp_path / "models"
    model_registry_dir.mkdir(parents=True, exist_ok=True)

    def _build(argv: list[str] | None = None) -> int:
        called["build"] += 1
        # simulate dataset.csv emitted
        out_dir = None
        if argv is not None and "--out_dir" in argv:
            out_dir = Path(argv[argv.index("--out_dir") + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "dataset.csv").write_text("id,ts_event\n1,1\n")
            _write_dataset_metadata_file(out_dir)
            _write_dataset_metadata_file(out_dir)
            np.savez(
                out_dir / "features_npz.npz",
                X_train=np.array([[0.1]], dtype=np.float32),
                X_val=np.array([[0.2]], dtype=np.float32),
                feature_names=np.array(["f1"], dtype=object),
            )
            sidecar = {
                "feature_registry_dir": str(feature_registry_dir),
                "feature_set_id": "fs1",
                "feature_names": ["f1"],
            }
            (out_dir / "feature_registration.json").write_text(
                json.dumps(sidecar),
                encoding="utf-8",
            )
        return 0

    def _hpo(argv: list[str] | None = None) -> int:
        called["hpo"] += 1
        return 0

    def _train(argv: list[str] | None = None) -> int:
        called["train"] += 1
        if argv is not None and "--out_dir" in argv:
            out_dir = Path(argv[argv.index("--out_dir") + 1])
            np.savez(out_dir / "teacher_preds.npz", q_train=np.array([0.3], dtype=np.float32))
        return 0

    def _distill(argv: list[str] | None = None) -> int:
        called["distill"] += 1
        return 0

    monkeypatch.setattr("ml.training.distillation.cli.main", _distill)

    orch = MLPipelineOrchestrator(
        coverage=coverage,
        writer=writer,
        registry=registry,
        ingestor=ingestor,
        build_main=_CliWrapper(_build),
        hpo_main=_CliWrapper(_hpo),
        teacher_main=_CliWrapper(_train),
    )

    cfg = OrchestratorConfig(
        dataset=DatasetBuildConfig(
            data_dir=str(tmp_path),
            symbols="SPY.NYSE",
            out_dir=str(tmp_path / "out"),
            include_micro=True,
            feature_registry_dir=str(feature_registry_dir),
        ),
        hpo=HPOConfig(enabled=True, epochs=1, batch_size=8, tail_rows=100, limit_groups=10),
        teacher=TeacherTrainConfig(enabled=True, model_id="teacher_X", max_epochs=1),
        student=StudentDistillConfig(
            enabled=True,
            model_id="student_X",
            model_registry_dir=str(model_registry_dir),
        ),
    )
    rc = orch.run(cfg)
    assert rc == 0
    assert called == {"build": 1, "hpo": 1, "train": 1, "distill": 1}


def test_pipeline_orchestrator_attach_runtime_sets_components(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    coverage = _Coverage()
    writer = _Writer()

    called: dict[str, int] = {"build": 0}

    def _build(argv: list[str] | None = None) -> int:
        called["build"] += 1
        if argv is not None and "--out_dir" in argv:
            out_dir = Path(argv[argv.index("--out_dir") + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "dataset.csv").write_text("id,ts_event\n1,1\n", encoding="utf-8")
            _write_dataset_metadata_file(out_dir)
        return 0

    orch_metrics_calls: list[int] = []
    orch_events_calls: list[int] = []

    def _fake_metrics() -> int:
        orch_metrics_calls.append(1)
        return 0

    def _fake_events() -> int:
        orch_events_calls.append(1)
        return 0

    monkeypatch.setattr("tools.validate_metrics_bootstrap.main", _fake_metrics)
    monkeypatch.setattr("tools.validate_event_constants.main", _fake_events)

    factory_calls: list[dict[str, Any]] = []
    manager_instances: list[_Manager] = []

    class _Manager:
        def __init__(self, **kwargs: Any) -> None:
            factory_calls.append(kwargs)
            self.data_registry = object()
            self.feature_registry = object()
            self.model_registry = object()
            self.strategy_registry = object()
            self.feature_store = object()
            self.model_store = object()
            self.strategy_store = object()
            self.data_store = object()
            self.partition_manager = object()
            manager_instances.append(self)  # pragma: no mutate - tracking

    def _factory(**kwargs: Any) -> _Manager:
        return _Manager(**kwargs)

    orch = MLPipelineOrchestrator(
        coverage=coverage,
        writer=writer,
        data_registry=None,
        ingestor=None,
        build_main=_CliWrapper(_build),
        hpo_main=None,
        teacher_main=_CliWrapper(_ok),
        model_registry=None,
        feature_registry=None,
        integration_manager_factory=_factory,
    )

    cfg = OrchestratorConfig(
        dataset=DatasetBuildConfig(
            data_dir=str(tmp_path),
            symbols="SPY.NYSE",
            out_dir=str(tmp_path / "out"),
        ),
        hpo=HPOConfig(),
        teacher=TeacherTrainConfig(enabled=False),
        integration=IntegrationConfig(
            enabled=True,
            db_connection="postgresql://example",
            auto_start_postgres=True,
            auto_migrate=True,
            ensure_healthy=False,
            strict_protocol_validation=True,
            run_validators=True,
        ),
    )

    rc = orch.run(cfg)
    assert rc == 0
    assert called == {"build": 1}
    assert len(factory_calls) == 1
    assert factory_calls[0]["auto_start_postgres"] is True
    assert factory_calls[0]["auto_migrate"] is True
    assert factory_calls[0]["ensure_healthy"] is False
    assert factory_calls[0]["db_connection"] == "postgresql://example"
    assert factory_calls[0]["strict_protocol_validation"] is True

    manager = manager_instances[0]
    assert orch.data_registry is manager.data_registry
    assert orch.feature_registry is manager.feature_registry
    assert orch.model_registry is manager.model_registry
    assert orch.strategy_registry is manager.strategy_registry
    assert orch.feature_store is manager.feature_store
    assert orch.model_store is manager.model_store
    assert orch.strategy_store is manager.strategy_store
    assert orch.data_store is manager.data_store
    assert orch.partition_manager is manager.partition_manager
    assert orch_metrics_calls and orch_events_calls


def test_pipeline_orchestrator_attach_runtime_skips_validators_when_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    coverage = _Coverage()
    writer = _Writer()

    def _build(argv: list[str] | None = None) -> int:
        if argv is not None and "--out_dir" in argv:
            out_dir = Path(argv[argv.index("--out_dir") + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "dataset.csv").write_text("id,ts_event\n1,1\n", encoding="utf-8")
            _write_dataset_metadata_file(out_dir)
        return 0

    metrics_called = False
    events_called = False

    def _metrics() -> int:
        nonlocal metrics_called
        metrics_called = True
        return 0

    def _events() -> int:
        nonlocal events_called
        events_called = True
        return 0

    monkeypatch.setattr("tools.validate_metrics_bootstrap.main", _metrics)
    monkeypatch.setattr("tools.validate_event_constants.main", _events)

    class _Manager:
        def __init__(self) -> None:
            self.data_registry = object()
            self.feature_registry = object()
            self.model_registry = object()
            self.strategy_registry = object()
            self.feature_store = object()
            self.model_store = object()
            self.strategy_store = object()
            self.data_store = object()
            self.partition_manager = object()

    orch = MLPipelineOrchestrator(
        coverage=coverage,
        writer=writer,
        data_registry=None,
        ingestor=None,
        build_main=_CliWrapper(_build),
        hpo_main=None,
        teacher_main=_CliWrapper(_ok),
        integration_manager_factory=lambda **_: _Manager(),
    )

    cfg = OrchestratorConfig(
        dataset=DatasetBuildConfig(
            data_dir=str(tmp_path),
            symbols="QQQ.NASDAQ",
            out_dir=str(tmp_path / "skip"),
        ),
        hpo=HPOConfig(),
        teacher=TeacherTrainConfig(enabled=False),
        integration=IntegrationConfig(
            enabled=True,
            run_validators=False,
        ),
    )

    rc = orch.run(cfg)
    assert rc == 0
    assert metrics_called is False
    assert events_called is False


def test_auto_fill_universe_backfills_expected_schemas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    coverage = _Coverage()
    writer = _Writer()
    ingestor = _Ingestor()

    out_dir = tmp_path / "out"

    def _build(argv: list[str] | None = None) -> int:
        if argv is None:
            return 0
        if "--out_dir" in argv:
            target = Path(argv[argv.index("--out_dir") + 1])
            target.mkdir(parents=True, exist_ok=True)
            (target / "dataset.csv").write_text("id,ts_event\n1,1\n", encoding="utf-8")
            _write_dataset_metadata_file(target)
        return 0

    orch = MLPipelineOrchestrator(
        coverage=coverage,
        writer=writer,
        data_registry=_Registry(),
        ingestor=ingestor,
        build_main=_CliWrapper(_build),
        hpo_main=None,
        teacher_main=_CliWrapper(_ok),
    )

    backfill_calls: list[tuple[str, str, str, int]] = []

    def _fake_backfill(
        self: MLPipelineOrchestrator,
        *,
        dataset_id: str,
        schema: str,
        instrument_id: str,
        lookback_days: int,
        **_: object,
    ) -> BackfillWindowList:
        backfill_calls.append((dataset_id, schema, instrument_id, lookback_days))
        return BackfillWindowList((), requested=())

    monkeypatch.setattr(MLPipelineOrchestrator, "backfill", _fake_backfill)
    ensure_calls: list[tuple[str, StorageKind]] = []

    def _capture_register(
        self: MLPipelineOrchestrator,
        *,
        dataset_id: str,
        dataset_type: DatasetType,
        location: str,
        storage_kind: StorageKind = StorageKind.PARQUET,
    ) -> None:
        ensure_calls.append((dataset_id, storage_kind))

    monkeypatch.setattr(MLPipelineOrchestrator, "_ensure_dataset_registered", _capture_register)

    target_instrument = "SPY.NYSE"

    def _fake_backfill_binding(
        self: MLPipelineOrchestrator,
        *,
        binding: ResolvedMarketBinding,
        lookback_days: int,
        **_: object,
    ) -> dict[str, BackfillWindowList]:
        instruments = binding.instrument_ids or (binding.symbol,)
        primary = (
            target_instrument
            if target_instrument
            else (instruments[0] if instruments else binding.symbol)
        )
        backfill_calls.append(
            (
                binding.dataset_id,
                binding.schema or "",
                primary,
                lookback_days,
            ),
        )
        return {
            primary: BackfillWindowList(
                (),
                requested=((0, DAY_NS),),
            ),
        }

    monkeypatch.setattr(MLPipelineOrchestrator, "backfill_binding", _fake_backfill_binding)

    l2_configs: list[object] = []

    def _fake_populate_l2(config: object) -> object:
        l2_configs.append(config)
        return object()

    monkeypatch.setattr(
        "ml.orchestration.pipeline_orchestrator.populate_l2_efficient",
        _fake_populate_l2,
    )

    policy = CoveragePolicy(
        l0_max_lookback_days=14,
        l1_max_lookback_days=7,
        l2_max_lookback_days=3,
        l3_max_lookback_days=2,
    )
    monkeypatch.setattr(
        "ml.orchestration.pipeline_orchestrator.CoveragePolicy.from_env",
        staticmethod(lambda: policy),
    )

    cfg = OrchestratorConfig(
        dataset=DatasetBuildConfig(
            data_dir=str(tmp_path),
            symbols=target_instrument,
            out_dir=str(out_dir),
            include_l2=True,
            instrument_ids=(target_instrument,),
        ),
        hpo=HPOConfig(),
        teacher=TeacherTrainConfig(enabled=False),
        auto_fill=AutoFillUniverseConfig(
            enabled=True,
            include_l2=True,
        ),
    )

    rc = orch.run(cfg)
    assert rc == 0
    schemas = {(schema, lookback) for _, schema, _, lookback in backfill_calls}
    assert ("ohlcv-1m", 14) in schemas
    assert ("tbbo", 7) in schemas
    assert ("trades", 7) in schemas
    assert {instrument for _, _, instrument, _ in backfill_calls} == {
        "SPY.NYSE",
    }
    assert l2_configs, "Expected L2 auto-fill to run"
    l2_config = l2_configs[0]
    assert getattr(l2_config, "days") == 3
    assert Path(getattr(l2_config, "data_dir")).samefile(tmp_path)


def test_prepare_dataset_config_discovers_market_inputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    coverage = _Coverage()
    writer = _Writer()
    ingestor = _Ingestor()
    discovery = _DiscoveryPayload(
        dataset_id="DBEQ.BASIC",
        schema="ohlcv-1m",
        coverage_start_ns=0,
        coverage_end_ns=200 * DAY_NS,
        storage_kind=None,
        cost_usd=0.0,
    )
    service = _DiscoveryService(result=discovery)

    orch = MLPipelineOrchestrator(
        coverage=coverage,
        writer=writer,
        data_registry=_Registry(),
        ingestor=ingestor,
        build_main=_CliWrapper(_ok),
        hpo_main=None,
        teacher_main=_CliWrapper(_ok),
        service=service,
    )

    monkeypatch.setattr(
        "ml.orchestration.pipeline_orchestrator.load_market_feed_descriptors",
        lambda: MarketFeedDescriptorSet(descriptors=()),
    )

    cfg = DatasetBuildConfig(
        data_dir=str(tmp_path),
        symbols="AAPL",
        out_dir=str(tmp_path / "out"),
        start_iso="2024-01-01",
        end_iso="2024-01-05",
    )

    prepared = orch._prepare_dataset_config(cfg)
    assert prepared.market_inputs is not None
    mapping = {item.dataset_id: item.schema_override for item in prepared.market_inputs}
    assert mapping.get("DBEQ.BASIC") == "ohlcv-1m"
    assert service.calls == [("AAPL", "ohlcv-1m")]


def test_prepare_dataset_config_prefers_discovery_with_descriptor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    coverage = _Coverage()
    writer = _Writer()
    ingestor = _Ingestor()
    discovery = _DiscoveryPayload(
        dataset_id="DBEQ.BASIC",
        schema="ohlcv-1m",
        coverage_start_ns=0,
        coverage_end_ns=200 * DAY_NS,
        storage_kind=None,
        cost_usd=0.0,
    )
    service = _DiscoveryService(result=discovery)

    orch = MLPipelineOrchestrator(
        coverage=coverage,
        writer=writer,
        data_registry=_Registry(),
        ingestor=ingestor,
        build_main=_CliWrapper(_ok),
        hpo_main=None,
        teacher_main=_CliWrapper(_ok),
        service=service,
    )

    descriptor = MarketFeedDescriptor(
        descriptor_id="eq-mini",
        dataset_id="EQUS.MINI",
        storage_kind=StorageKind.POSTGRES,
        schema="ohlcv-1m",
        symbol_patterns=("AAPL",),
        instrument_id_templates=("AAPL.{venue}",),
    )

    monkeypatch.setattr(
        "ml.orchestration.pipeline_orchestrator.load_market_feed_descriptors",
        lambda: MarketFeedDescriptorSet(descriptors=(descriptor,)),
    )

    cfg = DatasetBuildConfig(
        data_dir=str(tmp_path),
        symbols="AAPL",
        out_dir=str(tmp_path / "out"),
        start_iso="2024-01-01",
        end_iso="2024-01-05",
    )

    prepared = orch._prepare_dataset_config(cfg)
    assert prepared.market_inputs is not None
    mapping = {item.dataset_id: item.schema_override for item in prepared.market_inputs}
    assert mapping.get("DBEQ.BASIC") == "ohlcv-1m"
    assert service.calls == [("AAPL", "ohlcv-1m")]


def test_auto_fill_schema_prefers_discovery_when_binding_empty(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    coverage = _Coverage()
    writer = _Writer()
    ingestor = _Ingestor()
    discovery = _DiscoveryPayload(
        dataset_id="DBEQ.BASIC",
        schema="ohlcv-1m",
        coverage_start_ns=0,
        coverage_end_ns=200 * DAY_NS,
        storage_kind=None,
        cost_usd=0.0,
    )
    service = _DiscoveryService(result=discovery)

    orch = MLPipelineOrchestrator(
        coverage=coverage,
        writer=writer,
        data_registry=_Registry(),
        ingestor=ingestor,
        build_main=_CliWrapper(_ok),
        hpo_main=None,
        teacher_main=_CliWrapper(_ok),
        service=service,
    )

    calls: list[tuple[str, str]] = []
    ensure_calls: list[tuple[str, StorageKind]] = []

    def _fake_backfill_binding(
        self: MLPipelineOrchestrator,
        *,
        binding: ResolvedMarketBinding,
        lookback_days: int,
        **_: object,
    ) -> dict[str, BackfillWindowList]:
        calls.append((binding.dataset_id, binding.schema or ""))
        return {
            instrument_id: BackfillWindowList(
                (),
                requested=((0, DAY_NS),),
            )
            for instrument_id in binding.instrument_ids or (binding.symbol,)
        }

    def _capture_register(
        self: MLPipelineOrchestrator,
        *,
        dataset_id: str,
        dataset_type: DatasetType,
        location: str,
        storage_kind: StorageKind = StorageKind.PARQUET,
    ) -> None:
        ensure_calls.append((dataset_id, storage_kind))

    monkeypatch.setattr(MLPipelineOrchestrator, "backfill_binding", _fake_backfill_binding)
    monkeypatch.setattr(MLPipelineOrchestrator, "_ensure_dataset_registered", _capture_register)
    dataset_cfg = DatasetBuildConfig(
        data_dir=str(tmp_path),
        symbols="AAPL",
        out_dir=str(tmp_path / "out"),
        start_iso="2024-01-01",
        end_iso="2024-01-05",
        market_inputs=(
            MarketDatasetInput(
                descriptor_id="EQUS.MINI",
                dataset_id="EQUS.MINI",
                schema_override="ohlcv-1m",
            ),
        ),
    )

    orch._auto_fill_schema(
        dataset_id="EQUS.MINI",
        schema="ohlcv-1m",
        instrument_id="AAPL.ARCX",
        lookback_days=5,
        metrics=_AutoFillMetrics.default(),
        dataset_cfg=dataset_cfg,
        processed_bindings=set(),
    )

    assert calls
    assert calls[0][0] == "EQUS.MINI"
    assert len(calls) >= 2
    assert calls[1][0] == "DBEQ.BASIC"
    assert service.calls
    assert ensure_calls
    first_dataset_id, first_storage_kind = ensure_calls[0]
    assert first_dataset_id == "EQUS.MINI"
    assert first_storage_kind == StorageKind.POSTGRES
    assert ensure_calls
    first_dataset_id, first_storage_kind = ensure_calls[0]
    assert first_dataset_id == "EQUS.MINI"
    assert first_storage_kind == StorageKind.POSTGRES


def test_auto_fill_schema_retries_discovery_on_zero_frame_binding(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    coverage = _CoverageWithAvailability(
        {
            ("EQUS.MINI", "ohlcv-1m", "AAPL.ARCX"): {0},
        },
    )
    writer = _Writer()
    ingestor = _Ingestor()
    discovery = _DiscoveryPayload(
        dataset_id="DBEQ.BASIC",
        schema="ohlcv-1m",
        coverage_start_ns=0,
        coverage_end_ns=200 * DAY_NS,
        storage_kind=None,
        cost_usd=0.0,
    )
    service = _DiscoveryService(result=discovery)

    orch = MLPipelineOrchestrator(
        coverage=coverage,
        writer=writer,
        data_registry=_Registry(),
        ingestor=ingestor,
        build_main=_CliWrapper(_ok),
        hpo_main=None,
        teacher_main=_CliWrapper(_ok),
        service=service,
    )

    descriptor = MarketFeedDescriptor(
        descriptor_id="eq-mini",
        dataset_id="EQUS.MINI",
        storage_kind=StorageKind.POSTGRES,
        schema="ohlcv-1m",
        symbol_patterns=("AAPL",),
        instrument_id_templates=("AAPL.{venue}",),
    )
    monkeypatch.setattr(
        "ml.orchestration.pipeline_orchestrator.load_market_feed_descriptors",
        lambda: MarketFeedDescriptorSet(descriptors=(descriptor,)),
    )

    calls: list[tuple[str, str]] = []

    def _fake_backfill_binding(
        self: MLPipelineOrchestrator,
        *,
        binding: ResolvedMarketBinding,
        lookback_days: int,
        **_: object,
    ) -> dict[str, BackfillWindowList]:
        calls.append((binding.dataset_id, binding.schema or ""))
        instrument_ids = binding.instrument_ids or (binding.symbol,)
        if len(calls) == 1:
            return {
                instrument: BackfillWindowList(
                    (),
                    requested=((0, DAY_NS),),
                )
                for instrument in instrument_ids
            }
        return {
            instrument: BackfillWindowList(
                ((0, DAY_NS),),
                requested=((0, DAY_NS),),
                frames_written=1,
                rows_written=12,
            )
            for instrument in instrument_ids
        }

    monkeypatch.setattr(MLPipelineOrchestrator, "backfill_binding", _fake_backfill_binding)
    monkeypatch.setattr(MLPipelineOrchestrator, "_ensure_dataset_registered", lambda *_, **__: None)

    dataset_cfg = DatasetBuildConfig(
        data_dir=str(tmp_path),
        symbols="AAPL",
        out_dir=str(tmp_path / "out"),
        start_iso="2024-01-01",
        end_iso="2024-01-05",
        market_inputs=(
            MarketDatasetInput(
                descriptor_id="eq-mini",
                dataset_id="EQUS.MINI",
                schema_override="ohlcv-1m",
            ),
        ),
    )

    orch._auto_fill_schema(
        dataset_id="EQUS.MINI",
        schema="ohlcv-1m",
        instrument_id="AAPL.ARCX",
        lookback_days=5,
        metrics=_AutoFillMetrics.default(),
        dataset_cfg=dataset_cfg,
        processed_bindings=set(),
    )

    assert len(calls) == 2
    assert calls[0][0] == "EQUS.MINI"
    assert calls[1][0] == "DBEQ.BASIC"
    assert service.calls


def test_auto_fill_universe_logs_warning_when_gaps_remain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    coverage = _Coverage()
    writer = _Writer()
    ingestor = _Ingestor()

    def _build(argv: list[str] | None = None) -> int:
        if argv and "--out_dir" in argv:
            target = Path(argv[argv.index("--out_dir") + 1])
            target.mkdir(parents=True, exist_ok=True)
            (target / "dataset.csv").write_text("id,ts_event\n1,1\n", encoding="utf-8")
            _write_dataset_metadata_file(target)
        return 0

    orch = MLPipelineOrchestrator(
        coverage=coverage,
        writer=writer,
        data_registry=_Registry(),
        ingestor=ingestor,
        build_main=_CliWrapper(_build),
        hpo_main=None,
        teacher_main=_CliWrapper(_ok),
    )

    def _gap_backfill(
        self: MLPipelineOrchestrator,
        *,
        dataset_id: str,
        schema: str,
        instrument_id: str,
        lookback_days: int,
    ) -> BackfillWindowList:
        return BackfillWindowList(
            ((0, 1),),
            requested=((0, 1),),
            frames_written=1,
            rows_written=1,
        )

    monkeypatch.setattr(MLPipelineOrchestrator, "backfill", _gap_backfill)

    def _gap_backfill_binding(
        self: MLPipelineOrchestrator,
        *,
        binding: ResolvedMarketBinding,
        lookback_days: int,
    ) -> dict[str, BackfillWindowList]:
        instruments = binding.instrument_ids or (binding.symbol,)
        return {
            instrument: BackfillWindowList(
                ((0, 1),),
                requested=((0, 1),),
            )
            for instrument in instruments
        }

    monkeypatch.setattr(MLPipelineOrchestrator, "backfill_binding", _gap_backfill_binding)

    cfg = OrchestratorConfig(
        dataset=DatasetBuildConfig(
            data_dir=str(tmp_path),
            symbols="SPY.NYSE",
            out_dir=str(tmp_path / "out"),
            instrument_ids=("SPY.NYSE",),
        ),
        hpo=HPOConfig(),
        teacher=TeacherTrainConfig(enabled=False),
        auto_fill=AutoFillUniverseConfig(
            enabled=True,
            include_bars=True,
        ),
    )
    caplog.set_level("WARNING")
    rc = orch.run(cfg)
    assert rc == 0
    assert any("coverage gaps" in record.getMessage() for record in caplog.records)


def test_ensure_dataset_registered_seeds_manifest(tmp_path: Path) -> None:
    coverage = _Coverage()
    writer = _Writer()
    registry = _CapturingRegistry()
    ingestor = _Ingestor()

    orch = MLPipelineOrchestrator(
        coverage=coverage,
        writer=writer,
        data_registry=registry,
        ingestor=ingestor,
        build_main=_CliWrapper(_ok),
        teacher_main=_CliWrapper(_ok),
    )

    orch._ensure_dataset_registered(
        dataset_id="EQUS.MINI",
        dataset_type=DatasetType.TRADES,
        location=str(tmp_path),
    )

    manifest = registry.manifests["EQUS.MINI"]
    assert manifest.dataset_type == DatasetType.TRADES
    assert manifest.storage_kind == StorageKind.PARQUET
    assert manifest.retention_days > 0
    assert {"instrument_id", "ts_event", "ts_init"}.issubset(manifest.schema)
    assert "price" in manifest.schema
    assert manifest.metadata.get("auto_registered") is True
    initial_calls = registry.register_calls

    orch._ensure_dataset_registered(
        dataset_id="EQUS.MINI",
        dataset_type=DatasetType.TRADES,
        location=str(tmp_path),
    )
    assert registry.register_calls == initial_calls
    orch._ensure_dataset_registered(
        dataset_id="DBEQ.BASIC",
        dataset_type=DatasetType.MBP1,
        location=str(tmp_path),
        storage_kind=StorageKind.POSTGRES,
    )
    mbp_manifest = registry.manifests["DBEQ.BASIC"]
    assert mbp_manifest.dataset_type == DatasetType.MBP1
    assert mbp_manifest.storage_kind == StorageKind.POSTGRES
    assert mbp_manifest.retention_days == 90
    assert mbp_manifest.partitioning.get("interval") == "hourly"
    assert {"instrument_id", "ts_event", "ts_init", "level", "side"}.issubset(mbp_manifest.schema)
    assert registry.register_calls == initial_calls + 1


def test_dataset_metadata_sync_updates_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    coverage = _Coverage()
    writer = _Writer()
    registry = _CapturingRegistry()
    ingestor = _Ingestor()

    # Seed manifest so metadata sync has a target
    manifest = DatasetManifest(
        dataset_id="tft_dataset",
        dataset_type=DatasetType.FEATURES,
        storage_kind=StorageKind.PARQUET,
        location=str(tmp_path / "out"),
        partitioning={"by": "ts_event", "interval": "daily"},
        retention_days=90,
        schema={"instrument_id": "str", "ts_event": "int64", "ts_init": "int64"},
        ts_field="ts_event",
        seq_field=None,
        primary_keys=["instrument_id", "ts_event"],
        schema_hash="hash",
        constraints={},
        lineage=[],
        pipeline_signature="seed",
        version="1.0.0",
        created_at=0,
        last_modified=0,
        metadata={},
    )
    registry.register_dataset(manifest)

    def _build(argv: list[str] | None = None) -> int:
        if argv is None:
            return 0
        out_dir = Path(argv[argv.index("--out_dir") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "dataset.csv").write_text("id,ts_event\n1,1\n", encoding="utf-8")
        metadata = {
            "dataset_id": "tft_dataset",
            "vintage_policy": "real_time",
            "vintage_cutoff": None,
            "build_ts": "2025-01-01T00:00:00",
            "ts_event_start": "2025-01-01T00:00:00",
            "ts_event_end": "2025-01-02T00:00:00",
            "overall_window": ["2025-01-01T00:00:00", "2025-01-02T00:00:00"],
            "train_window": ["2025-01-01T00:00:00", "2025-01-01T12:00:00"],
            "validation_window": ["2025-01-01T12:00:00", "2025-01-02T00:00:00"],
            "test_window": None,
            "macro_observation_counts": {},
        }
        (out_dir / "dataset_metadata.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )
        return 0

    orch = MLPipelineOrchestrator(
        coverage=coverage,
        writer=writer,
        data_registry=registry,
        ingestor=ingestor,
        build_main=_CliWrapper(_build),
        teacher_main=_CliWrapper(_ok),
    )

    cfg = OrchestratorConfig(
        dataset=DatasetBuildConfig(
            data_dir=str(tmp_path),
            symbols="SPY.NYSE",
            out_dir=str(tmp_path / "out"),
        ),
        hpo=HPOConfig(),
        teacher=TeacherTrainConfig(enabled=False),
    )

    rc = orch.run(cfg)
    assert rc == 0
    assert registry.updates, "Expected manifest metadata update"
    update_payload = registry.updates[-1]
    assert update_payload["dataset_id"] == "tft_dataset"
    changes = update_payload["changes"]
    assert "metadata" in changes
    assert changes["metadata"]["vintage"]["policy"] == "real_time"
    assert "windows" in changes["metadata"]
    overall_window = changes["metadata"]["windows"]["overall"]
    assert list(overall_window) == [
        "2025-01-01T00:00:00",
        "2025-01-02T00:00:00",
    ]
    assert changes["pipeline_signature"].startswith("tft_pipeline:")


def test_guard_dataset_metadata_normalizes_iso_bounds(tmp_path: Path) -> None:
    orch = MLPipelineOrchestrator(
        coverage=_Coverage(),
        writer=_Writer(),
        build_main=_CliWrapper(_ok),
        teacher_main=_CliWrapper(_ok),
    )

    cfg = DatasetBuildConfig(
        data_dir=str(tmp_path),
        symbols="SPY.NYSE",
        out_dir=str(tmp_path / "out"),
        dataset_id="tft_dataset",
        vintage_policy=VintagePolicy.REAL_TIME,
        vintage_as_of="2025-01-01",
        start_iso="2025-01-01",
        end_iso="2025-01-02",
    )

    metadata = DatasetMetadata(
        dataset_id="tft_dataset",
        vintage_policy=VintagePolicy.REAL_TIME,
        vintage_cutoff="2025-01-01T00:00:00+00:00",
        build_ts="2025-01-03T00:00:00+00:00",
        ts_event_start="2025-01-01T00:00:00+00:00",
        ts_event_end="2025-01-02T00:00:00+00:00",
        overall_window=(
            "2025-01-01T00:00:00+00:00",
            "2025-01-02T00:00:00+00:00",
        ),
        train_window=None,
        validation_window=None,
        test_window=None,
        macro_observation_counts={},
    )

    orch._guard_dataset_metadata(cfg=cfg, metadata=metadata)


def test_guard_dataset_metadata_requires_macro_counts(tmp_path: Path) -> None:
    orch = MLPipelineOrchestrator(
        coverage=_Coverage(),
        writer=_Writer(),
        build_main=_CliWrapper(_ok),
        teacher_main=_CliWrapper(_ok),
    )

    cfg = DatasetBuildConfig(
        data_dir=str(tmp_path),
        symbols="SPY.NYSE",
        out_dir=str(tmp_path / "macro"),
        dataset_id="tft_dataset",
        include_macro=True,
        macro_series_ids=("CPI",),
    )

    incomplete_metadata = DatasetMetadata(
        dataset_id="tft_dataset",
        vintage_policy=VintagePolicy.REAL_TIME,
        vintage_cutoff=None,
        build_ts="2025-01-03T00:00:00+00:00",
        ts_event_start=None,
        ts_event_end=None,
        overall_window=None,
        train_window=None,
        validation_window=None,
        test_window=None,
        macro_observation_counts={"CPI": 0},
    )

    with pytest.raises(ValueError, match="Missing macro observations"):
        orch._guard_dataset_metadata(cfg=cfg, metadata=incomplete_metadata)

    complete_metadata = DatasetMetadata(
        dataset_id="tft_dataset",
        vintage_policy=VintagePolicy.REAL_TIME,
        vintage_cutoff=None,
        build_ts="2025-01-03T00:00:00+00:00",
        ts_event_start=None,
        ts_event_end=None,
        overall_window=None,
        train_window=None,
        validation_window=None,
        test_window=None,
        macro_observation_counts={"CPI": 5},
    )

    orch._guard_dataset_metadata(cfg=cfg, metadata=complete_metadata)


def test_guard_dataset_metadata_requires_provenance(tmp_path: Path) -> None:
    orch = MLPipelineOrchestrator(
        coverage=_Coverage(),
        writer=_Writer(),
        build_main=_CliWrapper(_ok),
        teacher_main=_CliWrapper(_ok),
    )

    cfg = DatasetBuildConfig(
        data_dir=str(tmp_path),
        symbols="SPY.NYSE",
        out_dir=str(tmp_path / "eq"),
        dataset_id="EQUS.MINI",
    )

    incomplete_binding = MarketBindingMetadata(
        binding_id="binding",
        dataset_id="EQUS.MINI",
        descriptor_id="EQUS.MINI",
        schema="ohlcv-1m",
        storage_kind="postgres",
        symbols=("SPY",),
        instrument_ids=("SPY.NYSE",),
        source="descriptor",
        license_start=None,
        license_end=None,
        ts_event_start=None,
        ts_event_end=None,
        rows_from_store=100,
        rows_from_catalog=0,
    )

    incomplete_metadata = DatasetMetadata(
        dataset_id="EQUS.MINI",
        vintage_policy=VintagePolicy.REAL_TIME,
        vintage_cutoff=None,
        build_ts="2025-01-03T00:00:00+00:00",
        ts_event_start=None,
        ts_event_end=None,
        overall_window=None,
        train_window=None,
        validation_window=None,
        test_window=None,
        macro_observation_counts={},
        market_bindings=(incomplete_binding,),
    )

    with pytest.raises(ValueError, match="source_datasets provenance"):
        orch._guard_dataset_metadata(cfg=cfg, metadata=incomplete_metadata)

    complete_binding = MarketBindingMetadata(
        binding_id="binding",
        dataset_id="EQUS.MINI",
        descriptor_id="EQUS.MINI",
        schema="ohlcv-1m",
        storage_kind="postgres",
        symbols=("SPY",),
        instrument_ids=("SPY.NYSE",),
        source="descriptor",
        license_start=None,
        license_end=None,
        ts_event_start=None,
        ts_event_end=None,
        rows_from_store=100,
        rows_from_catalog=0,
        source_datasets=("XNAS.ITCH",),
    )

    complete_metadata = DatasetMetadata(
        dataset_id="EQUS.MINI",
        vintage_policy=VintagePolicy.REAL_TIME,
        vintage_cutoff=None,
        build_ts="2025-01-03T00:00:00+00:00",
        ts_event_start=None,
        ts_event_end=None,
        overall_window=None,
        train_window=None,
        validation_window=None,
        test_window=None,
        macro_observation_counts={},
        market_bindings=(complete_binding,),
    )

    orch._guard_dataset_metadata(cfg=cfg, metadata=complete_metadata)


def test_build_auto_fill_config_from_args_handles_cli(tmp_path: Path) -> None:
    args = parse_args(
        [
            "--auto_fill_universe",
            "--auto_fill_dataset_id",
            "EQUS.PRO",
            "--auto_fill_l2_days",
            "45",
            "--auto_fill_skip_l2",
            "--auto_fill_allow_dataset_l2_ingest",
            "--auto_fill_l2_dataset_id",
            "DBEQ.PRO",
            "--auto_fill_l2_schema",
            "mbp-1",
            "--auto_fill_l2_progress_file",
            str(tmp_path / "progress.json"),
        ],
    )

    dataset_cfg = DatasetBuildConfig(
        data_dir=str(tmp_path),
        symbols="SPY.NYSE",
        out_dir=str(tmp_path / "out"),
    )

    auto_fill_cfg = _build_auto_fill_config_from_args(args, dataset_cfg)
    assert auto_fill_cfg.enabled is True
    assert auto_fill_cfg.dataset_id == "EQUS.PRO"
    assert auto_fill_cfg.l2_days == 45
    assert auto_fill_cfg.include_l2 is False
    assert auto_fill_cfg.disable_dataset_l2_ingest is False
    assert auto_fill_cfg.l2_dataset_id == "DBEQ.PRO"
    assert auto_fill_cfg.l2_schema == "mbp-1"
    assert auto_fill_cfg.l2_progress_file == str(tmp_path / "progress.json")


def test_parse_args_handles_market_inputs_json() -> None:
    args = parse_args(
        [
            "--market_inputs_json",
            '[{"descriptor_id":"EQUS.MINI","symbols":["SPY","QQQ"],"schema":"ohlcv-1m"}]',
        ],
    )

    parsed = _parse_market_inputs_json(str(args.market_inputs_json))
    assert parsed is not None
    assert parsed[0].descriptor_id == "EQUS.MINI"
    assert parsed[0].symbols == ("SPY", "QQQ")
    assert parsed[0].schema_override == "ohlcv-1m"


def test_parse_market_inputs_json_normalizes_storage_kind_variants() -> None:
    payload = json.dumps(
        [
            {"descriptor_id": "EQUS.MINI", "storage_kind": "POSTGRES"},
            {"descriptor_id": "DBEQ.MINI", "storage_kind": "StorageKind.parquet"},
        ],
    )

    parsed = _parse_market_inputs_json(payload)
    assert parsed is not None
    assert parsed[0].storage_kind_override == StorageKind.POSTGRES
    assert parsed[1].storage_kind_override == StorageKind.PARQUET


def test_auto_fill_skips_without_databento(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    coverage = _Coverage()
    writer = _Writer()

    # Ensure environment lacks the Databento API key so ingestion is unavailable
    monkeypatch.delenv("DATABENTO_API_KEY", raising=False)

    def _build(argv: list[str] | None = None) -> int:
        if argv is not None and "--out_dir" in argv:
            out_dir = Path(argv[argv.index("--out_dir") + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "dataset.csv").write_text("id,ts_event\n1,1\n", encoding="utf-8")
            _write_dataset_metadata_file(out_dir)
        return 0

    orch = MLPipelineOrchestrator(
        coverage=coverage,
        writer=writer,
        data_registry=_Registry(),
        ingestor=None,
        build_main=_CliWrapper(_build),
        hpo_main=None,
        teacher_main=_CliWrapper(_ok),
    )

    def _fail_backfill(
        self: MLPipelineOrchestrator,
        *,
        dataset_id: str,
        schema: str,
        instrument_id: str,
        lookback_days: int,
    ) -> BackfillWindowList:
        raise AssertionError("backfill should not be invoked when Databento is unavailable")

    monkeypatch.setattr(MLPipelineOrchestrator, "backfill", _fail_backfill)

    cfg = OrchestratorConfig(
        dataset=DatasetBuildConfig(
            data_dir=str(tmp_path),
            symbols="SPY.NYSE",
            out_dir=str(tmp_path / "out"),
        ),
        hpo=HPOConfig(),
        teacher=TeacherTrainConfig(enabled=False),
        auto_fill=AutoFillUniverseConfig(
            enabled=True,
            include_bars=True,
            include_tbbo=True,
            include_trades=True,
            include_l2=False,
        ),
    )

    rc = orch.run(cfg)
    assert rc == 0


def test_resolve_write_mode_tokens_aliases() -> None:
    assert _resolve_write_mode_tokens("parquet") == ("datastore", "parquet")
    assert _resolve_write_mode_tokens("sql+datastore") == ("sql", "datastore")


def test_resolve_write_mode_tokens_invalid() -> None:
    with pytest.raises(SystemExit):
        _resolve_write_mode_tokens("invalid-mode")


def test_prepare_dataset_config_uses_coverage(tmp_path: Path) -> None:
    coverage = _CoverageWithAvailability(
        available={("XNAS.ITCH", "ohlcv-1m", "SPY.XNAS"): {1}},
    )
    orch = MLPipelineOrchestrator(
        coverage=coverage,
        writer=_Writer(),
        data_registry=_Registry(),
        ingestor=_Ingestor(),
        build_main=_CliWrapper(_ok),
        teacher_main=_CliWrapper(_ok),
    )
    cfg = DatasetBuildConfig(
        data_dir=str(tmp_path),
        symbols="SPY",
        out_dir=str(tmp_path / "out"),
        instrument_ids=("SPY.XNAS",),
        end_iso="2025-09-25",
        market_dataset_id="XNAS.ITCH",
    )
    prepared = orch._prepare_dataset_config(cfg)
    assert prepared.market_inputs is not None
    assert prepared.market_inputs[0].dataset_id == "XNAS.ITCH"
    assert prepared.market_inputs[0].symbols == ("SPY",)
    assert prepared.instrument_ids is not None
    assert "SPY.XNAS" in prepared.instrument_ids


def test_apply_default_market_inputs_requires_explicit_dataset() -> None:
    base_cfg = DatasetBuildConfig(
        data_dir="data",
        symbols="SPY,QQQ",
        out_dir="out",
    )
    updated = _apply_default_market_inputs(base_cfg)
    assert updated.market_inputs is None
    assert updated.market_dataset_id is None


def test_apply_default_market_inputs_respects_existing_inputs() -> None:
    custom_input = MarketDatasetInput(descriptor_id="CUSTOM.FEED", dataset_id="CUSTOM.FEED")
    base_cfg = DatasetBuildConfig(
        data_dir="data",
        symbols="SPY",
        out_dir="out",
        market_inputs=(custom_input,),
        market_dataset_id="CUSTOM.FEED",
    )
    updated = _apply_default_market_inputs(base_cfg)
    assert updated.market_inputs == (custom_input,)
    assert updated.market_dataset_id == "CUSTOM.FEED"


def test_prepare_dataset_config_skips_disallowed_dataset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    coverage = _Coverage()
    writer = _Writer()
    ingestor = _Ingestor()
    service = _ServiceStub(allowed_dataset="EQUS.MINI")
    orch = MLPipelineOrchestrator(
        coverage=coverage,
        writer=writer,
        data_registry=_Registry(),
        ingestor=ingestor,
        build_main=_CliWrapper(_ok),
        hpo_main=None,
        teacher_main=_CliWrapper(_ok),
        service=service,
    )

    descriptors = (
        MarketFeedDescriptor(
            descriptor_id="tier1_l0",
            dataset_id="tier1_l0",
            storage_kind=StorageKind.POSTGRES,
            schema="ohlcv-1m",
            symbol_patterns=("*",),
            instrument_id_templates=("{symbol}.XNAS",),
        ),
        MarketFeedDescriptor(
            descriptor_id="EQUS.MINI",
            dataset_id="EQUS.MINI",
            storage_kind=StorageKind.POSTGRES,
            schema="ohlcv-1m",
            symbol_patterns=("*",),
            instrument_id_templates=("{symbol}.XNAS",),
        ),
    )
    monkeypatch.setattr(
        "ml.orchestration.pipeline_orchestrator.load_market_feed_descriptors",
        lambda: MarketFeedDescriptorSet(descriptors=descriptors),
    )

    cfg = DatasetBuildConfig(
        data_dir=str(tmp_path),
        symbols="SPY",
        out_dir=str(tmp_path / "out"),
        start_iso="2024-01-01",
        end_iso="2024-01-05",
        market_dataset_id="EQUS.MINI",
    )

    prepared = orch._prepare_dataset_config(cfg)
    assert prepared.market_inputs is not None
    dataset_ids = {item.dataset_id for item in prepared.market_inputs}
    assert "EQUS.MINI" in dataset_ids
    assert "tier1_l0" not in dataset_ids
    assert service.calls


def test_execute_stage_dataset_disables_training_flags() -> None:
    cfg = OrchestratorConfig(
        dataset=DatasetBuildConfig(
            data_dir="data",
            symbols="SPY",
            out_dir="out",
        ),
        hpo=HPOConfig(enabled=True),
        teacher=TeacherTrainConfig(enabled=True),
        student=StudentDistillConfig(enabled=True),
    )
    recorder = _StageRecorder()
    rc = pipeline_orchestrator._execute_stage(
        orch=recorder,
        orchestrator_cfg=cfg,
        stage=Stage.DATASET,
        ds_cfg=cfg.dataset,
        auto_fill_cfg=AutoFillUniverseConfig(),
        args=SimpleNamespace(ingest=False),
        ingestor=None,
        ingestion_service=None,
    )
    assert rc == 0
    assert recorder.calls == ["run"]
    assert cfg.teacher.enabled is True
    mutated = recorder.last_config
    assert mutated is not None
    assert mutated.teacher.enabled is False
    assert mutated.hpo.enabled is False
    assert mutated.student.enabled is False
    assert mutated.promotions is None
    assert mutated.integration is None


def test_execute_stage_train_invokes_training_only() -> None:
    cfg = OrchestratorConfig(
        dataset=DatasetBuildConfig(
            data_dir="data",
            symbols="SPY",
            out_dir="out",
        ),
        hpo=HPOConfig(enabled=True),
        teacher=TeacherTrainConfig(enabled=True),
        student=StudentDistillConfig(enabled=True),
    )
    recorder = _StageRecorder()
    rc = pipeline_orchestrator._execute_stage(
        orch=recorder,
        orchestrator_cfg=cfg,
        stage=Stage.TRAIN,
        ds_cfg=cfg.dataset,
        auto_fill_cfg=AutoFillUniverseConfig(),
        args=SimpleNamespace(ingest=False),
        ingestor=None,
        ingestion_service=None,
    )
    assert rc == 0
    assert recorder.calls == ["train"]
    assert recorder.last_config is cfg


def test_extract_config_args_handles_equals_form() -> None:
    config, stage, rest = pipeline_orchestrator._extract_config_args(
        ["--config=orchestrator.toml", "--stage=train", "--foo", "bar"],
    )
    assert config == "orchestrator.toml"
    assert stage == "train"
    assert rest == ["--foo", "bar"]


def test_run_ingestion_stage_uses_resolved_bindings(monkeypatch: pytest.MonkeyPatch) -> None:
    binding = ResolvedMarketBinding(
        binding_id="b-1",
        symbol="SPY",
        instrument_ids=("SPY.NYSE",),
        dataset_id="EQUS.MINI",
        descriptor_id=None,
        schema="ohlcv-1m",
        storage_kind=None,
        license_start=None,
        license_end=None,
        start=None,
        end=None,
        source="test",
    )
    orch = _IngestionOrchStub(resolved_bindings=(binding,))
    ds_cfg = DatasetBuildConfig(data_dir="data", symbols="SPY", out_dir="out")
    ingestion_cfg = IngestionStageConfig(
        enabled=True,
        dataset_id="EQUS.MINI",
        schema="bars",
        instruments=("SPY.NYSE",),
        lookback_days=7,
        market_dataset_id="EQUS.MINI",
    )

    def _fake_plan(
        *,
        ds_cfg: DatasetBuildConfig | None,
        ingestion_cfg: IngestionStageConfig,
    ) -> tuple[pipeline_orchestrator._IngestionPlanItem, ...]:
        del ds_cfg, ingestion_cfg
        return (
            pipeline_orchestrator._IngestionPlanItem(
                binding=binding,
                dataset_id=binding.dataset_id,
                schema="bars",
                instrument_ids=binding.instrument_ids,
            ),
        )

    monkeypatch.setattr(
        pipeline_orchestrator,
        "_build_ingestion_plan",
        _fake_plan,
    )

    rc = pipeline_orchestrator._run_ingestion_stage(
        orch=cast(MLPipelineOrchestrator, orch),
        ds_cfg=ds_cfg,
        auto_fill_cfg=AutoFillUniverseConfig(),
        ingestion_cfg=ingestion_cfg,
        ingestor=object(),
        ingestion_service=object(),
    )

    assert rc == 0
    assert orch.binding_calls == 1
    assert orch.coverage_calls == 0
    assert orch.manual_calls == 0


def test_run_ingestion_stage_fallbacks_to_manual(monkeypatch: pytest.MonkeyPatch) -> None:
    orch = _IngestionOrchStub(
        resolved_bindings=(),
        coverage_exception=IngestionError("coverage failure"),
        manual_rows_written=5,
    )
    ds_cfg = DatasetBuildConfig(data_dir="data", symbols="SPY", out_dir="out")
    ingestion_cfg = IngestionStageConfig(
        enabled=True,
        dataset_id="EQUS.MINI",
        schema="bars",
        instruments=("SPY.NYSE",),
        lookback_days=3,
        market_dataset_id="EQUS.MINI",
    )

    def _manual_plan(
        *,
        ds_cfg: DatasetBuildConfig | None,
        ingestion_cfg: IngestionStageConfig,
    ) -> tuple[pipeline_orchestrator._IngestionPlanItem, ...]:
        del ds_cfg, ingestion_cfg
        return (
            pipeline_orchestrator._IngestionPlanItem(
                binding=None,
                dataset_id="EQUS.MINI",
                schema="bars",
                instrument_ids=("SPY.NYSE",),
            ),
        )

    monkeypatch.setattr(
        pipeline_orchestrator,
        "_build_ingestion_plan",
        _manual_plan,
    )

    rc = pipeline_orchestrator._run_ingestion_stage(
        orch=cast(MLPipelineOrchestrator, orch),
        ds_cfg=ds_cfg,
        auto_fill_cfg=AutoFillUniverseConfig(),
        ingestion_cfg=ingestion_cfg,
        ingestor=object(),
        ingestion_service=object(),
    )

    assert rc == 0
    assert orch.binding_calls == 0
    assert orch.coverage_calls == 1
    assert orch.manual_calls == 1


def test_run_ingestion_stage_degrades_without_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    orch = _IngestionOrchStub()
    ds_cfg = DatasetBuildConfig(data_dir="data", symbols="SPY", out_dir="out")
    auto_fill_cfg = AutoFillUniverseConfig()
    ingestion_cfg = IngestionStageConfig(
        enabled=True,
        dataset_id="EQUS.MINI",
        schema="bars",
        instruments=("SPY.NYSE",),
        lookback_days=2,
        market_dataset_id="EQUS.MINI",
    )
    monkeypatch.delenv("DATABENTO_API_KEY", raising=False)

    def _plan(
        *,
        ds_cfg: DatasetBuildConfig | None,
        ingestion_cfg: IngestionStageConfig,
    ) -> tuple[pipeline_orchestrator._IngestionPlanItem, ...]:
        del ds_cfg, ingestion_cfg
        return (
            pipeline_orchestrator._IngestionPlanItem(
                binding=None,
                dataset_id="EQUS.MINI",
                schema="bars",
                instrument_ids=("SPY.NYSE",),
            ),
        )

    monkeypatch.setattr(
        pipeline_orchestrator,
        "_build_ingestion_plan",
        _plan,
    )

    rc = pipeline_orchestrator._run_ingestion_stage(
        orch=cast(MLPipelineOrchestrator, orch),
        ds_cfg=ds_cfg,
        auto_fill_cfg=auto_fill_cfg,
        ingestion_cfg=ingestion_cfg,
        ingestor=None,
        ingestion_service=None,
    )

    assert rc == 0
    assert orch.binding_calls == 0
    assert orch.coverage_calls == 0
    assert orch.manual_calls == 0


def test_run_ingestion_stage_returns_error_when_fallbacks_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orch = _IngestionOrchStub(
        resolved_bindings=(),
        coverage_exception=IngestionError("coverage failure"),
        manual_exception=IngestionError("manual failure"),
    )
    ds_cfg = DatasetBuildConfig(
        data_dir=str(tmp_path / "data"),
        symbols="SPY",
        out_dir=str(tmp_path / "out"),
    )
    ingestion_cfg = IngestionStageConfig(
        enabled=True,
        dataset_id="EQUS.MINI",
        schema="bars",
        instruments=("SPY.NYSE",),
        lookback_days=2,
        market_dataset_id="EQUS.MINI",
    )

    def _failing_plan(
        *,
        ds_cfg: DatasetBuildConfig | None,
        ingestion_cfg: IngestionStageConfig,
    ) -> tuple[pipeline_orchestrator._IngestionPlanItem, ...]:
        del ds_cfg, ingestion_cfg
        return (
            pipeline_orchestrator._IngestionPlanItem(
                binding=None,
                dataset_id="EQUS.MINI",
                schema="bars",
                instrument_ids=("SPY.NYSE",),
            ),
        )

    monkeypatch.setattr(
        pipeline_orchestrator,
        "_build_ingestion_plan",
        _failing_plan,
    )

    rc = pipeline_orchestrator._run_ingestion_stage(
        orch=cast(MLPipelineOrchestrator, orch),
        ds_cfg=ds_cfg,
        auto_fill_cfg=AutoFillUniverseConfig(),
        ingestion_cfg=ingestion_cfg,
        ingestor=object(),
        ingestion_service=object(),
    )

    assert rc == 1
    assert orch.binding_calls == 0
    assert orch.coverage_calls == 1
    assert orch.manual_calls == 1


def test_run_ingestion_stage_auto_fill_without_ingest(monkeypatch: pytest.MonkeyPatch) -> None:
    orch = _IngestionOrchStub()
    ds_cfg = DatasetBuildConfig(data_dir="data", symbols="SPY", out_dir="out")
    auto_fill_cfg = AutoFillUniverseConfig(enabled=True)
    ingestion_cfg = IngestionStageConfig(
        enabled=False,
        dataset_id="EQUS.MINI",
        schema="bars",
        instruments=("SPY.NYSE",),
        lookback_days=1,
        market_dataset_id="EQUS.MINI",
    )

    def _auto_plan(
        *,
        ds_cfg: DatasetBuildConfig | None,
        ingestion_cfg: IngestionStageConfig,
    ) -> tuple[pipeline_orchestrator._IngestionPlanItem, ...]:
        del ds_cfg, ingestion_cfg
        return (
            pipeline_orchestrator._IngestionPlanItem(
                binding=None,
                dataset_id="EQUS.MINI",
                schema="bars",
                instrument_ids=("SPY.NYSE",),
            ),
        )

    monkeypatch.setattr(
        pipeline_orchestrator,
        "_build_ingestion_plan",
        _auto_plan,
    )

    rc = pipeline_orchestrator._run_ingestion_stage(
        orch=cast(MLPipelineOrchestrator, orch),
        ds_cfg=ds_cfg,
        auto_fill_cfg=auto_fill_cfg,
        ingestion_cfg=ingestion_cfg,
        ingestor=None,
        ingestion_service=None,
    )

    assert rc == 0
    assert orch.auto_fill_calls == 1
    assert orch.binding_calls == 0
    assert orch.manual_calls == 0


def test_pipeline_service_dispatch_dataset_stage() -> None:
    orch = _OrchestratorRecorder()
    service = PipelineIntegrationService(integration_manager=None)
    run_cfg = OrchestratorRunConfig(
        stage=Stage.DATASET,
        dataset=DatasetBuildConfig(data_dir="data", symbols="SPY", out_dir="out"),
        training=TrainingStageConfig(),
    )

    rc = service._dispatch_stage_run(cast(MLPipelineOrchestrator, orch), run_cfg)

    assert rc == 0
    assert len(orch.run_calls) == 1
    mutated = orch.run_calls[0]
    assert mutated.teacher.enabled is False
    assert mutated.hpo.enabled is False
    assert mutated.student.enabled is False
    assert mutated.promotions is None
    assert mutated.integration is None


def test_pipeline_service_dispatch_ingest_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    orch = _OrchestratorRecorder()
    service = PipelineIntegrationService(integration_manager=None)
    run_cfg = OrchestratorRunConfig(
        stage=Stage.INGEST,
        dataset=DatasetBuildConfig(data_dir="data", symbols="SPY", out_dir="out"),
        ingestion=IngestionStageConfig(
            enabled=True, schema="bars", instruments=("SPY.NYSE",), lookback_days=4
        ),
        training=TrainingStageConfig(),
    )

    captured: dict[str, object] = {}

    def _fake_run_ingestion_stage(**kwargs: object) -> int:
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(pipeline_orchestrator, "_run_ingestion_stage", _fake_run_ingestion_stage)

    rc = service._dispatch_stage_run(cast(MLPipelineOrchestrator, orch), run_cfg)

    assert rc == 0
    assert captured.get("orch") is orch
    assert captured.get("ds_cfg") == run_cfg.dataset
    ingestion_cfg = captured.get("ingestion_cfg")
    assert isinstance(ingestion_cfg, IngestionStageConfig)
    assert ingestion_cfg.enabled is True
    assert ingestion_cfg.schema == "bars"
    assert ingestion_cfg.instruments == ("SPY.NYSE",)


def test_pipeline_service_ingest_stage_without_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    orch = _OrchestratorRecorder()
    service = PipelineIntegrationService(integration_manager=None)
    run_cfg = OrchestratorRunConfig(
        stage=Stage.INGEST,
        dataset=None,
        ingestion=IngestionStageConfig(
            enabled=True,
            dataset_id="EQUS.MINI",
            schema="bars",
            instruments=("SPY.NYSE",),
            lookback_days=5,
        ),
    )

    captured: dict[str, object] = {}

    def _fake_run_ingestion_stage(**kwargs: object) -> int:
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(pipeline_orchestrator, "_run_ingestion_stage", _fake_run_ingestion_stage)

    rc = service._dispatch_stage_run(cast(MLPipelineOrchestrator, orch), run_cfg)

    assert rc == 0
    assert captured.get("ds_cfg") is None
    ingestion_cfg = captured.get("ingestion_cfg")
    assert isinstance(ingestion_cfg, IngestionStageConfig)
    assert ingestion_cfg.dataset_id == "EQUS.MINI"
    assert ingestion_cfg.instruments == ("SPY.NYSE",)


def test_build_ingestion_plan_uses_bindings(monkeypatch: pytest.MonkeyPatch) -> None:
    binding = ResolvedMarketBinding(
        binding_id="plan-001",
        symbol="SPY",
        instrument_ids=("SPY.NYSE",),
        dataset_id="EQUS.MINI",
        descriptor_id="EQUS.MINI",
        schema="ohlcv-1m",
        storage_kind=None,
        license_start=None,
        license_end=None,
        start=None,
        end=None,
        source="descriptor",
    )
    ds_cfg = DatasetBuildConfig(data_dir="data", symbols="SPY", out_dir="out")
    ingestion_cfg = IngestionStageConfig(
        enabled=True,
        dataset_id="EQUS.MINI",
        schema="bars",
        instruments=("SPY.NYSE",),
        market_dataset_id="EQUS.MINI",
    )

    def _fake_resolve(**_: object) -> tuple[ResolvedMarketBinding, ...]:
        return (binding,)

    monkeypatch.setattr(
        pipeline_orchestrator.IngestionOrchestrator,
        "resolve_market_bindings",
        staticmethod(_fake_resolve),
    )

    plan = pipeline_orchestrator._build_ingestion_plan(
        ds_cfg=ds_cfg,
        ingestion_cfg=ingestion_cfg,
    )

    assert len(plan) == 1
    item = plan[0]
    assert item.binding is binding
    assert item.dataset_id == "EQUS.MINI"
    assert item.schema == "ohlcv-1m"
    assert item.instrument_ids == ("SPY.NYSE",)


def test_build_ingestion_plan_manual_when_no_bindings(monkeypatch: pytest.MonkeyPatch) -> None:
    ds_cfg = DatasetBuildConfig(data_dir="data", symbols="SPY", out_dir="out")
    ingestion_cfg = IngestionStageConfig(
        enabled=True,
        dataset_id="EQUS.MINI",
        schema="bars",
        instruments=("SPY.NYSE",),
    )

    def _no_bindings(**_: object) -> tuple[ResolvedMarketBinding, ...]:
        return ()

    monkeypatch.setattr(
        pipeline_orchestrator.IngestionOrchestrator,
        "resolve_market_bindings",
        staticmethod(_no_bindings),
    )

    plan = pipeline_orchestrator._build_ingestion_plan(
        ds_cfg=ds_cfg,
        ingestion_cfg=ingestion_cfg,
    )

    assert len(plan) == 1
    item = plan[0]
    assert item.binding is None
    assert item.dataset_id == "EQUS.MINI"
    assert item.schema == "ohlcv-1m"
    assert item.instrument_ids == ("SPY.NYSE",)


def test_build_ingestion_plan_defaults_to_allowed_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ingestion_cfg = IngestionStageConfig(
        enabled=True,
        schema="bars",
        instruments=("INTC.XNAS",),
    )

    monkeypatch.setattr(
        pipeline_orchestrator,
        "_get_allowed_databento_datasets",
        lambda: frozenset({"XNAS.ITCH"}),
    )

    monkeypatch.setattr(
        pipeline_orchestrator.IngestionOrchestrator,
        "resolve_market_bindings",
        staticmethod(lambda **_: ()),
    )

    plan = pipeline_orchestrator._build_ingestion_plan(
        ds_cfg=None,
        ingestion_cfg=ingestion_cfg,
    )

    assert len(plan) == 1
    item = plan[0]
    assert item.binding is None
    assert item.dataset_id == "XNAS.ITCH"
    assert item.schema == "ohlcv-1m"
    assert item.instrument_ids == ("INTC.XNAS",)


def test_pipeline_service_infers_stage_and_ingestion_config() -> None:
    service = PipelineIntegrationService(integration_manager=None)
    payload = {
        "dataset": {
            "data_dir": "data",
            "symbols": "SPY.NYSE",
            "out_dir": "out",
            "dataset_id": "CUSTOM.DATASET",
        },
        "ingestion": {
            "enabled": False,
        },
    }

    run_cfg = service._build_run_config(pipeline_type="ingest_only", payload=payload)

    assert run_cfg.stage == Stage.INGEST
    assert run_cfg.ingestion is not None
    assert run_cfg.ingestion.enabled is True
    assert run_cfg.ingestion.dataset_id == "CUSTOM.DATASET"
    assert run_cfg.ingestion.instruments == ("SPY.NYSE",)


def test_pipeline_service_respects_explicit_stage_override() -> None:
    service = PipelineIntegrationService(integration_manager=None)
    payload = {
        "dataset": {
            "data_dir": "data",
            "symbols": "SPY.NYSE",
            "out_dir": "out",
        },
        "stage": "train",
    }

    run_cfg = service._build_run_config(pipeline_type="ingest", payload=payload)

    assert run_cfg.stage == Stage.TRAIN
    assert run_cfg.ingestion is None
