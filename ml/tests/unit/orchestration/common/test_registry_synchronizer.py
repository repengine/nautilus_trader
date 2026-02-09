"""RegistrySynchronizer component tests."""

from __future__ import annotations

import dataclasses
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from ml.data import DatasetMetadata
from ml.data.metadata import MarketBindingMetadata
from ml.data.vintage import VintagePolicy
from ml.orchestration.config_types import DatasetBuildConfig
from ml.orchestration.registry_synchronizer import RegistrySynchronizer
from ml.registry.dataclasses import DatasetType
from ml.registry.dataclasses import StorageKind
from ml.tests.utils.targets import build_default_target_semantics_payload


@pytest.fixture
def data_registry() -> Mock:
    """Provide a mock DataRegistry."""
    registry = Mock()
    registry.get_manifest.return_value = Mock(metadata={})
    return registry


@pytest.fixture
def registry_synchronizer(data_registry: Mock) -> RegistrySynchronizer:
    """Create RegistrySynchronizer instance for testing."""
    return RegistrySynchronizer(
        data_registry=data_registry,
        feature_registry=None,
        model_registry=None,
        message_bus=None,
    )


@pytest.fixture
def dataset_cfg(tmp_path: Path) -> DatasetBuildConfig:
    """Construct a minimal DatasetBuildConfig for registry sync operations."""
    return DatasetBuildConfig(
        data_dir=str(tmp_path / "data"),
        out_dir=str(tmp_path / "out"),
        dataset_id="test.dataset",
        symbols="SPY",
        target_semantics=build_default_target_semantics_payload(),
    )


@pytest.fixture
def dataset_metadata() -> DatasetMetadata:
    """Construct DatasetMetadata with required fields."""
    return DatasetMetadata(
        dataset_id="test.dataset",
        vintage_policy=VintagePolicy.REAL_TIME,
        vintage_cutoff=None,
        build_ts="2024-01-01T00:00:00Z",
        ts_event_start=None,
        ts_event_end=None,
        overall_window=None,
        train_window=None,
        validation_window=None,
        test_window=None,
        macro_observation_counts={},
        market_bindings=(),
    )


def _metadata_json_payload(metadata: DatasetMetadata) -> dict[str, Any]:
    """Convert DatasetMetadata to JSON-serializable payload."""
    return {
        "dataset_id": metadata.dataset_id,
        "vintage_policy": metadata.vintage_policy.value,
        "vintage_cutoff": metadata.vintage_cutoff,
        "build_ts": metadata.build_ts,
        "ts_event_start": metadata.ts_event_start,
        "ts_event_end": metadata.ts_event_end,
        "overall_window": list(metadata.overall_window) if metadata.overall_window else None,
        "train_window": list(metadata.train_window) if metadata.train_window else None,
        "validation_window": list(metadata.validation_window) if metadata.validation_window else None,
        "test_window": list(metadata.test_window) if metadata.test_window else None,
        "macro_observation_counts": metadata.macro_observation_counts,
        "market_bindings": [],
    }


def test_registry_synchronizer_initializes_with_registries(
    data_registry: Mock,
) -> None:
    """Ensure registries are retained on initialization."""
    synchronizer = RegistrySynchronizer(
        data_registry=data_registry,
        feature_registry=None,
        model_registry=None,
        message_bus=None,
    )

    assert synchronizer.data_registry is data_registry


def test_record_build_artifacts_sets_build_artifacts(
    registry_synchronizer: RegistrySynchronizer,
    dataset_cfg: DatasetBuildConfig,
) -> None:
    """_record_build_artifacts should store artifacts for later access."""
    registry_synchronizer._record_build_artifacts(
        cfg=dataset_cfg,
        feature_set_id="fs1",
        feature_names=("a", "b"),
        feature_registry_dir="/tmp/features",
        dataset_metadata=None,
    )

    artifacts = registry_synchronizer.build_artifacts
    assert artifacts is not None
    assert artifacts.feature_set_id == "fs1"
    assert artifacts.feature_names == ("a", "b")


def test_guard_dataset_metadata_accepts_valid_metadata(
    registry_synchronizer: RegistrySynchronizer,
    dataset_cfg: DatasetBuildConfig,
    dataset_metadata: DatasetMetadata,
) -> None:
    """_guard_dataset_metadata should not raise for valid metadata."""
    registry_synchronizer._guard_dataset_metadata(cfg=dataset_cfg, metadata=dataset_metadata)


def test_synchronize_dataset_manifest_updates_registry(
    registry_synchronizer: RegistrySynchronizer,
    data_registry: Mock,
    dataset_cfg: DatasetBuildConfig,
    dataset_metadata: DatasetMetadata,
) -> None:
    """synchronize_dataset_manifest should call registry.update_manifest."""
    registry_synchronizer.synchronize_dataset_manifest(cfg=dataset_cfg, metadata=dataset_metadata)

    assert data_registry.update_manifest.called


def test_emit_feature_refresh_event_uses_attached_message_bus(
    registry_synchronizer: RegistrySynchronizer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Attached message bus should be used when available."""
    from ml.config.bus import MessageBusConfig

    publisher = Mock()
    registry_synchronizer.message_bus = publisher

    def _from_env(_cls: type[MessageBusConfig]) -> MessageBusConfig:
        return MessageBusConfig(scheme="topic", topic_prefix="ml")

    monkeypatch.setattr(MessageBusConfig, "from_env", classmethod(_from_env))
    monkeypatch.setattr(
        "ml.common.message_topics.build_topic_for_stage",
        lambda _stage, dataset_id, scheme, prefix: f"{prefix}.{scheme}.{dataset_id}",
    )

    registry_synchronizer._emit_feature_refresh_event("test.dataset", ("f1", "f2"))

    publisher.publish.assert_called_once()
    topic, payload = publisher.publish.call_args.args
    assert topic == "ml.topic.test.dataset"
    assert payload["dataset_id"] == "test.dataset"
    assert payload["features"] == ["f1", "f2"]


def test_emit_feature_refresh_event_falls_back_to_factory_publisher(
    registry_synchronizer: RegistrySynchronizer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Publisher factory path should be used when no message bus is attached."""
    from ml.config.bus import MessageBusConfig

    publisher = Mock()

    def _raise_from_env(_cls: type[MessageBusConfig]) -> MessageBusConfig:
        raise RuntimeError("env unavailable")

    monkeypatch.setattr(MessageBusConfig, "from_env", classmethod(_raise_from_env))
    monkeypatch.setattr("ml.common.message_bus.publisher_from_config", lambda _cfg: publisher)
    monkeypatch.setattr("ml.common.message_topics.build_topic_for_stage", lambda *_args, **_kwargs: "topic")

    registry_synchronizer.message_bus = None
    registry_synchronizer._emit_feature_refresh_event("test.dataset", ("f1",))

    publisher.publish.assert_called_once()


def test_emit_feature_refresh_event_swallows_publish_error(
    registry_synchronizer: RegistrySynchronizer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Publish errors are best-effort and must not bubble up."""
    publisher = Mock()
    publisher.publish.side_effect = RuntimeError("publish failed")
    registry_synchronizer.message_bus = publisher
    monkeypatch.setattr("ml.common.message_topics.build_topic_for_stage", lambda *_args, **_kwargs: "topic")
    registry_synchronizer._emit_feature_refresh_event("test.dataset", ("f1",))


def test_ensure_dataset_registered_registers_missing_manifest(
    registry_synchronizer: RegistrySynchronizer,
    data_registry: Mock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing manifests should trigger best-effort auto registration."""
    data_registry.get_manifest.side_effect = RuntimeError("missing")
    captured: dict[str, Any] = {}
    manifest = SimpleNamespace(dataset_id="test.dataset")

    def _build_auto_dataset_manifest(**kwargs: Any) -> object:
        captured.update(kwargs)
        return manifest

    monkeypatch.setattr(
        "ml.data.dataset_manifest_defaults.build_auto_dataset_manifest",
        _build_auto_dataset_manifest,
    )

    registry_synchronizer._ensure_dataset_registered(
        dataset_id="test.dataset",
        metadata={"dataset_type": "bars", "retention_days": 30, "out_dir": "/tmp/output"},
        storage_kind="postgres",
    )

    data_registry.register_dataset.assert_called_once_with(manifest)
    assert captured["dataset_type"] is DatasetType.BARS
    assert captured["storage_kind"] is StorageKind.POSTGRES
    assert captured["retention_days"] == 30
    assert str(captured["location"]).endswith("/tmp/output")


def test_ensure_dataset_registered_returns_when_manifest_exists(
    registry_synchronizer: RegistrySynchronizer,
    data_registry: Mock,
) -> None:
    """Existing manifests should skip registration."""
    data_registry.get_manifest.return_value = object()

    registry_synchronizer._ensure_dataset_registered("test.dataset")

    data_registry.register_dataset.assert_not_called()


def test_ensure_dataset_registered_swallows_register_errors(
    registry_synchronizer: RegistrySynchronizer,
    data_registry: Mock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Registration backend errors should be swallowed."""
    data_registry.get_manifest.side_effect = RuntimeError("missing")
    data_registry.register_dataset.side_effect = RuntimeError("backend unavailable")
    monkeypatch.setattr(
        "ml.data.dataset_manifest_defaults.build_auto_dataset_manifest",
        lambda **_kwargs: object(),
    )

    registry_synchronizer._ensure_dataset_registered("test.dataset")

    data_registry.register_dataset.assert_called_once()


def test_capture_cli_build_artifacts_handles_metadata_parse_failure(
    registry_synchronizer: RegistrySynchronizer,
    dataset_cfg: DatasetBuildConfig,
    tmp_path: Path,
) -> None:
    """Metadata parse failures should not abort artifact capture."""
    cfg = dataclasses.replace(dataset_cfg, out_dir=str(tmp_path), register_features=False)
    (tmp_path / "feature_set.json").write_text("{invalid-json", encoding="utf-8")
    (tmp_path / "dataset_metadata.json").write_text("{oops", encoding="utf-8")

    artifacts = registry_synchronizer.capture_cli_build_artifacts(cfg)

    assert artifacts is not None
    assert artifacts.dataset_metadata is None
    assert artifacts.feature_names == ()


def test_capture_cli_build_artifacts_handles_feature_registration_write_failure(
    registry_synchronizer: RegistrySynchronizer,
    dataset_cfg: DatasetBuildConfig,
    dataset_metadata: DatasetMetadata,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Persist failures for registration metadata should remain best-effort."""
    cfg = dataclasses.replace(
        dataset_cfg,
        out_dir=str(tmp_path),
        register_features=True,
        feature_registry_dir=str(tmp_path / "registry"),
    )
    (tmp_path / "dataset_metadata.json").write_text(
        json.dumps(_metadata_json_payload(dataset_metadata), indent=2),
        encoding="utf-8",
    )
    monkeypatch.setattr(registry_synchronizer, "_infer_feature_names", lambda _out_dir: ("f1", "f2"))
    monkeypatch.setattr(registry_synchronizer, "_export_feature_manifest", lambda _cfg, _result: "manifest-1")

    original_write_text = Path.write_text

    def _write_text_with_failure(
        path: Path,
        data: str,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> int:
        if path.name == "feature_registration.json":
            raise OSError("disk full")
        return original_write_text(path, data, encoding=encoding, errors=errors, newline=newline)

    monkeypatch.setattr(Path, "write_text", _write_text_with_failure)

    artifacts = registry_synchronizer.capture_cli_build_artifacts(cfg)
    assert artifacts is not None
    assert artifacts.feature_set_id == "manifest-1"
    assert artifacts.feature_names == ("f1", "f2")


def test_guard_dataset_metadata_raises_for_missing_macro_observations(
    registry_synchronizer: RegistrySynchronizer,
    dataset_cfg: DatasetBuildConfig,
    dataset_metadata: DatasetMetadata,
) -> None:
    """Macro-enabled builds require observations for each requested series."""
    cfg = dataclasses.replace(
        dataset_cfg,
        include_macro=True,
        macro_series_ids=("CPI", "PCE"),
    )
    metadata = dataclasses.replace(dataset_metadata, macro_observation_counts={"CPI": 1})

    with pytest.raises(ValueError, match="Missing macro observations"):
        registry_synchronizer._guard_dataset_metadata(cfg=cfg, metadata=metadata)


def test_guard_dataset_metadata_requires_equs_source_provenance(
    registry_synchronizer: RegistrySynchronizer,
    dataset_cfg: DatasetBuildConfig,
    dataset_metadata: DatasetMetadata,
) -> None:
    """EQUS.MINI bindings must include source dataset provenance."""
    binding = MarketBindingMetadata(
        binding_id="binding-1",
        dataset_id="EQUS.MINI",
        descriptor_id=None,
        schema=None,
        storage_kind="parquet",
        symbols=("SPY",),
        instrument_ids=("SPY.XNAS",),
        source="catalog",
        license_start=None,
        license_end=None,
        ts_event_start=None,
        ts_event_end=None,
        rows_from_store=0,
        rows_from_catalog=1,
        source_datasets=None,
        provider_dataset_id=None,
        provider_schema=None,
    )
    metadata = dataclasses.replace(dataset_metadata, market_bindings=(binding,))

    with pytest.raises(ValueError, match="source_datasets"):
        registry_synchronizer._guard_dataset_metadata(cfg=dataset_cfg, metadata=metadata)


def test_export_feature_manifest_and_dataset_type_helpers(
    registry_synchronizer: RegistrySynchronizer,
    dataset_cfg: DatasetBuildConfig,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Cover feature export helper and dataset/storage coercion helpers."""
    cfg = dataclasses.replace(
        dataset_cfg,
        register_features=True,
        feature_registry_dir=str(tmp_path / "registry"),
        feature_role="unknown-role",
        include_l2=True,
    )
    result = SimpleNamespace(feature_names=["f1", "f2"])
    monkeypatch.setattr(
        "ml.data.feature_manifest_export.export_feature_manifest",
        lambda **_kwargs: "manifest-id",
    )

    manifest_id = registry_synchronizer._export_feature_manifest(cfg, result)

    assert manifest_id == "manifest-id"
    assert registry_synchronizer._resolve_storage_kind("POSTGRES") is StorageKind.POSTGRES
    assert registry_synchronizer._resolve_storage_kind("invalid") is StorageKind.PARQUET
    assert registry_synchronizer._resolve_dataset_type(dataset_type="bars", metadata=None) is DatasetType.BARS
    assert registry_synchronizer._coerce_dataset_type("bars") is DatasetType.BARS


def test_infer_feature_names_prefers_polars_and_falls_back_to_dependency_check(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """_infer_feature_names should use available engines and fall back cleanly."""
    dataset_path = tmp_path / "dataset.parquet"
    dataset_path.write_bytes(b"parquet-placeholder")

    class _Frame:
        columns = ["feat_a", "timestamp", "instrument_id", "y"]

    class _Polars:
        @staticmethod
        def read_parquet(_path: str) -> _Frame:
            return _Frame()

    import ml._imports as imports_module

    monkeypatch.setattr(imports_module, "HAS_POLARS", True)
    monkeypatch.setattr(imports_module, "pl", _Polars())
    monkeypatch.setattr(imports_module, "HAS_PANDAS", False)
    monkeypatch.setattr(imports_module, "pd", None)
    feature_names = RegistrySynchronizer._infer_feature_names(tmp_path)
    assert feature_names == ("feat_a",)

    monkeypatch.setattr(imports_module, "HAS_POLARS", False)
    monkeypatch.setattr(imports_module, "HAS_PANDAS", False)
    monkeypatch.setattr(imports_module, "check_ml_dependencies", lambda _deps: (_ for _ in ()).throw(RuntimeError("missing")))
    assert RegistrySynchronizer._infer_feature_names(tmp_path) == ()


def test_resolve_location_prefers_overrides_and_metadata() -> None:
    """Location resolution should use explicit, then metadata, then dataset fallback."""
    explicit = RegistrySynchronizer._resolve_location(
        dataset_id="test.dataset",
        location="~/tmp/dataset",
        metadata=None,
    )
    assert explicit.endswith("/tmp/dataset")

    from_metadata = RegistrySynchronizer._resolve_location(
        dataset_id="test.dataset",
        location=None,
        metadata={"out_dir": "/tmp/out"},
    )
    assert from_metadata == "/tmp/out"

    default_location = RegistrySynchronizer._resolve_location(
        dataset_id="test.dataset",
        location=None,
        metadata=None,
    )
    assert default_location.endswith("test.dataset")


def test_synchronize_dataset_manifest_swallows_update_errors(
    registry_synchronizer: RegistrySynchronizer,
    data_registry: Mock,
    dataset_cfg: DatasetBuildConfig,
    dataset_metadata: DatasetMetadata,
) -> None:
    """Manifest update failures should not propagate."""
    data_registry.update_manifest.side_effect = RuntimeError("update failed")
    registry_synchronizer.synchronize_dataset_manifest(cfg=dataset_cfg, metadata=dataset_metadata)
    data_registry.update_manifest.assert_called_once()


def test_emit_feature_refresh_event_empty_dataset_import_failure_and_no_publisher(
    registry_synchronizer: RegistrySynchronizer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Feature refresh event helper should guard empty inputs and dependency failures."""
    registry_synchronizer._emit_feature_refresh_event("", ("f1",))

    message_bus_module = importlib.import_module("ml.common.message_bus")
    monkeypatch.setitem(sys.modules, "ml.common.message_bus", None)
    registry_synchronizer._emit_feature_refresh_event("test.dataset", ("f1",))

    monkeypatch.setitem(sys.modules, "ml.common.message_bus", message_bus_module)
    monkeypatch.setattr("ml.common.message_bus.publisher_from_config", lambda _cfg: None)
    registry_synchronizer.message_bus = None
    registry_synchronizer._emit_feature_refresh_event("test.dataset", ("f1",))


def test_ensure_dataset_registered_guards_missing_registry_and_builder(
    data_registry: Mock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Registration helper should return when registry/builder is unavailable."""
    synchronizer = RegistrySynchronizer(data_registry=None)
    synchronizer._ensure_dataset_registered("test.dataset")

    data_registry.get_manifest.side_effect = RuntimeError("missing")
    synchronizer = RegistrySynchronizer(data_registry=data_registry)
    monkeypatch.setitem(sys.modules, "ml.data.dataset_manifest_defaults", None)
    synchronizer._ensure_dataset_registered("test.dataset")
    data_registry.register_dataset.assert_not_called()


def test_capture_cli_build_artifacts_raises_on_metadata_guard_violation(
    registry_synchronizer: RegistrySynchronizer,
    dataset_cfg: DatasetBuildConfig,
    dataset_metadata: DatasetMetadata,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Guardrail violations should surface as ValueError from capture helper."""
    cfg = dataclasses.replace(dataset_cfg, out_dir=str(tmp_path), register_features=False)
    (tmp_path / "dataset_metadata.json").write_text(
        json.dumps(_metadata_json_payload(dataset_metadata), indent=2),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        registry_synchronizer,
        "_guard_dataset_metadata",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("bad metadata")),
    )

    with pytest.raises(ValueError, match="Dataset metadata guardrail violation"):
        registry_synchronizer.capture_cli_build_artifacts(cfg)


def test_public_guard_and_record_wrappers(
    registry_synchronizer: RegistrySynchronizer,
    dataset_cfg: DatasetBuildConfig,
    dataset_metadata: DatasetMetadata,
) -> None:
    """Public wrapper methods should delegate to private helper implementations."""
    registry_synchronizer.guard_dataset_metadata(cfg=dataset_cfg, metadata=dataset_metadata)
    registry_synchronizer.record_build_artifacts(
        cfg=dataset_cfg,
        feature_set_id="set-1",
        feature_names=("f1",),
        feature_registry_dir="/tmp/features",
        dataset_metadata=dataset_metadata,
    )
    assert registry_synchronizer.build_artifacts is not None
    assert registry_synchronizer.build_artifacts.feature_set_id == "set-1"


def test_export_feature_manifest_branch_edges(
    registry_synchronizer: RegistrySynchronizer,
    dataset_cfg: DatasetBuildConfig,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Feature export helper should handle missing names/import and export failures."""
    cfg = dataclasses.replace(
        dataset_cfg,
        register_features=True,
        feature_registry_dir=str(tmp_path / "registry"),
    )

    assert registry_synchronizer._export_feature_manifest(cfg, SimpleNamespace()) is None
    assert registry_synchronizer._export_feature_manifest(cfg, SimpleNamespace(feature_names=[])) is None

    feature_manifest_module = importlib.import_module("ml.data.feature_manifest_export")
    monkeypatch.setitem(sys.modules, "ml.data.feature_manifest_export", None)
    assert registry_synchronizer._export_feature_manifest(cfg, SimpleNamespace(feature_names=["f1"])) is None
    monkeypatch.setitem(
        sys.modules,
        "ml.data.feature_manifest_export",
        feature_manifest_module,
    )

    monkeypatch.setattr(
        "ml.data.feature_manifest_export.export_feature_manifest",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("export failed")),
    )
    assert registry_synchronizer._export_feature_manifest(cfg, SimpleNamespace(feature_names=["f1"])) is None


def test_infer_feature_names_import_and_pandas_failure_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Infer helper should return empty on import failures or pandas parquet errors."""
    dataset_path = tmp_path / "dataset.parquet"
    dataset_path.write_bytes(b"placeholder")

    imports_module_ref = importlib.import_module("ml._imports")
    monkeypatch.setitem(sys.modules, "ml._imports", None)
    assert RegistrySynchronizer._infer_feature_names(tmp_path) == ()
    monkeypatch.setitem(sys.modules, "ml._imports", imports_module_ref)

    import ml._imports as imports_module

    class _FailingPandas:
        @staticmethod
        def read_parquet(_path: str) -> object:
            raise RuntimeError("read failed")

    monkeypatch.setattr(imports_module, "HAS_POLARS", False)
    monkeypatch.setattr(imports_module, "HAS_PANDAS", True)
    monkeypatch.setattr(imports_module, "pd", _FailingPandas())
    assert RegistrySynchronizer._infer_feature_names(tmp_path) == ()


def test_static_resolution_helpers_cover_additional_branches(
    registry_synchronizer: RegistrySynchronizer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cover retention/storage/type coercion edge branches."""
    assert RegistrySynchronizer._resolve_retention_days({"retention_days": -1}) == 90
    assert RegistrySynchronizer._resolve_storage_kind(StorageKind.POSTGRES) is StorageKind.POSTGRES
    assert registry_synchronizer._resolve_dataset_type(dataset_type=DatasetType.BARS, metadata=None) is DatasetType.BARS
    assert registry_synchronizer._resolve_dataset_type(dataset_type=None, metadata={"schema": "bars"}) is DatasetType.BARS

    monkeypatch.setattr(
        "ml.schema.map_schema_to_dataset_type",
        lambda _candidate: (_ for _ in ()).throw(RuntimeError("unknown")),
    )
    assert registry_synchronizer._coerce_dataset_type("unknown-token") is None
    assert RegistrySynchronizer._resolve_location(
        dataset_id="demo.dataset",
        location=None,
        metadata={},
    ).endswith("demo.dataset")
