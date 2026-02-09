from __future__ import annotations

import dataclasses
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock
import sys
import types

import pytest

from ml.config.market_data import MarketDatasetInput
from ml.data import DatasetMetadataExpectations
from ml.data.metadata import DatasetMetadata
from ml.data.metadata import MarketBindingMetadata
from ml.data.validation import DatasetValidationConfig
from ml.data.vintage import VintagePolicy
from ml.orchestration.config_types import DatasetBuildConfig
from ml.orchestration.dataset_builder import DatasetBuilder
from ml.registry.dataclasses import StorageKind
from ml.tests.utils.targets import build_default_target_semantics_payload


@pytest.fixture
def dataset_cfg(tmp_path: Path) -> DatasetBuildConfig:
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "out"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    return DatasetBuildConfig(
        data_dir=str(data_dir),
        symbols="SPY",
        out_dir=str(out_dir),
        dataset_id="test_dataset",
        target_semantics=build_default_target_semantics_payload(),
    )


@pytest.fixture
def dataset_builder() -> DatasetBuilder:
    return DatasetBuilder(build_main=Mock(return_value=0))


def test_build_dataset_rejects_non_config(dataset_builder: DatasetBuilder) -> None:
    with pytest.raises(TypeError):
        dataset_builder.build_dataset(object())  # type: ignore[arg-type]


def test_build_dataset_falls_back_to_cli_when_api_fails(
    dataset_builder: DatasetBuilder,
    dataset_cfg: DatasetBuildConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_builder.build_main.return_value = 3  # type: ignore[assignment]
    monkeypatch.setattr("ml.data.build_tft_dataset", Mock(side_effect=RuntimeError("api failure")))

    rc = dataset_builder.build_dataset(dataset_cfg)

    assert rc == 3
    dataset_builder.build_main.assert_called_once()


def test_build_dataset_records_artifacts_from_api_success(
    dataset_builder: DatasetBuilder,
    dataset_cfg: DatasetBuildConfig,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    api_result = SimpleNamespace(
        dataset_parquet=tmp_path / "dataset.parquet",
        dataset_csv=tmp_path / "dataset.csv",
        features_npz=tmp_path / "features.npz",
        feature_names=["f1", "f2"],
        feature_set_id="fs-123",
        metadata=None,
    )
    monkeypatch.setattr("ml.data.build_tft_dataset", Mock(return_value=api_result))

    rc = dataset_builder.build_dataset(dataset_cfg)

    assert rc == 0
    dataset_builder.build_main.assert_not_called()
    artifacts = dataset_builder.build_artifacts
    assert artifacts is not None
    assert artifacts.out_dir == Path(dataset_cfg.out_dir)
    assert artifacts.feature_set_id == "fs-123"
    assert artifacts.feature_names == ("f1", "f2")


def test_validate_dataset_returns_false_without_metadata(tmp_path: Path) -> None:
    builder = DatasetBuilder()
    dataset_path = tmp_path / "dataset.parquet"
    dataset_path.touch()

    passed, metadata = builder.validate_dataset(
        dataset_path,
        DatasetMetadataExpectations(),
        DatasetValidationConfig(),
    )

    assert passed is False
    assert metadata is None


def test_validate_dataset_passes_with_matching_metadata(tmp_path: Path) -> None:
    builder = DatasetBuilder()
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = dataset_dir / "dataset.parquet"
    dataset_path.touch()

    ts_start = "2024-01-01T00:00:00+00:00"
    ts_end = "2024-01-02T00:00:00+00:00"
    metadata_payload = {
        "dataset_id": "test_dataset",
        "vintage_policy": VintagePolicy.REAL_TIME.value,
        "vintage_cutoff": None,
        "build_ts": "2024-01-03T00:00:00+00:00",
        "ts_event_start": ts_start,
        "ts_event_end": ts_end,
        "overall_window": [ts_start, ts_end],
        "train_window": [ts_start, ts_end],
        "validation_window": None,
        "test_window": None,
        "macro_observation_counts": {"CPI": 2},
        "market_bindings": [],
    }
    (dataset_dir / "dataset_metadata.json").write_text(
        json.dumps(metadata_payload, indent=2),
        encoding="utf-8",
    )

    expectations = DatasetMetadataExpectations(
        dataset_id="test_dataset",
        vintage_policy=VintagePolicy.REAL_TIME,
        vintage_cutoff=None,
        ts_event_start=ts_start,
        ts_event_end=ts_end,
    )
    validation_cfg = DatasetValidationConfig(
        min_rows=1,
        expected_vintage_policy=VintagePolicy.REAL_TIME,
        require_macro_series=("CPI",),
        macro_min_vintage_observations=1,
    )

    passed, metadata = builder.validate_dataset(
        dataset_path,
        expectations,
        validation_cfg,
    )

    assert passed is True
    assert metadata is not None
    assert metadata.dataset_id == "test_dataset"
    assert metadata.vintage_policy is VintagePolicy.REAL_TIME


def _dataset_metadata_payload(metadata: DatasetMetadata) -> dict[str, Any]:
    """Serialize DatasetMetadata to JSON payload."""
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
        "column_info": {"vintage_timestamp_columns": ["macro__value_vintage_ts"]},
    }


def test_require_target_semantics_payload_validates_inputs(
    dataset_cfg: DatasetBuildConfig,
) -> None:
    """Target semantics payload parser should enforce JSON-object payloads."""
    cfg_missing = replace(dataset_cfg, target_semantics=None)
    with pytest.raises(ValueError, match="target_semantics must be provided"):
        DatasetBuilder._require_target_semantics_payload(cfg_missing)

    cfg_invalid_json = replace(dataset_cfg, target_semantics="{invalid")
    with pytest.raises(ValueError, match="JSON object payload"):
        DatasetBuilder._require_target_semantics_payload(cfg_invalid_json)

    cfg_list_payload = replace(dataset_cfg, target_semantics='["not-an-object"]')
    with pytest.raises(ValueError, match="JSON object payload"):
        DatasetBuilder._require_target_semantics_payload(cfg_list_payload)

    cfg_dict = replace(dataset_cfg, target_semantics={"version": "v1"})
    assert DatasetBuilder._require_target_semantics_payload(cfg_dict) == {"version": "v1"}


def test_build_via_cli_serializes_optional_flags_and_market_inputs(
    dataset_cfg: DatasetBuildConfig,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """CLI fallback should serialize optional knobs and invoke capture hooks."""
    call_args: list[str] = []

    def _build_main(args: list[str]) -> int:
        call_args.extend(args)
        return 0

    builder = DatasetBuilder(build_main=_build_main)
    monkeypatch.setattr(builder, "_capture_cli_build_artifacts", lambda _cfg: None)
    monkeypatch.setattr(builder, "_maybe_convert_vintage_dataset", lambda _cfg, path: path)

    cfg = replace(
        dataset_cfg,
        include_macro=True,
        include_micro=True,
        include_l2=True,
        include_events=True,
        include_calendar=True,
        include_macro_deltas=True,
        include_calendar_lags=True,
        include_clustering_tags=True,
        include_context_features=True,
        fred_vintage_dir=str(tmp_path / "fred"),
        events_dir=str(tmp_path / "events"),
        student_mode=True,
        market_dataset_id="EQUS.MINI",
        market_inputs=(
            MarketDatasetInput(
                descriptor_id="desc",
                dataset_id="EQUS.MINI",
                symbols=("SPY",),
                schema_override="bars",
                storage_kind_override=StorageKind.PARQUET,
                start="2024-01-01T00:00:00+00:00",
                end="2024-01-02T00:00:00+00:00",
            ),
        ),
        auto_refresh_macro=False,
        macro_staleness_hours=6,
        macro_series_ids=("CPIAUCSL", "UNRATE"),
        macro_fred_path=str(tmp_path / "macro.parquet"),
        validation=DatasetValidationConfig(
            min_rows=3,
            min_positive_rate=0.1,
            max_positive_rate=0.9,
            min_feature_coverage=0.5,
        ),
        start_iso="2024-01-01T00:00:00+00:00",
        end_iso="2024-01-02T00:00:00+00:00",
        chunk_days=5,
        emit_dataset_events=True,
        register_features=True,
        feature_registry_dir=None,
        convert_vintage_to_age=True,
    )

    rc = builder._build_via_cli(cfg)

    assert rc == 0
    assert "--include_macro" in call_args
    assert "--include_micro" in call_args
    assert "--include_l2" in call_args
    assert "--include_events" in call_args
    assert "--include_calendar" in call_args
    assert "--include_macro_deltas" in call_args
    assert "--include_calendar_lags" in call_args
    assert "--include_clustering_tags" in call_args
    assert "--include_context_features" in call_args
    assert "--market_inputs_json" in call_args
    assert "--skip_macro_refresh" in call_args
    assert "--macro_freshness_hours" in call_args
    assert "--macro_series_ids" in call_args
    assert "--macro_fred_path" in call_args
    assert "--validation_min_rows" in call_args
    assert "--validation_min_positive_rate" in call_args
    assert "--validation_max_positive_rate" in call_args
    assert "--validation_min_feature_coverage" in call_args
    assert "--start" in call_args
    assert "--end" in call_args
    assert "--chunk_days" in call_args
    assert "--emit_dataset_events" in call_args
    assert "--register_features" in call_args
    assert "--feature_registry_dir" in call_args
    assert "--convert-vintage-age" in call_args


def test_build_via_cli_requires_build_main(dataset_cfg: DatasetBuildConfig) -> None:
    """CLI path should fail fast when build_main is not configured."""
    builder = DatasetBuilder(build_main=None)
    with pytest.raises(RuntimeError, match="build_main not configured"):
        builder._build_via_cli(dataset_cfg)


def test_infer_dataset_row_count_from_metadata_and_csv(tmp_path: Path) -> None:
    """Row count inference should cover metadata and CSV fallback paths."""
    metadata_only = SimpleNamespace(
        metadata=SimpleNamespace(overall_window=None, ts_event_start=None, ts_event_end=None),
        dataset_parquet=None,
        dataset_csv=None,
    )
    assert DatasetBuilder._infer_dataset_row_count(metadata_only) == 0

    csv_only_header = tmp_path / "dataset_header.csv"
    csv_only_header.write_text("col\n", encoding="utf-8")
    csv_result_header = SimpleNamespace(metadata=None, dataset_parquet=None, dataset_csv=csv_only_header)
    assert DatasetBuilder._infer_dataset_row_count(csv_result_header) == 0

    csv_with_rows = tmp_path / "dataset_rows.csv"
    csv_with_rows.write_text("col\n1\n", encoding="utf-8")
    csv_result_rows = SimpleNamespace(metadata=None, dataset_parquet=None, dataset_csv=csv_with_rows)
    assert DatasetBuilder._infer_dataset_row_count(csv_result_rows) is None


def test_infer_dataset_row_count_from_parquet(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Parquet metadata branch should return explicit num_rows."""
    parquet_path = tmp_path / "dataset.parquet"
    parquet_path.write_bytes(b"placeholder")

    fake_pyarrow = types.ModuleType("pyarrow")
    fake_parquet = types.ModuleType("pyarrow.parquet")

    class _Metadata:
        num_rows = 7

    class _ParquetFile:
        def __init__(self, _path: str) -> None:
            self.metadata = _Metadata()

    fake_parquet.ParquetFile = _ParquetFile  # type: ignore[attr-defined]
    fake_pyarrow.parquet = fake_parquet  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "pyarrow", fake_pyarrow)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", fake_parquet)

    result = SimpleNamespace(metadata=None, dataset_parquet=parquet_path, dataset_csv=None)
    assert DatasetBuilder._infer_dataset_row_count(result) == 7


def test_export_feature_manifest_handles_unknown_role_and_missing_features(
    dataset_cfg: DatasetBuildConfig,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Feature export helper should tolerate role/import edge cases."""
    cfg_disabled = replace(dataset_cfg, register_features=False, feature_registry_dir=None)
    assert DatasetBuilder._export_feature_manifest(cfg_disabled, SimpleNamespace(feature_names=["f1"])) is None

    cfg_enabled = replace(
        dataset_cfg,
        register_features=True,
        feature_registry_dir=str(tmp_path / "registry"),
        feature_role="unknown-role",
        include_l2=True,
    )
    missing_names = DatasetBuilder._export_feature_manifest(cfg_enabled, SimpleNamespace())
    assert missing_names is None

    monkeypatch.setattr(
        "ml.data.feature_manifest_export.export_feature_manifest",
        lambda **_kwargs: "manifest-xyz",
    )
    manifest_id = DatasetBuilder._export_feature_manifest(
        cfg_enabled,
        SimpleNamespace(feature_names=["f1", "f2"]),
    )
    assert manifest_id == "manifest-xyz"


def test_maybe_convert_vintage_dataset_paths(
    dataset_cfg: DatasetBuildConfig,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Vintage conversion should handle already-converted and fresh conversion paths."""
    builder = DatasetBuilder()
    dataset_path = tmp_path / "dataset.parquet"
    dataset_path.write_bytes(b"placeholder")
    metadata_path = tmp_path / "dataset_metadata.json"
    metadata_payload = {
        "dataset_id": "test_dataset",
        "vintage_policy": VintagePolicy.REAL_TIME.value,
        "vintage_cutoff": None,
        "build_ts": "2024-01-03T00:00:00+00:00",
        "ts_event_start": None,
        "ts_event_end": None,
        "overall_window": None,
        "train_window": None,
        "validation_window": None,
        "test_window": None,
        "macro_observation_counts": {},
        "column_info": {"vintage_timestamp_columns": ["macro__value_vintage_ts"]},
    }
    metadata_path.write_text(json.dumps(metadata_payload), encoding="utf-8")

    cfg = replace(dataset_cfg, convert_vintage_to_age=True, out_dir=str(tmp_path))
    destination = dataset_path.with_name("dataset_with_vintage_age.parquet")
    destination.write_bytes(b"existing")

    written: dict[str, Any] = {}
    monkeypatch.setattr(
        "ml.orchestration.dataset_builder.update_metadata_with_vintage_age",
        lambda metadata, vintage_columns, age_columns: {
            **metadata,
            "column_info": {
                "vintage_timestamp_columns": list(vintage_columns),
                "vintage_age_columns": list(age_columns),
            },
        },
    )
    monkeypatch.setattr(
        "ml.orchestration.dataset_builder.write_metadata",
        lambda _path, payload: written.update(payload),
    )

    assert builder._maybe_convert_vintage_dataset(cfg, dataset_path) == destination
    assert "column_info" in written

    destination.unlink()
    monkeypatch.setattr(
        "ml.orchestration.dataset_builder.convert_vintage_timestamps_to_age",
        lambda _src, _dst: SimpleNamespace(
            vintage_columns=("macro__value_vintage_ts",),
            age_columns=("macro__vintage_age_minutes",),
        ),
    )
    assert builder._maybe_convert_vintage_dataset(cfg, dataset_path) == destination


def test_maybe_convert_vintage_dataset_requires_metadata_file(
    dataset_cfg: DatasetBuildConfig,
    tmp_path: Path,
) -> None:
    """Conversion requires dataset_metadata.json to exist."""
    builder = DatasetBuilder()
    dataset_path = tmp_path / "dataset.parquet"
    dataset_path.write_bytes(b"placeholder")
    cfg = replace(dataset_cfg, convert_vintage_to_age=True, out_dir=str(tmp_path))

    with pytest.raises(FileNotFoundError):
        builder._maybe_convert_vintage_dataset(cfg, dataset_path)


def test_guard_dataset_metadata_raises_on_macro_and_equs_gaps(
    dataset_cfg: DatasetBuildConfig,
) -> None:
    """Guardrails enforce macro observations and EQUS provenance."""
    builder = DatasetBuilder()
    cfg_macro = replace(
        dataset_cfg,
        include_macro=True,
        macro_series_ids=("CPI", "PCE"),
        vintage_as_of="not-an-iso",
    )
    metadata_macro = DatasetMetadata(
        dataset_id=dataset_cfg.dataset_id,
        vintage_policy=VintagePolicy.REAL_TIME,
        vintage_cutoff="not-an-iso",
        build_ts="2024-01-01T00:00:00+00:00",
        ts_event_start=None,
        ts_event_end=None,
        overall_window=None,
        train_window=None,
        validation_window=None,
        test_window=None,
        macro_observation_counts={"CPI": 1},
        market_bindings=None,
    )
    with pytest.raises(ValueError, match="Missing macro observations"):
        builder._guard_dataset_metadata(cfg=cfg_macro, metadata=metadata_macro)

    binding = MarketBindingMetadata(
        binding_id="b1",
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
    metadata_equs = dataclasses.replace(metadata_macro, market_bindings=(binding,))
    with pytest.raises(ValueError, match="source_datasets"):
        builder._guard_dataset_metadata(cfg=dataset_cfg, metadata=metadata_equs)


def test_synchronize_dataset_manifest_handles_registry_edges(
    dataset_cfg: DatasetBuildConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Manifest synchronization should handle missing manifest and update failures."""
    metadata = DatasetMetadata(
        dataset_id=dataset_cfg.dataset_id,
        vintage_policy=VintagePolicy.REAL_TIME,
        vintage_cutoff=None,
        build_ts="2024-01-01T00:00:00+00:00",
        ts_event_start="2024-01-01T00:00:00+00:00",
        ts_event_end="2024-01-02T00:00:00+00:00",
        overall_window=("2024-01-01T00:00:00+00:00", "2024-01-02T00:00:00+00:00"),
        train_window=None,
        validation_window=None,
        test_window=None,
        macro_observation_counts={},
        market_bindings=(),
    )

    registry = Mock()
    registry.get_manifest.return_value = SimpleNamespace(metadata={"existing": True})
    builder = DatasetBuilder(data_registry=registry)
    monkeypatch.setattr(builder, "_compute_dataset_pipeline_signature", lambda _cfg, _meta: "sig")
    builder._synchronize_dataset_manifest(cfg=dataset_cfg, metadata=metadata)
    registry.update_manifest.assert_called_once()

    registry_missing = Mock()
    registry_missing.get_manifest.side_effect = RuntimeError("missing")
    DatasetBuilder(data_registry=registry_missing)._synchronize_dataset_manifest(cfg=dataset_cfg, metadata=metadata)
    registry_missing.update_manifest.assert_not_called()

    registry_update_fail = Mock()
    registry_update_fail.get_manifest.return_value = SimpleNamespace(metadata={})
    registry_update_fail.update_manifest.side_effect = RuntimeError("write failed")
    DatasetBuilder(data_registry=registry_update_fail)._synchronize_dataset_manifest(
        cfg=dataset_cfg,
        metadata=metadata,
    )
    registry_update_fail.update_manifest.assert_called_once()


def test_infer_feature_names_prefers_polars_then_dependency_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Feature-name inference should use available readers and degrade cleanly."""
    dataset_path = tmp_path / "dataset.parquet"
    dataset_path.write_bytes(b"placeholder")

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
    assert DatasetBuilder._infer_feature_names(tmp_path) == ("feat_a",)

    monkeypatch.setattr(imports_module, "HAS_POLARS", False)
    monkeypatch.setattr(imports_module, "HAS_PANDAS", False)
    monkeypatch.setattr(
        imports_module,
        "check_ml_dependencies",
        lambda _deps: (_ for _ in ()).throw(RuntimeError("dependency missing")),
    )
    assert DatasetBuilder._infer_feature_names(tmp_path) == ()


def test_capture_cli_build_artifacts_handles_parse_and_guardrail_failures(
    dataset_cfg: DatasetBuildConfig,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """CLI artifact capture should tolerate parse issues and raise on guardrail failures."""
    builder = DatasetBuilder()
    cfg = replace(
        dataset_cfg,
        out_dir=str(tmp_path),
        register_features=True,
        feature_registry_dir=str(tmp_path / "registry"),
    )
    (tmp_path / "feature_set.json").write_text("{broken", encoding="utf-8")
    (tmp_path / "feature_registration.json").write_text('{"feature_names": "not-a-list"}', encoding="utf-8")
    (tmp_path / "dataset_metadata.json").write_text("{broken", encoding="utf-8")

    builder._capture_cli_build_artifacts(cfg)
    artifacts = builder.build_artifacts
    assert artifacts is not None
    assert artifacts.feature_names == ()

    metadata = DatasetMetadata(
        dataset_id=cfg.dataset_id,
        vintage_policy=VintagePolicy.REAL_TIME,
        vintage_cutoff=None,
        build_ts="2024-01-01T00:00:00+00:00",
        ts_event_start=None,
        ts_event_end=None,
        overall_window=None,
        train_window=None,
        validation_window=None,
        test_window=None,
        macro_observation_counts={},
        market_bindings=(),
    )
    (tmp_path / "dataset_metadata.json").write_text(
        json.dumps(_dataset_metadata_payload(metadata), indent=2),
        encoding="utf-8",
    )
    monkeypatch.setattr(builder, "_guard_dataset_metadata", lambda **_kwargs: (_ for _ in ()).throw(ValueError("guardrail")))
    with pytest.raises(ValueError, match="guardrail"):
        builder._capture_cli_build_artifacts(cfg)
