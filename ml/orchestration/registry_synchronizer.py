#!/usr/bin/env python3

"""
Registry synchronization for ML pipeline orchestrator.

This module provides dataset manifest synchronization, feature export,
and build artifact recording for ML pipelines.

This component is extracted from the MLPipelineOrchestrator god class to provide
focused, testable registry synchronization functionality.

"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

from ml.data import DatasetMetadata
from ml.data import DatasetMetadataExpectations
from ml.data import compute_dataset_pipeline_signature
from ml.data import load_dataset_metadata
from ml.data import validate_dataset_metadata_expectations
from ml.data.vintage import format_dt
from ml.data.vintage import parse_dt
from ml.orchestration.config_types import DatasetBuildConfig
from ml.orchestration.dataset_builder import BuildArtifacts
from ml.registry.dataclasses import DatasetType
from ml.registry.dataclasses import StorageKind


if TYPE_CHECKING:
    from ml.registry.protocols import RegistryProtocol


logger = logging.getLogger(__name__)


# ========================================================================
# Protocol Definition
# ========================================================================


class RegistrySynchronizerProtocol(Protocol):
    """
    Protocol for registry synchronization operations.
    """

    def synchronize_dataset_manifest(
        self,
        *,
        cfg: DatasetBuildConfig,
        metadata: DatasetMetadata,
    ) -> None:
        """
        Synchronize dataset manifest metadata with the data registry.

        Parameters
        ----------
        cfg : DatasetBuildConfig
            Dataset build configuration
        metadata : DatasetMetadata
            Dataset metadata to synchronize

        """
        ...

    def capture_cli_build_artifacts(
        self,
        cfg: DatasetBuildConfig,
    ) -> BuildArtifacts | None:
        """
        Capture build artifacts from CLI output directory.

        Parameters
        ----------
        cfg : DatasetBuildConfig
            Dataset build configuration

        Returns
        -------
        BuildArtifacts | None
            Captured build artifacts or None if capture failed

        """
        ...


# ========================================================================
# RegistrySynchronizer Implementation
# ========================================================================


class RegistrySynchronizer:
    """
    Synchronizes dataset manifests with registries and captures build artifacts.

    Handles manifest metadata synchronization, feature export, and build artifact
    recording for ML pipelines.

    This component is extracted from the MLPipelineOrchestrator god class to
    provide focused, testable registry synchronization functionality.

    Parameters
    ----------
    data_registry : RegistryProtocol | None
        Data registry for manifest synchronization
    feature_registry : object | None
        Feature registry for feature manifest export

    """

    def __init__(
        self,
        *,
        data_registry: RegistryProtocol | None = None,
        feature_registry: object | None = None,
        model_registry: object | None = None,
        message_bus: object | None = None,
    ) -> None:
        """
        Initialize registry synchronizer.

        Parameters
        ----------
        data_registry : RegistryProtocol | None
            Data registry instance
        feature_registry : object | None
            Feature registry instance

        """
        self._data_registry = data_registry
        self._feature_registry = feature_registry
        self.data_registry = data_registry
        self.feature_registry = feature_registry
        self.model_registry = model_registry
        self.message_bus = message_bus
        self._build_artifacts: BuildArtifacts | None = None

        logger.debug("Initialized RegistrySynchronizer")

    @property
    def build_artifacts(self) -> BuildArtifacts | None:
        """Return the current build artifacts."""
        return self._build_artifacts

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def synchronize_dataset_manifest(
        self,
        *,
        cfg: DatasetBuildConfig,
        metadata: DatasetMetadata,
    ) -> None:
        """
        Synchronize dataset manifest metadata with the data registry.

        Parameters
        ----------
        cfg : DatasetBuildConfig
            Dataset build configuration
        metadata : DatasetMetadata
            Dataset metadata to synchronize

        """
        if self._data_registry is None or not cfg.dataset_id:
            return
        registry = self._data_registry

        try:
            manifest = registry.get_manifest(cfg.dataset_id)
        except Exception:
            logger.debug(
                "Data registry manifest missing for dataset_id=%s; skipping metadata sync",
                cfg.dataset_id,
                exc_info=True,
            )
            return

        manifest_metadata = dict(getattr(manifest, "metadata", {}) or {})
        manifest_metadata.update(
            {
                "dataset_id": cfg.dataset_id,
                "vintage": {
                    "policy": metadata.vintage_policy.value,
                    "cutoff": metadata.vintage_cutoff,
                    "build_ts": metadata.build_ts,
                },
                "windows": {
                    "overall": metadata.overall_window,
                    "train": metadata.train_window,
                    "validation": metadata.validation_window,
                    "test": metadata.test_window,
                    "ts_event_start": metadata.ts_event_start,
                    "ts_event_end": metadata.ts_event_end,
                },
                "market_bindings": [
                    {
                        "binding_id": binding.binding_id,
                        "dataset_id": binding.dataset_id,
                        "descriptor_id": binding.descriptor_id,
                        "source": binding.source,
                        "storage_kind": binding.storage_kind,
                        "symbols": list(binding.symbols),
                        "instrument_ids": list(binding.instrument_ids),
                    }
                    for binding in (metadata.market_bindings or ())
                ],
            },
        )

        try:
            registry.update_manifest(
                cfg.dataset_id,
                {
                    "metadata": manifest_metadata,
                    "pipeline_signature": self._compute_dataset_pipeline_signature(cfg, metadata),
                },
            )
        except Exception as exc:
            logger.debug(
                "Failed to update dataset manifest metadata: %s",
                exc,
                exc_info=True,
            )

    def _emit_feature_refresh_event(
        self,
        dataset_id: str,
        features: Sequence[str] | None = None,
    ) -> None:
        """
        Emit a feature refresh event to the configured message bus.

        Best-effort: returns silently if dependencies are unavailable or publishing
        fails. Topics honor MessageBusConfig and use the canonical Stage/Source/Status
        enums per AGENTS.md.
        """
        if not dataset_id:
            return

        try:
            from ml.common.message_bus import MessagePublisherProtocol
            from ml.common.message_bus import publisher_from_config
            from ml.common.message_topics import build_topic_for_stage
            from ml.config.bus import MessageBusConfig
            from ml.config.events import EventStatus
            from ml.config.events import Source
            from ml.config.events import Stage
        except Exception:
            logger.debug(
                "Message bus dependencies unavailable; skipping feature refresh event",
                exc_info=True,
            )
            return

        try:
            bus_cfg = MessageBusConfig.from_env()
        except Exception as exc:
            logger.debug(
                "MessageBusConfig.from_env failed; using defaults",
                exc_info=True,
                extra={"reason": str(exc)},
            )
            bus_cfg = MessageBusConfig()

        features_list = tuple(str(feature) for feature in (features or ()))
        publisher_obj: MessagePublisherProtocol | None = None
        if hasattr(self.message_bus, "publish"):
            publisher_obj = cast(MessagePublisherProtocol, self.message_bus)
        else:
            publisher_obj = publisher_from_config(bus_cfg)

        if publisher_obj is None:
            return

        topic = build_topic_for_stage(
            Stage.FEATURE_COMPUTED,
            dataset_id,
            scheme=bus_cfg.scheme,
            prefix=bus_cfg.topic_prefix,
        )
        payload = {
            "event_type": "feature_refresh",
            "dataset_id": dataset_id,
            "features": list(features_list),
            "stage": Stage.FEATURE_COMPUTED.value,
            "source": Source.HISTORICAL.value,
            "status": EventStatus.SUCCESS.value,
            "timestamp_ns": time.time_ns(),
        }

        try:
            publisher_obj.publish(topic, payload)
        except Exception as exc:  # pragma: no cover - defensive best-effort
            logger.debug(
                "Feature refresh publish failed",
                exc_info=True,
                extra={
                    "dataset_id": dataset_id,
                    "topic": topic,
                    "reason": str(exc),
                },
            )

    def _ensure_dataset_registered(
        self,
        dataset_id: str,
        metadata: Mapping[str, Any] | None = None,
        *,
        dataset_type: DatasetType | str | None = None,
        location: str | Path | None = None,
        storage_kind: StorageKind | str | None = None,
        retention_days: int | None = None,
    ) -> None:
        """
        Ensure a dataset manifest exists in the registry (best-effort).

        Args:
            dataset_id: Identifier of the dataset to register.
            metadata: Optional metadata used to infer dataset type/location.
            dataset_type: Explicit DatasetType or schema token override.
            location: Storage location override for the manifest.
            storage_kind: Storage backend override (parquet/postgres).
            retention_days: Retention override (defaults to 90 days).
        """
        registry_obj = self._data_registry
        if registry_obj is None or not dataset_id:
            return
        registry = registry_obj

        try:
            registry.get_manifest(dataset_id)
            return
        except Exception:
            logger.debug(
                "Dataset manifest missing; attempting registration",
                exc_info=True,
                extra={"dataset_id": dataset_id},
            )

        dataset_type_enum = self._resolve_dataset_type(
            dataset_type=dataset_type,
            metadata=metadata,
        )
        storage_kind_enum = self._resolve_storage_kind(storage_kind)
        resolved_location = self._resolve_location(
            dataset_id=dataset_id,
            location=location,
            metadata=metadata,
        )
        retention = retention_days or self._resolve_retention_days(metadata)

        try:
            from ml.data.dataset_manifest_defaults import build_auto_dataset_manifest
        except Exception:
            logger.debug(
                "Dataset manifest builder unavailable; skipping registration",
                exc_info=True,
            )
            return

        manifest = build_auto_dataset_manifest(
            dataset_id=dataset_id,
            dataset_type=dataset_type_enum,
            location=resolved_location,
            storage_kind=storage_kind_enum,
            pipeline_signature="registry_synchronizer",
            retention_days=retention,
            metadata={
                "auto_registered": True,
                "storage_path": resolved_location,
                "source": "registry_synchronizer",
            },
        )

        try:
            registry.register_dataset(manifest)
        except Exception as exc:  # pragma: no cover - best-effort path
            logger.debug(
                "Dataset registration skipped",
                exc_info=True,
                extra={
                    "dataset_id": dataset_id,
                    "dataset_type": dataset_type_enum.value,
                    "reason": str(exc),
                },
            )

    def capture_cli_build_artifacts(
        self,
        cfg: DatasetBuildConfig,
    ) -> BuildArtifacts | None:
        """
        Capture build artifacts from CLI output directory.

        Parameters
        ----------
        cfg : DatasetBuildConfig
            Dataset build configuration

        Returns
        -------
        BuildArtifacts | None
            Captured build artifacts or None if capture failed

        """
        out_dir = Path(cfg.out_dir)
        feature_registry_dir = cfg.feature_registry_dir
        feature_set_id: str | None = None
        feature_names: tuple[str, ...] = ()

        for candidate in (
            out_dir / "feature_set.json",
            out_dir / "feature_registration.json",
        ):
            if not candidate.exists():
                continue
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.debug("Failed to parse feature metadata %s: %s", candidate, exc)
                continue
            feature_registry_dir = feature_registry_dir or data.get("feature_registry_dir")
            feature_set_id = feature_set_id or data.get("feature_set_id")
            names = data.get("feature_names")
            if isinstance(names, list):
                feature_names = tuple(str(name) for name in names)

        if not feature_names:
            feature_names = self._infer_feature_names(out_dir)

        manifest_id: str | None = None
        dataset_metadata: DatasetMetadata | None = None
        metadata_path = out_dir / "dataset_metadata.json"
        if metadata_path.exists():
            try:
                dataset_metadata = load_dataset_metadata(metadata_path)
            except Exception as exc:
                logger.warning(
                    "Failed to load dataset metadata from %s: %s",
                    metadata_path,
                    exc,
                )
        else:
            logger.debug("Dataset metadata not found at %s; continuing without it", metadata_path)

        if cfg.register_features and feature_names:
            sentinel = type("_Result", (), {"feature_names": feature_names})
            manifest_id = self._export_feature_manifest(cfg, sentinel)
            if manifest_id:
                feature_set_id = feature_set_id or manifest_id
                feature_registry_dir = feature_registry_dir or cfg.feature_registry_dir
                payload = {
                    "feature_set_id": feature_set_id,
                    "feature_registry_dir": feature_registry_dir,
                    "feature_names": list(feature_names),
                    "manifest_id": manifest_id,
                }
                try:
                    (out_dir / "feature_registration.json").write_text(
                        json.dumps(payload, indent=2),
                        encoding="utf-8",
                    )
                except Exception as exc:
                    logger.debug(
                        "Failed to persist feature registration metadata: %s",
                        exc,
                    )

        if dataset_metadata is not None:
            try:
                self._guard_dataset_metadata(cfg=cfg, metadata=dataset_metadata)
            except Exception as exc:
                raise ValueError(f"Dataset metadata guardrail violation: {exc}") from exc
            self.synchronize_dataset_manifest(cfg=cfg, metadata=dataset_metadata)

        self._record_build_artifacts(
            cfg=cfg,
            feature_set_id=feature_set_id,
            feature_names=feature_names,
            feature_registry_dir=feature_registry_dir,
            dataset_metadata=dataset_metadata,
        )

        return self._build_artifacts

    def guard_dataset_metadata(
        self,
        *,
        cfg: DatasetBuildConfig,
        metadata: DatasetMetadata,
    ) -> None:
        """
        Validate dataset metadata against configuration guardrails.

        Parameters
        ----------
        cfg : DatasetBuildConfig
            Dataset build configuration
        metadata : DatasetMetadata
            Dataset metadata to validate

        Raises
        ------
        ValueError
            If metadata fails guardrail validation

        """
        self._guard_dataset_metadata(cfg=cfg, metadata=metadata)

    def record_build_artifacts(
        self,
        *,
        cfg: DatasetBuildConfig,
        feature_set_id: str | None,
        feature_names: Sequence[str] | None,
        feature_registry_dir: str | None,
        dataset_metadata: DatasetMetadata | None = None,
    ) -> None:
        """
        Record build artifacts for later use.

        Parameters
        ----------
        cfg : DatasetBuildConfig
            Dataset build configuration
        feature_set_id : str | None
            Feature set ID
        feature_names : Sequence[str] | None
            List of feature names
        feature_registry_dir : str | None
            Path to feature registry
        dataset_metadata : DatasetMetadata | None
            Dataset metadata

        """
        self._record_build_artifacts(
            cfg=cfg,
            feature_set_id=feature_set_id,
            feature_names=feature_names,
            feature_registry_dir=feature_registry_dir,
            dataset_metadata=dataset_metadata,
        )

    # -------------------------------------------------------------------------
    # Private Helpers
    # -------------------------------------------------------------------------

    def _guard_dataset_metadata(
        self,
        *,
        cfg: DatasetBuildConfig,
        metadata: DatasetMetadata,
    ) -> None:
        """Validate dataset metadata against configuration guardrails."""

        def _normalize(value: str | None) -> str | None:
            if not value:
                return None
            try:
                dt_value = parse_dt(value)
            except ValueError:
                return value
            if dt_value is None:
                return value
            formatted = format_dt(dt_value)
            return formatted or value

        expectations = DatasetMetadataExpectations(
            dataset_id=cfg.dataset_id,
            vintage_policy=cfg.vintage_policy,
            vintage_cutoff=_normalize(cfg.vintage_as_of),
            ts_event_start=_normalize(cfg.start_iso),
            ts_event_end=_normalize(cfg.end_iso),
        )
        validate_dataset_metadata_expectations(
            metadata,
            expectations,
            context="orchestrator.dataset",
        )
        if cfg.include_macro and cfg.macro_series_ids:
            missing = []
            for series in cfg.macro_series_ids:
                key = str(series)
                if metadata.macro_observation_counts.get(key, 0) <= 0:
                    missing.append(key)
            if missing:
                missing_str = ", ".join(sorted(missing))
                raise ValueError(
                    f"Missing macro observations for series: {missing_str}",
                )
        if metadata.market_bindings:
            for binding in metadata.market_bindings:
                if (binding.dataset_id or "").upper() != "EQUS.MINI":
                    continue
                if not binding.source_datasets:
                    raise ValueError(
                        "EQUS.MINI metadata missing source_datasets provenance",
                    )

    def _record_build_artifacts(
        self,
        *,
        cfg: DatasetBuildConfig,
        feature_set_id: str | None,
        feature_names: Sequence[str] | None,
        feature_registry_dir: str | None,
        dataset_metadata: DatasetMetadata | None = None,
    ) -> None:
        """Record build artifacts for later use."""
        names_tuple = tuple(feature_names or [])
        self._build_artifacts = BuildArtifacts(
            out_dir=Path(cfg.out_dir),
            feature_registry_dir=feature_registry_dir,
            feature_set_id=feature_set_id,
            feature_names=names_tuple,
            dataset_metadata=dataset_metadata,
        )

    @staticmethod
    def _export_feature_manifest(
        cfg: DatasetBuildConfig,
        result: object,
    ) -> str | None:
        """Export a feature manifest when registry configuration is provided."""
        if not cfg.register_features or not cfg.feature_registry_dir:
            return None

        try:
            feature_names = getattr(result, "feature_names")
        except AttributeError:
            logger.warning("Feature manifest export skipped: result missing feature_names")
            return None
        if not feature_names:
            logger.warning("Feature manifest export skipped: no feature names returned")
            return None

        try:
            from ml.data.feature_manifest_export import FeatureExportConfig
            from ml.data.feature_manifest_export import export_feature_manifest
            from ml.registry.base import DataRequirements
            from ml.registry.feature_registry import FeatureRole
        except Exception as exc:
            logger.warning("Feature manifest export unavailable: %s", exc)
            return None

        try:
            role = FeatureRole(cfg.feature_role)
        except ValueError:
            logger.warning("Unknown feature_role '%s'; defaulting to TEACHER", cfg.feature_role)
            role = FeatureRole.TEACHER

        data_requirements = DataRequirements.L1_L2 if cfg.include_l2 else DataRequirements.L1_ONLY
        flags = {
            "include_macro": cfg.include_macro,
            "macro_lag_days": cfg.macro_lag_days,
            "include_events": cfg.include_events,
            "include_l2": cfg.include_l2,
            "student_mode": cfg.student_mode,
            "fred_vintages": bool(cfg.fred_vintage_dir),
            "events_dir": bool(cfg.events_dir),
        }

        export_cfg = FeatureExportConfig(
            registry_path=Path(cfg.feature_registry_dir),
            role=role,
            data_requirements=data_requirements,
        )

        try:
            manifest_id = export_feature_manifest(
                feature_names=list(feature_names),
                flags=flags,
                cfg=export_cfg,
            )
            logger.info("Exported feature manifest %s", manifest_id)
            return manifest_id
        except Exception as exc:
            logger.warning("Feature manifest export failed: %s", exc, exc_info=True)
            return None

    @staticmethod
    def _compute_dataset_pipeline_signature(
        cfg: DatasetBuildConfig,
        metadata: DatasetMetadata,
    ) -> str:
        """Derive a stable pipeline signature covering vintage policy and scope."""
        return compute_dataset_pipeline_signature(
            dataset_id=cfg.dataset_id,
            symbols=cfg.symbols,
            instrument_ids=cfg.instrument_ids,
            macro_series_ids=cfg.macro_series_ids,
            include_macro=cfg.include_macro,
            macro_lag_days=cfg.macro_lag_days,
            vintage_policy=metadata.vintage_policy,
            vintage_cutoff=metadata.vintage_cutoff,
            ts_event_start=metadata.ts_event_start,
            ts_event_end=metadata.ts_event_end,
            market_bindings=metadata.market_bindings,
        )

    @staticmethod
    def _infer_feature_names(out_dir: Path) -> tuple[str, ...]:
        """Infer feature names from dataset parquet file."""
        dataset_path = out_dir / "dataset.parquet"
        if not dataset_path.exists():
            logger.debug("Dataset parquet missing after CLI build: %s", dataset_path)
            return ()
        try:
            from ml._imports import HAS_PANDAS
            from ml._imports import HAS_POLARS
            from ml._imports import check_ml_dependencies
            from ml._imports import pd
            from ml._imports import pl
        except Exception as exc:
            logger.debug("Failed to import dataset engines: %s", exc)
            return ()

        exclude = {"y", "time_index", "timestamp", "instrument_id", "ts_event"}
        try:
            if HAS_POLARS and pl is not None:
                frame = pl.read_parquet(str(dataset_path))
                return tuple(col for col in frame.columns if col not in exclude)
            if HAS_PANDAS and pd is not None:
                frame_pd = pd.read_parquet(str(dataset_path))
                return tuple(col for col in frame_pd.columns if col not in exclude)
        except Exception as exc:
            logger.warning("Failed to inspect dataset parquet: %s", exc)
            return ()
        try:
            check_ml_dependencies(["polars"])
        except Exception:
            logger.debug("Missing polars dependency when inspecting dataset parquet", exc_info=True)
        return ()

    @staticmethod
    def _resolve_retention_days(metadata: Mapping[str, Any] | None) -> int:
        """
        Resolve retention days from metadata, defaulting to 90.
        """
        if metadata is not None:
            value = metadata.get("retention_days")
            if isinstance(value, int) and value > 0:
                return value
        return 90

    @staticmethod
    def _resolve_storage_kind(storage_kind: StorageKind | str | None) -> StorageKind:
        """
        Normalize storage kind input to StorageKind enum.
        """
        if isinstance(storage_kind, StorageKind):
            return storage_kind
        if isinstance(storage_kind, str):
            normalized = storage_kind.strip().lower()
            for kind in StorageKind:
                if normalized in {kind.value, kind.name.lower()}:
                    return kind
        return StorageKind.PARQUET

    def _resolve_dataset_type(
        self,
        *,
        dataset_type: DatasetType | str | None,
        metadata: Mapping[str, Any] | None,
    ) -> DatasetType:
        """
        Infer DatasetType from explicit input or metadata.
        """
        if isinstance(dataset_type, DatasetType):
            return dataset_type

        candidate: str | None = None
        if isinstance(dataset_type, str):
            candidate = dataset_type
        elif metadata is not None:
            for key in ("dataset_type", "type", "schema"):
                value = metadata.get(key)
                if isinstance(value, str):
                    candidate = value
                    break

        coerced = self._coerce_dataset_type(candidate) if candidate else None
        return coerced or DatasetType.BARS

    @staticmethod
    def _coerce_dataset_type(candidate: str) -> DatasetType | None:
        """
        Attempt to coerce a string into a DatasetType, falling back to schema mapping.
        """
        try:
            return DatasetType(candidate)
        except Exception:
            pass
        try:
            from ml.schema import map_schema_to_dataset_type

            return map_schema_to_dataset_type(candidate)
        except Exception:
            return None

    @staticmethod
    def _resolve_location(
        *,
        dataset_id: str,
        location: str | Path | None,
        metadata: Mapping[str, Any] | None,
    ) -> str:
        """
        Resolve storage location from explicit input or metadata.
        """
        if location:
            return str(Path(location).expanduser())

        candidate_fields = (
            "location",
            "path",
            "catalog_path",
            "data_dir",
            "out_dir",
        )
        if metadata is not None:
            for field in candidate_fields:
                value = metadata.get(field)
                if isinstance(value, (str, Path)):
                    return str(Path(value).expanduser())

        return str(Path(f"./{dataset_id}").expanduser())
