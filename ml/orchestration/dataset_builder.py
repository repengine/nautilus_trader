#!/usr/bin/env python3

"""
Dataset building for ML pipeline orchestrator.

This module provides comprehensive dataset building including construction from market
data, feature engineering, validation against expectations, metadata management, and storage.

This component is extracted from the MLPipelineOrchestrator god class to provide
focused, testable dataset building functionality.

"""

from __future__ import annotations

import dataclasses
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from ml.data import DatasetMetadata
from ml.data import DatasetMetadataExpectations
from ml.data import DatasetValidationConfig
from ml.data import compute_dataset_pipeline_signature
from ml.data import load_dataset_metadata
from ml.data import validate_dataset_metadata_expectations
from ml.data.vintage import format_dt
from ml.data.vintage import parse_dt
from ml.orchestration.config_types import DatasetBuildConfig
from ml.preprocessing.vintage_age import convert_vintage_timestamps_to_age
from ml.preprocessing.vintage_age import update_metadata_with_vintage_age
from ml.preprocessing.vintage_age import write_metadata
from ml.registry.protocols import RegistryProtocol
from ml.stores.protocols import DataStoreFacadeProtocol


if TYPE_CHECKING:  # pragma: no cover - type-only imports
    from collections.abc import Callable


logger = logging.getLogger(__name__)


# ========================================================================
# Build Artifacts Dataclass
# ========================================================================


@dataclass(slots=True, frozen=True)
class BuildArtifacts:
    """Build artifacts from dataset construction."""

    out_dir: Path
    feature_set_id: str | None = None
    feature_names: tuple[str, ...] = ()
    feature_registry_dir: str | None = None
    dataset_metadata: DatasetMetadata | None = None


# ========================================================================
# Exceptions
# ========================================================================


@dataclass(slots=True, frozen=True)
class _EmptyDatasetError(Exception):
    """Dataset build produced zero rows."""

    message: str
    row_count: int | None = None

    def __str__(self) -> str:
        return self.message


# ========================================================================
# Protocol Definition
# ========================================================================


class DatasetBuilderProtocol(Protocol):
    """
    Protocol for dataset building operations.
    """

    def build_dataset(self, cfg: DatasetBuildConfig) -> int:
        """
        Build ML dataset.

        Parameters
        ----------
        cfg : DatasetBuildConfig
            Dataset build configuration

        Returns
        -------
        int
            Exit code (0 for success)

        """
        ...

    def validate_dataset(
        self,
        dataset_path: Path,
        expectations: DatasetMetadataExpectations,
        validation_config: DatasetValidationConfig,
    ) -> tuple[bool, DatasetMetadata | None]:
        """
        Validate dataset against expectations.

        Parameters
        ----------
        dataset_path : Path
            Path to dataset file
        expectations : DatasetMetadataExpectations
            Expected dataset metadata
        validation_config : DatasetValidationConfig
            Validation configuration

        Returns
        -------
        tuple[bool, DatasetMetadata | None]
            (validation_passed, dataset_metadata)

        """
        del dataset_path, expectations, validation_config
        raise NotImplementedError


# ========================================================================
# DatasetBuilder Implementation
# ========================================================================


class DatasetBuilder:
    """
    Builds and validates ML datasets.

    Handles dataset construction from market data, feature engineering,
    validation against expectations, metadata management, and storage.

    This component is extracted from the MLPipelineOrchestrator god class to
    provide focused, testable dataset building functionality.

    """

    def __init__(
        self,
        *,
        data_store: DataStoreFacadeProtocol | None = None,
        data_registry: RegistryProtocol | None = None,
        build_main: Callable[[list[str]], int] | None = None,
    ) -> None:
        """
        Initialize dataset builder.

        Parameters
        ----------
        data_store : DataStoreFacadeProtocol | None
            Data store for dataset persistence
        data_registry : RegistryProtocol | None
            Registry for dataset registration
        build_main : Callable[[list[str]], int] | None
            CLI main function for dataset building

        """
        self.data_store = data_store
        self.data_registry = data_registry
        self.build_main = build_main
        self._build_artifacts: BuildArtifacts | None = None

        logger.debug("Initialized DatasetBuilder")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def build_dataset(self, cfg: DatasetBuildConfig) -> int:
        """
        Build ML dataset from configuration.

        Attempts API build first, falls back to CLI if API unavailable or fails.

        Parameters
        ----------
        cfg : DatasetBuildConfig
            Dataset build configuration

        Returns
        -------
        int
            Exit code (0 for success)

        """
        # Prefer the public API to capture BuildResult (feature_set_id)
        try:
            from ml.data import DatasetBuildConfig as APICfg
            from ml.data import build_tft_dataset as api_build

            symbols_list = [s.strip().upper() for s in cfg.symbols.split(",") if s.strip()]
            instrument_ids_list = (
                [item.strip() for item in cfg.instrument_ids] if cfg.instrument_ids else None
            )
            vintage_as_of_dt = parse_dt(cfg.vintage_as_of)

            api_cfg = APICfg(
                data_dir=Path(cfg.data_dir),
                out_dir=Path(cfg.out_dir),
                dataset_id=cfg.dataset_id,
                symbols=symbols_list,
                instrument_ids=instrument_ids_list,
                include_macro=cfg.include_macro,
                macro_lag_days=cfg.macro_lag_days,
                include_micro=cfg.include_micro,
                include_l2=cfg.include_l2,
                include_events=cfg.include_events,
                include_calendar=cfg.include_calendar,
                fred_vintage_dir=(
                    Path(cfg.fred_vintage_dir).expanduser() if cfg.fred_vintage_dir else None
                ),
                events_base_dir=(Path(cfg.events_dir).expanduser() if cfg.events_dir else None),
                student_mode=cfg.student_mode,
                horizon_minutes=cfg.horizon_minutes,
                threshold=cfg.threshold,
                lookback_periods=cfg.lookback_periods,
                start=(
                    None
                    if not cfg.start_iso
                    else __import__("datetime").datetime.fromisoformat(cfg.start_iso)
                ),
                end=(
                    None
                    if not cfg.end_iso
                    else __import__("datetime").datetime.fromisoformat(cfg.end_iso)
                ),
                chunk_days=int(cfg.chunk_days or 0),
                emit_dataset_events=cfg.emit_dataset_events,
                register_features=cfg.register_features,
                feature_registry_dir=(
                    None if cfg.feature_registry_dir is None else Path(cfg.feature_registry_dir)
                ),
                feature_role=cfg.feature_role,
                market_dataset_id=cfg.market_dataset_id,
                auto_refresh_macro=cfg.auto_refresh_macro,
                macro_staleness_hours=cfg.macro_staleness_hours,
                macro_series_ids=cfg.macro_series_ids,
                macro_fred_path=(
                    Path(cfg.macro_fred_path).expanduser() if cfg.macro_fred_path else None
                ),
                validation=cfg.validation,
                vintage_policy=cfg.vintage_policy,
                vintage_as_of=vintage_as_of_dt,
            )

            logger.info(
                "Dataset readiness | macro=%s events=%s student_mode=%s vintages=%s events_dir=%s",
                cfg.include_macro,
                cfg.include_events,
                getattr(cfg, "student_mode", False),
                bool(cfg.fred_vintage_dir),
                bool(cfg.events_dir),
            )

            result = api_build(
                api_cfg,
                data_store=self.data_store,
            )

            if not result.feature_names:
                row_count = self._infer_dataset_row_count(result)
                metadata = getattr(result, "metadata", None)
                dataset_empty = row_count == 0
                if not dataset_empty and metadata is not None:
                    overall_window = getattr(metadata, "overall_window", None)
                    ts_start = getattr(metadata, "ts_event_start", None)
                    ts_end = getattr(metadata, "ts_event_end", None)
                    dataset_empty = overall_window is None and ts_start is None and ts_end is None
                if dataset_empty:
                    raise _EmptyDatasetError(
                        "Dataset build via API returned zero rows",
                        row_count=row_count,
                    )
                raise ValueError("API dataset build returned no features; falling back to CLI")

            manifest_id = self._export_feature_manifest(cfg, result)

            # Persist feature registration metadata for HPO
            try:
                meta_path = Path(cfg.out_dir) / "feature_registration.json"
                payload = {
                    "feature_set_id": result.feature_set_id,
                    "feature_registry_dir": cfg.feature_registry_dir,
                    "feature_role": cfg.feature_role,
                }
                if manifest_id:
                    payload["manifest_id"] = manifest_id
                meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            except Exception as exc:
                logger.debug(
                    "Unable to persist feature registration metadata",
                    exc_info=True,
                    extra={
                        "dataset_id": cfg.dataset_id,
                        "out_dir": str(cfg.out_dir),
                        "reason": str(exc),
                    },
                )

            dataset_metadata = getattr(result, "metadata", None)
            if dataset_metadata is not None:
                try:
                    self._guard_dataset_metadata(cfg=cfg, metadata=dataset_metadata)
                except Exception as exc:
                    raise ValueError(f"Dataset metadata guardrail violation: {exc}") from exc
                self._synchronize_dataset_manifest(cfg=cfg, metadata=dataset_metadata)
                try:
                    logger.info(
                        "Dataset metadata recorded | vintage_policy=%s vintage_cutoff=%s train_window=%s validation_window=%s",
                        dataset_metadata.vintage_policy.value,
                        dataset_metadata.vintage_cutoff,
                        dataset_metadata.train_window,
                        dataset_metadata.validation_window,
                    )
                except Exception:  # pragma: no cover - defensive logging
                    logger.debug("Failed to log dataset metadata", exc_info=True)

            converted_path = self._maybe_convert_vintage_dataset(cfg, result.dataset_parquet)
            if converted_path != result.dataset_parquet:
                result = dataclasses.replace(result, dataset_parquet=converted_path)

            self._record_build_artifacts(
                cfg=cfg,
                feature_set_id=getattr(result, "feature_set_id", None),
                feature_names=list(getattr(result, "feature_names", [])),
                feature_registry_dir=cfg.feature_registry_dir,
                dataset_metadata=dataset_metadata,
            )
            return 0

        except _EmptyDatasetError as empty_exc:
            row_info = f" rows={empty_exc.row_count}" if empty_exc.row_count is not None else ""
            logger.error(
                "Dataset build produced no rows%s; extend the build window or ensure catalog coverage before rerunning.",
                row_info,
                extra={
                    "dataset_id": cfg.dataset_id,
                    "symbols": cfg.symbols,
                    "start_iso": cfg.start_iso,
                    "end_iso": cfg.end_iso,
                },
            )
            return 1
        except Exception as exc:  # pragma: no cover - defensive fallback to CLI path
            logger.warning("API dataset build failed; falling back to CLI: %s", exc, exc_info=True)

        # Fallback to CLI build
        return self._build_via_cli(cfg)

    def validate_dataset(
        self,
        dataset_path: Path,
        expectations: DatasetMetadataExpectations,
        validation_config: DatasetValidationConfig,
    ) -> tuple[bool, DatasetMetadata | None]:
        """
        Validate dataset against expectations.

        Parameters
        ----------
        dataset_path : Path
            Path to dataset file
        expectations : DatasetMetadataExpectations
            Expected dataset metadata
        validation_config : DatasetValidationConfig
            Validation configuration

        Returns
        -------
        tuple[bool, DatasetMetadata | None]
            (validation_passed, dataset_metadata)

        """
        metadata_path = dataset_path.parent / "dataset_metadata.json"
        if not metadata_path.exists():
            logger.warning("Dataset metadata not found at %s", metadata_path)
            return False, None

        try:
            metadata = load_dataset_metadata(metadata_path)
            validate_dataset_metadata_expectations(
                metadata,
                expectations,
                context="dataset_builder.validate",
            )
            if validation_config.expected_vintage_policy is not None:
                if metadata.vintage_policy != validation_config.expected_vintage_policy:
                    raise ValueError(
                        f"Expected vintage policy {validation_config.expected_vintage_policy.value} "
                        f"but dataset uses {metadata.vintage_policy.value}",
                    )
            if validation_config.require_macro_series:
                required_series = tuple(str(series) for series in validation_config.require_macro_series)
                min_observations = validation_config.macro_min_vintage_observations or 1
                missing_macro = [
                    series
                    for series in required_series
                    if metadata.macro_observation_counts.get(series, 0) < min_observations
                ]
                if missing_macro:
                    raise ValueError(
                        "Missing macro series observations: "
                        f"{', '.join(sorted(missing_macro))} "
                        f"(threshold={min_observations})",
                    )
            return True, metadata
        except Exception as exc:
            logger.error("Dataset validation failed: %s", exc, exc_info=True)
            return False, None

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _build_via_cli(self, cfg: DatasetBuildConfig) -> int:
        """
        Build dataset via CLI fallback.

        Parameters
        ----------
        cfg : DatasetBuildConfig
            Dataset build configuration

        Returns
        -------
        int
            Exit code

        """
        if self.build_main is None:
            raise RuntimeError("CLI build_main not configured")

        args: list[str] = [
            "--data_dir",
            cfg.data_dir,
            "--symbols",
            cfg.symbols,
            "--out_dir",
            cfg.out_dir,
            "--horizon_minutes",
            str(cfg.horizon_minutes),
            "--threshold",
            str(cfg.threshold),
            "--lookback_periods",
            str(cfg.lookback_periods),
        ]

        if cfg.include_macro:
            args += ["--include_macro", "--macro_lag_days", str(cfg.macro_lag_days)]
        if cfg.include_micro:
            args += ["--include_micro"]
        if cfg.include_l2:
            args += ["--include_l2"]
        if getattr(cfg, "include_events", False):
            args += ["--include_events"]
        if getattr(cfg, "include_calendar", False):
            args += ["--include_calendar"]
        if cfg.fred_vintage_dir:
            args += ["--fred_vintage_dir", cfg.fred_vintage_dir]
        if cfg.events_dir:
            args += ["--events_dir", cfg.events_dir]
        if cfg.student_mode:
            args += ["--student_mode"]
        if cfg.market_dataset_id:
            args += ["--market_dataset_id", cfg.market_dataset_id]

        if cfg.market_inputs:
            inputs_payload: list[object] = []
            for item in cfg.market_inputs:
                entry: dict[str, object] = {}
                if item.descriptor_id is not None:
                    entry["descriptor_id"] = item.descriptor_id
                if item.dataset_id is not None:
                    entry["dataset_id"] = item.dataset_id
                if item.symbols is not None:
                    entry["symbols"] = list(item.symbols)
                if item.schema_override is not None:
                    entry["schema"] = item.schema_override
                if item.storage_kind_override is not None:
                    entry["storage_kind"] = item.storage_kind_override.value
                if item.start is not None:
                    entry["start"] = item.start
                if item.end is not None:
                    entry["end"] = item.end
                inputs_payload.append(entry or (item.descriptor_id or item.dataset_id or ""))
            args += ["--market_inputs_json", json.dumps(inputs_payload)]

        if not cfg.auto_refresh_macro:
            args += ["--skip_macro_refresh"]
        if cfg.macro_staleness_hours != 24:
            args += ["--macro_freshness_hours", str(cfg.macro_staleness_hours)]
        if cfg.macro_series_ids:
            args += ["--macro_series_ids", ",".join(cfg.macro_series_ids)]
        if cfg.macro_fred_path:
            args += ["--macro_fred_path", cfg.macro_fred_path]
        if cfg.vintage_policy:
            args += ["--vintage_policy", cfg.vintage_policy.value]
        if cfg.vintage_as_of:
            args += ["--vintage_as_of", cfg.vintage_as_of]

        if cfg.validation is not None:
            args += ["--validation_min_rows", str(cfg.validation.min_rows)]
            if cfg.validation.min_positive_rate is not None:
                args += ["--validation_min_positive_rate", str(cfg.validation.min_positive_rate)]
            if cfg.validation.max_positive_rate is not None:
                args += ["--validation_max_positive_rate", str(cfg.validation.max_positive_rate)]
            if cfg.validation.min_feature_coverage is not None:
                args += [
                    "--validation_min_feature_coverage",
                    str(cfg.validation.min_feature_coverage),
                ]

        if getattr(cfg, "start_iso", None):
            args += ["--start", str(cfg.start_iso)]
        if getattr(cfg, "end_iso", None):
            args += ["--end", str(cfg.end_iso)]
        if int(getattr(cfg, "chunk_days", 0) or 0) > 0:
            args += ["--chunk_days", str(int(cfg.chunk_days))]
        if cfg.emit_dataset_events:
            args += ["--emit_dataset_events"]
        if cfg.register_features:
            args += ["--register_features"]
            reg_dir = cfg.feature_registry_dir or str(Path.home() / ".nautilus" / "ml" / "features")
            args += ["--feature_registry_dir", reg_dir]

        if cfg.convert_vintage_to_age:
            args += ["--convert-vintage-age"]

        rc = self.build_main(args)
        if rc == 0:
            dataset_path = Path(cfg.out_dir) / "dataset.parquet"
            self._maybe_convert_vintage_dataset(cfg, dataset_path)
            self._capture_cli_build_artifacts(cfg)
        return rc

    @staticmethod
    def _infer_dataset_row_count(result: object) -> int | None:
        """
        Best-effort row count inference for API build results.

        Parameters
        ----------
        result : object
            API build result

        Returns
        -------
        int | None
            Inferred row count or None

        """
        metadata = getattr(result, "metadata", None)
        if metadata is not None:
            overall_window = getattr(metadata, "overall_window", None)
            ts_start = getattr(metadata, "ts_event_start", None)
            ts_end = getattr(metadata, "ts_event_end", None)
            if overall_window is None and ts_start is None and ts_end is None:
                return 0

        dataset_parquet = getattr(result, "dataset_parquet", None)
        if isinstance(dataset_parquet, Path) and dataset_parquet.exists():
            try:
                import pyarrow.parquet as pq
            except ModuleNotFoundError:  # pragma: no cover - optional dependency missing
                logger.debug(
                    "pyarrow unavailable for row count inference",
                    extra={"dataset_parquet": str(dataset_parquet)},
                )
            else:
                try:
                    return int(pq.ParquetFile(str(dataset_parquet)).metadata.num_rows)
                except Exception:  # pragma: no cover - defensive best effort
                    logger.debug(
                        "Unable to infer row count from dataset parquet",
                        exc_info=True,
                        extra={"dataset_parquet": str(dataset_parquet)},
                    )

        dataset_csv = getattr(result, "dataset_csv", None)
        if isinstance(dataset_csv, Path) and dataset_csv.exists():
            try:
                with dataset_csv.open(encoding="utf-8") as handle:
                    next(handle, None)  # header (if any)
                    has_data = next(handle, None)
                return 0 if has_data is None else None
            except Exception:  # pragma: no cover - defensive best effort
                logger.debug(
                    "Unable to infer row count from dataset CSV",
                    exc_info=True,
                    extra={"dataset_csv": str(dataset_csv)},
                )

        return None

    @staticmethod
    def _export_feature_manifest(
        cfg: DatasetBuildConfig,
        result: object,
    ) -> str | None:
        """
        Export a feature manifest when registry configuration is provided.

        Parameters
        ----------
        cfg : DatasetBuildConfig
            Dataset build configuration
        result : object
            API build result

        Returns
        -------
        str | None
            Manifest ID or None

        """
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
        except Exception as exc:  # pragma: no cover - import guard
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
            "include_calendar": cfg.include_calendar,
            "include_events": cfg.include_events,
            "include_earnings": cfg.include_earnings,
            "include_micro": cfg.include_micro,
            "include_l2": cfg.include_l2,
            "include_macro_revisions": cfg.include_macro_revisions,
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
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Feature manifest export failed: %s", exc, exc_info=True)
            return None

    def _record_build_artifacts(
        self,
        *,
        cfg: DatasetBuildConfig,
        feature_set_id: str | None,
        feature_names: Sequence[str] | None,
        feature_registry_dir: str | None,
        dataset_metadata: DatasetMetadata | None = None,
    ) -> None:
        """
        Record build artifacts for downstream stages.

        Parameters
        ----------
        cfg : DatasetBuildConfig
            Dataset configuration
        feature_set_id : str | None
            Feature set identifier
        feature_names : Sequence[str] | None
            Feature names
        feature_registry_dir : str | None
            Feature registry directory
        dataset_metadata : DatasetMetadata | None
            Dataset metadata

        """
        names_tuple = tuple(feature_names or [])
        self._build_artifacts = BuildArtifacts(
            out_dir=Path(cfg.out_dir),
            feature_registry_dir=feature_registry_dir,
            feature_set_id=feature_set_id,
            feature_names=names_tuple,
            dataset_metadata=dataset_metadata,
        )

    def _maybe_convert_vintage_dataset(
        self,
        cfg: DatasetBuildConfig,
        dataset_parquet: Path,
    ) -> Path:
        if not getattr(cfg, "convert_vintage_to_age", False):
            return dataset_parquet

        destination = dataset_parquet.with_name("dataset_with_vintage_age.parquet")
        metadata_path = dataset_parquet.parent / "dataset_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(metadata_path)

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        column_info = metadata.get("column_info", {})
        vintage_columns = tuple(
            str(name)
            for name in column_info.get("vintage_timestamp_columns", [])
            if isinstance(name, str)
        )

        if destination.exists():
            if not vintage_columns:
                return destination
            age_columns = tuple(
                name.replace("__value_vintage_ts", "__vintage_age_minutes")
                for name in vintage_columns
            )
            updated_metadata = update_metadata_with_vintage_age(
                metadata,
                vintage_columns=vintage_columns,
                age_columns=age_columns,
            )
            write_metadata(metadata_path, updated_metadata)
            logger.info(
                "Dataset vintage timestamps already converted; metadata refreshed",
                extra={
                    "dataset_id": cfg.dataset_id,
                    "destination": str(destination),
                },
            )
            return destination

        try:
            conversion = convert_vintage_timestamps_to_age(dataset_parquet, destination)
        except Exception:  # pragma: no cover - defensive
            logger.error(
                "Vintage age conversion failed",
                exc_info=True,
                extra={
                    "dataset_id": cfg.dataset_id,
                    "dataset_parquet": str(dataset_parquet),
                },
            )
            raise

        updated_metadata = update_metadata_with_vintage_age(
            metadata,
            vintage_columns=conversion.vintage_columns,
            age_columns=conversion.age_columns,
        )
        write_metadata(metadata_path, updated_metadata)
        logger.info(
            "Dataset vintage timestamps converted to age features",
            extra={
                "dataset_id": cfg.dataset_id,
                "destination": str(destination),
                "age_columns": list(conversion.age_columns),
            },
        )
        return destination

    def _guard_dataset_metadata(
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
            Dataset configuration
        metadata : DatasetMetadata
            Dataset metadata

        Raises
        ------
        ValueError
            If validation fails

        """

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

    @staticmethod
    def _compute_dataset_pipeline_signature(
        cfg: DatasetBuildConfig,
        metadata: DatasetMetadata,
    ) -> str:
        """
        Derive a stable pipeline signature covering vintage policy and scope.

        Parameters
        ----------
        cfg : DatasetBuildConfig
            Dataset configuration
        metadata : DatasetMetadata
            Dataset metadata

        Returns
        -------
        str
            Pipeline signature

        """
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

    def _synchronize_dataset_manifest(
        self,
        *,
        cfg: DatasetBuildConfig,
        metadata: DatasetMetadata,
    ) -> None:
        """
        Synchronize dataset manifest with metadata.

        Parameters
        ----------
        cfg : DatasetBuildConfig
            Dataset configuration
        metadata : DatasetMetadata
            Dataset metadata

        """
        registry_obj = self.data_registry
        if registry_obj is None or not cfg.dataset_id:
            return
        registry = registry_obj

        try:
            manifest = registry.get_manifest(cfg.dataset_id)
        except Exception:
            logger.debug(
                "Data registry manifest missing for dataset_id=%s; skipping metadata sync",
                cfg.dataset_id,
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
        except Exception as exc:  # pragma: no cover - registry backend failures
            logger.debug(
                "Failed to update dataset manifest metadata: %s",
                exc,
                exc_info=True,
            )

    @staticmethod
    def _infer_feature_names(out_dir: Path) -> tuple[str, ...]:
        """
        Infer feature names from dataset parquet file.

        Parameters
        ----------
        out_dir : Path
            Output directory

        Returns
        -------
        tuple[str, ...]
            Feature names

        """
        dataset_path = out_dir / "dataset_with_vintage_age.parquet"
        if not dataset_path.exists():
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
        except Exception as exc:  # pragma: no cover - defensive import guard
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
        except Exception as exc:  # pragma: no cover - io errors
            logger.warning("Failed to inspect dataset parquet: %s", exc)
            return ()

        try:
            check_ml_dependencies(["polars"])
        except Exception as exc:
            logger.debug(
                "Optional dependency check failed",
                exc_info=True,
                extra={"dependency": "polars", "reason": str(exc)},
            )
        return ()

    def _capture_cli_build_artifacts(self, cfg: DatasetBuildConfig) -> None:
        """
        Capture build artifacts from CLI build.

        Parameters
        ----------
        cfg : DatasetBuildConfig
            Dataset configuration

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
            self._synchronize_dataset_manifest(cfg=cfg, metadata=dataset_metadata)

        self._record_build_artifacts(
            cfg=cfg,
            feature_set_id=feature_set_id,
            feature_names=feature_names,
            feature_registry_dir=feature_registry_dir,
            dataset_metadata=dataset_metadata,
        )

    @property
    def build_artifacts(self) -> BuildArtifacts | None:
        """
        Get build artifacts from last build.

        Returns
        -------
        BuildArtifacts | None
            Build artifacts or None

        """
        return self._build_artifacts
