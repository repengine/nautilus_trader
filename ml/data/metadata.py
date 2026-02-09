"""
Dataset metadata helpers for TFT dataset builds.

Centralizes metadata structures, serialization, and expectation validation
used by dataset builders and orchestration flows.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np

from ml.common.reproducibility import ReproducibilityValue
from ml.common.reproducibility import validate_reproducibility_provenance
from ml.config.targets import EXECUTION_UNRESOLVED_CONTEXT_FAIL
from ml.config.targets import EXECUTION_UNRESOLVED_CONTEXT_ZERO_RETURN
from ml.config.targets import HORIZON_RESOLUTION_BAR_INDEX
from ml.config.targets import HORIZON_RESOLUTION_WALL_CLOCK
from ml.config.targets import TARGET_SEMANTICS_CONTRACT_ID
from ml.config.targets import TARGET_SEMANTICS_CONTRACT_MAJOR
from ml.config.targets import TARGET_SEMANTICS_EPOCH_VERSION
from ml.data.ingest.market_bindings import MarketBindingStats
from ml.data.vintage import VintagePolicy
from ml.data.vintage import format_dt
from ml.data.vintage import parse_dt


class DatasetBuildConfigProtocol(Protocol):
    """
    Minimal dataset build configuration protocol for metadata expectations.
    """

    dataset_id: str
    vintage_policy: VintagePolicy
    vintage_as_of: datetime | None
    start: datetime | None
    end: datetime | None


@dataclass(frozen=True)
class MarketBindingMetadata:
    """Metadata describing a resolved market binding for a dataset build."""

    binding_id: str
    dataset_id: str
    descriptor_id: str | None
    schema: str | None
    storage_kind: str | None
    symbols: tuple[str, ...]
    instrument_ids: tuple[str, ...]
    source: str
    license_start: str | None
    license_end: str | None
    ts_event_start: str | None
    ts_event_end: str | None
    rows_from_store: int
    rows_from_catalog: int
    source_datasets: tuple[str, ...] | None = None
    provider_dataset_id: str | None = None
    provider_schema: str | None = None


@dataclass(frozen=True)
class DatasetMetadata:
    """Metadata describing dataset build windows and capabilities."""

    dataset_id: str | None
    vintage_policy: VintagePolicy
    vintage_cutoff: str | None
    build_ts: str
    ts_event_start: str | None
    ts_event_end: str | None
    overall_window: tuple[str, str] | None
    train_window: tuple[str, str] | None
    validation_window: tuple[str, str] | None
    test_window: tuple[str, str] | None
    macro_observation_counts: dict[str, int]
    capability_flags: dict[str, bool] = field(default_factory=dict)
    market_bindings: tuple[MarketBindingMetadata, ...] | None = None
    target_semantics: dict[str, Any] | None = None
    reproducibility: dict[str, ReproducibilityValue] | None = None


@dataclass(frozen=True)
class DatasetMetadataExpectations:
    """Metadata expectations used to validate dataset artifacts."""

    dataset_id: str | None = None
    vintage_policy: VintagePolicy | None = None
    vintage_cutoff: str | None = None
    ts_event_start: str | None = None
    ts_event_end: str | None = None


def build_metadata_expectations(
    cfg: DatasetBuildConfigProtocol,
) -> DatasetMetadataExpectations:
    """
    Create metadata expectations derived from the dataset build configuration.

    Args:
        cfg: Dataset build configuration.

    Returns:
        DatasetMetadataExpectations instance.
    """
    # Support both legacy datetime attributes (`start`/`end`) and the newer ISO
    # string fields (`start_iso`/`end_iso`).
    start_raw = getattr(cfg, "start_iso", None)
    end_raw = getattr(cfg, "end_iso", None)
    if not start_raw and hasattr(cfg, "start"):
        start_raw = getattr(cfg, "start")
    if not end_raw and hasattr(cfg, "end"):
        end_raw = getattr(cfg, "end")

    def _normalize(value: object | None) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if isinstance(value, datetime):
            return format_dt(value)
        return str(value)

    start_iso = _normalize(start_raw)
    end_iso = _normalize(end_raw)
    cutoff_iso = _normalize(getattr(cfg, "vintage_as_of", None))
    return DatasetMetadataExpectations(
        dataset_id=cfg.dataset_id,
        vintage_policy=cfg.vintage_policy,
        vintage_cutoff=cutoff_iso,
        ts_event_start=start_iso,
        ts_event_end=end_iso,
    )


def _ns_to_iso(value: int | None) -> str | None:
    if value is None:
        return None
    dt_value = datetime.fromtimestamp(value / 1_000_000_000, tz=UTC)
    return format_dt(dt_value)


def _binding_stats_to_metadata(
    stats: Sequence[MarketBindingStats],
) -> tuple[MarketBindingMetadata, ...]:
    entries: list[MarketBindingMetadata] = []
    for stat in stats:
        storage_kind_value = stat.storage_kind.value if stat.storage_kind else None
        entries.append(
            MarketBindingMetadata(
                binding_id=stat.binding_id,
                dataset_id=stat.dataset_id,
                descriptor_id=stat.descriptor_id,
                schema=stat.schema,
                storage_kind=storage_kind_value,
                symbols=(stat.symbol,),
                instrument_ids=stat.instrument_ids,
                source=stat.source,
                license_start=stat.license_start,
                license_end=stat.license_end,
                ts_event_start=_ns_to_iso(stat.ts_event_start_ns),
                ts_event_end=_ns_to_iso(stat.ts_event_end_ns),
                rows_from_store=stat.rows_from_store,
                rows_from_catalog=stat.rows_from_catalog,
                source_datasets=tuple(sorted(stat.source_datasets)) if stat.source_datasets else None,
                provider_dataset_id=stat.provider_dataset_id,
                provider_schema=stat.provider_schema,
            ),
        )
    return tuple(entries)


def load_dataset_metadata(path: Path) -> DatasetMetadata:
    """
    Load dataset metadata from a JSON file.

    Args:
        path: Metadata JSON path.

    Returns:
        DatasetMetadata instance.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    try:
        policy_raw = raw.get("vintage_policy", VintagePolicy.REAL_TIME.value)
        if not policy_raw:
            policy_token = VintagePolicy.REAL_TIME.value
        elif isinstance(policy_raw, VintagePolicy):
            policy_token = policy_raw.value
        else:
            policy_token = str(policy_raw).strip().lower()
        policy = VintagePolicy(policy_token)
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"Invalid vintage_policy in metadata: {raw.get('vintage_policy')}") from exc

    def _as_tuple(value: object | None) -> tuple[str, str] | None:
        if not value:
            return None
        if isinstance(value, list | tuple) and len(value) == 2:
            return (str(value[0]), str(value[1]))
        raise ValueError(f"Metadata window must be length-2 sequence, got {value!r}")

    macro_counts_raw = raw.get("macro_observation_counts") or {}
    macro_counts: dict[str, int] = {
        str(key): int(value)
        for key, value in macro_counts_raw.items()
    }

    def _normalize_bool(value: object) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        if isinstance(value, int):
            return bool(value)
        if isinstance(value, str):
            token = value.strip().lower()
            if token in {"1", "true", "yes", "y", "t", "on"}:
                return True
            if token in {"0", "false", "no", "n", "f", "off", ""}:
                return False
        return bool(value)

    capability_raw = raw.get("capability_flags") or {}
    capability_flags: dict[str, bool] = {}
    if isinstance(capability_raw, dict):
        for key, value in capability_raw.items():
            capability_flags[str(key)] = _normalize_bool(value)

    target_semantics_raw = raw.get("target_semantics")
    target_semantics: dict[str, Any] | None = (
        target_semantics_raw if isinstance(target_semantics_raw, dict) else None
    )

    reproducibility_raw = raw.get("reproducibility")
    reproducibility: dict[str, ReproducibilityValue] | None = None
    if reproducibility_raw is not None:
        if not isinstance(reproducibility_raw, Mapping):
            raise ValueError("dataset metadata reproducibility must be a mapping")
        reproducibility = validate_reproducibility_provenance(
            payload=cast(Mapping[str, object], reproducibility_raw),
            context="dataset metadata reproducibility",
        )

    bindings_raw = raw.get("market_bindings")
    market_bindings: tuple[MarketBindingMetadata, ...] | None = None
    if isinstance(bindings_raw, list):
        converted: list[MarketBindingMetadata] = []
        for entry in bindings_raw:
            if not isinstance(entry, dict):
                continue
            symbols_field = entry.get("symbols")
            instrument_field = entry.get("instrument_ids")
            symbols_tuple = (
                tuple(str(item) for item in symbols_field)
                if isinstance(symbols_field, list | tuple)
                else ()
            )
            instruments_tuple = (
                tuple(str(item) for item in instrument_field)
                if isinstance(instrument_field, list | tuple)
                else ()
            )
            converted.append(
                MarketBindingMetadata(
                    binding_id=str(entry.get("binding_id")),
                    dataset_id=str(entry.get("dataset_id")),
                    descriptor_id=(
                        str(entry.get("descriptor_id")) if entry.get("descriptor_id") is not None else None
                    ),
                    schema=str(entry.get("schema")) if entry.get("schema") is not None else None,
                    storage_kind=(
                        str(entry.get("storage_kind")) if entry.get("storage_kind") is not None else None
                    ),
                    symbols=symbols_tuple,
                    instrument_ids=instruments_tuple,
                    source=str(entry.get("source", "")),
                    license_start=(
                        str(entry.get("license_start")) if entry.get("license_start") is not None else None
                    ),
                    license_end=(
                        str(entry.get("license_end")) if entry.get("license_end") is not None else None
                    ),
                    ts_event_start=(
                        str(entry.get("ts_event_start")) if entry.get("ts_event_start") is not None else None
                    ),
                    ts_event_end=(
                        str(entry.get("ts_event_end")) if entry.get("ts_event_end") is not None else None
                    ),
                    rows_from_store=int(entry.get("rows_from_store", 0) or 0),
                    rows_from_catalog=int(entry.get("rows_from_catalog", 0) or 0),
                    provider_dataset_id=(
                        str(entry.get("provider_dataset_id"))
                        if entry.get("provider_dataset_id") is not None
                        else None
                    ),
                    provider_schema=(
                        str(entry.get("provider_schema"))
                        if entry.get("provider_schema") is not None
                        else None
                    ),
                ),
            )
        market_bindings = tuple(converted)

    return DatasetMetadata(
        dataset_id=str(raw.get("dataset_id")) if raw.get("dataset_id") else None,
        vintage_policy=policy,
        vintage_cutoff=str(raw.get("vintage_cutoff")) if raw.get("vintage_cutoff") else None,
        build_ts=str(raw.get("build_ts", "")),
        ts_event_start=str(raw.get("ts_event_start")) if raw.get("ts_event_start") else None,
        ts_event_end=str(raw.get("ts_event_end")) if raw.get("ts_event_end") else None,
        overall_window=_as_tuple(raw.get("overall_window")),
        train_window=_as_tuple(raw.get("train_window")),
        validation_window=_as_tuple(raw.get("validation_window")),
        test_window=_as_tuple(raw.get("test_window")),
        macro_observation_counts=macro_counts,
        capability_flags=capability_flags,
        market_bindings=market_bindings,
        target_semantics=target_semantics,
        reproducibility=reproducibility,
    )


def validate_dataset_metadata_expectations(
    metadata: DatasetMetadata,
    expectations: DatasetMetadataExpectations,
    *,
    context: str | None = None,
) -> None:
    """
    Validate that metadata satisfies the supplied expectations.

    Args:
        metadata: Dataset metadata to validate.
        expectations: Expected metadata boundaries.
        context: Optional context prefix for errors.
    """
    prefix = f"{context}: " if context else ""

    if expectations.dataset_id and metadata.dataset_id != expectations.dataset_id:
        raise ValueError(
            f"{prefix}dataset_id mismatch (expected {expectations.dataset_id}, got {metadata.dataset_id})",
        )

    if expectations.vintage_policy and metadata.vintage_policy is not expectations.vintage_policy:
        raise ValueError(
            f"{prefix}vintage_policy mismatch (expected {expectations.vintage_policy.value}, got {metadata.vintage_policy.value})",
        )

    if expectations.vintage_cutoff is not None:
        expected_cutoff = expectations.vintage_cutoff
        actual_cutoff = metadata.vintage_cutoff or ""
        if actual_cutoff != expected_cutoff:
            raise ValueError(
                f"{prefix}vintage_cutoff mismatch (expected {expected_cutoff}, got {actual_cutoff or 'None'})",
            )

    def _ensure_bounds(
        label: str,
        expected: str | None,
        actual: str | None,
        *,
        comparator: str,
    ) -> None:
        if not expected or not actual:
            return
        expected_dt = parse_dt(expected)
        actual_dt = parse_dt(actual)
        if expected_dt is None or actual_dt is None:
            return
        if comparator == "gte" and actual_dt < expected_dt:
            raise ValueError(
                f"{prefix}{label} {actual} earlier than expected {expected}",
            )
        if comparator == "lte" and actual_dt > expected_dt:
            raise ValueError(
                f"{prefix}{label} {actual} later than expected {expected}",
            )

    _ensure_bounds("ts_event_start", expectations.ts_event_start, metadata.ts_event_start, comparator="gte")
    _ensure_bounds("ts_event_end", expectations.ts_event_end, metadata.ts_event_end, comparator="lte")


def require_reproducibility_metadata(
    metadata: DatasetMetadata,
    *,
    context: str | None = None,
) -> dict[str, ReproducibilityValue]:
    """
    Ensure dataset metadata includes canonical reproducibility provenance payload.

    Args:
        metadata: Dataset metadata to validate.
        context: Optional context prefix for errors.

    Returns:
        Validated reproducibility payload.
    """
    prefix = f"{context}: " if context else ""
    reproducibility = metadata.reproducibility
    if not isinstance(reproducibility, Mapping):
        raise ValueError(f"{prefix}dataset metadata missing reproducibility payload")

    validation_context = (
        f"{context}.dataset_metadata.reproducibility"
        if context
        else "dataset_metadata.reproducibility"
    )
    return validate_reproducibility_provenance(
        payload=cast(Mapping[str, object], reproducibility),
        context=validation_context,
    )


def _normalize_target_semantics_execution_payload(
    execution_payload: Mapping[str, object],
    *,
    context: str,
) -> dict[str, Any]:
    """
    Validate and normalize target semantics execution metadata payload.
    """
    entry_price_column_raw = execution_payload.get("entry_price_column")
    if not isinstance(entry_price_column_raw, str) or not entry_price_column_raw.strip():
        raise ValueError(
            f"{context}target semantics execution.entry_price_column must be non-empty string",
        )
    entry_price_column = entry_price_column_raw.strip()

    exit_price_column_raw = execution_payload.get("exit_price_column")
    if not isinstance(exit_price_column_raw, str) or not exit_price_column_raw.strip():
        raise ValueError(
            f"{context}target semantics execution.exit_price_column must be non-empty string",
        )
    exit_price_column = exit_price_column_raw.strip()

    latency_bars_raw = execution_payload.get("latency_bars")
    if not isinstance(latency_bars_raw, int):
        raise ValueError(f"{context}target semantics execution.latency_bars must be an int")
    if latency_bars_raw < 0:
        raise ValueError(f"{context}target semantics execution.latency_bars must be >= 0")

    latency_unit = execution_payload.get("latency_unit")
    if latency_unit != "bars":
        raise ValueError(
            f"{context}target semantics execution.latency_unit must be 'bars', got {latency_unit!r}",
        )

    unresolved_mode_raw = execution_payload.get("unresolved_context_mode")
    if not isinstance(unresolved_mode_raw, str) or not unresolved_mode_raw.strip():
        raise ValueError(
            f"{context}target semantics execution.unresolved_context_mode must be non-empty string",
        )
    unresolved_mode = unresolved_mode_raw.strip().lower()
    if unresolved_mode not in (
        EXECUTION_UNRESOLVED_CONTEXT_ZERO_RETURN,
        EXECUTION_UNRESOLVED_CONTEXT_FAIL,
    ):
        raise ValueError(
            f"{context}target semantics execution.unresolved_context_mode must be one of "
            f"{(EXECUTION_UNRESOLVED_CONTEXT_ZERO_RETURN, EXECUTION_UNRESOLVED_CONTEXT_FAIL)}, "
            f"got {unresolved_mode_raw!r}",
        )

    normalized: dict[str, Any] = {
        "entry_price_column": entry_price_column,
        "exit_price_column": exit_price_column,
        "latency_bars": int(latency_bars_raw),
        "latency_unit": "bars",
        "unresolved_context_mode": unresolved_mode,
    }
    if unresolved_mode == EXECUTION_UNRESOLVED_CONTEXT_ZERO_RETURN:
        unresolved_return_raw = execution_payload.get("unresolved_context_return")
        if not isinstance(unresolved_return_raw, int | float):
            raise ValueError(
                f"{context}target semantics execution.unresolved_context_return must be numeric "
                "when unresolved_context_mode='zero_return'",
            )
        unresolved_return = float(unresolved_return_raw)
        if unresolved_return != 0.0:
            raise ValueError(
                f"{context}target semantics execution.unresolved_context_return must be 0.0 "
                "for deterministic zero-return fallback",
            )
        normalized["unresolved_context_return"] = 0.0

    return normalized


def require_target_semantics_metadata(
    metadata: DatasetMetadata,
    *,
    context: str | None = None,
) -> dict[str, Any]:
    """
    Ensure dataset metadata includes strict target semantics payload.

    Args:
        metadata: Dataset metadata to validate.
        context: Optional context prefix for errors.

    Returns:
        Target semantics metadata payload.

    Example:
        >>> payload = require_target_semantics_metadata(metadata, context="training")
        >>> assert "horizons" in payload
    """
    prefix = f"{context}: " if context else ""
    target_semantics = metadata.target_semantics
    if not isinstance(target_semantics, dict) or not target_semantics:
        raise ValueError(f"{prefix}dataset metadata missing target_semantics payload")

    required_keys = ("contract", "horizons", "labels", "returns", "execution")
    missing = [key for key in required_keys if key not in target_semantics]
    if missing:
        raise ValueError(
            f"{prefix}dataset metadata target_semantics missing required keys: {missing}",
        )

    horizon_resolution_mode = target_semantics.get("horizon_resolution_mode")
    if not isinstance(horizon_resolution_mode, str) or not horizon_resolution_mode:
        raise ValueError(
            f"{prefix}dataset metadata target_semantics horizon_resolution_mode must be non-empty string",
        )
    if horizon_resolution_mode not in (
        HORIZON_RESOLUTION_BAR_INDEX,
        HORIZON_RESOLUTION_WALL_CLOCK,
    ):
        raise ValueError(
            f"{prefix}dataset metadata target_semantics horizon_resolution_mode must be one of "
            f"{(HORIZON_RESOLUTION_BAR_INDEX, HORIZON_RESOLUTION_WALL_CLOCK)}, "
            f"got {horizon_resolution_mode!r}",
        )

    horizon_alignment = target_semantics.get("horizon_alignment")
    if not isinstance(horizon_alignment, Mapping):
        raise ValueError(
            f"{prefix}dataset metadata target_semantics horizon_alignment must be a mapping",
        )
    alignment_mode = horizon_alignment.get("mode")
    if alignment_mode != horizon_resolution_mode:
        raise ValueError(
            f"{prefix}dataset metadata target_semantics horizon_alignment.mode mismatch "
            f"(expected {horizon_resolution_mode!r}, got {alignment_mode!r})",
        )
    future_anchor = horizon_alignment.get("future_anchor")
    if not isinstance(future_anchor, str) or not future_anchor.strip():
        raise ValueError(
            f"{prefix}dataset metadata target_semantics horizon_alignment.future_anchor must be non-empty string",
        )
    insufficient_future_handling = horizon_alignment.get("insufficient_future_handling")
    if (
        not isinstance(insufficient_future_handling, str)
        or not insufficient_future_handling.strip()
    ):
        raise ValueError(
            f"{prefix}dataset metadata target_semantics horizon_alignment."
            "insufficient_future_handling must be non-empty string",
        )
    if horizon_resolution_mode == HORIZON_RESOLUTION_WALL_CLOCK:
        timestamp_column = horizon_alignment.get("timestamp_column")
        if not isinstance(timestamp_column, str) or not timestamp_column.strip():
            raise ValueError(
                f"{prefix}dataset metadata target_semantics horizon_alignment.timestamp_column must "
                "be non-empty string for wall_clock mode",
            )

    execution = target_semantics.get("execution")
    if not isinstance(execution, Mapping):
        raise ValueError(
            f"{prefix}dataset metadata target_semantics execution must be a mapping",
        )
    _normalize_target_semantics_execution_payload(
        cast(Mapping[str, object], execution),
        context=f"{prefix}dataset metadata ",
    )

    version = target_semantics.get("version")
    if version is not None and (
        not isinstance(version, str) or version != TARGET_SEMANTICS_EPOCH_VERSION
    ):
        raise ValueError(
            f"{prefix}dataset metadata target_semantics version must be "
            f"{TARGET_SEMANTICS_EPOCH_VERSION!r} when present",
        )

    contract = target_semantics.get("contract")
    if not isinstance(contract, Mapping):
        raise ValueError(f"{prefix}dataset metadata target_semantics contract must be a mapping")
    contract_id = contract.get("id")
    if not isinstance(contract_id, str) or not contract_id:
        raise ValueError(f"{prefix}dataset metadata target_semantics contract.id must be non-empty string")
    contract_major = contract.get("major")
    if not isinstance(contract_major, int) or contract_major < 1:
        raise ValueError(f"{prefix}dataset metadata target_semantics contract.major must be >= 1")
    capabilities = contract.get("capabilities")
    if not isinstance(capabilities, list) or not capabilities:
        raise ValueError(
            f"{prefix}dataset metadata target_semantics contract.capabilities must be non-empty list",
        )
    if any(not isinstance(cap, str) or not cap.strip() for cap in capabilities):
        raise ValueError(
            f"{prefix}dataset metadata target_semantics contract.capabilities must contain non-empty strings",
        )
    if len(set(capabilities)) != len(capabilities):
        raise ValueError(
            f"{prefix}dataset metadata target_semantics contract.capabilities must be unique",
        )

    horizons = target_semantics.get("horizons")
    if not isinstance(horizons, list) or not horizons:
        raise ValueError(f"{prefix}dataset metadata target_semantics horizons must be non-empty list")

    labels = target_semantics.get("labels")
    if not isinstance(labels, dict) or not labels:
        raise ValueError(f"{prefix}dataset metadata target_semantics labels must be non-empty mapping")

    returns = target_semantics.get("returns")
    if not isinstance(returns, dict) or not returns:
        raise ValueError(f"{prefix}dataset metadata target_semantics returns must be non-empty mapping")
    return target_semantics


def require_target_semantics_contract(
    metadata: DatasetMetadata,
    *,
    required_capabilities: Sequence[str],
    expected_contract_id: str = TARGET_SEMANTICS_CONTRACT_ID,
    expected_contract_major: int = TARGET_SEMANTICS_CONTRACT_MAJOR,
    context: str | None = None,
) -> dict[str, Any]:
    """
    Ensure dataset target semantics use the canonical contract and capabilities.

    Args:
        metadata: Dataset metadata to validate.
        required_capabilities: Capability names required by the caller.
        expected_contract_id: Expected contract identifier.
        expected_contract_major: Expected contract major version.
        context: Optional context prefix for errors.

    Returns:
        Canonical contract payload.
    """
    prefix = f"{context}: " if context else ""
    target_semantics = require_target_semantics_metadata(metadata, context=context)
    version = target_semantics.get("version")
    if version != TARGET_SEMANTICS_EPOCH_VERSION:
        raise ValueError(
            f"{prefix}target semantics version mismatch "
            f"(expected {TARGET_SEMANTICS_EPOCH_VERSION!r}, got {version!r})",
        )
    contract = target_semantics.get("contract")
    if not isinstance(contract, Mapping):
        raise ValueError(f"{prefix}dataset metadata target_semantics contract must be a mapping")

    contract_id = contract.get("id")
    if contract_id != expected_contract_id:
        raise ValueError(
            f"{prefix}target semantics contract.id mismatch "
            f"(expected {expected_contract_id!r}, got {contract_id!r})",
        )

    contract_major = contract.get("major")
    if contract_major != expected_contract_major:
        raise ValueError(
            f"{prefix}target semantics contract.major mismatch "
            f"(expected {expected_contract_major}, got {contract_major!r})",
        )

    capabilities_raw = contract.get("capabilities")
    capabilities = {
        str(cap).strip()
        for cap in capabilities_raw
        if isinstance(cap, str) and cap.strip()
    } if isinstance(capabilities_raw, list) else set()
    required = [str(cap).strip() for cap in required_capabilities if str(cap).strip()]
    missing = [cap for cap in required if cap not in capabilities]
    if missing:
        raise ValueError(
            f"{prefix}target semantics contract missing required capabilities: {missing}",
        )
    return dict(contract)


def require_target_semantics_horizon_mode(
    metadata: DatasetMetadata,
    *,
    expected_mode: str | None = None,
    context: str | None = None,
) -> str:
    """
    Resolve and optionally enforce target semantics horizon resolution mode.

    Args:
        metadata: Dataset metadata to validate.
        expected_mode:
            Optional expected mode (`bar_index` or `wall_clock`).
            When provided, mismatches raise ValueError.
        context: Optional context prefix for errors.

    Returns:
        Declared horizon resolution mode.
    """
    prefix = f"{context}: " if context else ""
    target_semantics = require_target_semantics_metadata(metadata, context=context)
    mode = target_semantics.get("horizon_resolution_mode")
    if not isinstance(mode, str):
        raise ValueError(
            f"{prefix}dataset metadata target_semantics horizon_resolution_mode must be a string",
        )

    if expected_mode is None:
        return mode

    normalized_expected = str(expected_mode).strip().lower()
    if normalized_expected not in (
        HORIZON_RESOLUTION_BAR_INDEX,
        HORIZON_RESOLUTION_WALL_CLOCK,
    ):
        raise ValueError(
            "expected_mode must be one of "
            f"{(HORIZON_RESOLUTION_BAR_INDEX, HORIZON_RESOLUTION_WALL_CLOCK)}, "
            f"got {expected_mode!r}",
        )
    if mode != normalized_expected:
        raise ValueError(
            f"{prefix}target semantics horizon_resolution_mode mismatch "
            f"(expected {normalized_expected!r}, got {mode!r})",
        )
    return mode


def require_target_semantics_execution_contract(
    metadata: DatasetMetadata,
    *,
    expected_execution: Mapping[str, object] | None = None,
    context: str | None = None,
) -> dict[str, Any]:
    """
    Resolve and optionally enforce target semantics execution contract payload.

    Args:
        metadata: Dataset metadata to validate.
        expected_execution:
            Optional expected execution contract payload. When provided,
            normalized execution fields must match exactly.
        context: Optional context prefix for errors.

    Returns:
        Normalized execution contract payload.
    """
    prefix = f"{context}: " if context else ""
    target_semantics = require_target_semantics_metadata(metadata, context=context)
    execution = target_semantics.get("execution")
    if not isinstance(execution, Mapping):
        raise ValueError(f"{prefix}dataset metadata target_semantics execution must be a mapping")

    normalized_execution = _normalize_target_semantics_execution_payload(
        cast(Mapping[str, object], execution),
        context=f"{prefix}",
    )
    if expected_execution is None:
        return normalized_execution

    normalized_expected = _normalize_target_semantics_execution_payload(
        expected_execution,
        context=f"{prefix}expected ",
    )
    comparison_keys = (
        "entry_price_column",
        "exit_price_column",
        "latency_bars",
        "latency_unit",
        "unresolved_context_mode",
    )
    for key in comparison_keys:
        if normalized_execution.get(key) != normalized_expected.get(key):
            raise ValueError(
                f"{prefix}target semantics execution.{key} mismatch "
                f"(expected {normalized_expected.get(key)!r}, "
                f"got {normalized_execution.get(key)!r})",
            )
    if normalized_execution["unresolved_context_mode"] == EXECUTION_UNRESOLVED_CONTEXT_ZERO_RETURN:
        if normalized_execution.get("unresolved_context_return") != normalized_expected.get(
            "unresolved_context_return",
        ):
            raise ValueError(
                f"{prefix}target semantics execution.unresolved_context_return mismatch "
                f"(expected {normalized_expected.get('unresolved_context_return')!r}, "
                f"got {normalized_execution.get('unresolved_context_return')!r})",
            )
    return normalized_execution


def require_target_column_in_semantics(
    metadata: DatasetMetadata,
    target_col: str,
    *,
    context: str | None = None,
) -> None:
    """
    Ensure target column exists in target semantics metadata.

    Args:
        metadata: Dataset metadata to validate.
        target_col: Target column name expected in dataset.
        context: Optional context prefix for errors.

    Returns:
        None. Raises ValueError on mismatch.

    Example:
        >>> require_target_column_in_semantics(metadata, "target_bin_15m", context="training")
    """
    prefix = f"{context}: " if context else ""
    target_col = str(target_col).strip()
    if not target_col:
        raise ValueError(f"{prefix}target_col must be non-empty string")

    target_semantics = require_target_semantics_metadata(metadata, context=context)
    labels = target_semantics.get("labels")
    if isinstance(labels, dict) and target_col in labels:
        return

    legacy_aliases = target_semantics.get("legacy_aliases")
    if isinstance(legacy_aliases, dict) and target_col in legacy_aliases:
        return

    raise ValueError(
        f"{prefix}target_col '{target_col}' not declared in target_semantics labels or legacy_aliases",
    )


def resolve_target_col_from_metadata(
    metadata_path: Path,
    *,
    context: str | None = None,
) -> str:
    """
    Resolve the target column name from dataset metadata.

    Args:
        metadata_path: Path to dataset_metadata.json.
        context: Optional context prefix for errors.

    Returns:
        Target column name declared in column_info.

    Example:
        >>> target_col = resolve_target_col_from_metadata(Path("dataset_metadata.json"), context="training")
        >>> assert target_col
    """
    prefix = f"{context}: " if context else ""
    payload = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    column_info = payload.get("column_info", {})
    if not isinstance(column_info, Mapping):
        raise ValueError(f"{prefix}dataset metadata column_info must be an object")
    target_col = str(column_info.get("target_col", "y")).strip()
    if not target_col:
        raise ValueError(f"{prefix}target_col must be non-empty")
    return target_col


def _ensure_datetime(value: datetime | float | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    if isinstance(value, int | float | np.generic):
        try:
            return datetime.fromtimestamp(float(value) / 1_000_000_000, tz=UTC)
        except (OSError, OverflowError, ValueError):  # pragma: no cover - defensive
            return None
    return None


def _format_window(start: datetime | float | None, end: datetime | float | None) -> tuple[str, str] | None:
    start_dt = _ensure_datetime(start)
    end_dt = _ensure_datetime(end)
    if start_dt is None or end_dt is None:
        return None
    start_iso = format_dt(start_dt)
    end_iso = format_dt(end_dt)
    if start_iso is None or end_iso is None:
        return None
    return (start_iso, end_iso)


def build_dataset_metadata_from_windows(
    *,
    dataset_id: str | None,
    vintage_policy: VintagePolicy,
    vintage_as_of: datetime | None,
    build_ts: datetime,
    overall_start: datetime | None,
    overall_end: datetime | None,
    train_window_end: datetime | None,
    validation_window_start: datetime | None,
    macro_observation_counts: dict[str, int] | None,
    capability_flags: dict[str, bool] | None = None,
    market_bindings: tuple[MarketBindingMetadata, ...] | None = None,
    target_semantics: dict[str, Any] | None = None,
    reproducibility: dict[str, ReproducibilityValue] | None = None,
) -> DatasetMetadata:
    """
    Build dataset metadata from precomputed window boundaries.

    Args:
        dataset_id: Dataset identifier.
        vintage_policy: Vintage policy for macro joins.
        vintage_as_of: Optional vintage cutoff timestamp.
        build_ts: Build timestamp.
        overall_start: Overall window start.
        overall_end: Overall window end.
        train_window_end: Training window end.
        validation_window_start: Validation window start.
        macro_observation_counts: Macro observation counts by series.
        capability_flags: Optional capability flags.
        market_bindings: Optional market binding metadata.
        target_semantics: Optional target semantics metadata payload.
        reproducibility: Optional reproducibility provenance payload.

    Returns:
        DatasetMetadata instance.
    """
    build_ts_iso = format_dt(build_ts) or build_ts.isoformat()
    vintage_cutoff = format_dt(vintage_as_of) if vintage_as_of else None

    return DatasetMetadata(
        dataset_id=dataset_id,
        vintage_policy=vintage_policy,
        vintage_cutoff=vintage_cutoff,
        build_ts=build_ts_iso,
        ts_event_start=format_dt(overall_start) if overall_start else None,
        ts_event_end=format_dt(overall_end) if overall_end else None,
        overall_window=_format_window(overall_start, overall_end),
        train_window=_format_window(overall_start, train_window_end),
        validation_window=_format_window(validation_window_start, overall_end),
        test_window=None,
        macro_observation_counts=dict(macro_observation_counts or {}),
        capability_flags=capability_flags or {},
        market_bindings=market_bindings,
        target_semantics=target_semantics,
        reproducibility=reproducibility,
    )


def _compute_dataset_metadata(
    df_pd_sorted: Any,
    cutoff: int,
    vintage_policy: VintagePolicy,
    vintage_as_of: datetime | None,
    build_ts: datetime,
    dataset_id: str | None,
    macro_observation_counts: dict[str, int] | None,
    target_semantics: dict[str, Any] | None,
    reproducibility: dict[str, ReproducibilityValue] | None = None,
) -> DatasetMetadata:
    """
    Compute dataset metadata from a sorted dataframe.

    Args:
        df_pd_sorted: Polars or pandas dataframe sorted by time.
        cutoff: Train/validation split index.
        vintage_policy: Vintage policy used for macro joins.
        vintage_as_of: Optional vintage cutoff timestamp.
        build_ts: Dataset build timestamp.
        dataset_id: Dataset identifier.
        macro_observation_counts: Macro observation counts by series.
        target_semantics: Target semantics metadata payload.
        reproducibility: Reproducibility provenance payload.

    Returns:
        DatasetMetadata instance.
    """
    from ml._imports import pd
    from ml._imports import pl

    overall_window = None
    train_window = None
    validation_window = None
    ts_start = None
    ts_end = None

    if pl is not None and isinstance(df_pd_sorted, pl.DataFrame):
        if "ts_event" in df_pd_sorted.columns:
            ts_col = "ts_event"
        elif "timestamp" in df_pd_sorted.columns:
            ts_col = "timestamp"
        else:
            ts_col = None

        if ts_col is not None and df_pd_sorted.height > 0:
            ts_series = df_pd_sorted.select(pl.col(ts_col)).to_series()
            start_dt_raw = ts_series[0]
            end_dt_raw = ts_series[len(ts_series) - 1]
            start_dt = _ensure_datetime(start_dt_raw)
            end_dt = _ensure_datetime(end_dt_raw)
            overall_window = _format_window(start_dt, end_dt)
            ts_start = format_dt(start_dt) if start_dt is not None else None
            ts_end = format_dt(end_dt) if end_dt is not None else None

            if cutoff > 0:
                train_start_dt = start_dt
                train_end_dt = _ensure_datetime(ts_series[min(cutoff - 1, len(ts_series) - 1)])
                train_window = _format_window(train_start_dt, train_end_dt)

            if cutoff < len(ts_series):
                val_start_dt = _ensure_datetime(ts_series[cutoff])
                val_end_dt = end_dt
                validation_window = _format_window(val_start_dt, val_end_dt)
    else:
        ts_series = None
        if pd is not None and hasattr(df_pd_sorted, "columns") and "ts_event" in df_pd_sorted.columns:
            try:
                ts_series = pd.to_datetime(df_pd_sorted["ts_event"], utc=True)
            except Exception:
                ts_series = None

        if ts_series is not None and hasattr(ts_series, "iloc") and len(ts_series) > 0:
            start_dt = ts_series.iloc[0].to_pydatetime()
            end_dt = ts_series.iloc[-1].to_pydatetime()
            overall_window = _format_window(start_dt, end_dt)
            ts_start = format_dt(start_dt)
            ts_end = format_dt(end_dt)

            if cutoff > 0:
                train_start = start_dt
                train_end = ts_series.iloc[max(cutoff - 1, 0)].to_pydatetime()
                train_window = _format_window(train_start, train_end)

            if cutoff < len(ts_series):
                val_start = ts_series.iloc[cutoff].to_pydatetime()
                val_end = end_dt
                validation_window = _format_window(val_start, val_end)

    # Placeholder for explicit test split metadata (future extension)
    build_ts_iso = format_dt(build_ts)
    if build_ts_iso is None:
        build_ts_iso = ""

    vintage_iso = format_dt(vintage_as_of) if vintage_as_of else None
    macro_counts = dict(macro_observation_counts or {})

    metadata = DatasetMetadata(
        dataset_id=dataset_id,
        vintage_policy=vintage_policy,
        vintage_cutoff=vintage_iso,
        build_ts=build_ts_iso,
        ts_event_start=ts_start,
        ts_event_end=ts_end,
        overall_window=overall_window,
        train_window=train_window,
        validation_window=validation_window,
        test_window=None,
        macro_observation_counts=macro_counts,
        target_semantics=target_semantics,
        reproducibility=reproducibility,
    )
    return metadata


def _metadata_to_dict(metadata: DatasetMetadata) -> dict[str, Any]:
    payload: dict[str, Any] = {
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
        "capability_flags": metadata.capability_flags,
        "target_semantics": metadata.target_semantics,
        "reproducibility": metadata.reproducibility,
    }
    if metadata.market_bindings is not None:
        payload["market_bindings"] = [
            {
                "binding_id": binding.binding_id,
                "dataset_id": binding.dataset_id,
                "descriptor_id": binding.descriptor_id,
                "schema": binding.schema,
                "storage_kind": binding.storage_kind,
                "symbols": list(binding.symbols),
                "instrument_ids": list(binding.instrument_ids),
                "source": binding.source,
                "license_start": binding.license_start,
                "license_end": binding.license_end,
                "ts_event_start": binding.ts_event_start,
                "ts_event_end": binding.ts_event_end,
                "rows_from_store": binding.rows_from_store,
                "rows_from_catalog": binding.rows_from_catalog,
                "provider_dataset_id": binding.provider_dataset_id,
                "provider_schema": binding.provider_schema,
            }
            for binding in metadata.market_bindings
        ]
    else:
        payload["market_bindings"] = None
    return payload


def metadata_to_dict(metadata: DatasetMetadata) -> dict[str, Any]:
    """
    Convert dataset metadata into a JSON-serializable dictionary.

    Args:
        metadata: Dataset metadata to serialize.

    Returns:
        Dictionary payload ready for JSON encoding.

    Example:
        >>> payload = metadata_to_dict(metadata)
        >>> assert "dataset_id" in payload
    """
    return _metadata_to_dict(metadata)


def serialize_dataset_metadata(metadata: DatasetMetadata) -> str:
    """
    Serialize dataset metadata to a JSON string.

    Args:
        metadata: Dataset metadata to serialize.

    Returns:
        JSON string containing the metadata payload.

    Example:
        >>> payload = serialize_dataset_metadata(metadata)
        >>> assert payload.startswith("{")
    """
    return json.dumps(metadata_to_dict(metadata), indent=2)


def write_dataset_metadata(metadata: DatasetMetadata, out_dir: Path) -> Path:
    """
    Persist dataset metadata as `dataset_metadata.json` under the output directory.

    Args:
        metadata: Dataset metadata to persist.
        out_dir: Output directory for the metadata file.

    Returns:
        Path to the written metadata file.

    Example:
        >>> path = write_dataset_metadata(metadata, Path("ml_out/example"))
        >>> assert path.name == "dataset_metadata.json"
    """
    path = Path(out_dir) / "dataset_metadata.json"
    path.write_text(serialize_dataset_metadata(metadata), encoding="utf-8")
    return path


def _validate_dataset_metadata(metadata: DatasetMetadata) -> None:
    """Ensure computed metadata windows are internally consistent."""

    def _parse_window(window: tuple[str, str] | None) -> tuple[datetime | None, datetime | None]:
        if window is None:
            return (None, None)
        start_raw, end_raw = window
        start = parse_dt(start_raw)
        end = parse_dt(end_raw)
        if start is not None and end is not None and start > end:
            msg = f"Window start {start_raw} must be <= end {end_raw}"
            raise ValueError(msg)
        return (start, end)

    overall_start, overall_end = _parse_window(metadata.overall_window)
    ts_start = parse_dt(metadata.ts_event_start) if metadata.ts_event_start else None
    ts_end = parse_dt(metadata.ts_event_end) if metadata.ts_event_end else None

    if ts_start and ts_end and ts_start > ts_end:
        raise ValueError("ts_event_start must be <= ts_event_end")

    if metadata.overall_window and (ts_start or ts_end):
        if ts_start and overall_start and ts_start < overall_start:
            raise ValueError("ts_event_start earlier than overall_window start")
        if ts_end and overall_end and ts_end > overall_end:
            raise ValueError("ts_event_end later than overall_window end")

    for label, window in (
        ("train", metadata.train_window),
        ("validation", metadata.validation_window),
        ("test", metadata.test_window),
    ):
        start, end = _parse_window(window)
        if start and overall_start and start < overall_start:
            raise ValueError(f"{label}_window starts before overall window")
        if end and overall_end and end > overall_end:
            raise ValueError(f"{label}_window ends after overall window")


__all__ = [
    "DatasetMetadata",
    "DatasetMetadataExpectations",
    "MarketBindingMetadata",
    "build_metadata_expectations",
    "load_dataset_metadata",
    "metadata_to_dict",
    "require_reproducibility_metadata",
    "require_target_column_in_semantics",
    "require_target_semantics_contract",
    "require_target_semantics_execution_contract",
    "require_target_semantics_horizon_mode",
    "require_target_semantics_metadata",
    "resolve_target_col_from_metadata",
    "serialize_dataset_metadata",
    "validate_dataset_metadata_expectations",
    "write_dataset_metadata",
]
