"""
Databento ingestion safety policy loading utilities.

This module centralizes loading and validation of the repository's
``databento_safe_config.json`` file.  The configuration constrains which
Datasets/Schemas may be requested as part of historical ingestion and the
maximum free-range window that should be queried for each schema.  The values
are used in conjunction with environment-driven policy settings provided by
``ml.data.ingest.policy``.

The safe configuration intentionally lives on the cold path and is consumed by
command line tooling and data orchestrators.  It should *never* be imported in
hot-path actor code.

"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_SAFE_CONFIG_PATH = Path(__file__).with_name("databento_safe_config.json")


@dataclass(frozen=True, slots=True)
class SchemaSafetyConfig:
    """
    Schema specific safety configuration.

    Parameters
    ----------
    max_days:
        Maximum number of days that may be requested for a single historical
        call.  ``None`` indicates no explicit limit in the configuration.
    max_cost_usd:
        Optional schema specific cost ceiling.  When provided, the ingestion
        service will refuse to run if the Databento cost estimate exceeds this
        amount.  ``None`` indicates the global limit should be used.

    """

    max_days: int | None = None
    max_cost_usd: float | None = None


@dataclass(frozen=True, slots=True)
class DatabentoSafetyConfig:
    """
    Safety configuration derived from ``databento_safe_config.json``.
    """

    datasets: tuple[str, ...]
    schemas: dict[str, SchemaSafetyConfig]
    max_cost_usd: float | None
    max_symbols: int | None


class DatabentoSafetyConfigError(RuntimeError):
    """
    Raised when the safety configuration file is malformed.
    """


def _coerce_schema_config(raw: Any) -> SchemaSafetyConfig:
    if raw is None:
        return SchemaSafetyConfig()
    if not isinstance(raw, dict):  # pragma: no cover - defensive branch
        raise DatabentoSafetyConfigError("Schema configuration entries must be objects")
    max_days_raw = raw.get("max_days")
    max_cost_raw = raw.get("max_cost_usd")
    max_days = int(max_days_raw) if isinstance(max_days_raw, int) else None
    max_cost_usd = float(max_cost_raw) if isinstance(max_cost_raw, (int, float)) else None
    return SchemaSafetyConfig(max_days=max_days, max_cost_usd=max_cost_usd)


def load_databento_safety_config(path: Path | None = None) -> DatabentoSafetyConfig:
    """
    Load ``databento_safe_config.json`` from the repository.

    Parameters
    ----------
    path:
        Optional override path.  When ``None`` the repository default is used.

    Returns
    -------
    DatabentoSafetyConfig
        Parsed, immutable configuration structure.

    Raises
    ------
    DatabentoSafetyConfigError
        If the configuration cannot be parsed or contains invalid data.

    """
    cfg_path = path or DEFAULT_SAFE_CONFIG_PATH
    if not cfg_path.exists():  # pragma: no cover - validated via tests
        raise DatabentoSafetyConfigError(f"Databento safety config missing: {cfg_path}")
    try:
        raw = json.loads(cfg_path.read_text())
    except Exception as exc:  # pragma: no cover - validated via tests
        raise DatabentoSafetyConfigError(f"Failed to parse {cfg_path}: {exc}") from exc

    datasets_raw = raw.get("datasets", [])
    if not isinstance(datasets_raw, list) or not all(isinstance(s, str) for s in datasets_raw):
        raise DatabentoSafetyConfigError("'datasets' must be a list of strings")

    schemas_raw = raw.get("schemas", {})
    if not isinstance(schemas_raw, dict):
        raise DatabentoSafetyConfigError("'schemas' must be an object mapping schema->config")
    schemas = {str(name): _coerce_schema_config(cfg) for name, cfg in schemas_raw.items()}

    global_raw = raw.get("global", {})
    if not isinstance(global_raw, dict):
        raise DatabentoSafetyConfigError("'global' must be an object when present")
    max_cost_raw = global_raw.get("max_cost_usd")
    max_symbols_raw = global_raw.get("max_symbols")
    max_cost_usd = float(max_cost_raw) if isinstance(max_cost_raw, (int, float)) else None
    max_symbols = int(max_symbols_raw) if isinstance(max_symbols_raw, int) else None

    datasets = tuple(sorted({ds.strip() for ds in datasets_raw if ds.strip()}))

    return DatabentoSafetyConfig(
        datasets=datasets,
        schemas=schemas,
        max_cost_usd=max_cost_usd,
        max_symbols=max_symbols,
    )


__all__ = [
    "DEFAULT_SAFE_CONFIG_PATH",
    "DatabentoSafetyConfig",
    "DatabentoSafetyConfigError",
    "SchemaSafetyConfig",
    "load_databento_safety_config",
]
