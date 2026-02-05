"""
Macro and vintage orchestration helpers for dataset builds.

Centralizes macro refresh preparation, validation tuning, and ALFRED window
derivation so dataset builders share one implementation.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import Protocol

from ml.data.validation import DatasetValidationConfig
from ml.data.validation import DatasetValidationError
from ml.data.vintage import VintagePolicy
from ml.stores.protocols import DataStoreFacadeProtocol


DEFAULT_FRED_PARQUET_PATH = Path("data/features/macro/fred_indicators_ml_format.parquet")


class MacroRefreshConfig(Protocol):
    """
    Minimal configuration required for macro refresh orchestration.
    """

    @property
    def include_macro(self) -> bool:  # pragma: no cover - protocol
        ...

    @property
    def auto_refresh_macro(self) -> bool:  # pragma: no cover - protocol
        ...

    @property
    def student_mode(self) -> bool:  # pragma: no cover - protocol
        ...

    @property
    def macro_staleness_hours(self) -> int:  # pragma: no cover - protocol
        ...

    @property
    def macro_series_ids(self) -> tuple[str, ...] | None:  # pragma: no cover - protocol
        ...

    @property
    def fred_vintage_dir(self) -> Path | None:  # pragma: no cover - protocol
        ...

    @property
    def macro_fred_path(self) -> Path | None:  # pragma: no cover - protocol
        ...

    @property
    def chunk_days(self) -> int:  # pragma: no cover - protocol
        ...

    @property
    def start(self) -> datetime | None:  # pragma: no cover - protocol
        ...

    @property
    def end(self) -> datetime | None:  # pragma: no cover - protocol
        ...


class MacroValidationConfig(Protocol):
    """
    Minimal configuration required for macro/vintage validation tuning.
    """

    @property
    def include_macro(self) -> bool:  # pragma: no cover - protocol
        ...

    @property
    def macro_series_ids(self) -> tuple[str, ...] | None:  # pragma: no cover - protocol
        ...

    @property
    def vintage_policy(self) -> VintagePolicy:  # pragma: no cover - protocol
        ...


class LoggerProtocol(Protocol):
    """
    Minimal logger protocol for structured warnings.
    """

    def warning(self, msg: str, **kwargs: object) -> None:  # pragma: no cover - protocol
        ...


def resolve_fred_parquet_path(cfg: MacroRefreshConfig) -> Path:
    """
    Resolve the FRED parquet path for macro joins.

    Args:
        cfg: Dataset build configuration.

    Returns:
        Path to the FRED parquet artifact.
    """
    return cfg.macro_fred_path or DEFAULT_FRED_PARQUET_PATH


def derive_alfred_range(cfg: MacroRefreshConfig) -> tuple[str | None, str | None]:
    """
    Derive ALFRED refresh window bounds based on build start/end.

    Args:
        cfg: Dataset build configuration.

    Returns:
        Tuple of (start_iso, end_iso) for ALFRED refresh.
    """
    start_dt = getattr(cfg, "start", None)
    end_dt = getattr(cfg, "end", None)
    start_iso = getattr(cfg, "start_iso", None)
    end_iso = getattr(cfg, "end_iso", None)
    if start_dt is None and start_iso:
        start_dt = datetime.fromisoformat(start_iso)
    if end_dt is None and end_iso:
        end_dt = datetime.fromisoformat(end_iso)

    buffer = timedelta(days=30)
    if start_dt is not None:
        start_dt = start_dt - buffer
    if end_dt is not None:
        end_dt = end_dt + buffer

    today_utc = datetime.now(tz=UTC).date()

    def _normalize(dt: datetime | None) -> str | None:
        if dt is None:
            return None
        if dt.tzinfo is not None:
            dt = dt.astimezone(UTC)
        date_value = dt.date()
        if date_value > today_utc:
            date_value = today_utc
        return date_value.isoformat()

    return _normalize(start_dt), _normalize(end_dt)


def resolve_alfred_window_days(cfg: MacroRefreshConfig) -> int:
    """
    Resolve the ALFRED refresh window size in days.

    Args:
        cfg: Dataset build configuration.

    Returns:
        Window span in days for ALFRED refresh.
    """
    return 180 if cfg.chunk_days else 365


def normalize_vintage_as_of(value: datetime | None) -> datetime | None:
    """
    Normalize vintage cutoff timestamps to UTC-aware datetimes.

    Args:
        value: Vintage cutoff datetime.

    Returns:
        Timezone-aware datetime or None.
    """
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def refresh_macro_artifacts_if_needed(
    *,
    cfg: MacroRefreshConfig,
    data_store: DataStoreFacadeProtocol | None,
    fred_parquet_path: Path,
    alfred_start: str | None,
    alfred_end: str | None,
    alfred_window_days: int,
    logger: LoggerProtocol,
) -> None:
    """
    Refresh macro artifacts when configured and staleness thresholds are met.

    Args:
        cfg: Dataset build configuration.
        data_store: Optional DataStore for macro refresh dependencies.
        fred_parquet_path: Path to the FRED parquet artifact.
        alfred_start: ALFRED refresh start date (ISO).
        alfred_end: ALFRED refresh end date (ISO).
        alfred_window_days: Refresh window span in days.
        logger: Logger for warnings.
    """
    if not (cfg.include_macro and cfg.auto_refresh_macro and not cfg.student_mode):
        return

    from ml.data.ingest.macro_refresh import ensure_macro_ready

    refresh_window = timedelta(hours=max(cfg.macro_staleness_hours, 0))
    macro_refresh = ensure_macro_ready(
        fred_path=fred_parquet_path,
        vintage_dir=cfg.fred_vintage_dir,
        max_age=refresh_window,
        data_store=data_store,
        series_ids=cfg.macro_series_ids,
        alfred_realtime_start=alfred_start,
        alfred_realtime_end=alfred_end,
        alfred_window_days=alfred_window_days,
    )
    if macro_refresh.fred_error is not None:
        logger.warning(
            "FRED macro refresh failed; proceeding with existing artifacts",
            error=str(macro_refresh.fred_error),
            path=str(fred_parquet_path),
        )
    if macro_refresh.alfred_error is not None:
        logger.warning(
            "ALFRED macro refresh failed; proceeding with existing artifacts",
            error=str(macro_refresh.alfred_error),
            base_dir=str(macro_refresh.alfred_base_dir),
        )


def prepare_validation_config(
    *,
    cfg: MacroValidationConfig,
    validation_cfg: DatasetValidationConfig | None,
) -> DatasetValidationConfig:
    """
    Prepare validation config by injecting macro/vintage expectations.

    Args:
        cfg: Dataset build configuration.
        validation_cfg: Optional base validation config.

    Returns:
        Updated DatasetValidationConfig with macro/vintage expectations applied.
    """
    validation_cfg = validation_cfg or DatasetValidationConfig(
        require_macro_series=cfg.macro_series_ids,
    )
    if validation_cfg.require_macro_series is None and cfg.macro_series_ids:
        validation_cfg = replace(validation_cfg, require_macro_series=cfg.macro_series_ids)
    if validation_cfg.expected_vintage_policy is None:
        validation_cfg = replace(validation_cfg, expected_vintage_policy=cfg.vintage_policy)
    if (
        validation_cfg.macro_min_vintage_observations is None
        and cfg.include_macro
        and cfg.vintage_policy is VintagePolicy.REAL_TIME
        and cfg.macro_series_ids
    ):
        validation_cfg = replace(validation_cfg, macro_min_vintage_observations=1)
    if cfg.vintage_policy is not VintagePolicy.REAL_TIME and validation_cfg.macro_min_vintage_observations is not None:
        validation_cfg = replace(validation_cfg, macro_min_vintage_observations=None)
    return validation_cfg


def resolve_macro_presence(
    *,
    config: DatasetValidationConfig,
    macro_counts: dict[str, int],
) -> tuple[str, ...]:
    """
    Resolve macro series presence and enforce vintage observation thresholds.

    Args:
        config: Dataset validation config.
        macro_counts: Observations per macro series.

    Returns:
        Tuple of macro series present in the dataset.
    """
    if config.require_macro_series:
        required = set(config.require_macro_series)
        actual = {name for name, count in macro_counts.items() if count > 0}
        missing = required - actual
        if missing:
            msg = f"Missing macro series: {sorted(missing)}"
            raise DatasetValidationError(msg)
        macro_present = tuple(sorted(actual))
        min_obs = config.macro_min_vintage_observations
        policy = config.expected_vintage_policy or VintagePolicy.REAL_TIME
        if min_obs is not None and policy is VintagePolicy.REAL_TIME:
            failing = [name for name in required if macro_counts.get(name, 0) < min_obs]
            if failing:
                weakest_series = min(failing, key=lambda name: macro_counts.get(name, 0))
                msg = (
                    "Macro vintage coverage below threshold; "
                    f"series {weakest_series} has {macro_counts.get(weakest_series, 0)} observations < {min_obs}"
                )
                raise DatasetValidationError(msg)
        return macro_present

    return tuple(sorted(macro_counts.keys()))
