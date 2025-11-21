"""\
Macro data refresh utilities for TFT dataset builds.

These helpers ensure that FRED macro parquet artifacts and ALFRED vintage releases
are present and within an acceptable staleness window before datasets are built. The
functions are cold-path only and integrate with the standardized metrics bootstrap so
operators can monitor refresh outcomes.

Example
-------
>>> from datetime import timedelta
>>> from pathlib import Path
>>> from ml.data.ingest.macro_refresh import ensure_macro_ready
>>> ensure_macro_ready(
...     fred_path=Path("data/fred/fred_indicators_ml_format.parquet"),
...     vintage_dir=Path("data/fred/vintages"),
...     max_age=timedelta(hours=24),
... )  # doctest: +SKIP
MacroRefreshResult(fred_refreshed=False, alfred_refreshed=False, ...)
"""

from __future__ import annotations

import time
from collections.abc import Callable
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import Any, Final, Protocol, cast

import structlog

from ml._imports import check_ml_dependencies
from ml._imports import pl
from ml.common.metrics_bootstrap import get_counter
from ml.common.metrics_bootstrap import get_histogram
from ml.common.timestamps import sanitize_timestamp_ns
from ml.config.dataset_ids import MACRO_OBSERVATIONS_DATASET_ID
from ml.config.dataset_ids import MACRO_RELEASES_DATASET_ID
from ml.config.events import Source
from ml.config.macro_universe import MARKET_BASED_MACRO_SERIES
from ml.stores.protocols import DataStoreFacadeProtocol


POLARS = cast(Any, pl)

logger = structlog.get_logger(__name__)

_REFRESH_COUNTER = get_counter(
    "ml_macro_refresh_total",
    "Macro artifact refresh attempts",
    ["target", "status"],
)
_REFRESH_SECONDS = get_histogram(
    "ml_macro_refresh_seconds",
    "Macro artifact refresh durations (seconds)",
    ["target", "status"],
)

FRED_TARGET: Final[str] = "fred"
ALFRED_TARGET: Final[str] = "alfred"


@dataclass(slots=True, frozen=True)
class MacroRefreshResult:
    """Result of ensuring macro data freshness."""

    fred_refreshed: bool
    alfred_refreshed: bool
    fred_path: Path
    alfred_base_dir: Path | None
    fred_error: Exception | None = None
    alfred_error: Exception | None = None


class _FREDLoaderProtocol(Protocol):
    """Minimal interface required from :class:`FREDDataLoader`."""

    def fetch_all_indicators(
        self,
        start_date: datetime | None = ...,
        end_date: datetime | None = ...,
        use_cache: bool = ...,
        **_: object,
    ) -> object:
        ...

    def export_ml_parquet(
        self,
        data: object = ...,
        out_path: Path | None = ...,
        **_: object,
    ) -> Path:
        ...


class _ALFREDLoaderProtocol(Protocol):
    """Minimal interface required from :class:`ALFREDDataLoader`."""

    def refresh(self) -> object:
        ...


def refresh_fred_if_stale(
    *,
    parquet_path: Path,
    max_age: timedelta,
    series_ids: Sequence[str] | None = None,
    loader_factory: Callable[[Sequence[str] | None], _FREDLoaderProtocol] | None = None,
) -> tuple[bool, Exception | None]:
    """
    Refresh FRED parquet when the file is missing or older than *max_age*.

    Parameters
    ----------
    parquet_path : Path
        Destination parquet expected by the TFT dataset builder.
    max_age : timedelta
        Maximum allowed age for the parquet modification timestamp.
    series_ids : Sequence[str] | None, optional
        Optional series identifiers to constrain the refresh. Defaults to the loader's
        configured indicator set.
    loader_factory : Callable[[Sequence[str] | None], object] | None, optional
        Factory that returns a configured ``FREDDataLoader``. Primarily for testing;
        defaults to constructing a loader with ``FREDConfig``.

    Returns
    -------
    tuple[bool, Exception | None]
        Tuple of ``(was_refreshed, error)``. Errors are captured instead of raised so
        cold-path callers can fall back gracefully.
    """
    if max_age <= timedelta(0):
        is_stale = True
    else:
        if not parquet_path.exists():
            is_stale = True
        else:
            mtime = datetime.fromtimestamp(parquet_path.stat().st_mtime, tz=UTC)
            is_stale = (datetime.now(tz=UTC) - mtime) > max_age

    if not is_stale:
        logger.debug("FRED macro file fresh", path=str(parquet_path))
        _REFRESH_COUNTER.labels(target=FRED_TARGET, status="skipped").inc()
        return False, None

    logger.info("Refreshing FRED macro data", path=str(parquet_path))
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    if loader_factory is None:
        loader_factory = _build_fred_loader

    start = time.perf_counter()
    try:
        loader = loader_factory(series_ids)
        data = loader.fetch_all_indicators(use_cache=False)
        loader.export_ml_parquet(data=data, out_path=parquet_path)
        duration = time.perf_counter() - start
        _REFRESH_COUNTER.labels(target=FRED_TARGET, status="success").inc()
        _REFRESH_SECONDS.labels(target=FRED_TARGET, status="success").observe(duration)
        logger.info("FRED macro data refreshed", path=str(parquet_path), duration_s=duration)
        return True, None
    except Exception as exc:  # pragma: no cover - network/IO errors
        duration = time.perf_counter() - start
        _REFRESH_COUNTER.labels(target=FRED_TARGET, status="error").inc()
        _REFRESH_SECONDS.labels(target=FRED_TARGET, status="error").observe(duration)
        logger.warning(
            "FRED macro refresh failed",
            path=str(parquet_path),
            duration_s=duration,
            exc_info=True,
        )
        return False, exc


def refresh_alfred_if_stale(
    *,
    base_dir: Path,
    max_age: timedelta,
    series_ids: Sequence[str],
    loader_factory: Callable[[Sequence[str]], _ALFREDLoaderProtocol] | None = None,
) -> tuple[bool, Exception | None]:
    """
    Refresh ALFRED vintage releases if any tracked series is stale.

    Parameters
    ----------
    base_dir : Path
        Base directory where ALFRED loader writes release calendars.
    max_age : timedelta
        Maximum allowed age for per-series ``release_calendar.parquet`` files.
    series_ids : Sequence[str]
        Series identifiers that the dataset expects parity for.
    loader_factory : Callable[[Sequence[str]], object] | None, optional
        Optional factory producing an ``ALFREDDataLoader`` instance.

    Returns
    -------
    tuple[bool, Exception | None]
        Tuple of ``(was_refreshed, error)``.
    """
    if not series_ids:
        logger.debug("No ALFRED series configured; skipping refresh")
        _REFRESH_COUNTER.labels(target=ALFRED_TARGET, status="skipped").inc()
        return False, None

    missing_or_stale = False
    for series_id in series_ids:
        cal_path = base_dir / series_id / "release_calendar.parquet"
        if not cal_path.exists():
            missing_or_stale = True
            break
        mtime = datetime.fromtimestamp(cal_path.stat().st_mtime, tz=UTC)
        if max_age <= timedelta(0) or (datetime.now(tz=UTC) - mtime) > max_age:
            missing_or_stale = True
            break

    if not missing_or_stale:
        logger.debug("ALFRED vintage data fresh", base_dir=str(base_dir))
        _REFRESH_COUNTER.labels(target=ALFRED_TARGET, status="skipped").inc()
        return False, None

    logger.info("Refreshing ALFRED vintage data", base_dir=str(base_dir), series=len(series_ids))
    base_dir.mkdir(parents=True, exist_ok=True)

    if loader_factory is None:
        def _default_loader(series: Sequence[str]) -> _ALFREDLoaderProtocol:
            return _build_alfred_loader(series, None, None, 365)

        loader_factory = _default_loader

    start = time.perf_counter()
    try:
        loader = loader_factory(series_ids)
        loader.refresh()
        duration = time.perf_counter() - start
        _REFRESH_COUNTER.labels(target=ALFRED_TARGET, status="success").inc()
        _REFRESH_SECONDS.labels(target=ALFRED_TARGET, status="success").observe(duration)
        logger.info(
            "ALFRED vintage data refreshed",
            base_dir=str(base_dir),
            duration_s=duration,
            series=len(series_ids),
        )
        return True, None
    except Exception as exc:  # pragma: no cover - network/IO errors
        duration = time.perf_counter() - start
        _REFRESH_COUNTER.labels(target=ALFRED_TARGET, status="error").inc()
        _REFRESH_SECONDS.labels(target=ALFRED_TARGET, status="error").observe(duration)
        logger.warning(
            "ALFRED vintage refresh failed",
            base_dir=str(base_dir),
            duration_s=duration,
            exc_info=True,
        )
        return False, exc


def ensure_macro_ready(
    *,
    fred_path: Path,
    vintage_dir: Path | None,
    max_age: timedelta,
    series_ids: Sequence[str] | None = None,
    fred_loader_factory: Callable[[Sequence[str] | None], _FREDLoaderProtocol] | None = None,
    alfred_loader_factory: Callable[[Sequence[str]], _ALFREDLoaderProtocol] | None = None,
    alfred_realtime_start: str | None = None,
    alfred_realtime_end: str | None = None,
    alfred_window_days: int = 365,
    data_store: DataStoreFacadeProtocol | None = None,
    ingest_run_id: str | None = None,
) -> MacroRefreshResult:
    """
    Ensure macro artifacts exist within *max_age* and refresh when necessary.

    Args:
        fred_path: Location of the ML-format parquet.
        vintage_dir: Directory containing ALFRED release calendars.
        max_age: Maximum allowed staleness for artifacts.
        series_ids: Optional subset of macro series to refresh/ingest.
        data_store: When provided, refreshed artifacts are ingested into SQL
            via :class:`FeatureDatasetStore`.
        ingest_run_id: Optional run identifier to stamp on dataset events.

    """
    fred_refreshed, fred_error = refresh_fred_if_stale(
        parquet_path=fred_path,
        max_age=max_age,
        series_ids=series_ids,
        loader_factory=fred_loader_factory,
    )

    alfred_refreshed = False
    alfred_error: Exception | None = None
    if vintage_dir is not None:
        loader_factory = alfred_loader_factory
        if loader_factory is None:
            def _factory(series: Sequence[str]) -> _ALFREDLoaderProtocol:
                return _build_alfred_loader(
                    series,
                    alfred_realtime_start,
                    alfred_realtime_end,
                    alfred_window_days,
                )

            loader_factory = _factory
        alfred_refreshed, alfred_error = refresh_alfred_if_stale(
            base_dir=vintage_dir,
            max_age=max_age,
            series_ids=tuple(series_ids or ()),
            loader_factory=loader_factory,
        )

    if data_store is not None:
        run_id_value = ingest_run_id or f"macro_refresh_{int(time.time())}"
        try:
            _ingest_macro_datasets(
                data_store=data_store,
                fred_path=fred_path,
                vintage_dir=vintage_dir,
                series_ids=series_ids,
                run_id=run_id_value,
            )
        except Exception:
            logger.warning("macro_sql_ingest_failed", exc_info=True)

    return MacroRefreshResult(
        fred_refreshed=fred_refreshed,
        alfred_refreshed=alfred_refreshed,
        fred_path=fred_path,
        alfred_base_dir=vintage_dir,
        fred_error=fred_error,
        alfred_error=alfred_error,
    )


def _ingest_macro_datasets(
    *,
    data_store: DataStoreFacadeProtocol,
    fred_path: Path,
    vintage_dir: Path | None,
    series_ids: Sequence[str] | None,
    run_id: str,
) -> None:
    _pl = _require_polars()
    ts_init_ns = sanitize_timestamp_ns(time.time_ns(), context="macro_ingest")
    _ingest_macro_releases(
        data_store=data_store,
        vintage_dir=vintage_dir,
        series_ids=series_ids,
        run_id=run_id,
        ts_init_ns=ts_init_ns,
        polars_module=_pl,
    )
    _ingest_macro_observations(
        data_store=data_store,
        fred_path=fred_path,
        series_ids=series_ids,
        run_id=run_id,
        ts_init_ns=ts_init_ns,
        polars_module=_pl,
    )


def _ingest_macro_releases(
    *,
    data_store: DataStoreFacadeProtocol,
    vintage_dir: Path | None,
    series_ids: Sequence[str] | None,
    run_id: str,
    ts_init_ns: int,
    polars_module: Any,
) -> None:
    if vintage_dir is None or not vintage_dir.exists():
        return
    targets = _normalize_series_ids(series_ids) or tuple(
        sorted(child.name for child in vintage_dir.iterdir() if child.is_dir())
    )
    if not targets:
        return
    for series in targets:
        cal_path = vintage_dir / series / "release_calendar.parquet"
        if not cal_path.exists():
            continue
        try:
            frame = polars_module.read_parquet(str(cal_path))
        except Exception:
            logger.warning(
                "macro_release.parquet_read_failed",
                series=series,
                path=str(cal_path),
                exc_info=True,
            )
            continue
        if "release_end_ts" not in frame.columns:
            frame = frame.with_columns(polars_module.lit(None).alias("release_end_ts"))
        if frame.is_empty():
            continue
        prepared = (
            frame.with_columns(
                [
                    polars_module.lit(series).alias("series_id"),
                    polars_module.col("observation_ts").cast(polars_module.Datetime("ns")).cast(polars_module.Int64),
                    polars_module.col("release_ts").cast(polars_module.Datetime("ns")).cast(polars_module.Int64),
                    polars_module.col("release_end_ts")
                    .cast(polars_module.Datetime("ns"))
                    .cast(polars_module.Int64),
                    polars_module.col("value").cast(polars_module.Float64),
                ],
            )
            .with_columns(
                [
                    polars_module.col("release_ts").alias("ts_event"),
                    polars_module.lit(ts_init_ns).alias("ts_init"),
                    polars_module.lit("macro_refresh").alias("source"),
                    polars_module.lit(run_id).alias("run_id"),
                ],
            )
            .select(
                [
                    "series_id",
                    "observation_ts",
                    "release_ts",
                    "release_end_ts",
                    "value",
                    "ts_event",
                    "ts_init",
                    "source",
                    "run_id",
                ],
            )
        )
        if prepared.is_empty():
            continue
        data_store.write_ingestion(
            dataset_id=MACRO_RELEASES_DATASET_ID,
            records=prepared,
            source=Source.HISTORICAL.value,
            run_id=f"{run_id}_release",
            instrument_id=series,
        )


def _ingest_macro_observations(
    *,
    data_store: DataStoreFacadeProtocol,
    fred_path: Path,
    series_ids: Sequence[str] | None,
    run_id: str,
    ts_init_ns: int,
    polars_module: Any,
) -> None:
    if not fred_path.exists():
        return
    try:
        frame = polars_module.read_parquet(str(fred_path))
    except Exception:
        logger.warning("macro_observations.parquet_read_failed", path=str(fred_path), exc_info=True)
        return
    if frame.is_empty():
        return
    if series_ids:
        normalized = _normalize_series_ids(series_ids)
        if normalized:
            frame = frame.filter(polars_module.col("series_id").is_in(normalized))
    if frame.is_empty():
        return
    prepared = (
        frame.rename({"timestamp": "observation_ts"})
        .with_columns(
            [
                polars_module.col("observation_ts").cast(polars_module.Datetime("ns")).cast(polars_module.Int64),
                polars_module.col("value").cast(polars_module.Float64),
            ],
        )
        .with_columns(
            [
                polars_module.col("observation_ts").alias("ts_event"),
                polars_module.lit(ts_init_ns).alias("ts_init"),
                polars_module.lit("macro_refresh").alias("source"),
                polars_module.lit(run_id).alias("run_id"),
            ],
        )
        .select(
            [
                "series_id",
                "observation_ts",
                "value",
                "ts_event",
                "ts_init",
                "source",
                "run_id",
            ],
        )
    )
    if prepared.is_empty():
        return
    available_series = prepared.select("series_id").unique().to_series().to_list()
    targets = _normalize_series_ids(series_ids) or tuple(str(series).strip() for series in available_series if series)
    for series in targets:
        subset = prepared.filter(polars_module.col("series_id") == series)
        if subset.is_empty():
            continue
        data_store.write_ingestion(
            dataset_id=MACRO_OBSERVATIONS_DATASET_ID,
            records=subset,
            source=Source.HISTORICAL.value,
            run_id=f"{run_id}_observations",
            instrument_id=series,
        )


def _normalize_series_ids(series_ids: Sequence[str] | None) -> tuple[str, ...]:
    if not series_ids:
        return tuple()
    ordered: list[str] = []
    seen: set[str] = set()
    for raw in series_ids:
        series = raw.strip().upper()
        if not series or series in seen:
            continue
        seen.add(series)
        ordered.append(series)
    return tuple(ordered)


def _require_polars() -> Any:
    if POLARS is not None:
        return POLARS
    check_ml_dependencies(["polars"])
    from ml._imports import pl as _pl

    return cast(Any, _pl)


def _build_fred_loader(series_ids: Sequence[str] | None) -> _FREDLoaderProtocol:
    from ml.data.loaders.fred_loader import FREDConfig
    from ml.data.loaders.fred_loader import FREDDataLoader
    from ml.data.loaders.fred_loader import FREDIndicator

    indicators: list[FREDIndicator] | None
    if series_ids is None:
        indicators = None
    else:
        default_map = {ind.series_id: ind for ind in FREDDataLoader.DEFAULT_INDICATORS}
        indicators = []
        for series_id in series_ids:
            base = default_map.get(series_id)
            if base is not None:
                indicators.append(base)
                continue
            indicators.append(
                FREDIndicator(
                    series_id=series_id,
                    name=series_id,
                    category="custom",
                ),
            )
    return cast(_FREDLoaderProtocol, FREDDataLoader(config=FREDConfig(), indicators=indicators))


def _build_alfred_loader(
    series_ids: Sequence[str],
    realtime_start: str | None,
    realtime_end: str | None,
    window_days: int,
) -> _ALFREDLoaderProtocol:
    from ml.data.loaders.alfred_loader import ALFREDConfig
    from ml.data.loaders.alfred_loader import ALFREDDataLoader

    cfg = ALFREDConfig(
        series_ids=tuple(series_ids),
        start_date=realtime_start,
        end_date=realtime_end,
        window_days=window_days,
        fallback_to_fred_series=MARKET_BASED_MACRO_SERIES,
    )
    return cast(_ALFREDLoaderProtocol, ALFREDDataLoader(cfg))
