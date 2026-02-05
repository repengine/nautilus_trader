"""
Dataset build event emission helpers.

Provides a focused helper for emitting dataset lineage events after a build
completes. This keeps dataset build orchestration lean while preserving
best-effort event emission semantics.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import cast

from ml.config.events import EventStatus
from ml.config.events import Source
from ml.config.events import Stage
from ml.ml_types import PolarsDF


def _extract_event_bounds_ns(df: PolarsDF) -> tuple[int, int]:
    """
    Extract min/max timestamps (ns) from a dataset frame.

    Falls back to ``0, 0`` if timestamps are unavailable or parsing fails.
    """
    from ml._imports import pl

    if pl is None or not isinstance(df, pl.DataFrame):
        return 0, 0

    try:
        if "timestamp" in df.columns:
            ts_min_ns = int(
                df.select(pl.col("timestamp").cast(pl.Datetime("ns")).min()).item(),
            )
            ts_max_ns = int(
                df.select(pl.col("timestamp").cast(pl.Datetime("ns")).max()).item(),
            )
            return ts_min_ns, ts_max_ns
        if "ts_event" in df.columns:
            ts_min_ns = int(df.select(pl.col("ts_event").min()).item())
            ts_max_ns = int(df.select(pl.col("ts_event").max()).item())
            return ts_min_ns, ts_max_ns
    except Exception:
        return 0, 0

    return 0, 0


def emit_dataset_build_event(
    *,
    df: PolarsDF,
    dataset_id: str | None,
    symbols: Sequence[str],
    include_macro: bool,
    include_micro: bool,
    include_l2: bool,
    lookback_periods: int,
    primary_horizon_minutes: int | None,
    stage: Stage = Stage.FEATURE_COMPUTED,
    source: Source = Source.HISTORICAL,
    instrument_id: str = "GLOBAL",
    component: str = "dataset_builder",
    dataset_type: str = "dataset",
) -> None:
    """
    Emit a dataset lineage event for a completed build.

    Args:
        df: Dataset frame used to derive event bounds.
        dataset_id: Dataset identifier (defaults to ``tft_dataset`` if None).
        symbols: Symbols included in the dataset.
        include_macro: Whether macro features were included.
        include_micro: Whether microstructure features were included.
        include_l2: Whether L2 features were included.
        lookback_periods: Lookback window length.
        primary_horizon_minutes: Primary target horizon (minutes) if available.
        stage: Event stage to emit.
        source: Event source.
        instrument_id: Instrument identifier for emission.
        component: Component name emitting the event.
        dataset_type: Dataset type label.

    Returns:
        None.

    Example:
        >>> emit_dataset_build_event(
        ...     df=dataset_df,
        ...     dataset_id="tft_dataset",
        ...     symbols=["SPY"],
        ...     include_macro=True,
        ...     include_micro=False,
        ...     include_l2=False,
        ...     lookback_periods=30,
        ...     primary_horizon_minutes=15,
        ... )
    """
    try:
        from ml.common.event_emitter import emit_dataset_event
        from ml.core.integration import MLIntegrationManager
        from ml.registry.protocols import RegistryProtocol

        mgr = MLIntegrationManager(
            auto_start_postgres=False,
            auto_migrate=False,
            ensure_healthy=False,
        )
        data_registry = cast(RegistryProtocol, mgr.data_registry)
        count = int(getattr(df, "height", 0))
        ts_min_ns, ts_max_ns = _extract_event_bounds_ns(df)
        emit_dataset_event(
            data_registry,
            dataset_id=dataset_id or "tft_dataset",
            instrument_id=instrument_id,
            stage=stage,
            source=source,
            run_id=f"build_tft_{int(time.time() * 1e6):d}",
            ts_min=ts_min_ns,
            ts_max=ts_max_ns,
            count=count,
            status=EventStatus.SUCCESS,
            metadata={
                "symbols": ",".join(symbols),
                "include_macro": bool(include_macro),
                "include_micro": bool(include_micro),
                "include_l2": bool(include_l2),
                "horizon_minutes": int(primary_horizon_minutes)
                if primary_horizon_minutes is not None
                else None,
                "lookback_periods": int(lookback_periods),
            },
            dataset_type=dataset_type,
            component=component,
        )
    except Exception:
        logging.getLogger(__name__).debug(
            "Emit dataset event failed (ignored)",
            exc_info=True,
        )
