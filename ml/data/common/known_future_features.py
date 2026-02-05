"""
Known-future feature component for TFT dataset building.

This component delegates calendar and event schedule features to the canonical
pipeline batch backend. Legacy time_index/tod_sin aliases have been removed; a
``timestamp`` or ``ts_event`` column is required.

Guardrail: do not add ad-hoc feature math here. Register new transforms in
`ml/features/pipeline.py` and rely on the batch/stream executors instead.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from ml._imports import pd as pd_runtime
from ml._imports import pl as pl_runtime
from ml.data.common.event_provider import build_event_provider
from ml.data.common.feature_config_utils import normalize_feature_config
from ml.data.common.pipeline_batch import PipelineBatchContext
from ml.data.common.pipeline_batch import PipelineBatchExecutor
from ml.data.providers.calendar import MarketCalendarProvider
from ml.data.sources.calendar import CalendarSource
from ml.data.sources.calendar import MockCalendarSource
from ml.data.sources.calendar import PandasCalendarSource
from ml.features.config import FeatureConfig
from ml.features.pipeline import PipelineSpec
from ml.features.pipeline import TransformSpec


if TYPE_CHECKING:
    import pandas as _pd
    import polars as _pl

    from ml.data.providers.events import EventScheduleProvider
else:  # pragma: no cover - typing fallback
    _pd = Any
    _pl = Any
    EventScheduleProvider = Any


# Runtime aliases
pl: Any = cast(Any, pl_runtime)
pd: Any = cast(Any, pd_runtime)


logger = logging.getLogger(__name__)


class KnownFutureFeatureComponent:
    """
    Component for generating known-future features for TFT models.

    This component computes time-based and calendar features that are
    known in advance, making them suitable for TFT's "known future"
    input category.

    Canonical outputs are defined by the pipeline spec (calendar/event_schedule).
    """

    def __init__(
        self,
        *,
        include_calendar: bool = False,
        include_event_schedule: bool = False,
        feature_config: FeatureConfig | None = None,
        events_base_dir: Path | None = None,
        calendar_exchange: str = "NYSE",
    ) -> None:
        """
        Initialize KnownFutureFeatureComponent.

        Args:
            include_calendar: Whether to include precise calendar features
                from MarketCalendarProvider. Default False for performance.
            include_event_schedule: Whether to include event schedule features.
            feature_config: Optional FeatureConfig for canonical batch execution.
            events_base_dir: Base directory for event data sources.
            calendar_exchange: Exchange identifier for calendar features.

        """
        self.include_calendar = include_calendar
        self.include_event_schedule = include_event_schedule
        self._feature_config = normalize_feature_config(feature_config)
        self._events_base_dir = events_base_dir
        self._calendar_exchange = calendar_exchange
        self._calendar_provider: MarketCalendarProvider | None = None
        self._event_provider: EventScheduleProvider | None = None

    def _get_calendar_provider(self) -> MarketCalendarProvider:
        if self._calendar_provider is not None:
            return self._calendar_provider
        source: CalendarSource
        if self.include_calendar:
            try:
                source = cast(CalendarSource, PandasCalendarSource())
            except Exception as exc:
                logger.debug(
                    "known_future_features.calendar_source_fallback error=%s",
                    exc,
                    exc_info=True,
                )
                source = cast(CalendarSource, MockCalendarSource())
        else:
            source = cast(CalendarSource, MockCalendarSource())
        self._calendar_provider = MarketCalendarProvider(source)
        return self._calendar_provider

    def _get_event_provider(self) -> EventScheduleProvider | None:
        if self._event_provider is not None:
            return self._event_provider
        self._event_provider = build_event_provider(self._events_base_dir)
        return self._event_provider

    def _build_canonical_spec(self) -> PipelineSpec:
        transforms = [
            TransformSpec(
                name="calendar",
                params={"encoding": getattr(self._feature_config, "calendar_encoding", "cyclic")},
            ),
        ]
        if self.include_event_schedule:
            transforms.append(TransformSpec(name="event_schedule", params={}))
        return PipelineSpec(transforms=transforms)

    def add_known_future_features_canonical_polars(
        self,
        df: _pl.DataFrame,
    ) -> _pl.DataFrame:
        """
        Add canonical known-future features using the pipeline batch backend (Polars).

        Args:
            df: Polars DataFrame with timestamp column.

        Returns:
            DataFrame with canonical calendar/event features appended.
        """
        if df.is_empty():
            return df

        spec = self._build_canonical_spec()
        if not spec.transforms:
            return df

        context = PipelineBatchContext(
            feature_config=self._feature_config,
            calendar_provider=self._get_calendar_provider(),
            event_provider=self._get_event_provider() if self.include_event_schedule else None,
            calendar_exchange=self._calendar_exchange,
        )
        executor = PipelineBatchExecutor(
            spec,
            allowable=self._feature_config.resolved_data_requirements(),
            context=context,
        )
        return executor.execute_polars(df)

    def add_known_future_features_canonical_pandas(
        self,
        df: _pd.DataFrame,
    ) -> _pd.DataFrame:
        """
        Add canonical known-future features using the pipeline batch backend (Pandas).

        Args:
            df: Pandas DataFrame with timestamp column.

        Returns:
            DataFrame with canonical calendar/event features appended.
        """
        if len(df) == 0:
            return df

        spec = self._build_canonical_spec()
        if not spec.transforms:
            return df

        context = PipelineBatchContext(
            feature_config=self._feature_config,
            calendar_provider=self._get_calendar_provider(),
            event_provider=self._get_event_provider() if self.include_event_schedule else None,
            calendar_exchange=self._calendar_exchange,
        )
        executor = PipelineBatchExecutor(
            spec,
            allowable=self._feature_config.resolved_data_requirements(),
            context=context,
        )
        return executor.execute_pandas(df)

__all__ = ["KnownFutureFeatureComponent"]
