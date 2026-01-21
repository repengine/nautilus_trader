"""
Macro data cache for fast real-time feature access.

Provides pre-loaded ALFRED vintages and FRED data for low-latency macro feature
computation in the hot path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias

from ml._imports import check_ml_dependencies
from ml._imports import pl


if pl is None:
    check_ml_dependencies(["polars"])  # pragma: no cover

if TYPE_CHECKING:
    import polars as _pl

    PolarsDataFrame: TypeAlias = _pl.DataFrame
else:  # pragma: no cover - runtime type erasure
    PolarsDataFrame = Any  # type: ignore[assignment]


logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_relativedelta() -> Any:
    """Return the ``dateutil.relativedelta`` callable lazily."""
    try:
        module = import_module("dateutil.relativedelta")
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError("python-dateutil is required for macro revisions") from exc
    return getattr(module, "relativedelta")


@dataclass(slots=True)
class MacroSeriesSnapshot:
    """
    Snapshot of macro series current state for real-time inference.

    Attributes
    ----------
    series_id : str
        FRED series identifier (e.g., "PAYEMS", "CPIAUCSL").
    current_value : float
        Latest released value.
    observation_ts : datetime
        Observation period of current value.
    release_ts : datetime
        When current value was released.
    prior_1m_value : float | None
        Value from 1 month ago.
    prior_3m_value : float | None
        Value from 3 months ago.
    prior_12m_value : float | None
        Value from 12 months ago.
    revision_1m : float | None
        Revision of prior month (revised - initial).
    revision_3m : float | None
        Cumulative revisions over last 3 months.
    initial_value : float | None
        Initial release value (before any revisions).

    """

    series_id: str
    current_value: float
    observation_ts: datetime
    release_ts: datetime
    prior_1m_value: float | None = None
    prior_3m_value: float | None = None
    prior_12m_value: float | None = None
    revision_1m: float | None = None
    revision_3m: float | None = None
    initial_value: float | None = None
    history: tuple[float, ...] = ()


@dataclass(slots=True)
class MacroDataCache:
    """
    Fast cache for real-time macro feature access.

    Pre-loads ALFRED vintages on initialization and provides O(1) lookups for
    latest values, prior periods, and revisions. Designed for hot-path usage
    with <1ms P99 latency.

    Parameters
    ----------
    vintage_base_dir : Path
        Directory containing ALFRED vintage data (data/features/macro/fred/vintages/).
    series_ids : list[str]
        List of FRED series to cache.
    enable_revisions : bool, default True
        Whether to compute and cache revision features.

    """

    vintage_base_dir: Path
    series_ids: list[str]
    enable_revisions: bool = True
    aux_series_ids: list[str] = field(default_factory=list)
    history_window: int = 400
    _snapshots: dict[str, MacroSeriesSnapshot] = field(default_factory=dict, init=False)
    _loaded: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        """Load all vintages on initialization."""
        self.refresh()

    def refresh(self) -> None:
        """
        Reload all vintages from disk.

        Call this periodically (e.g., daily) to pick up new releases.
        """
        if pl is None:
            check_ml_dependencies(["polars"])  # pragma: no cover
        _pl = pl
        assert _pl is not None

        all_series: list[str] = list(dict.fromkeys([*self.series_ids, *self.aux_series_ids]))

        logger.info("Loading macro data cache for %d series", len(all_series))

        for series_id in all_series:
            try:
                snapshot = self._load_series_snapshot(series_id, _pl)
                if snapshot is not None:
                    self._snapshots[series_id] = snapshot
            except Exception as e:
                logger.warning("Failed to load %s: %s", series_id, e)

        self._loaded = True
        logger.info(
            "Macro cache loaded: %d/%d series",
            len(self._snapshots),
            len(all_series),
        )

    def _load_series_snapshot(
        self,
        series_id: str,
        _pl: Any,
    ) -> MacroSeriesSnapshot | None:
        """Load snapshot for a single series."""
        calendar_path = self.vintage_base_dir / series_id / "release_calendar.parquet"

        if not calendar_path.exists():
            logger.debug("No vintage data for %s (market-based series)", series_id)
            return None

        # Load release calendar
        df: PolarsDataFrame = _pl.read_parquet(str(calendar_path))

        if df.is_empty():
            return None

        # Sort by release to get chronological order
        df = df.sort(["observation_ts", "release_ts"])

        latest_per_observation = df.unique(
            subset=["observation_ts"],
            keep="last",
        ).sort("observation_ts")
        history_values = self._extract_history(latest_per_observation)

        # Get latest observation (most recent release)
        latest = df.filter(
            _pl.col("observation_ts") == _pl.col("observation_ts").max(),
        ).sort("release_ts").tail(1)

        if latest.is_empty():
            return None

        current_value = float(latest["value"][0])
        observation_ts = latest["observation_ts"][0]
        release_ts = latest["release_ts"][0]

        # Get initial value for this observation (first release)
        initial_df = df.filter(
            _pl.col("observation_ts") == observation_ts,
        ).sort("release_ts").head(1)
        initial_value = float(initial_df["value"][0]) if not initial_df.is_empty() else current_value

        # Get prior period values
        prior_1m = self._get_prior_value(df, observation_ts, months=1, _pl=_pl)
        prior_3m = self._get_prior_value(df, observation_ts, months=3, _pl=_pl)
        prior_12m = self._get_prior_value(df, observation_ts, months=12, _pl=_pl)

        # Compute revisions if enabled
        revision_1m = None
        revision_3m = None

        if self.enable_revisions and prior_1m is not None:
            # Revision = (current value for prior obs) - (initial value for prior obs)
            relativedelta = _load_relativedelta()
            prior_obs_ts = observation_ts - relativedelta(months=1)
            prior_initial = self._get_initial_value(df, prior_obs_ts, _pl=_pl)
            if prior_initial is not None:
                revision_1m = prior_1m - prior_initial

            # 3-month cumulative revisions
            if prior_3m is not None:
                revision_3m = self._compute_cumulative_revisions(
                    df,
                    observation_ts,
                    months=3,
                    _pl=_pl,
                )

        return MacroSeriesSnapshot(
            series_id=series_id,
            current_value=current_value,
            observation_ts=observation_ts,
            release_ts=release_ts,
            prior_1m_value=prior_1m,
            prior_3m_value=prior_3m,
            prior_12m_value=prior_12m,
            revision_1m=revision_1m,
            revision_3m=revision_3m,
            initial_value=initial_value,
            history=history_values,
        )

    def _extract_history(self, df: PolarsDataFrame) -> tuple[float, ...]:
        """Return trailing value history limited by configured window."""
        if self.history_window <= 0:
            return tuple()

        history_slice = df.tail(self.history_window)
        if history_slice.is_empty():
            return tuple()

        values = history_slice["value"].to_list()
        return tuple(float(v) for v in values)

    def _get_prior_value(
        self,
        df: PolarsDataFrame,
        current_obs_ts: datetime,
        months: int,
        _pl: Any,
    ) -> float | None:
        """Get value from N months ago (latest release for that observation)."""
        relativedelta = _load_relativedelta()

        prior_obs_ts = current_obs_ts - relativedelta(months=months)

        prior_df = df.filter(
            _pl.col("observation_ts") == prior_obs_ts,
        ).sort("release_ts").tail(1)

        if prior_df.is_empty():
            return None

        return float(prior_df["value"][0])

    def _get_initial_value(
        self,
        df: PolarsDataFrame,
        obs_ts: datetime,
        _pl: Any,
    ) -> float | None:
        """Get initial (first) release value for an observation."""
        initial_df = df.filter(
            _pl.col("observation_ts") == obs_ts,
        ).sort("release_ts").head(1)

        if initial_df.is_empty():
            return None

        return float(initial_df["value"][0])

    def _compute_cumulative_revisions(
        self,
        df: PolarsDataFrame,
        current_obs_ts: datetime,
        months: int,
        _pl: Any,
    ) -> float | None:
        """Compute cumulative revisions over last N months."""
        relativedelta = _load_relativedelta()
        total_revision = 0.0
        count = 0

        for month_offset in range(1, months + 1):
            obs_ts = current_obs_ts - relativedelta(months=month_offset)

            # Get latest value
            latest_val = self._get_prior_value(df, current_obs_ts, months=month_offset, _pl=_pl)
            if latest_val is None:
                continue

            # Get initial value
            initial_val = self._get_initial_value(df, obs_ts, _pl=_pl)
            if initial_val is None:
                continue

            total_revision += latest_val - initial_val
            count += 1

        return total_revision if count > 0 else None

    def get_snapshot(self, series_id: str) -> MacroSeriesSnapshot | None:
        """
        Get cached snapshot for a series.

        Returns
        -------
        MacroSeriesSnapshot | None
            Cached snapshot, or None if series not available.

        """
        return self._snapshots.get(series_id)

    def get_features(
        self,
        series_id: str,
        mode: str = "core",
    ) -> dict[str, float]:
        """
        Get all features for a series in real-time.

        Parameters
        ----------
        series_id : str
            FRED series identifier.
        mode : {"minimal", "core", "full"}
            Feature mode (same as batch).

        Returns
        -------
        dict[str, float]
            Feature name → value mapping.

        """
        snapshot = self._snapshots.get(series_id)
        if snapshot is None:
            return {}

        features: dict[str, float] = {
            f"{series_id}__value_real_time": snapshot.current_value,
        }

        # Minimal mode: current, prior_1m, revision_1m
        if mode in ["minimal", "core", "full"]:
            if snapshot.prior_1m_value is not None:
                features[f"{series_id}_prior_1m"] = snapshot.prior_1m_value

            if self.enable_revisions and snapshot.revision_1m is not None:
                features[f"{series_id}_revision_1m"] = snapshot.revision_1m

        # Core mode: add momentum, pct, net_signal
        if mode in ["core", "full"]:
            if snapshot.prior_1m_value is not None:
                # Month-over-month change
                features[f"{series_id}_mom_1m"] = (
                    snapshot.current_value - snapshot.prior_1m_value
                )

                # Percentage change
                if snapshot.prior_1m_value != 0:
                    features[f"{series_id}_pct_1m"] = (
                        (snapshot.current_value / snapshot.prior_1m_value) - 1.0
                    )

            # Net signal (headline adjusted for revision)
            if self.enable_revisions and snapshot.revision_1m is not None:
                features[f"{series_id}_net_signal_1m"] = (
                    snapshot.current_value - snapshot.revision_1m
                )

        # Full mode: add 3m and 12m features
        if mode == "full":
            if snapshot.prior_3m_value is not None:
                features[f"{series_id}_prior_3m"] = snapshot.prior_3m_value
                features[f"{series_id}_mom_3m"] = (
                    snapshot.current_value - snapshot.prior_3m_value
                )

            if snapshot.prior_12m_value is not None:
                features[f"{series_id}_prior_12m"] = snapshot.prior_12m_value
                features[f"{series_id}_mom_12m"] = (
                    snapshot.current_value - snapshot.prior_12m_value
                )

                if snapshot.prior_12m_value != 0:
                    features[f"{series_id}_pct_12m"] = (
                        (snapshot.current_value / snapshot.prior_12m_value) - 1.0
                    )

            if self.enable_revisions and snapshot.revision_3m is not None:
                features[f"{series_id}_revision_3m"] = snapshot.revision_3m

        return features

    def get_all_features(self, mode: str = "core") -> dict[str, float]:
        """
        Get features for all cached series.

        Parameters
        ----------
        mode : {"minimal", "core", "full"}
            Feature mode.

        Returns
        -------
        dict[str, float]
            All macro features for real-time inference.

        """
        all_features: dict[str, float] = {}

        for series_id in self.series_ids:
            features = self.get_features(series_id, mode=mode)
            all_features.update(features)

        return all_features

    def is_loaded(self) -> bool:
        """Check if cache has been loaded."""
        return self._loaded

    def get_coverage(self) -> dict[str, bool]:
        """Get coverage map showing which series are available."""
        return {series_id: series_id in self._snapshots for series_id in self.series_ids}
