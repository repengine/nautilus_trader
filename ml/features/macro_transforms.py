"""
Macro feature transforms for ALFRED vintage and FRED data integration.

Provides training/inference parity for macro features by implementing both batch
(historical) and real-time computation paths.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, cast

from ml._imports import check_ml_dependencies
from ml._imports import pl
from ml.data.vintage import VintagePolicy
from ml.features.macro_cache import MacroDataCache


if pl is None:
    check_ml_dependencies(["polars"])  # pragma: no cover

if TYPE_CHECKING:
    import polars as _pl
    from nautilus_trader.model.data import Bar

    PolarsDataFrame: TypeAlias = _pl.DataFrame
else:  # pragma: no cover - runtime aliasing
    PolarsDataFrame = Any  # type: ignore[assignment]


logger = logging.getLogger(__name__)


class MacroFeatureTransform:
    """
    Transform that adds ALFRED/FRED macro features with training/inference parity.

    This transform ensures that macro features are computed identically in both
    training (batch) and inference (real-time) modes, maintaining feature parity.

    Parameters
    ----------
    macro_series_ids : list[str]
        FRED series identifiers to include (e.g., ["PAYEMS", "UNRATE", "CPIAUCSL"]).
    vintage_base_dir : Path | str
        Directory containing ALFRED vintage data (data/fred/vintages/).
    fred_path : Path | str | None
        Path to FRED indicators parquet file (for series without vintages).
    include_revisions : bool, default False
        Whether to include revision-aware features.
    revision_mode : {"minimal", "core", "full"}, default "core"
        Revision feature mode:
        - minimal: current, prior_1m, revision_1m
        - core: + mom_1m, pct_1m, net_signal_1m
        - full: + prior_3m/12m, revision_3m, mom_3m/12m, pct_12m
    lag_days : int, default 1
        Publication lag for non-vintage series.
    vintage_policy : VintagePolicy, default REAL_TIME
        Policy for vintage selection.

    """

    def __init__(
        self,
        macro_series_ids: list[str],
        vintage_base_dir: Path | str,
        fred_path: Path | str | None = None,
        include_revisions: bool = False,
        revision_mode: Literal["minimal", "core", "full"] = "core",
        lag_days: int = 1,
        vintage_policy: VintagePolicy = VintagePolicy.REAL_TIME,
    ) -> None:
        self.macro_series_ids = macro_series_ids
        self.vintage_base_dir = Path(vintage_base_dir).expanduser()
        self.fred_path = Path(fred_path).expanduser() if fred_path else None
        self.include_revisions = include_revisions
        self.revision_mode = revision_mode
        self.lag_days = lag_days
        self.vintage_policy = vintage_policy

        # Real-time cache (lazy-loaded)
        self._cache: MacroDataCache | None = None

    def _get_cache(self) -> MacroDataCache:
        """Get or create real-time cache."""
        if self._cache is None:
            self._cache = MacroDataCache(
                vintage_base_dir=self.vintage_base_dir,
                series_ids=self.macro_series_ids,
                enable_revisions=self.include_revisions,
            )
        return self._cache

    def compute_batch(
        self,
        df: PolarsDataFrame,
        timestamp_col: str = "timestamp",
        vintage_cutoff: Any = None,
    ) -> PolarsDataFrame:
        """
        Compute macro features for batch (historical) data.

        Uses join_fred_asof to apply point-in-time vintage logic.

        Parameters
        ----------
        df : pl.DataFrame
            Market data with timestamp column.
        timestamp_col : str, default "timestamp"
            Name of timestamp column.
        vintage_cutoff : datetime | None
            Cutoff date for vintage selection (for backtesting).

        Returns
        -------
        pl.DataFrame
            DataFrame with macro features added as columns.

        """
        from ml.data.fred_join import join_fred_asof

        result = join_fred_asof(
            df,
            timestamp_col=timestamp_col,
            lag_days=self.lag_days,
            fred_path=self.fred_path,
            vintage_base_dir=self.vintage_base_dir,
            series_filter=set(self.macro_series_ids),
            vintage_policy=self.vintage_policy,
            vintage_cutoff=vintage_cutoff,
            include_revisions=self.include_revisions,
            revision_mode=self.revision_mode,
            revision_windows=None,  # Use defaults
        )
        if pl is None:
            check_ml_dependencies(["polars"])  # pragma: no cover
        assert pl is not None
        if not isinstance(result, pl.DataFrame):
            raise TypeError("join_fred_asof must return a polars DataFrame for polars input")
        return cast(PolarsDataFrame, result)

    def compute_realtime(
        self,
        bar: Bar | None = None,
        ts_event: int | None = None,
    ) -> dict[str, float]:
        """
        Compute macro features for real-time inference.

        Uses cached latest values - no point-in-time filtering needed since we're
        always at "now".

        Parameters
        ----------
        bar : Bar | None
            Current bar (unused, kept for signature compatibility).
        ts_event : int | None
            Event timestamp (unused in real-time - always uses latest).

        Returns
        -------
        dict[str, float]
            Feature name → value mapping for all macro features.

        """
        cache = self._get_cache()

        if not cache.is_loaded():
            logger.warning("Macro cache not loaded, returning empty features")
            return {}

        # Get all features from cache (uses latest released values)
        return cache.get_all_features(mode=self.revision_mode)

    def get_feature_names(self) -> list[str]:
        """
        Get all feature names that will be produced.

        Returns
        -------
        list[str]
            Ordered list of feature names.

        """
        # Build feature names based on configuration
        feature_names: list[str] = []

        for series_id in self.macro_series_ids:
            # Base value (always included)
            feature_names.append(f"{series_id}__value_real_time")

            # Minimal mode features
            if self.revision_mode in ["minimal", "core", "full"]:
                feature_names.append(f"{series_id}_prior_1m")
                if self.include_revisions:
                    feature_names.append(f"{series_id}_revision_1m")

            # Core mode features
            if self.revision_mode in ["core", "full"]:
                feature_names.extend([
                    f"{series_id}_mom_1m",
                    f"{series_id}_pct_1m",
                ])
                if self.include_revisions:
                    feature_names.append(f"{series_id}_net_signal_1m")

            # Full mode features
            if self.revision_mode == "full":
                feature_names.extend([
                    f"{series_id}_prior_3m",
                    f"{series_id}_prior_12m",
                    f"{series_id}_mom_3m",
                    f"{series_id}_mom_12m",
                    f"{series_id}_pct_12m",
                ])
                if self.include_revisions:
                    feature_names.append(f"{series_id}_revision_3m")

        return feature_names

    def get_feature_dtypes(self) -> list[str]:
        """
        Get dtypes for all features.

        Returns
        -------
        list[str]
            List of dtype strings (all float64 for macro features).

        """
        return ["float64"] * len(self.get_feature_names())

    def refresh_cache(self) -> None:
        """
        Refresh the real-time cache with latest data.

        Call this periodically (e.g., daily) to pick up new FRED/ALFRED releases.
        """
        if self._cache is not None:
            logger.info("Refreshing macro feature cache")
            self._cache.refresh()

    def get_cache_coverage(self) -> dict[str, bool]:
        """
        Get cache coverage showing which series are available.

        Returns
        -------
        dict[str, bool]
            Series ID → available mapping.

        """
        cache = self._get_cache()
        return cache.get_coverage()

    def get_transform_config(self) -> dict[str, Any]:
        """
        Get transform configuration for serialization.

        Returns
        -------
        dict[str, Any]
            Configuration dictionary.

        """
        return {
            "transform_type": "macro_features",
            "macro_series_ids": self.macro_series_ids,
            "vintage_base_dir": str(self.vintage_base_dir),
            "fred_path": str(self.fred_path) if self.fred_path else None,
            "include_revisions": self.include_revisions,
            "revision_mode": self.revision_mode,
            "lag_days": self.lag_days,
            "vintage_policy": self.vintage_policy.value,
        }


def create_macro_transform_from_config(
    macro_series_ids: list[str] | tuple[str, ...] | None,
    vintage_base_dir: Path | str | None,
    fred_path: Path | str | None = None,
    include_revisions: bool = False,
    revision_mode: str = "core",
    lag_days: int = 1,
) -> MacroFeatureTransform | None:
    """
    Factory function to create MacroFeatureTransform from config.

    Parameters
    ----------
    macro_series_ids : list[str] | tuple[str, ...] | None
        Series to include (None or empty list disables macro features).
    vintage_base_dir : Path | str | None
        Vintage directory path.
    fred_path : Path | str | None
        FRED data path.
    include_revisions : bool
        Enable revision features.
    revision_mode : str
        Revision mode.
    lag_days : int
        Publication lag.

    Returns
    -------
    MacroFeatureTransform | None
        Transform instance, or None if macro features disabled.

    """
    if not macro_series_ids or vintage_base_dir is None:
        return None

    series_list = list(macro_series_ids) if isinstance(macro_series_ids, tuple) else macro_series_ids

    return MacroFeatureTransform(
        macro_series_ids=series_list,
        vintage_base_dir=vintage_base_dir,
        fred_path=fred_path,
        include_revisions=include_revisions,
        revision_mode=revision_mode,  # type: ignore[arg-type]
        lag_days=lag_days,
    )
