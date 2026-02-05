"""
Feature alignment component for TFT dataset building.

This component delegates OHLCV feature computation to the canonical pipeline batch
backend to keep training/inference parity and avoid drift. Static and macro-delta
enrichment are delegated to shared helpers for backward compatibility.

Guardrail: do not add ad-hoc feature math here. Register new transforms in
`ml/features/pipeline.py` and rely on the batch/stream executors instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from ml._imports import pd as pd_runtime
from ml._imports import pl as pl_runtime
from ml.data.common.feature_config_utils import normalize_feature_config
from ml.data.common.macro_deltas import append_macro_delta_features_polars as _append_macro_delta_features_polars
from ml.data.common.pipeline_batch import PipelineBatchContext
from ml.data.common.pipeline_batch import PipelineBatchExecutor
from ml.data.common.static_features import add_static_features_pandas as _add_static_features_pandas
from ml.data.common.static_features import add_static_features_polars as _add_static_features_polars
from ml.features.config import FeatureConfig
from ml.features.config import build_pipeline_spec_from_feature_config
from ml.features.pipeline import PipelineRunner
from ml.features.pipeline import PipelineSpec


if TYPE_CHECKING:
    import pandas as _pd
    import polars as _pl
else:  # pragma: no cover - typing fallback
    _pd = Any
    _pl = Any


# Runtime aliases
pl: Any = cast(Any, pl_runtime)
pd: Any = cast(Any, pd_runtime)


_OHLCV_TRANSFORMS: set[str] = {
    "returns",
    "momentum",
    "volatility",
    "volume_ratio",
    "core_indicators",
    "microstructure",
    "trade_flow",
}


class FeatureAlignmentComponent:
    """
    Component for feature computation and alignment for TFT datasets.

    This component provides methods for:
    - Computing canonical OHLCV features via the pipeline batch backend
    - Adding static instrument features (asset class, tick size, exchange)
    - Supporting both Polars and Pandas DataFrames with identical outputs

    """

    def __init__(self, *, feature_config: FeatureConfig | None = None) -> None:
        """
        Initialize the feature alignment component.

        Args:
            feature_config: Optional FeatureConfig for canonical batch execution.
                Defaults to FeatureConfig() when omitted.

        """
        self._feature_config = normalize_feature_config(feature_config)
        self._ohlcv_spec = self._build_ohlcv_spec(self._feature_config)

    @staticmethod
    def _build_ohlcv_spec(config: FeatureConfig) -> PipelineSpec:
        full_spec = build_pipeline_spec_from_feature_config(config)
        ohlcv_transforms = [ts for ts in full_spec.transforms if ts.name in _OHLCV_TRANSFORMS]
        return PipelineSpec(transforms=ohlcv_transforms)

    def compute_features_canonical_polars(self, df: _pl.DataFrame) -> _pl.DataFrame:
        """
        Compute canonical OHLCV features using the pipeline batch backend (Polars).

        Args:
            df: Polars DataFrame with OHLCV columns.

        Returns:
            Polars DataFrame with canonical feature columns.
        """
        if pl is None:
            raise RuntimeError("Polars is required for canonical feature computation")

        allowable = self._feature_config.resolved_data_requirements()
        runner = PipelineRunner(self._ohlcv_spec, allowable=allowable)
        feature_names = runner.compute_feature_names()
        if df.is_empty():
            empty_df: _pl.DataFrame = pl.DataFrame({name: [] for name in feature_names})
            return empty_df

        executor = PipelineBatchExecutor(
            self._ohlcv_spec,
            allowable=allowable,
            context=PipelineBatchContext(feature_config=self._feature_config),
        )
        out = executor.execute_polars(df)
        return out.select(feature_names)

    def compute_features_canonical_pandas(self, df: _pd.DataFrame) -> _pd.DataFrame:
        """
        Compute canonical OHLCV features using the pipeline batch backend (Pandas).

        Args:
            df: Pandas DataFrame with OHLCV columns.

        Returns:
            Pandas DataFrame with canonical feature columns.
        """
        if pd is None:
            raise RuntimeError("Pandas is required for canonical feature computation")

        allowable = self._feature_config.resolved_data_requirements()
        runner = PipelineRunner(self._ohlcv_spec, allowable=allowable)
        feature_names = runner.compute_feature_names()
        if len(df) == 0:
            empty_frame = cast(
                "_pd.DataFrame",
                pd.DataFrame(
                    {name: pd.Series([], dtype=float) for name in feature_names},
                ),
            )
            return empty_frame

        executor = PipelineBatchExecutor(
            self._ohlcv_spec,
            allowable=allowable,
            context=PipelineBatchContext(feature_config=self._feature_config),
        )
        out = executor.execute_pandas(df)
        return out[feature_names]

    def compute_features_polars(self, df: _pl.DataFrame) -> _pl.DataFrame:
        """
        Compute canonical OHLCV features using Polars.

        This is a compatibility wrapper around ``compute_features_canonical_polars``.
        """
        return self.compute_features_canonical_polars(df)

    def compute_features_pandas(self, df: _pd.DataFrame) -> _pd.DataFrame:
        """
        Compute canonical OHLCV features using Pandas.

        This is a compatibility wrapper around ``compute_features_canonical_pandas``.
        """
        return self.compute_features_canonical_pandas(df)

    def add_static_features_polars(self, df: _pl.DataFrame) -> _pl.DataFrame:
        """
        Add static instrument features using Polars.

        Adds asset_class, tick_size, and exchange columns based on the
        instrument_id column. Uses default values for unknown symbols.

        Args:
            df: Polars DataFrame with instrument_id column

        Returns:
            DataFrame with added static feature columns.

        Raises:
            ValueError: If instrument_id column is missing

        Example:
            >>> df = pl.DataFrame({
            ...     "instrument_id": ["SPY", "SPY"],
            ...     "close": [450.0, 451.0],
            ... })
            >>> result = component.add_static_features_polars(df)
            >>> assert result["asset_class"][0] == "ETF"
            >>> assert result["exchange"][0] == "ARCA"

        """
        return _add_static_features_polars(df)

    def add_static_features_pandas(self, df: _pd.DataFrame) -> _pd.DataFrame:
        """
        Add static instrument features using Pandas.

        Adds the same static features as the Polars implementation.

        Args:
            df: Pandas DataFrame with instrument_id column

        Returns:
            DataFrame with added static feature columns.

        Raises:
            ValueError: If instrument_id column is missing

        Example:
            >>> df = pd.DataFrame({
            ...     "instrument_id": ["SPY", "SPY"],
            ...     "close": [450.0, 451.0],
            ... })
            >>> result = component.add_static_features_pandas(df)
            >>> assert result["asset_class"].iloc[0] == "ETF"

        """
        return _add_static_features_pandas(df)

    def append_macro_delta_features_polars(
        self,
        df: _pl.DataFrame,
        *,
        include_macro: bool,
        include_macro_deltas: bool,
        macro_series_ids: tuple[str, ...] | None,
    ) -> _pl.DataFrame:
        """
        Append 1-day delta features for configured macro series columns.

        The deltas are computed per-instrument, ordered by the timestamp column
        (``timestamp`` or ``ts_event``). The first delta in each instrument group
        is filled with ``0.0``.

        Args:
            df: Polars DataFrame containing macro series columns.
            include_macro: Whether macro features are enabled.
            include_macro_deltas: Whether macro delta features are enabled.
            macro_series_ids: Macro series identifiers to compute deltas for.

        Returns:
            DataFrame with appended ``*_delta_1d`` columns when enabled.

        Example:
            >>> df = pl.DataFrame({"timestamp": [1, 2], "PAYEMS": [100.0, 101.0]})
            >>> comp = FeatureAlignmentComponent()
            >>> out = comp.append_macro_delta_features_polars(
            ...     df,
            ...     include_macro=True,
            ...     include_macro_deltas=True,
            ...     macro_series_ids=("PAYEMS",),
            ... )
            >>> assert "PAYEMS_delta_1d" in out.columns

        """
        return _append_macro_delta_features_polars(
            df,
            include_macro=include_macro,
            include_macro_deltas=include_macro_deltas,
            macro_series_ids=macro_series_ids,
        )


__all__ = ["FeatureAlignmentComponent"]
