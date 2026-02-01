"""
Target generation for training datasets.

This module provides dual Polars/Pandas implementations for generating forward
returns and target labels across multiple horizons, with optional cost-aware
returns and multiclass/regression variants.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from ml.config.targets import TargetSemanticsConfig
from ml.config.targets import build_binary_target_column
from ml.config.targets import build_cost_return_column
from ml.config.targets import build_forward_return_column
from ml.config.targets import build_multiclass_target_column
from ml.config.targets import build_regression_target_column


if TYPE_CHECKING:
    import pandas as pd
    import polars as pl
else:
    pd = Any
    pl = Any


logger = logging.getLogger(__name__)


# ========================================================================
# Protocol Definition
# ========================================================================


class TargetGeneratorProtocol(Protocol):
    """Protocol for target generation operations."""

    def generate_targets_polars(
        self,
        df: Any,
        horizon_minutes: int,
        threshold: float,
    ) -> Any:
        """
        Generate legacy binary targets using Polars.

        Parameters
        ----------
        df : polars.DataFrame
            Input dataframe
        horizon_minutes : int
            Horizon in minutes
        threshold : float
            Return threshold for binary classification

        Returns
        -------
        polars.DataFrame
            DataFrame with target columns

        """
        ...

    def generate_targets_pandas(
        self,
        df: Any,
        horizon_minutes: int,
        threshold: float,
    ) -> Any:
        """
        Generate legacy binary targets using Pandas.

        Parameters
        ----------
        df : pandas.DataFrame
            Input dataframe
        horizon_minutes : int
            Horizon in minutes
        threshold : float
            Return threshold for binary classification

        Returns
        -------
        pandas.DataFrame
            DataFrame with target columns

        """
        ...

    def generate_targets(
        self,
        df: Any,
        horizon_minutes: int,
        threshold: float,
        use_polars: bool = True,
    ) -> Any:
        """
        Generate legacy binary targets using the specified implementation.

        Parameters
        ----------
        df : polars.DataFrame | pandas.DataFrame
            Input dataframe with price data
        horizon_minutes : int
            Forward-looking horizon in minutes
        threshold : float
            Binary classification threshold (e.g., 0.001 = 10bps)
        use_polars : bool
            Use Polars (True) or Pandas (False) implementation

        Returns
        -------
        polars.DataFrame | pandas.DataFrame
            DataFrame with target columns added

        """
        ...

    def generate_targets_with_semantics(
        self,
        df: Any,
        config: TargetSemanticsConfig,
        use_polars: bool = True,
        *,
        include_input: bool = False,
    ) -> TargetGenerationResult:
        """
        Generate multi-horizon targets using explicit target semantics.

        Parameters
        ----------
        df : polars.DataFrame | pandas.DataFrame
            Input dataframe with price data
        config : TargetSemanticsConfig
            Target semantics configuration
        use_polars : bool
            Use Polars (True) or Pandas (False) implementation
        include_input : bool
            If True, append target columns to the input dataframe

        Returns
        -------
        TargetGenerationResult
            Target generation output and metadata

        """
        ...


@dataclass(frozen=True)
class TargetGenerationResult:
    """
    Result container for target generation.

    Attributes
    ----------
    frame : Any
        DataFrame containing generated targets (and optional input columns).
    semantics : dict[str, Any]
        Target semantics metadata dictionary.
    return_columns : tuple[str, ...]
        Return column names generated.
    label_columns : tuple[str, ...]
        Target label column names generated.
    primary_target : str | None
        Primary target column name, if resolved.
    """

    frame: Any
    semantics: dict[str, Any]
    return_columns: tuple[str, ...]
    label_columns: tuple[str, ...]
    primary_target: str | None


def build_target_semantics_metadata(config: TargetSemanticsConfig) -> dict[str, Any]:
    """
    Build target semantics metadata from configuration.

    Args:
        config: Target semantics configuration.

    Returns:
        Target semantics metadata dictionary.
    """
    horizons: list[dict[str, Any]] = []
    returns: dict[str, Any] = {}
    labels: dict[str, Any] = {}

    emit_cost = config.should_emit_cost_return()
    for spec in config.horizons:
        label = spec.label or f"{spec.minutes}m"
        horizons.append({"label": label, "minutes": int(spec.minutes)})
        forward_col = build_forward_return_column(label)
        returns[forward_col] = {
            "horizon_minutes": int(spec.minutes),
            "basis": "raw",
        }
        if emit_cost:
            cost_col = build_cost_return_column(label)
            cost_entry: dict[str, Any] = {
                "horizon_minutes": int(spec.minutes),
                "basis": "net",
            }
            if config.cost_model is not None:
                cost_entry["cost_model"] = config.cost_model.as_metadata()
            returns[cost_col] = cost_entry

    if config.binary.enabled:
        threshold = config.binary.threshold
        for spec in config.horizons:
            label = spec.label or f"{spec.minutes}m"
            target_col = build_binary_target_column(label)
            return_col = config.return_column_for_basis(label, config.binary.return_basis)
            labels[target_col] = {
                "type": "binary",
                "return_col": return_col,
                "threshold": threshold,
                "threshold_bps": float(config.binary.threshold_bps),
                "classes": {"0": "neutral", "1": "long"},
            }

    if config.multiclass.enabled:
        short_threshold = config.multiclass.short_threshold
        long_threshold = config.multiclass.long_threshold
        for spec in config.horizons:
            label = spec.label or f"{spec.minutes}m"
            target_col = build_multiclass_target_column(label)
            return_col = config.return_column_for_basis(label, config.multiclass.return_basis)
            labels[target_col] = {
                "type": "multiclass",
                "return_col": return_col,
                "thresholds": {"short": short_threshold, "long": long_threshold},
                "thresholds_bps": {
                    "short": float(config.multiclass.short_threshold_bps),
                    "long": float(config.multiclass.long_threshold_bps),
                },
                "classes": {"-1": "short", "0": "neutral", "1": "long"},
            }

    if config.regression.enabled:
        for spec in config.horizons:
            label = spec.label or f"{spec.minutes}m"
            target_col = build_regression_target_column(label)
            return_col = config.return_column_for_basis(label, config.regression.return_basis)
            labels[target_col] = {
                "type": "regression",
                "return_col": return_col,
            }

    semantics: dict[str, Any] = {
        "version": config.version,
        "horizons": horizons,
        "returns": returns,
        "labels": labels,
    }

    primary = config.resolved_primary_target()
    if primary is not None:
        semantics["primary_target"] = primary

    if config.legacy_aliases and config.horizons:
        first_label = config.horizons[0].label or f"{config.horizons[0].minutes}m"
        legacy: dict[str, str] = {
            "forward_return": build_forward_return_column(first_label),
        }
        if config.binary.enabled:
            legacy["y"] = build_binary_target_column(first_label)
        semantics["legacy_aliases"] = legacy

    return semantics


# ========================================================================
# TargetGenerator Implementation
# ========================================================================


class TargetGenerator:
    """
    Generates prediction targets for training datasets.

    Supports forward returns, cost-aware returns, and binary/multiclass/regression
    labels across multiple horizons.
    """

    def __init__(self) -> None:
        """Initialize target generator."""
        logger.debug("TargetGenerator initialized")

    def generate_targets(
        self,
        df: Any,
        horizon_minutes: int,
        threshold: float,
        use_polars: bool = True,
    ) -> Any:
        """
        Generate legacy binary targets.

        Parameters
        ----------
        df : polars.DataFrame | pandas.DataFrame
            Input dataframe with price data
        horizon_minutes : int
            Forward-looking horizon in minutes
        threshold : float
            Binary classification threshold (e.g., 0.001 = 10bps)
        use_polars : bool
            Use Polars (True) or Pandas (False) implementation

        Returns
        -------
        polars.DataFrame | pandas.DataFrame
            DataFrame with target columns added

        """
        config = TargetSemanticsConfig.from_legacy(
            horizon_minutes=horizon_minutes,
            threshold=threshold,
            legacy_aliases=True,
        )
        result = self.generate_targets_with_semantics(
            df,
            config,
            use_polars=use_polars,
            include_input=True,
        )
        return result.frame

    def generate_targets_with_semantics(
        self,
        df: Any,
        config: TargetSemanticsConfig,
        use_polars: bool = True,
        *,
        include_input: bool = False,
    ) -> TargetGenerationResult:
        """
        Generate multi-horizon targets using explicit target semantics.

        Parameters
        ----------
        df : polars.DataFrame | pandas.DataFrame
            Input dataframe with price data
        config : TargetSemanticsConfig
            Target semantics configuration
        use_polars : bool
            Use Polars (True) or Pandas (False) implementation
        include_input : bool
            If True, append target columns to the input dataframe

        Returns
        -------
        TargetGenerationResult
            Target generation output and metadata

        """
        if use_polars:
            targets, return_cols, label_cols = self._generate_targets_polars_with_config(
                df,
                config,
            )
        else:
            targets, return_cols, label_cols = self._generate_targets_pandas_with_config(
                df,
                config,
            )

        if include_input:
            targets = self._combine_frames(df, targets, use_polars=use_polars)

        semantics = build_target_semantics_metadata(config)
        return TargetGenerationResult(
            frame=targets,
            semantics=semantics,
            return_columns=return_cols,
            label_columns=label_cols,
            primary_target=config.resolved_primary_target(),
        )

    def generate_targets_polars(
        self,
        df: Any,
        horizon_minutes: int,
        threshold: float,
    ) -> Any:
        """
        Generate legacy binary targets using Polars.

        Parameters
        ----------
        df : polars.DataFrame
            Input dataframe with 'close' column
        horizon_minutes : int
            Horizon in minutes (how many periods to look ahead)
        threshold : float
            Return threshold for binary classification

        Returns
        -------
        polars.DataFrame
            Input dataframe with added 'y' and 'forward_return' columns

        """
        config = TargetSemanticsConfig.from_legacy(
            horizon_minutes=horizon_minutes,
            threshold=threshold,
            legacy_aliases=True,
        )
        result = self.generate_targets_with_semantics(
            df,
            config,
            use_polars=True,
            include_input=True,
        )
        return result.frame

    def generate_targets_pandas(
        self,
        df: Any,
        horizon_minutes: int,
        threshold: float,
    ) -> Any:
        """
        Generate legacy binary targets using Pandas.

        Parameters
        ----------
        df : pandas.DataFrame
            Input dataframe with 'close' column
        horizon_minutes : int
            Horizon in minutes (how many periods to look ahead)
        threshold : float
            Return threshold for binary classification

        Returns
        -------
        pandas.DataFrame
            Input dataframe with added 'y' and 'forward_return' columns

        """
        config = TargetSemanticsConfig.from_legacy(
            horizon_minutes=horizon_minutes,
            threshold=threshold,
            legacy_aliases=True,
        )
        result = self.generate_targets_with_semantics(
            df,
            config,
            use_polars=False,
            include_input=True,
        )
        return result.frame

    def _combine_frames(
        self,
        df: Any,
        targets: Any,
        *,
        use_polars: bool,
    ) -> Any:
        """
        Combine input and target dataframes.
        """
        if use_polars:
            try:
                import polars as pl
            except ImportError as e:  # pragma: no cover - dependency guard
                msg = "Polars is required for TargetGenerator but not installed"
                raise ImportError(msg) from e
            if isinstance(df, pl.DataFrame) and isinstance(targets, pl.DataFrame):
                return pl.concat([df, targets], how="horizontal")
            raise TypeError(f"Expected Polars DataFrame, got {type(df)} and {type(targets)}")

        try:
            import pandas as pd
        except ImportError as e:  # pragma: no cover - dependency guard
            msg = "Pandas is required for TargetGenerator but not installed"
            raise ImportError(msg) from e
        if isinstance(df, pd.DataFrame) and isinstance(targets, pd.DataFrame):
            return pd.concat([df.reset_index(drop=True), targets.reset_index(drop=True)], axis=1)
        raise TypeError(f"Expected Pandas DataFrame, got {type(df)} and {type(targets)}")

    def _generate_targets_polars_with_config(
        self,
        df: Any,
        config: TargetSemanticsConfig,
    ) -> tuple[Any, tuple[str, ...], tuple[str, ...]]:
        """
        Generate targets using Polars with target semantics.
        """
        try:
            import polars as pl
        except ImportError as e:
            msg = "Polars is required for TargetGenerator but not installed"
            raise ImportError(msg) from e

        if not isinstance(df, pl.DataFrame):
            raise TypeError(f"Expected Polars DataFrame, got {type(df)}")
        if "close" not in df.columns:
            raise KeyError("Missing required 'close' column for target generation")

        horizon_labels = config.horizon_labels
        return_cols = list(build_forward_return_column(label) for label in horizon_labels)
        if config.should_emit_cost_return():
            return_cols.extend(build_cost_return_column(label) for label in horizon_labels)
        label_cols = list(config.label_columns())

        if df.is_empty():
            empty: dict[str, Any] = {}
            for name in return_cols:
                empty[name] = pl.Series([], dtype=pl.Float32)
            for name in label_cols:
                dtype: Any = pl.Float32
                if name.startswith("target_bin_") or name.startswith("target_class_"):
                    dtype = pl.Int32
                empty[name] = pl.Series([], dtype=dtype)
            if config.legacy_aliases and horizon_labels:
                empty["forward_return"] = pl.Series([], dtype=pl.Float32)
                if config.binary.enabled:
                    empty["y"] = pl.Series([], dtype=pl.Int32)
            return pl.DataFrame(empty), tuple(return_cols), tuple(label_cols)

        future_return_exprs: list[Any] = []
        for spec in config.horizons:
            label = spec.label or f"{spec.minutes}m"
            future_prices = pl.col("close").shift(-int(spec.minutes))
            current_prices = pl.col("close")
            forward_returns = (future_prices - current_prices) / current_prices
            future_return_exprs.append(
                forward_returns.cast(pl.Float32).alias(build_forward_return_column(label)),
            )

        targets = df.select(future_return_exprs)

        # Sanitize forward returns
        targets = targets.with_columns(
            [
                pl.when(pl.col(name).is_infinite() | pl.col(name).is_nan())
                .then(0.0)
                .otherwise(pl.col(name))
                .cast(pl.Float32)
                .alias(name)
                for name in list(build_forward_return_column(label) for label in horizon_labels)
            ],
        ).with_columns(
            [
                pl.col(name).fill_null(0.0).alias(name)
                for name in list(build_forward_return_column(label) for label in horizon_labels)
            ],
        )

        if config.should_emit_cost_return():
            cost_decimal = (
                config.cost_model.round_trip_decimal if config.cost_model is not None else 0.0
            )
            cost_exprs: list[Any] = []
            for label in horizon_labels:
                forward_col = build_forward_return_column(label)
                cost_col = build_cost_return_column(label)
                cost_exprs.append((pl.col(forward_col) - cost_decimal).cast(pl.Float32).alias(cost_col))
            targets = targets.with_columns(cost_exprs).with_columns(
                [
                    pl.when(pl.col(name).is_infinite() | pl.col(name).is_nan())
                    .then(0.0)
                    .otherwise(pl.col(name))
                    .cast(pl.Float32)
                    .alias(name)
                    for name in list(build_cost_return_column(label) for label in horizon_labels)
                ],
            ).with_columns(
                [
                    pl.col(name).fill_null(0.0).alias(name)
                    for name in list(build_cost_return_column(label) for label in horizon_labels)
                ],
            )

        label_exprs: list[Any] = []
        if config.binary.enabled:
            threshold = config.binary.threshold
            for label in horizon_labels:
                return_col = config.return_column_for_basis(label, config.binary.return_basis)
                label_exprs.append(
                    (pl.col(return_col) > threshold)
                    .cast(pl.Int32)
                    .alias(build_binary_target_column(label)),
                )

        if config.multiclass.enabled:
            short_threshold = config.multiclass.short_threshold
            long_threshold = config.multiclass.long_threshold
            for label in horizon_labels:
                return_col = config.return_column_for_basis(label, config.multiclass.return_basis)
                label_exprs.append(
                    pl.when(pl.col(return_col) > long_threshold)
                    .then(1)
                    .when(pl.col(return_col) < short_threshold)
                    .then(-1)
                    .otherwise(0)
                    .cast(pl.Int32)
                    .alias(build_multiclass_target_column(label)),
                )

        if config.regression.enabled:
            for label in horizon_labels:
                return_col = config.return_column_for_basis(label, config.regression.return_basis)
                label_exprs.append(
                    pl.col(return_col)
                    .cast(pl.Float32)
                    .alias(build_regression_target_column(label)),
                )

        if label_exprs:
            targets = targets.with_columns(label_exprs)

        if config.legacy_aliases and horizon_labels:
            first_label = horizon_labels[0]
            alias_exprs: list[Any] = [
                pl.col(build_forward_return_column(first_label)).alias("forward_return"),
            ]
            if config.binary.enabled:
                alias_exprs.append(pl.col(build_binary_target_column(first_label)).alias("y"))
            targets = targets.with_columns(alias_exprs)

        return targets, tuple(return_cols), tuple(label_cols)

    def _generate_targets_pandas_with_config(
        self,
        df: Any,
        config: TargetSemanticsConfig,
    ) -> tuple[Any, tuple[str, ...], tuple[str, ...]]:
        """
        Generate targets using Pandas with target semantics.
        """
        try:
            import pandas as pd
        except ImportError as e:
            msg = "Pandas is required for TargetGenerator but not installed"
            raise ImportError(msg) from e

        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected Pandas DataFrame, got {type(df)}")
        if "close" not in df.columns:
            raise KeyError("Missing required 'close' column for target generation")

        horizon_labels = config.horizon_labels
        return_cols = list(build_forward_return_column(label) for label in horizon_labels)
        if config.should_emit_cost_return():
            return_cols.extend(build_cost_return_column(label) for label in horizon_labels)
        label_cols = list(config.label_columns())

        if len(df) == 0:
            empty = {name: pd.Series([], dtype=float) for name in return_cols}
            for name in label_cols:
                dtype: type[float] | type[int] = float
                if name.startswith("target_bin_") or name.startswith("target_class_"):
                    dtype = int
                empty[name] = pd.Series([], dtype=dtype)
            if config.legacy_aliases and horizon_labels:
                empty["forward_return"] = pd.Series([], dtype=float)
                if config.binary.enabled:
                    empty["y"] = pd.Series([], dtype=int)
            return pd.DataFrame(empty), tuple(return_cols), tuple(label_cols)

        close = df["close"]
        data: dict[str, Any] = {}
        for spec in config.horizons:
            label = spec.label or f"{spec.minutes}m"
            forward = (close.shift(-int(spec.minutes)) - close) / close
            forward = forward.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(float)
            data[build_forward_return_column(label)] = forward

        if config.should_emit_cost_return():
            cost_decimal = (
                config.cost_model.round_trip_decimal if config.cost_model is not None else 0.0
            )
            for label in horizon_labels:
                forward_col = build_forward_return_column(label)
                cost_series = data[forward_col] - cost_decimal
                cost_series = cost_series.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(float)
                data[build_cost_return_column(label)] = cost_series

        if config.binary.enabled:
            threshold = config.binary.threshold
            for label in horizon_labels:
                return_col = config.return_column_for_basis(label, config.binary.return_basis)
                data[build_binary_target_column(label)] = (data[return_col] > threshold).astype(int)

        if config.multiclass.enabled:
            short_threshold = config.multiclass.short_threshold
            long_threshold = config.multiclass.long_threshold
            for label in horizon_labels:
                return_col = config.return_column_for_basis(label, config.multiclass.return_basis)
                series = data[return_col]
                data[build_multiclass_target_column(label)] = np.select(
                    [series > long_threshold, series < short_threshold],
                    [1, -1],
                    default=0,
                ).astype(int)

        if config.regression.enabled:
            for label in horizon_labels:
                return_col = config.return_column_for_basis(label, config.regression.return_basis)
                data[build_regression_target_column(label)] = data[return_col].astype(float)

        if config.legacy_aliases and horizon_labels:
            first_label = horizon_labels[0]
            data["forward_return"] = data[build_forward_return_column(first_label)]
            if config.binary.enabled:
                data["y"] = data[build_binary_target_column(first_label)]

        return pd.DataFrame(data), tuple(return_cols), tuple(label_cols)
