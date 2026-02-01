"""
Target generation component for TFT dataset building.

This component wraps the canonical TargetGenerator and preserves the legacy
binary target API while exposing explicit target semantics support.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

from ml.config.targets import TargetSemanticsConfig
from ml.training.datasets.target_generator import TargetGenerationResult
from ml.training.datasets.target_generator import TargetGenerator


if TYPE_CHECKING:
    import pandas as _pd
    import polars as _pl
else:  # pragma: no cover - typing fallback
    _pd = Any
    _pl = Any


logger = logging.getLogger(__name__)


class TargetGenerationComponent:
    """
    Component for generating targets for TFT models.

    Legacy binary targets remain available via `generate_targets_polars` and
    `generate_targets_pandas`, while `generate_targets_with_semantics` supports
    explicit multi-horizon target semantics.
    """

    def __init__(self) -> None:
        """Initialize target generation component."""
        self._generator = TargetGenerator()

    def generate_targets_polars(
        self,
        df: _pl.DataFrame,
        horizon_minutes: int,
        threshold: float,
    ) -> _pl.DataFrame:
        """
        Generate legacy binary targets using Polars.

        Args:
            df: Polars DataFrame with 'close' column containing price data.
            horizon_minutes: Number of periods to look ahead for target calculation.
            threshold: Minimum return threshold for positive classification (y=1).

        Returns:
            Polars DataFrame with target columns (legacy + explicit names).
        """
        config = TargetSemanticsConfig.from_legacy(
            horizon_minutes=horizon_minutes,
            threshold=threshold,
            legacy_aliases=True,
        )
        result = self._generator.generate_targets_with_semantics(
            df,
            config,
            use_polars=True,
        )
        return cast("_pl.DataFrame", result.frame)

    def generate_targets_pandas(
        self,
        df: _pd.DataFrame,
        horizon_minutes: int,
        threshold: float,
    ) -> _pd.DataFrame:
        """
        Generate legacy binary targets using Pandas.

        Args:
            df: Pandas DataFrame with 'close' column containing price data.
            horizon_minutes: Number of periods to look ahead for target calculation.
            threshold: Minimum return threshold for positive classification (y=1).

        Returns:
            Pandas DataFrame with target columns (legacy + explicit names).
        """
        config = TargetSemanticsConfig.from_legacy(
            horizon_minutes=horizon_minutes,
            threshold=threshold,
            legacy_aliases=True,
        )
        result = self._generator.generate_targets_with_semantics(
            df,
            config,
            use_polars=False,
        )
        return cast("_pd.DataFrame", result.frame)

    def generate_targets_with_semantics(
        self,
        df: Any,
        config: TargetSemanticsConfig,
        *,
        use_polars: bool = True,
    ) -> TargetGenerationResult:
        """
        Generate targets using explicit target semantics.

        Args:
            df: Input dataframe with price data.
            config: Target semantics configuration.
            use_polars: Whether to use Polars implementation.

        Returns:
            TargetGenerationResult containing target frame and metadata.
        """
        return self._generator.generate_targets_with_semantics(
            df,
            config,
            use_polars=use_polars,
        )


__all__ = [
    "TargetGenerationComponent",
]
