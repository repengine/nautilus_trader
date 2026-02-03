"""
Target generation component for TFT dataset building.

This component wraps the canonical TargetGenerator and exposes explicit target
semantics support for both Polars and Pandas inputs.
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

    Target generation is driven by explicit target semantics configuration.
    """

    def __init__(self) -> None:
        """Initialize target generation component."""
        self._generator = TargetGenerator()

    def generate_targets_polars(
        self,
        df: _pl.DataFrame,
        config: TargetSemanticsConfig,
    ) -> _pl.DataFrame:
        """
        Generate targets using Polars.

        Args:
            df: Polars DataFrame with 'close' column containing price data.
            config: Target semantics configuration.

        Returns:
            Polars DataFrame with target columns.
        """
        result = self._generator.generate_targets_with_semantics(
            df,
            config,
            use_polars=True,
        )
        return cast("_pl.DataFrame", result.frame)

    def generate_targets_pandas(
        self,
        df: _pd.DataFrame,
        config: TargetSemanticsConfig,
    ) -> _pd.DataFrame:
        """
        Generate targets using Pandas.

        Args:
            df: Pandas DataFrame with 'close' column containing price data.
            config: Target semantics configuration.

        Returns:
            Pandas DataFrame with target columns.
        """
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
