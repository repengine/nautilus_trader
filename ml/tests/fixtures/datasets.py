#!/usr/bin/env python3
"""
Dataset builder fixtures that provide deterministic sample data.

These helpers replace bespoke inline stubs across tests to guarantee that
dataset builders receive non-empty inputs without touching external sources.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import Callable, Final, Iterable, Sequence

import polars as pl
import pytest

DEFAULT_DATASET_BAR_MODULES: Final[tuple[str, str]] = (
    "ml.data.catalog_utils",
    "ml.data.tft_dataset_builder",
)


@dataclass(frozen=True, slots=True)
class SampleBarSeriesConfig:
    """Configuration for generating sample bar data."""

    instrument_id: str = "SPY"
    start: datetime = datetime(2025, 1, 1, 9, 30, tzinfo=UTC)
    rows: int = 10
    freq_minutes: int = 1


def build_sample_bars(config: SampleBarSeriesConfig) -> pl.DataFrame:
    """
    Generate a deterministic set of bar data for dataset builder tests.
    """

    timestamps: list[datetime] = [
        config.start + timedelta(minutes=config.freq_minutes * offset) for offset in range(config.rows)
    ]
    open_prices = [100.0 + 0.05 * idx for idx in range(config.rows)]
    close_prices = [op + 0.02 for op in open_prices]
    high_prices = [cp + 0.03 for cp in close_prices]
    low_prices = [op - 0.03 for op in open_prices]
    volumes = [1000 + 25 * idx for idx in range(config.rows)]

    return pl.DataFrame(
        {
            "instrument_id": [config.instrument_id] * config.rows,
            "timestamp": timestamps,
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
            "volume": volumes,
        },
    )


@pytest.fixture
def sample_bars_dataframe_factory() -> Callable[[SampleBarSeriesConfig | None], pl.DataFrame]:
    """
    Fixture returning a factory that creates deterministic bar dataframes.
    """

    def _factory(config: SampleBarSeriesConfig | None = None) -> pl.DataFrame:
        return build_sample_bars(config or SampleBarSeriesConfig())

    return _factory


@pytest.fixture
def sample_bar_series_config_cls() -> type[SampleBarSeriesConfig]:
    """Expose the SampleBarSeriesConfig dataclass via fixture injection."""

    return SampleBarSeriesConfig


@pytest.fixture
def sample_bar_series_config_factory(
    sample_bar_series_config_cls: type[SampleBarSeriesConfig],
) -> Callable[..., SampleBarSeriesConfig]:
    """
    Provide a factory for creating SampleBarSeriesConfig instances with overrides.
    """

    def _factory(**overrides: object) -> SampleBarSeriesConfig:
        return replace(sample_bar_series_config_cls(), **overrides)

    return _factory


@pytest.fixture
def patch_bars_to_dataframe(
    sample_bars_dataframe_factory: Callable[[SampleBarSeriesConfig | None], pl.DataFrame],
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[str, SampleBarSeriesConfig | None], None]:
    """
    Fixture providing a helper to patch ``bars_to_dataframe`` for a module.

    Usage::

        def test_something(patch_bars_to_dataframe):
            patch_bars_to_dataframe("ml.data.tft_dataset_builder")
            ...
    """

    def _patch(module_path: str, config: SampleBarSeriesConfig | None = None) -> None:
        import importlib

        module = importlib.import_module(module_path)

        def _stub(
            _catalog: object,
            instrument_ids: Iterable[str],
            start: datetime | None = None,
            end: datetime | None = None,
        ) -> pl.DataFrame:
            del _catalog, start, end
            instrument = next(iter(instrument_ids), (config or SampleBarSeriesConfig()).instrument_id)
            actual_config = SampleBarSeriesConfig(
                instrument_id=instrument,
                start=(config.start if config else SampleBarSeriesConfig().start),
                rows=(config.rows if config else SampleBarSeriesConfig().rows),
                freq_minutes=(config.freq_minutes if config else SampleBarSeriesConfig().freq_minutes),
            )
            return sample_bars_dataframe_factory(actual_config)

        monkeypatch.setattr(module, "bars_to_dataframe", _stub)

    return _patch


@pytest.fixture
def patch_dataset_bars(
    patch_bars_to_dataframe: Callable[[str, SampleBarSeriesConfig | None], None],
    sample_bar_series_config_factory: Callable[..., SampleBarSeriesConfig],
) -> Callable[[Sequence[str] | None, SampleBarSeriesConfig | None], SampleBarSeriesConfig]:
    """
    Patch ``bars_to_dataframe`` for multiple dataset modules at once.

    Args:
        modules: Optional list of module import paths to patch. Defaults to
            ``ml.data.catalog_utils`` and ``ml.data.tft_dataset_builder``.
        config: Optional ``SampleBarSeriesConfig`` instance to drive the patch.

    Returns:
        The resolved ``SampleBarSeriesConfig`` used for the patched modules.
    """

    def _patch(
        modules: Sequence[str] | None = None,
        config: SampleBarSeriesConfig | None = None,
    ) -> SampleBarSeriesConfig:
        resolved = config or sample_bar_series_config_factory()
        targets = tuple(modules or DEFAULT_DATASET_BAR_MODULES)
        for module_path in targets:
            patch_bars_to_dataframe(module_path, resolved)
        return resolved

    return _patch


__all__ = [
    "SampleBarSeriesConfig",
    "build_sample_bars",
    "patch_bars_to_dataframe",
    "patch_dataset_bars",
    "sample_bar_series_config_cls",
    "sample_bar_series_config_factory",
    "sample_bars_dataframe_factory",
]
