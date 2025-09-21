"""
Task helpers for Yahoo-style supplementary data generation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ml.data.loaders.supplementary import SUPPLEMENTARY_SYMBOLS
from ml.data.loaders.supplementary import SupplementaryDataConfig
from ml.data.loaders.supplementary import SupplementaryOutputs
from ml.data.loaders.supplementary import calculate_correlations
from ml.data.loaders.supplementary import calculate_spreads
from ml.data.loaders.supplementary import create_synthetic_supplementary_data
from ml.data.loaders.supplementary import write_supplementary_outputs


@dataclass(slots=True, frozen=True)
class PopulateYahooDataTaskConfig:
    """
    Arguments for :func:`populate_yahoo_data`.
    """

    output_dir: Path
    categories: Sequence[str] | None = None
    synthetic_years: int = 2


def _select_symbols(categories: Sequence[str] | None) -> tuple[str, ...]:
    if not categories:
        symbols: list[str] = []
        for value in SUPPLEMENTARY_SYMBOLS.values():
            symbols.extend(value)
        return tuple(symbols)
    invalid = [cat for cat in categories if cat not in SUPPLEMENTARY_SYMBOLS]
    if invalid:
        raise ValueError(f"Unknown Yahoo categories: {', '.join(invalid)}")
    selected: list[str] = []
    for category in categories:
        selected.extend(SUPPLEMENTARY_SYMBOLS[category])
    return tuple(selected)


def populate_yahoo_data(config: PopulateYahooDataTaskConfig) -> SupplementaryOutputs:
    """
    Generate Yahoo-style supplementary data and persist parquet outputs.
    """
    symbols = _select_symbols(config.categories)
    data_config = SupplementaryDataConfig(
        output_dir=config.output_dir,
        synthetic_years=config.synthetic_years,
    )
    data = create_synthetic_supplementary_data(data_config)
    data = data[data["symbol"].isin(symbols)].reset_index(drop=True)
    if data.empty:
        raise ValueError("No synthetic Yahoo data generated for requested categories")

    correlations = calculate_correlations(data, data_config.base_symbols)
    spreads = calculate_spreads(data, data_config.spread_definitions)
    return write_supplementary_outputs(data, correlations, spreads, data_config)


__all__ = [
    "PopulateYahooDataTaskConfig",
    "populate_yahoo_data",
]
