"""
Task orchestration for supplementary data population.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ml.data.loaders.supplementary import DEFAULT_BASE_SYMBOLS
from ml.data.loaders.supplementary import SupplementaryDataConfig
from ml.data.loaders.supplementary import SupplementaryOutputs
from ml.data.loaders.supplementary import calculate_correlations
from ml.data.loaders.supplementary import calculate_spreads
from ml.data.loaders.supplementary import create_synthetic_supplementary_data
from ml.data.loaders.supplementary import write_supplementary_outputs


@dataclass(slots=True, frozen=True)
class PopulateSupplementaryTaskConfig:
    """
    Arguments accepted by :func:`populate_supplementary_data`.
    """

    output_dir: Path
    base_symbols: tuple[str, ...] = DEFAULT_BASE_SYMBOLS
    synthetic_years: int = 2


def populate_supplementary_data(config: PopulateSupplementaryTaskConfig) -> SupplementaryOutputs:
    """
    Generate synthetic supplementary data and persist parquet outputs.
    """
    data_config = SupplementaryDataConfig(
        output_dir=config.output_dir,
        base_symbols=config.base_symbols,
        synthetic_years=config.synthetic_years,
    )
    data = create_synthetic_supplementary_data(data_config)
    if data.empty:
        raise ValueError("Supplementary data generation produced no rows")

    correlations = calculate_correlations(data, data_config.base_symbols)
    spreads = calculate_spreads(data, data_config.spread_definitions)
    outputs = write_supplementary_outputs(data, correlations, spreads, data_config)
    return outputs


__all__ = [
    "PopulateSupplementaryTaskConfig",
    "populate_supplementary_data",
]
