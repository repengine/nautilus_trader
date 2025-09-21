"""
Task entry points for alternative data population.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ml.data.loaders.alternative import AlternativeDataConfig
from ml.data.loaders.alternative import AlternativeDataResult
from ml.data.loaders.alternative import AlternativeSource
from ml.data.loaders.alternative import load_tier1_symbols
from ml.data.loaders.alternative import populate_alternative_data
from ml.data.loaders.alternative import save_alternative_data


@dataclass(slots=True, frozen=True)
class PopulateAlternativeDataTaskConfig:
    """
    Arguments accepted by :func:`populate_alternative_data_task`.
    """

    output_dir: Path
    symbols: Sequence[str] | None = None
    sources: Sequence[str] | None = None
    populate_all: bool = False
    tier1_progress_path: Path | None = None


def _resolve_sources(config: PopulateAlternativeDataTaskConfig) -> tuple[AlternativeSource, ...]:
    if config.populate_all:
        return tuple(AlternativeSource)
    if not config.sources:
        raise ValueError("No sources specified; pass populate_all=True or provide --source")
    resolved: list[AlternativeSource] = []
    for raw in config.sources:
        try:
            resolved.append(AlternativeSource(raw))
        except ValueError as exc:
            raise ValueError(f"Unsupported alternative data source: {raw}") from exc
    return tuple(resolved)


def _resolve_symbols(config: PopulateAlternativeDataTaskConfig) -> tuple[str, ...]:
    if config.symbols:
        symbols = tuple({symbol.upper() for symbol in config.symbols if symbol})
        if symbols:
            return symbols
    tier_symbols = load_tier1_symbols(config.tier1_progress_path)
    if tier_symbols:
        return tier_symbols
    raise ValueError("No symbols provided and Tier 1 progress file was not found or empty")


def populate_alternative_data_task(
    config: PopulateAlternativeDataTaskConfig,
) -> AlternativeDataResult:
    """
    Populate alternative data sources and persist datasets.
    """
    symbols = _resolve_symbols(config)
    sources = _resolve_sources(config)
    loader_config = AlternativeDataConfig(symbols=symbols, sources=sources)
    result = populate_alternative_data(loader_config)
    save_alternative_data(result, config.output_dir)
    return result


__all__ = [
    "PopulateAlternativeDataTaskConfig",
    "populate_alternative_data_task",
]
