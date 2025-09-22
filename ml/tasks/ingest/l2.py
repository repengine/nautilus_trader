"""
Task wrapper for efficient L2 data population.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta
from pathlib import Path

from ml.data.ingest.api import ensure_service
from ml.data.ingest.l2_efficient import L2PopulateConfig
from ml.data.ingest.l2_efficient import L2PopulateResult
from ml.data.ingest.l2_efficient import get_tier1_symbols
from ml.data.ingest.l2_efficient import populate_l2_data


@dataclass(slots=True, frozen=True)
class PopulateL2TaskConfig:
    """
    Arguments accepted by :func:`populate_l2_efficient`.
    """

    data_dir: Path
    progress_file: Path
    symbols: Sequence[str] | None = None
    tier: int | None = None
    days: int = 30
    start_date: datetime | None = None
    end_date: datetime | None = None
    resume: bool = True
    check_gaps: bool = True
    force: bool = False
    max_symbols: int | None = None
    symbol_offset: int = 0
    shuffle: bool = False
    rate_limit: int = 60
    dataset: str = "DBEQ.BASIC"
    schema: str = "mbp-10"
    sleep_between_symbols: float = 0.0


def populate_l2_efficient(config: PopulateL2TaskConfig) -> L2PopulateResult:
    """
    Populate L2 data using the loader helpers and return a summary result.
    """
    if config.symbols:
        symbols = tuple(str(symbol).upper() for symbol in config.symbols if symbol)
    elif config.tier == 1:
        symbols = tuple(get_tier1_symbols())
    else:
        raise ValueError("Populate L2 requires --symbols or tier=1 configuration")
    if not symbols:
        raise ValueError("No symbols resolved for L2 population")

    if config.start_date and config.end_date:
        start = config.start_date
        end = config.end_date
    else:
        end = datetime.now() - timedelta(days=1)
        start = end - timedelta(days=max(config.days, 1) - 1)

    loader_config = L2PopulateConfig(
        symbols=symbols,
        data_dir=config.data_dir,
        progress_file=config.progress_file,
        resume=config.resume,
        start_date=start,
        end_date=end,
        check_gaps=config.check_gaps,
        force=config.force,
        max_symbols=config.max_symbols,
        symbol_offset=config.symbol_offset,
        shuffle=config.shuffle,
        rate_limit=config.rate_limit,
        dataset=config.dataset,
        schema=config.schema,
        sleep_between_symbols=config.sleep_between_symbols,
    )

    service = ensure_service()
    return populate_l2_data(loader_config, service=service)


__all__ = ["PopulateL2TaskConfig", "populate_l2_efficient"]
