#!/usr/bin/env python3
"""
CLI wrapper for building TFT datasets via :mod:`ml.tasks.datasets.tft_cli`.
"""

from __future__ import annotations

from collections.abc import Sequence

from ml.stores.data_store import DataStore
from ml.tasks.datasets import build_tft_dataset
from ml.tasks.datasets import TFTDatasetTaskConfig
from ml.tasks.datasets import tft_cli as _tft_cli


__all__ = ["DataStore", "TFTDatasetTaskConfig", "build_tft_dataset", "main"]


def main(argv: Sequence[str] | None = None) -> int:
    """
    CLI entrypoint delegating to the shared task helper.
    """
    return _tft_cli.main(argv, build_fn=build_tft_dataset, data_store_cls=DataStore)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
