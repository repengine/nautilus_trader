"""Tests for DataStore earnings fallback chain."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ml._imports import HAS_POLARS
from ml.stores.data_store import DataStore


@pytest.mark.skipif(not HAS_POLARS, reason="polars required for file fallback")
def test_data_store_falls_back_to_file_store(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Ensure DataStore uses FileEarningsStore when PostgreSQL earnings store fails."""

    def _failing_earnings_store(*_args: object, **_kwargs: object) -> None:  # pragma: no cover - patch target
        raise RuntimeError("unavailable")

    monkeypatch.setenv("ML_FILE_STORE_PATH", str(tmp_path / "file_store"))
    monkeypatch.setattr("ml.stores.data_store.EarningsStore", _failing_earnings_store)

    feature_store = MagicMock()
    model_store = MagicMock()
    strategy_store = MagicMock()
    data_processor = MagicMock()
    registry = MagicMock()

    store = DataStore(
        connection_string="postgresql://unused",
        registry=registry,
        feature_store=feature_store,
        model_store=model_store,
        strategy_store=strategy_store,
        data_processor=data_processor,
    )

    from ml.stores.file_backed import FileEarningsStore

    assert isinstance(store._earnings_store, FileEarningsStore)
    store._earnings_store.flush()

    monkeypatch.delenv("ML_FILE_STORE_PATH", raising=False)
