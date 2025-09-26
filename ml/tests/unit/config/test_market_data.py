from __future__ import annotations

import pytest

from ml.config.market_data import coerce_storage_kind
from ml.registry.dataclasses import StorageKind


def test_coerce_storage_kind_accepts_enum_member() -> None:
    assert coerce_storage_kind(StorageKind.POSTGRES) is StorageKind.POSTGRES


def test_coerce_storage_kind_accepts_uppercase_name() -> None:
    assert coerce_storage_kind("POSTGRES") is StorageKind.POSTGRES


def test_coerce_storage_kind_accepts_prefixed_string() -> None:
    assert coerce_storage_kind("StorageKind.PARQUET") is StorageKind.PARQUET


def test_coerce_storage_kind_none_returns_none() -> None:
    assert coerce_storage_kind(None) is None


def test_coerce_storage_kind_invalid_raises() -> None:
    with pytest.raises(ValueError):
        coerce_storage_kind("invalid-kind")
