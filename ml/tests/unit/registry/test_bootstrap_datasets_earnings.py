from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml.registry.bootstrap_datasets import bootstrap_datasets
from ml.registry.persistence import BackendType
from ml.stores.data_store import EARNINGS_ACTUALS_DATASET_ID
from ml.stores.data_store import EARNINGS_ESTIMATES_DATASET_ID


def test_bootstrap_datasets_json_includes_earnings(tmp_path: Path) -> None:
    registry_root = tmp_path / "registry"
    registry_root.mkdir()

    bootstrap_datasets(backend=BackendType.JSON, registry_path=registry_root)

    registry_file = registry_root / "data_registry.json"
    payload = json.loads(registry_file.read_text(encoding="utf-8"))
    manifests = payload.get("manifests", {})
    contracts = payload.get("contracts", {})

    assert EARNINGS_ACTUALS_DATASET_ID in manifests
    assert EARNINGS_ESTIMATES_DATASET_ID in manifests
    assert EARNINGS_ACTUALS_DATASET_ID in contracts
    assert EARNINGS_ESTIMATES_DATASET_ID in contracts


def test_bootstrap_datasets_postgres_registers_earnings(monkeypatch: pytest.MonkeyPatch) -> None:
    records: list[list[str]] = []

    class _StubRegistry:
        def __init__(self, *args, **kwargs) -> None:
            self.registered: list[str] = []
            records.append(self.registered)

        def get_manifest(self, dataset_id: str) -> None:
            return None

        def register_dataset(self, manifest) -> str:  # pragma: no cover - simple stub
            self.registered.append(manifest.dataset_id)
            return manifest.dataset_id

    monkeypatch.setenv("NAUTILUS_REGISTRY_DB_URL", "postgresql://registry")
    monkeypatch.setattr("ml.registry.bootstrap_datasets.DataRegistry", _StubRegistry)
    monkeypatch.setattr("ml.registry.data_registry.DataRegistry", _StubRegistry)

    bootstrap_datasets(backend=BackendType.POSTGRES)

    assert records, "Expected stub registry to be instantiated"
    registered = records[0]
    assert EARNINGS_ACTUALS_DATASET_ID in registered
    assert EARNINGS_ESTIMATES_DATASET_ID in registered
