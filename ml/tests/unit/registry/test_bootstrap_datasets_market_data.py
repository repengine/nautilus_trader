"""
Tests ensuring the dataset bootstrap registers EQUS.MINI metadata.
"""

from __future__ import annotations

from ml.config.dataset_ids import EQUS_MINI_DATASET_ID
from ml.registry import bootstrap_datasets as bootstrap


def _get_manifest(dataset_id: str):
    for manifest in bootstrap.create_standard_manifests():
        if manifest.dataset_id == dataset_id:
            return manifest
    raise AssertionError(f"Manifest for {dataset_id} not found")


def test_equs_mini_manifest_registered() -> None:
    """
    EQUS.MINI should be part of the standard manifests so registries can seed it.
    """

    manifest = _get_manifest(EQUS_MINI_DATASET_ID)
    assert manifest.location == "market_data"
    assert manifest.primary_keys == ["instrument_id", "ts_event"]
    assert {"open", "high", "low", "close"} <= manifest.schema.keys()


def test_equs_mini_contract_present() -> None:
    """
    EQUS.MINI requires a dedicated data contract to enforce ingestion quality.
    """

    contracts = bootstrap.create_standard_contracts()
    contract = contracts[EQUS_MINI_DATASET_ID]
    assert contract.enforcement_mode == "lenient"
    assert any(rule.field_name == "close" for rule in contract.validation_rules)
