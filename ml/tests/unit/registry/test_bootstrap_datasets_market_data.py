"""
Tests ensuring the dataset bootstrap registers EQUS.MINI metadata.
"""

from __future__ import annotations

from ml.config.dataset_ids import EQUS_MINI_DATASET_ID
from ml.config.dataset_ids import EQUS_MINI_MBP1_DATASET_ID
from ml.config.dataset_ids import EQUS_MINI_QUOTES_DATASET_ID
from ml.config.dataset_ids import EQUS_MINI_TBBO_DATASET_ID
from ml.config.dataset_ids import EQUS_MINI_TRADES_DATASET_ID
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
    assert manifest.location == "market_data_bar"
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


def test_equs_mini_tbbo_manifest_registered() -> None:
    """
    EQUS.MINI_TBBO should be registered to the per-class TBBO table.
    """
    manifest = _get_manifest(EQUS_MINI_TBBO_DATASET_ID)
    assert manifest.location == "market_data_tbbo"
    assert manifest.primary_keys == ["instrument_id", "ts_event"]
    assert {"bid", "ask"} <= manifest.schema.keys()


def test_equs_mini_tbbo_contract_present() -> None:
    """
    EQUS.MINI_TBBO requires a lenient contract for bid/ask validation.
    """
    contracts = bootstrap.create_standard_contracts()
    contract = contracts[EQUS_MINI_TBBO_DATASET_ID]
    assert contract.enforcement_mode == "lenient"
    assert any(rule.field_name == "bid" for rule in contract.validation_rules)


def test_equs_mini_mbp1_manifest_registered() -> None:
    """
    EQUS.MINI_MBP1 should be registered to the per-class MBP-1 table.
    """
    manifest = _get_manifest(EQUS_MINI_MBP1_DATASET_ID)
    assert manifest.location == "market_data_mbp1"
    assert manifest.primary_keys == ["instrument_id", "ts_event"]
    assert {"bid", "ask"} <= manifest.schema.keys()


def test_equs_mini_mbp1_contract_present() -> None:
    """
    EQUS.MINI_MBP1 requires a lenient contract for bid/ask validation.
    """
    contracts = bootstrap.create_standard_contracts()
    contract = contracts[EQUS_MINI_MBP1_DATASET_ID]
    assert contract.enforcement_mode == "lenient"
    assert any(rule.field_name == "bid" for rule in contract.validation_rules)


def test_equs_mini_quotes_manifest_registered() -> None:
    """
    EQUS.MINI_QUOTES should be registered to the per-class quote table.
    """
    manifest = _get_manifest(EQUS_MINI_QUOTES_DATASET_ID)
    assert manifest.location == "market_data_quote_tick"
    assert manifest.primary_keys == ["instrument_id", "ts_event"]
    assert {"bid", "ask"} <= manifest.schema.keys()


def test_equs_mini_quotes_contract_present() -> None:
    """
    EQUS.MINI_QUOTES requires a lenient contract for bid/ask validation.
    """
    contracts = bootstrap.create_standard_contracts()
    contract = contracts[EQUS_MINI_QUOTES_DATASET_ID]
    assert contract.enforcement_mode == "lenient"
    assert any(rule.field_name == "bid" for rule in contract.validation_rules)


def test_equs_mini_trades_manifest_registered() -> None:
    """
    EQUS.MINI_TRADES should be registered to the per-class trade table.
    """
    manifest = _get_manifest(EQUS_MINI_TRADES_DATASET_ID)
    assert manifest.location == "market_data_trade_tick"
    assert manifest.primary_keys == ["instrument_id", "ts_event"]
    assert "last" in manifest.schema


def test_equs_mini_trades_contract_present() -> None:
    """
    EQUS.MINI_TRADES requires a lenient contract for trade validation.
    """
    contracts = bootstrap.create_standard_contracts()
    contract = contracts[EQUS_MINI_TRADES_DATASET_ID]
    assert contract.enforcement_mode == "lenient"
    assert any(rule.field_name == "last" for rule in contract.validation_rules)
