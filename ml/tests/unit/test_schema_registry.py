from __future__ import annotations

import pytest
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.data import TradeTick

from ml.registry.dataclasses import DatasetType
from ml.schema import DEFAULT_BAR_IDENTIFIER_TEMPLATE
from ml.schema import default_identifier_template_for_dataset_type
from ml.schema import map_schema_to_dataset_type
from ml.schema import schema_spec_for
from ml.schema import schema_to_dataclass
from ml.schema import schema_to_identifier_template
from ml.stores.providers import resolve_catalog_identifier


@pytest.mark.unit
def test_schema_spec_includes_dataset_type_and_template() -> None:
    spec = schema_spec_for("mbp-1")

    assert spec.dataset_type is DatasetType.MBP1
    assert spec.data_class is QuoteTick
    assert spec.identifier_template == "{instrument_id}"
    assert map_schema_to_dataset_type("tbbo") is DatasetType.TBBO


@pytest.mark.unit
def test_schema_identifier_templates_default_to_registry() -> None:
    assert schema_to_identifier_template("ohlcv-1m") == DEFAULT_BAR_IDENTIFIER_TEMPLATE
    assert schema_to_dataclass("trades") is TradeTick
    assert default_identifier_template_for_dataset_type(DatasetType.TBBO) == "{instrument_id}"


@pytest.mark.unit
def test_resolve_catalog_identifier_uses_registry_default() -> None:
    instrument = "AAPL.NASDAQ"

    identifier = resolve_catalog_identifier(schema="ohlcv-1m", instrument_id=instrument)

    assert identifier == DEFAULT_BAR_IDENTIFIER_TEMPLATE.format(instrument_id=instrument)


@pytest.mark.unit
def test_schema_registry_rejects_unknown_schema() -> None:
    with pytest.raises(ValueError):
        map_schema_to_dataset_type("unknown-schema")


@pytest.mark.unit
def test_resolve_catalog_identifier_prefers_schema_override() -> None:
    instrument = "AAPL.NASDAQ"
    schema_templates = {"tbbo": "{schema}:{instrument_id}"}

    identifier = resolve_catalog_identifier(
        schema="tbbo",
        instrument_id=instrument,
        schema_templates=schema_templates,
    )

    assert identifier == f"tbbo:{instrument}"


@pytest.mark.unit
def test_resolve_catalog_identifier_prefers_dataset_template_when_schema_missing() -> None:
    instrument = "AAPL.NASDAQ"
    dataset_templates = {DatasetType.TRADES: "{instrument_id}-trades"}

    identifier = resolve_catalog_identifier(
        schema="trades",
        instrument_id=instrument,
        dataset_templates=dataset_templates,
    )

    assert identifier == f"{instrument}-trades"
