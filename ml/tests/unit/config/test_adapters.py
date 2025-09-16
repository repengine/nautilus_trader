from __future__ import annotations

from types import SimpleNamespace

from ml.config.adapters import ConfigurationHelper, create_actor_config


class _Cfg(SimpleNamespace):
    pass


def test_create_actor_config_defaults() -> None:
    cfg = _Cfg(component_id="comp-1")
    ac = create_actor_config(cfg)  # type: ignore[arg-type]
    assert ac.component_id == "comp-1"
    assert ac.log_events is True
    assert ac.log_commands is True


def test_configuration_helper_accessors() -> None:
    # Minimal stand-ins with required attributes
    from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue

    # Use a simple sentinel object for bar_type to avoid strict parsing
    bar_type = object()
    instrument_id = InstrumentId(Symbol("EUR/USD"), Venue("SIM"))
    cfg = _Cfg(bar_type=bar_type, instrument_id=instrument_id, model_path="/tmp/model.onnx")

    assert ConfigurationHelper.get_bar_type(cfg) is bar_type
    assert ConfigurationHelper.get_instrument_id(cfg) == instrument_id
    assert ConfigurationHelper.get_model_path(cfg).endswith("model.onnx")
