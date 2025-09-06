from __future__ import annotations

from ml.config.base import MLActorConfig
from nautilus_trader.model.data import BarType
from nautilus_trader.model.identifiers import InstrumentId


def test_ml_actor_config_has_integration_fields() -> None:
    cfg = MLActorConfig(
        model_path="/tmp/model.onnx",
        model_id="m-1",
        bar_type=BarType.from_str("SPY.XNAS-1-MINUTE-LAST-EXTERNAL"),
        instrument_id=InstrumentId.from_str("SPY.XNAS"),
        db_connection=None,
        use_dummy_stores=True,
    )
    assert cfg.db_connection is None
    assert cfg.use_dummy_stores is True

