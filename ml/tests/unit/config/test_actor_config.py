from __future__ import annotations

from ml.config.actors import MLSignalActorConfig
from ml.config.base import MLActorConfig
from nautilus_trader.model.data import BarType
from nautilus_trader.model.identifiers import InstrumentId
from pytest import MonkeyPatch


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


def test_ml_signal_actor_config_default_db_connection_from_env(monkeypatch: MonkeyPatch) -> None:
    url = "postgresql://env-user:env-pass@env-host:5434/env_db"
    monkeypatch.setenv("ML_DB_CONNECTION", url)
    monkeypatch.setenv("DATABASE_URL", url)

    cfg = MLSignalActorConfig(
        model_path="/models/default.onnx",
        model_id="model-env",
        bar_type=BarType.from_str("SPY.XNAS-1-MINUTE-LAST-EXTERNAL"),
        instrument_id=InstrumentId.from_str("SPY.XNAS"),
    )

    assert cfg.db_connection == url
