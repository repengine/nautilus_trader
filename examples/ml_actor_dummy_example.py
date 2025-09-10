#!/usr/bin/env python3
"""
Minimal example: instantiate MLSignalActor with dummy stores (no persistence).
"""
from __future__ import annotations

from ml.actors.signal import MLSignalActor
from ml.actors.signal import MLSignalActorConfig
from nautilus_trader.model.data import BarType
from nautilus_trader.model.identifiers import InstrumentId


def main() -> None:
    inst = InstrumentId.from_str("EURUSD.SIM")
    bar_type = BarType.from_str("EURUSD.SIM-1-MINUTE-MID-EXTERNAL")

    cfg = MLSignalActorConfig(
        model_path="/tmp/nonexistent.onnx",  # demo path; loader won't be used here
        model_id="demo_model",
        bar_type=bar_type,
        instrument_id=inst,
        db_connection=None,  # trigger progressive fallback
        use_dummy_stores=True,  # force dummy stores (no persistence)
        log_predictions=False,
    )

    actor = MLSignalActor(cfg)
    print("Actor:", type(actor).__name__)
    print(
        "Stores:",
        type(actor.feature_store).__name__,
        type(actor.model_store).__name__,
        type(actor.strategy_store).__name__,
    )


if __name__ == "__main__":
    main()
