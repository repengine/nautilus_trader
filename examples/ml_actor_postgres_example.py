#!/usr/bin/env python3
"""
Minimal example: instantiate MLSignalActor backed by PostgreSQL.

Requires a running Postgres instance. By default expects:
  postgresql://postgres:postgres@localhost:5433/nautilus
which matches ml/deployment/docker-compose.yml port mapping.
"""
from __future__ import annotations

import os

from ml.actors.signal import MLSignalActor
from ml.actors.signal import MLSignalActorConfig
from nautilus_trader.model.data import BarType
from nautilus_trader.model.identifiers import InstrumentId


def main() -> None:
    inst = InstrumentId.from_str("EURUSD.SIM")
    bar_type = BarType.from_str("EURUSD.SIM-1-MINUTE-MID-EXTERNAL")

    db_url = os.environ.get(
        "DB_CONNECTION",
        "postgresql://postgres:postgres@localhost:5433/nautilus",
    )

    cfg = MLSignalActorConfig(
        model_path="/tmp/nonexistent.onnx",  # demo path
        model_id="demo_model",
        bar_type=bar_type,
        instrument_id=inst,
        db_connection=db_url,
        use_dummy_stores=False,
        log_predictions=False,
    )

    actor = MLSignalActor(cfg)
    print("Actor:", type(actor).__name__)
    print("FeatureStore engine:", actor.feature_store.engine)


if __name__ == "__main__":
    main()
