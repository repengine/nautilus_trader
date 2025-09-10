"""
Functional test for EnsembleStrategy weighted voting.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from ml.actors.signal import EnsembleStrategy
from ml.actors.signal import ThresholdSignalStrategy
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import Symbol
from nautilus_trader.model.identifiers import Venue


def _stub_bar() -> object:
    inst = InstrumentId(Symbol("EURUSD"), Venue("SIM"))
    return SimpleNamespace(bar_type=SimpleNamespace(instrument_id=inst), ts_event=1)


def test_ensemble_weighted_vote_emits_signal() -> None:
    # Two sub-strategies will both pass threshold and be combined
    s1 = ThresholdSignalStrategy(threshold=0.0)
    s2 = ThresholdSignalStrategy(threshold=0.0)
    ens = EnsembleStrategy(
        strategies={"s1": s1, "s2": s2},
        weights={"s1": 0.7, "s2": 0.3},
        threshold=0.0,
    )

    ctx = {"timestamp_ns": 1, "model_id": "m1"}
    sig = ens.generate_signal(
        _stub_bar(),
        prediction=0.6,
        confidence=0.9,
        features=np.zeros(1, dtype=np.float32),
        context=ctx,
    )
    assert sig is not None
