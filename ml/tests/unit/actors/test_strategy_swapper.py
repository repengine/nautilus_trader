from __future__ import annotations

from typing import Any, MutableMapping

import numpy as np
import numpy.typing as npt
from nautilus_trader.model.data import Bar

from ml.actors.base import MLSignal
from ml.actors.signal import SignalGenerationStrategy
from ml.actors.signal import SignalPolicySwapper


class _NoOpStrategy(SignalGenerationStrategy):
    def generate_signal(
        self,
        bar: Bar,  # pragma: no cover - not exercised in this unit
        prediction: float,
        confidence: float,
        features: npt.NDArray[np.float32],
        context: MutableMapping[str, Any],
    ) -> MLSignal | None:
        return None


def test_signal_policy_swapper_execute_swap() -> None:
    swapper = SignalPolicySwapper()

    s1 = _NoOpStrategy()
    s2 = _NoOpStrategy()

    swapper.set_current(s1, {"version": "1"})
    assert swapper.current_strategy is s1
    assert swapper.current_metadata == {"version": "1"}

    swapper.prepare_swap(s2, {"version": "2"})
    assert swapper.swap_pending is True
    assert swapper.load_error is None

    changed = swapper.execute_swap()
    assert changed is True
    assert swapper.swap_pending is False
    assert swapper.current_strategy is s2
    assert swapper.current_metadata == {"version": "2"}
