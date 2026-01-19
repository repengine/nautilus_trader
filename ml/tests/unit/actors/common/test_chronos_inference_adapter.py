from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from ml.actors.common.chronos_inference import ChronosInferenceAdapter
from ml.config.autogluon import AutoGluonDataConfig


class _StubSeries:
    def __init__(self, values: npt.NDArray[np.float32]) -> None:
        self.values = values


class _StubFrame:
    def __init__(self, values: npt.NDArray[np.float32]) -> None:
        self._values = values
        self.columns = ["mean"]

    def __getitem__(self, key: str) -> _StubSeries:
        if key != "mean":
            raise KeyError(key)
        return _StubSeries(self._values)

    @property
    def values(self) -> npt.NDArray[np.float32]:
        return self._values


class _StubPredictor:
    def __init__(self, result: Any) -> None:
        self._result = result
        self.calls: list[tuple[Any, dict[str, Any]]] = []

    def predict(self, data: Any, **kwargs: Any) -> Any:
        self.calls.append((data, dict(kwargs)))
        return self._result


def test_predict_from_timeseries_extracts_mean() -> None:
    values = np.array([0.25, 0.5, 0.75], dtype=np.float32)
    predictor = _StubPredictor(_StubFrame(values))
    adapter = ChronosInferenceAdapter(
        predictor=predictor,
        data_config=AutoGluonDataConfig(),
    )

    output = adapter.predict_from_timeseries({"dummy": "tsdf"})

    assert np.array_equal(output, values)
    assert predictor.calls
