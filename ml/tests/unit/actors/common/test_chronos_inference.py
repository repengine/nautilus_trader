from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from ml.actors.common import chronos_inference as chronos_module
from ml.actors.common.chronos_inference import ChronosInferenceAdapter
from ml.actors.common.chronos_inference import _extract_prediction_array
from ml.config.autogluon import AutoGluonDataConfig


class _Predictor:
    def __init__(self, output: object) -> None:
        self._output = output

    def predict(self, _data: object, **_kwargs: object) -> object:
        return self._output


def test_extract_prediction_array_from_mean_column() -> None:
    class _Frame:
        columns = ("mean",)

        def __getitem__(self, _key: str) -> object:
            return SimpleNamespace(values=np.array([1.0, 2.0], dtype=np.float64))

    result = _extract_prediction_array(_Frame())

    assert result.dtype == np.float32
    assert np.allclose(result, np.array([1.0, 2.0], dtype=np.float32))


def test_extract_prediction_array_from_values() -> None:
    data = SimpleNamespace(values=np.array([3.0, 4.0], dtype=np.float64))

    result = _extract_prediction_array(data)

    assert result.dtype == np.float32
    assert np.allclose(result, np.array([3.0, 4.0], dtype=np.float32))


def test_extract_prediction_array_fallback() -> None:
    result = _extract_prediction_array([5.0, 6.0])

    assert result.dtype == np.float32
    assert np.allclose(result, np.array([5.0, 6.0], dtype=np.float32))


def test_ensure_timeseries_dataframe_returns_existing(monkeypatch: pytest.MonkeyPatch) -> None:
    class _TSDF:
        pass

    tsdf = _TSDF()
    monkeypatch.setattr(chronos_module, "TimeSeriesDataFrame", _TSDF)

    adapter = ChronosInferenceAdapter(
        predictor=_Predictor(np.array([1.0])),
        data_config=AutoGluonDataConfig(),
    )

    assert adapter._ensure_timeseries_dataframe(tsdf) is tsdf


def test_ensure_timeseries_dataframe_requires_autogluon(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chronos_module, "TimeSeriesDataFrame", None)
    monkeypatch.setattr(chronos_module, "HAS_AUTOGLUON", False)
    called: list[list[str]] = []

    def _check(deps: list[str]) -> None:
        called.append(deps)

    monkeypatch.setattr(chronos_module, "check_ml_dependencies", _check)

    adapter = ChronosInferenceAdapter(
        predictor=_Predictor(np.array([1.0])),
        data_config=AutoGluonDataConfig(),
    )

    with pytest.raises(ImportError):
        adapter._ensure_timeseries_dataframe({"x": 1})

    assert called == [["autogluon"]]


def test_ensure_timeseries_dataframe_converts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chronos_module, "TimeSeriesDataFrame", None)
    monkeypatch.setattr(chronos_module, "HAS_AUTOGLUON", True)
    sentinel = object()
    called: list[tuple[object, AutoGluonDataConfig]] = []

    def _convert(data: object, cfg: AutoGluonDataConfig) -> object:
        called.append((data, cfg))
        return sentinel

    monkeypatch.setattr(chronos_module, "convert_to_timeseries_dataframe", _convert)

    adapter = ChronosInferenceAdapter(
        predictor=_Predictor(np.array([1.0])),
        data_config=AutoGluonDataConfig(),
    )

    data = {"x": 1}
    result = adapter._ensure_timeseries_dataframe(data)

    assert result is sentinel
    assert called == [(data, adapter.data_config)]


def test_predict_from_raw_data_flows_through_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = object()
    monkeypatch.setattr(chronos_module, "TimeSeriesDataFrame", None)
    monkeypatch.setattr(chronos_module, "HAS_AUTOGLUON", True)
    monkeypatch.setattr(chronos_module, "convert_to_timeseries_dataframe", lambda _data, _cfg: sentinel)

    adapter = ChronosInferenceAdapter(
        predictor=_Predictor(np.array([7.0, 8.0], dtype=np.float64)),
        data_config=AutoGluonDataConfig(),
    )

    result = adapter.predict({"row": 1})

    assert result.dtype == np.float32
    assert np.allclose(result, np.array([7.0, 8.0], dtype=np.float32))
