from __future__ import annotations

pytest_plugins = ("ml.tests.fixtures.pytest_plugins",)

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Protocol

import numpy as np
import numpy.typing as npt
import pytest

from ml.actors.multi_signal import MultiInstrumentSignalActor
from ml.actors.multi_signal import MultiInstrumentSignalActorConfig


pytestmark = pytest.mark.usefixtures(
    "isolated_prometheus_registry",
    "mock_tracing_backend",
    "isolated_orchestrator_env",
)


class _OnnxSessionStub(Protocol):
    """Subset of the onnx_session_stub_factory surface used in this test."""

    prediction: float
    confidence: float

    def get_inputs(self) -> list[SimpleNamespace]:
        ...

    def get_outputs(self) -> list[SimpleNamespace]:
        ...

    def run(
        self,
        output_names: object,
        input_feed: dict[str, npt.NDArray[np.float32]],
    ) -> list[npt.NDArray[np.float32]]:
        ...


@pytest.mark.integration
def test_vectorized_infer_with_mocked_onnxruntime(
    mock_onnx_runtime: Any,
    onnx_session_stub_factory: Callable[..., _OnnxSessionStub],
    tmp_path: Path,
) -> None:
    feature_dim = 4
    model_file = tmp_path / "mini_sum.onnx"
    model_file.write_bytes(b"mock-model")

    stub_session = onnx_session_stub_factory(
        prediction=0.75,
        confidence=0.88,
    )
    mock_onnx_runtime.ort.InferenceSession.side_effect = lambda *args, **kwargs: stub_session

    cfg = MultiInstrumentSignalActorConfig(
        actor_id="onnx-integration",
        max_batch_size=8,
        feature_dim=feature_dim,
        use_dummy_stores=True,
        model_path=str(model_file),
        model_id="mini_sum",
        instrument_id=None,  # not used here
        bar_type=None,  # not used here
    )
    actor = MultiInstrumentSignalActor(cfg)  # type: ignore[arg-type]

    # Load ORT session via the mocked runtime and wire metadata
    session = mock_onnx_runtime.ort.InferenceSession(str(model_file))
    metadata = {
        "input_names": [info.name for info in session.get_inputs()],
        "output_names": [info.name for info in session.get_outputs()],
    }
    setattr(actor, "_model", session)
    setattr(actor, "_model_metadata", metadata)

    # Two rows ensure vectorized inference feeds the stub session once.
    batch: npt.NDArray[np.float32] = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [0.5, 0.5, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    preds, confs = actor._infer_batch(batch)
    expected_preds = np.full(batch.shape[0], stub_session.prediction, dtype=np.float32)
    expected_confs = np.full(batch.shape[0], stub_session.confidence, dtype=np.float32)
    np.testing.assert_allclose(preds, expected_preds)
    np.testing.assert_allclose(confs, expected_confs)
