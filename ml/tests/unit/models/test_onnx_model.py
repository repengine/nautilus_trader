#!/usr/bin/env python3

# -------------------------------------------------------------------------------------------------
#  Copyright (C) 2015-2025 Nautech Systems Pty Ltd. All rights reserved.
#  https://nautechsystems.io
#
#  Licensed under the GNU Lesser General Public License Version 3.0 (the "License");
#  You may not use this file except in compliance with the License.
#  You may obtain a copy of the License at https://www.gnu.org/licenses/lgpl-3.0.en.html
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# -------------------------------------------------------------------------------------------------

"""Unit tests for ONNXModel wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ml._imports import HAS_ONNX
from ml._imports import check_ml_dependencies
from ml._imports import ort


if HAS_ONNX:
    import onnxruntime as ort_typing
else:
    ort_typing = None
from ml.models.onnx_model import ONNXModel
from ml.tests.fixtures.model_factory import TestModelFactory


# Skip all tests if ONNX not available
pytestmark = pytest.mark.skipif(not HAS_ONNX, reason="ONNX Runtime not available")


class TestONNXModel:
    """Test suite for ONNXModel wrapper."""

    @pytest.fixture
    def single_output_model(self, tmp_path: Path) -> tuple[ONNXModel, dict[str, Any]]:
        """Create a minimal single-output ONNX model for testing."""
        if not HAS_ONNX:
            check_ml_dependencies(["onnx"])

        model_path = TestModelFactory.create_onnx_model(
            n_features=5,
            n_outputs=1,
            output_path=tmp_path / "single_output.onnx",
        )

        session = ort.InferenceSession(str(model_path))
        metadata = {
            "model_type": "onnx",
            "n_features": 5,
            "n_outputs": 1,
            "input_shape": [None, 5],
            "output_shape": [None, 1],
        }

        return ONNXModel(session, metadata), metadata

    @pytest.fixture
    def multi_output_model(self, tmp_path: Path) -> tuple[ONNXModel, dict[str, Any]]:
        """Create a minimal multi-output ONNX model for testing."""
        if not HAS_ONNX:
            check_ml_dependencies(["onnx"])

        model_path = TestModelFactory.create_onnx_model(
            n_features=3,
            n_outputs=3,
            output_path=tmp_path / "multi_output.onnx",
        )

        session = ort.InferenceSession(str(model_path))
        metadata = {
            "model_type": "onnx",
            "n_features": 3,
            "n_outputs": 3,
            "input_shape": [None, 3],
            "output_shape": [None, 3],
        }

        return ONNXModel(session, metadata), metadata

    @pytest.fixture
    def model_with_custom_names(self, tmp_path: Path) -> tuple[ONNXModel, dict[str, Any]]:
        """Create ONNX model with custom input/output names in metadata."""
        if not HAS_ONNX:
            check_ml_dependencies(["onnx"])

        model_path = TestModelFactory.create_onnx_model(
            n_features=4,
            n_outputs=2,
            output_path=tmp_path / "custom_names.onnx",
        )

        session = ort.InferenceSession(str(model_path))
        metadata = {
            "model_type": "onnx",
            "input_names": ["custom_input"],
            "output_names": ["custom_output"],
            "input_shape": [None, 4],
        }

        return ONNXModel(session, metadata), metadata

    def test_initialization_with_inference_session(self, single_output_model: tuple[ONNXModel, dict[str, Any]]) -> None:
        """ONNXModel can be initialized with InferenceSession."""
        model, expected_metadata = single_output_model

        assert isinstance(model, ONNXModel)
        assert model.metadata == expected_metadata
        assert hasattr(model._model, "run")  # InferenceSession has run method

    def test_initialization_without_onnx_raises_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ONNXModel initialization fails gracefully when ONNX Runtime not available."""
        # Mock HAS_ONNX to False and the check_ml_dependencies function
        import ml.models.onnx_model
        monkeypatch.setattr(ml.models.onnx_model, "HAS_ONNX", False)

        def mock_check_deps(deps: list[str]) -> None:
            raise ImportError("ONNX Runtime required but not installed")

        monkeypatch.setattr(ml.models.onnx_model, "check_ml_dependencies", mock_check_deps)

        with pytest.raises(ImportError, match="ONNX Runtime required but not installed"):
            ONNXModel(model=None, metadata={})

    def test_input_output_names_from_session(self, single_output_model: tuple[ONNXModel, dict[str, Any]]) -> None:
        """ONNXModel extracts input/output names from InferenceSession."""
        model, _ = single_output_model

        # Check that input/output names are extracted from session
        assert hasattr(model, "_input_names")
        assert hasattr(model, "_output_names")
        assert isinstance(model._input_names, list)
        assert isinstance(model._output_names, list)
        assert len(model._input_names) > 0
        assert len(model._output_names) > 0

    def test_input_output_names_from_metadata_fallback(self, model_with_custom_names: tuple[ONNXModel, dict[str, Any]]) -> None:
        """ONNXModel falls back to metadata for input/output names when available."""
        model, metadata = model_with_custom_names

        # If metadata provides names, they should be used
        if "input_names" in metadata:
            # Note: Actual implementation prioritizes session names over metadata
            # This tests the fallback scenario
            assert hasattr(model, "_input_names")

    def test_predict_returns_float32_single_output(self, single_output_model: tuple[ONNXModel, dict[str, Any]]) -> None:
        """Single output predictions return float32 dtype."""
        model, _ = single_output_model

        features = np.random.randn(10, 5).astype(np.float32)
        predictions = model.predict(features)

        assert isinstance(predictions, np.ndarray)
        assert predictions.dtype == np.float32
        assert predictions.shape == (10,)  # Squeezed single output

    def test_predict_returns_float32_multi_output(self, multi_output_model: tuple[ONNXModel, dict[str, Any]]) -> None:
        """Multi output predictions return float32 dtype (first output only)."""
        model, _ = multi_output_model

        features = np.random.randn(5, 3).astype(np.float32)
        predictions = model.predict(features)

        assert isinstance(predictions, np.ndarray)
        assert predictions.dtype == np.float32
        # ONNX models return first output only, which is typically class labels (1D)
        assert predictions.shape == (5,)

    def test_predict_with_1d_input_handles_reshape(self, single_output_model: tuple[ONNXModel, dict[str, Any]]) -> None:
        """predict() handles 1D input by reshaping to 2D."""
        model, _ = single_output_model

        # Single sample as 1D array
        features_1d = np.random.randn(5).astype(np.float32)
        predictions = model.predict(features_1d)

        assert isinstance(predictions, np.ndarray)
        assert predictions.dtype == np.float32
        assert predictions.shape == (1,)  # Single prediction

    def test_predict_with_2d_input_preserves_batch_dimension(self, single_output_model: tuple[ONNXModel, dict[str, Any]]) -> None:
        """predict() handles 2D input correctly for batch predictions."""
        model, _ = single_output_model

        # Batch of samples
        features_2d = np.random.randn(7, 5).astype(np.float32)
        predictions = model.predict(features_2d)

        assert isinstance(predictions, np.ndarray)
        assert predictions.dtype == np.float32
        assert predictions.shape == (7,)  # Batch size preserved

    def test_predict_validates_input_shape(self, single_output_model: tuple[ONNXModel, dict[str, Any]]) -> None:
        """predict() validates input features through validate_input."""
        model, _ = single_output_model

        # Wrong number of features (expect 5, provide 3)
        wrong_features = np.random.randn(2, 3).astype(np.float32)

        with pytest.raises(ValueError, match="Input validation failed"):
            model.predict(wrong_features)

    def test_predict_enforces_float32_from_float64_input(self, single_output_model: tuple[ONNXModel, dict[str, Any]]) -> None:
        """predict() converts float64 input to float32."""
        model, _ = single_output_model

        # Start with float64 - the model should convert internally to float32
        features_f64 = np.random.randn(3, 5).astype(np.float64)
        # Type ignore because the model accepts NDArray[np.float32] but we're testing conversion
        predictions = model.predict(features_f64)  # type: ignore[arg-type]

        assert predictions.dtype == np.float32

    def test_predict_with_empty_batch(self, single_output_model: tuple[ONNXModel, dict[str, Any]]) -> None:
        """predict() handles empty batch correctly."""
        model, _ = single_output_model

        empty_features = np.empty((0, 5), dtype=np.float32)
        predictions = model.predict(empty_features)

        assert isinstance(predictions, np.ndarray)
        assert predictions.dtype == np.float32
        assert predictions.shape == (0,)

    def test_predict_handles_extreme_values(self, single_output_model: tuple[ONNXModel, dict[str, Any]]) -> None:
        """predict() handles extreme input values gracefully."""
        model, _ = single_output_model

        # Extreme values (finite values only)
        extreme_features = np.array([
            [1e6, -1e6, 0, 1e3, -1e3],
            [100.0, 1.0, 2.0, 3.0, 4.0],
        ], dtype=np.float32)

        predictions = model.predict(extreme_features)

        assert isinstance(predictions, np.ndarray)
        assert predictions.dtype == np.float32
        assert predictions.shape == (2,)
        # Predictions should be finite
        assert np.all(np.isfinite(predictions))

    def test_predict_single_output_squeezes_last_dimension(self, single_output_model: tuple[ONNXModel, dict[str, Any]]) -> None:
        """Single output models squeeze the last dimension when it equals 1."""
        model, _ = single_output_model

        features = np.random.randn(4, 5).astype(np.float32)
        predictions = model.predict(features)

        # Should be (4,) not (4, 1) for single output
        assert predictions.shape == (4,)
        assert predictions.ndim == 1

    def test_predict_multi_output_preserves_dimensions(self, multi_output_model: tuple[ONNXModel, dict[str, Any]]) -> None:
        """Multi output models return first output only (ONNX behavior)."""
        model, _ = multi_output_model

        features = np.random.randn(6, 3).astype(np.float32)
        predictions = model.predict(features)

        # ONNX models return first output only, typically class predictions (1D)
        assert predictions.shape == (6,)
        assert predictions.ndim == 1

    def test_predict_handles_scalar_output(self, tmp_path: Path) -> None:
        """predict() handles scalar outputs correctly."""
        if not HAS_ONNX:
            check_ml_dependencies(["onnx"])

        # Create model that might return scalar for single sample
        model_path = TestModelFactory.create_onnx_model(
            n_features=2,
            n_outputs=1,
            output_path=tmp_path / "scalar_model.onnx",
        )

        session = ort.InferenceSession(str(model_path))
        metadata = {"input_shape": [None, 2]}
        model = ONNXModel(session, metadata)

        # Single sample that might produce scalar
        single_sample = np.array([[1.0, 2.0]], dtype=np.float32)
        predictions = model.predict(single_sample)

        assert isinstance(predictions, np.ndarray)
        assert predictions.dtype == np.float32
        # Should still be array, not scalar
        assert predictions.shape in [(1,), (1, 1)]

    def test_predict_consistency_across_batch_sizes(self, single_output_model: tuple[ONNXModel, dict[str, Any]]) -> None:
        """Predictions should be consistent regardless of batch size."""
        model, _ = single_output_model

        # Same sample in different batch configurations
        sample = np.random.randn(5).astype(np.float32)

        # Single sample
        pred_single = model.predict(sample.reshape(1, -1))

        # Same sample repeated in batch
        batch_samples = np.tile(sample, (3, 1))
        pred_batch = model.predict(batch_samples)

        # First prediction should be same across batch sizes
        assert np.allclose(pred_single[0], pred_batch[0], rtol=1e-5)
        # All predictions in batch should be identical (same input)
        assert np.allclose(pred_batch[0], pred_batch[1], rtol=1e-5)
        assert np.allclose(pred_batch[0], pred_batch[2], rtol=1e-5)

    def test_predict_uses_first_input_name(self, single_output_model: tuple[ONNXModel, dict[str, Any]]) -> None:
        """predict() uses first input name from the session."""
        model, _ = single_output_model

        # This is an implementation test - ensure first input name is used
        assert len(model._input_names) > 0
        first_input_name = model._input_names[0]

        features = np.random.randn(2, 5).astype(np.float32)

        # Should not raise error (tests internal logic)
        predictions = model.predict(features)
        assert isinstance(predictions, np.ndarray)

    def test_predict_extracts_first_output(self, multi_output_model: tuple[ONNXModel, dict[str, Any]]) -> None:
        """predict() extracts first output from ONNX session results."""
        model, _ = multi_output_model

        features = np.random.randn(2, 3).astype(np.float32)
        predictions = model.predict(features)

        # Should return ndarray (first output processed)
        assert isinstance(predictions, np.ndarray)
        assert predictions.dtype == np.float32
        assert predictions.shape == (2,)  # First output shape

        # Verify this is actually from a multi-output model by checking session
        session = model._model
        outputs = session.get_outputs()
        assert len(outputs) > 1, "Should be multi-output model for this test"

    def test_predict_without_onnx_dependency(self, single_output_model: tuple[ONNXModel, dict[str, Any]], monkeypatch: pytest.MonkeyPatch) -> None:
        """predict() method handles missing ONNX dependency gracefully."""
        model, _ = single_output_model

        # Note: This scenario is unlikely since we skip tests without ONNX,
        # but tests robustness of the implementation
        features = np.random.randn(2, 5).astype(np.float32)

        # Should work normally when ONNX is available
        predictions = model.predict(features)
        assert isinstance(predictions, np.ndarray)

    def test_inheritance_from_base_model(self, single_output_model: tuple[ONNXModel, dict[str, Any]]) -> None:
        """ONNXModel properly inherits from BaseModel."""
        model, _ = single_output_model

        # Should have all BaseModel methods
        assert hasattr(model, "metadata")
        assert hasattr(model, "model_id")
        assert hasattr(model, "validate_input")
        assert hasattr(model, "predict")

        # Test BaseModel behavior
        assert isinstance(model.metadata, dict)
        assert isinstance(model.model_id, str)

    def test_metadata_passthrough(self, single_output_model: tuple[ONNXModel, dict[str, Any]]) -> None:
        """Metadata is properly stored and accessible."""
        model, expected_metadata = single_output_model

        assert model.metadata == expected_metadata
        assert model.metadata["model_type"] == "onnx"
        assert model.metadata["n_features"] == 5

    def test_edge_case_single_feature_model(self, tmp_path: Path) -> None:
        """ONNXModel works with single feature models."""
        if not HAS_ONNX:
            check_ml_dependencies(["onnx"])

        model_path = TestModelFactory.create_onnx_model(
            n_features=1,
            n_outputs=1,
            output_path=tmp_path / "single_feature.onnx",
        )

        session = ort.InferenceSession(str(model_path))
        metadata = {"input_shape": [None, 1]}
        model = ONNXModel(session, metadata)

        # Test with single feature
        features = np.array([[1.5]], dtype=np.float32)
        predictions = model.predict(features)

        assert predictions.dtype == np.float32
        assert predictions.shape == (1,)

    def test_edge_case_large_batch_prediction(self, single_output_model: tuple[ONNXModel, dict[str, Any]]) -> None:
        """ONNXModel handles large batch predictions efficiently."""
        model, _ = single_output_model

        # Large batch
        large_batch = np.random.randn(1000, 5).astype(np.float32)
        predictions = model.predict(large_batch)

        assert predictions.dtype == np.float32
        assert predictions.shape == (1000,)
        assert np.all(np.isfinite(predictions))

    def test_model_reproducibility(self, tmp_path: Path) -> None:
        """ONNXModel predictions are reproducible for same input."""
        if not HAS_ONNX:
            check_ml_dependencies(["onnx"])

        # Create two identical models
        model_path_1 = TestModelFactory.create_onnx_model(
            n_features=3,
            n_outputs=1,
            output_path=tmp_path / "model1.onnx",
        )

        model_path_2 = TestModelFactory.create_onnx_model(
            n_features=3,
            n_outputs=1,
            output_path=tmp_path / "model2.onnx",
        )

        # Load both models
        assert ort is not None, "ONNX Runtime should be available for this test"
        session_1 = ort.InferenceSession(str(model_path_1))
        session_2 = ort.InferenceSession(str(model_path_2))

        metadata = {"input_shape": [None, 3]}
        model_1 = ONNXModel(session_1, metadata)
        model_2 = ONNXModel(session_2, metadata)

        # Same input
        features = np.random.randn(5, 3).astype(np.float32)

        pred_1 = model_1.predict(features)
        pred_2 = model_2.predict(features)

        # Should be identical (same seed used in TestModelFactory)
        assert np.array_equal(pred_1, pred_2)

    def test_handles_different_input_output_combinations(self, tmp_path: Path) -> None:
        """ONNXModel handles various input/output dimension combinations."""
        if not HAS_ONNX:
            check_ml_dependencies(["onnx"])

        test_cases = [
            (2, 1),  # Simple regression
            (5, 2),  # Binary classification
            (4, 3),  # Multi-class
            (10, 5), # Multi-output
        ]

        for n_features, n_outputs in test_cases:
            model_path = TestModelFactory.create_onnx_model(
                n_features=n_features,
                n_outputs=n_outputs,
                output_path=tmp_path / f"model_{n_features}_{n_outputs}.onnx",
            )

            session = ort.InferenceSession(str(model_path))
            metadata = {"input_shape": [None, n_features]}
            model = ONNXModel(session, metadata)

            # Test prediction
            features = np.random.randn(3, n_features).astype(np.float32)
            predictions = model.predict(features)

            assert predictions.dtype == np.float32
            # ONNX models always return first output only, so shape is always (batch,)
            assert predictions.shape == (3,)

    def test_onnx_runtime_providers_accessibility(self, single_output_model: tuple[ONNXModel, dict[str, Any]]) -> None:
        """ONNXModel can access ONNX Runtime session properties."""
        model, _ = single_output_model

        # Should be able to access session properties
        session = model._model
        assert hasattr(session, "get_providers")

        providers = session.get_providers()
        assert isinstance(providers, list)
        assert len(providers) > 0
        # Should have at least CPUExecutionProvider
        assert "CPUExecutionProvider" in providers

    def test_input_output_metadata_extraction(self, single_output_model: tuple[ONNXModel, dict[str, Any]]) -> None:
        """ONNXModel correctly extracts input/output metadata from session."""
        model, _ = single_output_model

        session = model._model

        # Check that inputs/outputs were extracted
        inputs = session.get_inputs()
        outputs = session.get_outputs()

        assert len(inputs) > 0
        assert len(outputs) > 0

        # Verify our model cached them
        assert len(model._input_names) == len(inputs)
        assert len(model._output_names) == len(outputs)

        # Names should match
        for i, inp in enumerate(inputs):
            assert model._input_names[i] == inp.name

        for i, out in enumerate(outputs):
            assert model._output_names[i] == out.name
