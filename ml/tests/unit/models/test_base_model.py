#!/usr/bin/env python3

"""
Unit tests for BaseModel abstract class.

This module tests the behavioral contracts that all ML model wrappers must follow,
including abstract class enforcement, input validation, metadata management,
and type safety requirements.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from ml.models.base import BaseModel


class ConcreteModel(BaseModel):
    """Concrete implementation of BaseModel for testing."""

    def __init__(self, model: Any, metadata: dict[str, Any]) -> None:
        """Initialize concrete model for testing."""
        super().__init__(model, metadata)

    def predict(self, features: NDArray[np.float32]) -> NDArray[np.float32]:
        """Return features multiplied by 2.0 for testing."""
        # Validate input first
        self.validate_input(features)

        # Simple transformation for testing
        if features.size == 0:
            # Handle empty array case
            return np.array([], dtype=np.float32)
        elif features.ndim == 1:
            result: NDArray[np.float32] = (features * 2.0).astype(np.float32)
            return result
        else:
            # Return mean of features for each sample
            result = (features.mean(axis=1) * 2.0).astype(np.float32)
            return result


class TestBaseModel:
    """Test suite for BaseModel abstract class."""

    def test_cannot_instantiate_abstract_class(self) -> None:
        """BaseModel cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseModel(None, {})  # type: ignore

    def test_concrete_subclass_instantiation(self) -> None:
        """Concrete subclass can be instantiated successfully."""
        metadata = {
            "model_type": "test",
            "version": "1.0",
            "input_shape": [None, 5]
        }
        model = ConcreteModel(model="dummy_model", metadata=metadata)

        assert model is not None
        assert isinstance(model, BaseModel)
        assert isinstance(model, ConcreteModel)

    def test_metadata_property_access(self) -> None:
        """Metadata property returns stored metadata."""
        expected_metadata = {
            "model_type": "xgboost",
            "version": "1.2.0",
            "n_features": 10,
            "input_shape": [None, 10],
            "custom_field": "test_value"
        }

        model = ConcreteModel(model="dummy", metadata=expected_metadata)
        actual_metadata = model.metadata

        assert actual_metadata == expected_metadata
        assert actual_metadata is expected_metadata  # Returns same reference

    def test_model_id_from_model_id_field(self) -> None:
        """model_id property prefers model_id field in metadata."""
        metadata = {
            "model_id": "xgb_v2.1",
            "version": "1.0",
            "model_type": "xgboost"
        }

        model = ConcreteModel(model="dummy", metadata=metadata)

        assert model.model_id == "xgb_v2.1"

    def test_model_id_fallback_to_version(self) -> None:
        """model_id property falls back to version if model_id not present."""
        metadata = {
            "version": "2.5.1",
            "model_type": "lightgbm"
        }

        model = ConcreteModel(model="dummy", metadata=metadata)

        assert model.model_id == "2.5.1"

    def test_model_id_fallback_to_unknown(self) -> None:
        """model_id property returns 'unknown' if neither model_id nor version present."""
        metadata = {
            "model_type": "onnx"
        }

        model = ConcreteModel(model="dummy", metadata=metadata)

        assert model.model_id == "unknown"

    def test_model_id_string_conversion(self) -> None:
        """model_id property converts non-string values to strings."""
        metadata = {
            "model_id": 123,  # Integer
            "model_type": "test"
        }

        model = ConcreteModel(model="dummy", metadata=metadata)

        assert model.model_id == "123"
        assert isinstance(model.model_id, str)

    def test_validate_input_with_correct_shape(self) -> None:
        """validate_input passes with correctly shaped features."""
        metadata = {
            "input_shape": [None, 10]  # Expects 10 features
        }
        model = ConcreteModel(model="dummy", metadata=metadata)

        # 2D array with correct features
        rng = np.random.default_rng(42)
        features_2d = rng.random((5, 10)).astype(np.float32)
        model.validate_input(features_2d)  # Should not raise

        # 1D array with correct features
        features_1d = rng.random(10).astype(np.float32)
        model.validate_input(features_1d)  # Should not raise

    def test_validate_input_with_wrong_feature_count(self) -> None:
        """validate_input raises ValueError with wrong number of features."""
        metadata = {
            "input_shape": [None, 10]  # Expects 10 features
        }
        model = ConcreteModel(model="dummy", metadata=metadata)

        # Wrong number of features
        rng = np.random.default_rng(42)
        wrong_features = rng.random((5, 8)).astype(np.float32)

        with pytest.raises(ValueError, match="Input validation failed"):
            model.validate_input(wrong_features)

    def test_validate_input_error_message_format(self) -> None:
        """validate_input error message contains expected and actual feature counts."""
        metadata = {
            "input_shape": [None, 15]  # Expects 15 features
        }
        model = ConcreteModel(model="dummy", metadata=metadata)

        rng = np.random.default_rng(42)
        wrong_features = rng.random((3, 7)).astype(np.float32)  # 7 features

        with pytest.raises(ValueError) as exc_info:
            model.validate_input(wrong_features)

        error_msg = str(exc_info.value)
        assert "expected 15 features" in error_msg
        assert "got 7" in error_msg

    def test_validate_input_with_no_input_shape(self) -> None:
        """validate_input skips validation when input_shape not in metadata."""
        metadata = {
            "model_type": "test",
            "version": "1.0"
            # No input_shape specified
        }
        model = ConcreteModel(model="dummy", metadata=metadata)

        # Any shape should pass
        rng = np.random.default_rng(42)
        features = rng.random((5, 42)).astype(np.float32)
        model.validate_input(features)  # Should not raise

    def test_validate_input_with_none_input_shape(self) -> None:
        """validate_input skips validation when input_shape is None."""
        metadata = {
            "input_shape": None
        }
        model = ConcreteModel(model="dummy", metadata=metadata)

        rng = np.random.default_rng(42)
        features = rng.random((3, 7)).astype(np.float32)
        model.validate_input(features)  # Should not raise

    def test_validate_input_with_empty_input_shape(self) -> None:
        """validate_input skips validation when input_shape is empty."""
        metadata: dict[str, Any] = {
            "input_shape": []
        }
        model = ConcreteModel(model="dummy", metadata=metadata)

        rng = np.random.default_rng(42)
        features = rng.random((2, 5)).astype(np.float32)
        model.validate_input(features)  # Should not raise

    def test_validate_input_with_scalar_features(self) -> None:
        """validate_input handles scalar input appropriately."""
        metadata = {
            "input_shape": [None, 1]  # Single feature expected
        }
        model = ConcreteModel(model="dummy", metadata=metadata)

        # 0-dimensional scalar should fail (no features dimension)
        scalar = np.array(5.0, dtype=np.float32)
        with pytest.raises(ValueError):
            model.validate_input(scalar)

        # 1D single element should pass
        single_feature = np.array([5.0], dtype=np.float32)
        model.validate_input(single_feature)  # Should not raise

    def test_validate_input_with_complex_input_shapes(self) -> None:
        """validate_input works with complex input shape specifications."""
        metadata: dict[str, Any] = {
            "input_shape": [1, 28, 28, 3]  # Like image data: batch, height, width, channels
        }
        model = ConcreteModel(model="dummy", metadata=metadata)

        # Check last dimension (channels = 3)
        rng = np.random.default_rng(42)
        correct_features = rng.random((5, 10, 15, 3)).astype(np.float32)
        model.validate_input(correct_features)  # Should not raise

        # Wrong channels
        wrong_features = rng.random((5, 10, 15, 1)).astype(np.float32)
        with pytest.raises(ValueError):
            model.validate_input(wrong_features)

    def test_concrete_predict_method_enforcement(self) -> None:
        """Subclasses must implement predict method."""
        class IncompleteModel(BaseModel):
            # Missing predict method implementation
            pass

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteModel(None, {})  # type: ignore

    def test_predict_method_signature(self) -> None:
        """Test predict method signature and type enforcement."""
        model = ConcreteModel(
            model="dummy",
            metadata={"input_shape": [None, 3]}
        )

        # Correct usage
        features = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        result = model.predict(features)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.shape == (1,)  # Single sample result
        assert np.allclose(result, [4.0])  # (1+2+3)/3 * 2 = 2 * 2 = 4

    def test_predict_with_1d_input(self) -> None:
        """Test predict method with 1D feature arrays."""
        model = ConcreteModel(
            model="dummy",
            metadata={"input_shape": [None, 4]}
        )

        features_1d = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        result = model.predict(features_1d)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.ndim == 1
        assert np.allclose(result, [2.0, 4.0, 6.0, 8.0])  # features * 2

    def test_predict_with_2d_input(self) -> None:
        """Test predict method with 2D feature arrays (batch)."""
        model = ConcreteModel(
            model="dummy",
            metadata={"input_shape": [None, 3]}
        )

        features_2d = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ], dtype=np.float32)
        result = model.predict(features_2d)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.shape == (2,)  # Two samples
        # Mean of [1,2,3] * 2 = 2 * 2 = 4
        # Mean of [4,5,6] * 2 = 5 * 2 = 10
        assert np.allclose(result, [4.0, 10.0])

    def test_predict_validates_input(self) -> None:
        """Test that predict method calls validate_input internally."""
        model = ConcreteModel(
            model="dummy",
            metadata={"input_shape": [None, 5]}
        )

        # Wrong shape should be caught by validate_input called from predict
        wrong_features = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)  # 3 features, expect 5

        with pytest.raises(ValueError, match="Input validation failed"):
            model.predict(wrong_features)

    def test_edge_case_empty_features(self) -> None:
        """Edge case: empty feature arrays."""
        model = ConcreteModel(
            model="dummy",
            metadata={}  # No input validation
        )

        empty_features = np.array([], dtype=np.float32).reshape(0, 0)

        # This should work with our simple implementation
        # (though real models might handle this differently)
        result = model.predict(empty_features)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32

    def test_edge_case_nan_in_features(self) -> None:
        """Edge case: NaN values in features."""
        model = ConcreteModel(
            model="dummy",
            metadata={"input_shape": [None, 2]}
        )

        features_with_nan = np.array([[1.0, np.nan]], dtype=np.float32)
        result = model.predict(features_with_nan)

        # Result should contain NaN due to our implementation
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert np.isnan(result[0])

    def test_edge_case_inf_in_features(self) -> None:
        """Edge case: infinite values in features."""
        model = ConcreteModel(
            model="dummy",
            metadata={"input_shape": [None, 2]}
        )

        features_with_inf = np.array([[1.0, np.inf]], dtype=np.float32)
        result = model.predict(features_with_inf)

        # Result should contain inf
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert np.isinf(result[0])

    def test_type_safety_float32_enforcement(self) -> None:
        """Type system enforces float32 for features and predictions."""
        model = ConcreteModel(
            model="dummy",
            metadata={"input_shape": [None, 3]}
        )

        # Our implementation should work with float32
        features_f32 = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        result = model.predict(features_f32)

        assert result.dtype == np.float32

        # Test that float64 input is handled (should still work)
        features_f64 = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
        result = model.predict(features_f64.astype(np.float32))

        assert result.dtype == np.float32

    def test_metadata_mutability(self) -> None:
        """Metadata is directly mutable through the property (current behavior)."""
        original_metadata = {
            "model_type": "test",
            "version": "1.0"
        }
        model = ConcreteModel(model="dummy", metadata=original_metadata)

        # Getting metadata returns same reference, so modifications affect original
        retrieved_metadata = model.metadata
        retrieved_metadata["modified"] = True

        # Changes are reflected in the model metadata
        assert "modified" in model.metadata
        assert model.metadata["modified"] is True

    def test_model_storage(self) -> None:
        """Model object is stored and accessible internally."""
        test_model_obj = {"type": "test", "weights": [1, 2, 3]}
        metadata = {"model_type": "test"}

        model = ConcreteModel(model=test_model_obj, metadata=metadata)

        # Access via private attribute (for testing internal state)
        assert model._model is test_model_obj
        assert model._model["type"] == "test"

    def test_string_representation_fields(self) -> None:
        """Model properties return expected string types."""
        metadata = {
            "model_id": 42,  # Non-string
            "version": 1.5,   # Non-string
            "model_type": "test"
        }

        model = ConcreteModel(model="dummy", metadata=metadata)

        # model_id should be string
        assert isinstance(model.model_id, str)
        assert model.model_id == "42"
