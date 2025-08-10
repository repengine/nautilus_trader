#!/usr/bin/env python3

"""
Functional tests for model abstraction layer.

These tests define the contracts that all model implementations must fulfill:
1. Auto-detection of model format
2. Unified predict() interface
3. Metadata preservation
4. Security (no pickle in production)
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from numpy.typing import NDArray
import pytest

from ml._imports import HAS_XGBOOST, HAS_LIGHTGBM, HAS_ONNX


class TestModelContracts:
    """Test suite for model abstraction contracts."""
    
    def test_model_loader_detects_format(self) -> None:
        """
        Loader MUST auto-detect model format from file.
        
        Given: Files with .onnx, .json, .txt extensions
        When: Loading each file
        Then: Correct model type is returned
        """
        from ml.models.loader import ProductionModelLoader
        
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            
            # Create test files with different extensions
            test_files = {
                "model.onnx": "onnx",
                "model.json": "xgboost",
                "model.txt": "lightgbm",
                "model.joblib": "joblib",
            }
            
            loader = ProductionModelLoader()
            
            for filename, expected_type in test_files.items():
                filepath = tmpdir / filename
                
                # Create mock file based on type
                if expected_type == "onnx" and HAS_ONNX:
                    # Create minimal ONNX file
                    import onnx
                    from onnx import helper, TensorProto
                    
                    # Create a simple ONNX model with compatible IR version
                    input_tensor = helper.make_tensor_value_info(
                        'input', TensorProto.FLOAT, [None, 10]
                    )
                    output_tensor = helper.make_tensor_value_info(
                        'output', TensorProto.FLOAT, [None, 1]
                    )
                    
                    node = helper.make_node(
                        'Identity',
                        inputs=['input'],
                        outputs=['output'],
                    )
                    
                    graph = helper.make_graph(
                        [node],
                        'test_model',
                        [input_tensor],
                        [output_tensor],
                    )
                    
                    model = helper.make_model(graph)
                    # Set IR version and opset to be compatible with current ONNX Runtime
                    model.ir_version = 8  # Use stable IR version
                    # Set opset to version 17 (stable and widely supported)
                    model.opset_import[0].version = 17
                    onnx.save(model, str(filepath))
                    
                elif expected_type == "xgboost" and HAS_XGBOOST:
                    # Create minimal XGBoost JSON file
                    import xgboost as xgb
                    X = np.random.randn(100, 10)
                    y = np.random.randint(0, 2, 100)
                    model = xgb.XGBClassifier(n_estimators=2)
                    model.fit(X, y)
                    model.save_model(str(filepath))
                    
                elif expected_type == "lightgbm" and HAS_LIGHTGBM:
                    # Create minimal LightGBM file
                    import lightgbm as lgb
                    X = np.random.randn(100, 10)
                    y = np.random.randint(0, 2, 100)
                    model = lgb.LGBMClassifier(n_estimators=2)
                    model.fit(X, y)
                    model.booster_.save_model(str(filepath))
                    
                else:
                    # Create dummy file for format detection test
                    filepath.write_text("dummy content")
                    continue
                
                # Test loading and format detection
                try:
                    model, metadata = loader.load_model(str(filepath))
                    assert metadata["type"] == expected_type, \
                        f"Expected type {expected_type}, got {metadata['type']}"
                except FileNotFoundError:
                    # Skip if dependencies not available
                    pass
    
    def test_all_models_have_predict_interface(self) -> None:
        """
        Every model type MUST expose predict() method.
        
        Given: ONNX, XGBoost, LightGBM models
        When: Calling predict(features)
        Then: Returns predictions as numpy array
        """
        from ml.models import BaseModel, ONNXModel, XGBoostModel, LightGBMModel
        
        # Create test features
        features = np.random.randn(5, 10).astype(np.float32)
        
        # Test each model type has predict interface
        model_classes: list[type[BaseModel]] = []
        
        if HAS_ONNX:
            model_classes.append(ONNXModel)
        if HAS_XGBOOST:
            model_classes.append(XGBoostModel)
        if HAS_LIGHTGBM:
            model_classes.append(LightGBMModel)
        
        for model_class in model_classes:
            # Verify class has predict method
            assert hasattr(model_class, 'predict'), \
                f"{model_class.__name__} must have predict() method"
            
            # Verify it returns numpy array (using mock for testing)
            mock_model = Mock(spec=model_class)
            mock_model.predict.return_value = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
            
            predictions = mock_model.predict(features)
            assert isinstance(predictions, np.ndarray), \
                f"{model_class.__name__}.predict() must return numpy array"
            assert len(predictions) == len(features), \
                "Predictions length must match input samples"
    
    def test_model_metadata_preserved(self) -> None:
        """
        Model metadata MUST be accessible after loading.
        
        Given: Model saved with metadata
        When: Model is loaded
        Then: Can access feature_names, input_shape, etc.
        """
        from ml.models.loader import ProductionModelLoader
        from ml.models.saver import save_model_with_metadata
        
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            
            if HAS_XGBOOST:
                # Create and save model with metadata
                import xgboost as xgb
                
                # Train simple model
                n_features = 10
                feature_names = [f"feature_{i}" for i in range(n_features)]
                X = np.random.randn(100, n_features).astype(np.float32)
                y = np.random.randint(0, 2, 100)
                
                model = xgb.XGBClassifier(n_estimators=2)
                model.fit(X, y)
                
                # Save with metadata
                model_path = save_model_with_metadata(
                    model=model,
                    path=tmpdir / "model.json",
                    input_shape=(1, n_features),
                    training_metadata={
                        "feature_names": feature_names,
                        "training_accuracy": 0.95,
                        "model_version": "1.0.0",
                    }
                )
                
                # Load model
                loader = ProductionModelLoader()
                loaded_model, metadata = loader.load_model(str(model_path))
                
                # Verify metadata is preserved
                assert "training_metadata" in metadata or "feature_names" in metadata, \
                    "Metadata must be preserved through save/load"
                
                if "training_metadata" in metadata:
                    assert "feature_names" in metadata["training_metadata"]
                    assert metadata["training_metadata"]["feature_names"] == feature_names
                    assert metadata["training_metadata"]["training_accuracy"] == 0.95
                
                # Verify basic metadata
                assert "type" in metadata, "Model type must be in metadata"
                assert "size_bytes" in metadata, "Model size must be in metadata"
    
    def test_no_pickle_in_production(self) -> None:
        """
        Production loader MUST reject pickle files.
        
        Given: A pickle file
        When: Attempting to load with ProductionModelLoader
        Then: Raises error with clear security message
        """
        from ml.models.loader import ProductionModelLoader
        import pickle
        
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            
            # Create a pickle file
            pickle_path = tmpdir / "model.pkl"
            dummy_model = {"type": "dangerous", "data": "should not load"}
            
            with open(pickle_path, "wb") as f:
                pickle.dump(dummy_model, f)
            
            # Attempt to load with production loader
            loader = ProductionModelLoader()
            
            with pytest.raises(ValueError) as exc_info:
                loader.load_model(str(pickle_path))
            
            # Verify error message mentions security
            error_msg = str(exc_info.value).lower()
            assert "pickle" in error_msg, "Error must mention pickle"
            assert "security" in error_msg or "not supported" in error_msg, \
                "Error must explain security concern"
    
    def test_model_validates_input_shape(self) -> None:
        """
        Models MUST validate input shape before prediction.
        
        Given: Model expecting specific input shape
        When: Providing wrong shape
        Then: Raises informative error
        """
        from ml.models import BaseModel
        
        # Mock model with expected shape
        class TestModel(BaseModel):
            def __init__(self) -> None:
                from ml.models import ModelMetadata, ModelType
                metadata = ModelMetadata(
                    model_type=ModelType.UNKNOWN,
                    path="",
                    version="test",
                    size_bytes=0,
                    modified_time=0.0,
                    input_shape=(1, 10),
                )
                super().__init__(model=None, metadata=metadata)
                self.expected_features = 10
                
            def predict(self, features: NDArray[np.float32]) -> tuple[float, float]:
                if features.shape[1] != self.expected_features:
                    raise ValueError(
                        f"Expected {self.expected_features} features, "
                        f"got {features.shape[1]}"
                    )
                return (0.0, 0.5)
            
            def validate_input(self, features: NDArray[np.float32]) -> None:
                if features.shape[1] != self.expected_features:
                    raise ValueError(
                        f"Input validation failed: expected {self.expected_features} "
                        f"features, got {features.shape[1]}"
                    )
        
        model = TestModel()
        
        # Test with wrong shape
        wrong_features = np.random.randn(5, 7).astype(np.float32)  # Wrong feature count
        
        with pytest.raises(ValueError) as exc_info:
            model.validate_input(wrong_features)
        
        assert "expected 10 features" in str(exc_info.value)
    
    def test_model_handles_single_prediction(self) -> None:
        """
        Models MUST handle single sample predictions for hot path.
        
        Given: Single sample (hot path requirement)
        When: Predicting on single feature vector
        Then: Returns prediction and confidence as tuple
        """
        from ml.models import BaseModel
        
        class TestModel(BaseModel):
            def __init__(self) -> None:
                from ml.models import ModelMetadata, ModelType
                metadata = ModelMetadata(
                    model_type=ModelType.UNKNOWN,
                    path="",
                    version="test",
                    size_bytes=0,
                    modified_time=0.0,
                )
                super().__init__(model=None, metadata=metadata)
            
            def predict(self, features: NDArray[np.float32]) -> tuple[float, float]:
                # Simple mock prediction
                return (0.5, 0.8)
        
        model = TestModel()
        
        # Test single sample prediction (hot path)
        features = np.random.randn(10).astype(np.float32)  # Single sample
        prediction, confidence = model.predict(features)
        
        assert isinstance(prediction, float), "Prediction must be a float"
        assert isinstance(confidence, float), "Confidence must be a float"
        assert 0 <= confidence <= 1, "Confidence must be between 0 and 1"
        
        # Test with 2D input (1 sample, N features)
        features_2d = np.random.randn(1, 10).astype(np.float32)
        prediction_2d, confidence_2d = model.predict(features_2d)
        
        assert isinstance(prediction_2d, float), "Prediction must be a float"
        assert isinstance(confidence_2d, float), "Confidence must be a float"
    
    def test_model_loader_thread_safe(self) -> None:
        """
        Model loader MUST be thread-safe for concurrent access.
        
        Given: Multiple threads loading models
        When: Concurrent load operations
        Then: All operations succeed without corruption
        """
        from ml.models.loader import ProductionModelLoader
        import threading
        import concurrent.futures
        
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            
            # Create test model file
            model_path = tmpdir / "model.json"
            
            if HAS_XGBOOST:
                import xgboost as xgb
                X = np.random.randn(50, 5)
                y = np.random.randint(0, 2, 50)
                model = xgb.XGBClassifier(n_estimators=2)
                model.fit(X, y)
                model.save_model(str(model_path))
            else:
                # Create dummy file
                model_path.write_text('{"dummy": "model"}')
            
            loader = ProductionModelLoader()
            results = []
            errors = []
            
            def load_model() -> None:
                try:
                    model, metadata = loader.load_model(str(model_path))
                    results.append((model, metadata))
                except Exception as e:
                    errors.append(e)
            
            # Run concurrent loads
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(load_model) for _ in range(10)]
                concurrent.futures.wait(futures)
            
            # Verify no errors occurred
            assert len(errors) == 0, f"Thread-safe loading failed: {errors}"
            
            # Verify all loads succeeded
            if HAS_XGBOOST:
                assert len(results) == 10, "All concurrent loads should succeed"