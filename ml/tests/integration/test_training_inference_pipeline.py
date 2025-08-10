#!/usr/bin/env python3

"""
Test the complete training to inference pipeline.

This validates that models trained with our training modules can be
successfully loaded and used by inference actors.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from ml._imports import HAS_XGBOOST, HAS_LIGHTGBM, HAS_ONNX
from ml.models.loader import ProductionModelLoader
from ml.models.saver import save_model_with_metadata, convert_to_onnx
from ml.training.model_exporter import ModelExportMixin


class TestTrainingInferencePipeline:
    """Test the complete ML pipeline from training to inference."""
    
    @pytest.mark.skipif(not HAS_XGBOOST, reason="XGBoost not installed")
    def test_xgboost_native_pipeline(self) -> None:
        """Test XGBoost model can be saved and loaded in native format."""
        import xgboost as xgb
        
        # Create and train a simple model
        X = np.random.randn(100, 10).astype(np.float32)
        y = np.random.randint(0, 2, 100)
        
        model = xgb.XGBClassifier(n_estimators=10, max_depth=3)
        model.fit(X, y)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save using production saver
            model_path = save_model_with_metadata(
                model=model,
                path=Path(tmpdir) / "model.xgb",
                input_shape=(1, 10),
                training_metadata={
                    "feature_names": [f"feature_{i}" for i in range(10)],
                    "training_accuracy": 0.95,
                },
            )
            
            # Load using production loader
            loader = ProductionModelLoader()
            loaded_model, metadata = loader.load_model(str(model_path))
            
            # Verify metadata
            assert metadata["type"] == "xgboost"
            assert metadata["n_features"] == 10
            assert "feature_names" in metadata.get("training_metadata", {})
            
            # Verify prediction works
            test_features = np.random.randn(1, 10).astype(np.float32)
            
            # Handle raw XGBoost Booster
            import xgboost as xgb
            dtest = xgb.DMatrix(test_features)
            prediction = loaded_model.predict(dtest)
            assert prediction is not None
            assert len(prediction) == 1  # One sample
    
    @pytest.mark.skipif(not HAS_LIGHTGBM, reason="LightGBM not installed")
    def test_lightgbm_native_pipeline(self) -> None:
        """Test LightGBM model can be saved and loaded in native format."""
        import lightgbm as lgb
        
        # Create and train a simple model
        X = np.random.randn(100, 10).astype(np.float32)
        y = np.random.randint(0, 2, 100)
        
        model = lgb.LGBMClassifier(n_estimators=10, num_leaves=10)
        model.fit(X, y)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save using production saver
            model_path = save_model_with_metadata(
                model=model,
                path=Path(tmpdir) / "model.lgb",
                input_shape=(1, 10),
                training_metadata={
                    "feature_names": [f"feature_{i}" for i in range(10)],
                    "training_auc": 0.88,
                },
            )
            
            # Load using production loader
            loader = ProductionModelLoader()
            loaded_model, metadata = loader.load_model(str(model_path))
            
            # Verify metadata
            assert metadata["type"] == "lightgbm"
            assert metadata["n_features"] == 10
            
            # Verify prediction works
            test_features = np.random.randn(1, 10).astype(np.float32)
            
            # Handle raw LightGBM Booster
            prediction = loaded_model.predict(test_features)
            assert prediction is not None
            assert len(prediction) == 1  # One sample
    
    @pytest.mark.skipif(not HAS_ONNX or not HAS_XGBOOST, reason="ONNX or XGBoost not installed")
    def test_onnx_conversion_pipeline(self) -> None:
        """Test model can be converted to ONNX and loaded."""
        import xgboost as xgb
        
        # Create and train a simple model
        X = np.random.randn(100, 10).astype(np.float32)
        y = np.random.randint(0, 2, 100)
        
        model = xgb.XGBClassifier(n_estimators=10, max_depth=3)
        model.fit(X, y)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Convert to ONNX
            onnx_path = convert_to_onnx(
                model=model,
                sample_input=X[:1],
                output_path=Path(tmpdir) / "model.onnx",
            )
            
            # Load using production loader
            loader = ProductionModelLoader()
            loaded_model, metadata = loader.load_model(str(onnx_path))
            
            # Verify metadata
            assert metadata["type"] == "onnx"
            assert len(metadata["input_names"]) > 0
            
            # Verify prediction works
            test_features = np.random.randn(1, 10).astype(np.float32)
            # ONNX models need 2D input
            outputs = loaded_model.run(None, {
                metadata["input_names"][0]: test_features
            })
            assert outputs is not None
    
    def test_model_export_mixin(self) -> None:
        """Test the ModelExportMixin interface."""
        
        class TestTrainer(ModelExportMixin):
            def __init__(self) -> None:
                self.model = "dummy_model"
                self.features = ["f1", "f2", "f3"]
                self.metadata = {"accuracy": 0.9}
            
            def get_model(self) -> str:
                return self.model
            
            def get_feature_names(self) -> list[str]:
                return self.features
            
            def get_training_metadata(self) -> dict[str, float]:
                return self.metadata
        
        trainer = TestTrainer()
        
        # Test methods are accessible
        assert trainer.get_model() == "dummy_model"
        assert trainer.get_feature_names() == ["f1", "f2", "f3"]
        assert trainer.get_training_metadata() == {"accuracy": 0.9}
    
    @pytest.mark.skipif(not HAS_XGBOOST, reason="XGBoost not installed")
    def test_complete_training_to_actor_pipeline(self) -> None:
        """Test the complete pipeline from training to actor loading.
        
        This test validates the functional requirements:
        1. A model can be trained using XGBoostTrainer
        2. The trained model can be saved in production format
        3. The saved model can be loaded by ProductionModelLoader
        4. The loaded model produces predictions
        5. Metadata is preserved through the pipeline
        """
        import xgboost as xgb
        from ml.training.xgboost import XGBoostTrainer
        from ml.config.xgboost import XGBoostTrainingConfig
        
        # Setup training data
        n_samples, n_features = 100, 10
        X = np.random.randn(n_samples, n_features).astype(np.float32)
        y = np.random.randint(0, 2, n_samples)  # Binary classification
        feature_names = [f"feature_{i}" for i in range(n_features)]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.json"
            
            # FUNCTIONAL REQUIREMENT 1: Train a model
            config = XGBoostTrainingConfig(
                data_source="test_data",
                n_estimators=10,
                max_depth=3,
                objective="binary:logistic",
                save_model_path=str(model_path),
            )
            
            trainer = XGBoostTrainer(config)
            
            # Create a DataFrame for training (trainer's expected input format)
            import pandas as pd
            df = pd.DataFrame(X, columns=feature_names)
            df["target"] = y
            
            trainer.train(df)
            
            # FUNCTIONAL REQUIREMENT 2: Save the model
            trainer.save_model(model_path)
            assert model_path.exists(), "Model file should be created"
            
            # FUNCTIONAL REQUIREMENT 3: Load with ProductionModelLoader
            loader = ProductionModelLoader()
            loaded_model, metadata = loader.load_model(str(model_path))
            
            # FUNCTIONAL REQUIREMENT 4: Verify loaded model can make predictions
            assert loaded_model is not None, "Model should be loaded"
            
            # Create test data for prediction
            test_X = np.random.randn(5, n_features).astype(np.float32)
            
            # Make predictions (implementation detail: XGBoost needs DMatrix with feature names)
            import xgboost as xgb
            dtest = xgb.DMatrix(test_X, feature_names=feature_names)
            predictions = loaded_model.predict(dtest)
            
            # Validate predictions have expected shape and range
            assert predictions.shape == (5,), f"Expected 5 predictions, got {predictions.shape}"
            assert np.all((predictions >= 0) & (predictions <= 1)), "Binary predictions should be in [0,1]"
            
            # FUNCTIONAL REQUIREMENT 5: Verify metadata preservation
            assert metadata["type"] == "xgboost", "Model type should be identified"
            assert "n_features" in metadata or "input_shape" in metadata, "Feature count should be preserved"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])