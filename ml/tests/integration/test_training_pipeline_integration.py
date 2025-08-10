#!/usr/bin/env python3

"""
Integration test demonstrating the complete training-to-inference pipeline.

This test verifies that:
1. Models can be trained using XGBoostTrainer/LightGBMTrainer
2. Models are saved in production formats (never pickle)
3. Models can be loaded using ProductionModelLoader
4. Loaded models can make predictions
5. Metadata is properly preserved
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml._imports import HAS_XGBOOST, HAS_LIGHTGBM


class TestTrainingPipelineIntegration:
    """Integration tests for the complete training pipeline."""
    
    def test_xgboost_training_to_inference_pipeline(self) -> None:
        """
        Test the complete XGBoost training-to-inference pipeline.
        
        This verifies:
        1. Training with XGBoostTrainer
        2. Saving in JSON format with metadata
        3. Loading with ProductionModelLoader
        4. Making predictions with loaded model
        """
        if not HAS_XGBOOST:
            pytest.skip("XGBoost not installed")
            
        from ml.training.xgboost import XGBoostTrainer
        from ml.config.xgboost import XGBoostTrainingConfig
        from ml.models.loader import ProductionModelLoader
        
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            
            # 1. Train model
            config = XGBoostTrainingConfig(
                data_source="test_data",
                n_estimators=10,
                max_depth=3,
                save_model_path=str(tmpdir / "model.json"),
            )
            
            trainer = XGBoostTrainer(config)
            
            # Create training data
            n_samples, n_features = 200, 15
            feature_names = [f"feature_{i}" for i in range(n_features)]
            X = np.random.randn(n_samples, n_features)
            y = (X[:, 0] + X[:, 1] * 0.5 + np.random.randn(n_samples) * 0.1 > 0).astype(int)
            
            df = pd.DataFrame(X, columns=feature_names)
            df["target"] = y
            
            # Train
            result = trainer.train(df)
            
            # 2. Save model using new production methods
            model_path = tmpdir / "model.json"
            trainer.save_model(model_path)
            
            # Verify JSON format (not pickle)
            assert model_path.exists()
            assert model_path.suffix == ".json"
            
            # Verify metadata file
            metadata_path = model_path.with_suffix(".json.meta.json")
            assert metadata_path.exists()
            
            # 3. Load with ProductionModelLoader
            loader = ProductionModelLoader()
            loaded_model, metadata = loader.load_model(str(model_path))
            
            # 4. Verify metadata
            assert metadata["type"] == "xgboost"
            assert "training_metadata" in metadata or "feature_names" in metadata
            
            # 5. Make predictions
            test_X = np.random.randn(10, n_features).astype(np.float32)
            
            # Create DMatrix for XGBoost prediction
            import xgboost as xgb
            dtest = xgb.DMatrix(test_X)
            predictions = loaded_model.predict(dtest)
            
            assert predictions is not None
            assert len(predictions) == 10
            
    def test_lightgbm_training_to_inference_pipeline(self) -> None:
        """
        Test the complete LightGBM training-to-inference pipeline.
        
        This verifies:
        1. Training with LightGBMTrainer
        2. Saving in TXT format with metadata
        3. Loading with ProductionModelLoader
        4. Making predictions with loaded model
        """
        if not HAS_LIGHTGBM:
            pytest.skip("LightGBM not installed")
            
        from ml.training.lightgbm import LightGBMTrainer
        from ml.config.lightgbm import LightGBMTrainingConfig
        from ml.models.loader import ProductionModelLoader
        
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            
            # 1. Train model
            config = LightGBMTrainingConfig(
                data_source="test_data",
                n_estimators=10,
                num_leaves=31,
                save_model_path=str(tmpdir / "model.txt"),
            )
            
            trainer = LightGBMTrainer(config)
            
            # Create training data
            n_samples, n_features = 200, 15
            feature_names = [f"feature_{i}" for i in range(n_features)]
            X = np.random.randn(n_samples, n_features)
            y = (X[:, 0] + X[:, 1] * 0.5 + np.random.randn(n_samples) * 0.1 > 0).astype(int)
            
            df = pd.DataFrame(X, columns=feature_names)
            df["target"] = y
            
            # Train
            result = trainer.train(df)
            
            # 2. Save model using new production methods
            model_path = tmpdir / "model.txt"
            trainer.save_model(model_path)
            
            # Verify TXT format (not pickle)
            assert model_path.exists()
            assert model_path.suffix == ".txt"
            
            # Verify metadata file
            metadata_path = model_path.with_suffix(".txt.meta.json")
            assert metadata_path.exists()
            
            # 3. Load with ProductionModelLoader
            loader = ProductionModelLoader()
            loaded_model, metadata = loader.load_model(str(model_path))
            
            # 4. Verify metadata
            assert metadata["type"] == "lightgbm"
            assert "training_metadata" in metadata or "feature_names" in metadata
            
            # 5. Make predictions
            test_X = np.random.randn(10, n_features).astype(np.float32)
            predictions = loaded_model.predict(test_X)
            
            assert predictions is not None
            assert len(predictions) == 10
            
    def test_model_export_mixin_functionality(self) -> None:
        """
        Test the ModelExportMixin functionality.
        
        This verifies:
        1. save_for_production() method works
        2. validate_inference_compatibility() works
        3. Proper metadata is saved
        """
        if not HAS_XGBOOST:
            pytest.skip("XGBoost not installed")
            
        from ml.training.xgboost import XGBoostTrainer
        from ml.config.xgboost import XGBoostTrainingConfig
        
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            
            # Train a simple model
            config = XGBoostTrainingConfig(
                data_source="test_data",
                n_estimators=5,
                max_depth=2,
            )
            
            trainer = XGBoostTrainer(config)
            
            # Training data
            df = pd.DataFrame({
                "f1": np.random.randn(100),
                "f2": np.random.randn(100),
                "f3": np.random.randn(100),
                "target": np.random.randint(0, 2, 100),
            })
            
            trainer.train(df)
            
            # Test save_for_production
            prod_path = tmpdir / "production_model"
            saved_path = trainer.save_for_production(prod_path, format="native")
            
            assert saved_path.exists()
            assert saved_path.suffix in {".json", ".xgb"}
            
            # Test validate_inference_compatibility
            test_features = np.random.randn(5, 3).astype(np.float32)
            is_compatible = trainer.validate_inference_compatibility(
                saved_path,
                test_features,
            )
            
            assert is_compatible, "Model should be inference-compatible"
            
            # Verify metadata
            assert trainer.get_feature_names() == ["f1", "f2", "f3"]
            assert isinstance(trainer.get_training_metadata(), dict)