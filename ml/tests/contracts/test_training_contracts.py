#!/usr/bin/env python3

"""
Functional tests for training pipeline.

These tests define the contracts that training implementations must fulfill:
1. Export to production formats (ONNX or native)
2. Save comprehensive metadata
3. Ensure training-to-inference compatibility
4. Never use pickle for production models
"""

import json
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml._imports import HAS_XGBOOST, HAS_LIGHTGBM, HAS_ONNX


class TestTrainingContracts:
    """Test suite for training pipeline contracts."""
    
    def test_trainer_exports_to_production_format(self) -> None:
        """
        Trainers MUST export to ONNX or native format.
        
        Given: Trained XGBoost/LightGBM model
        When: Calling save_model()
        Then: Saves as .json/.txt (native) or .onnx, NOT pickle
        """
        if HAS_XGBOOST:
            from ml.training.xgboost import XGBoostTrainer
            from ml.config.xgboost import XGBoostTrainingConfig
            
            with tempfile.TemporaryDirectory() as tmpdir_str:
                tmpdir = Path(tmpdir_str)
                
                # Configure trainer
                config = XGBoostTrainingConfig(
                    data_source="test_data",
                    n_estimators=2,
                    max_depth=3,
                    save_model_path=str(tmpdir / "model.json"),
                )
                
                trainer = XGBoostTrainer(config)
                
                # Create training data
                n_samples, n_features = 100, 10
                X = np.random.randn(n_samples, n_features)
                y = np.random.randint(0, 2, n_samples)
                df = pd.DataFrame(X, columns=[f"f_{i}" for i in range(n_features)])
                df["target"] = y
                
                # Train and save
                trainer.train(df)
                model_path = tmpdir / "model.json"
                trainer.save_model(model_path)
                
                # Verify saved format
                assert model_path.exists(), "Model file must be created"
                
                # Ensure it's NOT pickle
                try:
                    with open(model_path, "rb") as f:
                        pickle.load(f)
                    pytest.fail("Model should NOT be saved as pickle")
                except (pickle.UnpicklingError, json.JSONDecodeError):
                    # Good - it's not pickle format
                    pass
                
                # Verify it's valid JSON (XGBoost native format)
                with open(model_path, "r") as f:
                    model_data = json.load(f)
                    assert "learner" in model_data or "booster" in str(model_data), \
                        "Should be valid XGBoost JSON format"
    
    def test_trainer_saves_metadata(self) -> None:
        """
        Training metadata MUST be saved with model.
        
        Given: Training with specific features
        When: Model is saved
        Then: .meta.json file contains features, metrics
        """
        if HAS_XGBOOST:
            from ml.training.xgboost import XGBoostTrainer
            from ml.config.xgboost import XGBoostTrainingConfig
            
            with tempfile.TemporaryDirectory() as tmpdir_str:
                tmpdir = Path(tmpdir_str)
                
                # Configure trainer
                config = XGBoostTrainingConfig(
                    data_source="test_data",
                    n_estimators=2,
                    max_depth=3,
                    save_model_path=str(tmpdir / "model.json"),
                )
                
                trainer = XGBoostTrainer(config)
                
                # Create training data with specific features
                feature_names = ["sma_10", "rsi_14", "volume_ratio", "price_change"]
                n_samples = 100
                X = np.random.randn(n_samples, len(feature_names))
                y = np.random.randint(0, 2, n_samples)
                df = pd.DataFrame(X, columns=feature_names)
                df["target"] = y
                
                # Train and save
                trainer.train(df)
                model_path = tmpdir / "model.json"
                trainer.save_model(model_path)
                
                # Check for metadata file
                metadata_path = model_path.with_suffix(".meta")
                if not metadata_path.exists():
                    # Try alternative metadata location
                    metadata_path = model_path.with_suffix(".json.meta.json")
                
                assert metadata_path.exists() or (model_path.parent / "metadata.json").exists(), \
                    "Metadata file must be saved with model"
                
                # If metadata exists, verify contents
                if metadata_path.exists():
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                    
                    # Verify essential metadata
                    assert "feature_names" in metadata or \
                           ("training_metadata" in metadata and 
                            "feature_names" in metadata["training_metadata"]), \
                        "Feature names must be in metadata"
                    
                    # Check for training metrics (if available)
                    if "training_metrics" in metadata:
                        assert isinstance(metadata["training_metrics"], dict), \
                            "Training metrics should be a dictionary"
    
    def test_training_to_inference_compatibility(self) -> None:
        """
        Models trained MUST be loadable by actors.
        
        Given: Model trained with XGBoostTrainer
        When: Loading with ProductionModelLoader
        Then: Model loads and can make predictions
        """
        if HAS_XGBOOST:
            from ml.training.xgboost import XGBoostTrainer
            from ml.config.xgboost import XGBoostTrainingConfig
            from ml.models.loader import ProductionModelLoader
            
            with tempfile.TemporaryDirectory() as tmpdir_str:
                tmpdir = Path(tmpdir_str)
                
                # Train model
                config = XGBoostTrainingConfig(
                    data_source="test_data",
                    n_estimators=5,
                    max_depth=3,
                    save_model_path=str(tmpdir / "model.json"),
                )
                
                trainer = XGBoostTrainer(config)
                
                # Training data
                n_samples, n_features = 100, 10
                feature_names = [f"feature_{i}" for i in range(n_features)]
                X = np.random.randn(n_samples, n_features)
                y = np.random.randint(0, 2, n_samples)
                df = pd.DataFrame(X, columns=feature_names)
                df["target"] = y
                
                # Train and save
                trainer.train(df)
                model_path = tmpdir / "model.json"
                trainer.save_model(model_path)
                
                # Load with production loader
                loader = ProductionModelLoader()
                loaded_model, metadata = loader.load_model(str(model_path))
                
                # Verify model can make predictions
                test_features = np.random.randn(5, n_features).astype(np.float32)
                
                # For XGBoost, need to use DMatrix
                import xgboost as xgb
                dtest = xgb.DMatrix(test_features, feature_names=feature_names)
                predictions = loaded_model.predict(dtest)
                
                assert predictions is not None, "Model must make predictions"
                assert len(predictions) == 5, "Must predict for all samples"
    
    def test_trainer_never_uses_pickle_for_production(self) -> None:
        """
        Production save methods MUST never use pickle.
        
        Given: Any trainer's save_model method
        When: Saving for production
        Then: Never creates pickle files
        """
        if HAS_XGBOOST:
            from ml.training.xgboost import XGBoostTrainer
            from ml.config.xgboost import XGBoostTrainingConfig
            
            with tempfile.TemporaryDirectory() as tmpdir_str:
                tmpdir = Path(tmpdir_str)
                
                # Configure for production save
                config = XGBoostTrainingConfig(
                    data_source="test_data",
                    n_estimators=2,
                    save_model_path=str(tmpdir / "production_model"),
                )
                
                trainer = XGBoostTrainer(config)
                
                # Train
                df = pd.DataFrame({
                    "f1": np.random.randn(50),
                    "f2": np.random.randn(50),
                    "target": np.random.randint(0, 2, 50),
                })
                trainer.train(df)
                
                # Try different save methods
                save_paths = [
                    tmpdir / "model.json",  # Native format
                    tmpdir / "model",  # No extension
                ]
                
                for save_path in save_paths:
                    trainer.save_model(save_path)
                    
                    # Check no pickle files created
                    pickle_extensions = [".pkl", ".pickle", ".joblib"]
                    for ext in pickle_extensions:
                        potential_pickle = save_path.with_suffix(ext)
                        assert not potential_pickle.exists(), \
                            f"Should not create pickle file: {potential_pickle}"
    
    def test_trainer_supports_onnx_export(self) -> None:
        """
        Trainers SHOULD support ONNX export for cross-platform deployment.
        
        Given: Trained model
        When: Exporting to ONNX
        Then: Creates valid ONNX file
        """
        if HAS_XGBOOST and HAS_ONNX:
            from ml.training.xgboost import XGBoostTrainer
            from ml.config.xgboost import XGBoostTrainingConfig
            from ml.models.saver import convert_to_onnx
            
            with tempfile.TemporaryDirectory() as tmpdir_str:
                tmpdir = Path(tmpdir_str)
                
                # Train model
                config = XGBoostTrainingConfig(
                    data_source="test_data",
                    n_estimators=2,
                    max_depth=3,
                )
                
                trainer = XGBoostTrainer(config)
                
                # Training data
                n_features = 10
                df = pd.DataFrame({
                    **{f"f_{i}": np.random.randn(50) for i in range(n_features)},
                    "target": np.random.randint(0, 2, 50),
                })
                trainer.train(df)
                
                # Export to ONNX if method exists
                if hasattr(trainer, 'export_to_onnx'):
                    onnx_path = tmpdir / "model.onnx"
                    trainer.export_to_onnx(onnx_path)
                    assert onnx_path.exists(), "ONNX file should be created"
                else:
                    # Try using the saver utility
                    sample_input = np.random.randn(1, n_features).astype(np.float32)
                    onnx_path = convert_to_onnx(
                        model=trainer._booster,  # Access internal model
                        sample_input=sample_input,
                        output_path=tmpdir / "model.onnx",
                    )
                    assert onnx_path.exists(), "ONNX conversion should succeed"
    
    def test_trainer_validates_input_data(self) -> None:
        """
        Trainers MUST validate input data before training.
        
        Given: Invalid training data
        When: Attempting to train
        Then: Raises informative error
        """
        if HAS_XGBOOST:
            from ml.training.xgboost import XGBoostTrainer
            from ml.config.xgboost import XGBoostTrainingConfig
            
            config = XGBoostTrainingConfig(
                data_source="test_data",
                n_estimators=2,
            )
            
            trainer = XGBoostTrainer(config)
            
            # Test with missing target column
            df_no_target = pd.DataFrame({
                "f1": np.random.randn(50),
                "f2": np.random.randn(50),
                # Missing "target" column
            })
            
            with pytest.raises((ValueError, KeyError)) as exc_info:
                trainer.train(df_no_target)
            
            error_msg = str(exc_info.value).lower()
            assert "target" in error_msg or "label" in error_msg, \
                "Error should mention missing target/label"
            
            # Test with empty dataframe
            df_empty = pd.DataFrame()
            
            with pytest.raises(ValueError):
                trainer.train(df_empty)
    
    def test_trainer_tracks_training_metrics(self) -> None:
        """
        Trainers MUST track and save training metrics.
        
        Given: Training process
        When: Training completes
        Then: Metrics are available and saved
        """
        if HAS_XGBOOST:
            from ml.training.xgboost import XGBoostTrainer
            from ml.config.xgboost import XGBoostTrainingConfig
            
            with tempfile.TemporaryDirectory() as tmpdir_str:
                tmpdir = Path(tmpdir_str)
                
                config = XGBoostTrainingConfig(
                    data_source="test_data",
                    n_estimators=5,
                    max_depth=3,
                    save_model_path=str(tmpdir / "model.json"),
                )
                
                trainer = XGBoostTrainer(config)
                
                # Train
                df = pd.DataFrame({
                    "f1": np.random.randn(100),
                    "f2": np.random.randn(100),
                    "f3": np.random.randn(100),
                    "target": np.random.randint(0, 2, 100),
                })
                
                trainer.train(df)
                
                # Check metrics are tracked
                if hasattr(trainer, '_training_metrics'):
                    metrics = trainer._training_metrics
                    assert isinstance(metrics, dict), "Metrics should be a dictionary"
                    
                    # Common metrics to check
                    possible_metrics = ["accuracy", "auc", "loss", "train_time"]
                    assert any(m in str(metrics).lower() for m in possible_metrics), \
                        "Should track at least one common metric"
                
                # Save and check metadata includes metrics
                model_path = tmpdir / "model.json"
                trainer.save_model(model_path)
                
                metadata_path = model_path.with_suffix(".meta")
                if metadata_path.exists():
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                    
                    if "training_metrics" in metadata:
                        assert len(metadata["training_metrics"]) > 0, \
                            "Should save some training metrics"
    
    def test_trainer_supports_feature_importance(self) -> None:
        """
        Trainers SHOULD provide feature importance when available.
        
        Given: Trained tree-based model
        When: Requesting feature importance
        Then: Returns importance scores for features
        """
        if HAS_XGBOOST:
            from ml.training.xgboost import XGBoostTrainer
            from ml.config.xgboost import XGBoostTrainingConfig
            
            config = XGBoostTrainingConfig(
                data_source="test_data",
                n_estimators=10,
                max_depth=3,
            )
            
            trainer = XGBoostTrainer(config)
            
            # Train with named features
            feature_names = ["sma", "rsi", "volume", "volatility"]
            df = pd.DataFrame({
                **{name: np.random.randn(100) for name in feature_names},
                "target": np.random.randint(0, 2, 100),
            })
            
            trainer.train(df)
            
            # Check for feature importance
            if hasattr(trainer, 'get_feature_importance'):
                importance = trainer.get_feature_importance()
                assert isinstance(importance, dict), \
                    "Feature importance should be a dictionary"
                assert len(importance) == len(feature_names), \
                    "Should have importance for all features"
                
                # Verify all importances are non-negative
                for feat, score in importance.items():
                    assert score >= 0, f"Importance for {feat} should be non-negative"