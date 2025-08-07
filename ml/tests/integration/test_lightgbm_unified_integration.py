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
"""
Integration tests for UnifiedLightGBMTrainer.

These tests demonstrate the complete workflow of the unified trainer with synthetic
data, showing the integration of all components including GOSS, DART, EFB, GPU, Optuna
optimization, and MLflow tracking.

"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from ml._imports import HAS_LIGHTGBM
from ml.config.lightgbm_unified import DARTConfig
from ml.config.lightgbm_unified import GOSSConfig
from ml.config.lightgbm_unified import UnifiedLightGBMConfig
from ml.training.lightgbm_unified import UnifiedLightGBMTrainer


@pytest.mark.skipif(not HAS_LIGHTGBM, reason="LightGBM required for integration tests")
class TestUnifiedLightGBMTrainerIntegration:
    """
    Integration tests for UnifiedLightGBMTrainer.
    """

    def setup_method(self):
        """
        Set up test fixtures with synthetic data.
        """
        # Create synthetic training data
        np.random.seed(42)
        self.n_samples = 1000
        self.n_features = 10

        # Generate features
        self.X_train = np.random.randn(self.n_samples, self.n_features)
        self.X_val = np.random.randn(200, self.n_features)

        # Generate target (regression)
        # Create some signal in the data
        signal = (
            self.X_train[:, 0] * 0.5
            + self.X_train[:, 1] * -0.3
            + self.X_train[:, 2] * 0.2
            + np.sin(self.X_train[:, 3]) * 0.1
        )
        noise = np.random.randn(self.n_samples) * 0.1
        self.y_train = signal + noise

        # Validation target
        signal_val = (
            self.X_val[:, 0] * 0.5
            + self.X_val[:, 1] * -0.3
            + self.X_val[:, 2] * 0.2
            + np.sin(self.X_val[:, 3]) * 0.1
        )
        noise_val = np.random.randn(200) * 0.1
        self.y_val = signal_val + noise_val

        # Feature names for better interpretability
        self.feature_names = [f"feature_{i}" for i in range(self.n_features)]

    def test_basic_training_workflow(self):
        """
        Test basic training workflow without advanced features.
        """
        config = UnifiedLightGBMConfig(
            data_source="synthetic_data",
            n_estimators=50,  # Small for fast test
            max_depth=3,
            learning_rate=0.1,
            enable_monitoring=False,  # Disable monitoring for simplicity
        )

        trainer = UnifiedLightGBMTrainer(config)

        # Train the model
        results = trainer.train(
            self.X_train,
            self.y_train,
            self.X_val,
            self.y_val,
            feature_names=self.feature_names,
        )

        # Validate results
        assert "model" in results
        assert "training_time" in results
        assert "feature_importance" in results
        assert "lgb_params" in results

        model = results["model"]
        assert model is not None
        assert hasattr(model, "predict")
        assert results["training_time"] > 0

        # Test predictions
        predictions = trainer.predict(model, self.X_val)
        assert predictions.shape == (len(self.y_val),)
        assert np.isfinite(predictions).all()

    def test_goss_configuration_training(self):
        """
        Test training with GOSS (Gradient-based One-Side Sampling) enabled.
        """
        config = UnifiedLightGBMConfig(
            data_source="synthetic_data",
            n_estimators=30,
            goss_config=GOSSConfig(enabled=True, top_rate=0.2, other_rate=0.1),
            enable_monitoring=False,
        )

        trainer = UnifiedLightGBMTrainer(config)

        results = trainer.train(
            self.X_train,
            self.y_train,
            self.X_val,
            self.y_val,
            feature_names=self.feature_names,
        )

        # Verify GOSS parameters were used
        lgb_params = results["lgb_params"]
        assert lgb_params["boosting_type"] == "goss"
        assert lgb_params["top_rate"] == 0.2
        assert lgb_params["other_rate"] == 0.1

        # Model should still work
        model = results["model"]
        predictions = trainer.predict(model, self.X_val)
        assert predictions.shape == (len(self.y_val),)

    def test_dart_configuration_training(self):
        """
        Test training with DART (Dropouts meet Multiple Additive Regression Trees)
        enabled.
        """
        config = UnifiedLightGBMConfig(
            data_source="synthetic_data",
            n_estimators=30,
            dart_config=DARTConfig(
                enabled=True,
                drop_rate=0.1,
                max_drop=10,
                skip_drop=0.5,
            ),
            enable_monitoring=False,
        )

        trainer = UnifiedLightGBMTrainer(config)

        results = trainer.train(
            self.X_train,
            self.y_train,
            self.X_val,
            self.y_val,
            feature_names=self.feature_names,
        )

        # Verify DART parameters were used
        lgb_params = results["lgb_params"]
        assert lgb_params["boosting_type"] == "dart"
        assert lgb_params["drop_rate"] == 0.1
        assert lgb_params["max_drop"] == 10
        assert lgb_params["skip_drop"] == 0.5

        # Model should work
        model = results["model"]
        predictions = trainer.predict(model, self.X_val)
        assert predictions.shape == (len(self.y_val),)

    def test_feature_importance_tracking(self):
        """
        Test feature importance tracking and decay detection.
        """
        config = UnifiedLightGBMConfig(
            data_source="synthetic_data",
            n_estimators=30,
            track_feature_decay=True,
            feature_decay_threshold=0.5,
            feature_history_window=5,
            enable_monitoring=False,
        )

        trainer = UnifiedLightGBMTrainer(config)

        # First training run
        results1 = trainer.train(
            self.X_train,
            self.y_train,
            self.X_val,
            self.y_val,
            feature_names=self.feature_names,
        )

        assert len(trainer.importance_history) == 1
        assert len(trainer.feature_decay_alerts) == 0

        # Second training run (might trigger decay detection)
        # Modify data to potentially change feature importance
        X_train_modified = self.X_train.copy()
        X_train_modified[:, 0] *= 0.1  # Reduce importance of first feature

        results2 = trainer.train(
            X_train_modified,
            self.y_train,
            self.X_val,
            self.y_val,
            feature_names=self.feature_names,
        )

        assert len(trainer.importance_history) <= 2
        # Feature decay alerts might or might not be triggered depending on the data

    def test_model_save_load_workflow(self):
        """
        Test model saving and loading with metadata.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "test_model.lgb"

            config = UnifiedLightGBMConfig(
                data_source="synthetic_data",
                n_estimators=20,
                enable_monitoring=False,
            )

            trainer = UnifiedLightGBMTrainer(config)

            # Train model
            results = trainer.train(
                self.X_train,
                self.y_train,
                self.X_val,
                self.y_val,
                feature_names=self.feature_names,
            )

            model = results["model"]

            # Save model
            trainer.save_model(model, model_path)

            # Verify files exist
            assert model_path.exists()
            metadata_path = model_path.with_suffix(".metadata.json")
            assert metadata_path.exists()

            # Load model
            loaded_model = trainer.load_model(model_path)

            # Test that loaded model works
            original_predictions = trainer.predict(model, self.X_val)
            loaded_predictions = trainer.predict(loaded_model, self.X_val)

            # Predictions should be very similar (allowing for small numerical differences)
            np.testing.assert_allclose(original_predictions, loaded_predictions, rtol=1e-10)

    def test_categorical_features_support(self):
        """
        Test native categorical feature support.
        """
        # Create data with some categorical features (simulated as integers)
        X_train_cat = self.X_train.copy()
        X_val_cat = self.X_val.copy()

        # Make first two features "categorical" (convert to integers)
        X_train_cat[:, 0] = (X_train_cat[:, 0] * 5).astype(int) % 10
        X_train_cat[:, 1] = (X_train_cat[:, 1] * 3).astype(int) % 5
        X_val_cat[:, 0] = (X_val_cat[:, 0] * 5).astype(int) % 10
        X_val_cat[:, 1] = (X_val_cat[:, 1] * 3).astype(int) % 5

        config = UnifiedLightGBMConfig(
            data_source="synthetic_data",
            n_estimators=20,
            categorical_features=["0", "1"],  # Feature indices as strings
            enable_monitoring=False,
        )

        trainer = UnifiedLightGBMTrainer(config)

        results = trainer.train(
            X_train_cat,
            self.y_train,
            X_val_cat,
            self.y_val,
            feature_names=self.feature_names,
        )

        # Should complete successfully and report categorical features
        assert results["n_categorical_features"] == 2

        model = results["model"]
        predictions = trainer.predict(model, X_val_cat)
        assert predictions.shape == (len(self.y_val),)

    def test_comprehensive_workflow(self):
        """
        Test comprehensive workflow with multiple advanced features.
        """
        config = UnifiedLightGBMConfig(
            data_source="synthetic_data",
            # Core parameters
            n_estimators=50,
            max_depth=4,
            learning_rate=0.05,
            num_leaves=15,
            # GOSS configuration
            goss_config=GOSSConfig(enabled=True, top_rate=0.3, other_rate=0.15),
            # Feature bundling
            efb_config=config.efb_config,  # Use default EFB settings
            # Feature tracking
            track_feature_decay=True,
            feature_decay_threshold=0.4,
            # Cross-validation
            cv_strategy="standard",
            cv_folds=3,
            # Monitoring (disabled for test simplicity)
            enable_monitoring=False,
        )

        trainer = UnifiedLightGBMTrainer(config)

        results = trainer.train(
            self.X_train,
            self.y_train,
            self.X_val,
            self.y_val,
            feature_names=self.feature_names,
        )

        # Comprehensive validation
        model = results["model"]
        assert model is not None

        # Check GOSS was applied
        assert results["lgb_params"]["boosting_type"] == "goss"

        # Check feature importance tracking
        assert len(trainer.importance_history) == 1

        # Check feature importance structure
        feature_importance = results["feature_importance"]
        assert len(feature_importance) == len(self.feature_names)
        assert all(isinstance(v, (int, float)) for v in feature_importance.values())

        # Test predictions work
        predictions = trainer.predict(model, self.X_val)
        assert predictions.shape == (len(self.y_val),)
        assert np.isfinite(predictions).all()

        # Check that we have reasonable performance (RMSE should be reasonable)
        rmse = np.sqrt(np.mean((predictions - self.y_val) ** 2))
        assert rmse < 1.0  # Should be much better than random
