"""
Test the integrated FeatureStore with MLSignalActor and training pipeline.
"""

from __future__ import annotations

from datetime import UTC
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import numpy.typing as npt
import pytest

from ml.actors.signal import MLSignalActor
from ml.actors.signal import MLSignalActorConfig
from ml.actors.signal import SignalStrategy
from ml.config.base import MLTrainingConfig
from ml.features.config import FeatureConfig
from ml.stores.feature_store_facade import FeatureStore
from ml.training.base_facade import BaseMLTrainer
from nautilus_trader.model.data import Bar

pytestmark = pytest.mark.usefixtures(
    "isolated_prometheus_registry",
    "mock_tracing_backend",
    "isolated_orchestrator_env",
)


@pytest.mark.database
@pytest.mark.serial
@pytest.mark.usefixtures("cloned_test_database", "real_engine_manager")
class TestFeatureStoreIntegration:
    """
    Test FeatureStore integration with existing components.
    """

    @pytest.fixture
    def mock_bar(self) -> Any:
        """
        Create a mock bar for testing.
        """
        # Use a MagicMock with Bar spec to avoid strict C-extension type checks
        return MagicMock(spec=Bar)

    def test_ml_signal_actor_with_feature_store(
        self,
        mock_bar: Bar,
        cloned_test_database: str,
    ) -> None:
        """
        Test that MLSignalActor uses FeatureStore when configured.
        """
        # Create config with FeatureStore enabled
        # Provide required typed args via casts to satisfy mypy
        from nautilus_trader.model.data import BarType
        from nautilus_trader.model.identifiers import InstrumentId

        config = MLSignalActorConfig(
            component_id="TEST_ACTOR",
            model_id="model-1",
            model_path="./test_model.onnx",
            bar_type=cast(BarType, object()),
            instrument_id=cast(InstrumentId, object()),
            use_feature_store=True,
            db_connection=cloned_test_database,
            persist_features=True,
            prediction_threshold=0.7,
        )

        # Create actor
        with patch("ml.actors.signal.MLSignalActor._load_model_with_metadata"):
            actor = MLSignalActor(config)

            # Verify FeatureStore was initialized
            assert actor._feature_store is not None
            assert actor._persist_features is True

            # Mock the FeatureStore compute_realtime method
            from numpy.random import default_rng

            expected_features = default_rng(0).random(50).astype(np.float32)
            actor._feature_store.compute_realtime = MagicMock(
                return_value=expected_features,
            )

            # Compute features
            features = actor._compute_features(mock_bar)

            # Verify FeatureStore was called correctly
            actor._feature_store.compute_realtime.assert_called_once_with(
                bar=mock_bar,
                store=True,
            )

            # Verify returned features match
            assert np.array_equal(
                cast(npt.NDArray[np.float32], features),
                expected_features,
            )

    def test_ml_signal_actor_without_feature_store(
        self,
        mock_bar: Bar,
    ) -> None:
        """
        Test that MLSignalActor works without FeatureStore (backward compatibility).
        """
        # Create config without FeatureStore
        from nautilus_trader.model.data import BarType
        from nautilus_trader.model.identifiers import InstrumentId

        config = MLSignalActorConfig(
            component_id="TEST_ACTOR",
            model_id="model-2",
            model_path="./test_model.onnx",
            bar_type=cast(BarType, object()),
            instrument_id=cast(InstrumentId, object()),
            use_feature_store=False,  # Explicitly disabled
            prediction_threshold=0.7,
        )

        # Create actor
        with patch("ml.actors.signal.MLSignalActor._load_model_with_metadata"):
            actor = MLSignalActor(config)

            # Verify store was initialized (always mandatory, but may be DummyStore)
            assert actor._feature_store is not None  # Always exists now
            # Note: persist_features may still control whether to actually write

            # Verify actor still has FeatureEngineer
            assert actor._feature_engineer is not None

    def test_training_with_feature_store(self, cloned_test_database: str) -> None:
        """
        Test training pipeline with FeatureStore integration.
        """

        @pytest.mark.database
        @pytest.mark.serial
        class TestTrainer(BaseMLTrainer):
            """
            Test trainer implementation.
            """

            def prepare_data(
                self,
                data: Any,
                target_col: str = "target",
            ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], dict[str, Any]]:
                """
                Simple implementation for testing.
                """
                from numpy.random import default_rng

                _rng = default_rng(0)
                return _rng.random((100, 10)), _rng.integers(0, 2, 100).astype(np.float64), {}

            def _train_model(
                self,
                X_train: npt.NDArray[np.float64],
                y_train: npt.NDArray[np.float64],
                X_val: npt.NDArray[np.float64],
                y_val: npt.NDArray[np.float64],
                **kwargs: Any,
            ) -> dict[str, Any]:
                return {"model": MagicMock(), "metrics": {"accuracy": 0.85}}

            def predict(
                self,
                model: Any,
                X: npt.NDArray[np.float64],
                **_: Any,
            ) -> npt.NDArray[np.float32]:
                from numpy.random import default_rng

                return default_rng(1).integers(0, 2, len(X)).astype(np.float32)

            def evaluate(
                self,
                model: Any,
                X: npt.NDArray[np.float64],
                y: npt.NDArray[np.float64],
            ) -> dict[str, float]:
                return {"accuracy": 0.85}

            def _create_model(self, params: dict[str, Any]) -> Any:
                return MagicMock()

            def _get_model_params(self) -> dict[str, Any]:
                return {}

            def _convert_to_onnx(self, model: Any, path: Any) -> None:
                return None

            def _suggest_hyperparameters(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
                return {}

            def save_model(self, path: Any) -> None:
                """
                Simple implementation for testing.
                """
                Path(path).touch()

            def load_model(self, path: Any) -> Any:
                """
                Simple implementation for testing.
                """
                return MagicMock()

        # Create config with FeatureStore
        config = MLTrainingConfig(
            data_source="test_data",
            db_connection=cloned_test_database,
        )

        # Create trainer
        trainer = TestTrainer(config)

        # Verify FeatureStore was initialized
        assert trainer._feature_store is not None

        # Mock FeatureStore methods
        cast(Any, trainer._feature_store).compute_and_store_historical = MagicMock(
            return_value=100,
        )
        cast(Any, trainer._feature_store).get_training_data = MagicMock(
            return_value=(
                np.random.rand(100, 10),
                np.arange(100),
                ["feature_" + str(i) for i in range(10)],
            ),
        )

        # Prepare data with FeatureStore
        X, y, feature_names = trainer.prepare_data_with_feature_store(
            instrument_id="EURUSD",
            start=datetime.now(UTC) - timedelta(days=30),
            end=datetime.now(UTC),
        )

        # Verify FeatureStore methods were called
        cast(Any, trainer._feature_store).compute_and_store_historical.assert_called_once()
        cast(Any, trainer._feature_store).get_training_data.assert_called_once()

        # Verify data was returned correctly
        assert X.shape == (100, 10)
        assert y.shape == (100,)
        assert len(feature_names) == 10

    @pytest.mark.database
    @pytest.mark.serial
    def test_feature_store_config_propagation(self, cloned_test_database: str) -> None:
        """
        Test that FeatureStore configuration is properly propagated.
        """
        feature_config = FeatureConfig()
        # Test with MLSignalActor
        from nautilus_trader.model.data import BarType
        from nautilus_trader.model.identifiers import InstrumentId

        actor_config = MLSignalActorConfig(
            component_id="TEST_ACTOR",
            model_id="model-3",
            model_path="./test_model.onnx",
            bar_type=cast(BarType, object()),
            instrument_id=cast(InstrumentId, object()),
            db_connection=cloned_test_database,
            persist_features=False,
            feature_config=feature_config,
        )

        with patch("ml.actors.signal.MLSignalActor._load_model_with_metadata"):
            actor = MLSignalActor(actor_config)

            # Verify custom configuration
            assert actor._feature_store is not None
            assert actor._persist_features is False
            assert cast(Any, actor._feature_store).connection_string == cloned_test_database
            # Compare config values, not object identity (prevents pollution from other tests)
            import msgspec

            actual_config = cast(Any, actor._feature_store).feature_config
            expected_config = feature_config
            assert msgspec.to_builtins(actual_config) == msgspec.to_builtins(
                expected_config,
            ), f"FeatureConfig mismatch: {msgspec.to_builtins(actual_config)} != {msgspec.to_builtins(expected_config)}"

    @pytest.mark.database
    @pytest.mark.serial
    def test_parity_validation_in_training(self, cloned_test_database: str) -> None:
        """
        Test that training pipeline can validate parity.
        """
        # Create FeatureStore
        feature_store = FeatureStore(
            connection_string=cloned_test_database,
            feature_config=FeatureConfig(),
        )

        # Mock data
        import polars as pl

        bars_df = pl.DataFrame(
            {
                "close": [1.1000, 1.1005, 1.1010],
                "high": [1.1010, 1.1015, 1.1020],
                "low": [1.0990, 1.0995, 1.1000],
                "volume": [1000000, 1100000, 1200000],
                "ts_event": [1, 2, 3],
            },
        )

        # Mock the load method
        cast(Any, feature_store)._load_bars_from_nautilus = MagicMock(
            return_value=bars_df,
        )

        # Test batch computation
        batch_features, _ = feature_store.feature_engineer.calculate_features_batch(bars_df)

        # Test online computation (should match batch)
        from ml.features.indicators import IndicatorManager

        indicator_mgr = IndicatorManager(feature_store.feature_config)
        online_features = []

        for i in range(len(bars_df)):
            # Update indicators first
            close_val = float(bars_df["close"][i])
            high_val = float(bars_df["high"][i])
            low_val = float(bars_df["low"][i])
            volume_val = float(bars_df["volume"][i])

            indicator_mgr.update_from_values(
                close=close_val,
                high=high_val,
                low=low_val,
                volume=volume_val,
            )

            # Create current bar dict
            current_bar = {
                "open": close_val,  # Use close as open proxy per calculator logic
                "high": high_val,
                "low": low_val,
                "close": close_val,
                "volume": volume_val,
            }

            features = feature_store.feature_engineer.calculate_features_online(
                current_bar=current_bar,
                indicator_manager=indicator_mgr,
            )
            online_features.append(features.copy())

        online_features_array = np.array(online_features)

        # Verify parity (within tolerance)
        if hasattr(batch_features, "to_numpy"):
            batch_features = batch_features.to_numpy()
        max_diff = np.max(np.abs(batch_features - online_features_array))
        assert max_diff < 1e-10, f"Parity violation: max diff = {max_diff}"


@pytest.mark.database
@pytest.mark.serial
class TestBackwardCompatibility:
    """
    Test that existing code continues to work without FeatureStore.
    """

    @pytest.mark.database
    @pytest.mark.serial
    def test_ml_signal_actor_backward_compatibility(self) -> None:
        """
        Test that MLSignalActor works with old config (no FeatureStore fields).
        """
        # Old-style config without FeatureStore fields
        from nautilus_trader.model.identifiers import InstrumentId
        from nautilus_trader.model.identifiers import Symbol
        from nautilus_trader.model.identifiers import Venue

        config = MLSignalActorConfig(
            actor_id="TEST_ACTOR",
            model_id="model-legacy",
            model_path="./test_model.onnx",
            bar_type=MagicMock(),
            instrument_id=InstrumentId(Symbol("EURUSD"), Venue("IDEALPRO")),
            signal_strategy=SignalStrategy.THRESHOLD,
            prediction_threshold=0.7,
            # Explicitly disable persistence to preserve legacy behavior
            persist_features=False,
        )

        # Should not raise any errors
        with patch("ml.actors.signal.MLSignalActor._load_model_with_metadata"):
            actor = MLSignalActor(config)

            # Should fall back to original behavior (no persistence)
            assert actor._feature_store is not None  # Always initialized now
            assert actor._persist_features is False
            assert actor._feature_engineer is not None

    @pytest.mark.database
    @pytest.mark.serial
    def test_training_backward_compatibility(self) -> None:
        """
        Test that training works with old config (no db_connection).
        """

        @pytest.mark.database
        @pytest.mark.serial
        class TestTrainer(BaseMLTrainer):
            """
            Test trainer implementation.
            """

            def prepare_data(
                self,
                data: Any,
                target_col: str = "target",
            ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], dict[str, Any]]:
                return (
                    np.random.rand(100, 10).astype(np.float64),
                    np.random.randint(0, 2, 100).astype(np.float64),
                    {},
                )

            def _train_model(
                self,
                X_train: npt.NDArray[np.float64],
                y_train: npt.NDArray[np.float64],
                X_val: npt.NDArray[np.float64],
                y_val: npt.NDArray[np.float64],
                **kwargs: Any,
            ) -> dict[str, Any]:
                return {"model": MagicMock(), "metrics": {"accuracy": 0.85}}

            def predict(
                self,
                model: Any,
                X: npt.NDArray[np.float64],
                **_: Any,
            ) -> npt.NDArray[np.float32]:
                from numpy.random import default_rng

                return default_rng(2).integers(0, 2, len(X)).astype(np.float32)

            def evaluate(
                self,
                model: Any,
                X: npt.NDArray[np.float64],
                y: npt.NDArray[np.float64],
            ) -> dict[str, float]:
                return {"accuracy": 0.85}

            def _create_model(self, params: dict[str, Any]) -> Any:
                return MagicMock()

            def _get_model_params(self) -> dict[str, Any]:
                return {}

            def _convert_to_onnx(self, model: Any, path: Any) -> None:
                return None

            def save_model(self, path: Any) -> None:
                Path(path).touch()

            def load_model(self, path: Any) -> Any:
                return MagicMock()

            def _suggest_hyperparameters(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
                return {}

        # Old-style config without db_connection
        config = MLTrainingConfig(
            data_source="test_data",
        )

        # Should not raise any errors
        trainer = TestTrainer(config)

        # Should work without FeatureStore
        assert trainer._feature_store is None  # No DB connection in old config

        # prepare_data should still work
        _X, _y, _metadata = trainer.prepare_data(None)
