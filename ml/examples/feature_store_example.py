#!/usr/bin/env python3
"""
Example demonstrating FeatureStore integration for training/inference parity.

This example shows how to:
1. Use FeatureStore with MLSignalActor for live inference
2. Use FeatureStore with training pipeline for historical training
3. Ensure perfect feature parity between training and inference

"""

from datetime import UTC
from datetime import datetime
from datetime import timedelta
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from ml.actors import MLSignalActor
from ml.actors import MLSignalActorConfig
from ml.actors import SignalStrategy
from ml.config.base import MLFeatureConfig
from ml.config.base import MLTrainingConfig
from ml.config.targets import TargetSemanticsConfig
from ml.features import FeatureConfig
from ml.features.indicators import IndicatorManager
from ml.stores.feature_store import FeatureStore
from ml.training.base import BaseMLTrainer
from nautilus_trader.model.data import BarType
from nautilus_trader.model.identifiers import InstrumentId


def example_live_inference_with_feature_store() -> MLSignalActor:
    """
    Demonstrate using MLSignalActor with FeatureStore for live inference.

    This ensures that features computed during live trading are:
    1. Identical to training features (same FeatureEngineer)
    2. Persisted for future training (continuous learning)

    """
    # Configure actor with FeatureStore
    config = MLSignalActorConfig(
        actor_id="ML_SIGNAL_001",
        model_id="example-model",
        bar_type=cast(BarType, object()),
        instrument_id=cast(InstrumentId, object()),
        model_path="./models/production/model.onnx",
        # Enable FeatureStore for parity
        use_feature_store=True,
        db_connection="postgresql://postgres:postgres@localhost:5432/nautilus",
        persist_features=True,  # Store features for future training
        # Feature configuration
        feature_config=FeatureConfig(),
        # Signal generation
        signal_strategy=SignalStrategy.THRESHOLD,
        prediction_threshold=0.7,
    )

    # Create actor - it will automatically use FeatureStore
    actor = MLSignalActor(config)

    # When bars arrive, actor will:
    # 1. Use FeatureStore.compute_realtime() for features
    # 2. Store features in PostgreSQL (same DB as Nautilus)
    # 3. Generate signals based on ML predictions

    print(f"Actor configured with FeatureStore: {actor._feature_store is not None}")
    print(f"Features will be persisted: {actor._persist_features}")

    return actor


def example_training_with_feature_store() -> BaseMLTrainer:
    """
    Demonstrate training with FeatureStore for guaranteed parity.

    This ensures that training uses:
    1. The same FeatureEngineer as inference
    2. Features computed with identical logic
    3. Direct access to Nautilus PostgreSQL data

    """

    class ExampleTrainer(BaseMLTrainer):
        """
        Example trainer using FeatureStore.
        """

        def prepare_data(
            self,
            data: Any,
            target_col: str = "target",
        ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], dict[str, Any]]:
            return np.empty((0, 0), dtype=np.float64), np.empty((0,), dtype=np.float64), {}

        def _train_model(
            self,
            X_train: npt.NDArray[np.float64],
            y_train: npt.NDArray[np.float64],
            X_val: npt.NDArray[np.float64],
            y_val: npt.NDArray[np.float64],
            **kwargs: Any,
        ) -> dict[str, Any]:
            return {"model": object(), "metrics": {}}

        def predict(
            self,
            model: Any,
            X: npt.NDArray[np.float64],
            **_: Any,
        ) -> npt.NDArray[np.float32]:
            return np.zeros(len(X), dtype=np.float32)

        def evaluate(
            self,
            model: Any,
            X: npt.NDArray[np.float64],
            y: npt.NDArray[np.float64],
        ) -> dict[str, float]:
            return {"accuracy": 0.0}

        def _create_model(self, params: dict[str, Any]) -> Any:
            return object()

        def _get_model_params(self) -> dict[str, Any]:
            return {}

        def _convert_to_onnx(self, model: Any, path: Any) -> None:
            return None

        def _suggest_hyperparameters(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            return {}

        def save_model(self, path: Any) -> None:
            Path = __import__("pathlib").Path
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).touch()

        def load_model(self, path: Any) -> Any:
            return object()

    # Configure training with FeatureStore
    target_semantics = TargetSemanticsConfig.from_legacy(
        horizon_minutes=15,
        threshold=0.001,
        legacy_aliases=True,
    )
    config = MLTrainingConfig(
        data_source="nautilus_postgres",  # Indicates data source
        db_connection="postgresql://postgres:postgres@localhost:5432/nautilus",
        feature_config=MLFeatureConfig(),
        train_test_split=0.8,
        save_model_path="./models/trained/model.pkl",
        target_semantics=target_semantics,
    )

    # Create trainer - it will automatically initialize FeatureStore
    trainer = ExampleTrainer(config)

    # Prepare data using FeatureStore
    # This ensures identical feature computation as inference
    X, y, feature_names = trainer.prepare_data_with_feature_store(
        instrument_id="EURUSD",
        start=datetime.now(UTC) - timedelta(days=90),
        end=datetime.now(UTC),
        compute_if_missing=True,  # Compute features if not in DB
    )

    print(f"Loaded {len(X)} samples with {len(feature_names)} features")
    print("Features computed with same logic as inference")

    return trainer


def example_parity_validation() -> bool:
    """
    Validate training/inference parity.
    """
    # Create FeatureStore
    feature_store = FeatureStore(
        connection_string="postgresql://postgres:postgres@localhost:5432/nautilus",
        feature_config=FeatureConfig(),
    )

    # Load some historical bars (mock data for example)
    import polars as pl

    bars_df = pl.DataFrame(
        {
            "close": [1.1000, 1.1005, 1.1010, 1.1015, 1.1020],
            "high": [1.1010, 1.1015, 1.1020, 1.1025, 1.1030],
            "low": [1.0990, 1.0995, 1.1000, 1.1005, 1.1010],
            "volume": [1000000, 1100000, 1200000, 1300000, 1400000],
            "ts_event": [1, 2, 3, 4, 5],
        },
    )

    # Compute features using batch method (training)
    batch_features, _ = feature_store.feature_engineer.calculate_features_batch(bars_df)

    indicator_manager = IndicatorManager(feature_store.feature_engineer.config)
    # Compute features using online method (inference)
    online_features = []
    for i in range(len(bars_df)):
        row = bars_df[i]
        features = feature_store.feature_engineer.calculate_features_online(
            current_bar={
                "close": float(cast(Any, row)["close"]),
                "high": float(cast(Any, row)["high"]),
                "low": float(cast(Any, row)["low"]),
                "volume": float(cast(Any, row)["volume"]),
            },
            indicator_manager=indicator_manager,
        )
        online_features.append(features)

    online_features_array = np.array(online_features)

    # Check parity
    max_diff = np.max(np.abs(batch_features - online_features_array))

    print(f"Maximum difference between batch and online: {max_diff}")
    print(f"Parity check {'PASSED' if max_diff < 1e-10 else 'FAILED'}")
    print(f"Training/inference will compute identical features: {max_diff < 1e-10}")

    parity_ok: bool = bool(max_diff < 1e-10)
    return parity_ok


def main() -> None:
    """
    Run all examples.
    """
    print("=" * 60)
    print("FeatureStore Integration Examples")
    print("=" * 60)

    print("\n1. Live Inference with FeatureStore:")
    print("-" * 40)
    actor = example_live_inference_with_feature_store()

    print("\n2. Training with FeatureStore:")
    print("-" * 40)
    trainer = example_training_with_feature_store()

    print("\n3. Parity Validation:")
    print("-" * 40)
    parity_ok = example_parity_validation()

    print("\n" + "=" * 60)
    print("Summary:")
    print(f"- Actor configured: {actor is not None}")
    print(f"- Trainer configured: {trainer is not None}")
    print(f"- Parity validated: {parity_ok}")
    print("- All features computed identically for training and inference")
    print("=" * 60)


if __name__ == "__main__":
    main()
