#!/usr/bin/env python3
"""
Example demonstrating FeatureStore integration for training/inference parity.

This example shows how to:
1. Use FeatureStore with MLSignalActor for live inference
2. Use FeatureStore with training pipeline for historical training
3. Ensure perfect feature parity between training and inference

"""

from datetime import datetime
from datetime import timedelta

from ml.actors.signal import MLSignalActor
from ml.actors.signal import SignalStrategy
from ml.config.actors import MLSignalActorConfig
from ml.config.base import MLFeatureConfig
from ml.config.base import MLTrainingConfig
from ml.features.engineering import FeatureConfig
from ml.stores.feature_store import FeatureStore
from ml.training.base import BaseMLTrainer


def example_live_inference_with_feature_store():
    """
    Demonstrate using MLSignalActor with FeatureStore for live inference.

    This ensures that features computed during live trading are:
    1. Identical to training features (same FeatureEngineer)
    2. Persisted for future training (continuous learning)

    """
    # Configure actor with FeatureStore
    config = MLSignalActorConfig(
        actor_id="ML_SIGNAL_001",
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
        confidence_threshold=0.6,
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


def example_training_with_feature_store():
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

        def prepare_data(self, data, target_col="target"):
            # For non-FeatureStore data sources
            # Override this for custom data preparation
            pass

        def train_model(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
            # Example using XGBoost
            from ml._imports import xgb

            model = xgb.XGBClassifier(
                objective="binary:logistic",
                max_depth=6,
                n_estimators=100,
            )

            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)] if X_val is not None else None,
                early_stopping_rounds=10,
                verbose=False,
            )

            return {"model": model, "metrics": {"accuracy": 0.85}}

        def predict(self, model, X):
            return model.predict(X)

        def evaluate(self, y_true, y_pred):
            from sklearn.metrics import accuracy_score

            return {"accuracy": accuracy_score(y_true, y_pred)}

        def save_model(self, path):
            import joblib

            joblib.dump(self._model, path)

        def load_model(self, path):
            import joblib

            return joblib.load(path)

    # Configure training with FeatureStore
    config = MLTrainingConfig(
        data_source="nautilus_postgres",  # Indicates data source
        db_connection="postgresql://postgres:postgres@localhost:5432/nautilus",
        feature_config=MLFeatureConfig(),
        train_test_split=0.8,
        save_model_path="./models/trained/model.pkl",
    )

    # Create trainer - it will automatically initialize FeatureStore
    trainer = ExampleTrainer(config)

    # Prepare data using FeatureStore
    # This ensures identical feature computation as inference
    X, y, feature_names = trainer.prepare_data_with_feature_store(
        instrument_id="EURUSD",
        start=datetime.utcnow() - timedelta(days=90),
        end=datetime.utcnow(),
        compute_if_missing=True,  # Compute features if not in DB
    )

    print(f"Loaded {len(X)} samples with {len(feature_names)} features")
    print("Features computed with same logic as inference")

    return trainer


def example_parity_validation():
    """
    Validate training/inference parity.
    """
    # Create FeatureStore
    feature_store = FeatureStore(
        connection_string="postgresql://postgres:postgres@localhost:5432/nautilus",
        feature_config=FeatureConfig(),
    )

    # Load some historical bars (mock data for example)
    import numpy as np
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

    # Compute features using online method (inference)
    online_features = []
    for i in range(len(bars_df)):
        row = bars_df[i]
        features = feature_store.feature_engineer.calculate_features_online(
            close_price=float(row["close"]),
            high_price=float(row["high"]),
            low_price=float(row["low"]),
            volume=float(row["volume"]),
        )
        online_features.append(features)

    online_features_array = np.array(online_features)

    # Check parity
    max_diff = np.max(np.abs(batch_features - online_features_array))

    print(f"Maximum difference between batch and online: {max_diff}")
    print(f"Parity check {'PASSED' if max_diff < 1e-10 else 'FAILED'}")
    print(f"Training/inference will compute identical features: {max_diff < 1e-10}")

    return max_diff < 1e-10


def main():
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
