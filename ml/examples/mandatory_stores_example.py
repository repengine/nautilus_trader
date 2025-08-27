#!/usr/bin/env python3
"""
Example demonstrating the mandatory store integration in ML actors.

This example shows how all ML data is automatically persisted without any additional
configuration required from the user.

"""

from typing import cast

import numpy as np
import numpy.typing as npt
from numpy.random import default_rng

from ml.actors.base import BaseMLInferenceActor
from ml.config.base import MLActorConfig
from nautilus_trader.model.data import Bar


class ProductionMLActor(BaseMLInferenceActor):
    """
    Example production ML actor with automatic data persistence.

    All features, predictions, and signals are automatically stored to the configured
    database without any additional code needed.

    """

    def __init__(self, config: MLActorConfig):
        """
        Initialize the actor.

        Stores are automatically initialized by the base class:
        - self._feature_store: Persists all computed features
        - self._model_store: Persists all predictions
        - self._strategy_store: Persists all trading signals

        """
        super().__init__(config)

        # Your custom initialization here
        self.feature_names = [
            "return_1",
            "return_5",
            "return_10",
            "volatility_20",
            "rsi_14",
            "macd_signal",
        ]

        print("Initialized ProductionMLActor with automatic stores:")
        print(f"  - Feature store: {type(self._feature_store).__name__}")
        print(f"  - Model store: {type(self._model_store).__name__}")
        print(f"  - Strategy store: {type(self._strategy_store).__name__}")

    def on_start(self) -> None:
        """
        Start actor.

        The base class handles:
        - Model loading
        - Store connection verification
        - Health monitoring setup

        """
        super().on_start()
        print("Actor started with automatic data persistence enabled")

    def _compute_features(self, bar: Bar) -> npt.NDArray[np.float32]:
        """
        Compute features from market data.

        These features are AUTOMATICALLY persisted to the feature store by the base
        class. No additional code needed!

        """
        # Example feature computation
        rng = default_rng(0)
        features: npt.NDArray[np.float32] = rng.standard_normal(len(self.feature_names)).astype(
            np.float32,
        )

        # The base class will automatically store these features
        # No need to call self._feature_store.write_features()

        return features

    def _predict(self, features: npt.NDArray[np.float32]) -> tuple[float, float]:
        """
        Generate prediction from features.

        The prediction and confidence are AUTOMATICALLY persisted to the model store by
        the base class.

        """
        # Example prediction (replace with your model inference)
        prediction = float(np.sum(features) > 0)
        confidence = float(abs(np.mean(features)))

        # The base class will automatically store this prediction
        # No need to call self._model_store.write_prediction()

        return prediction, confidence

    def on_stop(self) -> None:
        """
        Stop actor.

        The base class automatically:
        - Flushes all pending writes to stores
        - Saves final metrics
        - Cleans up connections

        """
        print("Shutting down - all data automatically flushed to stores")
        super().on_stop()


def main() -> None:
    """
    Demonstrate the mandatory store integration.
    """
    # Example 1: Production deployment with PostgreSQL
    print("\n=== Example 1: Production with PostgreSQL ===")

    # For typing-only correctness in this example, use cast() for required types
    from nautilus_trader.model.data import BarType
    from nautilus_trader.model.identifiers import InstrumentId

    production_config = MLActorConfig(
        model_id="production_model_v1",
        model_path="/path/to/model.onnx",
        bar_type=cast(BarType, object()),
        instrument_id=cast(InstrumentId, object()),
    )

    # Create actor - stores are automatically initialized
    _actor = ProductionMLActor(production_config)
    print("Production actor created with PostgreSQL stores")

    # Example 2: Development/Testing (automatic fallback)
    print("\n=== Example 2: Testing with automatic fallback ===")

    test_config = MLActorConfig(
        model_id="test_model",
        model_path="./test_model.pkl",
        bar_type=cast(BarType, object()),
        instrument_id=cast(InstrumentId, object()),
    )

    _test_actor = ProductionMLActor(test_config)
    print("Test actor created with automatic store fallback")

    # Example 3: Verifying data persistence
    print("\n=== Example 3: Data Persistence Verification ===")

    # You can check what's being stored
    print("\nData automatically persisted:")
    print("1. Features: Every feature computation is stored")
    print("2. Predictions: Every model inference is stored")
    print("3. Signals: Every trading signal is stored")
    print("4. Metadata: Model versions, feature schemas, etc.")

    # Example 4: Querying stored data
    print("\n=== Example 4: Querying Stored Data ===")

    print(
        """
    Query examples (run in PostgreSQL):

    -- Get recent predictions
    SELECT * FROM ml_model_predictions
    WHERE model_id = 'production_model_v1'
    ORDER BY ts_event DESC
    LIMIT 10;

    -- Check feature statistics
    SELECT
        feature_set_id,
        COUNT(*) as count,
        AVG((features->>'return_1')::float) as avg_return_1
    FROM ml_feature_values
    GROUP BY feature_set_id;

    -- Analyze signal generation
    SELECT
        strategy_id,
        signal_type,
        COUNT(*) as signals,
        AVG(strength) as avg_strength
    FROM ml_strategy_signals
    WHERE ts_event > extract(epoch from now() - interval '1 hour') * 1e9
    GROUP BY strategy_id, signal_type;
    """,
    )

    # Example 5: Benefits of mandatory stores
    print("\n=== Benefits of Mandatory Stores ===")

    benefits = [
        "1. Zero Configuration: Stores work automatically",
        "2. No Data Loss: Every prediction is persisted",
        "3. Feature Parity: Training and inference use identical features",
        "4. Audit Trail: Complete history for compliance",
        "5. Performance Monitoring: Built-in latency tracking",
        "6. A/B Testing: Compare models using stored predictions",
        "7. Debugging: Replay any prediction with exact features",
        "8. Model Drift: Detect changes over time",
    ]

    for benefit in benefits:
        print(f"  ✓ {benefit}")

    # Example 6: Migration from optional stores
    print("\n=== Migration from Optional Stores ===")

    print(
        """
    If you have existing code with optional stores:

    OLD CODE:
    ```python
    class MyActor(Actor):
        def __init__(self, config, model_store=None):
            self._model_store = model_store  # Might be None!

        def on_bar(self, bar):
            prediction = self.predict(bar)
            if self._model_store:  # Conditional storage
                self._model_store.write(prediction)
    ```

    NEW CODE:
    ```python
    class MyActor(BaseMLInferenceActor):
        def __init__(self, config):
            super().__init__(config)  # Stores initialized automatically

        def on_bar(self, bar):
            # Just process - storage is automatic!
            super().on_bar(bar)
    ```
    """,
    )


if __name__ == "__main__":
    main()

    print("\n" + "=" * 60)
    print("Mandatory stores ensure that NO DATA IS EVER LOST!")
    print("All ML operations are automatically tracked and persisted.")
    print("=" * 60)
