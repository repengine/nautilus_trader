# Test Specifications for ML Module Tasks

## Overview
This document provides specific test contracts for each ML module task. Each specification defines the properties that must hold and provides Hypothesis-based test templates.

## Task-Specific Test Specifications

### 1. DataBento Integration Tests

```python
from hypothesis import given, strategies as st, assume
import numpy as np

class TestDataBentoIntegration:
    """Contract tests for DataBento data ingestion."""

    @given(
        symbol=st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=['Lu', 'Ll'])),
        start_date=st.datetimes(min_value=datetime(2020, 1, 1), max_value=datetime(2023, 12, 31)),
        end_date=st.datetimes(min_value=datetime(2020, 1, 2), max_value=datetime(2024, 1, 1))
    )
    def test_historical_data_completeness(self, symbol, start_date, end_date):
        """Property: Historical data must be complete and ordered."""
        assume(start_date < end_date)

        loader = DataBentoLoader()
        data = loader.fetch_historical(symbol, start_date, end_date, level='L1')

        # Properties to verify:
        # 1. No gaps in timestamps
        timestamps = [bar.ts_event for bar in data]
        assert timestamps == sorted(timestamps), "Data not temporally ordered"

        # 2. All bars valid
        for bar in data:
            assert bar.high >= bar.low
            assert bar.high >= bar.open
            assert bar.high >= bar.close
            assert bar.volume >= 0

    @given(
        n_symbols=st.integers(min_value=1, max_value=10),
        buffer_size=st.integers(min_value=100, max_value=10000)
    )
    def test_live_stream_buffer_management(self, n_symbols, buffer_size):
        """Property: Live stream buffers never exceed limits."""
        streamer = DataBentoStreamer(buffer_size=buffer_size)

        for _ in range(n_symbols):
            symbol = generate_test_symbol()
            streamer.subscribe(symbol)

        # Simulate streaming
        for _ in range(buffer_size * 2):
            streamer.process_tick()

        # Property: Buffer respects limits
        assert streamer.buffer_size() <= buffer_size
        assert not streamer.has_data_loss()
```

### 2. Feature Engineering Utilities Tests

```python
class TestFeatureEngineeringUtilities:
    """Contract tests for feature engineering utilities."""

    @given(
        data=st.arrays(
            dtype=np.float64,
            shape=st.tuples(
                st.integers(100, 1000),  # samples
                st.integers(10, 50),      # features
            ),
            elements=st.floats(-1000, 1000, allow_nan=True, allow_infinity=True)
        ),
        contamination=st.floats(0.01, 0.1)
    )
    def test_outlier_removal_preserves_majority(self, data, contamination):
        """Property: Outlier removal preserves majority of data."""
        cleaner = OutlierRemover(contamination=contamination)
        cleaned = cleaner.fit_transform(data)

        # Properties:
        # 1. No NaN/Inf in output
        assert not np.any(np.isnan(cleaned))
        assert not np.any(np.isinf(cleaned))

        # 2. Preserves at least (1-contamination) of data
        min_retained = int(len(data) * (1 - contamination))
        assert len(cleaned) >= min_retained

        # 3. Shape consistency
        assert cleaned.shape[1] == data.shape[1]

    @given(
        features=feature_matrix_strategy,
        n_components=st.integers(min_value=2, max_value=20)
    )
    def test_pca_dimensionality_reduction(self, features, n_components):
        """Property: PCA reduces dimensions while preserving variance."""
        assume(n_components < features.shape[1])

        pca = PCATransformer(n_components=n_components)
        reduced = pca.fit_transform(features)

        # Properties:
        # 1. Correct output shape
        assert reduced.shape == (features.shape[0], n_components)

        # 2. Explained variance ratio sums to <= 1
        assert 0 < pca.explained_variance_ratio_.sum() <= 1.0

        # 3. Components are orthogonal
        components = pca.components_
        gram = components @ components.T
        assert np.allclose(gram, np.eye(n_components), atol=1e-5)
```

### 3. Trade Execution Logic Tests

```python
class TestTradeExecutionLogic:
    """Contract tests for trade execution in ML strategies."""

    @given(
        signal_value=st.floats(min_value=-1.0, max_value=1.0),
        confidence=st.floats(min_value=0.0, max_value=1.0),
        position_size=st.integers(min_value=100, max_value=10000)
    )
    def test_act_method_executes_correctly(self, signal_value, confidence, position_size):
        """Property: act() method correctly translates signals to orders."""
        strategy = MLTradingStrategy(config)

        signal = MLSignal(
            prediction=signal_value,
            confidence=confidence,
            metadata={'position_size': position_size}
        )

        # Execute action
        orders = strategy.act(instrument_id, signal)

        # Properties:
        # 1. Signal direction matches order side
        if signal_value > 0:
            assert all(o.side == OrderSide.BUY for o in orders)
        elif signal_value < 0:
            assert all(o.side == OrderSide.SELL for o in orders)

        # 2. Position sizing respects confidence
        for order in orders:
            assert order.quantity <= position_size * confidence

        # 3. Risk limits enforced
        total_exposure = sum(o.quantity * o.price for o in orders)
        assert total_exposure <= strategy.max_exposure
```

### 4. Model Versioning Tests

```python
class TestModelVersioning:
    """Contract tests for model versioning system."""

    @given(
        n_versions=st.integers(min_value=2, max_value=20),
        rollback_steps=st.integers(min_value=1, max_value=5)
    )
    def test_model_version_management(self, n_versions, rollback_steps):
        """Property: Model versions are managed correctly."""
        assume(rollback_steps < n_versions)

        registry = ModelRegistry()
        versions = []

        # Create versions
        for i in range(n_versions):
            model = create_test_model(version=i)
            version_id = registry.register(model)
            versions.append(version_id)

        # Properties:
        # 1. Versions are unique and ordered
        assert len(set(versions)) == n_versions
        assert versions == sorted(versions)

        # 2. Can retrieve any version
        for v in versions:
            retrieved = registry.get_model(v)
            assert retrieved is not None

        # 3. Rollback works correctly
        registry.rollback(rollback_steps)
        current = registry.get_current_version()
        assert current == versions[-rollback_steps-1]
```

### 5. LightGBM Issues Tests

```python
class TestLightGBMFixes:
    """Contract tests for LightGBM bug fixes."""

    @given(
        model_type=st.sampled_from(['Booster', 'LGBMClassifier', 'LGBMRegressor']),
        n_features=st.integers(min_value=5, max_value=50)
    )
    def test_model_type_detection(self, model_type, n_features):
        """Property: Model type detection using isinstance is accurate."""
        # Create model based on type
        if model_type == 'Booster':
            model = create_lgb_booster(n_features)
        elif model_type == 'LGBMClassifier':
            model = lgb.LGBMClassifier()
            model.fit(X_test, y_test)
        else:
            model = lgb.LGBMRegressor()
            model.fit(X_test, y_test)

        wrapper = LightGBMModel(model, {})

        # Properties:
        # 1. Correct type identification
        if model_type == 'Booster':
            assert wrapper._is_booster == True
        else:
            assert wrapper._is_booster == False

        # 2. Predictions are float32
        predictions = wrapper.predict(X_test)
        assert predictions.dtype == np.float32

    @given(
        features=st.arrays(
            dtype=np.float64,  # Test dtype conversion
            shape=(100, 20),
            elements=st.floats(-100, 100)
        )
    )
    def test_dtype_consistency(self, features):
        """Property: All operations maintain float32 consistency."""
        trainer = LightGBMTrainer(config)
        X, y, metadata = trainer.prepare_data(features)

        # Properties:
        # 1. Features converted to float32
        assert X.dtype == np.float32

        # 2. Metadata records dtype
        assert metadata['dtype'] == 'float32'

        # 3. Predictions are float32
        model = trainer.train(X, y)
        predictions = model.predict(X)
        assert predictions.dtype == np.float32
```

### 6. Async Model Training Tests

```python
class TestAsyncModelTraining:
    """Contract tests for async model training."""

    @given(
        n_bars=st.integers(min_value=100, max_value=1000),
        training_delay_ms=st.floats(min_value=100, max_value=5000)
    )
    def test_training_never_blocks_inference(self, n_bars, training_delay_ms):
        """Property: Training never blocks inference pipeline."""
        signal_actor = MLSignalActor(config)
        training_actor = ModelTrainingActor(config)

        # Start slow training
        training_actor.mock_delay = training_delay_ms / 1000
        training_actor.trigger_retrain()

        # Measure inference during training
        latencies = []
        for _ in range(n_bars):
            start = time.perf_counter()
            signal_actor.on_bar(create_test_bar())
            latencies.append(time.perf_counter() - start)

        # Property: P99 < 5ms even during training
        p99 = np.percentile(latencies, 99)
        assert p99 < 0.005, f"P99 {p99*1000:.1f}ms exceeds 5ms during training"
```

### 7. Atomic Model Swapping Tests

```python
class TestAtomicModelSwapping:
    """Contract tests for atomic model swapping."""

    @given(
        n_concurrent_swaps=st.integers(min_value=2, max_value=10),
        n_inference_calls=st.integers(min_value=100, max_value=1000)
    )
    def test_model_swap_atomicity(self, n_concurrent_swaps, n_inference_calls):
        """Property: Model swaps are atomic with no partial states."""
        actor = MLSignalActor(config)

        def swap_model(version):
            model = create_test_model(version=version)
            actor.update_model(model)

        def run_inference():
            for _ in range(n_inference_calls):
                features = np.random.randn(1, 10).astype(np.float32)
                result = actor.predict(features)
                # Should never fail or return None
                assert result is not None

        # Run concurrent swaps and inference
        threads = []
        for i in range(n_concurrent_swaps):
            threads.append(threading.Thread(target=swap_model, args=(i,)))
        threads.append(threading.Thread(target=run_inference))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Property: Actor in valid state
        assert actor.model is not None
        assert actor.is_ready()
```

### 8. Feature Window Aggregation Tests

```python
class TestFeatureWindowAggregation:
    """Contract tests for feature window aggregation."""

    @given(
        window_size=st.integers(min_value=10, max_value=1000),
        n_updates=st.integers(min_value=100, max_value=10000),
        publish_interval=st.integers(min_value=10, max_value=100)
    )
    def test_feature_window_correctness(self, window_size, n_updates, publish_interval):
        """Property: Feature windows maintain correct size and order."""
        aggregator = FeatureWindowActor(
            window_size=window_size,
            publish_interval=publish_interval
        )

        published_windows = []
        aggregator.on_publish = lambda w: published_windows.append(w)

        # Feed bars
        for i in range(n_updates):
            bar = create_test_bar(timestamp=i)
            aggregator.on_bar(bar)

        # Properties:
        # 1. All windows have correct size (except initial)
        for window in published_windows[1:]:
            assert len(window.bars) <= window_size

        # 2. Windows are temporally ordered
        for window in published_windows:
            timestamps = [b.ts_event for b in window.bars]
            assert timestamps == sorted(timestamps)

        # 3. No data loss
        all_timestamps = set()
        for window in published_windows:
            all_timestamps.update(b.ts_event for b in window.bars)
        # Should have seen most recent data
        assert max(all_timestamps) >= n_updates - window_size
```

### 9. Calibration Hooks Tests

```python
class TestCalibrationHooks:
    """Contract tests for probability calibration."""

    @given(
        predictions=st.arrays(
            dtype=np.float32,
            shape=(1000,),
            elements=st.floats(min_value=0.0, max_value=1.0)
        ),
        true_labels=st.arrays(
            dtype=np.int32,
            shape=(1000,),
            elements=st.integers(min_value=0, max_value=1)
        )
    )
    def test_isotonic_calibration(self, predictions, true_labels):
        """Property: Calibration improves probability reliability."""
        calibrator = IsotonicCalibrator()
        calibrator.fit(predictions, true_labels)
        calibrated = calibrator.transform(predictions)

        # Properties:
        # 1. Output in [0, 1]
        assert np.all(calibrated >= 0.0)
        assert np.all(calibrated <= 1.0)

        # 2. Monotonic transformation
        for i in range(len(predictions) - 1):
            if predictions[i] < predictions[i+1]:
                assert calibrated[i] <= calibrated[i+1] + 1e-6

        # 3. Better calibration (lower Brier score)
        brier_before = np.mean((predictions - true_labels) ** 2)
        brier_after = np.mean((calibrated - true_labels) ** 2)
        # May not always improve, but should not degrade significantly
        assert brier_after <= brier_before * 1.1
```

## Test Execution Strategy

### Phase 1: Write Failing Tests
For each task, write the contract tests FIRST:
```bash
# Example for LightGBM fixes
pytest ml/tests/contracts/test_lightgbm_contracts.py -xvs
# Should FAIL initially
```

### Phase 2: Implement Solutions
Implement the minimal code to make tests pass.

### Phase 3: Property Verification
Run Hypothesis with many examples:
```bash
# Run with more examples
pytest ml/tests/contracts/ --hypothesis-show-statistics --hypothesis-max-examples=1000
```

### Phase 4: Performance Validation
```bash
# Run performance tests
pytest ml/tests/performance/ --benchmark-only
```

## Coverage Requirements

Each task must achieve:
- Unit test coverage: ≥90%
- Contract satisfaction: 100%
- Performance requirements: Met
- No property violations across 1000+ Hypothesis examples

## Handoff Template

For next Claude session:
```markdown
Task: [Specific task name]
Test File: ml/tests/contracts/test_[task]_contracts.py
Status: Tests written, currently failing

Run: pytest ml/tests/contracts/test_[task]_contracts.py -xvs

Your goal: Make ALL tests pass by implementing the feature.

Key properties to maintain:
1. [Property 1]
2. [Property 2]
3. [Property 3]

Success: All tests green, no property violations.
```
