 Detailed Plan for 90% ML Test Coverage Using Nautilus Trader Infrastructure

  Understanding the Current Architecture

  Based on my analysis, the ML module is already integrated with Nautilus Trader:

- Data Loading: Uses ParquetDataCatalog for historical data
- Market Data: Leverages Nautilus Bar, QuoteTick, TradeTick objects
- Indicators: Uses Nautilus indicators (RSI, MACD, Bollinger Bands, etc.)
- Strategies: Extends Nautilus Strategy base class
- DataBento: Available as an adapter for real market data

  Revised Detailed Testing Plan

  1. Integration Tests Using Nautilus Test Infrastructure

  A. Data Loading & Feature Engineering Tests

  ml/tests/integration/test_nautilus_data_pipeline.py

  Specific Tests:

```python
def test_parquet_catalog_to_ml_features():
    """Test complete data pipeline from ParquetDataCatalog to ML features."""
    # Setup
    catalog = ParquetDataCatalog("./test_data")
    loader = MLDataLoader(catalog)
    feature_engineer = FeatureEngineer(config)

    # Load real bars from catalog
    bars_df = loader.load_bars("EURUSD.SIM", start="2024-01-01", end="2024-01-31")

    # Convert to Nautilus Bar objects
    bars = [Bar.from_dict(row) for row in bars_df.to_dicts()]

    # Test batch feature calculation
    batch_features = feature_engineer.calculate_features_batch(bars_df)

    # Test online feature calculation with Nautilus indicators
    indicator_manager = IndicatorManager(config)
    online_features = []
    for bar in bars:
        features = feature_engineer.calculate_features_online(bar, indicator_manager)
        online_features.append(features.copy())

    # Validate parity
    assert_features_match(batch_features, online_features, tolerance=1e-10)
```

```python
def test_nautilus_indicators_consistency():
    """Test that Nautilus indicators produce consistent results batch vs streaming."""
    # Test RSI, MACD, Bollinger Bands with real market data
    bars = load_test_bars()  # From ParquetDataCatalog

    # Batch calculation
    rsi_batch = calculate_rsi_batch(bars, period=14)

    # Streaming calculation
    rsi = RelativeStrengthIndex(14)
    rsi_streaming = []
    for bar in bars:
        rsi.update(bar.close)
        rsi_streaming.append(rsi.value)

    # Must match exactly
    np.testing.assert_allclose(rsi_batch, rsi_streaming, rtol=1e-10)
```

  B. Strategy Backtesting Integration Tests

  ml/tests/integration/test_ml_strategy_backtest.py

  Specific Tests:

```python
def test_ml_strategy_with_real_signals():
    """Test ML strategy execution in Nautilus backtesting engine."""
    # Setup Nautilus backtest engine
    config = BacktestEngineConfig(
        strategies=[
            MLStrategyConfig(
                instrument_id=InstrumentId("EURUSD.SIM"),
                ml_signal_source="TEST_SIGNAL_ACTOR",
                position_size_pct=0.1,
                min_confidence=0.7
            )
        ]
    )

    engine = BacktestEngine(config)

    # Add historical data from ParquetDataCatalog
    catalog = ParquetDataCatalog("./test_data")
    engine.add_data(catalog.bars("EURUSD.SIM"))

    # Add pre-generated ML signals
    test_signals = generate_test_ml_signals()
    engine.add_data(test_signals)

    # Run backtest
    engine.run()

    # Validate results
    assert engine.portfolio.balance_total(USD) > Money(10000, USD)
    assert len(engine.cache.orders()) > 0
    assert engine.analyzer.sharpe_ratio() > 0
```

```python
def test_ml_signal_actor_in_backtest():
    """Test ML signal generation during backtest."""
    # Load pre-trained model
    model = load_test_model()  # XGBoost or LightGBM

    # Configure signal actor
    signal_actor = MLSignalActor(
        config=MLSignalActorConfig(
            model_path="./test_model.onnx",
            feature_config=FeatureConfig(),
            signal_strategy="threshold"
        )
    )

    # Run in backtest with real bars
    engine = BacktestEngine()
    engine.add_actor(signal_actor)
    engine.add_strategy(SimpleMLStrategy(config))
    engine.add_data(catalog.bars("EURUSD.SIM"))

    engine.run()

    # Validate signals were generated
    assert signal_actor.signals_generated > 0
    assert signal_actor.inference_latency_p99 < 5  # ms
```

  2. Training Pipeline Tests with Real Data

  A. End-to-End Training Tests

  ml/tests/integration/test_training_pipeline.py

  Specific Tests:

```python
def test_xgboost_training_with_nautilus_data():
    """Test XGBoost training with real market data from Nautilus."""
    # Load data from ParquetDataCatalog
    catalog = ParquetDataCatalog("./test_data")
    loader = MLDataLoader(catalog)

    # Load multiple instruments
    instruments = ["EURUSD.SIM", "GBPUSD.SIM", "USDJPY.SIM"]
    data = loader.load_multiple(instruments, data_type="bars")

    # Feature engineering with Nautilus indicators
    feature_engineer = FeatureEngineer(FeatureConfig())
    features_df = feature_engineer.calculate_features_batch(data)

    # Create labels (e.g., next bar returns)
    labels = create_labels_from_bars(data)

    # Train XGBoost model
    trainer = XGBoostTrainer(
        XGBoostTrainingConfig(
            data_source="nautilus_catalog",
            target_column="label",
            feature_columns=features_df.columns[:-1],
            hyperparameters={"n_estimators": 100, "max_depth": 5}
        )
    )

    model = trainer.train(features_df, labels)

    # Validate model
    assert model is not None
    assert trainer.get_feature_importance() is not None

    # Test ONNX conversion for hot path
    onnx_model = trainer.convert_to_onnx()
    assert onnx_model is not None

    # Validate inference latency
    test_features = features_df.head(100).to_numpy()
    start = time.perf_counter_ns()
    predictions = onnx_model.run(None, {"input": test_features})[0]
    latency_ms = (time.perf_counter_ns() - start) / 1_000_000
    assert latency_ms < 2  # Must be under 2ms for hot path
```

```python
def test_feature_parity_with_live_indicators():
    """Test that training features match live indicator calculations."""
    # Load historical bars
    bars = catalog.bars("EURUSD.SIM", start="2024-01-01", end="2024-01-31")

    # Calculate features in batch (training)
    batch_features = calculate_batch_features(bars)

    # Simulate live calculation
    indicator_manager = IndicatorManager()
    live_features = []

    for bar in bars:
        # This is what happens in production
        features = indicator_manager.calculate_features(bar)
        live_features.append(features)

    # Features MUST match with extreme precision
    np.testing.assert_allclose(
        batch_features,
        np.array(live_features),
        rtol=1e-10,
        err_msg="Feature parity violation detected!"
    )
```

  B. MLflow Integration with Nautilus Metadata

  ml/tests/integration/test_mlflow_nautilus.py

  Specific Tests:

```python
def test_mlflow_tracking_with_nautilus_metrics():
    """Test MLflow tracking of Nautilus-specific metrics."""
    with MLflowManager.run_context("nautilus_ml_test") as run:
        # Log Nautilus-specific parameters
        run.log_params({
            "instrument": "EURUSD.SIM",
            "bar_type": "1-MINUTE-LAST",
            "indicator_periods": {"rsi": 14, "ema_fast": 12, "ema_slow": 26},
            "data_source": "ParquetDataCatalog"
        })

        # Train model with Nautilus data
        model = train_with_nautilus_data()

        # Log Nautilus trading metrics
        backtest_results = run_backtest_with_model(model)
        run.log_metrics({
            "sharpe_ratio": backtest_results.sharpe_ratio,
            "max_drawdown": backtest_results.max_drawdown,
            "win_rate": backtest_results.win_rate,
            "profit_factor": backtest_results.profit_factor,
            "total_trades": backtest_results.total_trades
        })

        # Log model with Nautilus metadata
        run.log_model(
            model,
            artifact_path="model",
            metadata={
                "compatible_nautilus_version": "1.193.0",
                "required_indicators": ["RSI", "MACD", "BollingerBands"],
                "expected_bar_type": "1-MINUTE-LAST"
            }
        )
```

  3. Performance & Load Tests with Nautilus Components

  A. Hot Path Performance Tests

  ml/tests/performance/test_hot_path_nautilus.py

  Specific Tests:

```python
def test_signal_generation_latency():
    """Test ML signal generation meets Nautilus latency requirements."""
    # Setup
    model = load_onnx_model("test_model.onnx")
    signal_actor = MLSignalActor(config)

    # Pre-warm indicators with historical bars
    warm_up_bars = load_bars(limit=100)
    for bar in warm_up_bars:
        signal_actor.on_bar(bar)

    # Measure latency on hot path
    test_bars = load_bars(offset=100, limit=1000)
    latencies = []

    for bar in test_bars:
        start = time.perf_counter_ns()
        signal = signal_actor.on_bar(bar)  # This includes feature calc + inference
        latency_ns = time.perf_counter_ns() - start
        latencies.append(latency_ns)

    # Validate requirements
    latencies_ms = np.array(latencies) / 1_000_000
    assert np.percentile(latencies_ms, 50) < 2  # P50 < 2ms
    assert np.percentile(latencies_ms, 99) < 5  # P99 < 5ms
    assert np.max(latencies_ms) < 10  # Max < 10ms
```

```python
def test_concurrent_signal_processing():
    """Test ML actors handle concurrent bar updates."""
    # This tests thread safety with Nautilus message bus
    actors = [
        MLSignalActor(config) for _ in range(10)
    ]

    # Simulate concurrent bar updates
    import threading
    errors = []

    def process_bars(actor, bars):
        try:
            for bar in bars:
                actor.on_bar(bar)
        except Exception as e:
            errors.append(e)

    threads = []
    for actor in actors:
        t = threading.Thread(target=process_bars, args=(actor, test_bars))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    assert len(errors) == 0  # No thread safety issues
```

  4. DataBento Integration Tests

  A. Real Market Data Tests

  ml/tests/integration/test_databento_ml.py

  Specific Tests:

```python
def test_ml_features_from_databento():
    """Test feature calculation from DataBento market data."""
    # Load DataBento data
    from nautilus_trader.adapters.databento import databento_data

    bars = databento_data(
        symbols=["ES"],  # E-mini S&P futures
        start="2024-01-01",
        end="2024-01-31",
        schema="ohlcv-1m",
        stype="futures"
    )

    # Calculate ML features
    feature_engineer = FeatureEngineer()
    features = feature_engineer.calculate_features_batch(bars)

    # Validate features
    assert not features.is_empty()
    assert "rsi_14" in features.columns
    assert features["volume_ratio"].min() > 0
```

```python
def test_live_ml_signal_generation():
    """Test ML signal generation with live DataBento feed simulation."""
    # This would use historical data to simulate live feed
    pass
```

  5. Monitoring & Metrics Integration Tests

  A. Prometheus Metrics with Nautilus

  ml/tests/integration/test_metrics_nautilus.py

  Specific Tests:

```python
def test_ml_metrics_during_backtest():
    """Test that ML metrics are properly collected during backtesting."""
    # Setup Prometheus test registry
    test_registry = CollectorRegistry()

    # Run backtest with ML strategy
    engine = BacktestEngine()
    strategy = SimpleMLStrategy(config)
    engine.add_strategy(strategy)
    engine.run()

    # Verify metrics were collected
    metrics = get_metrics_from_registry(test_registry)
    assert metrics["ml_signals_received"] > 0
    assert metrics["ml_inference_latency_seconds_count"] > 0
    assert metrics["ml_feature_computation_seconds_sum"] > 0
```

  6. Specific Refactoring Needed

  A. Test-Friendly Configuration Factory

```python
# ml/config/test_utils.py
class TestConfigFactory:
    @staticmethod
    def create_training_config(**overrides) -> TrainingConfig:
        """Create training config with test defaults."""
        defaults = {
            "data_source": "test_catalog",
            "target_column": "label",
            "catalog_path": "./test_data"
        }
        defaults.update(overrides)
        return TrainingConfig(**defaults)

    @staticmethod
    def create_ml_strategy_config(**overrides) -> MLStrategyConfig:
        """Create ML strategy config for testing."""
        defaults = {
            "instrument_id": InstrumentId("TEST.SIM"),
            "ml_signal_source": "TEST_ACTOR",
            "position_size_pct": 0.1
        }
        defaults.update(overrides)
        return MLStrategyConfig(**defaults)
```

  B. Mock Nautilus Components

```python
# ml/tests/mocks/nautilus_mocks.py
class MockParquetCatalog:
    """Mock ParquetDataCatalog for unit tests."""
    def bars(self, instrument: str, **kwargs):
        return generate_test_bars()

class MockBacktestEngine:
    """Simplified backtest engine for ML testing."""
    def run_with_ml_strategy(self, strategy, bars, signals):
        # Simplified execution for testing
        pass
```

  Coverage Targets with This Plan

  | Component       | Current | After Unit Tests | After Integration | After E2E | Final |
  |-----------------|---------|------------------|-------------------|-----------|-------|
  | MLDataLoader    | 100%    | 100%             | 100%              | 100%      | 100%  |
  | FeatureEngineer | 95%     | 95%              | 98%               | 99%       | 99%   |
  | MLSignalActor   | 97%     | 97%              | 99%               | 99%       | 99%   |
  | Training        | 18%     | 60%              | 85%               | 90%       | 92%   |
  | MLflow/Tracking | 23%     | 40%              | 75%               | 85%       | 90%   |
  | Strategies      | 90%     | 90%              | 95%               | 98%       | 98%   |

## User approved Claude's plan
### Detailed Test Coverage Plan for ML Module - Leveraging Nautilus Infrastructure

#### Phase 1: Nautilus Integration Tests (Week 1)

**Focus:** Test ML components with real Nautilus data and indicators

1. **Data Pipeline Integration** (ml/tests/integration/test_nautilus_data_pipeline.py)
   - Test ParquetDataCatalog → MLDataLoader → FeatureEngineer pipeline
   - Validate Nautilus Bar/Quote/Trade objects flow through ML pipeline
   - Test batch vs online feature parity with real Nautilus indicators (RSI, MACD, BB)
   - Verify feature calculations match exactly (1e-10 tolerance)

2. **Indicator Consistency Tests** (ml/tests/integration/test_indicator_parity.py)
   - Test all 13+ Nautilus indicators used in ML features
   - Compare batch calculations vs streaming updates
   - Validate warm-up periods and state management
   - Test with edge cases (gaps, outliers, missing data)

3. **Strategy Backtesting Integration** (ml/tests/integration/test_ml_strategy_backtest.py)
   - Test ML strategies in Nautilus BacktestEngine
   - Validate signal generation → order execution flow
   - Test position sizing, risk management, stop losses
   - Verify performance metrics (Sharpe, drawdown, win rate)

#### Phase 2: Training Pipeline with Real Data (Week 2)

**Focus:** End-to-end training using Nautilus market data

1. **XGBoost/LightGBM Training Tests** (ml/tests/integration/test_training_pipeline.py)
   - Load multi-instrument data from ParquetDataCatalog
   - Train models with Nautilus-calculated features
   - Test ONNX conversion for production deployment
   - Validate inference latency < 2ms requirement

2. **Feature Engineering Validation** (ml/tests/integration/test_feature_validation.py)
   - Test feature parity between training and live paths
   - Validate microstructure features with real tick data
   - Test feature scaling and normalization consistency
   - Verify feature importance extraction

3. **MLflow Integration** (ml/tests/integration/test_mlflow_nautilus.py)
   - Track Nautilus-specific metrics (instrument, bar type, indicators)
   - Log backtest results alongside model metrics
   - Test model versioning with Nautilus metadata
   - Validate A/B testing with real trading metrics

#### Phase 3: Performance & Hot Path Tests (Week 3)

**Focus:** Validate latency and throughput requirements

1. **Hot Path Latency Tests** (ml/tests/performance/test_hot_path_nautilus.py)
   - Test feature computation < 500μs
   - Test model inference < 2ms
   - Test end-to-end signal generation < 5ms
   - Validate with 1000+ bars in rapid succession

2. **Concurrent Processing Tests** (ml/tests/performance/test_concurrency.py)
   - Test multiple ML actors processing bars simultaneously
   - Validate thread safety with Nautilus message bus
   - Test memory stability over 24h simulated operation
   - Verify zero allocations in hot path

3. **Load Testing** (ml/tests/performance/test_load.py)
   - Process 1M+ bars through ML pipeline
   - Test with 10+ concurrent strategies
   - Validate Prometheus metrics under load
   - Test graceful degradation under pressure

#### Phase 4: DataBento & Production Tests (Week 4)

**Focus:** Real market data and production scenarios

1. **DataBento Integration** (ml/tests/integration/test_databento_ml.py)
   - Test with real futures/options data from DataBento
   - Validate feature calculation from various schemas (ohlcv, mbp, trades)
   - Test with different asset classes (futures, options, equities)
   - Verify corporate action handling

2. **End-to-End Production Scenarios** (ml/tests/e2e/test_production_scenarios.py)
   - Complete flow: DataBento → Training → Deployment → Live Trading
   - Test model updates and rollbacks
   - Validate monitoring and alerting
   - Test disaster recovery scenarios

### Specific Refactoring Required

1. **Configuration Factories** (ml/config/test_utils.py)
   - TestConfigFactory with Nautilus-specific defaults
   - Mock data catalog configurations
   - Test instrument and venue configurations

2. **Nautilus Mock Components** (ml/tests/mocks/nautilus_mocks.py)
   - MockParquetCatalog for unit tests
   - MockBacktestEngine for strategy testing
   - MockMessageBus for actor communication tests

3. **Test Data Generators** (ml/tests/fixtures/data_generators.py)
   - Generate realistic Bar objects with proper timestamps
   - Create correlated multi-instrument data
   - Generate ML signals with various confidence levels

### Expected Coverage Improvements

| Component           | Current | Target | Method                                 |
|---------------------|---------|--------|----------------------------------------|
| MLDataLoader        | 100%    | 100%   | Already complete                       |
| FeatureEngineer     | 95%     | 99%    | Integration tests with real indicators |
| MLSignalActor       | 97%     | 99%    | Performance tests                      |
| Training (XGB/LGBM) | 18%     | 90%    | Integration tests with real data       |
| MLflow/Tracking     | 23%     | 90%    | Docker-based integration tests         |
| Strategies          | 90%     | 98%    | Backtest integration                   |
| Monitoring          | 86%     | 92%    | Load tests with metrics                |

### Key Differences from Generic Plan

1. **Uses Nautilus Components:** ParquetDataCatalog, BacktestEngine, Indicators
2. **Real Market Data:** DataBento integration for realistic testing
3. **Nautilus-Specific Metrics:** Sharpe ratio, drawdown, trade statistics
4. **Existing Infrastructure:** Leverages Nautilus test utilities and fixtures
5. **Performance Requirements:** Specific latency targets for trading systems

This plan provides concrete, actionable tests that validate the ML module works correctly with Nautilus Trader's existing infrastructure.
