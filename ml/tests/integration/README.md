# Integration Tests

This directory contains integration tests that verify ML components work correctly together and with external systems.

## Purpose

Integration tests validate that individual components interact properly when combined. They catch issues that unit tests miss, such as interface mismatches, data flow problems, and configuration conflicts.

## Test Categories

### Pipeline Integration Tests

#### `test_integration_pipeline.py`
End-to-end ML pipeline validation:
- Training → Model Saving → Loading → Inference → Signal → Trading
- Multi-model coordination and deployment
- Model hot-reloading without system interruption
- Performance requirements (<5ms end-to-end)

#### `test_training_inference_pipeline.py`
Training and inference pipeline integration:
- Feature parity between training and inference
- Model serialization/deserialization workflows
- Data preprocessing consistency
- Performance optimization validation

#### `test_training_pipeline_integration.py`
Training workflow integration:
- Data loading → Feature engineering → Model training
- Configuration propagation across pipeline stages
- Error handling and recovery mechanisms
- Model persistence and metadata management

### System Integration Tests

#### `test_ml_signal_pipeline.py`
ML signal generation and propagation:
- Actor → Signal → Strategy communication
- Signal quality and timing validation
- Multi-instrument signal coordination
- Circuit breaker and safety mechanisms

#### `test_ml_strategy_backtest.py`
Strategy integration with backtesting:
- ML signals → Trading decisions → P&L tracking
- Historical data pipeline integration
- Performance metrics calculation
- Strategy configuration validation

#### `test_nautilus_data_pipeline.py`
Integration with Nautilus data systems:
- Market data → Feature computation → Model inference
- Data catalog integration
- Indicator computation consistency
- Real-time vs historical data handling

### External System Integration

#### `test_databento_setup.py`
External data provider integration:
- Data source connectivity and authentication
- Data format compatibility
- Error handling for network issues
- Rate limiting and quota management

#### `test_registry_integration.py`
Model registry system integration:
- Model deployment workflows
- Version management across environments
- Metadata propagation
- Rollback and recovery procedures

### Infrastructure Integration

#### `test_infrastructure.py`
System infrastructure integration:
- Component lifecycle management
- Resource allocation and cleanup
- Service discovery and communication
- Health monitoring and alerting

## Integration Test Patterns

### Service Communication
```python
def test_signal_actor_to_strategy_communication():
    """Test actor publishes signals that strategies receive."""
    # Setup both components in test environment
    signal_actor = MLSignalActor(config)
    strategy = SimpleMLStrategy(config)

    # Connect via test message bus
    connect_components(signal_actor, strategy)

    # Send market data and verify signal flow
    signal_actor.on_bar(test_bar)
    assert strategy.received_signals
```

### Data Flow Validation
```python
def test_training_to_inference_parity():
    """Ensure training features match inference features exactly."""
    # Train model with specific feature configuration
    model = train_model(training_data, feature_config)

    # Run inference with same configuration
    inference_features = compute_inference_features(test_data, feature_config)
    training_features = compute_training_features(test_data, feature_config)

    # Verify exact parity (1e-10 tolerance)
    np.testing.assert_allclose(training_features, inference_features, rtol=1e-10)
```

## When to Add Integration Tests

Add integration tests when:
- Connecting new components together
- Integrating with external systems
- Adding new data flows or pipelines
- Implementing cross-component features
- Validating end-to-end workflows

## Coverage and Performance

Integration tests should:
- **Focus on interfaces** between components
- **Validate data flow** correctness
- **Test error propagation** across boundaries
- **Verify configuration** consistency
- **Ensure performance** meets requirements

Integration tests are slower than unit tests but essential for system reliability.
