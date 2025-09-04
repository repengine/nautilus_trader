# Integration Testing Strategy

## Overview

This document defines comprehensive integration testing strategies for ML pipelines in Nautilus Trader. It covers end-to-end testing patterns, cross-domain integration tests, performance testing strategies, and continuous integration patterns to ensure reliable ML operations across all domains.

## Testing Architecture

### Testing Pyramid for ML Systems

```
                    ┌─────────────────────┐
                    │  E2E Pipeline Tests │  ← Full workflow validation
                    └─────────────────────┘
                  ┌───────────────────────────┐
                  │  Cross-Domain Integration │  ← Domain interaction tests
                  └───────────────────────────┘
              ┌─────────────────────────────────────┐
              │     Component Integration Tests     │  ← Store/registry integration
              └─────────────────────────────────────┘
          ┌─────────────────────────────────────────────┐
          │           Unit Tests (Existing)             │  ← Individual component tests
          └─────────────────────────────────────────────┘
```

### Test Categories

1. **End-to-End Pipeline Tests**: Complete data flow from ingestion to signal generation
2. **Cross-Domain Integration Tests**: Interactions between domains (data → features → models → strategies)
3. **Performance Integration Tests**: Latency, throughput, and resource usage under load
4. **Resilience Integration Tests**: Fallback behavior, error propagation, recovery
5. **Configuration Integration Tests**: Environment-specific configuration validation

## End-to-End Testing Patterns

### Pipeline Integration Test Framework

```python
import pytest
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from pathlib import Path
import time
import numpy as np
import pandas as pd

from ml.core.integration import MLIntegrationManager
from ml.common.correlation import create_correlation_id, trace_correlation_chain
from ml.testing.fixtures import TestDataGenerator
from ml.testing.assertions import MLAssertions

@dataclass
class PipelineTestScenario:
    """Test scenario configuration for pipeline testing."""
    
    name: str
    description: str
    
    # Input data configuration
    instruments: List[str]
    data_duration_minutes: int
    bar_frequency_seconds: int
    
    # Expected outputs
    min_features_generated: int
    min_predictions_made: int
    min_signals_generated: int
    
    # Performance expectations
    max_end_to_end_latency_ms: float
    max_memory_usage_mb: float
    
    # Quality expectations
    min_data_quality_score: float
    min_feature_parity_score: float
    min_prediction_confidence: float

class E2EPipelineTestRunner:
    """End-to-end pipeline test execution framework."""
    
    def __init__(self, integration_manager: MLIntegrationManager):
        self.integration = integration_manager
        self.test_data_generator = TestDataGenerator()
        self.assertions = MLAssertions()
        
        # Test tracking
        self.correlation_ids: List[str] = []
        self.performance_metrics: Dict[str, List[float]] = {}
    
    async def run_pipeline_test(self, scenario: PipelineTestScenario) -> Dict[str, Any]:
        """Run complete pipeline test scenario."""
        test_results = {
            "scenario": scenario.name,
            "status": "running",
            "start_time": time.time(),
            "metrics": {},
            "assertions": {},
            "errors": [],
        }
        
        try:
            # 1. Setup test environment
            await self._setup_test_environment(scenario)
            
            # 2. Generate and inject test data
            test_data = await self._generate_test_data(scenario)
            correlation_id = await self._inject_data(test_data)
            
            # 3. Wait for pipeline processing
            await self._wait_for_pipeline_completion(correlation_id, scenario)
            
            # 4. Validate end-to-end results
            validation_results = await self._validate_pipeline_results(correlation_id, scenario)
            test_results["assertions"] = validation_results
            
            # 5. Collect performance metrics
            performance_metrics = await self._collect_performance_metrics(correlation_id)
            test_results["metrics"] = performance_metrics
            
            # 6. Determine test status
            test_results["status"] = "passed" if all(validation_results.values()) else "failed"
            
        except Exception as e:
            test_results["status"] = "error"
            test_results["errors"].append(str(e))
        
        finally:
            test_results["end_time"] = time.time()
            test_results["duration_seconds"] = test_results["end_time"] - test_results["start_time"]
            await self._cleanup_test_environment()
        
        return test_results
    
    async def _setup_test_environment(self, scenario: PipelineTestScenario) -> None:
        """Setup isolated test environment."""
        # Ensure all components are healthy
        self.integration.ensure_healthy()
        
        # Clear any existing test data
        await self._clear_test_data()
        
        # Initialize test correlation tracking
        self.correlation_ids = []
        self.performance_metrics = {}
    
    async def _generate_test_data(self, scenario: PipelineTestScenario) -> pd.DataFrame:
        """Generate realistic test data for scenario."""
        test_data = self.test_data_generator.generate_market_data(
            instruments=scenario.instruments,
            duration_minutes=scenario.data_duration_minutes,
            bar_frequency_seconds=scenario.bar_frequency_seconds,
            include_microstructure=True,
            data_quality="high"
        )
        
        return test_data
    
    async def _inject_data(self, test_data: pd.DataFrame) -> str:
        """Inject test data into pipeline and return correlation ID."""
        correlation_id = create_correlation_id()
        
        # Inject data through DataStore
        for _, row in test_data.iterrows():
            await self.integration.data_store.write_bar_data(
                bar_data=row.to_dict(),
                correlation_id=correlation_id,
                ts_event=int(row['ts_event']),
                ts_init=int(time.time_ns())
            )
        
        self.correlation_ids.append(correlation_id)
        return correlation_id
    
    async def _wait_for_pipeline_completion(self, correlation_id: str, scenario: PipelineTestScenario) -> None:
        """Wait for pipeline to process all data."""
        timeout_seconds = scenario.data_duration_minutes * 60 + 30  # Buffer time
        start_time = time.time()
        
        while time.time() - start_time < timeout_seconds:
            # Check if pipeline processing is complete
            events = trace_correlation_chain(correlation_id)
            
            # Look for terminal events in strategy domain
            strategy_events = [e for e in events if e.get('domain') == 'strategy']
            if strategy_events:
                # Pipeline completed
                return
            
            await asyncio.sleep(1)  # Check every second
        
        raise TimeoutError(f"Pipeline completion timeout after {timeout_seconds}s")
    
    async def _validate_pipeline_results(self, correlation_id: str, scenario: PipelineTestScenario) -> Dict[str, bool]:
        """Validate end-to-end pipeline results."""
        validations = {}
        
        # 1. Data domain validation
        data_events = await self._get_domain_events(correlation_id, 'data')
        validations["data_ingested"] = len(data_events) > 0
        validations["data_quality"] = await self._validate_data_quality(correlation_id, scenario)
        
        # 2. Feature domain validation
        feature_events = await self._get_domain_events(correlation_id, 'features')
        validations["features_computed"] = len(feature_events) >= scenario.min_features_generated
        validations["feature_parity"] = await self._validate_feature_parity(correlation_id, scenario)
        
        # 3. Model domain validation
        model_events = await self._get_domain_events(correlation_id, 'model')
        validations["predictions_made"] = len(model_events) >= scenario.min_predictions_made
        validations["prediction_quality"] = await self._validate_prediction_quality(correlation_id, scenario)
        
        # 4. Strategy domain validation
        strategy_events = await self._get_domain_events(correlation_id, 'strategy')
        validations["signals_generated"] = len(strategy_events) >= scenario.min_signals_generated
        validations["signal_consistency"] = await self._validate_signal_consistency(correlation_id)
        
        # 5. End-to-end validation
        validations["end_to_end_latency"] = await self._validate_end_to_end_latency(correlation_id, scenario)
        validations["lineage_integrity"] = await self._validate_lineage_integrity(correlation_id)
        
        return validations
    
    async def _collect_performance_metrics(self, correlation_id: str) -> Dict[str, float]:
        """Collect performance metrics for the test run."""
        events = trace_correlation_chain(correlation_id)
        
        if not events:
            return {}
        
        # Calculate domain-specific latencies
        domain_latencies = {}
        domain_events = {}
        
        for event in events:
            domain = event.get('domain', 'unknown')
            if domain not in domain_events:
                domain_events[domain] = []
            domain_events[domain].append(event)
        
        # Calculate latencies between domains
        for domain, domain_event_list in domain_events.items():
            if len(domain_event_list) >= 2:
                start_time = min(e.get('ts_event', 0) for e in domain_event_list)
                end_time = max(e.get('ts_event', 0) for e in domain_event_list)
                domain_latencies[f"{domain}_latency_ns"] = end_time - start_time
        
        # Calculate total end-to-end latency
        if events:
            total_latency_ns = events[-1].get('ts_event', 0) - events[0].get('ts_event', 0)
            domain_latencies['total_latency_ns'] = total_latency_ns
            domain_latencies['total_latency_ms'] = total_latency_ns / 1_000_000
        
        return domain_latencies
    
    async def _validate_data_quality(self, correlation_id: str, scenario: PipelineTestScenario) -> bool:
        """Validate data quality meets expectations."""
        # Get data quality metrics from DataStore
        quality_metrics = await self.integration.data_store.get_quality_metrics(correlation_id)
        quality_score = quality_metrics.get('overall_score', 0.0)
        
        return quality_score >= scenario.min_data_quality_score
    
    async def _validate_feature_parity(self, correlation_id: str, scenario: PipelineTestScenario) -> bool:
        """Validate batch/online feature parity."""
        # Get feature parity metrics
        parity_metrics = await self.integration.feature_store.get_parity_metrics(correlation_id)
        parity_score = parity_metrics.get('parity_score', 0.0)
        
        return parity_score >= scenario.min_feature_parity_score
    
    async def _validate_prediction_quality(self, correlation_id: str, scenario: PipelineTestScenario) -> bool:
        """Validate prediction quality."""
        predictions = await self.integration.model_store.get_predictions(correlation_id)
        
        if not predictions:
            return False
        
        # Check average confidence
        confidences = [p.get('confidence', 0.0) for p in predictions]
        avg_confidence = np.mean(confidences)
        
        return avg_confidence >= scenario.min_prediction_confidence
    
    async def _validate_signal_consistency(self, correlation_id: str) -> bool:
        """Validate signal consistency and logic."""
        signals = await self.integration.strategy_store.get_signals(correlation_id)
        
        if not signals:
            return False
        
        # Check for logical consistency
        # - No contradictory signals at same timestamp
        # - Signal strength matches prediction confidence
        # - Risk constraints are respected
        
        return True  # Simplified for example
    
    async def _validate_end_to_end_latency(self, correlation_id: str, scenario: PipelineTestScenario) -> bool:
        """Validate end-to-end latency meets SLA."""
        metrics = await self._collect_performance_metrics(correlation_id)
        total_latency_ms = metrics.get('total_latency_ms', float('inf'))
        
        return total_latency_ms <= scenario.max_end_to_end_latency_ms
    
    async def _validate_lineage_integrity(self, correlation_id: str) -> bool:
        """Validate complete lineage traceability."""
        events = trace_correlation_chain(correlation_id)
        
        # Check that all domains are represented
        domains = set(e.get('domain') for e in events)
        expected_domains = {'data', 'features', 'model', 'strategy'}
        
        return expected_domains.issubset(domains)

# Test scenarios
PIPELINE_TEST_SCENARIOS = [
    PipelineTestScenario(
        name="basic_single_instrument",
        description="Basic pipeline test with single EUR/USD instrument",
        instruments=["EUR/USD"],
        data_duration_minutes=5,
        bar_frequency_seconds=60,
        min_features_generated=5,
        min_predictions_made=5,
        min_signals_generated=3,
        max_end_to_end_latency_ms=100.0,
        max_memory_usage_mb=256,
        min_data_quality_score=0.95,
        min_feature_parity_score=0.999,
        min_prediction_confidence=0.6,
    ),
    
    PipelineTestScenario(
        name="multi_instrument_stress",
        description="Stress test with multiple instruments",
        instruments=["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD"],
        data_duration_minutes=10,
        bar_frequency_seconds=30,
        min_features_generated=40,
        min_predictions_made=40,
        min_signals_generated=20,
        max_end_to_end_latency_ms=200.0,
        max_memory_usage_mb=512,
        min_data_quality_score=0.9,
        min_feature_parity_score=0.999,
        min_prediction_confidence=0.55,
    ),
    
    PipelineTestScenario(
        name="high_frequency_load",
        description="High frequency data processing test",
        instruments=["EUR/USD"],
        data_duration_minutes=2,
        bar_frequency_seconds=1,  # 1-second bars
        min_features_generated=120,
        min_predictions_made=120,
        min_signals_generated=60,
        max_end_to_end_latency_ms=50.0,  # Stricter latency
        max_memory_usage_mb=128,
        min_data_quality_score=0.95,
        min_feature_parity_score=0.999,
        min_prediction_confidence=0.6,
    ),
]

# Pytest integration
@pytest.mark.integration
@pytest.mark.asyncio
class TestE2EPipeline:
    """End-to-end pipeline integration tests."""
    
    @pytest.fixture
    async def integration_manager(self):
        """Setup integration manager for testing."""
        manager = MLIntegrationManager(
            auto_start_postgres=True,
            auto_migrate=True,
            ensure_healthy=True
        )
        yield manager
        manager.shutdown()
    
    @pytest.fixture
    def test_runner(self, integration_manager):
        """Test runner fixture."""
        return E2EPipelineTestRunner(integration_manager)
    
    @pytest.mark.parametrize("scenario", PIPELINE_TEST_SCENARIOS)
    async def test_pipeline_scenario(self, test_runner, scenario):
        """Test pipeline with specific scenario."""
        results = await test_runner.run_pipeline_test(scenario)
        
        # Assert overall test success
        assert results["status"] in ["passed"], f"Test failed: {results}"
        
        # Assert specific validations
        assertions = results["assertions"]
        for validation_name, passed in assertions.items():
            assert passed, f"Validation failed: {validation_name}"
        
        # Assert performance metrics
        metrics = results["metrics"]
        if "total_latency_ms" in metrics:
            assert metrics["total_latency_ms"] <= scenario.max_end_to_end_latency_ms
```

## Cross-Domain Integration Tests

### Domain Interaction Testing

```python
class CrossDomainIntegrationTests:
    """Test interactions between ML domains."""
    
    def __init__(self, integration_manager: MLIntegrationManager):
        self.integration = integration_manager
    
    @pytest.mark.integration
    async def test_data_to_features_flow(self):
        """Test data flowing correctly to feature computation."""
        # 1. Inject data in data domain
        correlation_id = create_correlation_id()
        
        test_bar = {
            'instrument_id': 'EUR/USD',
            'open': 1.1000,
            'high': 1.1005,
            'low': 1.0995,
            'close': 1.1002,
            'volume': 10000,
            'ts_event': time.time_ns(),
        }
        
        await self.integration.data_store.write_bar_data(
            bar_data=test_bar,
            correlation_id=correlation_id,
            ts_event=test_bar['ts_event'],
            ts_init=time.time_ns()
        )
        
        # 2. Wait for feature computation trigger
        await asyncio.sleep(2)  # Allow processing time
        
        # 3. Verify features were computed
        feature_events = await self._get_domain_events(correlation_id, 'features')
        assert len(feature_events) > 0, "No feature computation events found"
        
        # 4. Verify feature store contains computed features
        features = await self.integration.feature_store.get_latest_features(
            'EUR/USD', test_bar['ts_event']
        )
        assert features is not None, "Features not found in store"
        assert len(features) > 0, "Empty features computed"
    
    @pytest.mark.integration
    async def test_features_to_model_flow(self):
        """Test features flowing correctly to model inference."""
        # 1. Setup test features
        correlation_id = create_correlation_id()
        
        test_features = {
            'close': 1.1002,
            'volume': 10000,
            'rsi_14': 65.5,
            'sma_20': 1.0998,
            'bb_upper': 1.1010,
            'bb_lower': 1.0990,
        }
        
        await self.integration.feature_store.write_features(
            instrument_id='EUR/USD',
            features=test_features,
            ts_event=time.time_ns(),
            ts_init=time.time_ns(),
            correlation_id=correlation_id
        )
        
        # 2. Wait for model inference trigger
        await asyncio.sleep(2)
        
        # 3. Verify model predictions were made
        model_events = await self._get_domain_events(correlation_id, 'model')
        assert len(model_events) > 0, "No model inference events found"
        
        # 4. Verify prediction store contains results
        predictions = await self.integration.model_store.get_predictions(correlation_id)
        assert len(predictions) > 0, "No predictions found in store"
        
        prediction = predictions[0]
        assert 0 <= prediction['probability'] <= 1, "Invalid prediction probability"
    
    @pytest.mark.integration
    async def test_model_to_strategy_flow(self):
        """Test model predictions flowing to strategy signals."""
        # 1. Setup test prediction
        correlation_id = create_correlation_id()
        
        test_prediction = {
            'model_id': 'test_model',
            'probability': 0.75,
            'confidence': 0.8,
            'features_used': ['close', 'volume', 'rsi_14'],
            'ts_event': time.time_ns(),
        }
        
        await self.integration.model_store.record_prediction(
            model_id='test_model',
            prediction=test_prediction,
            ts_event=test_prediction['ts_event'],
            instrument_id='EUR/USD',
            correlation_id=correlation_id
        )
        
        # 2. Wait for strategy signal generation
        await asyncio.sleep(2)
        
        # 3. Verify strategy signals were generated
        strategy_events = await self._get_domain_events(correlation_id, 'strategy')
        assert len(strategy_events) > 0, "No strategy signal events found"
        
        # 4. Verify signal store contains results
        signals = await self.integration.strategy_store.get_signals(correlation_id)
        assert len(signals) > 0, "No signals found in store"
        
        signal = signals[0]
        assert signal['signal_type'] in ['BUY', 'SELL', 'HOLD'], "Invalid signal type"
        assert signal['strength'] > 0, "Invalid signal strength"
    
    @pytest.mark.integration
    async def test_error_propagation_across_domains(self):
        """Test error propagation and handling across domains."""
        correlation_id = create_correlation_id()
        
        # 1. Inject invalid data that should cause errors
        invalid_data = {
            'instrument_id': 'INVALID/PAIR',
            'open': float('nan'),  # Invalid price
            'high': -1.0,          # Invalid price
            'low': None,           # Missing data
            'close': 'not_a_number',  # Wrong type
            'volume': -1000,       # Invalid volume
            'ts_event': time.time_ns(),
        }
        
        # 2. Verify error handling at each domain
        with pytest.raises(DataValidationError):
            await self.integration.data_store.write_bar_data(
                bar_data=invalid_data,
                correlation_id=correlation_id,
                ts_event=invalid_data['ts_event'],
                ts_init=time.time_ns()
            )
        
        # 3. Verify system remains healthy after error
        health = self.integration.check_health()
        assert health['postgres'], "Database should remain healthy"
        assert health['data_store'], "Data store should recover"
    
    @pytest.mark.integration
    async def test_cross_domain_transaction_consistency(self):
        """Test transaction consistency across domains."""
        correlation_id = create_correlation_id()
        
        # 1. Start cross-domain transaction
        async with self.integration.begin_cross_domain_transaction(correlation_id):
            
            # 2. Write to multiple domains within transaction
            await self.integration.data_store.write_bar_data(...)
            await self.integration.feature_store.write_features(...)
            await self.integration.model_store.record_prediction(...)
            
            # 3. Simulate error before commit
            if should_simulate_error:
                raise SimulatedError("Transaction rollback test")
        
        # 4. Verify rollback occurred across all domains
        # All data should be rolled back if transaction failed
        
    async def _get_domain_events(self, correlation_id: str, domain: str) -> List[Dict]:
        """Get events for specific domain and correlation ID."""
        all_events = trace_correlation_chain(correlation_id)
        return [e for e in all_events if e.get('domain') == domain]
```

## Performance Integration Tests

### Load and Stress Testing

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
import psutil
import resource

class PerformanceIntegrationTests:
    """Performance and load testing for ML pipeline."""
    
    def __init__(self, integration_manager: MLIntegrationManager):
        self.integration = integration_manager
        self.metrics_collector = PerformanceMetricsCollector()
    
    @pytest.mark.performance
    @pytest.mark.slow
    async def test_throughput_limits(self):
        """Test system throughput under increasing load."""
        throughput_results = []
        
        # Test different load levels
        for rps in [10, 50, 100, 500, 1000, 2000]:
            print(f"Testing {rps} requests per second...")
            
            # Run load test
            results = await self._run_load_test(
                requests_per_second=rps,
                duration_seconds=30,
                test_type="throughput"
            )
            
            throughput_results.append({
                'target_rps': rps,
                'actual_rps': results['actual_rps'],
                'error_rate': results['error_rate'],
                'p99_latency_ms': results['p99_latency_ms'],
                'memory_usage_mb': results['peak_memory_mb'],
            })
            
            # Stop if error rate becomes too high
            if results['error_rate'] > 0.05:  # 5% error threshold
                print(f"Stopping throughput test at {rps} RPS due to high error rate")
                break
        
        # Analyze results
        max_stable_rps = self._find_max_stable_throughput(throughput_results)
        assert max_stable_rps >= 100, f"Minimum throughput requirement not met: {max_stable_rps}"
        
        print(f"Maximum stable throughput: {max_stable_rps} RPS")
    
    @pytest.mark.performance
    async def test_latency_under_load(self):
        """Test latency distribution under sustained load."""
        
        # Run sustained load test
        results = await self._run_load_test(
            requests_per_second=200,  # Moderate load
            duration_seconds=300,     # 5 minutes
            test_type="latency"
        )
        
        # Validate latency SLAs
        assert results['p50_latency_ms'] < 20, "P50 latency SLA violation"
        assert results['p95_latency_ms'] < 50, "P95 latency SLA violation"
        assert results['p99_latency_ms'] < 100, "P99 latency SLA violation"
        
        # Check latency stability over time
        latency_variance = results['latency_variance']
        assert latency_variance < 0.2, "Latency too variable over time"
    
    @pytest.mark.performance
    async def test_memory_usage_stability(self):
        """Test memory usage stability under load."""
        
        # Monitor memory usage during load test
        memory_monitor = MemoryMonitor()
        memory_monitor.start()
        
        try:
            # Run extended load test
            await self._run_load_test(
                requests_per_second=100,
                duration_seconds=600,  # 10 minutes
                test_type="memory"
            )
            
            memory_stats = memory_monitor.get_statistics()
            
            # Check for memory leaks
            memory_growth_rate = memory_stats['growth_rate_mb_per_minute']
            assert memory_growth_rate < 5, f"Memory leak detected: {memory_growth_rate} MB/min"
            
            # Check memory usage stays within limits
            peak_memory_mb = memory_stats['peak_memory_mb']
            assert peak_memory_mb < 2048, f"Memory usage too high: {peak_memory_mb} MB"
            
        finally:
            memory_monitor.stop()
    
    @pytest.mark.performance
    async def test_cpu_utilization_efficiency(self):
        """Test CPU utilization efficiency."""
        
        cpu_monitor = CPUMonitor()
        cpu_monitor.start()
        
        try:
            # Run CPU-intensive load test
            await self._run_load_test(
                requests_per_second=500,
                duration_seconds=120,
                test_type="cpu"
            )
            
            cpu_stats = cpu_monitor.get_statistics()
            
            # Check CPU efficiency
            avg_cpu_usage = cpu_stats['average_cpu_percent']
            assert 20 < avg_cpu_usage < 80, f"CPU usage inefficient: {avg_cpu_usage}%"
            
            # Check CPU usage distribution across cores
            core_usage_variance = cpu_stats['core_usage_variance']
            assert core_usage_variance < 0.3, "Poor CPU load distribution across cores"
            
        finally:
            cpu_monitor.stop()
    
    async def _run_load_test(self, requests_per_second: int, duration_seconds: int, 
                            test_type: str) -> Dict[str, float]:
        """Run load test with specified parameters."""
        
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        # Performance tracking
        latencies = []
        error_count = 0
        success_count = 0
        
        # Memory tracking
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        peak_memory = initial_memory
        
        # Request generation
        request_interval = 1.0 / requests_per_second
        
        async def generate_requests():
            nonlocal error_count, success_count, peak_memory
            
            while time.time() < end_time:
                request_start = time.time()
                
                try:
                    # Generate test request
                    correlation_id = create_correlation_id()
                    await self._execute_test_request(correlation_id)
                    
                    # Record success and latency
                    success_count += 1
                    latency_ms = (time.time() - request_start) * 1000
                    latencies.append(latency_ms)
                    
                except Exception as e:
                    error_count += 1
                    print(f"Request error: {e}")
                
                # Update peak memory
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                peak_memory = max(peak_memory, current_memory)
                
                # Rate limiting
                elapsed = time.time() - request_start
                sleep_time = max(0, request_interval - elapsed)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
        
        # Run load test
        await generate_requests()
        
        # Calculate results
        total_requests = success_count + error_count
        actual_rps = total_requests / duration_seconds
        error_rate = error_count / total_requests if total_requests > 0 else 0
        
        latency_percentiles = {}
        if latencies:
            latency_percentiles = {
                'p50_latency_ms': np.percentile(latencies, 50),
                'p95_latency_ms': np.percentile(latencies, 95),
                'p99_latency_ms': np.percentile(latencies, 99),
                'latency_variance': np.var(latencies) / np.mean(latencies) if latencies else 0,
            }
        
        return {
            'actual_rps': actual_rps,
            'error_rate': error_rate,
            'peak_memory_mb': peak_memory,
            'memory_growth_mb': peak_memory - initial_memory,
            **latency_percentiles,
        }
    
    async def _execute_test_request(self, correlation_id: str) -> None:
        """Execute single test request through pipeline."""
        # Generate test data
        test_bar = self._generate_test_bar()
        
        # Execute through pipeline
        await self.integration.data_store.write_bar_data(
            bar_data=test_bar,
            correlation_id=correlation_id,
            ts_event=test_bar['ts_event'],
            ts_init=time.time_ns()
        )
        
        # Wait for processing (simplified)
        await asyncio.sleep(0.01)  # Small delay for processing
    
    def _find_max_stable_throughput(self, results: List[Dict]) -> float:
        """Find maximum stable throughput from test results."""
        for result in reversed(results):  # Start from highest load
            if result['error_rate'] <= 0.01 and result['p99_latency_ms'] <= 100:
                return result['actual_rps']
        return 0
```

## Resilience Integration Tests

### Fault Injection and Recovery Testing

```python
class ResilienceIntegrationTests:
    """Test system resilience and recovery capabilities."""
    
    @pytest.mark.resilience
    async def test_database_connection_failure(self):
        """Test behavior when database connection fails."""
        # 1. Verify system is initially healthy
        health = self.integration.check_health()
        assert health['postgres'], "Database should be initially healthy"
        
        # 2. Simulate database failure
        await self._simulate_database_failure()
        
        # 3. Verify graceful degradation to fallback stores
        health_after_failure = self.integration.check_health()
        assert not health_after_failure['postgres'], "Database should be detected as unhealthy"
        
        # 4. Verify system continues operating with fallbacks
        correlation_id = create_correlation_id()
        
        # Should not raise, should use dummy stores
        await self.integration.data_store.write_bar_data(
            bar_data=self._generate_test_bar(),
            correlation_id=correlation_id,
            ts_event=time.time_ns(),
            ts_init=time.time_ns()
        )
        
        # 5. Simulate database recovery
        await self._simulate_database_recovery()
        
        # 6. Verify system returns to normal operation
        await asyncio.sleep(5)  # Allow recovery time
        
        health_after_recovery = self.integration.check_health()
        assert health_after_recovery['postgres'], "Database should recover"
    
    @pytest.mark.resilience
    async def test_model_inference_failure(self):
        """Test behavior when model inference fails."""
        # 1. Setup test with failing model
        failing_model_config = ModelConfig(
            model_id="failing_model",
            model_path="nonexistent_model.onnx"  # Will cause loading failure
        )
        
        # 2. Verify system handles model loading failure gracefully
        with pytest.raises(ModelLoadingError):
            await self.integration.model_registry.load_model("failing_model")
        
        # 3. Verify fallback model is used
        fallback_model = await self.integration.model_registry.get_fallback_model()
        assert fallback_model is not None, "Fallback model should be available"
        
        # 4. Verify pipeline continues with fallback
        correlation_id = create_correlation_id()
        
        # Pipeline should complete despite model failure
        await self._run_minimal_pipeline_test(correlation_id)
        
        # Should have events indicating fallback was used
        events = trace_correlation_chain(correlation_id)
        fallback_events = [e for e in events if 'fallback' in e.get('event_type', '')]
        assert len(fallback_events) > 0, "Fallback events should be recorded"
    
    @pytest.mark.resilience
    async def test_feature_computation_timeout(self):
        """Test behavior when feature computation times out."""
        # 1. Configure feature computation with very short timeout
        original_timeout = self.integration.feature_domain.max_feature_computation_time_ms
        self.integration.feature_domain.max_feature_computation_time_ms = 1.0  # 1ms timeout
        
        try:
            # 2. Run test that should trigger timeout
            correlation_id = create_correlation_id()
            
            # Use complex feature configuration that will take longer than 1ms
            complex_features = self._generate_complex_features()
            
            # Should handle timeout gracefully
            result = await self.integration.feature_store.compute_features_with_timeout(
                instrument_id='EUR/USD',
                data=complex_features,
                correlation_id=correlation_id
            )
            
            # 3. Verify fallback behavior
            assert result is not None, "Should return cached/default features on timeout"
            
            # 4. Verify timeout is recorded in metrics
            timeout_metrics = await self.integration.feature_store.get_timeout_metrics()
            assert timeout_metrics['timeout_count'] > 0, "Timeout should be recorded"
            
        finally:
            # Restore original timeout
            self.integration.feature_domain.max_feature_computation_time_ms = original_timeout
    
    @pytest.mark.resilience
    async def test_memory_pressure_handling(self):
        """Test behavior under memory pressure."""
        # 1. Monitor initial memory usage
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # 2. Generate memory pressure by loading many models
        loaded_models = []
        try:
            for i in range(100):  # Try to load many models
                model_id = f"test_model_{i}"
                try:
                    model = await self.integration.model_registry.load_model(model_id)
                    loaded_models.append(model)
                except MemoryError:
                    # Expected when memory pressure occurs
                    break
                except ModelNotFoundError:
                    # Create dummy model for testing
                    dummy_model = self._create_dummy_model(model_id)
                    loaded_models.append(dummy_model)
        
            # 3. Verify system handles memory pressure gracefully
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_growth = current_memory - initial_memory
            
            # Should not grow unboundedly
            assert memory_growth < 1024, f"Memory growth too high: {memory_growth} MB"
            
            # 4. Verify LRU eviction is working
            cache_stats = await self.integration.model_registry.get_cache_statistics()
            assert cache_stats['evictions'] > 0, "LRU eviction should have occurred"
            
        finally:
            # Clean up loaded models
            loaded_models.clear()
    
    @pytest.mark.resilience 
    async def test_network_partition_recovery(self):
        """Test behavior during network partitions."""
        # 1. Simulate network partition between services
        await self._simulate_network_partition()
        
        try:
            # 2. Verify local fallbacks are used
            correlation_id = create_correlation_id()
            
            # Should continue operating locally
            await self._run_minimal_pipeline_test(correlation_id)
            
            # 3. Verify queuing of operations for later sync
            queued_operations = await self.integration.get_queued_operations()
            assert len(queued_operations) > 0, "Operations should be queued during partition"
            
        finally:
            # 4. Simulate network recovery
            await self._simulate_network_recovery()
        
        # 5. Verify sync occurs after recovery
        await asyncio.sleep(10)  # Allow sync time
        
        synced_operations = await self.integration.get_synced_operations()
        assert len(synced_operations) > 0, "Operations should be synced after recovery"
```

## Continuous Integration Patterns

### CI/CD Pipeline Integration

```python
# CI configuration example (.github/workflows/ml-integration-tests.yml)
"""
name: ML Integration Tests

on:
  pull_request:
    paths:
      - 'ml/**'
  push:
    branches: [main, develop]

jobs:
  integration-tests:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: nautilus_test
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r ml/requirements-test.txt
    
    - name: Run integration tests
      env:
        DB_CONNECTION: postgresql://postgres:postgres@localhost:5432/nautilus_test
        ML_ENVIRONMENT: testing
        ML_AUTO_MIGRATE: true
      run: |
        pytest ml/tests/integration/ -v --tb=short
    
    - name: Run performance tests
      env:
        DB_CONNECTION: postgresql://postgres:postgres@localhost:5432/nautilus_test
        ML_ENVIRONMENT: testing
      run: |
        pytest ml/tests/performance/ -v --tb=short --durations=10
    
    - name: Upload test results
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: test-results
        path: test-results.xml
"""

class ContinuousIntegrationTestRunner:
    """CI-specific test runner with optimizations."""
    
    def __init__(self):
        self.test_timeout_seconds = 300  # 5 minutes max per test
        self.parallel_test_count = 4
        
    async def run_ci_test_suite(self) -> Dict[str, Any]:
        """Run complete CI test suite with optimizations."""
        
        # Test categories to run in CI
        test_categories = [
            "smoke_tests",         # Fast smoke tests first
            "integration_tests",   # Core integration tests
            "performance_tests",   # Performance validation
            "resilience_tests",    # Fault injection tests
        ]
        
        results = {}
        
        for category in test_categories:
            print(f"Running {category}...")
            
            try:
                category_results = await self._run_test_category(category)
                results[category] = category_results
                
                # Stop on critical failures
                if category == "smoke_tests" and category_results['failed'] > 0:
                    raise RuntimeError("Smoke tests failed - stopping CI run")
                    
            except Exception as e:
                results[category] = {'status': 'error', 'message': str(e)}
                if category in ["smoke_tests", "integration_tests"]:
                    raise  # Critical categories must pass
        
        return results
    
    async def _run_test_category(self, category: str) -> Dict[str, Any]:
        """Run specific test category."""
        if category == "smoke_tests":
            return await self._run_smoke_tests()
        elif category == "integration_tests":
            return await self._run_integration_tests()
        elif category == "performance_tests":
            return await self._run_performance_tests()
        elif category == "resilience_tests":
            return await self._run_resilience_tests()
        else:
            raise ValueError(f"Unknown test category: {category}")
    
    async def _run_smoke_tests(self) -> Dict[str, Any]:
        """Run fast smoke tests to validate basic functionality."""
        smoke_tests = [
            self._test_system_startup,
            self._test_basic_connectivity,
            self._test_component_health,
            self._test_minimal_pipeline,
        ]
        
        results = {'passed': 0, 'failed': 0, 'errors': []}
        
        for test_func in smoke_tests:
            try:
                await asyncio.wait_for(test_func(), timeout=30)  # 30s timeout per smoke test
                results['passed'] += 1
            except Exception as e:
                results['failed'] += 1
                results['errors'].append(f"{test_func.__name__}: {str(e)}")
        
        return results
    
    async def _test_system_startup(self):
        """Test system can start up successfully."""
        integration = MLIntegrationManager(
            auto_start_postgres=False,  # Use CI postgres service
            auto_migrate=True,
            ensure_healthy=True
        )
        integration.shutdown()
    
    async def _test_basic_connectivity(self):
        """Test basic database connectivity."""
        import os
        db_connection = os.getenv('DB_CONNECTION')
        
        from sqlalchemy import create_engine, text
        engine = create_engine(db_connection)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            assert result.fetchone()[0] == 1
    
    async def _test_component_health(self):
        """Test all components report healthy status."""
        integration = MLIntegrationManager(
            auto_start_postgres=False,
            auto_migrate=True,
            ensure_healthy=True
        )
        
        try:
            health = integration.check_health()
            assert health['postgres'], "PostgreSQL should be healthy"
            assert health['data_store'], "Data store should be healthy"
            assert health['feature_store'], "Feature store should be healthy"
        finally:
            integration.shutdown()
    
    async def _test_minimal_pipeline(self):
        """Test minimal pipeline execution."""
        integration = MLIntegrationManager(
            auto_start_postgres=False,
            auto_migrate=True
        )
        
        try:
            # Simple data write and read
            test_data = {'instrument_id': 'EUR/USD', 'close': 1.1000, 'ts_event': time.time_ns()}
            
            await integration.data_store.write_bar_data(
                bar_data=test_data,
                correlation_id=create_correlation_id(),
                ts_event=test_data['ts_event'],
                ts_init=time.time_ns()
            )
            
        finally:
            integration.shutdown()
```

This comprehensive integration testing strategy ensures that all ML components work together reliably across different environments, load conditions, and failure scenarios. It provides the foundation for confident deployment and operation of ML systems in Nautilus Trader.