#!/usr/bin/env python3
"""
Comprehensive test script to validate ML actor functionality and performance claims.

This script tests:
1. BaseMLInferenceActor instantiation and configuration
2. MLSignalActor with different strategies 
3. ONNX model loading and inference
4. Feature computation and performance
5. Hot path performance benchmarks
6. Store and registry integration
7. Circuit breaker and health monitoring
8. Hot reload functionality
"""

import os
import sys
import time
import tempfile
import numpy as np
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, patch
import traceback

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set environment variables for testing
os.environ["ML_TEST_ALLOW_NON_ONNX"] = "1"
os.environ["PYTEST_CURRENT_TEST"] = "test_ml_functionality"

print("🔍 Testing ML Actor Functionality")
print("=" * 50)

class TestResults:
    def __init__(self):
        self.tests_run = 0
        self.tests_passed = 0
        self.tests_failed = 0
        self.failures = []
        self.performance_results = {}
        
    def add_test(self, name: str, passed: bool, error: str = None, performance: dict = None):
        self.tests_run += 1
        if passed:
            self.tests_passed += 1
            print(f"✅ {name}")
        else:
            self.tests_failed += 1
            self.failures.append(f"{name}: {error}")
            print(f"❌ {name}: {error}")
        
        if performance:
            self.performance_results[name] = performance
    
    def summary(self):
        print("\n" + "=" * 50)
        print("TEST SUMMARY")
        print("=" * 50)
        print(f"Tests run: {self.tests_run}")
        print(f"Tests passed: {self.tests_passed}")
        print(f"Tests failed: {self.tests_failed}")
        print(f"Success rate: {(self.tests_passed/self.tests_run)*100:.1f}%")
        
        if self.failures:
            print("\nFAILURES:")
            for failure in self.failures:
                print(f"  - {failure}")
        
        if self.performance_results:
            print("\nPERFORMANCE RESULTS:")
            for test, results in self.performance_results.items():
                print(f"  {test}:")
                for metric, value in results.items():
                    print(f"    {metric}: {value}")

results = TestResults()

def test_imports():
    """Test that all ML actor imports work"""
    try:
        from ml.actors.base import BaseMLInferenceActor, MLSignal
        from ml.actors.signal import MLSignalActor, MLSignalActorConfig
        from ml.config.actors import OptimizationConfig, StrategyConfig
        from ml.config.base import MLActorConfig, MLFeatureConfig
        from nautilus_trader.model.identifiers import InstrumentId
        from nautilus_trader.model.data import BarType, Bar
        results.add_test("Import ML actors and config", True)
        return True
    except Exception as e:
        results.add_test("Import ML actors and config", False, str(e))
        return False

def test_basic_actor_instantiation():
    """Test basic actor instantiation without external dependencies"""
    try:
        from ml.actors.base import BaseMLInferenceActor
        from ml.config.base import MLActorConfig
        from nautilus_trader.model.data import BarType
        from nautilus_trader.model.identifiers import InstrumentId
        from nautilus_trader.common.component import ComponentId
        
        # Create a concrete implementation for testing
        class TestMLInferenceActor(BaseMLInferenceActor):
            def _load_model(self):
                self._model = Mock()
                
            def _initialize_features(self):
                pass
                
            def _compute_features(self, bar):
                return np.array([1.0, 2.0, 3.0], dtype=np.float32)
                
            def _predict(self, features):
                return 0.7, 0.8
        
        # Create minimal config
        config = MLActorConfig(
            component_id=ComponentId("test_actor"),
            model_path="/tmp/dummy_model.pkl",  # Non-existent is OK for this test
            bar_type=BarType.from_str("EUR/USD.SIM-1-MINUTE-BID-INTERNAL"),
            use_dummy_stores=True  # Use dummy stores to avoid DB dependency
        )
        
        # This should work without external dependencies
        actor = TestMLInferenceActor(config)
        
        # Test property accessors
        assert hasattr(actor, 'feature_store')
        assert hasattr(actor, 'model_store')
        assert hasattr(actor, 'strategy_store')
        assert hasattr(actor, 'data_store')
        
        results.add_test("BaseMLInferenceActor instantiation", True)
        return True
        
    except Exception as e:
        results.add_test("BaseMLInferenceActor instantiation", False, str(e))
        print(f"Stacktrace: {traceback.format_exc()}")
        return False

def test_signal_actor_instantiation():
    """Test MLSignalActor instantiation with different strategies"""
    try:
        from ml.actors.signal import MLSignalActor, MLSignalActorConfig, SignalStrategy
        from ml.config.actors import OptimizationConfig, StrategyConfig
        from nautilus_trader.model.data import BarType
        from nautilus_trader.model.identifiers import InstrumentId
        from nautilus_trader.common.component import ComponentId
        
        # Test each strategy
        strategies = ["threshold", "extremes", "momentum", "ensemble", "adaptive"]
        
        for strategy in strategies:
            try:
                config = MLSignalActorConfig(
                    component_id=ComponentId(f"test_signal_{strategy}"),
                    model_path="/tmp/dummy_model.pkl",
                    bar_type=BarType.from_str("EUR/USD.SIM-1-MINUTE-BID-INTERNAL"),
                    signal_strategy=strategy,
                    optimization_config=OptimizationConfig(level="standard"),
                    strategy_config=StrategyConfig(),
                    use_dummy_stores=True
                )
                
                # Mock the model loading to avoid file dependencies
                with patch.object(MLSignalActor, '_load_model_with_metadata'):
                    actor = MLSignalActor(config)
                    
                    # Test strategy creation worked
                    assert hasattr(actor, '_signal_strategy')
                    assert actor._signal_strategy is not None
                    
                results.add_test(f"MLSignalActor with {strategy} strategy", True)
                
            except Exception as e:
                results.add_test(f"MLSignalActor with {strategy} strategy", False, str(e))
        
        return True
        
    except Exception as e:
        results.add_test("MLSignalActor instantiation", False, str(e))
        print(f"Stacktrace: {traceback.format_exc()}")
        return False

def test_onnx_model_loading():
    """Test ONNX model loading with actual test models"""
    try:
        from ml.actors.base import ONNXModelLoader, ProductionModelLoader
        from ml._imports import HAS_ONNX
        
        if not HAS_ONNX:
            results.add_test("ONNX model loading", False, "ONNX Runtime not available")
            return False
            
        # Test with existing test models
        test_models = [
            "/home/nate/projects/nautilus_trader/ml/tests/data/model_registry_rollout/models/prod.onnx",
            "/home/nate/projects/nautilus_trader/ml/tests/data/model_registry_rollout/models/new.onnx"
        ]
        
        loader = ONNXModelLoader()
        
        for model_path in test_models:
            if Path(model_path).exists():
                try:
                    model, metadata = loader.load_model(model_path)
                    
                    # Verify model and metadata
                    assert model is not None
                    assert "type" in metadata
                    assert metadata["type"] == "onnx"
                    assert "input_names" in metadata
                    assert "output_names" in metadata
                    
                    results.add_test(f"Load ONNX model {Path(model_path).name}", True)
                    
                except Exception as e:
                    results.add_test(f"Load ONNX model {Path(model_path).name}", False, str(e))
            else:
                results.add_test(f"ONNX model {Path(model_path).name} not found", False, "File does not exist")
                
        return True
        
    except Exception as e:
        results.add_test("ONNX model loading", False, str(e))
        return False

def test_model_loading_formats():
    """Test different model loading formats"""
    try:
        from ml.actors.base import ProductionModelLoader
        
        loader = ProductionModelLoader()
        
        # Test pickle model rejection (security feature)
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            import pickle
            pickle.dump({'dummy': 'model'}, f)
            pickle_path = f.name
        
        try:
            loader.load_model(pickle_path)
            results.add_test("Pickle model security rejection", False, "Pickle loading should be rejected")
        except ValueError as e:
            if "not supported" in str(e):
                results.add_test("Pickle model security rejection", True)
            else:
                results.add_test("Pickle model security rejection", False, f"Wrong error: {e}")
        finally:
            os.unlink(pickle_path)
            
        return True
        
    except Exception as e:
        results.add_test("Model loading formats", False, str(e))
        return False

def create_dummy_onnx_model():
    """Create a simple ONNX model for testing"""
    try:
        from ml._imports import HAS_ONNX, onnx, ort
        import numpy as np
        
        if not HAS_ONNX:
            return None
        
        # Create a simple model programmatically
        import onnx.helper as helper
        import onnx.numpy_helper as numpy_helper
        from onnx import TensorProto
        
        # Create a simple linear model: y = x * 2 + 1
        X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 3])
        Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 1])
        confidence = helper.make_tensor_value_info('confidence', TensorProto.FLOAT, [1, 1])
        
        # Create weight and bias tensors
        W = numpy_helper.from_array(np.array([[0.5], [0.3], [0.2]], dtype=np.float32), name='W')
        B = numpy_helper.from_array(np.array([0.1], dtype=np.float32), name='B')
        
        # MatMul node
        matmul_node = helper.make_node('MatMul', ['X', 'W'], ['matmul_result'])
        
        # Add node
        add_node = helper.make_node('Add', ['matmul_result', 'B'], ['Y'])
        
        # Confidence node (constant)
        conf_node = helper.make_node('Constant', [], ['confidence'], 
                                   value=numpy_helper.from_array(np.array([[0.8]], dtype=np.float32)))
        
        # Create graph
        graph_def = helper.make_graph(
            [matmul_node, add_node, conf_node],
            'simple_model',
            [X],
            [Y, confidence],
            [W, B]
        )
        
        # Create model
        model_def = helper.make_model(graph_def, producer_name='test')
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            onnx.save(model_def, f.name)
            return f.name
            
    except Exception as e:
        print(f"Failed to create dummy ONNX model: {e}")
        return None

def test_feature_computation_performance():
    """Test feature computation and performance claims"""
    try:
        from ml.features.engineering import FeatureEngineer, FeatureConfig, IndicatorManager
        from nautilus_trader.model.data import Bar, BarType
        from nautilus_trader.model.identifiers import InstrumentId
        from nautilus_trader.core.datetime import unix_nanos_to_dt
        from decimal import Decimal
        import time
        
        # Create feature engineer
        config = FeatureConfig()
        engineer = FeatureEngineer(config)
        indicator_manager = IndicatorManager(config)
        
        # Create dummy bar data
        bar_type = BarType.from_str("EUR/USD.SIM-1-MINUTE-BID-INTERNAL")
        
        # Test performance with multiple bars
        feature_times = []
        num_bars = 100
        
        for i in range(num_bars):
            # Create a bar
            ts = time.time_ns()
            bar = Bar(
                bar_type=bar_type,
                open=Decimal('1.1000') + Decimal(str(i * 0.0001)),
                high=Decimal('1.1005') + Decimal(str(i * 0.0001)),
                low=Decimal('1.0995') + Decimal(str(i * 0.0001)),
                close=Decimal('1.1002') + Decimal(str(i * 0.0001)),
                volume=Decimal('1000'),
                ts_event=ts,
                ts_init=ts
            )
            
            # Time feature computation
            start_time = time.perf_counter()
            
            indicator_manager.update_from_bar(bar)
            
            if indicator_manager.all_initialized():
                current_bar = {
                    "close": float(bar.close),
                    "volume": float(bar.volume),
                    "high": float(bar.high),
                    "low": float(bar.low),
                }
                
                features = engineer.calculate_features_online(
                    current_bar=current_bar,
                    indicator_manager=indicator_manager,
                    scaler=None,
                )
                
                if features is not None and len(features) > 0:
                    feature_time = (time.perf_counter() - start_time) * 1000  # ms
                    feature_times.append(feature_time)
        
        if feature_times:
            avg_time = np.mean(feature_times)
            p99_time = np.percentile(feature_times, 99)
            
            # Check performance claims
            p99_under_500us = p99_time < 0.5  # 500 microseconds = 0.5ms
            
            performance = {
                "avg_feature_time_ms": f"{avg_time:.3f}",
                "p99_feature_time_ms": f"{p99_time:.3f}",
                "meets_500us_claim": p99_under_500us,
                "num_features": len(features) if features is not None else 0
            }
            
            results.add_test("Feature computation performance", True, 
                           performance=performance)
        else:
            results.add_test("Feature computation performance", False, 
                           "No feature times recorded")
        
        return True
        
    except Exception as e:
        results.add_test("Feature computation performance", False, str(e))
        print(f"Stacktrace: {traceback.format_exc()}")
        return False

def test_inference_performance():
    """Test inference performance with ONNX model"""
    try:
        dummy_model_path = create_dummy_onnx_model()
        if not dummy_model_path:
            results.add_test("Inference performance", False, "Could not create test ONNX model")
            return False
        
        from ml.actors.base import ONNXModelLoader
        
        loader = ONNXModelLoader()
        model, metadata = loader.load_model(dummy_model_path)
        
        # Test inference times
        inference_times = []
        num_tests = 1000
        
        for _ in range(num_tests):
            # Create random features
            features = np.random.randn(3).astype(np.float32).reshape(1, -1)
            
            start_time = time.perf_counter()
            
            # Run inference
            input_name = metadata["input_names"][0]
            outputs = model.run(metadata["output_names"], {input_name: features})
            
            inference_time = (time.perf_counter() - start_time) * 1000  # ms
            inference_times.append(inference_time)
        
        # Calculate statistics
        avg_time = np.mean(inference_times)
        p99_time = np.percentile(inference_times, 99)
        
        # Check performance claims
        p99_under_2ms = p99_time < 2.0
        
        performance = {
            "avg_inference_time_ms": f"{avg_time:.3f}",
            "p99_inference_time_ms": f"{p99_time:.3f}",
            "meets_2ms_claim": p99_under_2ms,
            "num_inferences": num_tests
        }
        
        results.add_test("ONNX inference performance", True, performance=performance)
        
        # Clean up
        os.unlink(dummy_model_path)
        
        return True
        
    except Exception as e:
        results.add_test("Inference performance", False, str(e))
        return False

def test_end_to_end_performance():
    """Test end-to-end signal generation performance"""
    try:
        from ml.actors.signal import MLSignalActor, MLSignalActorConfig
        from nautilus_trader.model.data import Bar, BarType
        from nautilus_trader.common.component import ComponentId
        from decimal import Decimal
        import time
        
        # Create dummy ONNX model
        dummy_model_path = create_dummy_onnx_model()
        if not dummy_model_path:
            results.add_test("End-to-end performance", False, "Could not create test model")
            return False
        
        try:
            config = MLSignalActorConfig(
                component_id=ComponentId("perf_test_actor"),
                model_path=dummy_model_path,
                bar_type=BarType.from_str("EUR/USD.SIM-1-MINUTE-BID-INTERNAL"),
                signal_strategy="threshold",
                use_dummy_stores=True
            )
            
            actor = MLSignalActor(config)
            
            # Mock the necessary methods to avoid full initialization
            with patch.object(actor, '_load_model_with_metadata'):
                with patch.object(actor, 'subscribe_bars'):
                    with patch.object(actor, 'log'):
                        # Initialize manually
                        actor._model, actor._model_metadata = actor._model_loader.load_model(dummy_model_path)
                        actor._initialize_features()
                        
                        # Create test bars and time end-to-end processing
                        processing_times = []
                        num_bars = 100
                        
                        for i in range(num_bars):
                            ts = time.time_ns()
                            bar = Bar(
                                bar_type=config.bar_type,
                                open=Decimal('1.1000'),
                                high=Decimal('1.1005'),
                                low=Decimal('1.0995'),
                                close=Decimal('1.1002'),
                                volume=Decimal('1000'),
                                ts_event=ts,
                                ts_init=ts
                            )
                            
                            start_time = time.perf_counter()
                            
                            # Process bar (this calls the hot path)
                            with patch.object(actor, 'publish_data'):  # Mock publish
                                actor.on_bar(bar)
                            
                            processing_time = (time.perf_counter() - start_time) * 1000  # ms
                            processing_times.append(processing_time)
                        
                        # Calculate statistics
                        if processing_times:
                            avg_time = np.mean(processing_times)
                            p99_time = np.percentile(processing_times, 99)
                            
                            # Check 5ms claim
                            p99_under_5ms = p99_time < 5.0
                            
                            performance = {
                                "avg_processing_time_ms": f"{avg_time:.3f}",
                                "p99_processing_time_ms": f"{p99_time:.3f}",
                                "meets_5ms_claim": p99_under_5ms,
                                "num_bars_processed": len(processing_times)
                            }
                            
                            results.add_test("End-to-end performance", True, performance=performance)
                        else:
                            results.add_test("End-to-end performance", False, "No processing times recorded")
            
        finally:
            # Clean up
            if os.path.exists(dummy_model_path):
                os.unlink(dummy_model_path)
        
        return True
        
    except Exception as e:
        results.add_test("End-to-end performance", False, str(e))
        print(f"Stacktrace: {traceback.format_exc()}")
        return False

def test_health_monitoring():
    """Test health monitoring and circuit breaker functionality"""
    try:
        from ml.actors.base import HealthMonitor, CircuitBreaker
        from ml.config.base import HealthMonitorConfig, CircuitBreakerConfig
        
        # Test health monitor
        health_monitor = HealthMonitor()
        
        # Test initial state
        assert health_monitor.status.value == "healthy"
        assert health_monitor.get_success_rate() == 1.0
        
        # Test prediction success
        health_monitor.update_prediction_success()
        assert health_monitor.total_predictions == 1
        assert health_monitor.failed_predictions == 0
        
        # Test prediction failure
        health_monitor.update_prediction_failure()
        assert health_monitor.total_predictions == 2
        assert health_monitor.failed_predictions == 1
        assert health_monitor.get_success_rate() == 0.5
        
        # Test health status export
        status = health_monitor.to_dict()
        assert "status" in status
        assert "success_rate" in status
        
        results.add_test("Health monitoring", True)
        
        # Test circuit breaker
        circuit_breaker = CircuitBreaker()
        
        # Test initial state
        assert circuit_breaker.state.value == "closed"
        assert circuit_breaker.can_execute() == True
        
        # Test failure recording
        circuit_breaker.record_failure()
        stats = circuit_breaker.get_stats()
        assert "failure_count" in stats
        
        results.add_test("Circuit breaker", True)
        
        return True
        
    except Exception as e:
        results.add_test("Health monitoring and circuit breaker", False, str(e))
        return False

def test_stores_integration():
    """Test stores and registries integration"""
    try:
        from ml.stores.base import DummyStore
        from ml.registry.base import DummyRegistry
        
        # Test dummy store (used in testing)
        store = DummyStore()
        
        # Test basic store operations
        store.write_features("test_set", "EUR/USD", {"feature1": 1.0}, 123456789, 123456789)
        store.write_prediction("test_model", "EUR/USD", 0.7, 0.8, {"f1": 1.0}, 1.5, 123456789)
        store.flush()
        
        results.add_test("Dummy store operations", True)
        
        # Test dummy registry
        registry = DummyRegistry()
        
        # Test basic registry operations
        result = registry.list_items()  # Should not fail
        
        results.add_test("Dummy registry operations", True)
        
        return True
        
    except Exception as e:
        results.add_test("Stores integration", False, str(e))
        return False

def test_missing_dependencies():
    """Check for missing ML dependencies"""
    try:
        from ml._imports import (
            HAS_ONNX, HAS_XGBOOST, HAS_SKLEARN, HAS_PANDAS, HAS_POLARS,
            check_ml_dependencies
        )
        
        dependencies = {
            "ONNX Runtime": HAS_ONNX,
            "XGBoost": HAS_XGBOOST,
            "Scikit-learn": HAS_SKLEARN,
            "Pandas": HAS_PANDAS,
            "Polars": HAS_POLARS,
        }
        
        missing = []
        for name, available in dependencies.items():
            if not available:
                missing.append(name)
        
        if missing:
            results.add_test("ML dependencies check", False, f"Missing: {', '.join(missing)}")
        else:
            results.add_test("ML dependencies check", True)
        
        return len(missing) == 0
        
    except Exception as e:
        results.add_test("ML dependencies check", False, str(e))
        return False

def main():
    """Run all tests"""
    print("Starting ML Actor Functionality Tests...")
    print()
    
    # Run tests in order
    test_functions = [
        test_imports,
        test_missing_dependencies,
        test_basic_actor_instantiation,
        test_signal_actor_instantiation,
        test_onnx_model_loading,
        test_model_loading_formats,
        test_feature_computation_performance,
        test_inference_performance,
        test_end_to_end_performance,
        test_health_monitoring,
        test_stores_integration,
    ]
    
    for test_func in test_functions:
        try:
            test_func()
        except Exception as e:
            results.add_test(test_func.__name__, False, f"Test crashed: {e}")
            print(f"Test {test_func.__name__} crashed: {traceback.format_exc()}")
    
    # Update todo status
    from ml.actors.base import BaseMLInferenceActor
    
    # Print results
    results.summary()

if __name__ == "__main__":
    main()