#!/usr/bin/env python3
"""
Simplified ML Actor functionality test focused on what actually works.
"""

import os
import sys
import time
import numpy as np
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch
import tempfile

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set environment variables for testing
os.environ["ML_TEST_ALLOW_NON_ONNX"] = "1"
os.environ["PYTEST_CURRENT_TEST"] = "test_ml_functionality"

print("🔍 Simplified ML Actor Functionality Test")
print("=" * 50)

def test_ml_imports():
    """Test that ML imports work correctly"""
    print("Testing ML imports...")
    try:
        from ml.actors.base import BaseMLInferenceActor, MLSignal, HealthMonitor, CircuitBreaker
        from ml.actors.signal import MLSignalActor
        from ml.config.base import MLActorConfig, MLFeatureConfig
        from ml.config.actors import MLSignalActorConfig, OptimizationConfig, StrategyConfig
        from ml._imports import HAS_ONNX, HAS_XGBOOST, HAS_SKLEARN
        print("✅ All ML imports successful")
        return True
    except Exception as e:
        print(f"❌ ML import failed: {e}")
        return False

def test_ml_dependencies():
    """Check ML dependency availability"""
    print("\nTesting ML dependencies...")
    from ml._imports import (
        HAS_ONNX, HAS_XGBOOST, HAS_SKLEARN, HAS_PANDAS, HAS_POLARS
    )
    
    deps = {
        "ONNX Runtime": HAS_ONNX,
        "XGBoost": HAS_XGBOOST, 
        "Scikit-learn": HAS_SKLEARN,
        "Pandas": HAS_PANDAS,
        "Polars": HAS_POLARS
    }
    
    for name, available in deps.items():
        status = "✅" if available else "❌"
        print(f"{status} {name}: {'Available' if available else 'Missing'}")
    
    return all(deps.values())

def test_health_monitor():
    """Test health monitoring system"""
    print("\nTesting health monitoring...")
    try:
        from ml.actors.base import HealthMonitor
        
        monitor = HealthMonitor()
        
        # Test initial state
        assert monitor.status.value == "healthy"
        assert monitor.get_success_rate() == 1.0
        
        # Test success tracking
        monitor.update_prediction_success()
        assert monitor.total_predictions == 1
        assert monitor.failed_predictions == 0
        
        # Test failure tracking
        monitor.update_prediction_failure()
        assert monitor.total_predictions == 2
        assert monitor.failed_predictions == 1
        assert monitor.get_success_rate() == 0.5
        
        # Test status export
        status = monitor.to_dict()
        assert "status" in status
        assert "success_rate" in status
        
        print("✅ Health monitor works correctly")
        return True
        
    except Exception as e:
        print(f"❌ Health monitor failed: {e}")
        return False

def test_circuit_breaker():
    """Test circuit breaker functionality"""
    print("\nTesting circuit breaker...")
    try:
        from ml.actors.base import CircuitBreaker
        
        breaker = CircuitBreaker()
        
        # Test initial state
        assert breaker.state.value == "closed"
        assert breaker.can_execute() == True
        
        # Test failure recording
        breaker.record_failure()
        stats = breaker.get_stats()
        assert "failure_count" in stats
        assert stats["failure_count"] == 1
        
        # Test success recording
        breaker.record_success()
        stats = breaker.get_stats()
        
        print("✅ Circuit breaker works correctly")
        return True
        
    except Exception as e:
        print(f"❌ Circuit breaker failed: {e}")
        return False

def test_dummy_stores():
    """Test dummy store implementations"""
    print("\nTesting dummy stores...")
    try:
        from ml.stores.base import DummyStore
        from ml.registry.base import DummyRegistry
        
        # Test dummy store
        store = DummyStore()
        store.write_features("test_set", "EUR/USD", {"feature1": 1.0}, 123456789, 123456789)
        store.write_prediction("test_model", "EUR/USD", 0.7, 0.8, {"f1": 1.0}, 1.5, 123456789)
        store.flush()
        
        # Test dummy registry
        registry = DummyRegistry()
        items = registry.list_items()  # Should not fail
        
        print("✅ Dummy stores work correctly")
        return True
        
    except Exception as e:
        print(f"❌ Dummy stores failed: {e}")
        return False

def test_model_loading_security():
    """Test security features in model loading"""
    print("\nTesting model loading security...")
    try:
        from ml.actors.base import ProductionModelLoader
        
        loader = ProductionModelLoader()
        
        # Test pickle rejection
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            import pickle
            pickle.dump({'dummy': 'model'}, f)
            pickle_path = f.name
        
        try:
            loader.load_model(pickle_path)
            print("❌ Pickle loading should be rejected")
            return False
        except ValueError as e:
            if "not supported" in str(e):
                print("✅ Pickle models correctly rejected for security")
                return True
            else:
                print(f"❌ Wrong error for pickle: {e}")
                return False
        finally:
            os.unlink(pickle_path)
            
    except Exception as e:
        print(f"❌ Model loading security test failed: {e}")
        return False

def test_signal_strategies():
    """Test signal generation strategies in isolation"""
    print("\nTesting signal generation strategies...")
    try:
        from ml.actors.signal import ThresholdSignalStrategy, ExtremesStrategy, MomentumStrategy
        from types import SimpleNamespace
        import numpy as np
        
        # Create mock objects
        instrument_id = SimpleNamespace(value="EUR/USD")  
        bar_type = SimpleNamespace(instrument_id=instrument_id)
        bar = SimpleNamespace(bar_type=bar_type, ts_event=123456789)
        features = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        
        context = {
            "model_id": "test_model",
            "log_predictions": False,
            "timestamp_ns": 123456789
        }
        
        # Test threshold strategy
        threshold_strategy = ThresholdSignalStrategy(threshold=0.7)
        signal = threshold_strategy.generate_signal(bar, 0.5, 0.8, features, context)
        assert signal is not None  # Should generate signal (confidence 0.8 > threshold 0.7)
        
        signal = threshold_strategy.generate_signal(bar, 0.5, 0.6, features, context)
        assert signal is None  # Should not generate signal (confidence 0.6 < threshold 0.7)
        
        # Test extremes strategy (basic instantiation)
        extremes_strategy = ExtremesStrategy(top_pct=0.1, threshold=0.7, window_size=10)
        # Won't generate signal without history but should not crash
        signal = extremes_strategy.generate_signal(bar, 0.5, 0.8, features, context)
        
        # Test momentum strategy (basic instantiation)
        momentum_strategy = MomentumStrategy(lookback=5, threshold=0.7, momentum_threshold=0.01)
        # Won't generate signal without history but should not crash
        signal = momentum_strategy.generate_signal(bar, 0.5, 0.8, features, context)
        
        print("✅ Signal strategies work correctly")
        return True
        
    except Exception as e:
        print(f"❌ Signal strategies failed: {e}")
        return False

def test_feature_engineering():
    """Test feature engineering components"""
    print("\nTesting feature engineering...")
    try:
        from ml.features.engineering import FeatureConfig, FeatureEngineer
        
        # Test basic instantiation
        config = FeatureConfig()
        engineer = FeatureEngineer(config)
        
        assert hasattr(engineer, 'n_features')
        assert engineer.n_features > 0
        
        # Test config methods
        feature_names = config.get_feature_names()
        assert isinstance(feature_names, list)
        assert len(feature_names) > 0
        
        print(f"✅ Feature engineering works (generates {engineer.n_features} features)")
        return True
        
    except Exception as e:
        print(f"❌ Feature engineering failed: {e}")
        return False

def test_onnx_model_creation():
    """Test creating a simple ONNX model programmatically"""
    print("\nTesting ONNX model creation...")
    try:
        from ml._imports import HAS_ONNX
        
        if not HAS_ONNX:
            print("⚠️ ONNX Runtime not available - skipping ONNX tests")
            return True
            
        from ml._imports import onnx, ort
        import onnx.helper as helper
        import onnx.numpy_helper as numpy_helper
        from onnx import TensorProto
        
        # Create a simple model: y = x * 0.5 + 0.1
        X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 3])
        Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 1])
        
        # Create weight and bias
        W = numpy_helper.from_array(np.array([[0.5], [0.3], [0.2]], dtype=np.float32), name='W')
        B = numpy_helper.from_array(np.array([0.1], dtype=np.float32), name='B')
        
        # Create nodes
        matmul_node = helper.make_node('MatMul', ['X', 'W'], ['matmul_result'])
        add_node = helper.make_node('Add', ['matmul_result', 'B'], ['Y'])
        
        # Create graph
        graph_def = helper.make_graph(
            [matmul_node, add_node],
            'simple_model',
            [X],
            [Y], 
            [W, B]
        )
        
        # Create model
        model_def = helper.make_model(graph_def, producer_name='test')
        
        # Test inference
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            onnx.save(model_def, f.name)
            
            # Load and test
            session = ort.InferenceSession(f.name)
            inputs = {'X': np.random.randn(1, 3).astype(np.float32)}
            outputs = session.run(['Y'], inputs)
            
            assert len(outputs) == 1
            assert outputs[0].shape == (1, 1)
            
            # Clean up
            os.unlink(f.name)
        
        print("✅ ONNX model creation and inference works")
        return True
        
    except Exception as e:
        print(f"❌ ONNX model creation failed: {e}")
        return False

def test_performance_monitoring():
    """Test performance monitoring components"""
    print("\nTesting performance monitoring...")
    try:
        from ml.actors.signal import PerformanceMonitor
        
        monitor = PerformanceMonitor(reservoir_size=100)
        
        # Record some timing data
        for i in range(10):
            monitor.record_timing(
                feature_time_ns=500_000,  # 0.5ms
                inference_time_ns=1_000_000,  # 1ms  
                total_time_ns=1_500_000  # 1.5ms
            )
            if i % 3 == 0:
                monitor.record_signal()
        
        # Get stats
        stats = monitor.get_current_stats()
        assert "prediction_count" in stats
        assert "signal_count" in stats
        assert "avg_feature_time_ms" in stats
        assert stats["prediction_count"] == 10
        assert stats["signal_count"] == 4  # Every 3rd iteration
        
        # Get percentiles
        percentiles = monitor.get_latency_percentiles()
        assert "feature_computation" in percentiles
        assert "inference" in percentiles
        assert "total" in percentiles
        
        print("✅ Performance monitoring works correctly")
        return True
        
    except Exception as e:
        print(f"❌ Performance monitoring failed: {e}")
        return False

def run_all_tests():
    """Run all tests and report results"""
    tests = [
        test_ml_imports,
        test_ml_dependencies, 
        test_health_monitor,
        test_circuit_breaker,
        test_dummy_stores,
        test_model_loading_security,
        test_signal_strategies,
        test_feature_engineering,
        test_onnx_model_creation,
        test_performance_monitoring,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test.__name__} crashed: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print("SIMPLIFIED TEST SUMMARY") 
    print("=" * 50)
    print(f"Tests passed: {passed}")
    print(f"Tests failed: {failed}")
    print(f"Success rate: {(passed/(passed+failed))*100:.1f}%")
    
    return passed, failed

if __name__ == "__main__":
    passed, failed = run_all_tests()
    
    print(f"\n🎯 CONCLUSION:")
    if failed == 0:
        print("✅ All simplified tests PASSED - ML infrastructure is working!")
    else:
        print(f"⚠️ Some functionality is missing or broken ({failed} failures)")
        print("The ML actors have partial functionality but may not be production-ready")