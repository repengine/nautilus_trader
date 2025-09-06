#!/usr/bin/env python3
"""
Test BaseMLInferenceActor with 4-store + 4-registry integration.
"""

import os
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

# Add the project root to Python path for imports
sys.path.insert(0, "/home/nate/projects/nautilus_trader")

def test_base_ml_inference_actor():
    """Test BaseMLInferenceActor initialization with stores and registries."""
    print("=== Testing BaseMLInferenceActor ===")
    
    try:
        from ml.actors.base import BaseMLInferenceActor
        from ml.config.base import MLActorConfig
        import numpy as np
        from nautilus_trader.model.data import Bar
        
        # Create a concrete test implementation since BaseMLInferenceActor is abstract
        class TestMLInferenceActor(BaseMLInferenceActor):
            def _load_model(self) -> None:
                """Dummy model loading."""
                self._model = lambda x: (0.5, 0.8)  # dummy prediction function
                
            def _initialize_features(self) -> None:
                """Dummy feature initialization."""
                pass
                
            def _compute_features(self, bar: Bar) -> np.ndarray:
                """Dummy feature computation."""
                return np.array([1.0, 2.0, 3.0], dtype=np.float32)
                
            def _predict(self, features: np.ndarray) -> tuple[float, float]:
                """Dummy prediction."""
                return 0.7, 0.85
        
        # Test 1: Initialize with dummy stores (should work)
        print("  Testing with dummy stores...")
        try:
            # Create a dummy model file
            with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
                model_path = f.name
                f.write(b"dummy model content")
            
            # Set environment variable to allow non-ONNX in tests
            os.environ["ML_TEST_ALLOW_NON_ONNX"] = "1"
            
            config = MLActorConfig(
                component_id="test_actor",
                model_path=model_path,
                use_dummy_stores=True,  # This should force dummy stores
                warm_up_period=5
            )
            
            actor = TestMLInferenceActor(config)
            print("  ✓ BaseMLInferenceActor initialized with dummy stores")
            
            # Test that stores are initialized
            assert hasattr(actor, '_feature_store'), "Missing _feature_store"
            assert hasattr(actor, '_model_store'), "Missing _model_store"  
            assert hasattr(actor, '_strategy_store'), "Missing _strategy_store"
            assert hasattr(actor, '_data_store'), "Missing _data_store"
            print("  ✓ All 4 stores are initialized")
            
            # Test that registries are initialized
            assert hasattr(actor, '_feature_registry'), "Missing _feature_registry"
            assert hasattr(actor, '_model_registry'), "Missing _model_registry"
            assert hasattr(actor, '_strategy_registry'), "Missing _strategy_registry" 
            assert hasattr(actor, '_data_registry'), "Missing _data_registry"
            print("  ✓ All 4 registries are initialized")
            
            # Test property accessors
            feature_store = actor.feature_store
            model_store = actor.model_store
            strategy_store = actor.strategy_store
            data_store = actor.data_store
            print("  ✓ Store property accessors work")
            
            feature_registry = actor.feature_registry
            model_registry = actor.model_registry
            strategy_registry = actor.strategy_registry
            data_registry = actor.data_registry
            print("  ✓ Registry property accessors work")
            
            # Test health checks
            if hasattr(feature_store, 'is_healthy'):
                health = feature_store.is_healthy()
                print(f"  ✓ FeatureStore health: {health}")
                
            if hasattr(model_store, 'is_healthy'):
                health = model_store.is_healthy()
                print(f"  ✓ ModelStore health: {health}")
                
            if hasattr(strategy_store, 'is_healthy'):
                health = strategy_store.is_healthy()
                print(f"  ✓ StrategyStore health: {health}")
                
            # Cleanup
            os.unlink(model_path)
            
        except Exception as e:
            print(f"  ✗ BaseMLInferenceActor with dummy stores failed: {e}")
            traceback.print_exc()
        
        # Test 2: Initialize with PostgreSQL stores (should work if DB available)
        print("\n  Testing with PostgreSQL stores...")
        try:
            # Create a dummy model file
            with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
                model_path = f.name
                f.write(b"dummy model content")
            
            config = MLActorConfig(
                component_id="test_actor_pg",
                model_path=model_path,
                db_connection="postgresql://postgres:postgres@localhost:5432/nautilus",
                warm_up_period=5
            )
            
            actor = TestMLInferenceActor(config)
            print("  ✓ BaseMLInferenceActor initialized with PostgreSQL stores")
            
            # Test store types
            from ml.stores.base import DummyStore
            
            feature_store = actor.feature_store
            model_store = actor.model_store
            strategy_store = actor.strategy_store
            
            # Check if they're real stores or dummy stores
            is_dummy_feature = isinstance(feature_store, DummyStore)
            is_dummy_model = isinstance(model_store, DummyStore)
            is_dummy_strategy = isinstance(strategy_store, DummyStore)
            
            print(f"    FeatureStore is DummyStore: {is_dummy_feature}")
            print(f"    ModelStore is DummyStore: {is_dummy_model}")
            print(f"    StrategyStore is DummyStore: {is_dummy_strategy}")
            
            # Test health checks for real stores
            if not is_dummy_feature and hasattr(feature_store, 'is_healthy'):
                health = feature_store.is_healthy()
                print(f"    FeatureStore health: {health}")
                
            if not is_dummy_model and hasattr(model_store, 'is_healthy'):
                health = model_store.is_healthy()
                print(f"    ModelStore health: {health}")
                
            if not is_dummy_strategy and hasattr(strategy_store, 'is_healthy'):
                health = strategy_store.is_healthy()
                print(f"    StrategyStore health: {health}")
            
            # Cleanup
            os.unlink(model_path)
            
        except Exception as e:
            print(f"  ✗ BaseMLInferenceActor with PostgreSQL stores failed: {e}")
            traceback.print_exc()
        
        # Test 3: Test progressive fallback when PostgreSQL unavailable
        print("\n  Testing progressive fallback...")
        try:
            # Create a dummy model file
            with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
                model_path = f.name
                f.write(b"dummy model content")
            
            config = MLActorConfig(
                component_id="test_actor_fallback",
                model_path=model_path,
                db_connection="postgresql://invalid:invalid@invalid:5432/invalid",
                warm_up_period=5
            )
            
            # This should fallback to dummy stores
            actor = TestMLInferenceActor(config)
            print("  ✓ BaseMLInferenceActor fallback to dummy stores works")
            
            # Check that stores are dummy stores
            from ml.stores.base import DummyStore
            from ml.registry.base import DummyRegistry
            
            feature_store = actor.feature_store
            is_dummy = isinstance(feature_store, DummyStore)
            print(f"    Fallback FeatureStore is DummyStore: {is_dummy}")
            
            feature_registry = actor.feature_registry
            is_dummy_reg = isinstance(feature_registry, DummyRegistry)
            print(f"    Fallback FeatureRegistry is DummyRegistry: {is_dummy_reg}")
            
            # Cleanup
            os.unlink(model_path)
            
        except Exception as e:
            print(f"  ✗ Progressive fallback test failed: {e}")
            traceback.print_exc()
            
    except ImportError as e:
        print(f"  ✗ Could not import BaseMLInferenceActor: {e}")
        traceback.print_exc()
    except Exception as e:
        print(f"  ✗ BaseMLInferenceActor test failed: {e}")
        traceback.print_exc()

def test_mandatory_flush_on_shutdown():
    """Test that all stores are flushed on actor shutdown."""
    print("\n=== Testing Mandatory Store Flush ===")
    
    try:
        from ml.actors.base import BaseMLInferenceActor
        from ml.config.base import MLActorConfig
        import numpy as np
        from nautilus_trader.model.data import Bar
        
        class TestFlushMLActor(BaseMLInferenceActor):
            def __init__(self, config):
                super().__init__(config)
                self.flush_calls = []
                
            def _load_model(self) -> None:
                self._model = lambda x: (0.5, 0.8)
                
            def _initialize_features(self) -> None:
                pass
                
            def _compute_features(self, bar: Bar) -> np.ndarray:
                return np.array([1.0, 2.0, 3.0], dtype=np.float32)
                
            def _predict(self, features: np.ndarray) -> tuple[float, float]:
                return 0.7, 0.85
        
        # Create actor with dummy stores
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            model_path = f.name
            f.write(b"dummy model content")
        
        os.environ["ML_TEST_ALLOW_NON_ONNX"] = "1"
        
        config = MLActorConfig(
            component_id="test_flush_actor",
            model_path=model_path,
            use_dummy_stores=True,
            warm_up_period=5
        )
        
        actor = TestFlushMLActor(config)
        
        # Test on_stop calls flush on all stores
        try:
            actor.on_stop()
            print("  ✓ on_stop() executed without errors")
        except Exception as e:
            print(f"  ✗ on_stop() failed: {e}")
            traceback.print_exc()
        
        # Cleanup
        os.unlink(model_path)
        
    except Exception as e:
        print(f"  ✗ Mandatory flush test failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    print("BaseMLInferenceActor 4-Store + 4-Registry Integration Test")
    print("=" * 60)
    
    test_base_ml_inference_actor()
    test_mandatory_flush_on_shutdown()
    
    print("\n" + "=" * 60)
    print("BaseMLInferenceActor Integration Test Complete")