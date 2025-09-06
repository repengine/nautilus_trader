#!/usr/bin/env python3
"""
Comprehensive test of the ML stores and registries functionality.

This script tests the "mandatory 4-store + 4-registry integration" claims by:
1. Testing each store (FeatureStore, ModelStore, StrategyStore, DataStore) initialization
2. Testing each registry (FeatureRegistry, ModelRegistry, StrategyRegistry, DataRegistry) initialization  
3. Testing CRUD operations on each store
4. Testing progressive fallback to DummyStore when PostgreSQL is unavailable
5. Testing BaseMLInferenceActor integration
6. Testing database schema and migrations

"""

import os
import sys
import traceback
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add the project root to Python path for imports
sys.path.insert(0, "/home/nate/projects/nautilus_trader")

def test_feature_store():
    """Test FeatureStore initialization and basic operations."""
    print("\n=== Testing FeatureStore ===")
    
    try:
        from ml.stores.feature_store import FeatureStore
        
        # Test 1: Initialize with valid PostgreSQL connection
        try:
            connection_string = "postgresql://postgres:postgres@localhost:5432/nautilus"
            print(f"  Testing with PostgreSQL connection: {connection_string}")
            store = FeatureStore(connection_string=connection_string)
            print("  ✓ FeatureStore initialized with PostgreSQL")
            
            # Test basic health check
            if hasattr(store, 'is_healthy'):
                health = store.is_healthy()
                print(f"    Health check: {'✓' if health else '✗'} {health}")
            
        except Exception as e:
            print(f"  ✗ FeatureStore PostgreSQL init failed: {e}")
        
        # Test 2: Initialize with invalid connection (should fail gracefully)
        try:
            bad_connection = "postgresql://invalid:invalid@invalid:5432/invalid"
            print(f"  Testing with invalid connection: {bad_connection}")
            store = FeatureStore(connection_string=bad_connection)
            print("  ✗ FeatureStore should have failed with bad connection")
        except Exception as e:
            print(f"  ✓ FeatureStore correctly failed with bad connection: {type(e).__name__}")
        
        # Test 3: Test write operations (if connection works)
        try:
            connection_string = "postgresql://postgres:postgres@localhost:5432/nautilus"
            store = FeatureStore(connection_string=connection_string)
            
            # Test write_features
            test_features = {
                "sma_10": 1.5,
                "rsi_14": 0.6,
                "volume_ratio": 1.2
            }
            
            ts_event = int(time.time() * 1e9)  # nanoseconds
            store.write_features(
                feature_set_id="test_set",
                instrument_id="EURUSD.IDEALPRO",
                features=test_features,
                ts_event=ts_event,
                ts_init=ts_event
            )
            print("  ✓ write_features() succeeded")
            
            # Test flush
            store.flush()
            print("  ✓ flush() succeeded")
            
        except Exception as e:
            print(f"  ✗ FeatureStore operations failed: {e}")
            traceback.print_exc()
        
    except ImportError as e:
        print(f"  ✗ Could not import FeatureStore: {e}")
    except Exception as e:
        print(f"  ✗ FeatureStore test failed: {e}")
        traceback.print_exc()

def test_model_store():
    """Test ModelStore initialization and basic operations."""
    print("\n=== Testing ModelStore ===")
    
    try:
        from ml.stores.model_store import ModelStore
        from ml.registry.persistence import PersistenceConfig, BackendType
        
        # Test 1: Initialize with persistence config
        try:
            persistence_config = PersistenceConfig(
                backend=BackendType.POSTGRES,
                connection_string="postgresql://postgres:postgres@localhost:5432/nautilus"
            )
            store = ModelStore(persistence_config=persistence_config)
            print("  ✓ ModelStore initialized with PersistenceConfig")
            
            # Test health check
            if hasattr(store, 'is_healthy'):
                health = store.is_healthy()
                print(f"    Health check: {'✓' if health else '✗'} {health}")
                
        except Exception as e:
            print(f"  ✗ ModelStore PostgreSQL init failed: {e}")
        
        # Test 2: Test write operations
        try:
            persistence_config = PersistenceConfig(
                backend=BackendType.POSTGRES,
                connection_string="postgresql://postgres:postgres@localhost:5432/nautilus"
            )
            store = ModelStore(persistence_config=persistence_config)
            
            # Test write_prediction
            ts_event = int(time.time() * 1e9)
            store.write_prediction(
                model_id="test_model_v1",
                instrument_id="EURUSD.IDEALPRO",
                prediction=0.85,
                confidence=0.92,
                features={"sma_10": 1.5, "rsi_14": 0.6},
                inference_time_ms=2.5,
                ts_event=ts_event
            )
            print("  ✓ write_prediction() succeeded")
            
            # Test flush
            store.flush()
            print("  ✓ flush() succeeded")
            
        except Exception as e:
            print(f"  ✗ ModelStore operations failed: {e}")
            traceback.print_exc()
        
    except ImportError as e:
        print(f"  ✗ Could not import ModelStore: {e}")
    except Exception as e:
        print(f"  ✗ ModelStore test failed: {e}")
        traceback.print_exc()

def test_strategy_store():
    """Test StrategyStore initialization and basic operations."""
    print("\n=== Testing StrategyStore ===")
    
    try:
        from ml.stores.strategy_store import StrategyStore
        from ml.registry.persistence import PersistenceConfig, BackendType
        
        # Test 1: Initialize with persistence config
        try:
            persistence_config = PersistenceConfig(
                backend=BackendType.POSTGRES,
                connection_string="postgresql://postgres:postgres@localhost:5432/nautilus"
            )
            store = StrategyStore(persistence_config=persistence_config)
            print("  ✓ StrategyStore initialized with PersistenceConfig")
            
            # Test health check
            if hasattr(store, 'is_healthy'):
                health = store.is_healthy()
                print(f"    Health check: {'✓' if health else '✗'} {health}")
                
        except Exception as e:
            print(f"  ✗ StrategyStore PostgreSQL init failed: {e}")
        
        # Test 2: Test write operations
        try:
            persistence_config = PersistenceConfig(
                backend=BackendType.POSTGRES,
                connection_string="postgresql://postgres:postgres@localhost:5432/nautilus"
            )
            store = StrategyStore(persistence_config=persistence_config)
            
            # Test write_signal
            ts_event = int(time.time() * 1e9)
            store.write_signal(
                strategy_id="test_strategy_v1",
                instrument_id="EURUSD.IDEALPRO",
                signal_type="BUY",
                strength=0.8,
                confidence=0.9,
                risk_score=0.3,
                ts_event=ts_event
            )
            print("  ✓ write_signal() succeeded")
            
            # Test flush
            store.flush()
            print("  ✓ flush() succeeded")
            
        except Exception as e:
            print(f"  ✗ StrategyStore operations failed: {e}")
            traceback.print_exc()
        
    except ImportError as e:
        print(f"  ✗ Could not import StrategyStore: {e}")
    except Exception as e:
        print(f"  ✗ StrategyStore test failed: {e}")
        traceback.print_exc()

def test_data_store():
    """Test DataStore initialization and basic operations."""
    print("\n=== Testing DataStore ===")
    
    try:
        from ml.stores.data_store import DataStore
        from ml.registry.data_registry import DataRegistry
        from ml.registry.persistence import PersistenceConfig, BackendType
        
        # Test 1: Initialize DataStore
        try:
            # Create a temporary registry path
            with tempfile.TemporaryDirectory() as temp_dir:
                registry_path = Path(temp_dir) / "registry"
                
                persistence_config = PersistenceConfig(
                    backend=BackendType.POSTGRES,
                    connection_string="postgresql://postgres:postgres@localhost:5432/nautilus"
                )
                
                # Initialize DataRegistry first
                data_registry = DataRegistry(
                    registry_path=registry_path,
                    persistence_config=persistence_config
                )
                print("  ✓ DataRegistry initialized")
                
                # Initialize DataStore
                store = DataStore(
                    registry=data_registry,
                    connection_string="postgresql://postgres:postgres@localhost:5432/nautilus"
                )
                print("  ✓ DataStore initialized with DataRegistry")
                
                # Test health check
                if hasattr(store, 'is_healthy'):
                    health = store.is_healthy()
                    print(f"    Health check: {'✓' if health else '✗'} {health}")
                
        except Exception as e:
            print(f"  ✗ DataStore init failed: {e}")
            traceback.print_exc()
        
    except ImportError as e:
        print(f"  ✗ Could not import DataStore: {e}")
    except Exception as e:
        print(f"  ✗ DataStore test failed: {e}")
        traceback.print_exc()

def test_registries():
    """Test all 4 registries initialization and basic operations."""
    print("\n=== Testing Registries ===")
    
    try:
        from ml.registry.feature_registry import FeatureRegistry
        from ml.registry.model_registry import ModelRegistry
        from ml.registry.strategy_registry import StrategyRegistry
        from ml.registry.data_registry import DataRegistry
        from ml.registry.persistence import PersistenceConfig, BackendType
        
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_path = Path(temp_dir) / "registry"
            
            # Test FeatureRegistry
            try:
                feature_registry = FeatureRegistry(registry_path)
                print("  ✓ FeatureRegistry initialized")
            except Exception as e:
                print(f"  ✗ FeatureRegistry init failed: {e}")
            
            # Test ModelRegistry  
            try:
                model_registry = ModelRegistry(registry_path)
                print("  ✓ ModelRegistry initialized")
            except Exception as e:
                print(f"  ✗ ModelRegistry init failed: {e}")
            
            # Test StrategyRegistry
            try:
                strategy_registry = StrategyRegistry(registry_path)
                print("  ✓ StrategyRegistry initialized")
            except Exception as e:
                print(f"  ✗ StrategyRegistry init failed: {e}")
            
            # Test DataRegistry with PostgreSQL persistence
            try:
                persistence_config = PersistenceConfig(
                    backend=BackendType.POSTGRES,
                    connection_string="postgresql://postgres:postgres@localhost:5432/nautilus"
                )
                data_registry = DataRegistry(
                    registry_path=registry_path / "datasets",
                    persistence_config=persistence_config
                )
                print("  ✓ DataRegistry initialized with PostgreSQL")
            except Exception as e:
                print(f"  ✗ DataRegistry PostgreSQL init failed: {e}")
                
            # Test DataRegistry with JSON persistence (fallback)
            try:
                persistence_config = PersistenceConfig(
                    backend=BackendType.JSON,
                    json_path=registry_path
                )
                data_registry = DataRegistry(
                    registry_path=registry_path / "datasets",
                    persistence_config=persistence_config
                )
                print("  ✓ DataRegistry initialized with JSON fallback")
            except Exception as e:
                print(f"  ✗ DataRegistry JSON init failed: {e}")
                
    except ImportError as e:
        print(f"  ✗ Could not import registries: {e}")
    except Exception as e:
        print(f"  ✗ Registry test failed: {e}")
        traceback.print_exc()

def test_progressive_fallback():
    """Test progressive fallback to DummyStore when PostgreSQL unavailable."""
    print("\n=== Testing Progressive Fallback ===")
    
    try:
        from ml.stores.base import DummyStore
        from ml.registry.base import DummyRegistry
        
        # Test 1: DummyStore initialization
        try:
            dummy_store = DummyStore()
            print("  ✓ DummyStore initialized")
            
            # Test basic operations
            dummy_store.flush()
            print("  ✓ DummyStore.flush() works")
            
            if hasattr(dummy_store, 'is_healthy'):
                health = dummy_store.is_healthy()
                print(f"    DummyStore health: {'✓' if health else '✗'} {health}")
                
        except Exception as e:
            print(f"  ✗ DummyStore test failed: {e}")
        
        # Test 2: DummyRegistry initialization
        try:
            dummy_registry = DummyRegistry()
            print("  ✓ DummyRegistry initialized")
        except Exception as e:
            print(f"  ✗ DummyRegistry test failed: {e}")
            
        # Test 3: Test fallback in BaseMLInferenceActor initialization
        try:
            from ml.actors.base import BaseMLInferenceActor
            from ml.config.base import MLActorConfig
            
            # Create minimal config that forces dummy stores
            config = MLActorConfig(
                component_id="test_actor",
                model_path="/tmp/nonexistent_model.onnx",
                use_dummy_stores=True  # Force dummy stores
            )
            
            # This should be an abstract class, so instantiation might fail
            # Let's check if the _init_stores_and_registries method exists
            if hasattr(BaseMLInferenceActor, '_init_stores_and_registries'):
                print("  ✓ BaseMLInferenceActor has _init_stores_and_registries method")
            else:
                print("  ✗ BaseMLInferenceActor missing _init_stores_and_registries method")
                
        except Exception as e:
            print(f"  ✗ BaseMLInferenceActor fallback test failed: {e}")
            
    except ImportError as e:
        print(f"  ✗ Could not import fallback classes: {e}")
    except Exception as e:
        print(f"  ✗ Progressive fallback test failed: {e}")
        traceback.print_exc()

def test_database_schema():
    """Test database schema migrations and table creation."""
    print("\n=== Testing Database Schema ===")
    
    try:
        from sqlalchemy import text, MetaData
        from ml.core.db_engine import EngineManager
        
        # Test connection to PostgreSQL
        try:
            connection_string = "postgresql://postgres:postgres@localhost:5432/nautilus"
            engine = EngineManager.get_engine(connection_string)
            print("  ✓ Database engine created")
            
            with engine.connect() as conn:
                # Test basic connectivity
                result = conn.execute(text("SELECT 1"))
                print("  ✓ Database connection works")
                
                # Check if ML tables exist
                metadata = MetaData()
                metadata.reflect(bind=engine)
                table_names = list(metadata.tables.keys())
                
                ml_tables = [name for name in table_names if name.startswith('ml_')]
                print(f"  Found ML tables: {ml_tables}")
                
                # Check specific tables mentioned in the code
                expected_tables = [
                    'ml_feature_values',
                    'ml_model_predictions', 
                    'ml_strategy_signals'
                ]
                
                for table in expected_tables:
                    if table in table_names:
                        print(f"  ✓ Table {table} exists")
                    else:
                        print(f"  ✗ Table {table} missing")
                        
        except Exception as e:
            print(f"  ✗ Database schema test failed: {e}")
            traceback.print_exc()
            
    except ImportError as e:
        print(f"  ✗ Could not import database utilities: {e}")
    except Exception as e:
        print(f"  ✗ Database schema test failed: {e}")
        traceback.print_exc()

def test_protocols():
    """Test MLComponentProtocol integration and health/performance interfaces."""
    print("\n=== Testing Protocol Integration ===")
    
    try:
        from ml.stores.protocols import FeatureStoreProtocol, ModelStoreProtocol, StrategyStoreProtocol
        from ml.stores.feature_store import FeatureStore
        from ml.stores.model_store import ModelStore  
        from ml.stores.strategy_store import StrategyStore
        from ml.stores.base import DummyStore
        
        print("  ✓ Store protocols imported successfully")
        
        # Test if stores actually implement the protocols
        try:
            connection_string = "postgresql://postgres:postgres@localhost:5432/nautilus"
            feature_store = FeatureStore(connection_string=connection_string)
            
            # Check if feature store conforms to protocol
            if isinstance(feature_store, type(FeatureStoreProtocol)):
                print("  ✓ FeatureStore conforms to FeatureStoreProtocol")
            else:
                print("  ✗ FeatureStore does not conform to FeatureStoreProtocol")
                
        except Exception as e:
            print(f"  ✗ Protocol conformance test failed: {e}")
        
        # Test if DummyStore conforms to protocols  
        try:
            dummy = DummyStore()
            print("  ✓ DummyStore can be instantiated")
            
            # Test common protocol methods
            if hasattr(dummy, 'flush'):
                dummy.flush()
                print("  ✓ DummyStore.flush() works")
                
            if hasattr(dummy, 'is_healthy'):
                health = dummy.is_healthy()
                print(f"  ✓ DummyStore.is_healthy() returns {health}")
                
        except Exception as e:
            print(f"  ✗ DummyStore protocol test failed: {e}")
            
    except ImportError as e:
        print(f"  ✗ Could not import protocols: {e}")
    except Exception as e:
        print(f"  ✗ Protocol integration test failed: {e}")
        traceback.print_exc()

def main():
    """Run all tests and report results."""
    print("ML Stores and Registries Functionality Test")
    print("=" * 50)
    
    # Test each component
    test_feature_store()
    test_model_store() 
    test_strategy_store()
    test_data_store()
    test_registries()
    test_progressive_fallback()
    test_database_schema()
    test_protocols()
    
    print("\n" + "=" * 50)
    print("Test Summary:")
    print("Check the output above for ✓ (success) and ✗ (failure) indicators")
    print("This test reveals what actually works vs what's documented")

if __name__ == "__main__":
    main()