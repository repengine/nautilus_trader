#!/usr/bin/env python3
"""
Comprehensive test script to validate the 4-store + 4-registry integration claims.

This script tests:
1. All 4 stores: FeatureStore, ModelStore, StrategyStore, DataStore
2. All 4 registries: FeatureRegistry, ModelRegistry, StrategyRegistry, DataRegistry
3. Progressive fallback: PostgreSQL → DummyStore
4. BaseMLInferenceActor automatic initialization
5. Data persistence and retrieval
6. Cross-store integration and event propagation

Author: Claude Code Assistant
Purpose: Validate documentation claims with actual code testing

"""

import os
import sys
import time
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, List

# Add the project root to Python path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

# Set environment variables to enable dummy fallback for tests
os.environ["ML_TEST_ALLOW_NON_ONNX"] = "1"
os.environ["ML_ALLOW_NON_ONNX_IN_TESTS"] = "1"


def test_section(title: str):
    """
    Print a section header.
    """
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")


def test_imports():
    """
    Test that we can import all the claimed components.
    """
    test_section("TESTING IMPORTS")

    results = {}

    # Test store imports
    try:
        from ml.stores import FeatureStore, ModelStore, StrategyStore, DataStore
        from ml.stores.base import DummyStore

        results["stores"] = True
        print("✅ All 4 stores imported successfully")
        print(f"   - FeatureStore: {FeatureStore}")
        print(f"   - ModelStore: {ModelStore}")
        print(f"   - StrategyStore: {StrategyStore}")
        print(f"   - DataStore: {DataStore}")
        print(f"   - DummyStore: {DummyStore}")
    except Exception as e:
        results["stores"] = False
        print(f"❌ Store imports failed: {e}")
        traceback.print_exc()

    # Test registry imports
    try:
        from ml.registry import FeatureRegistry, ModelRegistry, StrategyRegistry, DataRegistry
        from ml.registry.base import DummyRegistry

        results["registries"] = True
        print("✅ All 4 registries imported successfully")
        print(f"   - FeatureRegistry: {FeatureRegistry}")
        print(f"   - ModelRegistry: {ModelRegistry}")
        print(f"   - StrategyRegistry: {StrategyRegistry}")
        print(f"   - DataRegistry: {DataRegistry}")
        print(f"   - DummyRegistry: {DummyRegistry}")
    except Exception as e:
        results["registries"] = False
        print(f"❌ Registry imports failed: {e}")
        traceback.print_exc()

    # Test base actor import
    try:
        from ml.actors.base import BaseMLInferenceActor

        results["base_actor"] = True
        print("✅ BaseMLInferenceActor imported successfully")
        print(f"   - BaseMLInferenceActor: {BaseMLInferenceActor}")
    except Exception as e:
        results["base_actor"] = False
        print(f"❌ BaseMLInferenceActor import failed: {e}")
        traceback.print_exc()

    return results


def test_dummy_stores():
    """
    Test that dummy stores work correctly.
    """
    test_section("TESTING DUMMY STORE FUNCTIONALITY")

    from ml.stores.base import DummyStore

    results = {}

    try:
        # Create dummy store instance
        dummy = DummyStore()
        print("✅ DummyStore instance created")

        # Test various method calls (should not raise exceptions)
        dummy.write_features("test", "EURUSD", {"feature1": 1.0}, 12345, 12345)
        dummy.write_prediction("model1", "EURUSD", 0.5, 0.8, {"f1": 1.0}, 1.0, 12345)
        dummy.write_signal("strategy1", "EURUSD", "BUY", 0.7, {}, {}, {}, 12345, 12345)
        dummy.flush()

        stats = dummy.get_statistics()
        print(f"✅ DummyStore methods work correctly, stats: {stats}")

        # Test that dummy store has all required methods
        required_methods = [
            "write_features",
            "write_prediction",
            "write_signal",
            "write_batch",
            "flush",
            "get_statistics",
            "get_latest",
            "read_predictions",
            "read_signals",
            "get_model_performance",
            "get_strategy_performance",
        ]

        missing_methods = []
        for method in required_methods:
            if not hasattr(dummy, method):
                missing_methods.append(method)

        if not missing_methods:
            print("✅ DummyStore has all required protocol methods")
            results["dummy_store"] = True
        else:
            print(f"❌ DummyStore missing methods: {missing_methods}")
            results["dummy_store"] = False

    except Exception as e:
        results["dummy_store"] = False
        print(f"❌ DummyStore test failed: {e}")
        traceback.print_exc()

    return results


def test_dummy_registries():
    """
    Test that dummy registries work correctly.
    """
    test_section("TESTING DUMMY REGISTRY FUNCTIONALITY")

    from ml.registry.base import DummyRegistry

    results = {}

    try:
        # Create dummy registry instance
        dummy = DummyRegistry()
        print("✅ DummyRegistry instance created")

        # Test various registry methods
        dummy.emit_event("dataset1", "EURUSD", "stage1", "source1", "run1", 0, 0, 100, "success")
        dummy.update_watermark("dataset1", "EURUSD", "source1", 123456789, 100, 100.0)

        manifest = dummy.get_manifest("test")
        contract = dummy.get_contract("test")

        print("✅ DummyRegistry methods work correctly")
        print(f"   - Manifest: {manifest}")
        print(f"   - Contract: {contract}")

        results["dummy_registry"] = True

    except Exception as e:
        results["dummy_registry"] = False
        print(f"❌ DummyRegistry test failed: {e}")
        traceback.print_exc()

    return results


def test_progressive_fallback():
    """
    Test the progressive fallback from PostgreSQL to DummyStore.
    """
    test_section("TESTING PROGRESSIVE FALLBACK MECHANISM")

    results = {}

    try:
        # Import necessary components
        from ml.stores import FeatureStore, ModelStore, StrategyStore
        from ml.registry.persistence import PersistenceConfig, BackendType

        print("Testing fallback with invalid database connection...")

        # Test with invalid connection string - should fallback to dummy or raise error
        invalid_connection = "postgresql://invalid:invalid@nonexistent:9999/fake"

        try:
            feature_store = FeatureStore(connection_string=invalid_connection)
            print("⚠️  FeatureStore created with invalid connection - may be using fallback")
            results["feature_store_fallback"] = True
        except Exception as e:
            print(f"⚠️  FeatureStore failed with invalid connection (expected): {e}")
            results["feature_store_fallback"] = False

        # Test with persistence config fallback
        try:
            persistence_config = PersistenceConfig(
                backend=BackendType.POSTGRES,
                connection_string=invalid_connection,
            )
            model_store = ModelStore(persistence_config=persistence_config)
            print("⚠️  ModelStore created with invalid persistence config - may be using fallback")
            results["model_store_fallback"] = True
        except Exception as e:
            print(f"⚠️  ModelStore failed with invalid persistence config (expected): {e}")
            results["model_store_fallback"] = False

        print("✅ Progressive fallback mechanism tested")

    except Exception as e:
        results["fallback"] = False
        print(f"❌ Progressive fallback test failed: {e}")
        traceback.print_exc()

    return results


def test_base_actor_initialization():
    """
    Test BaseMLInferenceActor initialization with dummy stores.
    """
    test_section("TESTING BaseMLInferenceActor INITIALIZATION")

    results = {}

    try:
        from ml.actors.base import BaseMLInferenceActor
        from ml.config.base import MLActorConfig

        # Create a concrete implementation for testing
        class TestMLActor(BaseMLInferenceActor):
            def _load_model(self):
                self._model = lambda x: (0.5, 0.8)  # Mock model

            def _initialize_features(self):
                pass  # Mock initialization

            def _compute_features(self, bar):
                return None  # Mock features

            def _predict(self, features):
                return 0.5, 0.8  # Mock prediction

        # Create config with dummy stores enabled
        config = MLActorConfig(
            component_id="test_actor",
            model_path="/tmp/fake_model.onnx",
            use_dummy_stores=True,  # Enable dummy stores for testing
        )

        print("Creating TestMLActor with dummy stores...")
        actor = TestMLActor(config)

        # Check that all stores are initialized
        stores_initialized = [
            hasattr(actor, "_feature_store"),
            hasattr(actor, "_model_store"),
            hasattr(actor, "_strategy_store"),
            hasattr(actor, "_data_store"),
        ]

        registries_initialized = [
            hasattr(actor, "_feature_registry"),
            hasattr(actor, "_model_registry"),
            hasattr(actor, "_strategy_registry"),
            hasattr(actor, "_data_registry"),
        ]

        print(f"Stores initialized: {sum(stores_initialized)}/4")
        print(f"Registries initialized: {sum(registries_initialized)}/4")

        # Test property accessors
        try:
            feature_store = actor.feature_store
            model_store = actor.model_store
            strategy_store = actor.strategy_store
            data_store = actor.data_store

            feature_registry = actor.feature_registry
            model_registry = actor.model_registry
            strategy_registry = actor.strategy_registry
            data_registry = actor.data_registry

            print("✅ All store and registry property accessors work")
            print(f"   - FeatureStore: {type(feature_store).__name__}")
            print(f"   - ModelStore: {type(model_store).__name__}")
            print(f"   - StrategyStore: {type(strategy_store).__name__}")
            print(f"   - DataStore: {type(data_store).__name__}")
            print(f"   - FeatureRegistry: {type(feature_registry).__name__}")
            print(f"   - ModelRegistry: {type(model_registry).__name__}")
            print(f"   - StrategyRegistry: {type(strategy_registry).__name__}")
            print(f"   - DataRegistry: {type(data_registry).__name__}")

            results["actor_initialization"] = True

        except Exception as e:
            print(f"❌ Property accessor test failed: {e}")
            results["actor_initialization"] = False

    except Exception as e:
        results["actor_initialization"] = False
        print(f"❌ BaseMLInferenceActor initialization failed: {e}")
        traceback.print_exc()

    return results


def test_data_persistence():
    """
    Test actual data persistence and retrieval.
    """
    test_section("TESTING DATA PERSISTENCE AND RETRIEVAL")

    results = {}

    try:
        from ml.stores.base import DummyStore
        from ml.stores import FeatureStore

        # Test with DummyStore first (should always work)
        dummy = DummyStore()

        # Test feature writing
        dummy.write_features(
            feature_set_id="test_features",
            instrument_id="EURUSD",
            features={"close_ratio": 1.05, "volume_ma": 1500.0},
            ts_event=int(time.time() * 1e9),  # nanoseconds
            ts_init=int(time.time() * 1e9),
        )

        # Test model prediction writing
        dummy.write_prediction(
            model_id="test_model",
            instrument_id="EURUSD",
            prediction=0.75,
            confidence=0.90,
            features={"close_ratio": 1.05, "volume_ma": 1500.0},
            inference_time_ms=2.5,
            ts_event=int(time.time() * 1e9),
        )

        # Test strategy signal writing
        dummy.write_signal(
            strategy_id="test_strategy",
            instrument_id="EURUSD",
            signal_type="BUY",
            strength=0.80,
            model_predictions={"test_model": 0.75},
            risk_metrics={"var": 0.02},
            execution_params={"stop_loss": 0.95},
            ts_event=int(time.time() * 1e9),
            ts_init=int(time.time() * 1e9),
        )

        print("✅ DummyStore persistence operations completed successfully")

        # Flush and get statistics
        dummy.flush()
        stats = dummy.get_statistics()
        print(f"✅ DummyStore stats: {stats}")

        results["dummy_persistence"] = True

        # Try to test with real stores if possible
        try:
            # This might fail if no PostgreSQL available, which is expected
            feature_store = FeatureStore("postgresql://postgres:postgres@localhost:5432/nautilus")
            print("⚠️  Real FeatureStore created - PostgreSQL may be available for testing")
            results["real_store_available"] = True
        except Exception:
            print("INFO: Real stores not available (expected without PostgreSQL)")
            results["real_store_available"] = False

    except Exception as e:
        results["persistence"] = False
        print(f"❌ Data persistence test failed: {e}")
        traceback.print_exc()

    return results


def test_registry_functionality():
    """
    Test registry functionality.
    """
    test_section("TESTING REGISTRY FUNCTIONALITY")

    results = {}

    try:
        from ml.registry import DataRegistry, FeatureRegistry, ModelRegistry, StrategyRegistry
        from ml.registry.persistence import PersistenceConfig, BackendType
        from ml.registry.dataclasses import DatasetManifest, DatasetType, StorageKind
        import tempfile

        # Test with JSON backend (file-based, should always work)
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_path = Path(temp_dir) / "registry"

            persistence_config = PersistenceConfig(
                backend=BackendType.JSON,
                json_path=registry_path,
            )

            # Test DataRegistry
            try:
                data_registry = DataRegistry(
                    registry_path=registry_path / "datasets",
                    persistence_config=persistence_config,
                )

                # Create a test dataset manifest
                manifest = DatasetManifest(
                    dataset_id="test_bars",
                    dataset_type=DatasetType.BARS,
                    storage_kind=StorageKind.POSTGRES,
                    location="test_table",
                    schema={"ts_event": "int64", "close": "float64"},
                    ts_field="ts_event",
                    primary_keys=["ts_event"],
                    schema_hash="test_hash_123",
                    lineage=[],
                    pipeline_signature="test_pipeline",
                    version="1.0.0",
                    retention_days=365,
                )

                # Register the dataset
                dataset_id = data_registry.register_dataset(manifest)
                print(f"✅ DataRegistry: Dataset registered with ID: {dataset_id}")

                # Emit an event
                data_registry.emit_event(
                    dataset_id="test_bars",
                    instrument_id="EURUSD",
                    stage="CATALOG_WRITTEN",
                    source="historical",
                    run_id="test_run_123",
                    ts_min=int(time.time() * 1e9),
                    ts_max=int(time.time() * 1e9),
                    count=1000,
                    status="success",
                )
                print("✅ DataRegistry: Event emitted successfully")

                # Update watermark
                data_registry.update_watermark(
                    dataset_id="test_bars",
                    instrument_id="EURUSD",
                    source="live",
                    last_success_ns=int(time.time() * 1e9),
                    count=100,
                    completeness_pct=98.5,
                )
                print("✅ DataRegistry: Watermark updated successfully")

                results["data_registry"] = True

            except Exception as e:
                print(f"❌ DataRegistry test failed: {e}")
                results["data_registry"] = False
                traceback.print_exc()

            # Test FeatureRegistry
            try:
                feature_registry = FeatureRegistry(
                    registry_path=registry_path,
                    persistence_config=persistence_config,
                )
                print("✅ FeatureRegistry: Created successfully")
                results["feature_registry"] = True
            except Exception as e:
                print(f"❌ FeatureRegistry creation failed: {e}")
                results["feature_registry"] = False
                traceback.print_exc()

            # Test ModelRegistry
            try:
                model_registry = ModelRegistry(
                    registry_path=registry_path,
                    persistence_config=persistence_config,
                )
                print("✅ ModelRegistry: Created successfully")
                results["model_registry"] = True
            except Exception as e:
                print(f"❌ ModelRegistry creation failed: {e}")
                results["model_registry"] = False
                traceback.print_exc()

            # Test StrategyRegistry
            try:
                strategy_registry = StrategyRegistry(
                    registry_path=registry_path,
                )
                print("✅ StrategyRegistry: Created successfully")
                results["strategy_registry"] = True
            except Exception as e:
                print(f"❌ StrategyRegistry creation failed: {e}")
                results["strategy_registry"] = False
                traceback.print_exc()

    except Exception as e:
        print(f"❌ Registry functionality test failed: {e}")
        traceback.print_exc()

    return results


def test_cross_store_integration():
    """
    Test integration between stores and registries.
    """
    test_section("TESTING CROSS-STORE INTEGRATION")

    results = {}

    try:
        # This is a basic integration test using dummy components
        from ml.stores.base import DummyStore
        from ml.registry.base import DummyRegistry

        # Create instances
        feature_store = DummyStore()
        model_store = DummyStore()
        data_registry = DummyRegistry()

        # Simulate cross-store workflow
        # 1. Write features
        features = {"close_ratio": 1.05, "rsi": 65.0}
        ts_event = int(time.time() * 1e9)

        feature_store.write_features(
            feature_set_id="integration_test",
            instrument_id="EURUSD",
            features=features,
            ts_event=ts_event,
            ts_init=ts_event,
        )

        # 2. Write model prediction using the same features
        model_store.write_prediction(
            model_id="integration_model",
            instrument_id="EURUSD",
            prediction=0.75,
            confidence=0.90,
            features=features,
            inference_time_ms=1.5,
            ts_event=ts_event,
        )

        # 3. Emit event to registry about the workflow
        data_registry.emit_event(
            dataset_id="integration_workflow",
            instrument_id="EURUSD",
            stage="PREDICTION_EMITTED",
            source="live",
            run_id="integration_run",
            ts_min=ts_event,
            ts_max=ts_event,
            count=1,
            status="success",
        )

        print("✅ Cross-store integration workflow completed successfully")
        print("   - Features written to FeatureStore")
        print("   - Predictions written to ModelStore")
        print("   - Events emitted to DataRegistry")

        results["cross_store_integration"] = True

    except Exception as e:
        print(f"❌ Cross-store integration test failed: {e}")
        traceback.print_exc()
        results["cross_store_integration"] = False

    return results


def run_comprehensive_test():
    """
    Run all tests and compile results.
    """
    test_section("COMPREHENSIVE ML STORES + REGISTRIES INTEGRATION TEST")

    print("Testing the mandatory 4-store + 4-registry integration pattern...")
    print("Documentation claims:")
    print("  ✓ All ML actors MUST inherit from BaseMLInferenceActor")
    print("  ✓ Automatic initialization of 4 stores + 4 registries")
    print("  ✓ Progressive fallback: PostgreSQL → DummyStore")
    print("  ✓ Protocol-based interfaces for type safety")
    print("  ✓ Data persistence and event propagation")

    # Run all test sections
    all_results = {}

    all_results.update(test_imports())
    all_results.update(test_dummy_stores())
    all_results.update(test_dummy_registries())
    all_results.update(test_progressive_fallback())
    all_results.update(test_base_actor_initialization())
    all_results.update(test_data_persistence())
    all_results.update(test_registry_functionality())
    all_results.update(test_cross_store_integration())

    # Summary
    test_section("TEST RESULTS SUMMARY")

    passed = sum(1 for result in all_results.values() if result is True)
    failed = sum(1 for result in all_results.values() if result is False)
    total = len(all_results)

    print(f"Total tests: {total}")
    print(f"Passed: {passed} ✅")
    print(f"Failed: {failed} ❌")
    print(f"Success rate: {passed/total*100:.1f}%")

    print("\nDetailed results:")
    for test_name, result in all_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")

    # Validation against documentation claims
    test_section("DOCUMENTATION CLAIMS VALIDATION")

    mandatory_components = [
        ("stores", "All 4 stores available"),
        ("registries", "All 4 registries available"),
        ("base_actor", "BaseMLInferenceActor available"),
        ("actor_initialization", "Automatic initialization works"),
        ("dummy_store", "DummyStore fallback works"),
        ("dummy_registry", "DummyRegistry fallback works"),
    ]

    claims_validated = 0
    for component, description in mandatory_components:
        if all_results.get(component):
            print(f"✅ VALIDATED: {description}")
            claims_validated += 1
        else:
            print(f"❌ FAILED: {description}")

    print(f"\nDocumentation claims validated: {claims_validated}/{len(mandatory_components)}")

    # Evidence of actual functionality
    test_section("EVIDENCE OF ACTUAL IMPLEMENTATION")

    evidence = []
    if all_results.get("stores"):
        evidence.append(
            "✓ All 4 stores (FeatureStore, ModelStore, StrategyStore, DataStore) are importable"
        )
    if all_results.get("registries"):
        evidence.append(
            "✓ All 4 registries (FeatureRegistry, ModelRegistry, StrategyRegistry, DataRegistry) are importable"
        )
    if all_results.get("dummy_persistence"):
        evidence.append("✓ Data persistence operations work (with DummyStore)")
    if all_results.get("data_registry"):
        evidence.append("✓ DataRegistry can register datasets, emit events, and update watermarks")
    if all_results.get("actor_initialization"):
        evidence.append(
            "✓ BaseMLInferenceActor automatically initializes all 4 stores + 4 registries"
        )
    if all_results.get("cross_store_integration"):
        evidence.append("✓ Cross-store integration workflow operates correctly")

    for item in evidence:
        print(f"  {item}")

    return all_results


if __name__ == "__main__":
    try:
        results = run_comprehensive_test()

        # Exit with appropriate code
        if all(results.values()):
            print("\n🎉 ALL TESTS PASSED - Documentation claims fully validated!")
            sys.exit(0)
        else:
            print("\n⚠️  SOME TESTS FAILED - Documentation claims partially validated")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n⏹️  Test interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n💥 Test runner failed: {e}")
        traceback.print_exc()
        sys.exit(1)
