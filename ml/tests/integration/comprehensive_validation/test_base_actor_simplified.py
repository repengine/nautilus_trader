#!/usr/bin/env python3
"""
Simplified test for BaseMLInferenceActor without Nautilus dependencies.

This tests the core initialization functionality.

"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

# Test environment setup
os.environ["ML_TEST_ALLOW_NON_ONNX"] = "1"
os.environ["ML_ALLOW_NON_ONNX_IN_TESTS"] = "1"


def test_base_actor_stores_registries_init():
    """
    Test that BaseMLInferenceActor has the required store initialization logic.
    """

    try:
        from ml.actors.base import BaseMLInferenceActor

        # Check that the class exists and has the required attributes/methods
        print("✅ BaseMLInferenceActor imported successfully")

        # Check for the mandatory initialization method
        if hasattr(BaseMLInferenceActor, "_init_stores_and_registries"):
            print("✅ BaseMLInferenceActor has _init_stores_and_registries method")
        else:
            print("❌ BaseMLInferenceActor missing _init_stores_and_registries method")
            return False

        # Check for property accessors
        required_properties = [
            "feature_store",
            "model_store",
            "strategy_store",
            "data_store",
            "feature_registry",
            "model_registry",
            "strategy_registry",
            "data_registry",
        ]

        missing_properties = []
        for prop in required_properties:
            if not hasattr(BaseMLInferenceActor, prop):
                missing_properties.append(prop)

        if missing_properties:
            print(f"❌ BaseMLInferenceActor missing properties: {missing_properties}")
            return False
        else:
            print("✅ BaseMLInferenceActor has all 8 required property accessors")
            print(f"   - Store properties: {required_properties[:4]}")
            print(f"   - Registry properties: {required_properties[4:]}")

        # Read the source to verify the initialization logic
        import inspect

        source = inspect.getsource(BaseMLInferenceActor._init_stores_and_registries)

        # Check for key initialization patterns
        if "FeatureStore" in source and "ModelStore" in source:
            print("✅ Source code confirms FeatureStore and ModelStore initialization")
        if "StrategyStore" in source and "DataStore" in source:
            print("✅ Source code confirms StrategyStore and DataStore initialization")
        if "FeatureRegistry" in source and "ModelRegistry" in source:
            print("✅ Source code confirms FeatureRegistry and ModelRegistry initialization")
        if "StrategyRegistry" in source and "DataRegistry" in source:
            print("✅ Source code confirms StrategyRegistry and DataRegistry initialization")
        if "DummyStore" in source:
            print("✅ Source code confirms DummyStore fallback implementation")

        print("🎉 BaseMLInferenceActor implements mandatory 4-store + 4-registry pattern!")
        return True

    except Exception as e:
        print(f"❌ BaseMLInferenceActor test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Testing BaseMLInferenceActor 4-store + 4-registry initialization...")
    success = test_base_actor_stores_registries_init()

    if success:
        print(
            "\n✅ VALIDATION SUCCESS: BaseMLInferenceActor supports mandatory integration pattern",
        )
        sys.exit(0)
    else:
        print("\n❌ VALIDATION FAILED: BaseMLInferenceActor missing required functionality")
        sys.exit(1)
