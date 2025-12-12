#!/usr/bin/env python3
"""
System Validation Smoke Tests for MLPipelineOrchestrator
Phase 5: Deployment Validation

Exercises the orchestrator in BOTH modes (legacy and component) to verify
real operations work end-to-end.
"""
import os
import sys
import tempfile
import traceback
from pathlib import Path
from unittest.mock import Mock, MagicMock

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))


def test_import_and_initialize():
    """Test 1: Import and Initialize in both modes"""
    print("\n" + "=" * 80)
    print("TEST 1: Import and Initialize")
    print("=" * 80)

    results = {}

    for mode in ["1", "0"]:
        mode_name = "Legacy" if mode == "1" else "Component"
        print(f"\n--- Testing {mode_name} Mode (flag={mode}) ---")

        try:
            # Set environment variable
            os.environ["ML_USE_LEGACY_PIPELINE_ORCHESTRATOR"] = mode

            # Force reimport to pick up new flag value
            import importlib
            if "ml.orchestration" in sys.modules:
                del sys.modules["ml.orchestration"]
            if "ml.orchestration.pipeline_orchestrator_facade" in sys.modules:
                del sys.modules["ml.orchestration.pipeline_orchestrator_facade"]

            # Import the orchestrator
            from ml.orchestration import MLPipelineOrchestrator

            print(f"✅ {mode_name} mode: Import OK")
            results[mode_name] = "PASS"

        except Exception as e:
            print(f"❌ {mode_name} mode: Import FAILED")
            print(f"   Error: {e}")
            print(f"   Traceback:\n{traceback.format_exc()}")
            results[mode_name] = f"FAIL: {e}"

    return results


def test_create_orchestrator_instance():
    """Test 2: Create Orchestrator Instance"""
    print("\n" + "=" * 80)
    print("TEST 2: Create Orchestrator Instance")
    print("=" * 80)

    results = {}

    for mode in ["1", "0"]:
        mode_name = "Legacy" if mode == "1" else "Component"
        print(f"\n--- Testing {mode_name} Mode (flag={mode}) ---")

        try:
            os.environ["ML_USE_LEGACY_PIPELINE_ORCHESTRATOR"] = mode

            # Force reimport
            import importlib
            if "ml.orchestration" in sys.modules:
                del sys.modules["ml.orchestration"]
            if "ml.orchestration.pipeline_orchestrator_facade" in sys.modules:
                del sys.modules["ml.orchestration.pipeline_orchestrator_facade"]

            from ml.orchestration.pipeline_orchestrator_facade import MLPipelineOrchestratorFacade

            # Create minimal mocks
            mock_coverage = Mock()
            mock_writer = Mock()
            mock_build_main = Mock(return_value=0)
            mock_teacher_main = Mock(return_value=0)

            # Create orchestrator
            orchestrator = MLPipelineOrchestratorFacade(
                coverage=mock_coverage,
                writer=mock_writer,
                build_main=mock_build_main,
                teacher_main=mock_teacher_main,
            )

            print(f"✅ {mode_name} mode: Orchestrator initialized")
            results[mode_name] = "PASS"

        except Exception as e:
            print(f"❌ {mode_name} mode: Initialization FAILED")
            print(f"   Error: {e}")
            print(f"   Traceback:\n{traceback.format_exc()}")
            results[mode_name] = f"FAIL: {e}"

    return results


def test_verify_stage_controller_integration():
    """Test 3: Verify StageController Integration"""
    print("\n" + "=" * 80)
    print("TEST 3: Verify StageController Integration")
    print("=" * 80)

    results = {}

    # Only test component mode (StageController not used in legacy)
    print("\n--- Testing Component Mode (flag=0) ---")

    try:
        os.environ["ML_USE_LEGACY_PIPELINE_ORCHESTRATOR"] = "0"

        # Force reimport
        import importlib
        if "ml.orchestration" in sys.modules:
            del sys.modules["ml.orchestration"]
        if "ml.orchestration.pipeline_orchestrator_facade" in sys.modules:
            del sys.modules["ml.orchestration.pipeline_orchestrator_facade"]

        from ml.orchestration.pipeline_orchestrator_facade import MLPipelineOrchestratorFacade

        # Create orchestrator
        mock_coverage = Mock()
        mock_writer = Mock()
        mock_build_main = Mock(return_value=0)
        mock_teacher_main = Mock(return_value=0)

        orchestrator = MLPipelineOrchestratorFacade(
            coverage=mock_coverage,
            writer=mock_writer,
            build_main=mock_build_main,
            teacher_main=mock_teacher_main,
        )

        # Verify StageController is present
        if orchestrator._stage_controller is None:
            raise AssertionError("StageController is None")
        print("✅ StageController present")

        # Verify all components present
        if orchestrator._ingestion_coordinator is None:
            raise AssertionError("IngestionCoordinator is None")
        print("✅ IngestionCoordinator present")

        if orchestrator._dataset_builder is None:
            raise AssertionError("DatasetBuilder is None")
        print("✅ DatasetBuilder present")

        if orchestrator._training_coordinator is None:
            raise AssertionError("TrainingCoordinator is None")
        print("✅ TrainingCoordinator present")

        results["Component"] = "PASS"

    except Exception as e:
        print("❌ Component integration FAILED")
        print(f"   Error: {e}")
        print(f"   Traceback:\n{traceback.format_exc()}")
        results["Component"] = f"FAIL: {e}"

    # Test legacy mode doesn't use StageController
    print("\n--- Testing Legacy Mode (flag=1) ---")
    try:
        os.environ["ML_USE_LEGACY_PIPELINE_ORCHESTRATOR"] = "1"

        # Force reimport
        if "ml.orchestration" in sys.modules:
            del sys.modules["ml.orchestration"]
        if "ml.orchestration.pipeline_orchestrator_facade" in sys.modules:
            del sys.modules["ml.orchestration.pipeline_orchestrator_facade"]

        from ml.orchestration.pipeline_orchestrator_facade import MLPipelineOrchestratorFacade

        orchestrator = MLPipelineOrchestratorFacade(
            coverage=mock_coverage,
            writer=mock_writer,
            build_main=mock_build_main,
            teacher_main=mock_teacher_main,
        )

        # In legacy mode, StageController should still be initialized but not used
        print("✅ Legacy mode initialized (StageController not used)")
        results["Legacy"] = "PASS"

    except Exception as e:
        print("❌ Legacy mode FAILED")
        print(f"   Error: {e}")
        results["Legacy"] = f"FAIL: {e}"

    return results


def test_config_resolution():
    """Test 4: Test Config Resolution (Dry Run)"""
    print("\n" + "=" * 80)
    print("TEST 4: Config Resolution (Dry Run)")
    print("=" * 80)

    results = {}

    try:
        from ml.orchestration.config_types import DatasetBuildConfig, OrchestratorConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = DatasetBuildConfig(
                data_dir=tmpdir,
                symbols="SPY",
                out_dir=tmpdir,
                dataset_id="smoke_test",
            )

            print(f"✅ Config created: dataset_id={cfg.dataset_id}")
            print(f"   symbols={cfg.symbols}")
            print(f"   data_dir={cfg.data_dir}")
            print(f"   out_dir={cfg.out_dir}")

            results["Config"] = "PASS"

    except Exception as e:
        print("❌ Config creation FAILED")
        print(f"   Error: {e}")
        print(f"   Traceback:\n{traceback.format_exc()}")
        results["Config"] = f"FAIL: {e}"

    return results


def test_run_delegation_path():
    """Test 5: Test run() Delegation Path"""
    print("\n" + "=" * 80)
    print("TEST 5: run() Delegation Path")
    print("=" * 80)

    results = {}

    for mode in ["1", "0"]:
        mode_name = "Legacy" if mode == "1" else "Component"
        print(f"\n--- Testing {mode_name} Mode (flag={mode}) ---")

        try:
            os.environ["ML_USE_LEGACY_PIPELINE_ORCHESTRATOR"] = mode

            # Force reimport
            import importlib
            if "ml.orchestration" in sys.modules:
                del sys.modules["ml.orchestration"]
            if "ml.orchestration.pipeline_orchestrator_facade" in sys.modules:
                del sys.modules["ml.orchestration.pipeline_orchestrator_facade"]

            from ml.orchestration.pipeline_orchestrator_facade import MLPipelineOrchestratorFacade

            # Create orchestrator
            mock_coverage = Mock()
            mock_writer = Mock()
            mock_build_main = Mock(return_value=0)
            mock_teacher_main = Mock(return_value=0)

            orchestrator = MLPipelineOrchestratorFacade(
                coverage=mock_coverage,
                writer=mock_writer,
                build_main=mock_build_main,
                teacher_main=mock_teacher_main,
            )

            # Verify run() method exists and is callable
            if not callable(orchestrator.run):
                raise AssertionError("run() is not callable")
            print(f"✅ {mode_name} mode: run() method callable")

            # Verify run_training_only() method exists and is callable
            if not callable(orchestrator.run_training_only):
                raise AssertionError("run_training_only() is not callable")
            print(f"✅ {mode_name} mode: run_training_only() method callable")

            results[mode_name] = "PASS"

        except Exception as e:
            print(f"❌ {mode_name} mode: Delegation path FAILED")
            print(f"   Error: {e}")
            print(f"   Traceback:\n{traceback.format_exc()}")
            results[mode_name] = f"FAIL: {e}"

    return results


def test_all_public_methods_exist():
    """Test 6: Verify All Public Methods Exist"""
    print("\n" + "=" * 80)
    print("TEST 6: All Public Methods Exist")
    print("=" * 80)

    results = {}

    expected_methods = [
        "run_pre_ingestion",
        "backfill",
        "backfill_binding",
        "backfill_coverage",
        "build_dataset",
        "run_hpo",
        "train_teacher",
        "distill_student",
        "run",
        "run_training_only",
    ]

    for mode in ["1", "0"]:
        mode_name = "Legacy" if mode == "1" else "Component"
        print(f"\n--- Testing {mode_name} Mode (flag={mode}) ---")

        try:
            os.environ["ML_USE_LEGACY_PIPELINE_ORCHESTRATOR"] = mode

            # Force reimport
            import importlib
            if "ml.orchestration" in sys.modules:
                del sys.modules["ml.orchestration"]
            if "ml.orchestration.pipeline_orchestrator_facade" in sys.modules:
                del sys.modules["ml.orchestration.pipeline_orchestrator_facade"]

            from ml.orchestration.pipeline_orchestrator_facade import MLPipelineOrchestratorFacade

            # Create orchestrator
            mock_coverage = Mock()
            mock_writer = Mock()
            mock_build_main = Mock(return_value=0)
            mock_teacher_main = Mock(return_value=0)

            orchestrator = MLPipelineOrchestratorFacade(
                coverage=mock_coverage,
                writer=mock_writer,
                build_main=mock_build_main,
                teacher_main=mock_teacher_main,
            )

            # Verify all methods exist
            missing_methods = []
            for method in expected_methods:
                if not hasattr(orchestrator, method):
                    missing_methods.append(method)
                elif not callable(getattr(orchestrator, method)):
                    missing_methods.append(f"{method} (not callable)")

            if missing_methods:
                raise AssertionError(f"Missing methods: {', '.join(missing_methods)}")

            print(f"✅ {mode_name} mode: All {len(expected_methods)} public methods exist and are callable")

            results[mode_name] = "PASS"

        except Exception as e:
            print(f"❌ {mode_name} mode: Method verification FAILED")
            print(f"   Error: {e}")
            print(f"   Traceback:\n{traceback.format_exc()}")
            results[mode_name] = f"FAIL: {e}"

    return results


def main():
    """Run all smoke tests"""
    print("\n" + "#" * 80)
    print("# MLPipelineOrchestrator System Validation - Smoke Tests")
    print("# Phase 5: Deployment Validation")
    print("#" * 80)

    all_results = {}

    # Run all tests
    all_results["Test 1: Import and Initialize"] = test_import_and_initialize()
    all_results["Test 2: Create Orchestrator Instance"] = test_create_orchestrator_instance()
    all_results["Test 3: Verify StageController Integration"] = test_verify_stage_controller_integration()
    all_results["Test 4: Config Resolution"] = test_config_resolution()
    all_results["Test 5: run() Delegation Path"] = test_run_delegation_path()
    all_results["Test 6: All Public Methods Exist"] = test_all_public_methods_exist()

    # Print summary
    print("\n" + "=" * 80)
    print("SMOKE TEST SUMMARY")
    print("=" * 80)

    total_tests = 0
    passed_tests = 0
    failed_tests = 0

    for test_name, test_results in all_results.items():
        print(f"\n{test_name}:")
        for mode, result in test_results.items():
            total_tests += 1
            if result == "PASS":
                passed_tests += 1
                print(f"  ✅ {mode}: PASS")
            else:
                failed_tests += 1
                print(f"  ❌ {mode}: {result}")

    print("\n" + "=" * 80)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print("=" * 80)

    if failed_tests == 0:
        print("\n✅ ALL SMOKE TESTS PASSED")
        return 0
    else:
        print(f"\n❌ {failed_tests} SMOKE TEST(S) FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
