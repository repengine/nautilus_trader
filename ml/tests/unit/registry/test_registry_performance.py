#!/usr/bin/env python3

"""
Performance tests for model registry.

These tests ensure the registry meets performance requirements for production use cases.

"""

from __future__ import annotations

import hashlib
import json
import tempfile
import threading
import time
from pathlib import Path

from ml.registry.base import DataRequirements
from ml.registry.base import ModelManifest
from ml.registry.base import ModelRole
from ml.registry.local_registry import LocalModelRegistry


class TestRegistryPerformance:
    """
    Test registry performance under load.
    """

    def test_registry_bulk_registration_performance(self) -> None:
        """
        Test registry can handle 100 models in under 5 seconds.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            registry_path = Path(tmp_dir)
            registry = LocalModelRegistry(registry_path)

            start_time = time.time()

            # Register 100 models
            for i in range(100):
                # Create minimal ONNX file
                model_path = registry_path / f"model_{i}.onnx"
                model_path.write_bytes(b"ONNX_MODEL_" + str(i).encode())

                # Create manifest
                feature_schema = {f"feature_{j}": "float32" for j in range(10)}
                schema_json = json.dumps(feature_schema, sort_keys=True)
                schema_hash = hashlib.sha256(schema_json.encode()).hexdigest()

                manifest = ModelManifest(
                    model_id=f"perf_model_{i}",
                    role=ModelRole.INFERENCE,
                    data_requirements=DataRequirements.L1_ONLY,
                    architecture="PerformanceTest",
                    feature_schema=feature_schema,
                    feature_schema_hash=schema_hash,
                    performance_metrics={"accuracy": 0.9},
                    version=f"1.0.{i}",
                    created_at=time.time(),
                    last_modified=time.time(),
                )

                registry.register_model(model_path, manifest)

            elapsed = time.time() - start_time

            # Should handle 100 models in under 5 seconds
            assert elapsed < 5.0, f"Registration took {elapsed:.2f}s, expected < 5s"

            # Verify all models registered
            all_models = registry.get_all_models()
            assert len(all_models) == 100

    def test_registry_concurrent_read_performance(self) -> None:
        """
        Test registry handles concurrent reads efficiently.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            registry_path = Path(tmp_dir)
            registry = LocalModelRegistry(registry_path)

            # Register 10 models
            model_ids = []
            for i in range(10):
                model_path = registry_path / f"model_{i}.onnx"
                model_path.write_bytes(b"ONNX_MODEL_" + str(i).encode())

                manifest = ModelManifest(
                    model_id=f"concurrent_model_{i}",
                    role=ModelRole.INFERENCE,
                    data_requirements=DataRequirements.L1_ONLY,
                    architecture="ConcurrentTest",
                    feature_schema={"feature": "float32"},
                    feature_schema_hash=f"hash_{i}",
                    version=f"1.0.{i}",
                    created_at=time.time(),
                    last_modified=time.time(),
                )

                model_id = registry.register_model(model_path, manifest)
                model_ids.append(model_id)

            # Concurrent reads
            read_times = []

            def read_models() -> None:
                """
                Read all models and measure time.
                """
                thread_start = time.time()
                for model_id in model_ids:
                    info = registry.get_model(model_id)
                    assert info is not None
                read_times.append(time.time() - thread_start)

            # Launch 20 concurrent readers
            threads = []
            start_time = time.time()

            for _ in range(20):
                thread = threading.Thread(target=read_models)
                threads.append(thread)
                thread.start()

            # Wait for all threads
            for thread in threads:
                thread.join()

            total_time = time.time() - start_time

            # Should handle 20 concurrent readers in under 1 second
            assert total_time < 1.0, f"Concurrent reads took {total_time:.2f}s"

            # All reads should complete quickly
            avg_read_time = sum(read_times) / len(read_times)
            assert avg_read_time < 0.1, f"Average read time {avg_read_time:.3f}s"

    def test_registry_path_validation_performance(self) -> None:
        """
        Test path validation doesn't significantly impact performance.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            registry_path = Path(tmp_dir)
            registry = LocalModelRegistry(registry_path)

            # Create valid model path
            model_path = registry_path / "model.onnx"
            model_path.write_bytes(b"ONNX_MODEL")

            manifest = ModelManifest(
                model_id="validation_test",
                role=ModelRole.INFERENCE,
                data_requirements=DataRequirements.L1_ONLY,
                architecture="ValidationTest",
                feature_schema={"feature": "float32"},
                feature_schema_hash="test_hash",
                version="1.0.0",
                created_at=time.time(),
                last_modified=time.time(),
            )

            # Measure registration with path validation
            start_time = time.time()
            for _ in range(100):
                # Path validation happens internally
                assert registry._validate_model_path(model_path)
            validation_time = time.time() - start_time

            # Path validation for 100 checks should be under 10ms
            assert validation_time < 0.01, f"Validation took {validation_time*1000:.2f}ms"

    def test_registry_security_rejection_performance(self) -> None:
        """
        Test security checks reject invalid paths quickly.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            registry_path = Path(tmp_dir)
            registry = LocalModelRegistry(registry_path)

            # Test path traversal attempts are rejected quickly
            malicious_paths = [
                Path("../../../etc/passwd"),
                Path("/etc/passwd"),
                Path("../../sensitive.onnx"),
            ]

            start_time = time.time()

            for bad_path in malicious_paths:
                # Should reject immediately
                assert not registry._validate_model_path(bad_path)

            rejection_time = time.time() - start_time

            # Security checks should be near-instant (< 1ms)
            assert rejection_time < 0.001, f"Rejection took {rejection_time*1000:.2f}ms"

    def test_registry_persistence_performance(self) -> None:
        """
        Test registry save/load performance.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            registry_path = Path(tmp_dir)

            # Create registry with 50 models
            registry = LocalModelRegistry(registry_path)

            for i in range(50):
                model_path = registry_path / f"model_{i}.onnx"
                model_path.write_bytes(b"ONNX")

                manifest = ModelManifest(
                    model_id=f"persist_model_{i}",
                    role=ModelRole.INFERENCE,
                    data_requirements=DataRequirements.L1_ONLY,
                    architecture="PersistTest",
                    feature_schema={"f": "float32"},
                    feature_schema_hash=f"h_{i}",
                    version="1.0.0",
                    created_at=time.time(),
                    last_modified=time.time(),
                )

                registry.register_model(model_path, manifest)

            # Flush any pending saves first
            registry.flush()

            # Measure save time
            start_time = time.time()
            registry._save_registry(immediate=True)
            save_time = time.time() - start_time

            # Save should be under 100ms for 50 models
            assert save_time < 0.1, f"Save took {save_time*1000:.2f}ms"

            # Measure load time
            del registry
            start_time = time.time()
            registry2 = LocalModelRegistry(registry_path)
            load_time = time.time() - start_time

            # Load should be under 100ms
            assert load_time < 0.1, f"Load took {load_time*1000:.2f}ms"

            # Verify all models loaded
            assert len(registry2.get_all_models()) == 50

    def test_batch_save_performance(self) -> None:
        """
        Test batch save reduces I/O operations.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            registry_path = Path(tmp_dir)

            # Create registry with 50ms batch interval
            registry = LocalModelRegistry(
                registry_path,
                batch_save_interval=0.05,  # 50ms
            )

            # Track saves by monitoring file modification time
            registry_file = registry_path / "registry.json"
            save_times = []

            def track_saves() -> None:
                """
                Monitor registry file modifications.
                """
                last_mtime = 0.0
                for _ in range(20):  # Check for 1 second
                    if registry_file.exists():
                        mtime = registry_file.stat().st_mtime
                        if mtime != last_mtime:
                            save_times.append(time.time())
                            last_mtime = float(mtime)
                    time.sleep(0.05)

            # Start monitoring in background
            monitor_thread = threading.Thread(target=track_saves)
            monitor_thread.start()

            # Register 10 models rapidly
            start_time = time.time()
            for i in range(10):
                model_path = registry_path / f"model_{i}.onnx"
                model_path.write_bytes(b"ONNX")

                manifest = ModelManifest(
                    model_id=f"batch_model_{i}",
                    role=ModelRole.INFERENCE,
                    data_requirements=DataRequirements.L1_ONLY,
                    architecture="BatchTest",
                    feature_schema={"f": "float32"},
                    feature_schema_hash=f"h_{i}",
                    version="1.0.0",
                    created_at=time.time(),
                    last_modified=time.time(),
                )

                registry.register_model(model_path, manifest)
                time.sleep(0.001)  # 1ms between registrations

            # Wait for batch save to complete
            time.sleep(0.1)

            # Flush to ensure all saves complete
            registry.flush()

            # Stop monitoring
            monitor_thread.join(timeout=1.0)

            # With batching, should have fewer saves than registrations
            # Without batching, we'd have 10 saves (one per registration)
            # With batching, we should have 2-3 saves maximum
            assert len(save_times) <= 3, f"Too many saves: {len(save_times)}, expected <= 3"

            # Verify all models were saved
            assert len(registry.get_all_models()) == 10
