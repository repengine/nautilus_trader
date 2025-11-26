#!/usr/bin/env python3

"""
Integration tests for model registry security and artifact integrity verification.

These tests verify that the ModelRegistry correctly calculates SHA-256 digests during
registration and verifies integrity during model loading.

"""

import hashlib
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from ml.common.security import ArtifactIntegrityError
from ml.registry import DataRequirements
from ml.registry import ModelManifest
from ml.registry import ModelRegistry
from ml.registry import ModelRole

pytestmark = pytest.mark.usefixtures(
    "isolated_prometheus_registry",
    "mock_tracing_backend",
    "isolated_orchestrator_env",
)


class TestModelRegistryIntegrity:
    """
    Test model registry integrity verification functionality.
    """

    @pytest.fixture
    def registry(self, tmp_path: Path) -> ModelRegistry:
        """
        Create a test model registry.
        """
        return ModelRegistry(registry_path=tmp_path)

    @pytest.fixture
    def sample_onnx_model(self, tmp_path: Path) -> tuple[Path, str]:
        """
        Create a sample ONNX model file and return path and digest.
        """
        model_file = tmp_path / "test_model.onnx"
        model_content = b"sample ONNX model content for testing"
        model_file.write_bytes(model_content)
        digest = hashlib.sha256(model_content).hexdigest()
        return model_file, digest

    @pytest.fixture
    def sample_manifest(self) -> ModelManifest:
        """
        Create a sample model manifest.
        """
        return ModelManifest(
            model_id="test_model_001",
            role=ModelRole.INFERENCE,
            data_requirements=DataRequirements.L1_ONLY,
            architecture="test_arch",
            feature_schema={"feature1": "float32", "feature2": "float32"},
            feature_schema_hash="test_hash_123",
            version="1.0.0",
        )

    def test_register_model_calculates_digest(
        self,
        registry: ModelRegistry,
        sample_onnx_model: tuple[Path, str],
        sample_manifest: ModelManifest,
    ) -> None:
        """
        Test that model registration calculates SHA-256 digest.
        """
        model_path, expected_digest = sample_onnx_model

        # Register the model
        model_id = registry.register_model(
            model_path=model_path,
            manifest=sample_manifest,
        )

        # Verify the model was registered
        assert model_id == sample_manifest.model_id

        # Get the registered model info
        model_info = registry.get_model(model_id)
        assert model_info is not None

        # Verify the digest was calculated and stored
        assert model_info.manifest.artifact_sha256_digest == expected_digest

    def test_register_non_onnx_model_no_digest(
        self,
        registry: ModelRegistry,
        sample_manifest: ModelManifest,
        tmp_path: Path,
    ) -> None:
        """
        Test that non-ONNX models don't get digests calculated.
        """
        # Create a non-ONNX model file
        model_file = tmp_path / "test_model.pkl"
        model_file.write_bytes(b"pickle model content")

        # Update manifest to be non-serveable
        sample_manifest.serveable = False

        # Register the model - should work without calculating digest
        model_id = registry.register_model(
            model_path=model_file,
            manifest=sample_manifest,
        )

        # Get the registered model info
        model_info = registry.get_model(model_id)
        assert model_info is not None

        # Verify no digest was calculated for non-ONNX file
        assert model_info.manifest.artifact_sha256_digest is None

    def test_load_model_verifies_integrity(
        self,
        registry: ModelRegistry,
        sample_onnx_model: tuple[Path, str],
        sample_manifest: ModelManifest,
        mock_onnx_runtime: Any,
    ) -> None:
        """
        Test that model loading verifies artifact integrity.
        """
        model_path, _ = sample_onnx_model

        # Register the model first
        model_id = registry.register_model(
            model_path=model_path,
            manifest=sample_manifest,
        )

        mock_session = mock_onnx_runtime.ort.InferenceSession.return_value

        # Load the model - should verify integrity and succeed
        loaded_model = registry.load_model(model_id)

        assert loaded_model == mock_session
        mock_onnx_runtime.ort.InferenceSession.assert_called_once()

    def test_load_model_detects_tampering(
        self,
        registry: ModelRegistry,
        sample_onnx_model: tuple[Path, str],
        sample_manifest: ModelManifest,
    ) -> None:
        """
        Test that model loading detects tampered artifacts.
        """
        model_path, _ = sample_onnx_model

        # Register the model first
        model_id = registry.register_model(
            model_path=model_path,
            manifest=sample_manifest,
        )

        # Tamper with the model file after registration
        tampered_content = b"tampered ONNX model content - this is malicious"
        model_path.write_bytes(tampered_content)

        # Try to load the model - should fail integrity verification
        with pytest.raises(ValueError) as exc_info:
            registry.load_model(model_id)

        error_msg = str(exc_info.value).lower()
        assert "integrity verification failed" in error_msg
        assert "tampered" in error_msg

    def test_load_model_missing_digest_warning(
        self,
        registry: ModelRegistry,
        sample_onnx_model: tuple[Path, str],
        sample_manifest: ModelManifest,
        mock_onnx_runtime: Any,
    ) -> None:
        """
        Test that model loading warns when digest is missing.
        """
        model_path, _ = sample_onnx_model

        # Manually clear the digest to simulate old models without digests
        sample_manifest.artifact_sha256_digest = None

        # Register the model
        model_id = registry.register_model(
            model_path=model_path,
            manifest=sample_manifest,
        )

        # Manually clear the digest from the stored model
        model_info = registry._models[model_id]
        model_info.manifest.artifact_sha256_digest = None

        mock_session = mock_onnx_runtime.ort.InferenceSession.return_value

        # Load the model - should succeed with warning
        loaded_model = registry.load_model(model_id)

        assert loaded_model == mock_session
        mock_onnx_runtime.ort.InferenceSession.assert_called_once()

    def test_register_model_file_not_found(
        self,
        registry: ModelRegistry,
        sample_manifest: ModelManifest,
        tmp_path: Path,
    ) -> None:
        """
        Test that registration fails gracefully for missing files.
        """
        nonexistent_file = tmp_path / "nonexistent_model.onnx"

        with pytest.raises(ValueError, match="Cannot calculate SHA-256 digest"):
            registry.register_model(
                model_path=nonexistent_file,
                manifest=sample_manifest,
            )

    def test_register_model_permission_error(
        self,
        registry: ModelRegistry,
        sample_onnx_model: tuple[Path, str],
        sample_manifest: ModelManifest,
    ) -> None:
        """
        Test that registration handles file permission errors.
        """
        model_path, _ = sample_onnx_model

        # Mock file reading to raise permission error during digest calculation
        # Patch both legacy (model_registry.open) and facade (model_persistence.open)
        with (
            patch("ml.registry.model_registry.open", side_effect=PermissionError("Access denied")),
            patch(
                "ml.registry.common.model_persistence.open",
                side_effect=PermissionError("Access denied"),
            ),
        ):
            with pytest.raises(ValueError, match="Cannot calculate SHA-256 digest"):
                registry.register_model(
                    model_path=model_path,
                    manifest=sample_manifest,
                )

    def test_model_registry_persistence_includes_digest(
        self,
        registry: ModelRegistry,
        sample_onnx_model: tuple[Path, str],
        sample_manifest: ModelManifest,
    ) -> None:
        """
        Test that model registry persistence includes digest information.
        """
        model_path, expected_digest = sample_onnx_model

        # Register the model
        model_id = registry.register_model(
            model_path=model_path,
            manifest=sample_manifest,
        )

        # Force save to disk
        registry._save_registry(immediate=True)

        # Create a new registry instance and verify digest is persisted
        new_registry = ModelRegistry(registry_path=registry.registry_path)
        model_info = new_registry.get_model(model_id)

        assert model_info is not None
        assert model_info.manifest.artifact_sha256_digest == expected_digest

    def test_integrity_verification_bypass_non_onnx(
        self,
        registry: ModelRegistry,
        sample_manifest: ModelManifest,
        tmp_path: Path,
    ) -> None:
        """
        Test that integrity verification is bypassed for non-ONNX files.
        """
        # Create a non-ONNX model file
        model_file = tmp_path / "test_model.joblib"
        model_file.write_bytes(b"joblib model content")

        # Update manifest to be non-serveable
        sample_manifest.serveable = False

        # Register the model
        model_id = registry.register_model(
            model_path=model_file,
            manifest=sample_manifest,
        )

        # Try to load the model - should return None for non-ONNX files
        loaded_model = registry.load_model(model_id)
        assert loaded_model is None


class TestModelRegistrySecurityEdgeCases:
    """
    Test edge cases and boundary conditions for security features.
    """

    @pytest.fixture
    def registry(self, tmp_path: Path) -> ModelRegistry:
        """
        Create a test model registry.
        """
        return ModelRegistry(registry_path=tmp_path)

    @pytest.fixture
    def sample_onnx_model(self, tmp_path: Path) -> tuple[Path, str]:
        """
        Create a sample ONNX model file and return path and digest.
        """
        model_file = tmp_path / "test_model.onnx"
        model_content = b"sample ONNX model content for testing"
        model_file.write_bytes(model_content)
        digest = hashlib.sha256(model_content).hexdigest()
        return model_file, digest

    @pytest.fixture
    def sample_manifest(self) -> ModelManifest:
        """
        Create a sample model manifest.
        """
        return ModelManifest(
            model_id="test_model_001",
            role=ModelRole.INFERENCE,
            data_requirements=DataRequirements.L1_ONLY,
            architecture="test_arch",
            feature_schema={"feature1": "float32", "feature2": "float32"},
            feature_schema_hash="test_hash_123",
            version="1.0.0",
        )

    def test_digest_calculation_large_file(
        self,
        registry: ModelRegistry,
        tmp_path: Path,
    ) -> None:
        """
        Test digest calculation for large model files.
        """
        # Create a large model file (>1MB to test chunked reading)
        model_file = tmp_path / "large_model.onnx"
        large_content = b"A" * (1024 * 1024 + 1000)  # 1MB + 1000 bytes
        model_file.write_bytes(large_content)

        expected_digest = hashlib.sha256(large_content).hexdigest()

        # Test digest calculation
        actual_digest = registry._calculate_file_sha256(model_file)
        assert actual_digest == expected_digest

    def test_concurrent_model_registration(
        self,
        registry: ModelRegistry,
        tmp_path: Path,
    ) -> None:
        """
        Test that concurrent model registration maintains integrity.
        """
        import threading
        from concurrent.futures import ThreadPoolExecutor

        # Create multiple model files
        model_files = []
        manifests = []
        expected_digests = []

        for i in range(5):
            model_file = tmp_path / f"concurrent_model_{i}.onnx"
            content = f"concurrent model {i} content".encode()
            model_file.write_bytes(content)

            manifest = ModelManifest(
                model_id=f"concurrent_model_{i}",
                role=ModelRole.INFERENCE,
                data_requirements=DataRequirements.L1_ONLY,
                architecture="test_arch",
                feature_schema={"feature1": "float32"},
                feature_schema_hash=f"hash_{i}",
                version="1.0.0",
            )

            model_files.append(model_file)
            manifests.append(manifest)
            expected_digests.append(hashlib.sha256(content).hexdigest())

        # Register models concurrently
        def register_model(i):
            return registry.register_model(
                model_path=model_files[i],
                manifest=manifests[i],
            )

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(register_model, i) for i in range(5)]
            model_ids = [future.result() for future in futures]

        # Verify all models were registered with correct digests
        for i, model_id in enumerate(model_ids):
            model_info = registry.get_model(model_id)
            assert model_info is not None
            assert model_info.manifest.artifact_sha256_digest == expected_digests[i]

    def test_registry_corruption_detection(
        self,
        registry: ModelRegistry,
        tmp_path: Path,
    ) -> None:
        """
        Test detection of registry data corruption.
        """
        # Create and register a model
        model_file = tmp_path / "test_model.onnx"
        content = b"test model content"
        model_file.write_bytes(content)

        manifest = ModelManifest(
            model_id="test_model",
            role=ModelRole.INFERENCE,
            data_requirements=DataRequirements.L1_ONLY,
            architecture="test_arch",
            feature_schema={"feature1": "float32"},
            feature_schema_hash="test_hash",
            version="1.0.0",
        )

        model_id = registry.register_model(model_path=model_file, manifest=manifest)

        # Simulate registry corruption by manually changing the stored digest
        model_info = registry._models[model_id]
        model_info.manifest.artifact_sha256_digest = "corrupted_digest_value"

        # Try to load the model - should detect the mismatch
        with pytest.raises(ValueError) as exc_info:
            registry.load_model(model_id)

        assert "integrity verification failed" in str(exc_info.value).lower()

    def test_security_logging_and_auditing(
        self,
        registry: ModelRegistry,
        sample_onnx_model: tuple[Path, str],
        sample_manifest: ModelManifest,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """
        Test that security events are properly logged for auditing.
        """
        model_path, _ = sample_onnx_model

        # Register the model
        model_id = registry.register_model(
            model_path=model_path,
            manifest=sample_manifest,
        )

        # Clear logs
        caplog.clear()

        # Tamper with the model file
        model_path.write_bytes(b"tampered content")

        # Try to load the model - should log security alert
        with pytest.raises(ValueError):
            registry.load_model(model_id)

        # Verify security alert was logged
        security_logs = [record for record in caplog.records if "SECURITY ALERT" in record.message]
        assert len(security_logs) > 0

        security_log = security_logs[0]
        assert "integrity verification failed" in security_log.message.lower()
        assert "tampered" in security_log.message.lower()
        assert security_log.levelname == "ERROR"
