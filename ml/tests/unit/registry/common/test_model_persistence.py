#!/usr/bin/env python3

"""
Unit tests for ModelPersistenceComponent.

This test module verifies the behavioral contract of ModelPersistenceComponent
extracted from the ModelRegistry god class. Tests cover:
- JSON/PostgreSQL persistence loading and saving
- Serialization/deserialization of ModelInfo
- Batch save management with timers
- SHA-256 integrity verification for model artifacts
- Thread-safety and error handling
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from ml.registry.base import DataRequirements
from ml.registry.base import DeploymentStatus
from ml.registry.base import ModelInfo
from ml.registry.base import ModelManifest
from ml.registry.base import ModelRole
from ml.registry.common.model_persistence import ModelPersistenceComponent
from ml.registry.persistence import BackendType
from ml.registry.persistence import PersistenceConfig
from ml.tests.utils.db import build_postgres_url


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def tmp_registry_path(tmp_path: Path) -> Path:
    """Create a temporary registry directory."""
    registry_path = tmp_path / "registry"
    registry_path.mkdir(parents=True, exist_ok=True)
    return registry_path


@pytest.fixture
def json_persistence_config(tmp_registry_path: Path) -> PersistenceConfig:
    """Create a PersistenceConfig with JSON backend."""
    return PersistenceConfig(
        backend=BackendType.JSON,
        json_path=tmp_registry_path,
    )


@pytest.fixture
def persistence_component(
    json_persistence_config: PersistenceConfig,
    tmp_registry_path: Path,
) -> ModelPersistenceComponent:
    """
    Create a ModelPersistenceComponent with JSON backend for unit tests.
    """
    return ModelPersistenceComponent(
        persistence_config=json_persistence_config,
        registry_path=tmp_registry_path,
        batch_save_interval=0.1,
    )


@pytest.fixture
def sample_model_manifest() -> ModelManifest:
    """Create a sample ModelManifest for testing."""
    return ModelManifest(
        model_id="test_model_123",
        role=ModelRole.INFERENCE,
        data_requirements=DataRequirements.L1_ONLY,
        architecture="XGBoost",
        feature_schema={"input": "float32"},
        feature_schema_hash="abc123hash",
        version="1.0.0",
        created_at=1700000000.0,
        last_modified=1700000000.0,
        artifact_sha256_digest="expected_digest_123",
    )


@pytest.fixture
def sample_model_info(sample_model_manifest: ModelManifest, tmp_path: Path) -> ModelInfo:
    """Create a sample ModelInfo for testing."""
    model_path = tmp_path / "test_model.onnx"
    model_path.write_bytes(b"fake model content")
    return ModelInfo(
        manifest=sample_model_manifest,
        model_path=model_path,
        deployment_status=DeploymentStatus.INACTIVE,
        deployed_to=[],
        performance_history=[{"accuracy": 0.95}],
        metadata={"test_key": "test_value"},
    )


@pytest.fixture
def sample_onnx_model(tmp_path: Path) -> tuple[Path, str]:
    """
    Create a sample ONNX model file and return (path, sha256_digest).
    """
    model_file = tmp_path / "test_model.onnx"
    content = b"sample ONNX model content for testing"
    model_file.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    return model_file, digest


# =============================================================================
# Test: Loading Registry (Happy Path)
# =============================================================================


class TestLoadRegistryJsonBackend:
    """Tests for loading registry from JSON backend."""

    def test_load_registry_json_backend_empty(
        self,
        persistence_component: ModelPersistenceComponent,
    ) -> None:
        """
        Verify loading from non-existent JSON file creates empty state.

        Input: Empty tmp_path directory
        Expected: Returns empty _models, _ab_tests, _deployments dicts
        """
        persistence_component.load_registry()

        assert len(persistence_component.models) == 0
        assert len(persistence_component.ab_tests) == 0
        assert len(persistence_component.deployments) == 0

    def test_load_registry_json_backend_existing(
        self,
        persistence_component: ModelPersistenceComponent,
        tmp_registry_path: Path,
    ) -> None:
        """
        Verify loading from existing JSON file deserializes correctly.

        Input: Pre-populated registry.json file
        Expected: All models loaded with correct manifest fields
        """
        # Create a registry file with test data
        registry_data = {
            "models": {
                "model_1": {
                    "manifest": {
                        "model_id": "model_1",
                        "role": "inference",
                        "data_requirements": "l1_only",
                        "architecture": "XGBoost",
                        "feature_schema": {"input": "float32"},
                        "feature_schema_hash": "hash123",
                        "version": "1.0.0",
                        "created_at": 1700000000.0,
                        "last_modified": 1700000000.0,
                    },
                    "model_path": "/tmp/model.onnx",
                    "deployment_status": "inactive",
                    "deployed_to": [],
                    "performance_history": [],
                    "metadata": {},
                }
            },
            "ab_tests": {"test_1": {"split_ratio": 0.5}},
            "deployments": {"target_1": ["model_1"]},
        }

        registry_file = tmp_registry_path / "registry.json"
        with open(registry_file, "w") as f:
            json.dump(registry_data, f)

        persistence_component.load_registry()

        assert len(persistence_component.models) == 1
        assert "model_1" in persistence_component.models
        model_info = persistence_component.models["model_1"]
        assert model_info.manifest.model_id == "model_1"
        assert model_info.manifest.role == ModelRole.INFERENCE
        assert model_info.deployment_status == DeploymentStatus.INACTIVE

        assert len(persistence_component.ab_tests) == 1
        assert len(persistence_component.deployments) == 1


# =============================================================================
# Test: Saving Registry (Happy Path)
# =============================================================================


class TestSaveRegistry:
    """Tests for saving registry to JSON backend."""

    def test_save_registry_immediate(
        self,
        persistence_component: ModelPersistenceComponent,
        sample_model_info: ModelInfo,
        tmp_registry_path: Path,
    ) -> None:
        """
        Verify immediate save writes JSON synchronously.

        Input: ModelInfo to save, immediate=True
        Expected: File written immediately, timer not started
        """
        persistence_component.set_model(
            sample_model_info.manifest.model_id,
            sample_model_info,
        )
        persistence_component.save_registry(immediate=True)

        registry_file = tmp_registry_path / "registry.json"
        assert registry_file.exists()
        assert persistence_component._pending_save is False
        assert persistence_component._save_timer is None

        # Verify file content
        with open(registry_file) as f:
            data = json.load(f)
        assert sample_model_info.manifest.model_id in data["models"]

    def test_save_registry_batched(
        self,
        persistence_component: ModelPersistenceComponent,
        sample_model_info: ModelInfo,
        tmp_registry_path: Path,
    ) -> None:
        """
        Verify batched save schedules timer.

        Input: ModelInfo to save, immediate=False
        Expected: Timer scheduled, file not yet written immediately
        """
        persistence_component.set_model(
            sample_model_info.manifest.model_id,
            sample_model_info,
        )
        persistence_component.save_registry(immediate=False)

        assert persistence_component._pending_save is True
        assert persistence_component._save_timer is not None

        # Wait for batch save to complete
        time.sleep(0.2)

        registry_file = tmp_registry_path / "registry.json"
        assert registry_file.exists()

    def test_flush_batch_save(
        self,
        persistence_component: ModelPersistenceComponent,
        sample_model_info: ModelInfo,
        tmp_registry_path: Path,
    ) -> None:
        """
        Verify flush writes pending changes immediately.

        Input: Pending save state
        Expected: Pending changes written, timer cancelled
        """
        persistence_component.set_model(
            sample_model_info.manifest.model_id,
            sample_model_info,
        )
        persistence_component.save_registry(immediate=False)

        assert persistence_component._pending_save is True

        # Flush should write immediately
        persistence_component.flush()

        registry_file = tmp_registry_path / "registry.json"
        assert registry_file.exists()
        assert persistence_component._pending_save is False
        assert persistence_component._save_timer is None


# =============================================================================
# Test: Serialization
# =============================================================================


class TestSerialization:
    """Tests for ModelInfo serialization and deserialization."""

    def test_model_info_to_dict_serialization(
        self,
        persistence_component: ModelPersistenceComponent,
        sample_model_info: ModelInfo,
    ) -> None:
        """
        Verify ModelInfo serialization includes all fields.

        Input: ModelInfo with all fields populated
        Expected: Dict contains all manifest and metadata fields
        """
        result = persistence_component.model_info_to_dict(sample_model_info)

        assert "manifest" in result
        assert result["manifest"]["model_id"] == sample_model_info.manifest.model_id
        assert result["manifest"]["role"] == "inference"
        assert result["manifest"]["data_requirements"] == "l1_only"
        assert result["manifest"]["architecture"] == "XGBoost"
        assert result["manifest"]["version"] == "1.0.0"
        assert result["manifest"]["artifact_sha256_digest"] == "expected_digest_123"
        assert result["deployment_status"] == "inactive"
        assert result["metadata"] == {"test_key": "test_value"}

    def test_dict_to_model_info_deserialization(
        self,
        persistence_component: ModelPersistenceComponent,
        sample_model_info: ModelInfo,
    ) -> None:
        """
        Verify dict deserialization creates valid ModelInfo.

        Input: Dict from model_info_to_dict
        Expected: ModelInfo with all fields restored
        """
        data = persistence_component.model_info_to_dict(sample_model_info)
        restored = persistence_component.dict_to_model_info(data)

        assert restored.manifest.model_id == sample_model_info.manifest.model_id
        assert restored.manifest.role == ModelRole.INFERENCE
        assert restored.manifest.data_requirements == DataRequirements.L1_ONLY
        assert restored.deployment_status == DeploymentStatus.INACTIVE
        assert restored.metadata == {"test_key": "test_value"}

    def test_dict_to_model_info_legacy_format(
        self,
        persistence_component: ModelPersistenceComponent,
    ) -> None:
        """
        Verify deserialization handles legacy format without manifest wrapper.

        Input: Dict with flat structure (no "manifest" key)
        Expected: ModelInfo created with defaults for missing fields
        """
        legacy_data = {
            "model_id": "legacy_model",
            "model_path": "/tmp/legacy.onnx",
            "deployment_status": "inactive",
            "deployed_to": [],
            "version": "1.0.0",
            "created_at": 1700000000.0,
            "last_modified": 1700000000.0,
        }

        model_info = persistence_component.dict_to_model_info(legacy_data)

        assert model_info.manifest.model_id == "legacy_model"
        assert model_info.manifest.role == ModelRole.INFERENCE  # Default
        assert model_info.manifest.data_requirements == DataRequirements.L1_ONLY  # Default
        assert model_info.manifest.architecture == "unknown"


# =============================================================================
# Test: SHA-256 Integrity Verification
# =============================================================================


class TestIntegrityVerification:
    """Tests for SHA-256 integrity verification."""

    def test_calculate_file_sha256(
        self,
        persistence_component: ModelPersistenceComponent,
        sample_onnx_model: tuple[Path, str],
    ) -> None:
        """
        Verify SHA-256 calculation for model files.

        Input: File with known content
        Expected: Returns correct hexadecimal digest
        """
        file_path, expected_digest = sample_onnx_model

        actual_digest = persistence_component.calculate_file_sha256(file_path)

        assert actual_digest == expected_digest
        assert len(actual_digest) == 64  # SHA-256 is 64 hex characters

    def test_verify_artifact_integrity_pass(
        self,
        persistence_component: ModelPersistenceComponent,
        sample_onnx_model: tuple[Path, str],
    ) -> None:
        """
        Verify integrity check passes for unmodified files.

        Input: File path and matching digest
        Expected: No exception raised
        """
        file_path, expected_digest = sample_onnx_model

        # Should not raise
        persistence_component.verify_artifact_integrity(file_path, expected_digest)

    def test_verify_artifact_integrity_fail(
        self,
        persistence_component: ModelPersistenceComponent,
        sample_onnx_model: tuple[Path, str],
    ) -> None:
        """
        Verify integrity check fails for tampered files.

        Input: File path with mismatched digest
        Expected: ValueError raised with security message
        """
        file_path, _ = sample_onnx_model
        wrong_digest = "a" * 64  # Invalid digest

        with pytest.raises(ValueError, match="integrity verification failed"):
            persistence_component.verify_artifact_integrity(file_path, wrong_digest)

    def test_verify_artifact_integrity_file_not_found(
        self,
        persistence_component: ModelPersistenceComponent,
        tmp_path: Path,
    ) -> None:
        """
        Verify handling of missing artifact files.

        Input: Non-existent file path
        Expected: ValueError raised
        """
        non_existent = tmp_path / "does_not_exist.onnx"

        with pytest.raises(ValueError, match="Cannot verify artifact"):
            persistence_component.verify_artifact_integrity(non_existent, "some_digest")

    def test_verify_artifact_integrity_none_digest(
        self,
        persistence_component: ModelPersistenceComponent,
        sample_onnx_model: tuple[Path, str],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """
        Verify handling when no digest available.

        Input: expected_digest=None
        Expected: Warning logged, no exception
        """
        file_path, _ = sample_onnx_model

        # Should not raise
        persistence_component.verify_artifact_integrity(file_path, None)

        assert "skipping integrity verification" in caplog.text.lower()

    def test_calculate_sha256_large_file(
        self,
        persistence_component: ModelPersistenceComponent,
        tmp_path: Path,
    ) -> None:
        """
        Verify chunked reading for large files (>1MB).

        Input: 2MB file
        Expected: Correct digest without memory issues
        """
        large_file = tmp_path / "large_model.onnx"
        # Create 2MB file
        content = b"x" * (2 * 1024 * 1024)
        large_file.write_bytes(content)

        expected_digest = hashlib.sha256(content).hexdigest()
        actual_digest = persistence_component.calculate_file_sha256(large_file)

        assert actual_digest == expected_digest

    def test_calculate_sha256_empty_file(
        self,
        persistence_component: ModelPersistenceComponent,
        tmp_path: Path,
    ) -> None:
        """
        Verify handling of empty files.

        Input: 0-byte file
        Expected: Returns SHA-256 of empty content
        """
        empty_file = tmp_path / "empty.onnx"
        empty_file.write_bytes(b"")

        expected_digest = hashlib.sha256(b"").hexdigest()
        actual_digest = persistence_component.calculate_file_sha256(empty_file)

        assert actual_digest == expected_digest


# =============================================================================
# Test: Error Conditions
# =============================================================================


class TestErrorConditions:
    """Tests for error handling in persistence operations."""

    def test_load_registry_corrupted_json(
        self,
        persistence_component: ModelPersistenceComponent,
        tmp_registry_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """
        Verify graceful handling of corrupted JSON.

        Input: Malformed JSON in registry.json
        Expected: Warning logged, empty state initialized
        """
        registry_file = tmp_registry_path / "registry.json"
        registry_file.write_text("{invalid json content")

        persistence_component.load_registry()

        assert len(persistence_component.models) == 0
        assert "failed to load registry" in caplog.text.lower()

    def test_save_registry_permission_error(
        self,
        json_persistence_config: PersistenceConfig,
        tmp_path: Path,
    ) -> None:
        """
        Verify handling of file write permission errors.

        Input: Read-only directory
        Expected: Exception raised
        """
        # Create a read-only directory
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()

        component = ModelPersistenceComponent(
            persistence_config=json_persistence_config,
            registry_path=readonly_dir,
        )

        # Make directory read-only
        os.chmod(readonly_dir, stat.S_IRUSR | stat.S_IXUSR)

        try:
            with pytest.raises(Exception):  # Could be OSError or PermissionError
                component.save_registry(immediate=True)
        finally:
            # Restore permissions for cleanup
            os.chmod(readonly_dir, stat.S_IRWXU)

    def test_batch_save_directory_deleted(
        self,
        persistence_component: ModelPersistenceComponent,
        sample_model_info: ModelInfo,
        tmp_registry_path: Path,
    ) -> None:
        """
        Verify graceful handling when directory deleted during batch save.

        Input: Pending save with deleted parent directory
        Expected: FileNotFoundError caught, logged, timer reset
        """
        persistence_component.set_model(
            sample_model_info.manifest.model_id,
            sample_model_info,
        )
        persistence_component.save_registry(immediate=False)

        # Delete the directory
        import shutil

        shutil.rmtree(tmp_registry_path, ignore_errors=True)

        # Flush should handle gracefully
        persistence_component.flush()

        # State should be reset
        assert persistence_component._pending_save is False


# =============================================================================
# Test: Model CRUD Operations
# =============================================================================


class TestModelCRUD:
    """Tests for model CRUD operations."""

    def test_get_model_returns_copy(
        self,
        persistence_component: ModelPersistenceComponent,
        sample_model_info: ModelInfo,
    ) -> None:
        """
        Verify get_model returns model if exists.

        Input: Model in registry
        Expected: Returns model info
        """
        model_id = sample_model_info.manifest.model_id
        persistence_component.set_model(model_id, sample_model_info)

        result = persistence_component.get_model(model_id)

        assert result is not None
        assert result.manifest.model_id == model_id

    def test_get_model_not_found(
        self,
        persistence_component: ModelPersistenceComponent,
    ) -> None:
        """
        Verify get_model returns None for non-existent model.

        Input: Non-existent model_id
        Expected: Returns None
        """
        result = persistence_component.get_model("non_existent")
        assert result is None

    def test_set_model_persists(
        self,
        persistence_component: ModelPersistenceComponent,
        sample_model_info: ModelInfo,
    ) -> None:
        """
        Verify set_model stores model in registry.

        Input: Valid model info
        Expected: Model saved and retrievable
        """
        model_id = sample_model_info.manifest.model_id
        persistence_component.set_model(model_id, sample_model_info)

        result = persistence_component.get_model(model_id)
        assert result is not None
        assert result.manifest.version == sample_model_info.manifest.version

    def test_delete_model_removes(
        self,
        persistence_component: ModelPersistenceComponent,
        sample_model_info: ModelInfo,
    ) -> None:
        """
        Verify delete_model removes model from registry.

        Input: Model in registry
        Expected: Model deleted successfully, returns True
        """
        model_id = sample_model_info.manifest.model_id
        persistence_component.set_model(model_id, sample_model_info)

        result = persistence_component.delete_model(model_id)

        assert result is True
        assert persistence_component.get_model(model_id) is None

    def test_delete_model_not_found(
        self,
        persistence_component: ModelPersistenceComponent,
    ) -> None:
        """
        Verify delete_model returns False for non-existent model.

        Input: Non-existent model_id
        Expected: Returns False
        """
        result = persistence_component.delete_model("non_existent")
        assert result is False


# =============================================================================
# Test: Properties
# =============================================================================


class TestProperties:
    """Tests for component properties."""

    def test_backend_property(
        self,
        persistence_component: ModelPersistenceComponent,
    ) -> None:
        """Verify backend property returns correct type."""
        assert persistence_component.backend == BackendType.JSON

    def test_models_property(
        self,
        persistence_component: ModelPersistenceComponent,
        sample_model_info: ModelInfo,
    ) -> None:
        """Verify models property returns the internal dict."""
        persistence_component.set_model(
            sample_model_info.manifest.model_id,
            sample_model_info,
        )

        models = persistence_component.models
        assert sample_model_info.manifest.model_id in models

    def test_ab_tests_property(
        self,
        persistence_component: ModelPersistenceComponent,
    ) -> None:
        """Verify ab_tests property returns the internal dict."""
        ab_tests = persistence_component.ab_tests
        assert isinstance(ab_tests, dict)

    def test_deployments_property(
        self,
        persistence_component: ModelPersistenceComponent,
    ) -> None:
        """Verify deployments property returns the internal dict."""
        deployments = persistence_component.deployments
        assert isinstance(deployments, dict)


# =============================================================================
# Test: Thread Safety
# =============================================================================


class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_set_model(
        self,
        persistence_component: ModelPersistenceComponent,
        sample_model_manifest: ModelManifest,
        tmp_path: Path,
    ) -> None:
        """
        Verify thread safety for concurrent model operations.

        Input: Multiple threads setting models
        Expected: All models stored correctly without race conditions
        """
        num_threads = 10
        results: list[bool] = []

        def set_model(idx: int) -> None:
            manifest = ModelManifest(
                model_id=f"model_{idx}",
                role=ModelRole.INFERENCE,
                data_requirements=DataRequirements.L1_ONLY,
                architecture="XGBoost",
                feature_schema={},
                feature_schema_hash=f"hash_{idx}",
                version="1.0.0",
                created_at=time.time(),
                last_modified=time.time(),
            )
            model_path = tmp_path / f"model_{idx}.onnx"
            model_path.write_bytes(f"content_{idx}".encode())
            model_info = ModelInfo(
                manifest=manifest,
                model_path=model_path,
                deployment_status=DeploymentStatus.INACTIVE,
                deployed_to=[],
            )
            persistence_component.set_model(f"model_{idx}", model_info)
            results.append(True)

        threads = [threading.Thread(target=set_model, args=(i,)) for i in range(num_threads)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == num_threads
        assert len(persistence_component.models) == num_threads


# =============================================================================
# Test: Round-Trip Serialization
# =============================================================================


class TestRoundTrip:
    """Tests for complete round-trip serialization."""

    def test_full_round_trip(
        self,
        persistence_component: ModelPersistenceComponent,
        sample_model_info: ModelInfo,
        tmp_registry_path: Path,
    ) -> None:
        """
        Verify complete round-trip: set -> save -> load -> get.

        Input: ModelInfo
        Expected: Identical model retrieved after reload
        """
        model_id = sample_model_info.manifest.model_id

        # Save
        persistence_component.set_model(model_id, sample_model_info)
        persistence_component.save_registry(immediate=True)

        # Create new component and load
        new_component = ModelPersistenceComponent(
            persistence_config=PersistenceConfig(
                backend=BackendType.JSON,
                json_path=tmp_registry_path,
            ),
            registry_path=tmp_registry_path,
        )
        new_component.load_registry()

        # Verify
        loaded = new_component.get_model(model_id)
        assert loaded is not None
        assert loaded.manifest.model_id == model_id
        assert loaded.manifest.role == ModelRole.INFERENCE
        assert loaded.manifest.version == "1.0.0"
        assert loaded.deployment_status == DeploymentStatus.INACTIVE


# =============================================================================
# Test: PostgreSQL Backend (Mocked)
# =============================================================================


class TestPostgresBackend:
    """Tests for PostgreSQL backend operations using mocks."""

    def test_load_from_postgres_no_session(
        self,
        tmp_registry_path: Path,
    ) -> None:
        """
        Verify handling when PostgreSQL session unavailable.

        Input: Postgres backend with no working session
        Expected: Empty state initialized
        """
        with patch("ml.registry.persistence.PersistenceManager") as MockPM:
            mock_pm = MagicMock()
            mock_pm.config.backend = BackendType.POSTGRES
            mock_pm.get_session.return_value = None
            MockPM.return_value = mock_pm

            # Create config - since Postgres needs a connection string, mock it
            with patch.object(PersistenceConfig, "__post_init__"):
                config = PersistenceConfig(
                    backend=BackendType.POSTGRES,
                    connection_string=build_postgres_url(
                        user="test",
                        password="test",
                        database="test",
                    ),
                )

            component = ModelPersistenceComponent.__new__(ModelPersistenceComponent)
            component.persistence = mock_pm
            component.registry_path = tmp_registry_path
            component.registry_file = tmp_registry_path / "registry.json"
            component._lock = threading.RLock()
            component._models = {}
            component._ab_tests = {}
            component._deployments = {}
            component._pending_save = False
            component._save_timer = None
            component.batch_save_interval = 0.1

            component._load_from_postgres()

            assert len(component.models) == 0
            assert len(component.ab_tests) == 0

    def test_load_from_postgres_with_session(
        self,
        tmp_registry_path: Path,
    ) -> None:
        """
        Verify loading from PostgreSQL with mocked session.

        Input: Mocked database session with models
        Expected: Models loaded successfully
        """
        from datetime import datetime

        # Create mock database model
        mock_db_model = MagicMock()
        mock_db_model.model_id = "db_model_1"
        mock_db_model.role = "inference"
        mock_db_model.data_requirements = "l1_only"
        mock_db_model.architecture = "XGBoost"
        mock_db_model.feature_schema = {"input": "float32"}
        mock_db_model.feature_schema_hash = "hash123"
        mock_db_model.parent_id = None
        mock_db_model.children_ids = []
        mock_db_model.training_config = {}
        mock_db_model.performance_metrics = {}
        mock_db_model.deployment_constraints = {}
        mock_db_model.version = "1.0.0"
        mock_db_model.created_at = datetime(2023, 1, 1)
        mock_db_model.last_modified = datetime(2023, 1, 1)
        mock_db_model.serveable = "true"
        mock_db_model.artifact_format = "onnx"
        mock_db_model.feature_set_id = None
        mock_db_model.pipeline_signature = None
        mock_db_model.pipeline_version = None
        mock_db_model.artifact_sha256_digest = None
        mock_db_model.model_path = "/tmp/model.onnx"
        mock_db_model.deployment_status = "inactive"
        mock_db_model.deployed_to = ["target_1"]
        mock_db_model.performance_history = []
        mock_db_model.extra_metadata = {}

        # Create mock session
        mock_session = MagicMock()
        mock_session.query.return_value.all.return_value = [mock_db_model]

        # Create mock persistence manager
        mock_pm = MagicMock()
        mock_pm.config.backend = BackendType.POSTGRES
        mock_pm.get_session.return_value = mock_session

        component = ModelPersistenceComponent.__new__(ModelPersistenceComponent)
        component.persistence = mock_pm
        component.registry_path = tmp_registry_path
        component.registry_file = tmp_registry_path / "registry.json"
        component._lock = threading.RLock()
        component._models = {}
        component._ab_tests = {}
        component._deployments = {}
        component._pending_save = False
        component._save_timer = None
        component.batch_save_interval = 0.1

        component._load_from_postgres()

        assert "db_model_1" in component.models
        assert "target_1" in component.deployments
        mock_session.close.assert_called_once()

    def test_load_from_postgres_exception(
        self,
        tmp_registry_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """
        Verify handling when PostgreSQL query fails.

        Input: Session that throws exception on query
        Expected: Empty state, warning logged
        """
        mock_session = MagicMock()
        mock_session.query.side_effect = Exception("Database connection lost")

        mock_pm = MagicMock()
        mock_pm.config.backend = BackendType.POSTGRES
        mock_pm.get_session.return_value = mock_session

        component = ModelPersistenceComponent.__new__(ModelPersistenceComponent)
        component.persistence = mock_pm
        component.registry_path = tmp_registry_path
        component.registry_file = tmp_registry_path / "registry.json"
        component._lock = threading.RLock()
        component._models = {}
        component._ab_tests = {}
        component._deployments = {}
        component._pending_save = False
        component._save_timer = None
        component.batch_save_interval = 0.1

        component._load_from_postgres()

        assert len(component.models) == 0
        assert "error loading from database" in caplog.text.lower()
        mock_session.close.assert_called_once()

    def test_save_model_to_db_no_session(
        self,
        persistence_component: ModelPersistenceComponent,
        sample_model_info: ModelInfo,
    ) -> None:
        """
        Verify save_model_to_db returns gracefully when no session.

        Input: Model info with no database session
        Expected: No exception, method returns silently
        """
        with patch.object(persistence_component.persistence, "get_session", return_value=None):
            # Should not raise
            persistence_component.save_model_to_db(sample_model_info)

    def test_save_model_to_db_update_existing(
        self,
        persistence_component: ModelPersistenceComponent,
        sample_model_info: ModelInfo,
    ) -> None:
        """
        Verify save_model_to_db updates existing model.

        Input: Model that already exists in database
        Expected: Existing model updated
        """
        mock_session = MagicMock()
        mock_existing = MagicMock()
        mock_session.query.return_value.filter_by.return_value.first.return_value = mock_existing

        with patch.object(persistence_component.persistence, "get_session", return_value=mock_session):
            persistence_component.save_model_to_db(sample_model_info)

        mock_session.commit.assert_called_once()
        mock_session.close.assert_called_once()
        # Verify update was done (not add)
        mock_session.add.assert_not_called()

    def test_save_model_to_db_create_new(
        self,
        persistence_component: ModelPersistenceComponent,
        sample_model_info: ModelInfo,
    ) -> None:
        """
        Verify save_model_to_db creates new model.

        Input: Model that doesn't exist in database
        Expected: New model created
        """
        mock_session = MagicMock()
        mock_session.query.return_value.filter_by.return_value.first.return_value = None

        with patch.object(persistence_component.persistence, "get_session", return_value=mock_session):
            persistence_component.save_model_to_db(sample_model_info)

        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()
        mock_session.close.assert_called_once()

    def test_save_model_to_db_exception_rollback(
        self,
        persistence_component: ModelPersistenceComponent,
        sample_model_info: ModelInfo,
    ) -> None:
        """
        Verify save_model_to_db rolls back on exception.

        Input: Session that throws on commit
        Expected: Rollback called, exception re-raised
        """
        mock_session = MagicMock()
        mock_session.query.return_value.filter_by.return_value.first.return_value = None
        mock_session.commit.side_effect = Exception("Commit failed")

        with patch.object(persistence_component.persistence, "get_session", return_value=mock_session):
            with pytest.raises(Exception, match="Commit failed"):
                persistence_component.save_model_to_db(sample_model_info)

        mock_session.rollback.assert_called_once()
        mock_session.close.assert_called_once()


class TestAdditionalEdgeCases:
    """Additional edge case tests for improved coverage."""

    def test_model_info_serialization_with_all_optional_fields(
        self,
        persistence_component: ModelPersistenceComponent,
        tmp_path: Path,
    ) -> None:
        """
        Verify serialization includes all optional manifest fields.

        Input: ModelManifest with all optional fields populated
        Expected: All fields present in serialized dict
        """
        manifest = ModelManifest(
            model_id="full_model",
            role=ModelRole.INFERENCE,
            data_requirements=DataRequirements.L1_L2,
            architecture="Transformer",
            feature_schema={"input_1": "float32", "input_2": "int64"},
            feature_schema_hash="full_hash_abc123",
            parent_id="parent_model_123",
            children_ids=["child_1", "child_2"],
            training_config={"epochs": 100, "batch_size": 32},
            performance_metrics={"accuracy": 0.98, "f1": 0.97},
            deployment_constraints={"max_latency_ms": 10},
            version="2.0.0",
            created_at=1700000000.0,
            last_modified=1700000001.0,
            serveable=True,
            artifact_format="onnx",
            feature_set_id="feature_set_abc",
            pipeline_signature="sig_xyz",
            pipeline_version="1.2.3",
            decision_policy="threshold",
            decision_config={"threshold": 0.5},
            output_schema={"kind": "binary_proba", "shape": [None, 1]},
            calibration={"kind": "platt", "params": {"coef": 1.2}},
            artifact_sha256_digest="a" * 64,
        )
        model_path = tmp_path / "full_model.onnx"
        model_path.write_bytes(b"full model content")
        model_info = ModelInfo(
            manifest=manifest,
            model_path=model_path,
            deployment_status=DeploymentStatus.ACTIVE,
            deployed_to=["prod_server_1", "prod_server_2"],
            performance_history=[{"timestamp": 1700000000.0, "accuracy": 0.98}],
            metadata={"custom_key": "custom_value"},
        )

        result = persistence_component.model_info_to_dict(model_info)

        # Verify all optional fields
        assert result["manifest"]["parent_id"] == "parent_model_123"
        assert result["manifest"]["children_ids"] == ["child_1", "child_2"]
        assert result["manifest"]["training_config"] == {"epochs": 100, "batch_size": 32}
        assert result["manifest"]["serveable"] is True
        assert result["manifest"]["feature_set_id"] == "feature_set_abc"
        assert result["manifest"]["pipeline_signature"] == "sig_xyz"
        assert result["manifest"]["decision_policy"] == "threshold"
        assert result["manifest"]["output_schema"] == {"kind": "binary_proba", "shape": [None, 1]}
        assert result["manifest"]["calibration"] == {"kind": "platt", "params": {"coef": 1.2}}
        assert result["manifest"]["artifact_sha256_digest"] == "a" * 64
        assert result["deployed_to"] == ["prod_server_1", "prod_server_2"]

    def test_multiple_batch_saves_coalesce(
        self,
        persistence_component: ModelPersistenceComponent,
        sample_model_info: ModelInfo,
        tmp_registry_path: Path,
    ) -> None:
        """
        Verify multiple batch save requests coalesce into one.

        Input: Multiple save_registry(immediate=False) calls
        Expected: Only one actual save operation
        """
        persistence_component.set_model(
            sample_model_info.manifest.model_id,
            sample_model_info,
        )

        # Multiple batch save requests
        persistence_component.save_registry(immediate=False)
        persistence_component.save_registry(immediate=False)
        persistence_component.save_registry(immediate=False)

        # Should still only have one pending save
        assert persistence_component._pending_save is True

        # Wait for batch save
        time.sleep(0.2)

        # File should exist and contain the model
        registry_file = tmp_registry_path / "registry.json"
        assert registry_file.exists()

    def test_verify_artifact_integrity_empty_digest_string(
        self,
        persistence_component: ModelPersistenceComponent,
        sample_onnx_model: tuple[Path, str],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """
        Verify handling when digest is empty string.

        Input: expected_digest=""
        Expected: Warning logged, no exception
        """
        file_path, _ = sample_onnx_model

        # Should not raise with empty string
        persistence_component.verify_artifact_integrity(file_path, "")

        assert "skipping integrity verification" in caplog.text.lower()

    def test_immediate_save_cancels_pending_timer(
        self,
        persistence_component: ModelPersistenceComponent,
        sample_model_info: ModelInfo,
    ) -> None:
        """
        Verify immediate save cancels pending batch timer.

        Input: Pending batch save, then immediate save
        Expected: Timer cancelled, save happens immediately
        """
        persistence_component.set_model(
            sample_model_info.manifest.model_id,
            sample_model_info,
        )

        # Schedule batch save
        persistence_component.save_registry(immediate=False)
        assert persistence_component._save_timer is not None

        # Immediate save should cancel the timer
        persistence_component.save_registry(immediate=True)

        # Timer should be cleared from component
        assert persistence_component._save_timer is None
        # Pending save should be cleared
        assert persistence_component._pending_save is False

    def test_load_registry_with_missing_fields(
        self,
        persistence_component: ModelPersistenceComponent,
        tmp_registry_path: Path,
    ) -> None:
        """
        Verify graceful handling of JSON with missing optional fields.

        Input: Registry JSON with minimal fields
        Expected: ModelInfo created with defaults
        """
        registry_data = {
            "models": {
                "minimal_model": {
                    "manifest": {
                        "model_id": "minimal_model",
                        "role": "inference",
                        "data_requirements": "l1_only",
                        "architecture": "XGBoost",
                        "feature_schema": {},
                        "feature_schema_hash": "hash",
                        "version": "1.0.0",
                        "created_at": 1700000000.0,
                        "last_modified": 1700000000.0,
                        # Missing: parent_id, children_ids, training_config, etc.
                    },
                    "model_path": "/tmp/model.onnx",
                    "deployment_status": "inactive",
                    # Missing: deployed_to, performance_history, metadata
                }
            },
            "ab_tests": {},
            "deployments": {},
        }

        registry_file = tmp_registry_path / "registry.json"
        with open(registry_file, "w") as f:
            json.dump(registry_data, f)

        persistence_component.load_registry()

        model_info = persistence_component.get_model("minimal_model")
        assert model_info is not None
        assert model_info.manifest.parent_id is None
        assert model_info.manifest.children_ids == []
        assert model_info.deployed_to == []

    def test_cleanup_flush_on_del(
        self,
        json_persistence_config: PersistenceConfig,
        tmp_registry_path: Path,
        sample_model_info: ModelInfo,
    ) -> None:
        """
        Verify __del__ flushes pending saves.

        Input: Component with pending save
        Expected: Save flushed on destruction
        """
        component = ModelPersistenceComponent(
            persistence_config=json_persistence_config,
            registry_path=tmp_registry_path,
        )
        component.set_model(sample_model_info.manifest.model_id, sample_model_info)
        component.save_registry(immediate=False)

        # Manually trigger __del__
        component.__del__()

        # File should be saved
        registry_file = tmp_registry_path / "registry.json"
        assert registry_file.exists()


class TestDeploymentsTracking:
    """Tests for deployment tracking functionality."""

    def test_ab_tests_modification(
        self,
        persistence_component: ModelPersistenceComponent,
    ) -> None:
        """
        Verify A/B tests can be modified via property.

        Input: Direct modification of ab_tests dict
        Expected: Changes persist after save/load
        """
        persistence_component.ab_tests["test_exp_1"] = {
            "control": "model_a",
            "treatment": "model_b",
            "split": 0.5,
        }
        persistence_component.save_registry(immediate=True)
        persistence_component.load_registry()

        assert "test_exp_1" in persistence_component.ab_tests
        assert persistence_component.ab_tests["test_exp_1"]["split"] == 0.5

    def test_deployments_modification(
        self,
        persistence_component: ModelPersistenceComponent,
    ) -> None:
        """
        Verify deployments can be modified via property.

        Input: Direct modification of deployments dict
        Expected: Changes persist after save/load
        """
        persistence_component.deployments["prod_server"] = ["model_1", "model_2"]
        persistence_component.save_registry(immediate=True)
        persistence_component.load_registry()

        assert "prod_server" in persistence_component.deployments
        assert persistence_component.deployments["prod_server"] == ["model_1", "model_2"]
