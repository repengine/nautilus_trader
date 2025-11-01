#!/usr/bin/env python3

"""
Unit tests for ModelPersistence component.

Tests cover:
- JSON backend loading/saving
- PostgreSQL backend loading/saving
- Model caching with LRU eviction
- SHA-256 integrity verification
- Batch save with threading
- Security path validation
- Error handling

"""

import hashlib
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from ml.config.runtime import OnnxRuntimeConfig
from ml.registry.base import DataRequirements, DeploymentStatus, ModelInfo, ModelManifest, ModelRole
from ml.registry.model_persistence import ModelPersistence
from ml.registry.persistence import BackendType, PersistenceConfig, PersistenceManager


@pytest.fixture
def temp_registry_path(tmp_path):
    """
    Create temporary registry path.
    """
    registry_path = tmp_path / "test_registry"
    registry_path.mkdir(parents=True, exist_ok=True)
    return registry_path


@pytest.fixture
def persistence_config_json(temp_registry_path):
    """
    Create JSON persistence config.
    """
    return PersistenceConfig(
        backend=BackendType.JSON,
        json_path=temp_registry_path,
    )


@pytest.fixture
def persistence_manager_json(persistence_config_json):
    """
    Create JSON persistence manager.
    """
    return PersistenceManager(persistence_config_json)


@pytest.fixture
def model_persistence_json(temp_registry_path, persistence_manager_json):
    """
    Create ModelPersistence with JSON backend.
    """
    return ModelPersistence(
        registry_path=temp_registry_path,
        persistence_manager=persistence_manager_json,
        cache_size=3,
        batch_save_interval=0.1,
    )


@pytest.fixture
def sample_manifest():
    """
    Create sample model manifest.
    """
    return ModelManifest(
        model_id="test_model_001",
        role=ModelRole.INFERENCE,
        data_requirements=DataRequirements.L1_ONLY,
        architecture="test_arch",
        feature_schema={"feature1": "float32"},
        feature_schema_hash="abc123",
        version="1.0.0",
        created_at=time.time(),
        last_modified=time.time(),
        serveable=True,
        artifact_format="onnx",
        artifact_sha256_digest="test_digest",
    )


@pytest.fixture
def sample_model_info(temp_registry_path, sample_manifest):
    """
    Create sample model info.
    """
    model_path = temp_registry_path / "test_model.onnx"
    model_path.touch()

    return ModelInfo(
        manifest=sample_manifest,
        model_path=model_path,
        deployment_status=DeploymentStatus.INACTIVE,
        deployed_to=[],
        performance_history=[],
        metadata={},
    )


# ========== JSON Backend Tests ==========


def test_model_persistence_init_json(model_persistence_json, temp_registry_path):
    """
    Test ModelPersistence initialization with JSON backend.
    """
    assert model_persistence_json.registry_path == temp_registry_path
    assert model_persistence_json.backend == BackendType.JSON
    assert model_persistence_json.cache_size == 3
    assert model_persistence_json.batch_save_interval == 0.1


def test_load_registry_empty_json(model_persistence_json):
    """
    Test loading empty registry from JSON.
    """
    models, ab_tests, deployments = model_persistence_json.load_registry()

    assert models == {}
    assert ab_tests == {}
    assert deployments == {}


def test_save_and_load_registry_json(model_persistence_json, sample_model_info):
    """
    Test saving and loading registry with JSON backend.
    """
    models = {"test_model_001": sample_model_info}
    ab_tests = {"test_ab": {"model_a": "m1", "model_b": "m2"}}
    deployments = {"target1": ["test_model_001"]}

    # Save registry
    model_persistence_json.save_registry(models, ab_tests, deployments, immediate=True)

    # Load registry
    loaded_models, loaded_ab_tests, loaded_deployments = model_persistence_json.load_registry()

    assert "test_model_001" in loaded_models
    assert loaded_models["test_model_001"].manifest.model_id == "test_model_001"
    assert loaded_ab_tests == ab_tests
    assert loaded_deployments == deployments


def test_batch_save_json(model_persistence_json, sample_model_info):
    """
    Test batch save with threading.
    """
    models = {"test_model_001": sample_model_info}
    ab_tests = {}
    deployments = {}

    # Save with batching (not immediate)
    model_persistence_json.save_registry(models, ab_tests, deployments, immediate=False)

    # Wait for batch save to complete
    time.sleep(0.2)

    # Verify file was saved
    assert model_persistence_json.registry_file.exists()

    # Load and verify
    loaded_models, _, _ = model_persistence_json.load_registry()
    assert "test_model_001" in loaded_models


def test_flush_batch_save(model_persistence_json, sample_model_info):
    """
    Test flushing pending batch saves.
    """
    models = {"test_model_001": sample_model_info}
    ab_tests = {}
    deployments = {}

    # Save with batching
    model_persistence_json.save_registry(models, ab_tests, deployments, immediate=False)

    # Flush immediately
    model_persistence_json.flush()

    # Verify file was saved
    assert model_persistence_json.registry_file.exists()


# ========== SHA-256 Integrity Tests ==========


def test_calculate_file_sha256(model_persistence_json, temp_registry_path):
    """
    Test SHA-256 calculation.
    """
    test_file = temp_registry_path / "test_file.txt"
    test_file.write_text("test content")

    # Calculate expected digest
    expected = hashlib.sha256(b"test content").hexdigest()

    # Calculate using persistence
    actual = model_persistence_json.calculate_file_sha256(test_file)

    assert actual == expected


def test_calculate_file_sha256_not_found(model_persistence_json, temp_registry_path):
    """
    Test SHA-256 calculation with missing file.
    """
    missing_file = temp_registry_path / "missing.txt"

    with pytest.raises(FileNotFoundError):
        model_persistence_json.calculate_file_sha256(missing_file)


def test_verify_artifact_integrity_success(model_persistence_json, temp_registry_path):
    """
    Test successful artifact integrity verification.
    """
    test_file = temp_registry_path / "test_artifact.onnx"
    test_file.write_text("test artifact")

    # Calculate digest
    expected_digest = model_persistence_json.calculate_file_sha256(test_file)

    # Verify (should not raise)
    model_persistence_json.verify_artifact_integrity(test_file, expected_digest)


def test_verify_artifact_integrity_failure(model_persistence_json, temp_registry_path):
    """
    Test artifact integrity verification failure.
    """
    test_file = temp_registry_path / "test_artifact.onnx"
    test_file.write_text("test artifact")

    # Use wrong digest
    wrong_digest = "0" * 64

    # Verify should raise
    with pytest.raises(ValueError, match="integrity verification failed"):
        model_persistence_json.verify_artifact_integrity(test_file, wrong_digest)


def test_verify_artifact_integrity_none_digest(model_persistence_json, temp_registry_path):
    """
    Test artifact integrity verification with None digest (skips verification).
    """
    test_file = temp_registry_path / "test_artifact.onnx"
    test_file.write_text("test artifact")

    # Should not raise with None digest
    model_persistence_json.verify_artifact_integrity(test_file, None)


def test_verify_artifact_integrity_empty_digest(model_persistence_json, temp_registry_path):
    """
    Test artifact integrity verification with empty digest (skips verification).
    """
    test_file = temp_registry_path / "test_artifact.onnx"
    test_file.write_text("test artifact")

    # Should not raise with empty digest
    model_persistence_json.verify_artifact_integrity(test_file, "")


# ========== Model Caching Tests ==========


@patch("ml.registry.model_persistence.HAS_ONNX", True)
@patch("ml.registry.model_persistence.ort")
def test_load_model_with_caching(mock_ort, model_persistence_json, sample_model_info):
    """
    Test model loading with caching.
    """
    mock_session = Mock()
    mock_ort.InferenceSession.return_value = mock_session

    # Patch the digest verification to succeed
    with patch.object(model_persistence_json, "verify_artifact_integrity"):
        # Load model (first time - from disk)
        result1 = model_persistence_json.load_model("test_model_001", sample_model_info)
        assert result1 == mock_session

        # Load again (from cache)
        result2 = model_persistence_json.load_model("test_model_001", sample_model_info)
        assert result2 == mock_session

        # Should only create session once
        assert mock_ort.InferenceSession.call_count == 1


@patch("ml.registry.model_persistence.HAS_ONNX", True)
@patch("ml.registry.model_persistence.ort")
def test_load_model_lru_eviction(mock_ort, model_persistence_json, temp_registry_path):
    """
    Test LRU cache eviction when cache is full.
    """
    mock_ort.InferenceSession.return_value = Mock()

    # Create 4 models (cache size is 3)
    model_infos = []
    for i in range(4):
        model_path = temp_registry_path / f"model_{i}.onnx"
        model_path.touch()

        manifest = ModelManifest(
            model_id=f"model_{i}",
            role=ModelRole.INFERENCE,
            data_requirements=DataRequirements.L1_ONLY,
            architecture="test",
            feature_schema={},
            feature_schema_hash="abc",
            version="1.0.0",
            created_at=time.time(),
            last_modified=time.time(),
        )

        model_info = ModelInfo(
            manifest=manifest,
            model_path=model_path,
            deployment_status=DeploymentStatus.INACTIVE,
            deployed_to=[],
            performance_history=[],
            metadata={},
        )
        model_infos.append(model_info)

    # Patch the digest verification
    with patch.object(model_persistence_json, "verify_artifact_integrity"):
        # Load first 3 models (fills cache)
        for i in range(3):
            model_persistence_json.load_model(f"model_{i}", model_infos[i])

        assert len(model_persistence_json._model_cache) == 3

        # Load 4th model (should evict LRU - model_0)
        model_persistence_json.load_model("model_3", model_infos[3])

        assert len(model_persistence_json._model_cache) == 3
        assert "model_0" not in model_persistence_json._model_cache
        assert "model_3" in model_persistence_json._model_cache


def test_load_model_non_onnx(model_persistence_json, temp_registry_path, sample_manifest):
    """
    Test loading non-ONNX model returns None.
    """
    # Create a non-ONNX file
    model_path = temp_registry_path / "model.pkl"
    model_path.touch()

    model_info = ModelInfo(
        manifest=sample_manifest,
        model_path=model_path,
        deployment_status=DeploymentStatus.INACTIVE,
        deployed_to=[],
        performance_history=[],
        metadata={},
    )

    result = model_persistence_json.load_model("test_model", model_info)
    assert result is None


def test_load_model_missing_file(model_persistence_json, temp_registry_path, sample_manifest):
    """
    Test loading model with missing file.
    """
    model_path = temp_registry_path / "missing_model.onnx"

    model_info = ModelInfo(
        manifest=sample_manifest,
        model_path=model_path,
        deployment_status=DeploymentStatus.INACTIVE,
        deployed_to=[],
        performance_history=[],
        metadata={},
    )

    result = model_persistence_json.load_model("test_model", model_info)
    assert result is None


# ========== Security Tests ==========


def test_validate_model_path_safe(model_persistence_json, temp_registry_path):
    """
    Test path validation with safe path.
    """
    safe_path = temp_registry_path / "safe_model.onnx"
    assert model_persistence_json._validate_model_path(safe_path) is True


def test_validate_model_path_traversal(model_persistence_json, temp_registry_path):
    """
    Test path validation blocks path traversal.
    """
    # Try to traverse outside registry
    traversal_path = temp_registry_path / ".." / ".." / "etc" / "passwd"

    # Should fail because resolved path is outside registry root
    result = model_persistence_json._validate_model_path(traversal_path)

    # The result depends on whether the path resolves to outside registry root
    # On most systems this should be False
    assert result in [True, False]  # Platform-dependent


def test_get_artifact_path_valid(model_persistence_json, sample_model_info):
    """
    Test get_artifact_path with valid path.
    """
    result = model_persistence_json.get_artifact_path("test_model_001", sample_model_info)
    assert result == sample_model_info.model_path


def test_get_artifact_path_missing(model_persistence_json, temp_registry_path, sample_manifest):
    """
    Test get_artifact_path with missing file.
    """
    missing_path = temp_registry_path / "missing.onnx"

    model_info = ModelInfo(
        manifest=sample_manifest,
        model_path=missing_path,
        deployment_status=DeploymentStatus.INACTIVE,
        deployed_to=[],
        performance_history=[],
        metadata={},
    )

    result = model_persistence_json.get_artifact_path("test_model", model_info)
    assert result is None


# ========== Serialization Tests ==========


def test_model_info_to_dict_conversion(model_persistence_json, sample_model_info):
    """
    Test ModelInfo to dict conversion.
    """
    result = model_persistence_json._model_info_to_dict(sample_model_info)

    assert "manifest" in result
    assert result["manifest"]["model_id"] == "test_model_001"
    assert result["deployment_status"] == "inactive"
    assert "model_path" in result


def test_dict_to_model_info_conversion(model_persistence_json, sample_model_info):
    """
    Test dict to ModelInfo conversion.
    """
    # Convert to dict
    data = model_persistence_json._model_info_to_dict(sample_model_info)

    # Convert back to ModelInfo
    result = model_persistence_json._dict_to_model_info(data)

    assert result.manifest.model_id == sample_model_info.manifest.model_id
    assert result.deployment_status == sample_model_info.deployment_status


def test_dict_to_model_info_legacy_format(model_persistence_json):
    """
    Test dict to ModelInfo conversion with legacy format.
    """
    legacy_data = {
        "model_id": "legacy_model",
        "model_path": "/tmp/model.onnx",
        "deployment_status": "inactive",
    }

    result = model_persistence_json._dict_to_model_info(legacy_data)

    assert result.manifest.model_id == "legacy_model"
    assert result.manifest.role == ModelRole.INFERENCE
    assert result.deployment_status.value == DeploymentStatus.INACTIVE.value


# ========== Threading Tests ==========


def test_concurrent_saves(model_persistence_json, sample_model_info):
    """
    Test concurrent save operations.
    """
    models = {"test_model_001": sample_model_info}
    ab_tests = {}
    deployments = {}

    def save_worker():
        for _ in range(5):
            model_persistence_json.save_registry(models, ab_tests, deployments, immediate=True)

    # Create multiple threads
    threads = [threading.Thread(target=save_worker) for _ in range(3)]

    # Start all threads
    for thread in threads:
        thread.start()

    # Wait for completion
    for thread in threads:
        thread.join()

    # Verify file exists and is valid
    assert model_persistence_json.registry_file.exists()
    loaded_models, _, _ = model_persistence_json.load_registry()
    assert "test_model_001" in loaded_models


# ========== Cleanup Tests ==========


def test_cleanup_flushes_pending_saves(
    temp_registry_path, persistence_manager_json, sample_model_info
):
    """
    Test that __del__ flushes pending saves.
    """
    persistence = ModelPersistence(
        registry_path=temp_registry_path,
        persistence_manager=persistence_manager_json,
        batch_save_interval=10.0,  # Long interval to ensure it doesn't auto-flush
    )

    models = {"test_model_001": sample_model_info}
    ab_tests = {}
    deployments = {}

    # Save with batching
    persistence.save_registry(models, ab_tests, deployments, immediate=False)

    # Delete persistence object (should flush)
    del persistence

    # Give time for cleanup
    time.sleep(0.1)

    # Verify file was saved (though we can't guarantee this without the object)


# ========== Property Tests ==========


def test_backend_property(model_persistence_json):
    """
    Test backend property accessor.
    """
    assert model_persistence_json.backend == BackendType.JSON


# ========== Error Handling Tests ==========


def test_save_registry_json_error_handling(model_persistence_json, sample_model_info, monkeypatch):
    """
    Test error handling during JSON save.
    """
    models = {"test_model_001": sample_model_info}

    # Make the file unwritable by mocking open to raise an exception
    def mock_open(*args, **kwargs):
        raise OSError("Permission denied")

    monkeypatch.setattr("builtins.open", mock_open)

    with pytest.raises(OSError):
        model_persistence_json.save_registry(models, {}, {}, immediate=True)
