#!/usr/bin/env python3

"""End-to-end tests for pipeline signature validation."""

from pathlib import Path

import pytest

from ml.orchestration.signature import PipelineSignatureValidator
from ml.orchestration.signature import compute_pipeline_signature
from ml.registry.data_registry import DataRegistry
from ml.registry.dataclasses import DatasetManifest
from ml.registry.dataclasses import DatasetType
from ml.registry.dataclasses import StorageKind
from ml.registry.persistence import BackendType
from ml.registry.persistence import PersistenceConfig


@pytest.mark.e2e
@pytest.mark.slow
class TestOrchestratorSignature:
    """E2E tests for pipeline signature validation."""

    @pytest.fixture
    def data_registry(self, tmp_path: Path) -> DataRegistry:
        """Create a DataRegistry instance with JSON backend for testing.

        Args:
            tmp_path: Pytest temporary directory fixture

        Returns:
            DataRegistry instance with JSON backend

        """
        registry_path = tmp_path / "registry"
        registry_path.mkdir(parents=True, exist_ok=True)

        persistence_config = PersistenceConfig(
            backend=BackendType.JSON,
            json_path=registry_path,
        )

        registry = DataRegistry(
            registry_path=registry_path,
            batch_save_interval=0.0,  # Immediate saves for testing
            persistence_config=persistence_config,
        )

        return registry

    @pytest.fixture
    def test_manifest(self) -> DatasetManifest:
        """Create a test dataset manifest.

        Returns:
            DatasetManifest with minimal required fields

        """
        return DatasetManifest(
            dataset_id="test_dataset",
            dataset_type=DatasetType.FEATURES,
            storage_kind=StorageKind.PARQUET,
            location="/tmp/test_dataset",
            partitioning={},
            retention_days=30,
            schema={
                "instrument_id": "str",
                "close": "float64",
                "timestamp": "int64",
                "ts_event": "int64",
                "ts_init": "int64",
            },
            ts_field="timestamp",
            seq_field=None,
            primary_keys=["timestamp", "instrument_id"],
            schema_hash="test_hash",
            constraints={},
            lineage=[],
            pipeline_signature="",
            version="1.0.0",
            created_at=1705324800000000000,
            last_modified=1705324800000000000,
            metadata={},
        )

    def test_e2e_pipeline_signature_validation(
        self,
        data_registry: DataRegistry,
        test_manifest: DatasetManifest,
    ) -> None:
        """Test pipeline signature validation prevents data inconsistency.

        Property Under Test: Pipeline signatures prevent data inconsistency

        Given:
        - Dataset built with pipeline signature A: {"features": ["returns_1", "rsi_14"], "version": "1.0"}
        - Attempt to build another dataset with incompatible pipeline B: {"features": ["returns_1", "macd"], "version": "1.0"}
        - Attempt to build with compatible pipeline C: {"features": ["returns_1", "rsi_14"], "version": "1.0"} (same as A)

        When:
        - Building initial dataset with pipeline A → succeeds, signature saved
        - Attempting to build with pipeline B → validation should reject
        - Attempting to build with pipeline C → validation should accept

        Then:
        - Pipeline B rejected with clear error message: "Pipeline signature mismatch"
        - Pipeline C accepted and proceeds without error
        - Signature validation enforces feature consistency
        - Prevents training on inconsistent datasets

        """
        # Setup: Create validator
        validator = PipelineSignatureValidator()
        dataset_id = "test_dataset"

        # Register initial dataset
        data_registry.register_dataset(test_manifest)

        # Compute pipeline signature A
        features_a = ["returns_1", "rsi_14"]
        signature_a = compute_pipeline_signature(features=features_a, version="1.0")

        # Verify signature computation is deterministic and order-invariant
        signature_a_reordered = compute_pipeline_signature(
            features=["rsi_14", "returns_1"],
            version="1.0",
        )
        assert signature_a == signature_a_reordered, "Signature should be order-invariant"
        assert len(signature_a) == 64, "Signature should be SHA256 hex digest (64 chars)"

        # Given: First run - build dataset with pipeline A
        # This should pass because no signature is stored yet
        validator.validate_signature(
            dataset_id=dataset_id,
            new_signature=signature_a,
            registry=data_registry,
        )

        # Store signature A
        data_registry.set_pipeline_signature(dataset_id, signature_a)

        # Verify signature was stored
        stored_signature = data_registry.get_pipeline_signature(dataset_id)
        assert stored_signature == signature_a, "Stored signature should match signature_a"

        # When: Second run - attempt to build with incompatible pipeline B
        features_b = ["returns_1", "macd"]  # Different features!
        signature_b = compute_pipeline_signature(features=features_b, version="1.0")

        # Verify signatures are different
        assert signature_b != signature_a, "Different features should produce different signatures"

        # Then: Validation should reject pipeline B with clear error message
        with pytest.raises(ValueError, match="Pipeline signature mismatch"):
            validator.validate_signature(
                dataset_id=dataset_id,
                new_signature=signature_b,
                registry=data_registry,
            )

        # When: Third run - attempt to build with compatible pipeline C (same as A)
        features_c = ["returns_1", "rsi_14"]  # Same features as A
        signature_c = compute_pipeline_signature(features=features_c, version="1.0")

        # Verify signatures match
        assert signature_c == signature_a, "Same features should produce same signature"

        # Then: Validation should accept pipeline C
        validator.validate_signature(
            dataset_id=dataset_id,
            new_signature=signature_c,
            registry=data_registry,
        )  # Should not raise

        # Verify stored signature unchanged
        final_signature = data_registry.get_pipeline_signature(dataset_id)
        assert final_signature == signature_a, "Signature should remain unchanged"

    def test_signature_computation_with_config_hash(self) -> None:
        """Test signature computation with optional config hash.

        Property Under Test: Config hash adds specificity to signatures

        Given:
        - Same features and version
        - Different config hashes

        When:
        - Computing signatures with different config hashes

        Then:
        - Signatures should differ
        - Config hash adds additional specificity

        """
        features = ["returns_1", "rsi_14"]
        version = "1.0"

        # Compute signature without config hash
        sig_no_hash = compute_pipeline_signature(features=features, version=version)

        # Compute signature with config hash A
        sig_hash_a = compute_pipeline_signature(
            features=features,
            version=version,
            config_hash="abc123",
        )

        # Compute signature with config hash B
        sig_hash_b = compute_pipeline_signature(
            features=features,
            version=version,
            config_hash="def456",
        )

        # Verify signatures differ
        assert sig_no_hash != sig_hash_a, "Config hash should change signature"
        assert sig_hash_a != sig_hash_b, "Different config hashes should produce different signatures"

        # Verify signatures are deterministic
        sig_hash_a_repeat = compute_pipeline_signature(
            features=features,
            version=version,
            config_hash="abc123",
        )
        assert sig_hash_a == sig_hash_a_repeat, "Signature should be deterministic"

    def test_signature_validation_error_messages(
        self,
        data_registry: DataRegistry,
        test_manifest: DatasetManifest,
    ) -> None:
        """Test signature validation provides clear error messages.

        Property Under Test: Error messages clearly indicate mismatch

        Given:
        - Dataset with stored signature
        - Attempt with incompatible signature

        When:
        - Validation fails

        Then:
        - Error message includes dataset_id
        - Error message includes truncated signatures for debugging

        """
        validator = PipelineSignatureValidator()
        dataset_id = "test_dataset_errors"

        # Register dataset with different ID
        manifest = DatasetManifest(
            dataset_id=dataset_id,
            dataset_type=test_manifest.dataset_type,
            storage_kind=test_manifest.storage_kind,
            location=test_manifest.location,
            partitioning=test_manifest.partitioning,
            retention_days=test_manifest.retention_days,
            schema=test_manifest.schema,
            ts_field=test_manifest.ts_field,
            seq_field=test_manifest.seq_field,
            primary_keys=test_manifest.primary_keys,
            schema_hash=test_manifest.schema_hash,
            constraints=test_manifest.constraints,
            lineage=test_manifest.lineage,
            pipeline_signature=test_manifest.pipeline_signature,
            version=test_manifest.version,
            created_at=test_manifest.created_at,
            last_modified=test_manifest.last_modified,
            metadata=test_manifest.metadata,
        )
        data_registry.register_dataset(manifest)

        # Store signature A
        signature_a = compute_pipeline_signature(["feature1"], "1.0")
        data_registry.set_pipeline_signature(dataset_id, signature_a)

        # Attempt with signature B
        signature_b = compute_pipeline_signature(["feature2"], "1.0")

        # Verify error message content
        with pytest.raises(ValueError) as exc_info:
            validator.validate_signature(
                dataset_id=dataset_id,
                new_signature=signature_b,
                registry=data_registry,
            )

        error_message = str(exc_info.value)
        assert dataset_id in error_message, "Error should mention dataset_id"
        assert "Pipeline signature mismatch" in error_message, "Error should be clear"
        assert signature_a[:16] in error_message, "Error should show stored signature prefix"
        assert signature_b[:16] in error_message, "Error should show new signature prefix"

    def test_signature_validation_with_missing_dataset(
        self,
        data_registry: DataRegistry,
    ) -> None:
        """Test signature validation handles missing datasets gracefully.

        Property Under Test: Missing datasets pass validation (first run)

        Given:
        - Dataset doesn't exist in registry

        When:
        - Validating signature

        Then:
        - Validation passes (returns None)
        - No exception raised

        """
        validator = PipelineSignatureValidator()
        signature = compute_pipeline_signature(["feature1"], "1.0")

        # Should not raise - missing dataset is treated as first run
        validator.validate_signature(
            dataset_id="nonexistent_dataset",
            new_signature=signature,
            registry=data_registry,
        )

    def test_empty_features_raises_error(self) -> None:
        """Test that empty features list raises ValueError.

        Property Under Test: Input validation for signature computation

        Given:
        - Empty features list

        When:
        - Computing signature

        Then:
        - ValueError raised with clear message

        """
        with pytest.raises(ValueError, match="Features list cannot be empty"):
            compute_pipeline_signature(features=[], version="1.0")

    def test_empty_version_raises_error(self) -> None:
        """Test that empty version string raises ValueError.

        Property Under Test: Input validation for signature computation

        Given:
        - Empty version string

        When:
        - Computing signature

        Then:
        - ValueError raised with clear message

        """
        with pytest.raises(ValueError, match="Version string cannot be empty"):
            compute_pipeline_signature(features=["feature1"], version="")
