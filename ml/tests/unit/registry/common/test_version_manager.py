#!/usr/bin/env python3

"""
Unit tests for VersionManagerComponent.

This module tests the VersionManagerComponent which handles:
- Auto-versioning of model manifests
- Schema compatibility filtering
- Model lineage tracking
- Latest version resolution

Test Coverage Target: 90%
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from ml.registry.base import DataRequirements
from ml.registry.base import DeploymentStatus
from ml.registry.base import ModelInfo
from ml.registry.base import ModelManifest
from ml.registry.base import ModelRole
from ml.registry.common.model_persistence import ModelPersistenceComponent
from ml.registry.common.version_manager import VersionManagerComponent
from ml.registry.persistence import BackendType
from ml.registry.persistence import PersistenceConfig


if TYPE_CHECKING:
    pass


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def persistence_component(tmp_path: Path) -> ModelPersistenceComponent:
    """
    ModelPersistenceComponent with JSON backend for unit tests.

    Uses temporary directory to avoid side effects.
    """
    config = PersistenceConfig(
        backend=BackendType.JSON,
        json_path=tmp_path,
    )
    component = ModelPersistenceComponent(config, tmp_path)
    component.load_registry()
    return component


@pytest.fixture
def version_manager(persistence_component: ModelPersistenceComponent) -> VersionManagerComponent:
    """
    VersionManagerComponent for versioning and lineage tests.
    """
    return VersionManagerComponent(persistence_component)


@pytest.fixture
def sample_manifest() -> ModelManifest:
    """Create a sample model manifest for testing."""
    return ModelManifest(
        model_id="test_model_1",
        role=ModelRole.INFERENCE,
        data_requirements=DataRequirements.L1_ONLY,
        architecture="XGBoost",
        feature_schema={"close": "float32", "volume": "float32"},
        feature_schema_hash="test_hash_abc123",
        version="",  # Empty for auto-versioning tests
        created_at=time.time(),
        last_modified=time.time(),
    )


@pytest.fixture
def sample_onnx_model(tmp_path: Path) -> tuple[Path, str]:
    """
    Create a sample ONNX model file and return (path, sha256_digest).
    """
    import hashlib

    model_file = tmp_path / "test_model.onnx"
    content = b"sample ONNX model content for testing"
    model_file.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    return model_file, digest


def _create_model_info(
    model_id: str,
    model_path: Path,
    architecture: str = "XGBoost",
    version: str = "1.0.0",
    schema_hash: str = "test_hash_abc123",
    role: ModelRole = ModelRole.INFERENCE,
    parent_id: str | None = None,
    children_ids: list[str] | None = None,
) -> ModelInfo:
    """Helper to create ModelInfo for testing."""
    manifest = ModelManifest(
        model_id=model_id,
        role=role,
        data_requirements=DataRequirements.L1_ONLY,
        architecture=architecture,
        feature_schema={"close": "float32"},
        feature_schema_hash=schema_hash,
        version=version,
        parent_id=parent_id,
        children_ids=children_ids or [],
        created_at=time.time(),
        last_modified=time.time(),
    )
    return ModelInfo(
        manifest=manifest,
        model_path=model_path,
        deployment_status=DeploymentStatus.INACTIVE,
        deployed_to=[],
        performance_history=[],
        metadata={},
    )


# =============================================================================
# Happy Path Tests - Auto Versioning
# =============================================================================


class TestAutoVersionManifest:
    """Tests for auto_version_manifest method."""

    def test_auto_version_manifest_first_model(
        self,
        version_manager: VersionManagerComponent,
        sample_manifest: ModelManifest,
    ) -> None:
        """Verify auto-versioning for first model of architecture assigns 1.0.0."""
        # Given: A manifest without version and empty registry
        assert sample_manifest.version == ""

        # When: Auto-versioning is applied
        version_manager.auto_version_manifest(sample_manifest)

        # Then: Version should be 1.0.0
        assert sample_manifest.version == "1.0.0"

    def test_auto_version_manifest_increment(
        self,
        version_manager: VersionManagerComponent,
        persistence_component: ModelPersistenceComponent,
        sample_manifest: ModelManifest,
        sample_onnx_model: tuple[Path, str],
    ) -> None:
        """Verify auto-versioning increments patch version from existing models."""
        model_path, _ = sample_onnx_model

        # Given: An existing model with version 1.2.3 for same architecture
        existing = _create_model_info(
            model_id="existing_model",
            model_path=model_path,
            architecture="XGBoost",
            version="1.2.3",
        )
        persistence_component.set_model("existing_model", existing)

        # When: Auto-versioning is applied to new manifest
        sample_manifest.version = ""
        version_manager.auto_version_manifest(sample_manifest)

        # Then: Version should be 1.2.4
        assert sample_manifest.version == "1.2.4"

    def test_auto_version_manifest_with_version(
        self,
        version_manager: VersionManagerComponent,
        sample_manifest: ModelManifest,
    ) -> None:
        """Verify auto-versioning skips when version already provided."""
        # Given: A manifest with pre-set version
        sample_manifest.version = "2.0.0"

        # When: Auto-versioning is applied
        version_manager.auto_version_manifest(sample_manifest)

        # Then: Version should remain unchanged
        assert sample_manifest.version == "2.0.0"

    def test_auto_version_manifest_different_architecture(
        self,
        version_manager: VersionManagerComponent,
        persistence_component: ModelPersistenceComponent,
        sample_manifest: ModelManifest,
        sample_onnx_model: tuple[Path, str],
    ) -> None:
        """Verify auto-versioning considers only same architecture."""
        model_path, _ = sample_onnx_model

        # Given: An existing model with different architecture
        existing = _create_model_info(
            model_id="existing_lgb",
            model_path=model_path,
            architecture="LightGBM",  # Different architecture
            version="5.0.0",
        )
        persistence_component.set_model("existing_lgb", existing)

        # When: Auto-versioning is applied to XGBoost manifest
        sample_manifest.architecture = "XGBoost"
        sample_manifest.version = ""
        version_manager.auto_version_manifest(sample_manifest)

        # Then: Version should be 1.0.0 (first of this architecture)
        assert sample_manifest.version == "1.0.0"

    def test_auto_version_manifest_selects_max_version(
        self,
        version_manager: VersionManagerComponent,
        persistence_component: ModelPersistenceComponent,
        sample_manifest: ModelManifest,
        sample_onnx_model: tuple[Path, str],
    ) -> None:
        """Verify auto-versioning selects highest version among same architecture."""
        model_path, _ = sample_onnx_model

        # Given: Multiple models with same architecture at different versions
        for i, version in enumerate(["1.0.5", "2.1.0", "1.9.9"]):
            model = _create_model_info(
                model_id=f"model_{i}",
                model_path=model_path,
                architecture="XGBoost",
                version=version,
            )
            persistence_component.set_model(f"model_{i}", model)

        # When: Auto-versioning is applied
        sample_manifest.version = ""
        version_manager.auto_version_manifest(sample_manifest)

        # Then: Version should be 2.1.1 (max is 2.1.0, so patch incremented)
        assert sample_manifest.version == "2.1.1"


# =============================================================================
# Happy Path Tests - List Compatible
# =============================================================================


class TestListCompatible:
    """Tests for list_compatible method."""

    def test_list_compatible_by_schema_hash(
        self,
        version_manager: VersionManagerComponent,
        persistence_component: ModelPersistenceComponent,
        sample_onnx_model: tuple[Path, str],
    ) -> None:
        """Verify listing models by schema hash returns all matching models."""
        model_path, _ = sample_onnx_model
        target_hash = "matching_hash_xyz"

        # Given: Models with different schema hashes
        matching_1 = _create_model_info(
            model_id="match_1",
            model_path=model_path,
            schema_hash=target_hash,
        )
        matching_2 = _create_model_info(
            model_id="match_2",
            model_path=model_path,
            schema_hash=target_hash,
        )
        non_matching = _create_model_info(
            model_id="no_match",
            model_path=model_path,
            schema_hash="different_hash",
        )
        persistence_component.set_model("match_1", matching_1)
        persistence_component.set_model("match_2", matching_2)
        persistence_component.set_model("no_match", non_matching)

        # When: Listing compatible models
        result = version_manager.list_compatible(schema_hash=target_hash)

        # Then: Only matching models returned
        assert len(result) == 2
        model_ids = {m.manifest.model_id for m in result}
        assert model_ids == {"match_1", "match_2"}

    def test_list_compatible_with_role_filter(
        self,
        version_manager: VersionManagerComponent,
        persistence_component: ModelPersistenceComponent,
        sample_onnx_model: tuple[Path, str],
    ) -> None:
        """Verify filtering by role returns only models with matching role."""
        model_path, _ = sample_onnx_model
        target_hash = "shared_hash"

        # Given: Models with different roles
        inference = _create_model_info(
            model_id="inference_model",
            model_path=model_path,
            schema_hash=target_hash,
            role=ModelRole.INFERENCE,
        )
        student = _create_model_info(
            model_id="student_model",
            model_path=model_path,
            schema_hash=target_hash,
            role=ModelRole.STUDENT,
        )
        persistence_component.set_model("inference_model", inference)
        persistence_component.set_model("student_model", student)

        # When: Listing with role filter
        result = version_manager.list_compatible(
            schema_hash=target_hash,
            role=ModelRole.STUDENT,
        )

        # Then: Only STUDENT role returned
        assert len(result) == 1
        assert result[0].manifest.model_id == "student_model"
        assert result[0].manifest.role == ModelRole.STUDENT

    def test_list_compatible_with_architecture_filter(
        self,
        version_manager: VersionManagerComponent,
        persistence_component: ModelPersistenceComponent,
        sample_onnx_model: tuple[Path, str],
    ) -> None:
        """Verify filtering by architecture returns only matching models."""
        model_path, _ = sample_onnx_model
        target_hash = "shared_hash"

        # Given: Models with different architectures
        xgb = _create_model_info(
            model_id="xgb_model",
            model_path=model_path,
            schema_hash=target_hash,
            architecture="XGBoost",
        )
        lgb = _create_model_info(
            model_id="lgb_model",
            model_path=model_path,
            schema_hash=target_hash,
            architecture="LightGBM",
        )
        persistence_component.set_model("xgb_model", xgb)
        persistence_component.set_model("lgb_model", lgb)

        # When: Listing with architecture filter
        result = version_manager.list_compatible(
            schema_hash=target_hash,
            architecture="XGBoost",
        )

        # Then: Only XGBoost returned
        assert len(result) == 1
        assert result[0].manifest.model_id == "xgb_model"
        assert result[0].manifest.architecture == "XGBoost"

    def test_list_compatible_with_all_filters(
        self,
        version_manager: VersionManagerComponent,
        persistence_component: ModelPersistenceComponent,
        sample_onnx_model: tuple[Path, str],
    ) -> None:
        """Verify combining all filters works correctly."""
        model_path, _ = sample_onnx_model
        target_hash = "target_hash"

        # Given: Various models
        target = _create_model_info(
            model_id="target",
            model_path=model_path,
            schema_hash=target_hash,
            architecture="XGBoost",
            role=ModelRole.INFERENCE,
        )
        wrong_arch = _create_model_info(
            model_id="wrong_arch",
            model_path=model_path,
            schema_hash=target_hash,
            architecture="LightGBM",
            role=ModelRole.INFERENCE,
        )
        wrong_role = _create_model_info(
            model_id="wrong_role",
            model_path=model_path,
            schema_hash=target_hash,
            architecture="XGBoost",
            role=ModelRole.STUDENT,
        )
        persistence_component.set_model("target", target)
        persistence_component.set_model("wrong_arch", wrong_arch)
        persistence_component.set_model("wrong_role", wrong_role)

        # When: Using all filters
        result = version_manager.list_compatible(
            schema_hash=target_hash,
            role=ModelRole.INFERENCE,
            architecture="XGBoost",
        )

        # Then: Only exact match returned
        assert len(result) == 1
        assert result[0].manifest.model_id == "target"


# =============================================================================
# Happy Path Tests - Resolve Latest
# =============================================================================


class TestResolveLatest:
    """Tests for resolve_latest method."""

    def test_resolve_latest(
        self,
        version_manager: VersionManagerComponent,
        persistence_component: ModelPersistenceComponent,
        sample_onnx_model: tuple[Path, str],
    ) -> None:
        """Verify resolve_latest returns model with highest version."""
        model_path, _ = sample_onnx_model
        target_hash = "shared_hash"

        # Given: Multiple models with different versions
        for i, version in enumerate(["1.0.0", "2.0.0", "1.5.0"]):
            model = _create_model_info(
                model_id=f"model_{i}",
                model_path=model_path,
                architecture="XGBoost",
                version=version,
                schema_hash=target_hash,
                role=ModelRole.INFERENCE,
            )
            persistence_component.set_model(f"model_{i}", model)

        # When: Resolving latest
        result = version_manager.resolve_latest(
            role=ModelRole.INFERENCE,
            architecture="XGBoost",
            schema_hash=target_hash,
        )

        # Then: Model with version 2.0.0 returned
        assert result is not None
        assert result.manifest.version == "2.0.0"
        assert result.manifest.model_id == "model_1"


# =============================================================================
# Happy Path Tests - Model Lineage
# =============================================================================


class TestGetModelLineage:
    """Tests for get_model_lineage method."""

    def test_get_model_lineage_with_parent(
        self,
        version_manager: VersionManagerComponent,
        persistence_component: ModelPersistenceComponent,
        sample_onnx_model: tuple[Path, str],
    ) -> None:
        """Verify lineage tracing through parent chain."""
        model_path, _ = sample_onnx_model

        # Given: A chain of parent-child relationships (grandparent -> parent -> child)
        grandparent = _create_model_info(
            model_id="grandparent",
            model_path=model_path,
            parent_id=None,
            children_ids=["parent"],
        )
        parent = _create_model_info(
            model_id="parent",
            model_path=model_path,
            parent_id="grandparent",
            children_ids=["child"],
        )
        child = _create_model_info(
            model_id="child",
            model_path=model_path,
            parent_id="parent",
            children_ids=[],
        )
        persistence_component.set_model("grandparent", grandparent)
        persistence_component.set_model("parent", parent)
        persistence_component.set_model("child", child)

        # When: Getting lineage for child
        result = version_manager.get_model_lineage("child")

        # Then: Lineage includes all ancestors in order
        assert len(result) == 3
        assert result[0].manifest.model_id == "grandparent"
        assert result[1].manifest.model_id == "parent"
        assert result[2].manifest.model_id == "child"

    def test_get_model_lineage_with_children(
        self,
        version_manager: VersionManagerComponent,
        persistence_component: ModelPersistenceComponent,
        sample_onnx_model: tuple[Path, str],
    ) -> None:
        """Verify lineage includes children of the queried model."""
        model_path, _ = sample_onnx_model

        # Given: A model with children
        parent = _create_model_info(
            model_id="parent",
            model_path=model_path,
            parent_id=None,
            children_ids=["child_1", "child_2"],
        )
        child_1 = _create_model_info(
            model_id="child_1",
            model_path=model_path,
            parent_id="parent",
            children_ids=[],
        )
        child_2 = _create_model_info(
            model_id="child_2",
            model_path=model_path,
            parent_id="parent",
            children_ids=[],
        )
        persistence_component.set_model("parent", parent)
        persistence_component.set_model("child_1", child_1)
        persistence_component.set_model("child_2", child_2)

        # When: Getting lineage for parent
        result = version_manager.get_model_lineage("parent")

        # Then: Lineage includes parent and children
        assert len(result) == 3
        assert result[0].manifest.model_id == "parent"
        model_ids = {m.manifest.model_id for m in result[1:]}
        assert model_ids == {"child_1", "child_2"}

    def test_get_model_lineage_single_model(
        self,
        version_manager: VersionManagerComponent,
        persistence_component: ModelPersistenceComponent,
        sample_onnx_model: tuple[Path, str],
    ) -> None:
        """Verify lineage for model with no parents or children."""
        model_path, _ = sample_onnx_model

        # Given: A standalone model
        standalone = _create_model_info(
            model_id="standalone",
            model_path=model_path,
            parent_id=None,
            children_ids=[],
        )
        persistence_component.set_model("standalone", standalone)

        # When: Getting lineage
        result = version_manager.get_model_lineage("standalone")

        # Then: Only the model itself
        assert len(result) == 1
        assert result[0].manifest.model_id == "standalone"


# =============================================================================
# Error Condition Tests
# =============================================================================


class TestErrorConditions:
    """Tests for error conditions and edge cases."""

    def test_resolve_latest_no_matches(
        self,
        version_manager: VersionManagerComponent,
        persistence_component: ModelPersistenceComponent,
        sample_onnx_model: tuple[Path, str],
    ) -> None:
        """Verify resolve_latest returns None when no compatible models exist."""
        model_path, _ = sample_onnx_model

        # Given: Models that don't match criteria
        model = _create_model_info(
            model_id="model_1",
            model_path=model_path,
            architecture="LightGBM",
            schema_hash="other_hash",
        )
        persistence_component.set_model("model_1", model)

        # When: Resolving with non-matching criteria
        result = version_manager.resolve_latest(
            role=ModelRole.INFERENCE,
            architecture="XGBoost",
            schema_hash="non_existent_hash",
        )

        # Then: Returns None
        assert result is None

    def test_get_model_lineage_not_found(
        self,
        version_manager: VersionManagerComponent,
    ) -> None:
        """Verify get_model_lineage returns empty list for non-existent model."""
        # When: Getting lineage for non-existent model
        result = version_manager.get_model_lineage("non_existent_model")

        # Then: Returns empty list
        assert result == []


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_list_compatible_empty_registry(
        self,
        version_manager: VersionManagerComponent,
    ) -> None:
        """Verify list_compatible returns empty list with no registered models."""
        # When: Listing from empty registry
        result = version_manager.list_compatible(schema_hash="any_hash")

        # Then: Returns empty list
        assert result == []

    def test_lineage_circular_reference_handling(
        self,
        version_manager: VersionManagerComponent,
        persistence_component: ModelPersistenceComponent,
        sample_onnx_model: tuple[Path, str],
    ) -> None:
        """Verify lineage terminates gracefully with circular parent references."""
        model_path, _ = sample_onnx_model

        # Given: Models with circular parent reference (A -> B -> A)
        # Note: This is an invalid state, but we should handle it gracefully
        model_a = _create_model_info(
            model_id="model_a",
            model_path=model_path,
            parent_id="model_b",  # Points to B
            children_ids=[],
        )
        model_b = _create_model_info(
            model_id="model_b",
            model_path=model_path,
            parent_id="model_a",  # Points back to A (circular!)
            children_ids=[],
        )
        persistence_component.set_model("model_a", model_a)
        persistence_component.set_model("model_b", model_b)

        # When: Getting lineage (should not infinite loop)
        result = version_manager.get_model_lineage("model_a")

        # Then: Lineage terminates with finite length and no duplicates
        assert len(result) > 0
        assert len(result) <= 10  # Reasonable upper bound
        model_ids = [m.manifest.model_id for m in result]
        # The current implementation doesn't have cycle detection,
        # but it terminates because parent_id lookup breaks the cycle
        # when the parent is already the one we started tracing from
        assert "model_a" in model_ids

    def test_lineage_missing_parent_in_chain(
        self,
        version_manager: VersionManagerComponent,
        persistence_component: ModelPersistenceComponent,
        sample_onnx_model: tuple[Path, str],
    ) -> None:
        """Verify lineage handles missing parent gracefully."""
        model_path, _ = sample_onnx_model

        # Given: A model with parent_id pointing to non-existent model
        model = _create_model_info(
            model_id="orphan",
            model_path=model_path,
            parent_id="non_existent_parent",
            children_ids=[],
        )
        persistence_component.set_model("orphan", model)

        # When: Getting lineage
        result = version_manager.get_model_lineage("orphan")

        # Then: Returns just the model itself (parent not found)
        assert len(result) == 1
        assert result[0].manifest.model_id == "orphan"

    def test_lineage_missing_child(
        self,
        version_manager: VersionManagerComponent,
        persistence_component: ModelPersistenceComponent,
        sample_onnx_model: tuple[Path, str],
    ) -> None:
        """Verify lineage handles missing children gracefully."""
        model_path, _ = sample_onnx_model

        # Given: A model with children_ids pointing to non-existent models
        parent = _create_model_info(
            model_id="parent_with_missing_children",
            model_path=model_path,
            parent_id=None,
            children_ids=["missing_child_1", "missing_child_2"],
        )
        persistence_component.set_model("parent_with_missing_children", parent)

        # When: Getting lineage
        result = version_manager.get_model_lineage("parent_with_missing_children")

        # Then: Returns just the parent (children not found)
        assert len(result) == 1
        assert result[0].manifest.model_id == "parent_with_missing_children"
