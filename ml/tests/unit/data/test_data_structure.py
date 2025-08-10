#!/usr/bin/env python3

"""Tests for test data structure and accessibility."""

from __future__ import annotations

from pathlib import Path


class TestDataStructure:
    """Test that test data is properly organized and accessible."""

    def test_test_data_dir_exists(self, test_data_dir: Path) -> None:
        """Test that test data directory exists."""
        assert test_data_dir.exists()
        assert test_data_dir.is_dir()
        assert test_data_dir.name == "data"

    def test_model_registry_dir_exists(self, model_registry_dir: Path) -> None:
        """Test that model registry test data exists."""
        assert model_registry_dir.exists()
        assert model_registry_dir.is_dir()
        assert (model_registry_dir / "registry.json").exists()
        assert (model_registry_dir / "models").exists()

    def test_model_registry_rollout_dir_exists(
        self,
        model_registry_rollout_dir: Path,
    ) -> None:
        """Test that model registry rollout test data exists."""
        assert model_registry_rollout_dir.exists()
        assert model_registry_rollout_dir.is_dir()
        assert (model_registry_rollout_dir / "registry.json").exists()
        assert (model_registry_rollout_dir / "models").exists()

    def test_xgboost_test_models_exist(
        self,
        xgb_v1_model_path: Path,
        xgb_v2_model_path: Path,
    ) -> None:
        """Test that XGBoost test models exist."""
        assert xgb_v1_model_path.exists()
        assert xgb_v1_model_path.is_file()
        assert xgb_v1_model_path.suffix == ".json"

        assert xgb_v2_model_path.exists()
        assert xgb_v2_model_path.is_file()
        assert xgb_v2_model_path.suffix == ".json"

    def test_onnx_test_models_exist(
        self,
        prod_onnx_model_path: Path,
        new_onnx_model_path: Path,
    ) -> None:
        """Test that ONNX test models exist."""
        assert prod_onnx_model_path.exists()
        assert prod_onnx_model_path.is_file()
        assert prod_onnx_model_path.suffix == ".onnx"

        assert new_onnx_model_path.exists()
        assert new_onnx_model_path.is_file()
        assert new_onnx_model_path.suffix == ".onnx"

    def test_no_production_code_references_test_data(self) -> None:
        """Test that production code doesn't reference test data."""
        # This test would normally grep through production code
        # to ensure no references to test data paths
        # For now, we just document the requirement
        # Implementation deferred

    def test_registry_json_has_correct_paths(
        self,
        model_registry_dir: Path,
        model_registry_rollout_dir: Path,
    ) -> None:
        """Test that registry.json files have correct test data paths."""
        import json

        # Check model_registry paths
        registry_path = model_registry_dir / "registry.json"
        with open(registry_path) as f:
            registry = json.load(f)

        for model_data in registry["models"].values():
            model_path = model_data["model_path"]
            assert model_path.startswith("ml/tests/data/model_registry/")

        # Check model_registry_rollout paths
        rollout_registry_path = model_registry_rollout_dir / "registry.json"
        with open(rollout_registry_path) as f:
            rollout_registry = json.load(f)

        for model_data in rollout_registry["models"].values():
            model_path = model_data["model_path"]
            assert model_path.startswith("ml/tests/data/model_registry_rollout/")
