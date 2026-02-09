from __future__ import annotations

import hashlib
import importlib
import json
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from typing import cast
from unittest.mock import MagicMock

import pytest

import ml.registry.artifacts as artifacts
from ml.registry.base import DataRequirements
from ml.registry.base import DeploymentStatus
from ml.registry.base import ModelInfo
from ml.registry.base import ModelManifest
from ml.registry.base import ModelRole
from ml.registry.canary_deployment_mgr import CanaryDeploymentManager
from ml.registry.dataclasses import CanaryConfig
from ml.registry.lineage_manager import LineageManager
from ml.registry.model_deployment_mgr import ModelDeploymentManager
from ml.registry.model_deployment_mgr import ModelDeploymentManagerProtocol
from ml.registry.persistence import BackendType
from ml.registry.persistence import PersistenceConfig


class _CounterRecorder:
    def __init__(self) -> None:
        self.statuses: list[str] = []
        self.increments = 0

    def labels(self, *, status: str) -> _CounterRecorder:
        self.statuses.append(status)
        return self

    def inc(self) -> None:
        self.increments += 1


class _FakeSelect:
    def __init__(self) -> None:
        self.where_calls: list[object] = []
        self.order_by_calls: list[object] = []
        self.limit_calls: list[int] = []

    def where(self, criterion: object) -> _FakeSelect:
        self.where_calls.append(criterion)
        return self

    def order_by(self, criterion: object) -> _FakeSelect:
        self.order_by_calls.append(criterion)
        return self

    def limit(self, limit_value: int) -> _FakeSelect:
        self.limit_calls.append(limit_value)
        return self


class _Row:
    def __init__(self, mapping: dict[str, Any]) -> None:
        self._mapping = mapping


def _json_persistence() -> Any:
    return SimpleNamespace(config=SimpleNamespace(backend=BackendType.JSON))


def _postgres_persistence(session: Any) -> Any:
    return SimpleNamespace(
        config=SimpleNamespace(backend=BackendType.POSTGRES),
        get_session=lambda: session,
    )


def _fake_lineage_table() -> Any:
    created_at = MagicMock()
    created_at.desc.return_value = "created_at_desc"
    columns = SimpleNamespace(
        transform_id=MagicMock(),
        child_dataset_id=MagicMock(),
        parent_dataset_id=MagicMock(),
        ts_range=MagicMock(),
        parameters=MagicMock(),
        created_at=created_at,
    )
    return SimpleNamespace(c=columns)


def _ab_testing_module() -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return importlib.import_module("ml.registry.ab_testing_manager")


def _make_model_info(
    model_id: str,
    *,
    version: str = "1.0.0",
    role: ModelRole = ModelRole.INFERENCE,
    data_requirements: DataRequirements = DataRequirements.L1_ONLY,
    architecture: str = "test_arch",
    feature_schema_hash: str = "schema_hash",
    deployment_status: DeploymentStatus = DeploymentStatus.INACTIVE,
    deployed_to: list[str] | None = None,
    performance_history: list[dict[str, Any]] | None = None,
    performance_metrics: dict[str, float] | None = None,
    metadata: Any | None = None,
    parent_id: str | None = None,
    children_ids: list[str] | None = None,
) -> ModelInfo:
    manifest = ModelManifest(
        model_id=model_id,
        role=role,
        data_requirements=data_requirements,
        architecture=architecture,
        feature_schema={"feature": "float32"},
        feature_schema_hash=feature_schema_hash,
        parent_id=parent_id,
        children_ids=list(children_ids or []),
        version=version,
        created_at=1.0,
        last_modified=1.0,
        performance_metrics=performance_metrics or {},
    )
    return ModelInfo(
        manifest=manifest,
        model_path=Path(f"/tmp/{model_id}.onnx"),
        deployment_status=deployment_status,
        deployed_to=list(deployed_to or []),
        performance_history=list(performance_history or []),
        metadata={} if metadata is None else metadata,
    )


class TestArtifactsModule:
    def test_sha256_returns_expected_digest(self, tmp_path: Path) -> None:
        artifact_path = tmp_path / "model.onnx"
        artifact_path.write_bytes(b"nautilus-artifact")
        expected = hashlib.sha256(b"nautilus-artifact").hexdigest()
        assert artifacts._sha256(artifact_path) == expected

    def test_validate_onnx_returns_true_with_stubbed_runtime(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        artifact_path = tmp_path / "valid.onnx"
        artifact_path.write_bytes(b"stub")
        runtime_stub = SimpleNamespace(InferenceSession=lambda _path: object())
        monkeypatch.setitem(sys.modules, "onnxruntime", runtime_stub)
        assert artifacts.validate_onnx(artifact_path) is True

    def test_validate_onnx_returns_false_when_runtime_raises(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        artifact_path = tmp_path / "invalid.onnx"
        artifact_path.write_bytes(b"stub")

        def _raise(_path: str) -> object:
            raise RuntimeError("load failed")

        runtime_stub = SimpleNamespace(InferenceSession=_raise)
        monkeypatch.setitem(sys.modules, "onnxruntime", runtime_stub)
        assert artifacts.validate_onnx(artifact_path) is False

    def test_update_model_artifact_success_updates_registry_row(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        counter = _CounterRecorder()
        session = MagicMock()
        manager_stub = SimpleNamespace(get_session=lambda: session)
        artifact_path = tmp_path / "model.onnx"
        artifact_path.write_bytes(b"model")

        monkeypatch.setattr(artifacts, "_artifact_updates_total", counter)
        monkeypatch.setattr(artifacts, "PersistenceManager", lambda _cfg: manager_stub)
        monkeypatch.setattr(artifacts, "_sha256", lambda _path: "digest-123")
        validate_mock = MagicMock(return_value=True)
        monkeypatch.setattr(artifacts, "validate_onnx", validate_mock)

        request = artifacts.ArtifactUpdateRequest(
            model_id="model-1",
            artifact_path=artifact_path,
            artifact_format="onnx",
            validate=True,
        )
        result = artifacts.update_model_artifact(
            request=request,
            registry_config=PersistenceConfig(
                backend=BackendType.JSON,
                json_path=tmp_path,
            ),
        )

        assert result == {"model_id": "model-1", "digest": "digest-123", "valid": True}
        assert session.execute.call_count == 1
        session.commit.assert_called_once()
        session.rollback.assert_not_called()
        session.close.assert_called_once()
        validate_mock.assert_called_once_with(artifact_path)
        assert counter.statuses == ["ok"]
        assert counter.increments == 1

    def test_update_model_artifact_skips_validation_for_non_onnx(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        counter = _CounterRecorder()
        session = MagicMock()
        manager_stub = SimpleNamespace(get_session=lambda: session)
        artifact_path = tmp_path / "model.bin"
        artifact_path.write_bytes(b"model")

        monkeypatch.setattr(artifacts, "_artifact_updates_total", counter)
        monkeypatch.setattr(artifacts, "PersistenceManager", lambda _cfg: manager_stub)
        monkeypatch.setattr(artifacts, "_sha256", lambda _path: "digest-xyz")
        validate_mock = MagicMock(return_value=False)
        monkeypatch.setattr(artifacts, "validate_onnx", validate_mock)

        request = artifacts.ArtifactUpdateRequest(
            model_id="model-2",
            artifact_path=artifact_path,
            artifact_format="bin",
            validate=True,
        )
        result = artifacts.update_model_artifact(
            request=request,
            registry_config=PersistenceConfig(
                backend=BackendType.JSON,
                json_path=tmp_path,
            ),
        )

        assert result["valid"] is True
        validate_mock.assert_not_called()
        assert counter.statuses == ["ok"]

    def test_update_model_artifact_handles_execute_rollback_and_close_failures(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        counter = _CounterRecorder()
        session = MagicMock()
        session.execute.side_effect = RuntimeError("db-write-failed")
        session.rollback.side_effect = RuntimeError("rollback-failed")
        session.close.side_effect = RuntimeError("close-failed")
        manager_stub = SimpleNamespace(get_session=lambda: session)
        artifact_path = tmp_path / "model.onnx"
        artifact_path.write_bytes(b"model")

        monkeypatch.setattr(artifacts, "_artifact_updates_total", counter)
        monkeypatch.setattr(artifacts, "PersistenceManager", lambda _cfg: manager_stub)
        monkeypatch.setattr(artifacts, "_sha256", lambda _path: "digest-err")
        monkeypatch.setattr(artifacts, "validate_onnx", lambda _path: True)

        request = artifacts.ArtifactUpdateRequest(
            model_id="model-err",
            artifact_path=artifact_path,
            artifact_format="onnx",
            validate=True,
        )
        result = artifacts.update_model_artifact(
            request=request,
            registry_config=PersistenceConfig(
                backend=BackendType.JSON,
                json_path=tmp_path,
            ),
        )

        assert result["model_id"] == "model-err"
        assert result["error"] == "db-write-failed"
        assert counter.statuses == ["error"]
        assert counter.increments == 1

    def test_update_model_artifact_returns_error_when_session_missing(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        counter = _CounterRecorder()
        manager_stub = SimpleNamespace(get_session=lambda: None)
        artifact_path = tmp_path / "model.onnx"
        artifact_path.write_bytes(b"model")

        monkeypatch.setattr(artifacts, "_artifact_updates_total", counter)
        monkeypatch.setattr(artifacts, "PersistenceManager", lambda _cfg: manager_stub)
        monkeypatch.setattr(artifacts, "_sha256", lambda _path: "digest-none")
        monkeypatch.setattr(artifacts, "validate_onnx", lambda _path: True)

        request = artifacts.ArtifactUpdateRequest(
            model_id="model-none",
            artifact_path=artifact_path,
            artifact_format="onnx",
            validate=True,
        )
        result = artifacts.update_model_artifact(
            request=request,
            registry_config=PersistenceConfig(
                backend=BackendType.JSON,
                json_path=tmp_path,
            ),
        )

        assert result["model_id"] == "model-none"
        assert "error" in result
        assert counter.statuses == ["error"]


class TestLineageManagerModule:
    def test_get_lineage_table_raises_when_bind_missing(self) -> None:
        manager = LineageManager()
        session = MagicMock()
        session.get_bind.return_value = None
        with pytest.raises(RuntimeError, match="Failed to resolve database bind"):
            manager._get_lineage_table(session)

    def test_get_lineage_table_caches_table_per_bind(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        manager = LineageManager()
        session = MagicMock()
        bind = object()
        session.get_bind.return_value = bind
        table_sentinel = object()
        table_factory = MagicMock(return_value=table_sentinel)
        monkeypatch.setattr("ml.registry.lineage_manager.Table", table_factory)

        first = manager._get_lineage_table(session)
        second = manager._get_lineage_table(session)
        assert first is table_sentinel
        assert second is table_sentinel
        assert table_factory.call_count == 1

    def test_link_lineage_json_backend_trims_to_limit(self) -> None:
        manager = LineageManager()
        manager._lineage = [{"parent_dataset_id": f"old-{idx}"} for idx in range(5000)]
        manager.link_lineage(
            child_dataset_id="child",
            parent_ids=["parent-new"],
            transform_id="transform",
            ts_range={"start_ns": 1, "end_ns": 2},
            params={"lookback": 10},
            persistence=_json_persistence(),
        )

        assert len(manager._lineage) == 5000
        assert manager._lineage[0]["parent_dataset_id"] == "old-1"
        assert manager._lineage[-1]["parent_dataset_id"] == "parent-new"

    def test_link_lineage_postgres_backend_commits_after_last_parent(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        manager = LineageManager()
        session = MagicMock()
        table = MagicMock()
        insert_stmt = object()
        table.insert.return_value.values.return_value = insert_stmt
        monkeypatch.setattr(manager, "_get_lineage_table", lambda _session: table)
        persistence = SimpleNamespace(
            config=SimpleNamespace(backend=BackendType.POSTGRES),
            get_session=MagicMock(side_effect=[session, session]),
        )

        manager.link_lineage(
            child_dataset_id="child",
            parent_ids=["parent-1", "parent-2"],
            transform_id="transform",
            ts_range={"start_ns": 1, "end_ns": 2},
            params={"lookback": 5},
            persistence=persistence,
        )

        assert session.execute.call_count == 2
        session.commit.assert_called_once()
        session.close.assert_called_once()

    def test_link_lineage_postgres_backend_raises_when_session_missing(self) -> None:
        manager = LineageManager()
        persistence = SimpleNamespace(
            config=SimpleNamespace(backend=BackendType.POSTGRES),
            get_session=lambda: None,
        )
        with pytest.raises(RuntimeError, match="Failed to get database session"):
            manager.link_lineage(
                child_dataset_id="child",
                parent_ids=["parent-1"],
                transform_id="transform",
                ts_range={"start_ns": 1, "end_ns": 2},
                params={},
                persistence=persistence,
            )

    def test_link_lineage_postgres_backend_rolls_back_on_execute_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        manager = LineageManager()
        session = MagicMock()
        session.execute.side_effect = RuntimeError("insert-failed")
        table = MagicMock()
        table.insert.return_value.values.return_value = object()
        monkeypatch.setattr(manager, "_get_lineage_table", lambda _session: table)
        persistence = _postgres_persistence(session)

        with pytest.raises(RuntimeError, match="insert-failed"):
            manager.link_lineage(
                child_dataset_id="child",
                parent_ids=["parent-1"],
                transform_id="transform",
                ts_range={"start_ns": 1, "end_ns": 2},
                params={},
                persistence=persistence,
            )

        session.rollback.assert_called_once()

    def test_iter_lineage_returns_empty_when_persistence_is_none(self) -> None:
        manager = LineageManager()
        assert list(manager.iter_lineage()) == []

    def test_iter_lineage_json_filters_sorts_and_limits(self) -> None:
        manager = LineageManager()
        manager._lineage = [
            {
                "transform_id": "t-older",
                "child_dataset_id": "child-a",
                "parent_dataset_id": "parent-a",
                "ts_range": json.dumps({"start_ns": 1, "end_ns": 2}),
                "parameters": json.dumps({"lookback": 10}),
                "created_at": 10.0,
            },
            {
                "transform_id": "t-newer",
                "child_dataset_id": "child-a",
                "parent_dataset_id": "parent-a",
                "ts_range": "{invalid",
                "parameters": "{invalid",
                "created_at": 20.0,
            },
        ]

        records = list(
            manager.iter_lineage(
                child="child-a",
                parent="parent-a",
                limit=1,
                persistence=_json_persistence(),
            ),
        )
        assert len(records) == 1
        assert records[0].transform_id == "t-newer"
        assert records[0].ts_range == {}
        assert records[0].parameters == {}

    def test_iter_lineage_json_skips_non_matching_child_and_parent(self) -> None:
        manager = LineageManager()
        manager._lineage = [
            {
                "transform_id": "t-child-miss",
                "child_dataset_id": "child-x",
                "parent_dataset_id": "parent-a",
                "ts_range": {"start_ns": 1, "end_ns": 2},
                "parameters": {"lookback": 1},
                "created_at": 1.0,
            },
            {
                "transform_id": "t-parent-miss",
                "child_dataset_id": "child-a",
                "parent_dataset_id": "parent-x",
                "ts_range": {"start_ns": 3, "end_ns": 4},
                "parameters": {"lookback": 2},
                "created_at": 2.0,
            },
        ]
        records = list(
            manager.iter_lineage(
                child="child-a",
                parent="parent-a",
                persistence=_json_persistence(),
            ),
        )
        assert records == []

    def test_iter_lineage_postgres_raises_when_session_missing(self) -> None:
        manager = LineageManager()
        persistence = SimpleNamespace(
            config=SimpleNamespace(backend=BackendType.POSTGRES),
            get_session=lambda: None,
        )
        with pytest.raises(RuntimeError, match="Failed to get database session"):
            list(manager.iter_lineage(persistence=persistence))

    def test_iter_lineage_postgres_builds_records_from_rows(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        manager = LineageManager()
        session = MagicMock()
        fake_stmt = _FakeSelect()
        monkeypatch.setattr("ml.registry.lineage_manager.select", lambda *_args: fake_stmt)
        table = _fake_lineage_table()
        monkeypatch.setattr(manager, "_get_lineage_table", lambda _session: table)
        session.execute.return_value.all.return_value = [
            {
                "transform_id": "t1",
                "child_dataset_id": "child-a",
                "parent_dataset_id": "parent-a",
                "ts_range": json.dumps({"start_ns": 1, "end_ns": 2}),
                "parameters": json.dumps({"lookback": 10}),
                "created_at": 12.5,
            },
        ]

        records = list(
            manager.iter_lineage(
                child="child-a",
                parent="parent-a",
                limit=1,
                persistence=_postgres_persistence(session),
            ),
        )
        assert len(records) == 1
        assert records[0].transform_id == "t1"
        assert len(fake_stmt.where_calls) == 2
        assert fake_stmt.limit_calls == [1]
        session.close.assert_called_once()

    def test_lineage_from_row_handles_mapping_and_invalid_payloads(self) -> None:
        row = _Row(
            {
                "transform_id": "t-map",
                "child_dataset_id": "child",
                "parent_dataset_id": "parent",
                "ts_range": json.dumps({"start_ns": 5, "end_ns": 9}),
                "parameters": json.dumps({"alpha": 1}),
                "created_at": "4.5",
            },
        )
        mapped = LineageManager._lineage_from_row(row)
        assert mapped.created_at == 4.5
        assert mapped.ts_range == {"start_ns": 5, "end_ns": 9}
        assert mapped.parameters == {"alpha": 1}

        invalid = LineageManager._lineage_from_row(
            {
                "transform_id": "t-invalid",
                "child_dataset_id": "child",
                "parent_dataset_id": "parent",
                "ts_range": "{bad",
                "parameters": "{bad",
                "created_at": None,
            },
        )
        assert invalid.created_at == 0.0
        assert invalid.ts_range == {}
        assert invalid.parameters == {}

    def test_lineage_from_row_handles_dict_payloads(self) -> None:
        record = LineageManager._lineage_from_row(
            {
                "transform_id": "t-dict",
                "child_dataset_id": "child",
                "parent_dataset_id": "parent",
                "ts_range": {"start_ns": 7, "end_ns": 9},
                "parameters": {"beta": "on"},
                "created_at": 2.5,
            },
        )
        assert record.ts_range == {"start_ns": 7, "end_ns": 9}
        assert record.parameters == {"beta": "on"}


class TestModelRegistryShim:
    def test_model_registry_shim_exports_facade_aliases(self) -> None:
        model_registry = importlib.import_module("ml.registry.model_registry")
        model_registry_facade = importlib.import_module("ml.registry.model_registry_facade")
        assert model_registry.ModelRegistry is model_registry_facade.ModelRegistry
        assert model_registry.ModelRegistryFacade is model_registry_facade.ModelRegistryFacade
        assert model_registry.__all__ == ["ModelRegistry", "ModelRegistryFacade"]


class TestABTestingManagerModule:
    def test_configure_ab_test_validates_model_count(self) -> None:
        module = _ab_testing_module()
        manager = module.ABTestingManager(models={}, deployments={})
        assert manager.configure_ab_test(["model-1"], 0.5, 1, "ml_signal_actor") is None

    def test_configure_ab_test_returns_none_when_model_missing(self) -> None:
        module = _ab_testing_module()
        models = {"model-a": _make_model_info("model-a")}
        manager = module.ABTestingManager(models=models, deployments={})
        assert (
            manager.configure_ab_test(["model-a", "missing"], 0.5, 1, "ml_signal_actor")
            is None
        )

    def test_configure_ab_test_updates_model_state_and_deployments(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        module = _ab_testing_module()
        monkeypatch.setattr(module.time, "time", lambda: 100.0)
        save_callback = MagicMock()
        models = {
            "model-a": _make_model_info("model-a", deployed_to=["ml_signal_actor"]),
            "model-b": _make_model_info("model-b"),
        }
        deployments: dict[str, list[str]] = {}
        manager = module.ABTestingManager(
            models=models,
            deployments=deployments,
            save_callback=save_callback,
        )

        config = manager.configure_ab_test(
            models=["model-a", "model-b"],
            split_ratio=0.7,
            duration_hours=2,
            target="ml_signal_actor",
        )

        assert config is not None
        assert config["start_time"] == 100.0
        assert config["end_time"] == 7300.0
        assert models["model-a"].deployment_status == DeploymentStatus.TESTING
        assert models["model-b"].deployment_status == DeploymentStatus.TESTING
        assert models["model-a"].deployed_to.count("ml_signal_actor") == 1
        assert models["model-b"].deployed_to == ["ml_signal_actor"]
        assert deployments["ml_signal_actor"] == ["model-a", "model-b"]
        assert len(manager._ab_tests) == 1
        save_callback.assert_called_once()

    def test_compare_models_skips_missing_and_sorts_descending(self) -> None:
        module = _ab_testing_module()
        models = {
            "model-a": _make_model_info(
                "model-a",
                version="1.0.0",
                performance_history=[{"accuracy": 0.70}, {"accuracy": 0.82}],
            ),
            "model-b": _make_model_info(
                "model-b",
                version="1.1.0",
                performance_history=[{"accuracy": 0.78}],
            ),
            "model-c": _make_model_info("model-c", performance_history=[{"loss": 0.2}]),
        }
        manager = module.ABTestingManager(models=models, deployments={})

        comparison = manager.compare_models(
            ["missing", "model-c", "model-a", "model-b"],
            "accuracy",
        )

        assert comparison is not None
        assert comparison["metric"] == "accuracy"
        assert comparison["best_model"] == "model-a"
        assert [rank["model_id"] for rank in comparison["rankings"]] == ["model-a", "model-b"]

    def test_compare_models_returns_none_when_no_metric_values(self) -> None:
        module = _ab_testing_module()
        manager = module.ABTestingManager(
            models={"model-a": _make_model_info("model-a", performance_history=[{"loss": 1.0}])},
            deployments={},
        )
        assert manager.compare_models(["model-a"], "accuracy") is None

    def test_compare_models_statistically_requires_two_models(self) -> None:
        module = _ab_testing_module()
        manager = module.ABTestingManager(
            models={"model-a": _make_model_info("model-a")},
            deployments={},
        )
        assert manager.compare_models_statistically(["model-a"], "accuracy") is None

    def test_compare_models_statistically_returns_none_when_samples_missing(self) -> None:
        module = _ab_testing_module()
        manager = module.ABTestingManager(
            models={
                "model-a": _make_model_info("model-a", performance_history=[{"loss": 1.0}]),
                "model-b": _make_model_info("model-b", performance_history=[{"accuracy": 0.8}]),
            },
            deployments={},
        )
        assert manager.compare_models_statistically(["model-a", "model-b"], "accuracy") is None

    def test_compare_models_statistically_includes_model_metadata(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        module = _ab_testing_module()
        welch_mock = MagicMock(
            return_value={
                "statistically_significant": True,
                "p_value_approx": 0.01,
                "relative_improvement": 0.05,
            },
        )
        monkeypatch.setattr(module, "welch_t_test", welch_mock)
        manager = module.ABTestingManager(
            models={
                "model-a": _make_model_info(
                    "model-a",
                    performance_history=[{"accuracy": 0.80}, {"accuracy": 0.82}],
                ),
                "model-b": _make_model_info(
                    "model-b",
                    performance_history=[{"accuracy": 0.76}, {"accuracy": 0.77}],
                ),
            },
            deployments={},
        )

        result = manager.compare_models_statistically(["model-a", "model-b"], "accuracy")

        assert result is not None
        assert result["model_a"] == "model-a"
        assert result["model_b"] == "model-b"
        assert result["metric"] == "accuracy"
        welch_mock.assert_called_once()

    def test_run_ab_test_returns_empty_when_configuration_fails(self) -> None:
        module = _ab_testing_module()
        manager = module.ABTestingManager(models={}, deployments={})
        assert manager.run_ab_test("model-a", "model-b", 0.5, 1.0, "ml_signal_actor") == ""

    def test_run_track_and_analyze_ab_test_flow(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        module = _ab_testing_module()
        monkeypatch.setattr(module.time, "time", lambda: 200.0)
        monkeypatch.setattr(
            module,
            "welch_t_test",
            lambda _a, _b: {
                "relative_improvement": 0.10,
                "statistically_significant": True,
                "p_value_approx": 0.02,
            },
        )
        manager = module.ABTestingManager(
            models={
                "model-a": _make_model_info("model-a"),
                "model-b": _make_model_info("model-b"),
            },
            deployments={},
        )

        test_id = manager.run_ab_test(
            model_a_id="model-a",
            model_b_id="model-b",
            split_ratio=0.5,
            duration_hours=2.0,
            target="ml_signal_actor",
        )
        assert test_id == "ab_test_200"

        manager.track_ab_test_metric(test_id, "model-a", 0.70)
        manager.track_ab_test_metric(test_id, "model-a", 0.80)
        manager.track_ab_test_metric(test_id, "model-b", 0.90)
        manager.track_ab_test_metric("missing", "model-a", 1.0)
        manager.track_ab_test_metric(test_id, "missing", 1.0)

        analysis = manager.analyze_ab_test(test_id)
        assert analysis is not None
        assert analysis["control_model"] == "model-a"
        assert analysis["treatment_model"] == "model-b"
        assert analysis["control_mean"] == pytest.approx(0.75)
        assert analysis["treatment_mean"] == pytest.approx(0.90)
        assert analysis["relative_improvement"] == pytest.approx(0.10)
        assert analysis["statistical_significance"] is True
        assert analysis["p_value"] == pytest.approx(0.02)

    def test_analyze_ab_test_returns_none_for_unknown_or_incomplete_tests(self) -> None:
        module = _ab_testing_module()
        manager = module.ABTestingManager(models={}, deployments={})
        assert manager.analyze_ab_test("missing") is None

        manager._ab_test_metrics["one-model"] = {"model-a": [0.8]}
        assert manager.analyze_ab_test("one-model") is None

        manager._ab_test_metrics["empty-samples"] = {"model-a": [], "model-b": [0.7]}
        assert manager.analyze_ab_test("empty-samples") is None


class TestCanaryDeploymentManagerModule:
    def test_start_canary_deployment_raises_for_unknown_model(self) -> None:
        manager = CanaryDeploymentManager(models={}, ab_testing_manager=None)
        with pytest.raises(ValueError, match="not found"):
            manager.start_canary_deployment(
                model_id="missing",
                target="ml_signal_actor",
                config=CanaryConfig(),
            )

    def test_start_canary_deployment_uses_explicit_baseline_and_updates_state(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("ml.registry.canary_deployment_mgr.time.time", lambda: 50.0)
        save_callback = MagicMock()
        models = {
            "canary": _make_model_info("canary"),
            "baseline": _make_model_info(
                "baseline",
                deployment_status=DeploymentStatus.ACTIVE,
                deployed_to=["ml_signal_actor"],
                performance_metrics={"accuracy": 0.93},
            ),
        }
        manager = CanaryDeploymentManager(
            models=models,
            ab_testing_manager=MagicMock(),
            save_callback=save_callback,
        )

        deployment_id = manager.start_canary_deployment(
            model_id="canary",
            target="ml_signal_actor",
            config=CanaryConfig(success_metric="accuracy"),
            baseline_model_id="baseline",
        )

        assert deployment_id == "canary_50_canary"
        canary = manager.get_canary_deployment(deployment_id)
        assert canary is not None
        assert canary.baseline_model_id == "baseline"
        assert canary.baseline_performance == pytest.approx(0.93)
        assert models["canary"].deployment_status == DeploymentStatus.TESTING
        assert models["canary"].metadata["canary_deployment"] == deployment_id
        save_callback.assert_called_once()

    def test_start_canary_deployment_detects_active_baseline_when_omitted(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("ml.registry.canary_deployment_mgr.time.time", lambda: 60.0)
        models = {
            "candidate": _make_model_info("candidate"),
            "active-prod": _make_model_info(
                "active-prod",
                deployment_status=DeploymentStatus.ACTIVE,
                deployed_to=["ml_signal_actor"],
                performance_metrics={"accuracy": 0.88},
            ),
        }
        manager = CanaryDeploymentManager(models=models, ab_testing_manager=None)

        deployment_id = manager.start_canary_deployment(
            model_id="candidate",
            target="ml_signal_actor",
            config=CanaryConfig(success_metric="accuracy"),
        )
        canary = manager.get_canary_deployment(deployment_id)
        assert canary is not None
        assert canary.baseline_model_id == "active-prod"
        assert canary.baseline_performance == pytest.approx(0.88)

    def test_update_and_evaluate_canary_metrics(self) -> None:
        models = {"candidate": _make_model_info("candidate")}
        manager = CanaryDeploymentManager(models=models, ab_testing_manager=None)
        assert manager.evaluate_canary("missing") == (False, "deployment_not_found")
        assert manager.evaluate_canary_for_rollback("missing") == (
            False,
            "deployment_not_found",
        )

        deployment_id = manager.start_canary_deployment(
            model_id="candidate",
            target="ml_signal_actor",
            config=CanaryConfig(min_samples=1, monitoring_duration_hours=0.0),
        )
        manager.update_canary_metrics("missing", metric_value=0.0)  # no-op branch
        manager.update_canary_metrics(deployment_id, metric_value=0.8)

        assert manager.evaluate_canary(deployment_id) == (True, "monitoring_period_complete")
        assert manager.evaluate_canary_for_rollback(deployment_id) == (False, "metrics_acceptable")

    def test_auto_promote_canary_handles_missing_success_and_failure(self) -> None:
        deploy_callback = MagicMock(return_value=True)
        retire_callback = MagicMock()
        models = {
            "candidate": _make_model_info("candidate"),
            "baseline": _make_model_info("baseline"),
        }
        manager = CanaryDeploymentManager(
            models=models,
            ab_testing_manager=None,
            deploy_callback=deploy_callback,
            retire_callback=retire_callback,
        )
        assert manager.auto_promote_canary("missing") is False

        deployment_id = manager.start_canary_deployment(
            model_id="candidate",
            target="ml_signal_actor",
            config=CanaryConfig(),
            baseline_model_id="baseline",
        )
        assert manager.auto_promote_canary(deployment_id) is True
        deployment = manager.get_canary_deployment(deployment_id)
        assert deployment is not None
        assert deployment.status == "promoted"
        retire_callback.assert_called_once_with("baseline")

        failing = CanaryDeploymentManager(
            models={"candidate": _make_model_info("candidate")},
            ab_testing_manager=None,
            deploy_callback=MagicMock(return_value=False),
        )
        failing_id = failing.start_canary_deployment(
            model_id="candidate",
            target="ml_signal_actor",
            config=CanaryConfig(),
        )
        assert failing.auto_promote_canary(failing_id) is False
        failing_deployment = failing.get_canary_deployment(failing_id)
        assert failing_deployment is not None
        assert failing_deployment.status == "active"

    def test_start_gradual_rollout_and_get_status(self) -> None:
        ab_testing_mgr = MagicMock()
        manager = CanaryDeploymentManager(
            models={
                "current": _make_model_info("current"),
                "new": _make_model_info("new"),
            },
            ab_testing_manager=ab_testing_mgr,
        )
        with pytest.raises(ValueError, match="not found"):
            manager.start_gradual_rollout(
                current_model_id="missing",
                new_model_id="new",
                target="ml_signal_actor",
                stages=[0.1, 0.5, 1.0],
                stage_duration_minutes=30,
            )

        rollout_id = manager.start_gradual_rollout(
            current_model_id="current",
            new_model_id="new",
            target="ml_signal_actor",
            stages=[0.1, 0.5, 1.0],
            stage_duration_minutes=30,
        )
        status = manager.get_rollout_status(rollout_id)
        assert status is not None
        assert status["current_stage"] == 0
        assert status["traffic_split"] == pytest.approx(0.1)
        assert manager.get_rollout_status("missing") is None

        assert ab_testing_mgr.configure_ab_test.call_count == 1
        kwargs = ab_testing_mgr.configure_ab_test.call_args.kwargs
        assert kwargs["split_ratio"] == pytest.approx(0.9)
        assert kwargs["duration_hours"] == 1800

        manager_without_ab = CanaryDeploymentManager(
            models={
                "current": _make_model_info("current"),
                "new": _make_model_info("new"),
            },
            ab_testing_manager=None,
        )
        manager_without_ab.start_gradual_rollout(
            current_model_id="current",
            new_model_id="new",
            target="ml_signal_actor",
            stages=[],
            stage_duration_minutes=10,
        )

    def test_advance_rollout_stage_paths(self) -> None:
        ab_testing_mgr = MagicMock()
        manager = CanaryDeploymentManager(
            models={
                "current": _make_model_info("current"),
                "new": _make_model_info("new"),
            },
            ab_testing_manager=ab_testing_mgr,
        )
        assert manager.advance_rollout_stage("missing") is False

        rollout_id = manager.start_gradual_rollout(
            current_model_id="current",
            new_model_id="new",
            target="ml_signal_actor",
            stages=[0.2, 0.6, 1.0],
            stage_duration_minutes=30,
        )
        assert manager.advance_rollout_stage(rollout_id) is True
        assert ab_testing_mgr.configure_ab_test.call_count == 2
        kwargs = ab_testing_mgr.configure_ab_test.call_args.kwargs
        assert kwargs["split_ratio"] == pytest.approx(0.4)
        assert kwargs["duration_hours"] == 1

        rollout = manager._rollout_plans[rollout_id]
        rollout.current_stage = len(rollout.stages) - 1
        assert manager.advance_rollout_stage(rollout_id) is False


class _RaisingMetadata(dict[str, Any]):
    def items(self) -> Any:  # type: ignore[override]
        raise RuntimeError("metadata-items-failed")


class TestModelDeploymentManagerModule:
    def test_model_deployment_protocol_methods_are_callable(self) -> None:
        protocol_self = object()
        assert (
            ModelDeploymentManagerProtocol.deploy_model(protocol_self, "model-a", "ml_signal_actor")
            is None
        )
        assert (
            ModelDeploymentManagerProtocol.rollback(
                protocol_self,
                "ml_signal_actor",
                "model-a",
            )
            is None
        )
        assert ModelDeploymentManagerProtocol.retire_model(protocol_self, "model-a") is None
        assert (
            ModelDeploymentManagerProtocol.hot_reload_model(
                protocol_self,
                "ml_signal_actor",
                "model-b",
            )
            is None
        )
        assert ModelDeploymentManagerProtocol.get_active_models(protocol_self) is None
        assert ModelDeploymentManagerProtocol.get_all_models(protocol_self) is None
        assert ModelDeploymentManagerProtocol.get_model(protocol_self, "model-a") is None
        assert (
            ModelDeploymentManagerProtocol.get_models_by_role(
                protocol_self,
                ModelRole.INFERENCE,
            )
            is None
        )
        assert (
            ModelDeploymentManagerProtocol.get_models_by_data_requirements(
                protocol_self,
                DataRequirements.L1_ONLY,
            )
            is None
        )
        assert ModelDeploymentManagerProtocol.get_model_lineage(protocol_self, "model-a") is None
        assert (
            ModelDeploymentManagerProtocol.track_performance(
                protocol_self,
                "model-a",
                {"accuracy": 0.9},
            )
            is None
        )
        assert (
            ModelDeploymentManagerProtocol.update_metadata(
                protocol_self,
                "model-a",
                {"key": "value"},
            )
            is None
        )
        assert (
            ModelDeploymentManagerProtocol.get_performance_history(protocol_self, "model-a")
            is None
        )
        assert (
            ModelDeploymentManagerProtocol.list_compatible(
                protocol_self,
                "schema-hash",
                role=ModelRole.INFERENCE,
                architecture="xgboost",
            )
            is None
        )
        assert (
            ModelDeploymentManagerProtocol.resolve_latest(
                protocol_self,
                ModelRole.INFERENCE,
                "xgboost",
                "schema-hash",
            )
            is None
        )

    def test_deploy_model_returns_false_when_model_missing(self) -> None:
        manager = ModelDeploymentManager(models={}, deployments={})
        assert manager.deploy_model("missing", "ml_signal_actor") is False

    def test_deploy_model_updates_state_and_optional_config(self) -> None:
        save_callback = MagicMock()
        models = {"model-a": _make_model_info("model-a")}
        deployments: dict[str, list[str]] = {"ml_signal_actor": ["stale-model"]}
        manager = ModelDeploymentManager(
            models=models,
            deployments=deployments,
            save_callback=save_callback,
        )

        success = manager.deploy_model(
            model_id="model-a",
            target="ml_signal_actor",
            config={"traffic_percentage": 100.0},
        )

        assert success is True
        assert models["model-a"].deployment_status == DeploymentStatus.ACTIVE
        assert models["model-a"].deployed_to == ["ml_signal_actor"]
        assert deployments["ml_signal_actor"] == ["model-a"]
        assert models["model-a"].metadata["deployment_config"] == {"traffic_percentage": 100.0}
        save_callback.assert_called_once()

        fresh_models = {"model-b": _make_model_info("model-b")}
        fresh_deployments: dict[str, list[str]] = {}
        fresh_manager = ModelDeploymentManager(
            models=fresh_models,
            deployments=fresh_deployments,
        )
        assert fresh_manager.deploy_model("model-b", "ml_strategy") is True
        assert fresh_deployments["ml_strategy"] == ["model-b"]
        assert "deployment_config" not in fresh_models["model-b"].metadata

    def test_query_helpers_and_compatibility_filters(self) -> None:
        models = {
            "model-a": _make_model_info(
                "model-a",
                version="1.0.0",
                role=ModelRole.INFERENCE,
                architecture="xgboost",
                feature_schema_hash="schema-a",
                deployment_status=DeploymentStatus.ACTIVE,
            ),
            "model-b": _make_model_info(
                "model-b",
                version="1.1.0",
                role=ModelRole.INFERENCE,
                architecture="xgboost",
                feature_schema_hash="schema-a",
                data_requirements=DataRequirements.STREAMING,
            ),
            "model-c": _make_model_info(
                "model-c",
                role=ModelRole.STUDENT,
                architecture="lightgbm",
                feature_schema_hash="schema-b",
            ),
        }
        manager = ModelDeploymentManager(models=models, deployments={})

        assert len(manager.get_active_models()) == 1
        assert len(manager.get_all_models()) == 3
        assert manager.get_model("model-a") is models["model-a"]
        assert manager.get_model("missing") is None
        assert manager.get_models_by_role(ModelRole.INFERENCE) == [models["model-a"], models["model-b"]]
        assert manager.get_models_by_data_requirements(DataRequirements.STREAMING) == [models["model-b"]]
        assert manager.list_compatible("schema-a") == [models["model-a"], models["model-b"]]
        assert manager.list_compatible("schema-a", architecture="xgboost") == [
            models["model-a"],
            models["model-b"],
        ]
        assert manager.list_compatible("schema-a", role=ModelRole.STUDENT) == []
        assert manager.resolve_latest(
            role=ModelRole.INFERENCE,
            architecture="xgboost",
            schema_hash="schema-a",
        ) is models["model-b"]
        assert manager.resolve_latest(
            role=ModelRole.STUDENT,
            architecture="xgboost",
            schema_hash="schema-a",
        ) is None

    def test_get_model_lineage_handles_missing_and_parent_child_chain(self) -> None:
        models = {
            "parent": _make_model_info("parent"),
            "mid": _make_model_info("mid", parent_id="parent", children_ids=["child"]),
            "child": _make_model_info("child", parent_id="mid"),
        }
        manager = ModelDeploymentManager(models=models, deployments={})
        assert manager.get_model_lineage("missing") == []
        lineage = manager.get_model_lineage("mid")
        assert [model.manifest.model_id for model in lineage] == ["parent", "mid", "child"]

    def test_track_performance_adds_timestamp_and_persists(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("ml.registry.model_deployment_mgr.time.time", lambda: 321.0)
        save_callback = MagicMock()
        model = _make_model_info("model-a")
        manager = ModelDeploymentManager(
            models={"model-a": model},
            deployments={},
            save_callback=save_callback,
        )

        manager.track_performance("missing", {"accuracy": 0.5})
        manager.track_performance("model-a", {"accuracy": 0.9})
        manager.track_performance("model-a", {"accuracy": 0.95, "timestamp": 100.0})

        history = manager.get_performance_history("model-a")
        assert history[0]["timestamp"] == 321.0
        assert history[1]["timestamp"] == 100.0
        assert save_callback.call_count == 2

    def test_update_metadata_handles_not_found_merge_and_exception(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        save_callback = MagicMock()
        model = _make_model_info("model-a", metadata=cast(Any, "not-a-dict"))
        manager = ModelDeploymentManager(
            models={"model-a": model},
            deployments={},
            save_callback=save_callback,
        )

        manager.update_metadata("missing", {"key": "value"})
        manager.update_metadata("model-a", {"training_dataset_id": "dataset-1"})
        assert model.metadata == {"training_dataset_id": "dataset-1"}
        save_callback.assert_called_once()

        manager.update_metadata("model-a", cast(dict[str, Any], _RaisingMetadata({"x": 1})))
        assert any("Failed updating metadata for model-a" in record.message for record in caplog.records)

    def test_get_performance_history_returns_copy(self) -> None:
        model = _make_model_info("model-a", performance_history=[{"accuracy": 0.5}])
        manager = ModelDeploymentManager(models={"model-a": model}, deployments={})

        copied = manager.get_performance_history("model-a")
        copied.append({"accuracy": 1.0})
        assert manager.get_performance_history("model-a") == [{"accuracy": 0.5}]
        assert manager.get_performance_history("missing") == []

    def test_rollback_paths(self) -> None:
        save_callback = MagicMock()
        current = _make_model_info(
            "current",
            deployment_status=DeploymentStatus.ACTIVE,
            deployed_to=["ml_signal_actor"],
        )
        rollback = _make_model_info("rollback")
        models = {"current": current, "rollback": rollback}
        manager = ModelDeploymentManager(
            models=models,
            deployments={"ml_signal_actor": ["current"]},
            save_callback=save_callback,
        )

        assert manager.rollback("ml_signal_actor", "missing") is False
        assert manager.rollback("ml_signal_actor", "rollback") is True
        assert current.deployment_status == DeploymentStatus.INACTIVE
        assert "ml_signal_actor" not in current.deployed_to
        assert rollback.deployment_status == DeploymentStatus.ACTIVE
        assert rollback.deployed_to == ["ml_signal_actor"]
        assert manager._deployments["ml_signal_actor"] == ["rollback"]
        assert save_callback.call_count == 1

    def test_retire_model_paths(self) -> None:
        save_callback = MagicMock()
        deployed = _make_model_info(
            "model-a",
            deployment_status=DeploymentStatus.ACTIVE,
            deployed_to=["ml_signal_actor", "ml_strategy"],
        )
        manager = ModelDeploymentManager(
            models={"model-a": deployed},
            deployments={"ml_signal_actor": ["model-a"], "ml_strategy": ["model-a"]},
            save_callback=save_callback,
        )

        assert manager.retire_model("missing") is False
        assert manager.retire_model("model-a") is True
        assert deployed.deployment_status == DeploymentStatus.RETIRED
        assert deployed.deployed_to == []
        assert manager._deployments["ml_signal_actor"] == []
        assert manager._deployments["ml_strategy"] == []
        assert save_callback.call_count == 1

    def test_hot_reload_paths(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        manager = ModelDeploymentManager(
            models={"new": _make_model_info("new")},
            deployments={},
        )
        assert manager.hot_reload_model("ml_signal_actor", "missing") is False

        deploy_spy = MagicMock(return_value=True)
        manager.deploy_model = deploy_spy
        assert manager.hot_reload_model("ml_signal_actor", "new") is True
        deploy_spy.assert_called_once_with("new", "ml_signal_actor")

        current = _make_model_info(
            "current",
            deployment_status=DeploymentStatus.ACTIVE,
            deployed_to=["ml_signal_actor"],
            feature_schema_hash="schema-a",
        )
        replacement = _make_model_info("replacement", feature_schema_hash="schema-b")
        manager_with_active = ModelDeploymentManager(
            models={"current": current, "replacement": replacement},
            deployments={},
        )
        manager_with_active.deploy_model = MagicMock(return_value=True)
        manager_with_active.retire_model = MagicMock(return_value=True)
        assert manager_with_active.hot_reload_model("ml_signal_actor", "replacement") is True
        manager_with_active.retire_model.assert_called_once_with("current")
        assert any("Feature schema mismatch during hot reload" in rec.message for rec in caplog.records)

        manager_failure = ModelDeploymentManager(
            models={"current": current, "replacement": replacement},
            deployments={},
        )
        manager_failure.deploy_model = MagicMock(return_value=False)
        manager_failure.retire_model = MagicMock(return_value=True)
        assert manager_failure.hot_reload_model("ml_signal_actor", "replacement") is False
        manager_failure.retire_model.assert_not_called()
