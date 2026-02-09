from __future__ import annotations

import json
import threading
from dataclasses import replace
from itertools import count
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from typing import cast
from unittest.mock import MagicMock

import numpy as np
import pytest

from ml.config.events import EventStatus
from ml.config.events import Source
from ml.config.events import Stage
from ml.config.base import StatsConfig
from ml.registry._typing_utils import expect_bool
from ml.registry._typing_utils import expect_dict
from ml.registry._typing_utils import expect_dict_list
from ml.registry._typing_utils import expect_float
from ml.registry._typing_utils import expect_float_dict
from ml.registry._typing_utils import expect_optional_str
from ml.registry._typing_utils import expect_str
from ml.registry._typing_utils import expect_str_dict
from ml.registry._typing_utils import expect_str_list
from ml.registry.abstract_registry import AbstractRegistry
from ml.registry.contract_manager import ContractManager
from ml.registry.data_registry import DataRegistry
from ml.registry.data_registry import _is_legacy_dataset_id
from ml.registry.dataclasses import DataContract
from ml.registry.dataclasses import DatasetManifest
from ml.registry.dataclasses import DatasetType
from ml.registry.dataclasses import QualityFlag
from ml.registry.dataclasses import StorageKind
from ml.registry.dataclasses import ValidationRule
from ml.registry.dataclasses import ValidationRuleType
from ml.registry.event_manager import EventManager
from ml.registry.protocols import RegistryProtocol
from ml.registry.protocols import TypedRegistryProtocol
from ml.registry.common.manifest_defaults import resolve_primary_keys
from ml.registry.common.watermark_manager import WatermarkManagerComponent
from ml.registry.mixins import ArtifactMixin
from ml.registry.mixins import CacheMixin
from ml.registry.mixins import StageLifecycleMixin
from ml.registry.persistence import BackendType
from ml.registry.persistence import PersistenceConfig
from ml.registry.persistence import PersistenceManager
from ml.registry.statistics import calculate_sample_size
from ml.registry.statistics import compare_models
from ml.registry.statistics import welch_t_test
from ml.registry.utils import REGISTRY_PATH_ENV_VAR
from ml.registry.utils import assert_features_compatible
from ml.registry.utils import build_feature_schema
from ml.registry.utils import build_student_manifest
from ml.registry.utils import compute_dataset_schema_hash
from ml.registry.utils import get_default_registry_path
from ml.registry.watermark import Watermark


def _make_manifest(
    dataset_id: str = "dataset.features",
    constraints: dict[str, Any] | None = None,
) -> DatasetManifest:
    return DatasetManifest(
        dataset_id=dataset_id,
        dataset_type=DatasetType.FEATURES,
        storage_kind=StorageKind.PARQUET,
        location="/tmp/features.parquet",
        partitioning={},
        retention_days=7,
        schema={
            "instrument_id": "str",
            "ts_event": "int64",
            "ts_init": "int64",
            "close": "float64",
            "symbol": "str",
        },
        ts_field="ts_event",
        seq_field=None,
        primary_keys=["instrument_id", "ts_event"],
        schema_hash="",
        constraints=constraints or {},
        lineage=[],
        pipeline_signature="unit_test_pipeline",
        version="1.0.0",
    )


class _ManifestManagerStub:
    def __init__(self, manifest: DatasetManifest) -> None:
        self._manifest = manifest
        self.calls = 0

    def get_manifest(self, dataset_id: str, persistence: Any) -> DatasetManifest:
        del persistence
        self.calls += 1
        if dataset_id != self._manifest.dataset_id:
            raise ValueError(f"Unknown dataset {dataset_id}")
        return self._manifest


class _EntryWithStage:
    def __init__(self) -> None:
        self.stage: str = "initial"
        self.last_modified: float = 0.0


class _BrokenLastModifiedEntry:
    def __init__(self) -> None:
        self.stage: str = "initial"

    @property
    def last_modified(self) -> float:
        return 0.0

    @last_modified.setter
    def last_modified(self, value: float) -> None:
        del value
        raise RuntimeError("cannot set last_modified")


class _AbstractRegistryStub(AbstractRegistry):
    def __init__(self, persistence: PersistenceManager) -> None:
        super().__init__(persistence=persistence)

    def _health_snapshot(self) -> tuple[int, float | None]:
        return 3, 123.0


def _make_watermark_persistence_stub(
    *,
    backend: BackendType,
    session: Any | None = None,
) -> Any:
    return SimpleNamespace(
        backend=backend,
        _lock=threading.RLock(),
        _watermarks={},
        _save_registry=MagicMock(),
        persistence=SimpleNamespace(get_session=lambda: session),
        _watermark_from_row=DataRegistry._watermark_from_row,
    )


def _make_data_registry_json_backend(tmp_path: Path) -> DataRegistry:
    registry_dir = tmp_path / "registry_support_modules"
    return DataRegistry(
        registry_path=registry_dir,
        batch_save_interval=0.0,
        persistence_config=PersistenceConfig(
            backend=BackendType.JSON,
            json_path=registry_dir,
        ),
    )


class TestTypingUtils:
    def test_expect_str_and_optional_str(self) -> None:
        assert expect_str("abc", "field") == "abc"
        assert expect_optional_str(None, "field") is None
        assert expect_optional_str("xyz", "field") == "xyz"
        with pytest.raises(TypeError, match="field must be a string"):
            expect_str(123, "field")

    def test_expect_float_and_bool(self) -> None:
        assert expect_float(2, "n") == 2.0
        assert expect_float(1.5, "n") == 1.5
        assert expect_float(None, "n", default=2.5) == 2.5
        with pytest.raises(TypeError, match="n must be numeric"):
            expect_float(None, "n")
        with pytest.raises(TypeError, match="n must be numeric"):
            expect_float("bad", "n")

        assert expect_bool(True, "b") is True
        assert expect_bool(" yes ", "b") is True
        assert expect_bool("0", "b") is False
        assert expect_bool(None, "b", default=False) is False
        with pytest.raises(TypeError, match="b must be boolean"):
            expect_bool(None, "b")
        with pytest.raises(TypeError, match="b must be boolean-compatible"):
            expect_bool("maybe", "b")
        with pytest.raises(TypeError, match="b must be boolean-compatible"):
            expect_bool(1, "b")

    def test_expect_list_and_dict_helpers(self) -> None:
        assert expect_str_list(["a", "b"], "items") == ["a", "b"]
        assert expect_str_list(None, "items") == []
        with pytest.raises(TypeError, match="items\\[1\\] must be a string"):
            expect_str_list(["a", 1], "items")
        with pytest.raises(TypeError, match="items must be a sequence of strings"):
            expect_str_list("not-a-sequence", "items")

        assert expect_dict(None, "meta") == {}
        assert expect_dict({"k": 1}, "meta") == {"k": 1}
        with pytest.raises(TypeError, match="meta key must be a string"):
            expect_dict({1: "x"}, "meta")
        with pytest.raises(TypeError, match="meta must be a mapping"):
            expect_dict("invalid", "meta")

        assert expect_float_dict(None, "metrics") == {}
        assert expect_float_dict({"a": 1, "b": 2.5}, "metrics") == {"a": 1.0, "b": 2.5}
        with pytest.raises(TypeError, match="metrics values must be numeric"):
            expect_float_dict({"a": "bad"}, "metrics")
        with pytest.raises(TypeError, match="metrics must be a mapping of floats"):
            expect_float_dict("invalid", "metrics")

        assert expect_str_dict(None, "labels") == {}
        assert expect_str_dict({"a": "x", "b": "y"}, "labels") == {"a": "x", "b": "y"}
        with pytest.raises(TypeError, match="labels value must be a string"):
            expect_str_dict({"a": 1}, "labels")
        with pytest.raises(TypeError, match="labels must be a mapping of strings"):
            expect_str_dict("invalid", "labels")

        assert expect_dict_list(None, "records") == []
        assert expect_dict_list([{"a": 1}, {"b": 2}], "records") == [{"a": 1}, {"b": 2}]
        with pytest.raises(TypeError, match="records\\[0\\] must be a mapping"):
            expect_dict_list(["not-mapping"], "records")
        with pytest.raises(TypeError, match="records must be a sequence of mappings"):
            expect_dict_list("bad", "records")


class TestRegistryProtocols:
    def test_registry_protocol_stub_methods_are_callable(self) -> None:
        registry_protocol = cast(Any, RegistryProtocol)
        manifest = _make_manifest()

        assert (
            registry_protocol.emit_event(
                object(),
                "dataset.features",
                "EUR/USD",
                Stage.CATALOG_WRITTEN,
                Source.HISTORICAL,
                "run-1",
                1,
                2,
                3,
                EventStatus.SUCCESS,
            )
            is None
        )
        assert (
            registry_protocol.update_watermark(
                object(),
                "dataset.features",
                "EUR/USD",
                Source.HISTORICAL,
                1,
                2,
                99.0,
            )
            is None
        )
        assert registry_protocol.get_manifest(object(), "dataset.features") is None
        assert registry_protocol.get_contract(object(), "dataset.features") is None
        assert registry_protocol.register_dataset(object(), manifest) is None
        assert registry_protocol.update_manifest(object(), "dataset.features", {"version": "2.0.0"}) is None

    def test_typed_registry_protocol_stub_methods_are_callable(self) -> None:
        typed_protocol = cast(Any, TypedRegistryProtocol)

        assert typed_protocol.get(object(), "dataset.features") is None
        assert typed_protocol.save(object(), _make_manifest()) is None
        assert typed_protocol.delete(object(), "dataset.features") is None
        assert typed_protocol.list_manifests(object(), prefix="dataset", limit=2) is None
        assert typed_protocol.batch_save(object(), [_make_manifest()]) is None
        assert (
            typed_protocol.emit_event(
                object(),
                dataset_id="dataset.features",
                instrument_id="EUR/USD",
                stage=Stage.CATALOG_WRITTEN,
                source=Source.HISTORICAL,
                run_id="run-typed",
                ts_min=1,
                ts_max=2,
                count=3,
                status=EventStatus.SUCCESS,
            )
            is None
        )
        assert (
            typed_protocol.update_watermark(
                object(),
                dataset_id="dataset.features",
                instrument_id="EUR/USD",
                source=Source.HISTORICAL,
                last_success_ns=1,
                count=3,
                completeness_pct=100.0,
            )
            is None
        )


class TestAbstractRegistryModule:
    def test_json_helpers_and_health_status(self, tmp_path: Path) -> None:
        manager = PersistenceManager(
            PersistenceConfig(backend=BackendType.JSON, json_path=tmp_path / "registry"),
        )
        registry = _AbstractRegistryStub(manager)

        payload = {"status": "ok"}
        registry._json_save("state.json", payload)
        assert registry._json_load("state.json") == payload

        health = registry.get_health_status()
        assert health["component"] == "_AbstractRegistryStub"
        assert health["backend"] == "json"
        assert health["count"] == 3
        assert health["last_modified"] == 123.0

    def test_json_helpers_are_noop_for_non_json_backend(self) -> None:
        persistence_stub = cast(
            PersistenceManager,
            SimpleNamespace(
                config=SimpleNamespace(backend=BackendType.POSTGRES),
                load_json=MagicMock(return_value={"unexpected": True}),
                save_json=MagicMock(),
                log_audit=MagicMock(),
            ),
        )
        registry = _AbstractRegistryStub(persistence_stub)

        assert registry._json_load("ignored.json") is None
        registry._json_save("ignored.json", {"k": "v"})
        persistence_stub.load_json.assert_not_called()  # type: ignore[attr-defined]
        persistence_stub.save_json.assert_not_called()  # type: ignore[attr-defined]

    def test_log_audit_passthrough(self) -> None:
        persistence_stub = cast(
            PersistenceManager,
            SimpleNamespace(
                config=SimpleNamespace(backend=BackendType.JSON),
                load_json=MagicMock(),
                save_json=MagicMock(),
                log_audit=MagicMock(),
            ),
        )
        registry = _AbstractRegistryStub(persistence_stub)

        registry.log_audit(
            entity_type="feature",
            entity_id="fs_1",
            action="update",
            changes={"version": "2.0.0"},
            user_id="tester",
        )
        persistence_stub.log_audit.assert_called_once_with(  # type: ignore[attr-defined]
            entity_type="feature",
            entity_id="fs_1",
            action="update",
            changes={"version": "2.0.0"},
            user_id="tester",
        )


class TestMixins:
    def test_stage_lifecycle_set_stage_and_last_modified(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("ml.registry.mixins.time.time", lambda: 123.0)
        entry = _EntryWithStage()
        StageLifecycleMixin._set_stage(entry, "active")
        assert entry.stage == "active"
        assert entry.last_modified == 123.0

    def test_stage_lifecycle_handles_last_modified_setter_failure(self) -> None:
        entry = _BrokenLastModifiedEntry()
        StageLifecycleMixin._set_stage(entry, "active")
        assert entry.stage == "active"

    def test_artifact_mixin_attach_and_merge(self) -> None:
        container = SimpleNamespace()
        ArtifactMixin._attach_artifacts(container, {"model": "m.onnx"})
        assert container.artifacts == {"model": "m.onnx"}

        ArtifactMixin._attach_artifacts(container, {"report": "r.json"})
        assert container.artifacts == {"model": "m.onnx", "report": "r.json"}

    def test_cache_mixin_put_get_pop_and_lru(self, monkeypatch: pytest.MonkeyPatch) -> None:
        tick = count(1)
        monkeypatch.setattr("ml.registry.mixins.time.time", lambda: float(next(tick)))
        cache = CacheMixin(cache_size=2)

        cache._evict_lru()  # No-op on empty cache

        cache.cache_put("a", "A")
        cache.cache_put("b", "B")
        assert cache.cache_get("a") == "A"  # Refresh "a", making "b" LRU

        cache.cache_put("c", "C")
        assert cache.cache_get("b") is None
        assert cache.cache_get("a") == "A"
        assert cache.cache_get("c") == "C"

        cache.cache_pop("a")
        assert cache.cache_get("a") is None


class TestContractManager:
    def test_create_contract_from_manifest_builds_rules_and_threshold(self) -> None:
        manager = ContractManager()
        manifest = _make_manifest(
            constraints={
                "ranges": {"close": {"min": 0.0, "max": 100.0}, "ignored": {}},
                "nullability": {"close": False, "symbol": True},
                "regex": {"symbol": "^[A-Z]+$", "blank": ""},
                "null_rate_threshold": 0.2,
            },
        )

        contract = manager.create_contract_from_manifest(manifest)
        rule_types = [rule.rule_type for rule in contract.validation_rules]
        assert rule_types == [
            ValidationRuleType.RANGE,
            ValidationRuleType.NULLABILITY,
            ValidationRuleType.REGEX,
        ]
        assert all(rule.severity == QualityFlag.FAIL for rule in contract.validation_rules)
        assert contract.quality_thresholds == {"null_rate": 0.2}

    def test_create_contract_from_manifest_uses_default_rule_when_no_constraints(self) -> None:
        manager = ContractManager()
        contract = manager.create_contract_from_manifest(_make_manifest())
        assert len(contract.validation_rules) == 1
        assert contract.validation_rules[0].rule_type == ValidationRuleType.TYPE_CHECK
        assert contract.validation_rules[0].severity == QualityFlag.WARN

    def test_create_contract_from_manifest_ignores_invalid_null_rate_threshold(self) -> None:
        manager = ContractManager()
        manifest = _make_manifest(constraints={"null_rate_threshold": 1.2})
        contract = manager.create_contract_from_manifest(manifest)
        assert contract.quality_thresholds == {}

    def test_get_contract_uses_cache_after_first_fetch(self) -> None:
        manager = ContractManager()
        manifest = _make_manifest()
        manifest_manager = _ManifestManagerStub(manifest)
        persistence = object()

        first = manager.get_contract(manifest.dataset_id, manifest_manager, persistence)
        second = manager.get_contract(manifest.dataset_id, manifest_manager, persistence)
        assert first == second
        assert manifest_manager.calls == 1

    def test_contract_dict_round_trip(self) -> None:
        manager = ContractManager()
        contract = manager.create_contract_from_manifest(
            _make_manifest(
                constraints={"ranges": {"close": {"min": 0.0}}},
            ),
        )
        payload = manager._contract_to_dict(contract)
        rebuilt = manager._dict_to_contract(payload)
        assert rebuilt == contract


class TestRegistryDataclassValidations:
    def _rule(self) -> ValidationRule:
        return ValidationRule(
            rule_type=ValidationRuleType.TYPE_CHECK,
            field_name="close",
            parameters={"dtype": "float64"},
            severity=QualityFlag.WARN,
            description="close should be float64",
        )

    def test_validation_rule_rejects_invalid_configurations(self) -> None:
        with pytest.raises(ValueError, match="Invalid severity"):
            ValidationRule(
                rule_type=ValidationRuleType.TYPE_CHECK,
                field_name="close",
                parameters={},
                severity=QualityFlag.PASS,
                description="invalid severity",
            )

        with pytest.raises(ValueError, match="Range rule requires"):
            ValidationRule(
                rule_type=ValidationRuleType.RANGE,
                field_name="close",
                parameters={},
                severity=QualityFlag.FAIL,
                description="missing min max",
            )

        with pytest.raises(ValueError, match="Monotonicity rule requires"):
            ValidationRule(
                rule_type=ValidationRuleType.MONOTONICITY,
                field_name="ts_event",
                parameters={},
                severity=QualityFlag.FAIL,
                description="missing direction",
            )

        with pytest.raises(ValueError, match="Monotonicity direction must be"):
            ValidationRule(
                rule_type=ValidationRuleType.MONOTONICITY,
                field_name="ts_event",
                parameters={"direction": "sideways"},
                severity=QualityFlag.FAIL,
                description="invalid direction",
            )

        with pytest.raises(ValueError, match="Lateness rule requires"):
            ValidationRule(
                rule_type=ValidationRuleType.LATENESS,
                field_name="ts_event",
                parameters={},
                severity=QualityFlag.FAIL,
                description="missing max lateness",
            )

        with pytest.raises(ValueError, match="Regex rule requires"):
            ValidationRule(
                rule_type=ValidationRuleType.REGEX,
                field_name="instrument_id",
                parameters={"pattern": 123},
                severity=QualityFlag.FAIL,
                description="invalid regex",
            )

    def test_data_contract_rejects_invalid_configuration(self) -> None:
        valid_rule = self._rule()

        with pytest.raises(ValueError, match="Invalid enforcement_mode"):
            DataContract(
                contract_id="contract.invalid.mode",
                dataset_id="dataset.features",
                version="1.0.0",
                validation_rules=[valid_rule],
                enforcement_mode="invalid",
            )

        with pytest.raises(ValueError, match="at least one validation rule"):
            DataContract(
                contract_id="contract.empty",
                dataset_id="dataset.features",
                version="1.0.0",
                validation_rules=[],
            )

        with pytest.raises(ValueError, match="must be between 0 and 1"):
            DataContract(
                contract_id="contract.threshold",
                dataset_id="dataset.features",
                version="1.0.0",
                validation_rules=[valid_rule],
                quality_thresholds={"null_rate": 1.2},
            )

    def test_dataset_manifest_rejects_invalid_required_fields(self) -> None:
        manifest = _make_manifest()

        with pytest.raises(ValueError, match="Timestamp field"):
            replace(manifest, ts_field="missing_ts")

        with pytest.raises(ValueError, match="Sequence field"):
            replace(manifest, seq_field="sequence")

        with pytest.raises(ValueError, match="Primary key 'missing_pk'"):
            replace(manifest, primary_keys=["missing_pk"])

        with pytest.raises(ValueError, match="Nautilus required field 'ts_init'"):
            replace(
                manifest,
                schema={
                    "instrument_id": "str",
                    "ts_event": "int64",
                    "close": "float64",
                },
                primary_keys=["instrument_id", "ts_event"],
                schema_hash="",
            )

        with pytest.raises(ValueError, match="retention_days must be positive"):
            replace(manifest, retention_days=0)

    def test_dataset_manifest_compatibility_detects_hash_mismatch(self) -> None:
        manifest = _make_manifest()
        incompatible = replace(
            manifest,
            schema={
                "instrument_id": "str",
                "ts_event": "int64",
                "ts_init": "int64",
                "open": "float64",
            },
            primary_keys=["instrument_id", "ts_event"],
            schema_hash="",
        )
        assert manifest.is_compatible_with(incompatible) is False


class TestPersistenceModule:
    def test_persistence_config_sets_defaults_and_validates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("NAUTILUS_REGISTRY_DB_URL", raising=False)
        postgres = PersistenceConfig(backend=BackendType.POSTGRES)
        assert postgres.connection_string == "postgresql://postgres:postgres@localhost:5432/nautilus"

        with pytest.raises(ValueError, match="json_path is required"):
            PersistenceConfig(backend=BackendType.JSON, json_path=None)

    def test_manager_init_routes_postgres_backend_to_initializer(self) -> None:
        config = PersistenceConfig(
            backend=BackendType.POSTGRES,
            connection_string="postgresql://localhost/test",
        )
        with pytest.MonkeyPatch.context() as m:
            init_mock = MagicMock()
            m.setattr("ml.registry.persistence.PersistenceManager._init_postgres", init_mock)
            PersistenceManager(config)
            init_mock.assert_called_once()

    def test_init_postgres_validates_connection_string(self) -> None:
        config = PersistenceConfig(
            backend=BackendType.POSTGRES,
            connection_string="postgresql://localhost/test",
        )
        object.__setattr__(config, "connection_string", None)
        manager = cast(PersistenceManager, PersistenceManager.__new__(PersistenceManager))
        manager.config = config

        with pytest.raises(ValueError, match="Connection string is required"):
            manager._init_postgres()

    def test_init_postgres_sets_engine_session_factory_and_tables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        config = PersistenceConfig(
            backend=BackendType.POSTGRES,
            connection_string="postgresql://localhost/test",
            pool_size=7,
            max_overflow=3,
            echo=True,
        )
        manager = cast(PersistenceManager, PersistenceManager.__new__(PersistenceManager))
        manager.config = config
        manager._engine = None
        manager._session_factory = None

        engine = MagicMock()
        session_factory = MagicMock()
        create_all = MagicMock()
        monkeypatch.setattr("ml.registry.persistence.get_or_create_engine", MagicMock(return_value=engine))
        monkeypatch.setattr("ml.registry.persistence.sessionmaker", MagicMock(return_value=session_factory))
        monkeypatch.setattr("ml.registry.persistence.Base.metadata.create_all", create_all)

        manager._init_postgres()

        assert manager._engine is engine
        assert manager._session_factory is session_factory
        create_all.assert_called_once_with(engine)

    def test_get_session_and_close_cover_postgres_paths(self) -> None:
        config = PersistenceConfig(
            backend=BackendType.POSTGRES,
            connection_string="postgresql://localhost/test",
        )
        manager = cast(PersistenceManager, PersistenceManager.__new__(PersistenceManager))
        manager.config = config
        manager._engine = MagicMock()
        manager._session_factory = MagicMock(return_value="session")

        assert manager.get_session() == "session"
        manager.close()
        manager._engine.dispose.assert_called_once()

    def test_json_helpers_return_early_for_non_json_and_missing_path(self, tmp_path: Path) -> None:
        config_non_json = PersistenceConfig(
            backend=BackendType.JSON,
            json_path=tmp_path / "registry_non_json",
        )
        object.__setattr__(config_non_json, "backend", BackendType.POSTGRES)
        manager_non_json = cast(PersistenceManager, PersistenceManager.__new__(PersistenceManager))
        manager_non_json.config = config_non_json
        manager_non_json.save_json({"x": 1}, "ignored.json")
        assert manager_non_json.load_json("ignored.json") is None

        config_missing = PersistenceConfig(
            backend=BackendType.JSON,
            json_path=tmp_path / "registry_missing_path",
        )
        object.__setattr__(config_missing, "json_path", None)
        manager_missing = PersistenceManager(config_missing)
        manager_missing.save_json({"x": 1}, "ignored.json")
        assert manager_missing.load_json("ignored.json") is None

    def test_load_json_returns_none_for_missing_file(self, tmp_path: Path) -> None:
        manager = PersistenceManager(
            PersistenceConfig(backend=BackendType.JSON, json_path=tmp_path / "registry"),
        )
        assert manager.load_json("does_not_exist.json") is None

    def test_log_audit_postgres_and_json_branches(self, tmp_path: Path) -> None:
        config_pg = PersistenceConfig(
            backend=BackendType.POSTGRES,
            connection_string="postgresql://localhost/test",
        )
        manager_pg = cast(PersistenceManager, PersistenceManager.__new__(PersistenceManager))
        manager_pg.config = config_pg
        manager_pg._engine = None
        manager_pg._session_factory = None
        session = MagicMock()
        manager_pg.get_session = MagicMock(return_value=session)  # type: ignore[assignment]

        manager_pg.log_audit(
            entity_type="feature",
            entity_id="fs_1",
            action="register",
            changes={"version": "1.0.0"},
            user_id="tester",
        )
        session.add.assert_called_once()
        session.commit.assert_called_once()
        session.close.assert_called_once()

        manager_json = PersistenceManager(
            PersistenceConfig(backend=BackendType.JSON, json_path=tmp_path / "registry_json"),
        )
        (tmp_path / "registry_json").mkdir(parents=True, exist_ok=True)
        manager_json.log_audit(
            entity_type="feature",
            entity_id="fs_2",
            action="register",
            changes={"version": "1.0.1"},
        )
        audit_file = (tmp_path / "registry_json" / "audit_log.jsonl")
        assert audit_file.exists()

        config_json_missing_path = PersistenceConfig(
            backend=BackendType.JSON,
            json_path=tmp_path / "registry_json_missing",
        )
        object.__setattr__(config_json_missing_path, "json_path", None)
        manager_json_missing = PersistenceManager(config_json_missing_path)
        manager_json_missing.log_audit(entity_type="feature", entity_id="fs_3", action="skip")


class TestEventManager:
    def test_emit_event_no_persistence_is_noop(self) -> None:
        manager = EventManager()
        manager.emit_event(
            dataset_id="dataset.features",
            instrument_id="EUR/USD",
            stage=Stage.CATALOG_WRITTEN,
            source=Source.HISTORICAL,
            run_id="run-1",
            ts_min=1,
            ts_max=2,
            count=10,
            status=EventStatus.SUCCESS,
            persistence=None,
        )
        assert manager._events == []

    def test_emit_event_json_backend_appends_normalized_event(self) -> None:
        manager = EventManager()
        persistence = SimpleNamespace(
            config=SimpleNamespace(backend=BackendType.JSON),
        )
        manager.emit_event(
            dataset_id="dataset.features",
            instrument_id="EUR/USD",
            stage=Stage.CATALOG_WRITTEN,
            source=Source.HISTORICAL,
            run_id="run-1",
            ts_min=1,
            ts_max=2,
            count=10,
            status=EventStatus.SUCCESS,
            metadata=None,
            persistence=persistence,
        )

        assert len(manager._events) == 1
        event = manager._events[0]
        assert event["dataset_id"] == "dataset.features"
        assert event["stage"] == Stage.CATALOG_WRITTEN.value
        assert event["source"] == Source.HISTORICAL.value
        assert event["status"] == EventStatus.SUCCESS.value
        assert event["metadata"] == {}
        assert event["ts_event"] == 1

    def test_emit_event_json_backend_trims_to_last_ten_thousand(self) -> None:
        manager = EventManager()
        manager._events = [{"run_id": f"old-{idx}"} for idx in range(10000)]
        persistence = SimpleNamespace(
            config=SimpleNamespace(backend=BackendType.JSON),
        )

        manager.emit_event(
            dataset_id="dataset.features",
            instrument_id="EUR/USD",
            stage=Stage.CATALOG_WRITTEN,
            source=Source.HISTORICAL,
            run_id="new-run",
            ts_min=10,
            ts_max=20,
            count=1,
            status=EventStatus.SUCCESS,
            persistence=persistence,
        )

        assert len(manager._events) == 10000
        assert manager._events[0]["run_id"] == "old-1"
        assert manager._events[-1]["run_id"] == "new-run"

    def test_emit_event_postgres_uses_extended_function_when_available(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("ml.registry.event_manager.set_instrumentation_search_path", lambda _: None)
        session = MagicMock()
        persistence = SimpleNamespace(
            config=SimpleNamespace(backend=BackendType.POSTGRES),
            get_session=lambda: session,
        )

        EventManager().emit_event(
            dataset_id="dataset.features",
            instrument_id="EUR/USD",
            stage=Stage.CATALOG_WRITTEN,
            source=Source.HISTORICAL,
            run_id="run-1",
            ts_min=1,
            ts_max=2,
            count=3,
            status=EventStatus.SUCCESS,
            metadata={"k": "v"},
            persistence=persistence,
        )

        statement_text = str(session.execute.call_args_list[0].args[0])
        assert "emit_data_event_ext" in statement_text
        assert session.execute.call_args_list[0].args[1]["metadata"] == '{"k": "v"}'
        session.commit.assert_called_once()
        assert session.close.call_count == 0

    def test_emit_event_postgres_falls_back_to_legacy_function(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("ml.registry.event_manager.set_instrumentation_search_path", lambda _: None)
        session = MagicMock()
        session.execute.side_effect = [RuntimeError("ext failed"), None]
        persistence = SimpleNamespace(
            config=SimpleNamespace(backend=BackendType.POSTGRES),
            get_session=lambda: session,
        )

        EventManager().emit_event(
            dataset_id="dataset.features",
            instrument_id="EUR/USD",
            stage=Stage.CATALOG_WRITTEN,
            source=Source.HISTORICAL,
            run_id="run-1",
            ts_min=1,
            ts_max=2,
            count=3,
            status=EventStatus.SUCCESS,
            persistence=persistence,
        )

        assert session.execute.call_count == 2
        fallback_statement = str(session.execute.call_args_list[1].args[0])
        assert "emit_data_event(" in fallback_statement
        assert session.rollback.call_count == 1
        session.commit.assert_called_once()
        assert session.close.call_count == 0

    def test_emit_event_postgres_falls_back_to_insert_when_functions_fail(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("ml.registry.event_manager.set_instrumentation_search_path", lambda _: None)
        session = MagicMock()
        session.execute.side_effect = [
            RuntimeError("ext failed"),
            RuntimeError("legacy failed"),
            None,
        ]
        persistence = SimpleNamespace(
            config=SimpleNamespace(backend=BackendType.POSTGRES),
            get_session=lambda: session,
        )

        EventManager().emit_event(
            dataset_id="dataset.features",
            instrument_id="EUR/USD",
            stage=Stage.CATALOG_WRITTEN,
            source=Source.HISTORICAL,
            run_id="run-1",
            ts_min=1,
            ts_max=2,
            count=3,
            status=EventStatus.SUCCESS,
            persistence=persistence,
        )

        assert session.execute.call_count == 3
        insert_statement = str(session.execute.call_args_list[2].args[0])
        assert "INSERT INTO ml_data_events" in insert_statement
        assert session.rollback.call_count == 2
        session.commit.assert_called_once()
        session.close.assert_called_once()

    def test_emit_event_postgres_raises_when_insert_fallback_fails(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("ml.registry.event_manager.set_instrumentation_search_path", lambda _: None)
        session = MagicMock()
        session.execute.side_effect = [
            RuntimeError("ext failed"),
            RuntimeError("legacy failed"),
            RuntimeError("insert failed"),
        ]
        persistence = SimpleNamespace(
            config=SimpleNamespace(backend=BackendType.POSTGRES),
            get_session=lambda: session,
        )

        with pytest.raises(RuntimeError, match="insert failed"):
            EventManager().emit_event(
                dataset_id="dataset.features",
                instrument_id="EUR/USD",
                stage=Stage.CATALOG_WRITTEN,
                source=Source.HISTORICAL,
                run_id="run-1",
                ts_min=1,
                ts_max=2,
                count=3,
                status=EventStatus.SUCCESS,
                persistence=persistence,
            )

        assert session.rollback.call_count == 3
        session.close.assert_called_once()

    def test_emit_event_postgres_raises_when_session_missing(self) -> None:
        persistence = SimpleNamespace(
            config=SimpleNamespace(backend=BackendType.POSTGRES),
            get_session=lambda: None,
        )
        with pytest.raises(RuntimeError, match="Failed to get database session"):
            EventManager().emit_event(
                dataset_id="dataset.features",
                instrument_id="EUR/USD",
                stage=Stage.CATALOG_WRITTEN,
                source=Source.HISTORICAL,
                run_id="run-1",
                ts_min=1,
                ts_max=2,
                count=3,
                status=EventStatus.SUCCESS,
                persistence=persistence,
            )


class TestRegistryUtilsModule:
    def test_get_default_registry_path_honors_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(REGISTRY_PATH_ENV_VAR, "~/custom-registry")
        assert get_default_registry_path() == Path("~/custom-registry").expanduser()

    def test_get_default_registry_path_uses_home_when_env_absent(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv(REGISTRY_PATH_ENV_VAR, raising=False)
        assert get_default_registry_path() == Path.home() / ".nautilus" / "ml" / "registry"

    def test_build_feature_schema_and_hash_are_deterministic(self) -> None:
        schema = build_feature_schema(["f1", "f2"], ["float32", "int64"])
        assert schema == {"f1": "float32", "f2": "int64"}

        with pytest.raises(ValueError, match="must be same length"):
            build_feature_schema(["f1"], ["float32", "int64"])

        hash_a = compute_dataset_schema_hash(
            schema={"b": "int64", "a": "float64"},
            primary_keys=["ts_event", "instrument_id"],
            ts_field="ts_event",
            seq_field="sequence",
            pipeline_signature="pipeline-v1",
        )
        hash_b = compute_dataset_schema_hash(
            schema={"a": "float64", "b": "int64"},
            primary_keys=["instrument_id", "ts_event"],
            ts_field="ts_event",
            seq_field="sequence",
            pipeline_signature="pipeline-v1",
        )
        hash_without_seq = compute_dataset_schema_hash(
            schema={"a": "float64", "b": "int64"},
            primary_keys=["instrument_id", "ts_event"],
            ts_field="ts_event",
            pipeline_signature="pipeline-v1",
        )
        assert hash_a == hash_b
        assert hash_a != hash_without_seq

    def test_build_student_manifest_and_feature_compatibility_guards(self) -> None:
        decision_config = {"positive_class_index": 1}
        output_schema = {"prediction": "float32"}
        calibration = {"method": "platt"}

        manifest = build_student_manifest(
            model_id="student-1",
            architecture="xgboost",
            feature_schema={"f1": "float32", "f2": "int64"},
            feature_schema_hash="hash-1",
            parent_id="teacher-1",
            decision_policy="ml.policy.binary",
            decision_config=decision_config,
            output_schema=output_schema,
            calibration=calibration,
        )
        decision_config["positive_class_index"] = 2

        assert manifest.feature_schema["f1"] == "float32"
        assert manifest.decision_config["positive_class_index"] == 1
        assert manifest.output_schema == output_schema
        assert manifest.calibration == calibration

        assert_features_compatible(manifest, ["f1", "f2"], ["float32", "int64"])

        with pytest.raises(ValueError, match="Feature names/order mismatch"):
            assert_features_compatible(manifest, ["f2", "f1"])

        with pytest.raises(ValueError, match="Feature dtypes mismatch"):
            assert_features_compatible(manifest, ["f1", "f2"], ["float32", "float32"])


class TestRegistryStatisticsModule:
    def test_welch_t_test_handles_insufficient_and_zero_variance_samples(self) -> None:
        insufficient = welch_t_test(
            np.array([1.0], dtype=np.float64),
            np.array([2.0], dtype=np.float64),
        )
        assert insufficient["error"] == "Insufficient samples for test"
        assert insufficient["p_value_approx"] == 1.0

        zero_variance = welch_t_test(
            np.array([1.0, 1.0], dtype=np.float64),
            np.array([2.0, 2.0], dtype=np.float64),
        )
        assert zero_variance["error"] == "Zero variance in samples"
        assert zero_variance["statistically_significant"] is False

    def test_welch_t_test_uses_small_sample_threshold_and_zero_mean_guard(self) -> None:
        result = welch_t_test(
            np.array([-1.0, 1.0], dtype=np.float64),
            np.array([0.5, 1.5], dtype=np.float64),
        )
        stats = StatsConfig()
        assert result["degrees_of_freedom"] < float(stats.small_sample_df_threshold)
        assert result["critical_value"] == float(stats.conservative_critical_value)
        assert result["relative_improvement"] == 0.0

    def test_welch_t_test_uses_default_critical_value_for_larger_samples(self) -> None:
        sample_a = np.linspace(0.0, 99.0, num=100, dtype=np.float64)
        sample_b = sample_a + 0.5
        result = welch_t_test(sample_a, sample_b, significance_level=0.10)
        stats = StatsConfig()
        assert result["degrees_of_freedom"] >= float(stats.small_sample_df_threshold)
        assert result["critical_value"] == float(stats.z_alpha_default)

    def test_compare_models_covers_error_sorting_and_winner_paths(self) -> None:
        assert compare_models([], "accuracy") == {"error": "No models provided"}
        assert compare_models([{"model_id": "m0", "metrics": {}}], "accuracy", baseline_index=1) == {
            "error": "Invalid baseline index 1",
        }

        ranked = compare_models(
            [
                {"model_id": "baseline", "metrics": {"accuracy": 0.5}},
                {"model_id": "winner", "metrics": {"accuracy": 0.8}},
                {"model_id": "missing", "metrics": {}},
            ],
            "accuracy",
        )
        assert ranked["winner"] == "winner"
        assert ranked["models"][0]["model_id"] == "winner"
        assert "relative_improvement" in ranked["models"][0]

        all_missing = compare_models(
            [
                {"model_id": "a", "metrics": {}},
                {"model_id": "b", "metrics": {}},
            ],
            "accuracy",
        )
        assert "winner" not in all_missing
        assert all("relative_improvement" not in model for model in all_missing["models"])

    def test_calculate_sample_size_handles_zero_mapped_and_interpolated_paths(self) -> None:
        assert calculate_sample_size(0.0) == 100000
        assert calculate_sample_size(0.5, power=0.90, significance_level=0.01) >= 30
        assert calculate_sample_size(10.0, power=0.83, significance_level=0.03) == 30


class TestManifestDefaultsModule:
    def test_resolve_primary_keys_dataset_specific_defaults(self) -> None:
        schema = {"ticker": "str", "period_end": "str"}
        assert resolve_primary_keys(DatasetType.EARNINGS_ACTUALS, schema) == [
            "ticker",
            "period_end",
        ]

    def test_resolve_primary_keys_schema_fallbacks(self) -> None:
        assert resolve_primary_keys(
            DatasetType.FEATURES,
            {"instrument_id": "str", "ts_event": "int64", "value": "float64"},
        ) == ["instrument_id", "ts_event"]
        assert resolve_primary_keys(
            DatasetType.FEATURES,
            {"ts_event": "int64", "value": "float64"},
        ) == ["ts_event"]
        assert resolve_primary_keys(
            DatasetType.FEATURES,
            {"series_id": "str", "value": "float64"},
        ) == ["series_id"]


class TestWatermarkManagerComponentModule:
    def test_update_watermark_json_persists_to_cache_and_save_hook(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("ml.registry.common.watermark_manager.time.time", lambda: 42.0)
        persistence = _make_watermark_persistence_stub(backend=BackendType.JSON)
        manager = WatermarkManagerComponent(cast(Any, persistence))

        manager.update_watermark(
            dataset_id="dataset.features",
            instrument_id="EUR/USD",
            source=Source.LIVE,
            last_success_ns=10,
            count=2,
            completeness_pct=95.0,
        )

        key = "dataset.features:EUR/USD:live"
        assert persistence._watermarks[key].updated_at == 42.0
        persistence._save_registry.assert_called_once_with(immediate=True)

    def test_update_watermark_postgres_commits_and_caches(self) -> None:
        session = MagicMock()
        persistence = _make_watermark_persistence_stub(
            backend=BackendType.POSTGRES,
            session=session,
        )
        manager = WatermarkManagerComponent(cast(Any, persistence))

        manager.update_watermark(
            dataset_id="dataset.features",
            instrument_id="EUR/USD",
            source=Source.HISTORICAL,
            last_success_ns=20,
            count=3,
            completeness_pct=99.0,
        )

        call_args = session.execute.call_args
        assert "update_watermark" in str(call_args.args[0])
        assert call_args.args[1]["source"] == "historical"
        session.commit.assert_called_once()
        session.close.assert_called_once()
        assert "dataset.features:EUR/USD:historical" in persistence._watermarks

    def test_update_watermark_postgres_rolls_back_on_failure(self) -> None:
        session = MagicMock()
        session.execute.side_effect = RuntimeError("update failed")
        persistence = _make_watermark_persistence_stub(
            backend=BackendType.POSTGRES,
            session=session,
        )
        manager = WatermarkManagerComponent(cast(Any, persistence))

        with pytest.raises(RuntimeError, match="update failed"):
            manager.update_watermark(
                dataset_id="dataset.features",
                instrument_id="EUR/USD",
                source=Source.HISTORICAL,
                last_success_ns=20,
                count=3,
                completeness_pct=99.0,
            )

        session.rollback.assert_called_once()
        session.close.assert_called_once()

    def test_get_watermark_postgres_paths(self) -> None:
        row = {
            "dataset_id": "dataset.features",
            "instrument_id": "EUR/USD",
            "source": "live",
            "last_success_ns": 100,
            "last_attempt_ns": 100,
            "last_count": 4,
            "completeness_pct": 98.0,
            "updated_at": 1.5,
        }

        session = MagicMock()
        session.execute.return_value.fetchone.return_value = row
        persistence = _make_watermark_persistence_stub(
            backend=BackendType.POSTGRES,
            session=session,
        )
        manager = WatermarkManagerComponent(cast(Any, persistence))

        fetched = manager.get_watermark("dataset.features", "EUR/USD", Source.LIVE)
        assert fetched is not None
        assert fetched.last_success_ns == 100
        assert "dataset.features:EUR/USD:live" in persistence._watermarks
        session.close.assert_called_once()

        session_cached = MagicMock()
        persistence_cached = _make_watermark_persistence_stub(
            backend=BackendType.POSTGRES,
            session=session_cached,
        )
        persistence_cached._watermarks["dataset.features:EUR/USD:live"] = fetched
        cached_manager = WatermarkManagerComponent(cast(Any, persistence_cached))
        cached = cached_manager.get_watermark("dataset.features", "EUR/USD", Source.LIVE)
        assert cached is fetched
        session_cached.execute.assert_not_called()

        session_none = MagicMock()
        session_none.execute.return_value.fetchone.return_value = None
        persistence_none = _make_watermark_persistence_stub(
            backend=BackendType.POSTGRES,
            session=session_none,
        )
        none_manager = WatermarkManagerComponent(cast(Any, persistence_none))
        assert none_manager.get_watermark("dataset.features", "EUR/USD", Source.LIVE) is None

    def test_get_watermark_postgres_raises_when_session_missing(self) -> None:
        persistence = _make_watermark_persistence_stub(
            backend=BackendType.POSTGRES,
            session=None,
        )
        manager = WatermarkManagerComponent(cast(Any, persistence))
        with pytest.raises(RuntimeError, match="Failed to get database session"):
            manager.get_watermark("dataset.features", "EUR/USD", Source.LIVE)

    def test_iter_watermarks_postgres_applies_filters_and_caches_rows(self) -> None:
        session = MagicMock()
        session.execute.return_value.fetchall.return_value = [
            {
                "dataset_id": "dataset.features",
                "instrument_id": "EUR/USD",
                "source": "live",
                "last_success_ns": 101,
                "last_attempt_ns": 102,
                "last_count": 2,
                "completeness_pct": 97.0,
                "updated_at": 11.0,
            },
        ]
        persistence = _make_watermark_persistence_stub(
            backend=BackendType.POSTGRES,
            session=session,
        )
        manager = WatermarkManagerComponent(cast(Any, persistence))

        results = list(
            manager.iter_watermarks(
                dataset_id="dataset.features",
                instrument_id="EUR/USD",
                source=Source.LIVE,
                limit=1,
            ),
        )

        assert len(results) == 1
        assert results[0].source == "live"
        execute_args = session.execute.call_args.args
        assert "WHERE dataset_id = :dataset_id" in str(execute_args[0])
        assert "LIMIT :limit" in str(execute_args[0])
        assert execute_args[1] == {
            "dataset_id": "dataset.features",
            "instrument_id": "EUR/USD",
            "source": "live",
            "limit": 1,
        }
        assert "dataset.features:EUR/USD:live" in persistence._watermarks
        session.close.assert_called_once()

    def test_iter_watermarks_json_clamps_negative_limit_to_zero(self) -> None:
        persistence = _make_watermark_persistence_stub(backend=BackendType.JSON)
        persistence._watermarks["dataset.features:EUR/USD:live"] = Watermark(
            dataset_id="dataset.features",
            instrument_id="EUR/USD",
            source="live",
            last_success_ns=1,
            last_attempt_ns=1,
            last_count=1,
            completeness_pct=100.0,
            updated_at=1.0,
        )
        manager = WatermarkManagerComponent(cast(Any, persistence))
        assert list(manager.iter_watermarks(limit=-3)) == []

    def test_iter_watermarks_postgres_raises_when_session_missing(self) -> None:
        persistence = _make_watermark_persistence_stub(
            backend=BackendType.POSTGRES,
            session=None,
        )
        manager = WatermarkManagerComponent(cast(Any, persistence))
        with pytest.raises(RuntimeError, match="Failed to get database session"):
            list(manager.iter_watermarks(dataset_id="dataset.features"))


class TestDataRegistrySupportPaths:
    def test_is_legacy_dataset_id_predicate(self) -> None:
        assert _is_legacy_dataset_id("ohlcv_spy_xnas") is True
        assert _is_legacy_dataset_id(" mbp_foo ") is True
        assert _is_legacy_dataset_id("ml.features.spy") is False
        assert _is_legacy_dataset_id(None) is False

    def test_manifest_from_row_resolves_default_primary_keys_for_macro_dataset(
        self,
        tmp_path: Path,
    ) -> None:
        registry = _make_data_registry_json_backend(tmp_path)

        row = {
            "dataset_id": "ml.macro.observations",
            "dataset_type": "macro_observations",
            "storage_kind": "postgres",
            "location": "ml_macro_table",
            "partitioning": json.dumps({"by": "observation_ts"}),
            "retention_days": 30,
            "schema": json.dumps(
                {
                    "series_id": "str",
                    "observation_ts": "int64",
                    "ts_event": "int64",
                },
            ),
            "ts_field": "ts_event",
            "seq_field": None,
            "schema_hash": "",
            "constraints": json.dumps({}),
            "lineage": json.dumps([]),
            "pipeline_signature": "macro_ingestion",
            "version": "1.0.0",
            "created_at": 1_000,
            "last_modified": 2_000,
            "metadata": json.dumps({}),
        }

        manifest = registry._manifest_from_row(row)
        assert manifest.dataset_type == DatasetType.MACRO_OBSERVATIONS
        assert manifest.primary_keys == ["series_id", "observation_ts", "ts_event"]
        assert manifest.partitioning == {"by": "observation_ts"}
        assert manifest.constraints == {}

    def test_list_legacy_dataset_ids_postgres_paths(self, tmp_path: Path) -> None:
        registry = _make_data_registry_json_backend(tmp_path)
        registry_runtime = cast(Any, registry)

        registry_runtime.backend = BackendType.POSTGRES
        registry_runtime.persistence = SimpleNamespace(
            get_session=lambda: None,
            close=lambda: None,
        )
        empty = registry.list_legacy_dataset_ids()
        assert empty == {
            "registry": (),
            "events": (),
            "watermarks": (),
            "lineage_children": (),
            "lineage_parents": (),
        }

        session = MagicMock()

        def _result_rows(values: tuple[str, ...]) -> MagicMock:
            result = MagicMock()
            result.fetchall.return_value = [(value,) for value in values]
            return result

        session.execute.side_effect = [
            _result_rows(("ohlcv_spy_xnas",)),
            _result_rows(("mbp_spy_xnas",)),
            _result_rows(("ohlcv_spy_xnas",)),
            _result_rows(("ohlcv_spy_xnas",)),
            _result_rows(("mbp_spy_xnas",)),
        ]
        registry_runtime.persistence = SimpleNamespace(
            get_session=lambda: session,
            close=lambda: None,
        )

        report = registry.list_legacy_dataset_ids()
        assert report == {
            "registry": ("ohlcv_spy_xnas",),
            "events": ("mbp_spy_xnas",),
            "watermarks": ("ohlcv_spy_xnas",),
            "lineage_children": ("ohlcv_spy_xnas",),
            "lineage_parents": ("mbp_spy_xnas",),
        }
        assert session.execute.call_count == 5
        session.close.assert_called_once()
