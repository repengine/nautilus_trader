#!/usr/bin/env python3

"""
Feature registry with self-describing manifests.

Provides a local, file-based registry for feature sets with lifecycle management, schema
hashing, and simple lineage queries.

"""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from pathlib import Path
from typing import Any, cast

from ml.registry.base import DataRequirements
from ml.registry.dataclasses import QualityGate
from ml.registry.persistence import BackendType
from ml.registry.persistence import FeatureTable
from ml.registry.persistence import PersistenceConfig
from ml.registry.persistence import PersistenceManager


class FeatureRole(Enum):
    """
    Role of a feature set in the system.
    """

    TEACHER = "teacher"
    STUDENT = "student"
    INFERENCE_SUPPORT = "inference_support"


class FeatureStage(Enum):
    """
    Lifecycle stage for a feature set.
    """

    CANDIDATE = "candidate"
    STAGING = "staging"
    PROD = "prod"
    DEPRECATED = "deprecated"
    SCRAPPED = "scrapped"


@dataclass(slots=True)
class FeatureManifest:
    """
    Self-describing manifest for a feature set.

    Attributes
    ----------
    feature_set_id : str
        Unique identifier for the feature set.
    name : str
        Human-readable name.
    version : str
        Semantic version string.
    role : FeatureRole
        Intended role (teacher/student/etc.).
    data_requirements : DataRequirements
        Data requirements of the pipeline (e.g. L1_ONLY).
    feature_names : list[str]
        Ordered feature names.
    feature_dtypes : list[str]
        Dtypes for each feature (typically "float32").
    schema_hash : str
        Hash of schema used for validation.
    pipeline_signature : str
        Hash/signature of the transform graph and params.
    pipeline_version : str
        Version of the pipeline engine.
    capability_flags : dict[str, bool]
        Optional capability flags (e.g., microstructure, trade_flow).
    constraints : dict[str, Any]
        Runtime constraints (latency, memory, warmup bars).
    parity_tolerance : float
        Parity tolerance used during validation.
    parity_digest : dict[str, Any]
        Summary of parity validation results.
    perf_digest : dict[str, Any]
        Summary of latency/performance measurements.
    parent_feature_set_id : Optional[str]
        Parent feature set (e.g. teacher for a student set).
    metadata : dict[str, Any]
        Extra metadata for auditability.
    created_at : float
        Creation timestamp.
    last_modified : float
        Last modified timestamp.
    stage : FeatureStage
        Lifecycle stage.

    """

    feature_set_id: str
    name: str
    version: str
    role: FeatureRole
    data_requirements: DataRequirements
    feature_names: list[str]
    feature_dtypes: list[str]
    schema_hash: str
    pipeline_signature: str
    pipeline_version: str
    capability_flags: dict[str, bool] = field(default_factory=dict)
    constraints: dict[str, Any] = field(default_factory=dict)
    parity_tolerance: float = 0.0
    parity_digest: dict[str, Any] = field(default_factory=dict)
    perf_digest: dict[str, Any] = field(default_factory=dict)
    parent_feature_set_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0
    last_modified: float = 0.0
    stage: FeatureStage = FeatureStage.CANDIDATE


def compute_schema_hash(
    feature_names: list[str],
    feature_dtypes: list[str],
    pipeline_signature: str,
) -> str:
    """
    Compute a stable schema hash from names, dtypes, and pipeline signature.
    """
    h = hashlib.sha256()
    for n, t in zip(feature_names, feature_dtypes):
        h.update(n.encode("utf-8"))
        h.update(b"::")
        h.update(t.encode("utf-8"))
        h.update(b"\n")
    h.update(b"|sig|")
    h.update(pipeline_signature.encode("utf-8"))
    return h.hexdigest()


@dataclass(slots=True)
class FeatureInfo:
    """
    Container for manifest and storage information.
    """

    manifest: FeatureManifest
    artifacts: dict[str, str] = field(default_factory=dict)


class FeatureRegistry:
    """
    Feature registry with configurable persistence backend.

    Supports both JSON files and PostgreSQL for persistence, making it suitable for both
    development and production environments.

    Thread-safe for concurrent operations.

    """

    def __init__(
        self,
        registry_path: Path,
        persistence_config: PersistenceConfig | None = None,
    ) -> None:
        """
        Initialize feature registry with configurable persistence backend.

        Parameters
        ----------
        registry_path : Path
            Directory path for registry storage (used for artifact files)
        persistence_config : PersistenceConfig | None
            Persistence configuration. If None, defaults to JSON backend.

        """
        self._root = registry_path
        self._root.mkdir(parents=True, exist_ok=True)
        self._file = self._root / "feature_registry.json"
        self._lock = threading.RLock()
        self._features: dict[str, FeatureInfo] = {}

        # Setup persistence
        if persistence_config is None:
            persistence_config = PersistenceConfig(
                backend=BackendType.JSON,
                json_path=registry_path,
            )
        self.persistence = PersistenceManager(persistence_config)
        self.backend = persistence_config.backend

        self._load()

    # Persistence
    def _load(self) -> None:
        """
        Load registry from persistence backend.
        """
        if self.backend == BackendType.JSON:
            if not self._file.exists():
                self._save()
                return
            data = self.persistence.load_json("feature_registry.json")
            if data:
                self._features = {}
                for fid, feature_data in data.get("features", {}).items():
                    m = feature_data["manifest"]
                    manifest = FeatureManifest(
                        feature_set_id=fid,
                        name=m["name"],
                        version=m["version"],
                        role=FeatureRole(m["role"]),
                        data_requirements=DataRequirements(m["data_requirements"]),
                        feature_names=m["feature_names"],
                        feature_dtypes=m["feature_dtypes"],
                        schema_hash=m["schema_hash"],
                        pipeline_signature=m["pipeline_signature"],
                        pipeline_version=m["pipeline_version"],
                        capability_flags=m.get("capability_flags", {}),
                        constraints=m.get("constraints", {}),
                        parity_tolerance=m.get("parity_tolerance", 0.0),
                        parity_digest=m.get("parity_digest", {}),
                        perf_digest=m.get("perf_digest", {}),
                        parent_feature_set_id=m.get("parent_feature_set_id"),
                        metadata=m.get("metadata", {}),
                        created_at=m.get("created_at", 0.0),
                        last_modified=m.get("last_modified", 0.0),
                        stage=FeatureStage(m.get("stage", FeatureStage.CANDIDATE.value)),
                    )
                    self._features[fid] = FeatureInfo(
                        manifest=manifest,
                        artifacts=feature_data.get("artifacts", {}),
                    )
        elif self.backend == BackendType.POSTGRES:
            # Load all features from PostgreSQL
            session = self.persistence.get_session()
            if session is None:
                return
            try:
                self._features = {}
                features = session.query(FeatureTable).all()
                for feature in features:
                    feature_info = self._db_to_feature_info(feature)
                    self._features[feature_info.manifest.feature_set_id] = feature_info
            except Exception as e:
                print(f"Error loading from database: {e}")
                self._features = {}
            finally:
                session.close()

    def _save(self) -> None:
        """
        Save registry to persistence backend.
        """
        with self._lock:
            if self.backend == BackendType.JSON:
                serial: dict[str, Any] = {
                    "features": {
                        fid: {
                            "manifest": {
                                "name": info.manifest.name,
                                "version": info.manifest.version,
                                "role": info.manifest.role.value,
                                "data_requirements": info.manifest.data_requirements.value,
                                "feature_names": info.manifest.feature_names,
                                "feature_dtypes": info.manifest.feature_dtypes,
                                "schema_hash": info.manifest.schema_hash,
                                "pipeline_signature": info.manifest.pipeline_signature,
                                "pipeline_version": info.manifest.pipeline_version,
                                "capability_flags": info.manifest.capability_flags,
                                "constraints": info.manifest.constraints,
                                "parity_tolerance": info.manifest.parity_tolerance,
                                "parity_digest": info.manifest.parity_digest,
                                "perf_digest": info.manifest.perf_digest,
                                "parent_feature_set_id": info.manifest.parent_feature_set_id,
                                "metadata": info.manifest.metadata,
                                "created_at": info.manifest.created_at,
                                "last_modified": info.manifest.last_modified,
                                "stage": info.manifest.stage.value,
                            },
                            "artifacts": info.artifacts,
                        }
                        for fid, info in self._features.items()
                    },
                    "last_updated": time.time(),
                }
                self.persistence.save_json(serial, "feature_registry.json")
            elif self.backend == BackendType.POSTGRES:
                # PostgreSQL is updated on each operation
                pass

    def _db_to_feature_info(self, db_feature: FeatureTable) -> FeatureInfo:
        """
        Convert database feature to FeatureInfo.

        Parameters
        ----------
        db_feature : FeatureTable
            Database feature record

        Returns
        -------
        FeatureInfo
            Feature information object

        """
        manifest = FeatureManifest(
            feature_set_id=cast(str, db_feature.feature_set_id),
            name=cast(str, db_feature.name),
            version=cast(str, db_feature.version),
            role=FeatureRole(cast(str, db_feature.role)),
            data_requirements=DataRequirements(cast(str, db_feature.data_requirements)),
            feature_names=cast(list[str], db_feature.feature_names) or [],
            feature_dtypes=cast(list[str], db_feature.feature_dtypes) or [],
            schema_hash=cast(str, db_feature.schema_hash),
            pipeline_signature=cast(str, db_feature.pipeline_signature) or "",
            pipeline_version=cast(str, db_feature.pipeline_version) or "",
            capability_flags=cast(dict[str, Any], db_feature.capability_flags) or {},
            constraints=cast(dict[str, Any], db_feature.constraints) or {},
            parity_tolerance=cast(float, db_feature.parity_tolerance) or 0.0,
            parity_digest=cast(dict[str, Any], db_feature.parity_digest) or {},
            perf_digest=cast(dict[str, Any], db_feature.perf_digest) or {},
            parent_feature_set_id=db_feature.parent_feature_set_id,
            metadata=cast(dict[str, Any], db_feature.extra_metadata) or {},
            created_at=db_feature.created_at.timestamp() if db_feature.created_at else time.time(),
            last_modified=(
                db_feature.last_modified.timestamp() if db_feature.last_modified else time.time()
            ),
            stage=FeatureStage(cast(str, db_feature.stage)),
        )
        # Note: artifacts are stored as JSON in metadata for PostgreSQL
        artifacts = (
            db_feature.extra_metadata.get("artifacts", {}) if db_feature.extra_metadata else {}
        )
        return FeatureInfo(
            manifest=manifest,
            artifacts=artifacts,
        )

    def _save_feature_to_db(self, feature_info: FeatureInfo) -> None:
        """
        Save feature to PostgreSQL database.

        Parameters
        ----------
        feature_info : FeatureInfo
            Feature information to save

        """
        session = self.persistence.get_session()
        if session is None:
            return

        try:
            # Check if feature exists
            existing = (
                session.query(FeatureTable)
                .filter_by(
                    feature_set_id=feature_info.manifest.feature_set_id,
                )
                .first()
            )

            # Store artifacts in metadata
            metadata = feature_info.manifest.metadata.copy()
            metadata["artifacts"] = feature_info.artifacts

            if existing:
                # Update existing feature
                existing.name = feature_info.manifest.name
                existing.version = feature_info.manifest.version
                existing.role = feature_info.manifest.role.value
                existing.data_requirements = feature_info.manifest.data_requirements.value
                existing.feature_names = feature_info.manifest.feature_names
                existing.feature_dtypes = feature_info.manifest.feature_dtypes
                existing.schema_hash = feature_info.manifest.schema_hash
                existing.pipeline_signature = feature_info.manifest.pipeline_signature
                existing.pipeline_version = feature_info.manifest.pipeline_version
                existing.capability_flags = feature_info.manifest.capability_flags
                existing.constraints = feature_info.manifest.constraints
                existing.parity_tolerance = feature_info.manifest.parity_tolerance
                existing.parity_digest = feature_info.manifest.parity_digest
                existing.perf_digest = feature_info.manifest.perf_digest
                existing.parent_feature_set_id = feature_info.manifest.parent_feature_set_id
                existing.stage = feature_info.manifest.stage.value
                existing.extra_metadata = metadata
            else:
                # Create new feature
                new_feature = FeatureTable(
                    feature_set_id=feature_info.manifest.feature_set_id,
                    name=feature_info.manifest.name,
                    version=feature_info.manifest.version,
                    role=feature_info.manifest.role.value,
                    data_requirements=feature_info.manifest.data_requirements.value,
                    feature_names=feature_info.manifest.feature_names,
                    feature_dtypes=feature_info.manifest.feature_dtypes,
                    schema_hash=feature_info.manifest.schema_hash,
                    pipeline_signature=feature_info.manifest.pipeline_signature,
                    pipeline_version=feature_info.manifest.pipeline_version,
                    capability_flags=feature_info.manifest.capability_flags,
                    constraints=feature_info.manifest.constraints,
                    parity_tolerance=feature_info.manifest.parity_tolerance,
                    parity_digest=feature_info.manifest.parity_digest,
                    perf_digest=feature_info.manifest.perf_digest,
                    parent_feature_set_id=feature_info.manifest.parent_feature_set_id,
                    stage=feature_info.manifest.stage.value,
                    extra_metadata=metadata,
                )
                session.add(new_feature)

            session.commit()
        except Exception as e:
            session.rollback()
            raise RuntimeError(f"Failed to save feature to database: {e}") from e
        finally:
            session.close()

    # ID generation
    def _gen_id(self) -> str:
        return f"feature_set_{int(time.time() * 1_000_000)}"

    # Public API
    def register_feature_set(
        self,
        manifest: FeatureManifest,
        artifacts: dict[str, str] | None = None,
    ) -> str:
        with self._lock:
            fid = manifest.feature_set_id or self._gen_id()
            now = time.time()
            manifest.feature_set_id = fid
            if manifest.created_at == 0.0:
                manifest.created_at = now
            manifest.last_modified = now
            feature_info = FeatureInfo(manifest=manifest, artifacts=artifacts or {})
            self._features[fid] = feature_info

            # Persist to backend
            if self.backend == BackendType.POSTGRES:
                self._save_feature_to_db(feature_info)
            else:
                self._save()

            # Log audit
            self.persistence.log_audit(
                entity_type="feature",
                entity_id=fid,
                action="registered",
                changes={"role": manifest.role.value, "stage": manifest.stage.value},
            )

            return fid

    def promote(self, feature_set_id: str, stage: FeatureStage) -> None:
        with self._lock:
            info = self._features[feature_set_id]
            info.manifest.stage = stage
            info.manifest.last_modified = time.time()
            self._save()

    def deprecate(self, feature_set_id: str, reason: str | None = None) -> None:
        with self._lock:
            info = self._features[feature_set_id]
            info.manifest.stage = FeatureStage.DEPRECATED
            if reason:
                info.manifest.metadata["deprecation_reason"] = reason
            info.manifest.last_modified = time.time()
            self._save()

    def scrap(self, feature_set_id: str) -> None:
        with self._lock:
            if feature_set_id in self._features:
                self._features[feature_set_id].manifest.stage = FeatureStage.SCRAPPED
                self._save()

    def get_feature_set(self, feature_set_id: str) -> FeatureInfo | None:
        """
        Get complete feature information including manifest and artifacts.
        """
        return self._features.get(feature_set_id)

    def get_feature_manifest(self, feature_set_id: str) -> FeatureManifest | None:
        """
        Get only the feature manifest (backward compatibility).
        """
        info = self._features.get(feature_set_id)
        return None if info is None else info.manifest

    def resolve_by_schema_hash(self, schema_hash: str) -> list[FeatureInfo]:
        """
        Get all feature sets matching a schema hash.
        """
        return [fi for fi in self._features.values() if fi.manifest.schema_hash == schema_hash]

    def list_by_role(self, role: FeatureRole) -> list[FeatureInfo]:
        """
        List all feature sets with a specific role.
        """
        return [fi for fi in self._features.values() if fi.manifest.role == role]

    def list_all(self) -> list[FeatureInfo]:
        """
        List all feature sets in the registry.
        """
        return list(self._features.values())

    def get_lineage(self, feature_set_id: str) -> list[FeatureManifest]:
        """
        Return the lineage chain from parent to child including the requested ID if
        linked.
        """
        result: list[FeatureManifest] = []
        # Simple one-hop: include parent then the requested manifest and any children pointing to it
        target_info = self.get_feature_set(feature_set_id)
        if target_info is None:
            return result
        target = target_info.manifest
        if target.parent_feature_set_id:
            parent_info = self.get_feature_set(target.parent_feature_set_id)
            if parent_info is not None:
                result.append(parent_info.manifest)
        result.append(target)
        result.extend(
            [
                fi.manifest
                for fi in self._features.values()
                if fi.manifest.parent_feature_set_id == feature_set_id
            ],
        )
        return result

    # Quality gating
    def validate_and_promote(self, feature_set_id: str, gates: list[QualityGate]) -> bool:
        """
        Validate a feature set against quality gates and promote to PROD if passed.

        Gates can reference keys present in manifest.perf_digest,
        manifest.parity_digest, or manifest.constraints. The first matching key is used.

        """
        info = self._features.get(feature_set_id)
        if info is None:
            raise KeyError(f"Unknown feature_set_id: {feature_set_id}")

        def _get_metric(name: str) -> float | None:
            # Look into perf, then parity, then constraints
            for src in (
                info.manifest.perf_digest,
                info.manifest.parity_digest,
                info.manifest.constraints,
            ):
                if name in src:
                    val = src[name]
                    if isinstance(val, int | float):
                        return float(val)
            return None

        passed_all = True
        for gate in gates:
            val = _get_metric(gate.metric_name)
            ok = False
            if val is None:
                ok = False
            elif gate.comparison == "gte":
                ok = val >= gate.threshold
            elif gate.comparison == "lte":
                ok = val <= gate.threshold
            elif gate.comparison == "gt":
                ok = val > gate.threshold
            elif gate.comparison == "lt":
                ok = val < gate.threshold
            elif gate.comparison == "eq":
                ok = abs(val - gate.threshold) < 1e-12
            else:
                ok = False

            if gate.required and not ok:
                passed_all = False

        if passed_all:
            self.promote(feature_set_id, FeatureStage.PROD)
        return passed_all
