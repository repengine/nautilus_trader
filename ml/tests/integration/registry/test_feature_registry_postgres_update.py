from __future__ import annotations

import os
from pathlib import Path

import pytest

from ml.registry.base import DataRequirements
from ml.registry.feature_registry import FeatureManifest
from ml.registry.feature_registry import FeatureRegistry
from ml.registry.feature_registry import FeatureRole
from ml.registry.persistence import BackendType
from ml.registry.persistence import PersistenceConfig

pytestmark = pytest.mark.usefixtures(
    "isolated_prometheus_registry",
    "mock_tracing_backend",
    "isolated_orchestrator_env",
)


@pytest.mark.integration
def test_update_manifest_persists_postgres(tmp_path: Path) -> None:
    # Use default or env-provided connection string
    conn = os.getenv(
        "NAUTILUS_REGISTRY_DB_URL",
        "postgresql://postgres:postgres@localhost:5432/nautilus",
    )
    pc = PersistenceConfig(backend=BackendType.POSTGRES, connection_string=conn)
    # Skip if DB is not reachable
    try:
        from ml.core.db_engine import EngineManager

        eng = EngineManager.get_engine(conn)
        with eng.connect() as _:
            pass
    except Exception:
        pytest.skip("PostgreSQL is not available for integration test")

    reg = FeatureRegistry(tmp_path, persistence_config=pc)

    m = FeatureManifest(
        feature_set_id="",
        name="fs_pg",
        version="1.0.0",
        role=FeatureRole.TEACHER,
        data_requirements=DataRequirements.L1_ONLY,
        feature_names=["f1", "f2"],
        feature_dtypes=["float32", "float32"],
        schema_hash="abc123",
        pipeline_signature="sig",
        pipeline_version="1",
    )
    fid = reg.register_feature_set(m)
    reg.update_manifest(fid, perf_digest={"pr_auc": 0.66})

    # Re-instantiate and assert persistence
    reg2 = FeatureRegistry(tmp_path, persistence_config=pc)
    info = reg2.get_feature_set(fid)
    assert info is not None
    assert abs(info.manifest.perf_digest.get("pr_auc", 0.0) - 0.66) < 1e-12
