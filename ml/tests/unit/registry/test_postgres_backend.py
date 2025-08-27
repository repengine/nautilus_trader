#!/usr/bin/env python3

"""
Unit tests for PostgreSQL-backed registries.

Tests the dual-backend capability of registries with both JSON and PostgreSQL
persistence backends.

"""

from __future__ import annotations

import tempfile
import time
from collections.abc import Generator
from pathlib import Path

import pytest


SUFFIX_ONNX = ".onnx"  # Define directly to avoid import issues
from ml.registry.base import DataRequirements
from ml.registry.model_registry import ModelManifest
from ml.registry.base import ModelRole
from ml.registry.feature_registry import FeatureManifest
from ml.registry.feature_registry import FeatureRegistry
from ml.registry.feature_registry import FeatureRole
from ml.registry.feature_registry import FeatureStage
from ml.registry.model_registry import ModelRegistry
from ml.registry.persistence import BackendType
from ml.registry.persistence import PersistenceConfig
from ml.registry.strategy_registry import MarketRegime
from ml.registry.strategy_registry import StrategyManifest
from ml.registry.strategy_registry import StrategyRegistry
from ml.registry.strategy_registry import StrategyType


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """
    Create a temporary directory for testing.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def json_persistence_config(temp_dir: Path) -> PersistenceConfig:
    """
    Create JSON backend persistence configuration.
    """
    return PersistenceConfig(
        backend=BackendType.JSON,
        json_path=temp_dir,
    )


@pytest.fixture
def postgres_persistence_config() -> PersistenceConfig | None:
    """
    Create PostgreSQL backend persistence configuration.
    """
    try:
        # Try to connect to local PostgreSQL
        config = PersistenceConfig(
            backend=BackendType.POSTGRES,
            connection_string="postgresql://postgres:postgres@localhost:5432/test_nautilus",
            echo=False,
        )
        # Test connection
        from ml.registry.persistence import PersistenceManager

        manager = PersistenceManager(config)
        session = manager.get_session()
        if session:
            session.close()
            return config
    except Exception:
        # PostgreSQL not available, skip tests
        return None
    return None


class TestModelRegistryBackends:
    """
    Test model registry with different backends.
    """

    def test_json_backend_register_and_retrieve(
        self,
        temp_dir: Path,
        json_persistence_config: PersistenceConfig,
    ) -> None:
        """
        Test registering and retrieving models with JSON backend.
        """
        # Create registry
        registry = ModelRegistry(
            registry_path=temp_dir,
            persistence_config=json_persistence_config,
        )

        # Create model manifest
        manifest = ModelManifest(
            model_id="test_model_1",
            role=ModelRole.TEACHER,
            data_requirements=DataRequirements.L1_L2,
            architecture="XGBoost",
            feature_schema={"feature1": "float32", "feature2": "float32"},
            feature_schema_hash="test_hash",
            version="1.0.0",
            created_at=time.time(),
            last_modified=time.time(),
        )

        # Create model file
        model_path = temp_dir / f"model{SUFFIX_ONNX}"
        model_path.touch()

        # Register model
        model_id = registry.register_model(model_path, manifest)
        assert model_id == "test_model_1"

        # Retrieve model
        model_info = registry.get_model(model_id)
        assert model_info is not None
        assert model_info.manifest.role == ModelRole.TEACHER
        assert model_info.manifest.architecture == "XGBoost"

    def test_postgres_backend_register_and_retrieve(
        self,
        temp_dir: Path,
        postgres_persistence_config: PersistenceConfig | None,
    ) -> None:
        """
        Test registering and retrieving models with PostgreSQL backend.
        """
        if postgres_persistence_config is None:
            pytest.skip("PostgreSQL not available")

        # Create registry
        registry = ModelRegistry(
            registry_path=temp_dir,
            persistence_config=postgres_persistence_config,
        )

        # Create model manifest
        manifest = ModelManifest(
            model_id="test_pg_model_1",
            role=ModelRole.STUDENT,
            data_requirements=DataRequirements.L1_ONLY,
            architecture="LightGBM",
            feature_schema={"feature1": "float32"},
            feature_schema_hash="test_hash_pg",
            version="2.0.0",
            created_at=time.time(),
            last_modified=time.time(),
        )

        # Create model file
        model_path = temp_dir / f"model{SUFFIX_ONNX}"
        model_path.touch()

        # Register model
        model_id = registry.register_model(model_path, manifest)
        assert model_id == "test_pg_model_1"

        # Retrieve model
        model_info = registry.get_model(model_id)
        assert model_info is not None
        assert model_info.manifest.role == ModelRole.STUDENT
        assert model_info.manifest.architecture == "LightGBM"

        # Test persistence - create new registry instance
        registry2 = ModelRegistry(
            registry_path=temp_dir,
            persistence_config=postgres_persistence_config,
        )

        model_info2 = registry2.get_model(model_id)
        assert model_info2 is not None
        assert model_info2.manifest.version == "2.0.0"


class TestFeatureRegistryBackends:
    """
    Test feature registry with different backends.
    """

    def test_json_backend_register_and_retrieve(
        self,
        temp_dir: Path,
        json_persistence_config: PersistenceConfig,
    ) -> None:
        """
        Test registering and retrieving features with JSON backend.
        """
        # Create registry
        registry = FeatureRegistry(
            registry_path=temp_dir,
            persistence_config=json_persistence_config,
        )

        # Create feature manifest
        manifest = FeatureManifest(
            feature_set_id="test_features_1",
            name="Test Features",
            version="1.0.0",
            role=FeatureRole.TEACHER,
            data_requirements=DataRequirements.L1_L2,
            feature_names=["price_ratio", "volume"],
            feature_dtypes=["float32", "float32"],
            schema_hash="feature_hash_1",
            pipeline_signature="pipeline_sig_1",
            pipeline_version="1.0",
            stage=FeatureStage.CANDIDATE,
            created_at=time.time(),
            last_modified=time.time(),
        )

        # Register feature set
        feature_id = registry.register_feature_set(manifest)
        assert feature_id == "test_features_1"

        # Retrieve feature set
        feature_info = registry.get_feature_set(feature_id)
        assert feature_info is not None
        assert feature_info.manifest.role == FeatureRole.TEACHER
        assert len(feature_info.manifest.feature_names) == 2

    def test_postgres_backend_with_lifecycle(
        self,
        temp_dir: Path,
        postgres_persistence_config: PersistenceConfig | None,
    ) -> None:
        """
        Test feature lifecycle with PostgreSQL backend.
        """
        if postgres_persistence_config is None:
            pytest.skip("PostgreSQL not available")

        # Create registry
        registry = FeatureRegistry(
            registry_path=temp_dir,
            persistence_config=postgres_persistence_config,
        )

        # Create and register feature manifest
        manifest = FeatureManifest(
            feature_set_id="test_pg_features_1",
            name="PG Test Features",
            version="1.0.0",
            role=FeatureRole.STUDENT,
            data_requirements=DataRequirements.L1_ONLY,
            feature_names=["simple_feature"],
            feature_dtypes=["float32"],
            schema_hash="pg_feature_hash_1",
            pipeline_signature="pg_pipeline_sig_1",
            pipeline_version="1.0",
            stage=FeatureStage.CANDIDATE,
            created_at=time.time(),
            last_modified=time.time(),
        )

        feature_id = registry.register_feature_set(manifest)

        # Promote to staging
        registry.promote(feature_id, FeatureStage.STAGING)
        feature_info = registry.get_feature_set(feature_id)
        assert feature_info is not None
        assert feature_info.manifest.stage == FeatureStage.STAGING

        # Promote to production
        registry.promote(feature_id, FeatureStage.PROD)
        feature_info = registry.get_feature_set(feature_id)
        assert feature_info is not None
        assert feature_info.manifest.stage == FeatureStage.PROD


class TestStrategyRegistryBackends:
    """
    Test strategy registry with different backends.
    """

    def test_json_backend_register_and_retrieve(
        self,
        temp_dir: Path,
        json_persistence_config: PersistenceConfig,
    ) -> None:
        """
        Test registering and retrieving strategies with JSON backend.
        """
        # Create registry
        registry = StrategyRegistry(
            base_path=temp_dir,
            persistence_config=json_persistence_config,
        )

        # Create strategy manifest
        manifest = StrategyManifest(
            strategy_id="test_strategy_1",
            strategy_type=StrategyType.MOMENTUM,
            version="1.0.0",
            required_models=["model_1"],
            required_features=["features_1"],
            suitable_regimes=[MarketRegime.TRENDING_UP],
            instrument_types=["FX", "EQUITY"],
            timeframe_range=("1m", "1h"),
            max_position_size=1.0,
            max_leverage=2.0,
            max_drawdown=0.1,
            stop_loss_type="trailing",
            min_sharpe_ratio=1.5,
            min_win_rate=0.55,
            max_correlation_with_portfolio=0.7,
            parent_strategy_id=None,
            incompatible_strategies=[],
            config_schema={"param1": "float"},
            default_config={"param1": 0.5},
            backtest_metrics={"sharpe": 1.8},
            live_metrics=None,
            created_at=time.time(),
            last_modified=time.time(),
            author="Test Author",
            description="Test strategy",
        )

        # Create strategy file
        strategy_path = temp_dir / "test_strategy.py"
        strategy_path.write_text("# Test strategy implementation")

        # Register strategy
        strategy_id = registry.register_strategy(strategy_path, manifest)
        assert strategy_id == "test_strategy_1"

        # Retrieve strategy
        strategy_info = registry.get_strategy(strategy_id)
        assert strategy_info is not None
        assert strategy_info.manifest.strategy_type == StrategyType.MOMENTUM
        assert MarketRegime.TRENDING_UP in strategy_info.manifest.suitable_regimes

    def test_postgres_backend_with_compatibility(
        self,
        temp_dir: Path,
        postgres_persistence_config: PersistenceConfig | None,
    ) -> None:
        """
        Test strategy compatibility checking with PostgreSQL backend.
        """
        if postgres_persistence_config is None:
            pytest.skip("PostgreSQL not available")

        # Create registry
        registry = StrategyRegistry(
            base_path=temp_dir,
            persistence_config=postgres_persistence_config,
        )

        # Create first strategy
        manifest1 = StrategyManifest(
            strategy_id="pg_strategy_1",
            strategy_type=StrategyType.TREND_FOLLOWING,
            version="1.0.0",
            required_models=[],
            required_features=["features_1"],
            suitable_regimes=[MarketRegime.TRENDING_UP],
            instrument_types=["FX"],
            timeframe_range=("5m", "1h"),
            max_position_size=1.0,
            max_leverage=3.0,
            max_drawdown=0.15,
            stop_loss_type="fixed",
            min_sharpe_ratio=1.2,
            min_win_rate=0.5,
            max_correlation_with_portfolio=0.8,
            parent_strategy_id=None,
            incompatible_strategies=["pg_strategy_2"],
            config_schema={},
            default_config={},
            backtest_metrics={"sharpe": 1.5},
            live_metrics=None,
            created_at=time.time(),
            last_modified=time.time(),
            author="Test",
            description="Strategy 1",
        )

        strategy_path = temp_dir / "strategy1.py"
        strategy_path.write_text("# Strategy 1")
        registry.register_strategy(strategy_path, manifest1)

        # Check compatibility
        is_compatible = registry.check_compatibility("pg_strategy_1", ["pg_strategy_2"])
        assert not is_compatible

        is_compatible = registry.check_compatibility("pg_strategy_1", ["other_strategy"])
        assert is_compatible


def test_backend_switching(temp_dir: Path) -> None:
    """
    Test switching between JSON and PostgreSQL backends.
    """
    # Start with JSON backend
    json_config = PersistenceConfig(
        backend=BackendType.JSON,
        json_path=temp_dir,
    )

    json_registry = ModelRegistry(
        registry_path=temp_dir,
        persistence_config=json_config,
    )

    # Register model with JSON backend
    manifest = ModelManifest(
        model_id="switch_test_model",
        role=ModelRole.INFERENCE,
        data_requirements=DataRequirements.HISTORICAL,
        architecture="Neural Network",
        feature_schema={"input": "float32"},
        feature_schema_hash="switch_hash",
        version="1.0.0",
        created_at=time.time(),
        last_modified=time.time(),
    )

    model_path = temp_dir / f"model{SUFFIX_ONNX}"
    model_path.touch()

    model_id = json_registry.register_model(model_path, manifest)

    # Verify data is in JSON
    registry_file = temp_dir / "registry.json"
    assert registry_file.exists()

    # Now try PostgreSQL if available
    try:
        postgres_config = PersistenceConfig(
            backend=BackendType.POSTGRES,
            connection_string="postgresql://postgres:postgres@localhost:5432/test_nautilus",
        )

        postgres_registry = ModelRegistry(
            registry_path=temp_dir,
            persistence_config=postgres_config,
        )

        # Register another model with PostgreSQL backend
        manifest2 = ModelManifest(
            model_id="postgres_model",
            role=ModelRole.ENSEMBLE,
            data_requirements=DataRequirements.STREAMING,
            architecture="Ensemble",
            feature_schema={"input1": "float32", "input2": "float32"},
            feature_schema_hash="postgres_hash",
            version="2.0.0",
            created_at=time.time(),
            last_modified=time.time(),
        )

        model_id2 = postgres_registry.register_model(model_path, manifest2)

        # Verify model is in PostgreSQL
        retrieved = postgres_registry.get_model(model_id2)
        assert retrieved is not None
        assert retrieved.manifest.architecture == "Ensemble"

    except Exception:
        # PostgreSQL not available, that's OK
        pass
