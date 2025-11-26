#!/usr/bin/env python3

"""
Demo of using PostgreSQL-backed registries for ML components.

This example shows how to configure and use the PostgreSQL backend for storing model,
feature, and strategy manifests with full versioning, audit logging, and metadata
tracking.

"""

from __future__ import annotations

import time
from pathlib import Path

from ml.config.constants import SUFFIX_ONNX
from ml.registry import DataRequirements
from ml.registry import ModelManifest
from ml.registry import ModelRegistry
from ml.registry import ModelRole
from ml.registry.persistence import BackendType
from ml.registry.persistence import PersistenceConfig


def demo_json_backend() -> None:
    """
    Demo using traditional JSON file backend.
    """
    print("\n=== JSON Backend Demo ===")

    # Create registry with JSON backend (default)
    registry_path = Path(".cache/ml_registry_json")
    registry = ModelRegistry(
        registry_path=registry_path,
        cache_size=10,
        batch_save_interval=0.1,
    )

    # Create a model manifest
    manifest = ModelManifest(
        model_id="xgb_predictor_v1",
        role=ModelRole.TEACHER,
        data_requirements=DataRequirements.L1_L2,
        architecture="XGBoost",
        feature_schema={
            "price_ratio": "float32",
            "volume_imbalance": "float32",
            "spread": "float32",
        },
        feature_schema_hash="abc123def456",
        version="1.0.0",
        created_at=time.time(),
        last_modified=time.time(),
    )

    # Register model (assumes model file exists)
    model_path = registry_path / f"models/model{SUFFIX_ONNX}"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.touch()  # Create dummy file for demo

    model_id = registry.register_model(
        model_path=model_path,
        manifest=manifest,
    )

    print(f"Registered model: {model_id}")

    # Query model
    model_info = registry.get_model(model_id)
    if model_info is not None:
        print(f"Model role: {model_info.manifest.role}")
        print(f"Feature schema: {model_info.manifest.feature_schema}")
    else:
        print("Model not found in registry lookup")

    # List all models
    all_models = registry.get_all_models()
    print(f"Total models in registry: {len(all_models)}")


def demo_postgres_backend() -> None:
    """
    Demo using PostgreSQL backend for production.
    """
    print("\n=== PostgreSQL Backend Demo ===")

    # Configure PostgreSQL persistence
    persistence_config = PersistenceConfig(
        backend=BackendType.POSTGRES,
        connection_string="postgresql://postgres:postgres@localhost:5432/nautilus",
        pool_size=5,
        max_overflow=10,
        echo=False,  # Set to True to see SQL queries
    )

    # Create registry with PostgreSQL backend
    registry_path = Path(".cache/ml_registry_postgres")
    registry = ModelRegistry(
        registry_path=registry_path,  # Still needed for model file storage
        cache_size=10,
        batch_save_interval=0.1,
        persistence_config=persistence_config,
    )

    print("Connected to PostgreSQL backend")

    # Create teacher model manifest
    teacher_manifest = ModelManifest(
        model_id="teacher_xgb_v2",
        role=ModelRole.TEACHER,
        data_requirements=DataRequirements.L1_L2_L3,
        architecture="XGBoost",
        feature_schema={
            "microstructure_feature_1": "float32",
            "microstructure_feature_2": "float32",
            "order_flow_imbalance": "float32",
            "book_pressure": "float32",
        },
        feature_schema_hash="teacher_hash_789",
        performance_metrics={
            "accuracy": 0.92,
            "sharpe_ratio": 2.1,
            "max_drawdown": 0.08,
        },
        version="2.0.0",
        created_at=time.time(),
        last_modified=time.time(),
    )

    # Register teacher model
    teacher_path = registry_path / f"models/teacher{SUFFIX_ONNX}"
    teacher_path.parent.mkdir(parents=True, exist_ok=True)
    teacher_path.touch()

    teacher_id = registry.register_model(
        model_path=teacher_path,
        manifest=teacher_manifest,
    )

    print(f"Registered teacher model: {teacher_id}")

    # Create student model manifest (distilled from teacher)
    student_manifest = ModelManifest(
        model_id="student_lgb_v1",
        role=ModelRole.STUDENT,
        data_requirements=DataRequirements.L1_ONLY,
        architecture="LightGBM",
        feature_schema={
            "price_ratio": "float32",
            "volume_signal": "float32",
        },
        feature_schema_hash="student_hash_456",
        parent_id=teacher_id,  # Link to teacher
        performance_metrics={
            "accuracy": 0.88,
            "sharpe_ratio": 1.8,
            "max_drawdown": 0.10,
            "inference_latency_ms": 0.5,
        },
        deployment_constraints={
            "max_latency_ms": 1.0,
            "max_memory_mb": 50,
        },
        version="1.0.0",
        created_at=time.time(),
        last_modified=time.time(),
    )

    # Register student model
    student_path = registry_path / f"models/student{SUFFIX_ONNX}"
    student_path.touch()

    student_id = registry.register_model(
        model_path=student_path,
        manifest=student_manifest,
    )

    print(f"Registered student model: {student_id} (distilled from {teacher_id})")

    # Query models by role
    teachers = registry.get_models_by_role(ModelRole.TEACHER)
    students = registry.get_models_by_role(ModelRole.STUDENT)

    print(f"\nTeacher models: {len(teachers)}")
    for teacher in teachers:
        print(f"  - {teacher.manifest.model_id}: {teacher.manifest.architecture}")

    print(f"\nStudent models: {len(students)}")
    for student in students:
        print(f"  - {student.manifest.model_id}: parent={student.manifest.parent_id}")

    # Deploy a model
    registry.deploy_model(student_id, target="production_strategy_v1")
    print(f"\nDeployed {student_id} to production_strategy_v1")

    # Query deployment status
    model_info = registry.get_model(student_id)
    if model_info:
        print(f"Deployment status: {model_info.deployment_status}")
        print(f"Deployed to: {model_info.deployed_to}")

    # Benefits of PostgreSQL backend:
    print("\n=== PostgreSQL Backend Benefits ===")
    print("1. ACID compliance - all operations are transactional")
    print("2. Concurrent access - multiple processes can safely read/write")
    print("3. Rich querying - use SQL for complex lineage/dependency queries")
    print("4. Audit logging - automatic tracking of all changes")
    print("5. Scalability - handles thousands of models efficiently")
    print("6. Integration - can be queried by BI tools, monitoring systems")
    print("7. Backup/Recovery - standard PostgreSQL tools")


def demo_migration() -> None:
    """
    Demo migrating from JSON to PostgreSQL backend.
    """
    print("\n=== Migration Demo ===")

    # Step 1: Load existing JSON registry
    json_registry = ModelRegistry(
        registry_path=Path(".cache/ml_registry_json"),
    )

    # Step 2: Create PostgreSQL registry
    postgres_config = PersistenceConfig(
        backend=BackendType.POSTGRES,
        connection_string="postgresql://postgres:postgres@localhost:5432/nautilus",
    )

    postgres_registry = ModelRegistry(
        registry_path=Path(".cache/ml_registry_postgres"),
        persistence_config=postgres_config,
    )

    # Step 3: Migrate all models
    migrated_count = 0
    for model_info in json_registry.get_all_models():
        # Re-register in PostgreSQL backend
        postgres_registry.register_model(
            model_path=model_info.model_path,
            manifest=model_info.manifest,
        )
        migrated_count += 1

    print(f"Migrated {migrated_count} models from JSON to PostgreSQL")


if __name__ == "__main__":
    # Run demos
    demo_json_backend()

    # Note: PostgreSQL demo requires a running PostgreSQL instance
    # You can use the Nautilus PostgreSQL container:
    # docker compose -f ml/deployment/docker-compose.yml up -d postgres

    try:
        demo_postgres_backend()
        demo_migration()
    except Exception as e:
        print(f"\nPostgreSQL demo skipped: {e}")
        print("Make sure PostgreSQL is running and accessible")
        print("You can use: docker compose -f ml/deployment/docker-compose.yml up -d postgres")
