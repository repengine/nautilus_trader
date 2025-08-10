#!/usr/bin/env python3

"""
Example workflow demonstrating the Model Registry orchestration.

This shows how the registry coordinates all ML components from
training to deployment and monitoring.
"""

from __future__ import annotations

import time
from pathlib import Path

from ml.registry import LocalModelRegistry
from ml.registry import ModelDeploymentManager


def demonstrate_registry_workflow() -> None:
    """
    Demonstrate complete model lifecycle management.

    This example shows:
    1. Model registration after training
    2. Deployment to actors
    3. Performance tracking
    4. A/B testing
    5. Hot reload
    6. Rollback
    """
    # Initialize registry
    registry_path = Path("ml/tests/data/model_registry")
    registry = LocalModelRegistry(registry_path)
    deployment_manager = ModelDeploymentManager(registry)

    print("=" * 60)
    print("ML Model Registry Workflow Demonstration")
    print("=" * 60)

    # ========================================
    # 1. TRAINING PHASE (Agent 3's domain)
    # ========================================
    print("\n1. Training Phase:")
    print("-" * 40)

    # Simulate training completion
    model_v1_path = registry_path / "models" / "xgb_v1.json"
    model_v1_path.parent.mkdir(parents=True, exist_ok=True)
    model_v1_path.write_text('{"model": "v1"}')  # Dummy model file

    # Register trained model with metadata
    model_v1_id = registry.register_model(
        model_path=model_v1_path,
        metadata={
            "trained_by": "XGBoostTrainer",
            "features": ["sma_10", "rsi_14", "volume_ratio"],
            "training_metrics": {
                "accuracy": 0.92,
                "auc": 0.88,
                "f1_score": 0.85,
            },
            "training_samples": 100000,
            "training_date": time.time(),
        },
        version="1.0.0"
    )

    print(f"✓ Registered model: {model_v1_id}")
    print("  Version: 1.0.0")
    print("  Features: sma_10, rsi_14, volume_ratio")
    print("  Training accuracy: 0.92")

    # ========================================
    # 2. DEPLOYMENT PHASE (Actor integration)
    # ========================================
    print("\n2. Deployment Phase:")
    print("-" * 40)

    # Deploy to MLSignalActor
    deployment_config = {
        "target": "ml_signal_actor",
        "instruments": ["EURUSD", "GBPUSD"],
        "max_positions": 5,
        "confidence_threshold": 0.7,
    }

    deployment_id = deployment_manager.deploy(
        model_id=model_v1_id,
        config=deployment_config
    )

    print("✓ Deployed to MLSignalActor")
    print(f"  Deployment ID: {deployment_id}")
    print("  Instruments: EURUSD, GBPUSD")
    print("  Max positions: 5")

    # Check deployment status
    if deployment_id is not None:
        status = deployment_manager.get_deployment_status(deployment_id)
        print(f"  Status: {'ACTIVE' if status and status['is_active'] else 'INACTIVE'}")

    # ========================================
    # 3. MONITORING PHASE (Live performance)
    # ========================================
    print("\n3. Monitoring Phase:")
    print("-" * 40)

    # Track live performance metrics
    for hour in range(3):
        metrics = {
            "live_accuracy": 0.90 + hour * 0.01,
            "pnl": 1000 + hour * 500,
            "trades": 10 + hour * 5,
            "sharpe_ratio": 1.5 + hour * 0.1,
            "max_drawdown": 0.05 - hour * 0.01,
        }
        registry.track_performance(model_v1_id, metrics)
        print(f"  Hour {hour+1}: Accuracy={metrics['live_accuracy']:.2f}, PnL=${metrics['pnl']}")

    # Get performance history
    history = registry.get_performance_history(model_v1_id)
    print(f"✓ Tracked {len(history)} performance snapshots")

    # ========================================
    # 4. MODEL UPDATE (New version available)
    # ========================================
    print("\n4. Model Update Phase:")
    print("-" * 40)

    # Train and register v2 model
    model_v2_path = registry_path / "models" / "xgb_v2.json"
    model_v2_path.write_text('{"model": "v2"}')

    model_v2_id = registry.register_model(
        model_path=model_v2_path,
        metadata={
            "trained_by": "XGBoostTrainer",
            "features": ["sma_10", "rsi_14", "volume_ratio", "atr_20"],  # Added feature
            "training_metrics": {
                "accuracy": 0.94,
                "auc": 0.91,
                "f1_score": 0.89,
            },
            "improvements": "Added ATR feature, improved hyperparameters",
        },
        version="2.0.0"
    )

    print(f"✓ Registered new model: {model_v2_id}")
    print("  Version: 2.0.0")
    print("  New feature: atr_20")
    print("  Training accuracy: 0.94 (+2% improvement)")

    # ========================================
    # 5. A/B TESTING PHASE
    # ========================================
    print("\n5. A/B Testing Phase:")
    print("-" * 40)

    # Configure A/B test
    ab_config = registry.configure_ab_test(
        models=[model_v1_id, model_v2_id],
        split_ratio=0.5,  # 50/50 traffic split
        duration_hours=24,
        target="ml_signal_actor"
    )

    if ab_config:
        print("✓ Started A/B test")
        print(f"  Model A (current): {model_v1_id}")
        print(f"  Model B (challenger): {model_v2_id}")
        print("  Traffic split: 50/50")
        print("  Duration: 24 hours")

    # Simulate A/B test results
    registry.track_performance(model_v2_id, {
        "live_accuracy": 0.93,
        "pnl": 2000,
        "trades": 30,
    })

    # Compare models
    comparison = registry.compare_models(
        model_ids=[model_v1_id, model_v2_id],
        metric="pnl"
    )

    if comparison:
        print("\n  A/B Test Results:")
        print(f"  Best model: {comparison['best_model']}")
        for rank in comparison["rankings"]:
            print(f"    {rank['model_id']}: PnL=${rank.get('pnl', 0)}")

    # ========================================
    # 6. HOT RELOAD (Zero-downtime update)
    # ========================================
    print("\n6. Hot Reload Phase:")
    print("-" * 40)

    # Hot reload to v2 based on A/B test results
    if deployment_id:
        success = deployment_manager.hot_reload(
            deployment_id=deployment_id,
            new_model_id=model_v2_id
        )

        if success:
            print("✓ Hot reloaded to v2.0.0")
            print("  No downtime during transition")
            print("  All connections maintained")

    # ========================================
    # 7. ROLLBACK CAPABILITY
    # ========================================
    print("\n7. Rollback Capability:")
    print("-" * 40)

    # Simulate issue with v2
    print("  Simulating performance degradation...")
    registry.track_performance(model_v2_id, {
        "live_accuracy": 0.85,  # Degraded
        "pnl": -500,  # Loss
        "alert": "Accuracy dropped below threshold"
    })

    # Rollback to v1
    success = registry.rollback(
        target="ml_signal_actor",
        to_model_id=model_v1_id
    )

    if success:
        print("✓ Rolled back to v1.0.0")
        print("  Restored previous stable version")
        print("  Model v2 marked for investigation")

    # ========================================
    # 8. REGISTRY STATUS
    # ========================================
    print("\n8. Registry Status:")
    print("-" * 40)

    # Get all models
    all_models = registry.get_all_models()
    print(f"Total models registered: {len(all_models)}")

    for model in all_models:
        print(f"\n  Model: {model.model_id}")
        print(f"    Version: {model.version}")
        print(f"    Status: {model.deployment_status.value}")
        print(f"    Deployed to: {', '.join(model.deployed_to) if model.deployed_to else 'None'}")
        perf_count = len(model.performance_history)
        if perf_count > 0:
            print(f"    Performance entries: {perf_count}")

    # Get active models
    active_models = registry.get_active_models()
    print(f"\nActive models: {len(active_models)}")
    for model in active_models:
        print(f"  - {model.model_id} (v{model.version})")

    print("\n" + "=" * 60)
    print("Registry workflow demonstration complete!")
    print("=" * 60)


def demonstrate_gradual_rollout() -> None:
    """
    Demonstrate gradual rollout strategy.

    This shows how to safely deploy a new model
    with progressive traffic increases.
    """
    print("\n" + "=" * 60)
    print("Gradual Rollout Demonstration")
    print("=" * 60)

    registry_path = Path("ml/tests/data/model_registry_rollout")
    registry = LocalModelRegistry(registry_path)
    deployment_manager = ModelDeploymentManager(registry)

    # Create dummy models
    model_v1_path = registry_path / "models" / "prod.onnx"
    model_v2_path = registry_path / "models" / "new.onnx"
    model_v1_path.parent.mkdir(parents=True, exist_ok=True)
    model_v1_path.write_text('{"model": "production"}')
    model_v2_path.write_text('{"model": "new"}')

    # Register models
    prod_model = registry.register_model(
        model_path=model_v1_path,
        metadata={"name": "production_model"},
    )

    new_model = registry.register_model(
        model_path=model_v2_path,
        metadata={"name": "new_model"},
    )

    # Deploy production model
    deployment_id = deployment_manager.deploy(
        model_id=prod_model,
        config={"target": "ml_signal_actor"}
    )

    if deployment_id:
        print("\nStarting gradual rollout:")
        print(f"  Current model: {prod_model}")
        print(f"  New model: {new_model}")

        # Define rollout stages
        rollout_id = deployment_manager.gradual_rollout(
            deployment_id=deployment_id,
            new_model_id=new_model,
            stages=[0.1, 0.25, 0.5, 1.0],  # 10% -> 25% -> 50% -> 100%
            stage_duration_minutes=60,  # 1 hour per stage
        )

        print(f"\nRollout plan created: {rollout_id}")
        print("  Stage 1: 10% traffic (1 hour)")
        print("  Stage 2: 25% traffic (1 hour)")
        print("  Stage 3: 50% traffic (1 hour)")
        print("  Stage 4: 100% traffic (complete)")
        print("\nTotal rollout time: 4 hours")


if __name__ == "__main__":
    # Run the demonstrations
    demonstrate_registry_workflow()
    demonstrate_gradual_rollout()
