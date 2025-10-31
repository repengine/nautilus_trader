#!/usr/bin/env python3
"""
Validate orchestrator configuration files.

Usage:
    python ml/config/orchestrator/validate_config.py spy_tft_production.toml
    python ml/config/orchestrator/validate_config.py --all
"""

from __future__ import annotations

import sys
from pathlib import Path

from ml.orchestration.config_loader import load_orchestrator_run_config


def validate_config(config_path: Path) -> bool:
    """Validate a single config file."""
    print(f"\n{'='*80}")
    print(f"Validating: {config_path.name}")
    print("="*80)

    try:
        # Load the config
        run_cfg = load_orchestrator_run_config(config_path)

        print("✓ Config loaded successfully")
        print(f"  Stage: {run_cfg.stage.value}")

        # Check dataset config
        if run_cfg.dataset:
            print("✓ Dataset config present")
            print(f"  Symbols: {run_cfg.dataset.symbols}")
            print(f"  Output: {run_cfg.dataset.out_dir}")
            print(f"  Lookback: {run_cfg.dataset.lookback_periods}")
            print(f"  Horizon: {run_cfg.dataset.horizon_minutes}m")
            print(f"  Include macro: {run_cfg.dataset.include_macro}")

            if run_cfg.dataset.validation:
                print("✓ Validation rules present")
                print(f"  Min rows: {run_cfg.dataset.validation.min_rows}")
                print(f"  Positive rate: {run_cfg.dataset.validation.min_positive_rate}-{run_cfg.dataset.validation.max_positive_rate}")

        # Check ingestion config
        if run_cfg.ingestion and run_cfg.ingestion.enabled:
            print("✓ Ingestion enabled")
            print(f"  Dataset: {run_cfg.ingestion.dataset_id}")
            print(f"  Schema: {run_cfg.ingestion.schema}")
            print(f"  Instruments: {run_cfg.ingestion.instruments}")

        # Check training config
        if run_cfg.training:
            if run_cfg.training.teacher.enabled:
                print("✓ Teacher training enabled")
                print(f"  Model ID: {run_cfg.training.teacher.model_id}")
                print(f"  Max epochs: {run_cfg.training.teacher.max_epochs}")

            if run_cfg.training.student.enabled:
                print("✓ Student distillation enabled")
                print(f"  Model ID: {run_cfg.training.student.model_id}")
                print(f"  Parent: {run_cfg.training.student.parent_model_id}")

            if run_cfg.training.hpo.enabled:
                print("✓ HPO enabled")
                print(f"  Trials: {run_cfg.training.hpo.optuna_trials}")

        # Check integration
        if run_cfg.integration and run_cfg.integration.enabled:
            print("✓ Integration enabled")
            if run_cfg.integration.db_connection:
                print(f"  Database: {run_cfg.integration.db_connection}")
                print(
                    "  ⚠ Hardcoded db_connection detected — prefer ML_DB_CONNECTION / NAUTILUS_DB env overrides.",
                )
            print(f"  Auto-migrate: {run_cfg.integration.auto_migrate}")

        # Try to compose orchestrator config
        try:
            run_cfg.compose_orchestrator_config()
            print("✓ Can compose OrchestratorConfig")
        except Exception as e:
            print(f"✗ Failed to compose OrchestratorConfig: {e}")
            return False

        print("\n✅ Config is valid!\n")
        return True

    except Exception as e:
        print("\n❌ Config validation failed:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main() -> None:
    """Main entry point."""
    config_dir = Path(__file__).parent

    # Check if --all flag
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        configs = sorted(config_dir.glob("*.toml"))
        results: dict[str, bool] = {}

        for config_path in configs:
            results[config_path.name] = validate_config(config_path)

        # Summary
        print(f"\n{'='*80}")
        print("VALIDATION SUMMARY")
        print("="*80)

        passed = sum(1 for v in results.values() if v)
        total = len(results)

        for name, valid in sorted(results.items()):
            status = "✅ PASS" if valid else "❌ FAIL"
            print(f"{status}: {name}")

        print(f"\nTotal: {passed}/{total} passed")

        sys.exit(0 if passed == total else 1)

    # Validate single file
    if len(sys.argv) < 2:
        print("Usage: python validate_config.py <config.toml>")
        print("       python validate_config.py --all")
        sys.exit(1)

    config_name: str = sys.argv[1]
    config_path = config_dir / config_name

    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    valid = validate_config(config_path)
    sys.exit(0 if valid else 1)


if __name__ == "__main__":
    main()
