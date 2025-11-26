# ML Stores Migrations

This directory contains the SQL migrations for the ML persistence layer.

Two migration sets are supported to keep startup simple and production hardening opt‑in:

- Baseline (default): core schema needed for production
  - ml/registry/migrations/001_initial_schema.sql
  - ml/stores/migrations/001_stores_schema.sql
  - ml/stores/migrations/002_auto_partitioning.sql
  - ml/stores/migrations/003_market_data.sql
  - ml/stores/migrations/004_data_registry.sql
  - ml/stores/migrations/007_add_event_metadata.sql

- Optional (full): hardening, views, performance aids, and registry extensions
  - ml/stores/migrations/005_schema_hardening.sql
  - ml/stores/migrations/005_views.sql
  - ml/stores/migrations/006_disable_partition_triggers.sql
  - ml/stores/migrations/007_brin_indexes.sql
  - ml/registry/migrations/002_add_cold_path_fields.sql
  - ml/registry/migrations/003_add_artifact_digest.sql
  - ml/stores/migrations/008_predictions_alias.sql
  - ml/stores/migrations/009_update_parents_predictions.sql

How these are applied:

- CLI: `python -m ml.cli.apply_migrations --schema both` applies the Baseline plan.
  - Add `--full` to include the Optional plan.
- Integration (code): `MLIntegrationManager` uses the same plan builder.
  - Set `ML_MIGRATIONS_FULL=1` to include Optionals.
  - Set `ML_MIGRATIONS_SCHEMA=stores|registry|both` to scope the plan.

Notes
- All scripts are idempotent: re‑runs are safe; conflicting statements are tolerated.
- The `PartitionManager` runs a post‑migration maintenance pass to ensure partitions exist.
- Keep DDL‑heavy tests serial and reuse the SQL splitter from `ml/cli/apply_migrations.py`.
