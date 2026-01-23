# ML Stores Migrations

This directory now hosts **incremental** ML store migrations only.
The canonical layout is split into three tracks:

- `ml/stores/migrations_bootstrap` - single consolidated bootstrap for new databases
- `ml/stores/migrations` - incremental migrations after the bootstrap
- `ml/stores/migrations_legacy` - archived legacy history for existing databases

How they are applied:

- Default runner behavior (`ML_MIGRATIONS_PROFILE=auto`) applies bootstrap
  migrations when `ml_schema_migrations` is empty, then applies incremental
  migrations.
- To replay the legacy chain (existing DBs), set
  `ML_MIGRATIONS_PROFILE=legacy`.
- To skip bootstrap and run incremental only, set
  `ML_MIGRATIONS_PROFILE=incremental`.

Notes:
- Do not delete or reorder migrations that may already be applied.
- Keep test-only seeds in separate migrations (not in bootstrap).
- Use the SQL splitter in `ml/tasks/db.py` when executing migrations.
- Per-class market data tables are introduced in `021_market_data_class_tables.sql`;
  the bootstrap creates those tables plus a `market_data` compatibility view for
  legacy readers.
- Dataset type constraint updates (macro/features) are applied in
  `022_update_dataset_type_constraint.sql`.
