-- Update dataset lineage parents to reference canonical 'predictions' dataset
-- instead of the legacy 'predictions_xgboost'. Idempotent, safe to re-run.

UPDATE ml_dataset_registry
SET parents = jsonb_set(
    COALESCE(parents, '[]'::jsonb),
    '{0}',  -- first (and only) parent in seed
    '"predictions"'::jsonb,
    true
), last_modified = NOW()
WHERE dataset_id = 'signals_momentum'
  AND parents::text LIKE '%predictions_xgboost%';

