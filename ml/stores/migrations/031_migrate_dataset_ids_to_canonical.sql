-- Migrate registry dataset_id naming to the canonical schema registry.
-- Converts ohlcv_* -> bars_*, mbp_* -> mbp1_* or mbp10_* based on dataset_type.

CREATE TEMP TABLE tmp_dataset_id_mapping ON COMMIT DROP AS
SELECT
    mapped.old_dataset_id,
    mapped.new_dataset_id
FROM (
    SELECT
        dataset_id AS old_dataset_id,
        CASE
            WHEN dataset_id LIKE 'ohlcv\_%' THEN regexp_replace(dataset_id, '^ohlcv_', 'bars_')
            WHEN dataset_id LIKE 'mbp\_%' AND dataset_type = 'MBP1' THEN regexp_replace(dataset_id, '^mbp_', 'mbp1_')
            WHEN dataset_id LIKE 'mbp\_%' AND dataset_type = 'MBP10' THEN regexp_replace(dataset_id, '^mbp_', 'mbp10_')
            ELSE dataset_id
        END AS new_dataset_id
    FROM public.ml_dataset_registry
    WHERE dataset_id LIKE 'ohlcv\_%' OR dataset_id LIKE 'mbp\_%'
) AS mapped
WHERE mapped.new_dataset_id <> mapped.old_dataset_id;

INSERT INTO public.ml_dataset_registry (
    dataset_id,
    name,
    version,
    dataset_type,
    storage_kind,
    location,
    partitioning,
    retention_days,
    schema,
    schema_hash,
    constraints,
    parents,
    pipeline_signature,
    created_at,
    last_modified,
    metadata
)
SELECT
    mapping.new_dataset_id,
    registry.name,
    registry.version,
    registry.dataset_type,
    registry.storage_kind,
    registry.location,
    registry.partitioning,
    registry.retention_days,
    registry.schema,
    registry.schema_hash,
    registry.constraints,
    registry.parents,
    registry.pipeline_signature,
    registry.created_at,
    registry.last_modified,
    CASE
        WHEN registry.metadata IS NULL THEN jsonb_build_object('dataset_id', mapping.new_dataset_id)
        ELSE jsonb_set(registry.metadata, '{dataset_id}', to_jsonb(mapping.new_dataset_id), true)
    END AS metadata
FROM public.ml_dataset_registry AS registry
JOIN tmp_dataset_id_mapping AS mapping
    ON registry.dataset_id = mapping.old_dataset_id
ON CONFLICT (dataset_id) DO NOTHING;

UPDATE public.ml_dataset_registry AS registry
SET parents = COALESCE(
    (
        SELECT jsonb_agg(COALESCE(mapping.new_dataset_id, parent_token.value) ORDER BY parent_token.ordinality)
        FROM jsonb_array_elements_text(registry.parents) WITH ORDINALITY AS parent_token(value, ordinality)
        LEFT JOIN tmp_dataset_id_mapping AS mapping
            ON mapping.old_dataset_id = parent_token.value
    ),
    '[]'::jsonb
)
WHERE registry.parents IS NOT NULL
  AND jsonb_typeof(registry.parents) = 'array';

UPDATE public.ml_data_events AS events
SET dataset_id = mapping.new_dataset_id
FROM tmp_dataset_id_mapping AS mapping
WHERE events.dataset_id = mapping.old_dataset_id;

UPDATE public.ml_data_watermarks AS watermarks
SET dataset_id = mapping.new_dataset_id
FROM tmp_dataset_id_mapping AS mapping
WHERE watermarks.dataset_id = mapping.old_dataset_id;

UPDATE public.ml_data_lineage AS lineage
SET child_dataset_id = mapping.new_dataset_id
FROM tmp_dataset_id_mapping AS mapping
WHERE lineage.child_dataset_id = mapping.old_dataset_id;

UPDATE public.ml_data_lineage AS lineage
SET parent_dataset_id = mapping.new_dataset_id
FROM tmp_dataset_id_mapping AS mapping
WHERE lineage.parent_dataset_id = mapping.old_dataset_id;

DELETE FROM public.ml_dataset_registry AS registry
USING tmp_dataset_id_mapping AS mapping
WHERE registry.dataset_id = mapping.old_dataset_id;
