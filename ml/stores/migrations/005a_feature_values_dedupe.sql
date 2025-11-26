-- Optional Pre-migration Deduplication for ml_feature_values
-- Run BEFORE 005_schema_hardening.sql if unique index creation might fail.
-- This script helps identify and (optionally) remove duplicates for the
-- upsert key (feature_set_id, instrument_id, ts_event).

-- 1) Detect duplicates
SELECT feature_set_id, instrument_id, ts_event, COUNT(*) AS dup_count
FROM public.ml_feature_values
GROUP BY feature_set_id, instrument_id, ts_event
HAVING COUNT(*) > 1
ORDER BY dup_count DESC, feature_set_id, instrument_id, ts_event
LIMIT 100;

-- 2) Inspect rows for one duplicate key
-- Replace placeholders as needed
-- SELECT * FROM public.ml_feature_values
-- WHERE feature_set_id = '<FEATURE_SET>'
--   AND instrument_id = '<INSTRUMENT>'
--   AND ts_event = <TS_EVENT>
-- ORDER BY created_at DESC NULLS LAST;

-- 3) Optional deduplication strategy: keep the most recent row by created_at,
-- delete older rows. REVIEW BEFORE RUNNING.
-- WITH ranked AS (
--   SELECT ctid, ROW_NUMBER() OVER (
--     PARTITION BY feature_set_id, instrument_id, ts_event
--     ORDER BY created_at DESC NULLS LAST, id DESC NULLS LAST
--   ) AS rn
--   FROM public.ml_feature_values
-- )
-- DELETE FROM public.ml_feature_values m
-- USING ranked r
-- WHERE m.ctid = r.ctid AND r.rn > 1;

