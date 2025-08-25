-- DEPRECATED: See ml/stores/migrations/001_stores_schema.sql for the canonical
-- partitioned JSONB-based feature storage schema. This file is retained only
-- for historical reference and should not be used to provision databases.

-- Feature values table
-- Stores computed ML features alongside Nautilus market data
-- Legacy example (array-based) — do NOT use; provided for reference only.
CREATE TABLE IF NOT EXISTS ml_feature_values (
    instrument_id VARCHAR(100) NOT NULL,
    ts_event BIGINT NOT NULL,              -- Nautilus convention: nanoseconds since epoch
    feature_version VARCHAR(64) NOT NULL,   -- Hash of pipeline/config for versioning
    features FLOAT8[],                      -- Array of feature values
    feature_names JSONB,                    -- Mapping of index to feature name
    created_at BIGINT NOT NULL,            -- When features were computed (nanoseconds)

    PRIMARY KEY (instrument_id, ts_event, feature_version)
);

-- Indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_ml_features_instrument_time
    ON ml_feature_values(instrument_id, ts_event);

CREATE INDEX IF NOT EXISTS idx_ml_features_version
    ON ml_feature_values(feature_version);

-- Comments for documentation
COMMENT ON TABLE ml_feature_values IS 'Computed ML features for training and inference';
COMMENT ON COLUMN ml_feature_values.ts_event IS 'Event timestamp in nanoseconds (Nautilus convention)';
COMMENT ON COLUMN ml_feature_values.feature_version IS 'Hash identifying the feature computation pipeline version';
COMMENT ON COLUMN ml_feature_values.features IS 'Array of computed feature values';
COMMENT ON COLUMN ml_feature_values.feature_names IS 'JSON mapping of array indices to feature names';

-- Example query to join with Nautilus bar data for training:
-- SELECT
--     b.ts_event,
--     b.close,
--     b.volume,
--     f.features,
--     f.feature_names
-- FROM bar b
-- INNER JOIN ml_feature_values f
--     ON b.instrument_id = f.instrument_id
--     AND b.ts_event = f.ts_event
-- WHERE b.instrument_id = 'EURUSD'
-- AND b.ts_event BETWEEN 1704067200000000000 AND 1706745600000000000
-- ORDER BY b.ts_event;
