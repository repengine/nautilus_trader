-- Migration: Create macro observation table for long-format releases.
-- Rollback: DROP TABLE IF EXISTS ml.macro_observations CASCADE; DROP TABLE IF EXISTS ml.macro_observations_default CASCADE;

CREATE SCHEMA IF NOT EXISTS ml;
SET search_path TO public, pg_catalog, ml;

-- --------------------------------------------------------------------------
-- Macro observations (long format)
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ml.macro_observations (
    series_id VARCHAR(64) NOT NULL,
    observation_ts BIGINT NOT NULL,
    value DOUBLE PRECISION,
    ts_event BIGINT NOT NULL,
    ts_init BIGINT NOT NULL,
    source VARCHAR(32),
    run_id VARCHAR(64),
    PRIMARY KEY (series_id, observation_ts, ts_event)
) PARTITION BY RANGE (ts_event);

CREATE TABLE IF NOT EXISTS ml.macro_observations_default
    PARTITION OF ml.macro_observations DEFAULT;

CREATE INDEX IF NOT EXISTS idx_macro_observations_series
    ON ml.macro_observations (series_id);
CREATE INDEX IF NOT EXISTS idx_macro_observations_ts_event
    ON ml.macro_observations USING BRIN (ts_event);
