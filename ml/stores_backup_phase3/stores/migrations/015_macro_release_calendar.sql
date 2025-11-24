-- Migration: Create macro release calendar partitions for ALFRED vintages.
-- Rollback: DROP TABLE IF EXISTS ml.macro_release_calendar CASCADE; DROP TABLE IF EXISTS ml.macro_release_calendar_default CASCADE;

CREATE SCHEMA IF NOT EXISTS ml;
SET search_path TO public, pg_catalog, ml;

-- --------------------------------------------------------------------------
-- Macro release calendar (ALFRED vintages)
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ml.macro_release_calendar (
    series_id VARCHAR(64) NOT NULL,
    observation_ts BIGINT NOT NULL,
    release_ts BIGINT NOT NULL,
    release_end_ts BIGINT,
    value DOUBLE PRECISION,
    ts_event BIGINT NOT NULL,
    ts_init BIGINT NOT NULL,
    source VARCHAR(32),
    run_id VARCHAR(64),
    PRIMARY KEY (series_id, observation_ts, release_ts, ts_event)
) PARTITION BY RANGE (ts_event);

CREATE TABLE IF NOT EXISTS ml.macro_release_calendar_default
    PARTITION OF ml.macro_release_calendar DEFAULT;

CREATE INDEX IF NOT EXISTS idx_macro_release_series
    ON ml.macro_release_calendar (series_id);
CREATE INDEX IF NOT EXISTS idx_macro_release_ts_event
    ON ml.macro_release_calendar USING BRIN (ts_event);
