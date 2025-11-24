-- Migration: Add events calendar storage for structured macro/event ingest.
-- Rollback: DROP TABLE IF EXISTS ml.events_calendar CASCADE; DROP TABLE IF EXISTS ml.events_calendar_default CASCADE;

CREATE SCHEMA IF NOT EXISTS ml;
SET search_path TO public, pg_catalog, ml;

-- --------------------------------------------------------------------------
-- Events calendar
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ml.events_calendar (
    event_timestamp BIGINT NOT NULL,
    event_type VARCHAR(64) NOT NULL,
    name VARCHAR(255) NOT NULL,
    instrument_id VARCHAR(64) NOT NULL,
    importance VARCHAR(32),
    source VARCHAR(64),
    metadata JSONB,
    ts_event BIGINT NOT NULL,
    ts_init BIGINT NOT NULL,
    PRIMARY KEY (event_type, event_timestamp, instrument_id, name, ts_event)
) PARTITION BY RANGE (ts_event);

CREATE TABLE IF NOT EXISTS ml.events_calendar_default
    PARTITION OF ml.events_calendar DEFAULT;

CREATE INDEX IF NOT EXISTS idx_events_calendar_ts_event
    ON ml.events_calendar USING BRIN (ts_event);
