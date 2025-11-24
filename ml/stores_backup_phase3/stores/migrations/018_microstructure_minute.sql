-- Migration: Create microstructure per-minute feature table (L1 aggregates).
-- Rollback: DROP TABLE IF EXISTS ml.microstructure_minute CASCADE; DROP TABLE IF EXISTS ml.microstructure_minute_default CASCADE;

CREATE SCHEMA IF NOT EXISTS ml;
SET search_path TO public, pg_catalog, ml;

-- --------------------------------------------------------------------------
-- Microstructure per-minute features (L1 derived)
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ml.microstructure_minute (
    instrument_id VARCHAR(32) NOT NULL,
    "timestamp" BIGINT NOT NULL,
    ts_event BIGINT NOT NULL,
    ts_init BIGINT NOT NULL,
    midprice DOUBLE PRECISION,
    spread_bps DOUBLE PRECISION,
    quote_imbalance DOUBLE PRECISION,
    trade_imbalance DOUBLE PRECISION,
    realized_vol DOUBLE PRECISION,
    PRIMARY KEY (instrument_id, "timestamp", ts_event)
) PARTITION BY RANGE (ts_event);

CREATE TABLE IF NOT EXISTS ml.microstructure_minute_default
    PARTITION OF ml.microstructure_minute DEFAULT;

CREATE INDEX IF NOT EXISTS idx_microstructure_minute_brin
    ON ml.microstructure_minute USING BRIN (ts_event);
CREATE INDEX IF NOT EXISTS idx_microstructure_minute_instrument
    ON ml.microstructure_minute (instrument_id);
