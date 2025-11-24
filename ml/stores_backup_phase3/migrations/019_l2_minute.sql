-- Migration: Create L2 depth per-minute feature table.
-- Rollback: DROP TABLE IF EXISTS ml.l2_minute CASCADE; DROP TABLE IF EXISTS ml.l2_minute_default CASCADE;

CREATE SCHEMA IF NOT EXISTS ml;
SET search_path TO public, pg_catalog, ml;

-- --------------------------------------------------------------------------
-- L2 depth per-minute features
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ml.l2_minute (
    instrument_id VARCHAR(32) NOT NULL,
    "timestamp" BIGINT NOT NULL,
    ts_event BIGINT NOT NULL,
    ts_init BIGINT NOT NULL,
    midprice DOUBLE PRECISION,
    spread_bps DOUBLE PRECISION,
    microprice_bps DOUBLE PRECISION,
    depth_imbalance_top1 DOUBLE PRECISION,
    depth_imbalance_top3 DOUBLE PRECISION,
    depth_imbalance_top5 DOUBLE PRECISION,
    depth_imbalance_top10 DOUBLE PRECISION,
    dwp_bps_top1 DOUBLE PRECISION,
    dwp_bps_top3 DOUBLE PRECISION,
    dwp_bps_top5 DOUBLE PRECISION,
    dwp_bps_top10 DOUBLE PRECISION,
    bid_slope_top1 DOUBLE PRECISION,
    bid_slope_top3 DOUBLE PRECISION,
    bid_slope_top5 DOUBLE PRECISION,
    bid_slope_top10 DOUBLE PRECISION,
    ask_slope_top1 DOUBLE PRECISION,
    ask_slope_top3 DOUBLE PRECISION,
    ask_slope_top5 DOUBLE PRECISION,
    ask_slope_top10 DOUBLE PRECISION,
    PRIMARY KEY (instrument_id, "timestamp", ts_event)
) PARTITION BY RANGE (ts_event);

CREATE TABLE IF NOT EXISTS ml.l2_minute_default
    PARTITION OF ml.l2_minute DEFAULT;

CREATE INDEX IF NOT EXISTS idx_l2_minute_brin
    ON ml.l2_minute USING BRIN (ts_event);
CREATE INDEX IF NOT EXISTS idx_l2_minute_instrument
    ON ml.l2_minute (instrument_id);
