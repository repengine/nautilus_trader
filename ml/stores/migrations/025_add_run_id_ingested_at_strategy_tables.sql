ALTER TABLE IF EXISTS ml_strategy_signals
    ADD COLUMN IF NOT EXISTS run_id VARCHAR(255),
    ADD COLUMN IF NOT EXISTS ingested_at_ns BIGINT;

ALTER TABLE IF EXISTS ml_strategy_order_events
    ADD COLUMN IF NOT EXISTS run_id VARCHAR(255),
    ADD COLUMN IF NOT EXISTS ingested_at_ns BIGINT;

CREATE INDEX IF NOT EXISTS idx_ml_strategy_signals_run_id
    ON ml_strategy_signals (run_id);

CREATE INDEX IF NOT EXISTS idx_ml_strategy_order_events_run_id
    ON ml_strategy_order_events (run_id);
