-- Add strategy risk-halt events and replay summary tables, plus new stages.

CREATE TABLE IF NOT EXISTS public.ml_strategy_risk_halt_events (
    event_id VARCHAR(64) NOT NULL,
    strategy_id VARCHAR(255) NOT NULL,
    instrument_id VARCHAR(100) NOT NULL,
    event_type VARCHAR(32) NOT NULL,
    reason VARCHAR(255) NOT NULL,
    detail TEXT,
    ts_event BIGINT NOT NULL,
    ts_init BIGINT,
    is_live BOOLEAN DEFAULT FALSE,
    run_id VARCHAR(255),
    ingested_at_ns BIGINT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (event_id)
);

CREATE TABLE IF NOT EXISTS public.ml_strategy_replay_summary (
    run_id VARCHAR(255) NOT NULL,
    instrument_ids JSONB,
    started_ns BIGINT,
    finished_ns BIGINT,
    total_orders BIGINT,
    total_fills BIGINT,
    total_halts BIGINT,
    total_sizing_rejects BIGINT,
    total_positions BIGINT,
    ts_event BIGINT,
    ts_init BIGINT,
    ingested_at_ns BIGINT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (run_id)
);

CREATE INDEX IF NOT EXISTS idx_ml_strategy_risk_halt_events_lookup
    ON public.ml_strategy_risk_halt_events (strategy_id, instrument_id, ts_event);
CREATE INDEX IF NOT EXISTS idx_ml_strategy_risk_halt_events_type
    ON public.ml_strategy_risk_halt_events (event_type);
CREATE INDEX IF NOT EXISTS idx_ml_strategy_replay_summary
    ON public.ml_strategy_replay_summary (run_id);

ALTER TABLE public.ml_data_events
    DROP CONSTRAINT IF EXISTS check_stage;

ALTER TABLE public.ml_data_events
    ADD CONSTRAINT check_stage
    CHECK (
        (stage)::text = ANY (
            ARRAY[
                'INGESTED',
                'CATALOG_WRITTEN',
                'FEATURE_COMPUTED',
                'PREDICTION_EMITTED',
                'SIGNAL_EMITTED',
                'MODEL_INFERRED',
                'ORDER_EVENT_EMITTED',
                'RISK_HALT_EMITTED',
                'REPLAY_SUMMARY_EMITTED'
            ]::text[]
        )
    );
