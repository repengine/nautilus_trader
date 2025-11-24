-- Add metadata column to ml_data_events and provide extended emit function

-- Add metadata column if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_name = 'ml_data_events'
          AND column_name = 'metadata'
    ) THEN
        ALTER TABLE ml_data_events ADD COLUMN metadata JSONB;
    END IF;
END$$;

-- Extended function to emit data event with metadata JSON
CREATE OR REPLACE FUNCTION emit_data_event_ext(
    p_dataset_id VARCHAR(255),
    p_instrument_id VARCHAR(100),
    p_stage VARCHAR(50),
    p_source VARCHAR(50),
    p_run_id VARCHAR(255),
    p_ts_min BIGINT,
    p_ts_max BIGINT,
    p_count BIGINT,
    p_status VARCHAR(20),
    p_error TEXT DEFAULT NULL,
    p_metadata JSONB DEFAULT NULL,
    p_seq_min BIGINT DEFAULT NULL,
    p_seq_max BIGINT DEFAULT NULL
)
RETURNS BIGINT AS $$
DECLARE
    v_event_id BIGINT;
    v_ts_event BIGINT;
BEGIN
    -- Current timestamp in nanoseconds
    v_ts_event := EXTRACT(EPOCH FROM NOW()) * 1000000000;

    INSERT INTO ml_data_events (
        dataset_id, instrument_id, stage, source, run_id,
        ts_min, ts_max, ts_event, count, seq_min, seq_max,
        status, error, created_at, metadata
    )
    VALUES (
        p_dataset_id, p_instrument_id, p_stage, p_source, p_run_id,
        p_ts_min, p_ts_max, v_ts_event, p_count, p_seq_min, p_seq_max,
        p_status, p_error, NOW(), p_metadata
    )
    RETURNING event_id INTO v_event_id;

    IF p_status = 'success' THEN
        PERFORM update_watermark(
            p_dataset_id,
            p_instrument_id,
            p_source,
            p_ts_max,
            p_count,
            NULL
        );
    END IF;

    RETURN v_event_id;
END;
$$ LANGUAGE plpgsql;

