-- Update ml_data_events stage constraint to include order event emissions.

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
                'ORDER_EVENT_EMITTED'
            ]::text[]
        )
    );
