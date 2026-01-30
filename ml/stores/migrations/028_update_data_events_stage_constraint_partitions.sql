-- Refresh check_stage constraint on ml_data_events partitions for new stages.

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

DO $$
DECLARE
    partition REGCLASS;
    is_local BOOLEAN;
BEGIN
    FOR partition IN
        SELECT inhrelid::regclass
        FROM pg_inherits
        WHERE inhparent = 'public.ml_data_events'::regclass
    LOOP
        SELECT conislocal
            INTO is_local
            FROM pg_constraint
            WHERE conrelid = partition
              AND conname = 'check_stage';

        IF is_local THEN
            EXECUTE format('ALTER TABLE %s DROP CONSTRAINT check_stage', partition);
            EXECUTE format($sql$
                ALTER TABLE %s
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
                )
            $sql$, partition);
        END IF;
    END LOOP;
END $$;
