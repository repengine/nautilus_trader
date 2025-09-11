"""
Shared upsert + publish helper for ML stores.

This mixin factors the common pattern of sanitizing/deduplicating rows, performing a
SQLAlchemy INSERT .. ON CONFLICT DO UPDATE upsert, and publishing batch/row summaries
via the message bus utilities.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from sqlalchemy.dialects.postgresql import insert

from ml.stores._batch_utils import publish_batch_and_rows
from ml.stores._batch_utils import sanitize_and_dedup


class SQLUpsertMixin:
    """
    Mixin providing a reusable upsert-and-publish operation.
    """

    # Attributes are provided by consuming classes; intentionally untyped here
    # to avoid attribute type conflicts across mixins.

    def _execute_upsert_and_publish(
        self,
        *,
        values: list[dict[str, Any]],
        ts_event_field: str,
        ts_init_field: str,
        context: str,
        key_fields: tuple[str, str, str],
        table: Any,
        conflict_cols: Iterable[str],
        update_cols: Iterable[str],
        dataset_id: str,
        stage: Any,  # Stage enum value expected at call sites
        instrument_key: str,
        ts_field: str,
        run_id_batch: str,
        run_id_row: str,
        source: str,
        logger: Any,
    ) -> None:
        if not values:
            return

        # Normalize timestamps and remove duplicates within the batch
        values = sanitize_and_dedup(
            values,
            ts_event_field=ts_event_field,
            ts_init_field=ts_init_field,
            context=context,
            key_fields=key_fields,
        )

        # Build upsert statement
        stmt = insert(table)
        excluded = stmt.excluded  # shorthand for mapping update columns
        set_map = {col: getattr(excluded, col) for col in update_cols}
        stmt = stmt.on_conflict_do_update(index_elements=list(conflict_cols), set_=set_map)

        # Execute
        engine = getattr(self, "engine")
        with engine.begin() as conn:
            conn.execute(stmt, values)

        # Publish batch + per-row summaries (best-effort)
        publish_batch_and_rows(
            enable_publishing=bool(getattr(self, "_enable_publishing", False)),
            publisher=getattr(self, "publisher", None),
            publish_mode=getattr(self, "_publish_mode", "batch"),
            topic_scheme=getattr(self, "_topic_scheme", "domain_op"),
            topic_prefix=getattr(self, "_topic_prefix", "events.ml"),
            stage=stage,
            dataset_id=dataset_id,
            instrument_key=instrument_key,
            ts_field=ts_field,
            rows=values,
            run_id_batch=run_id_batch,
            run_id_row=run_id_row,
            source=source,
            logger=logger,
        )
