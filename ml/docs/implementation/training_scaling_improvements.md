# Training Scaling Improvements (Concrete Plan)

## Goals

- Eliminate DRAM OOM during dataset build and training without shrinking coverage.
- Maximize signal per byte from the existing market data catalog.
- Preserve reproducibility and parity between training and inference.

## Current Constraints (Observed)

- Chunked dataset build still materializes full DataFrames and converts to dense NumPy,
  which spikes memory on large datasets.
- Streaming training still aggregates large in-memory blocks (validation joins),
  though validation joins now run in bounded chunks and ensemble blending streams logits.
- High dataloader worker counts can degrade stability and increase memory pressure.

## Code Hotspots (Current)

- Dataset build + chunked writer: `ml/data/build.py:_build_dataset_chunked`
  - `df_chunk.select(...).to_numpy()` materializes full float matrices.
  - Per-chunk `.parquet` files are re-read as full DataFrames before final write.
- Dataset build per-symbol processing: `ml/data/tft_dataset_builder_facade.py:_build_training_dataset_direct`
  - Per-symbol DataFrames are accumulated then concatenated.
- Streaming loader: `ml/training/teacher/streaming_loader.py:TFTStreamingDataset._iter_shard_batches`
  - `scanner.to_table()` materializes full shard tables and then converts to numpy.
- Streaming worker: `ml/training/event_driven/worker.py`
  - Validation returns join collects full column arrays; ensemble blending loads
    full logits arrays into memory (not hot-path, but large artifacts).
  - Status: ensemble blending now accumulates logits incrementally; validation joins chunked.
- Dataset planner: `ml/training/event_driven/dataset_service.py`
  - Applies caps and shard budgets (best place to tighten limits).

## Near-Term Changes (Minimal Refactor)

1. Streaming dataset writer + bounded feature extraction
   - Replace `df_chunk.select(...).to_numpy()` with Arrow batch iteration and feed
     `_StreamingFeatureWriter` in bounded slices.
   - Write final dataset parquet via `pyarrow.parquet.ParquetWriter` as batches stream
     in, avoiding chunk re-loads.
   - Keep `feature_names` inference and `non_null_counts` by aggregating per batch.
   - Status: completed (chunked writer now streams batches + feature extraction).
2. Reduce chunk materialization
   - Avoid `.chunks` re-read when possible; if retained, store as Arrow IPC/Parquet
     with row-group size tuned to memory and read via `pyarrow.dataset` streaming.
   - Status: completed for chunked output merge (batch streaming from parquet).
3. Control row growth in streaming planner/worker
   - Enforce `max_total_rows`, `max_total_sequences`, `max_shards` at planning time.
   - Add a memory policy guard in worker to reduce `num_workers` when recent RSS is high.
   - Status: worker RSS guard completed; planning caps already applied via service caps.
4. Join late, store early
   - Ensure macro/events/micro/L2/earnings mirrors are persisted ahead of training;
     dataset build becomes "select + join + write" against durable stores.
   - Status: pending.

## Mid-Term Changes (Architecture)

1. Partitioned dataset format
   - Write training datasets as partitioned Parquet (instrument_id / time buckets).
   - Train directly from partitioned parquet via streaming readers and manifests.
2. Feature cache tiering
   - Separate "slow external" feature datasets (macro, filings, events) from
     "fast market" data; always hydrate those caches ahead of training.
3. Streaming trainer memory policy
   - Batch-level feature assembly only; no global materialization of tensors.
   - Keep per-worker memory caps; reduce dataloader workers when RSS grows.

## Quant Data-Utilization Strategy

- Maximize cross-sectional breadth and regime coverage first.
- Use multiple horizons/targets to extract more signal per row.
- Favor rolling time splits over single static splits to reduce leakage and
  increase sample efficiency.

## Metrics to Track

- Dataset build peak RSS and elapsed time per 100k rows.
- Training worker RSS and GPU peak usage per cohort.
- Feature cache hit rates (parquet/SQL vs API fetches).

## Risks / Open Questions

- Arrow/Parquet streaming must preserve row order and metadata alignment.
- Join order changes must keep parity with inference feature generation.
- Some feature families may still lack durable parquet mirrors.

## Next Steps (Future Research)
### Phase 0: Baseline + Guardrails

- Add RSS gauges around dataset build chunk loop (per-chunk peak).
- Extend streaming loader RSS gauge to include "scan" and "batch assembly" stages.
  - Status: completed (dataset build RSS + streaming scan/assembly gauges).

### Phase 1: Minimal Refactor (Targeted)

1. Dataset build (writer + features)
   - File: `ml/data/build.py`
   - Change `_build_dataset_chunked` to:
     - Use `table.to_batches(max_chunksize=...)` for feature extraction.
     - Stream batches into `_StreamingFeatureWriter` (float32 blocks).
     - Write batches to `ParquetWriter` directly without re-reading `.chunks`.
   - Status: completed.
2. Streaming loader (bounded shard iteration)
   - File: `ml/training/teacher/streaming_loader.py`
   - Change `_iter_shard_batches` to iterate `scanner.to_batches()` and build
     sequences from rolling buffers; avoid `scanner.to_table()`.
   - Status: completed.
3. Worker caps
   - Files: `ml/training/event_driven/worker.py`, `ml/training/event_driven/dataset_service.py`
   - Wire memory policy caps into `TFTStreamingConfig` and surface in plan caps.
   - Status: worker RSS guard completed; planner caps already surfaced.
4. Streaming worker ensemble blending
   - File: `ml/training/event_driven/worker.py`
   - Accumulate weighted logits incrementally to avoid storing all member arrays.
   - Status: completed.

### Phase 2: Mid-Term Architecture

- Partition parquet by instrument/time and add a manifest/index for shard metadata.
- Update `collect_streaming_metadata` to use partition filters instead of full scans.
- Standardize feature cache mirrors for macro/events/micro/L2/earnings.

## Testing / Validation Matrix
Types + Lint:

- `poetry run mypy ml --strict`
- `poetry run ruff check ml`

Focused Tests:

- Dataset build: `poetry run pytest ml/tests/unit/data/test_dataset_build_macro.py`
- Dataset validation: `poetry run pytest ml/tests/unit/data/test_dataset_validation.py`
- Builder integration: `poetry run pytest ml/tests/integration/data/test_tft_builder_with_events.py`
- Streaming event-driven: `poetry run pytest ml/tests/integration/training/event_driven/test_plan_to_result.py`
- Streaming unit tests: `poetry run pytest ml/tests/unit/training/event_driven/`
- Dataset E2E (if dataset build changed): `pytest -q ml/tests/e2e/test_tft_dataset_builder_e2e.py`

Validation Targets (as applicable):

- `make validate-metrics`
- `make validate-events`
