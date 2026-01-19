# Streaming Parity Validation Report

## Scope

This report captures lightweight parity and determinism checks for the TFT
streaming pipeline. It complements the existing streaming validation suites
by focusing on formatting parity (streaming bootstrap vs. offline formatter)
and deterministic shard ordering.

## Evidence

- **Parity formatting**: `ml/tests/unit/training/teacher/test_streaming_loader.py::test_materialize_streaming_frame_matches_time_series_formatter`
- **Deterministic shard order**: `ml/tests/unit/training/teacher/test_streaming_loader.py::test_resolve_shard_order_deterministic`
- **Loader parity (existing)**: `ml/tests/unit/training/teacher/test_streaming_loader.py::test_streaming_matches_pytorch_forecasting_batch`

## Notes

- Tests rely on local parquet fixtures and run only when pandas + pyarrow are available.
- The parity test uses `TimeSeriesFormatter` to confirm sequence/time index alignment.
