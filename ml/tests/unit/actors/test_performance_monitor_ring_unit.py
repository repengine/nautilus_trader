"""
Unit tests for PerformanceMonitor backed by ring buffers.

Contracts:
- record_timing increments counters
- last_*_time_ms equals last appended values
- averages and p99 match numpy.percentile over the last N samples
"""

from __future__ import annotations

import numpy as np
import pytest

from ml.actors.signal import PerformanceMonitor


@pytest.mark.unit
def test_performance_monitor_statistics() -> None:
    pm = PerformanceMonitor(reservoir_size=5)

    feature_ns = [1_000_000, 2_000_000, 3_000_000, 4_000_000, 5_000_000, 6_000_000]
    infer_ns = [2_000_000, 1_000_000, 3_000_000, 6_000_000, 5_000_000, 4_000_000]
    total_ns = [f + i for f, i in zip(feature_ns, infer_ns)]

    ms_window = []
    for f, i, t in zip(feature_ns, infer_ns, total_ns):
        pm.record_timing(f, i, t)
        pm.record_signal()
        ms_window.append(t / 1_000_000.0)

    stats = pm.get_current_stats()
    # prediction_count increments per record_timing
    assert stats["prediction_count"] == len(feature_ns)
    assert stats["signal_count"] == len(feature_ns)  # we called record_signal() each time

    # Last values equal last appended (converted to ms)
    assert stats["last_total_time_ms"] == pytest.approx(total_ns[-1] / 1_000_000.0)
    assert stats["last_feature_time_ms"] == pytest.approx(feature_ns[-1] / 1_000_000.0)
    assert stats["last_inference_time_ms"] == pytest.approx(infer_ns[-1] / 1_000_000.0)

    # Expected window is the last reservoir_size items
    tail = np.array(ms_window[-5:], dtype=np.float32)
    assert stats["avg_total_time_ms"] == pytest.approx(float(np.mean(tail)))
    assert stats["p99_total_time_ms"] == pytest.approx(float(np.percentile(tail, 99)))

    # Percentiles API aligns with numpy for all series
    perc = pm.get_latency_percentiles()
    assert set(perc.keys()) == {"feature_computation", "inference", "total"}

