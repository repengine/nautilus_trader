from __future__ import annotations

import time
from typing import Protocol

import pytest

from ml.common.gpu_monitor import GPUMemoryMonitor, GPUMemoryProbe


class _StubProbe:
    def __init__(self, readings: list[float | None]) -> None:
        self._readings = readings
        self._index = 0

    def sample(self) -> float | None:
        if self._index >= len(self._readings):
            return self._readings[-1]
        value = self._readings[self._index]
        self._index += 1
        return value


def test_gpu_memory_monitor_records_peak() -> None:
    probe = _StubProbe([10.0, 12.5, None, 9.0, 42.0])
    monitor = GPUMemoryMonitor(0.01, probe=probe)
    monitor.start()
    time.sleep(0.05)
    monitor.stop()
    assert monitor.max_memory_mb() == pytest.approx(42.0)


def test_gpu_memory_monitor_requires_positive_interval() -> None:
    with pytest.raises(ValueError):
        GPUMemoryMonitor(0.0)
