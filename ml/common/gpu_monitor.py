"""
Utilities for sampling GPU memory usage during long-running operations.

The :class:`GPUMemoryMonitor` class runs an asynchronous sampler which collects
memory readings (in MiB) using a pluggable probe. By default it shells out to
``nvidia-smi`` and records the maximum value observed.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
from dataclasses import dataclass
from typing import Protocol


class GPUMemoryProbe(Protocol):
    """Protocol describing a GPU sampler returning memory usage in MiB."""

    def sample(self) -> float | None:
        """Return the current GPU memory usage in MiB (or ``None`` on failure)."""


@dataclass(slots=True)
class NvidiaSmiProbe:
    """Probe implementation backed by the ``nvidia-smi`` CLI."""

    command: tuple[str, ...] = (
        "nvidia-smi",
        "--query-gpu=memory.used",
        "--format=csv,noheader,nounits",
    )

    def sample(self) -> float | None:
        """Return the maximum memory usage across all GPUs in MiB."""
        try:
            proc = subprocess.run(
                self.command,
                capture_output=True,
                check=False,
                text=True,
                timeout=2.0,
            )
        except Exception:
            return None
        if proc.returncode != 0:
            return None
        if not proc.stdout:
            return None
        readings: list[float] = []
        for line in proc.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                readings.append(float(line))
            except ValueError:
                continue
        if not readings:
            return None
        return max(readings)


class GPUMemoryMonitor:
    """Background sampler that records the peak GPU memory usage in MiB."""

    def __init__(
        self,
        interval_seconds: float,
        *,
        probe: GPUMemoryProbe | None = None,
    ) -> None:
        if interval_seconds <= 0.0:
            msg = "interval_seconds must be positive"
            raise ValueError(msg)
        self._interval = interval_seconds
        self._probe = probe or NvidiaSmiProbe()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._max_mb: float | None = None
        command = getattr(self._probe, "command", None)
        executable = command[0] if isinstance(command, (tuple, list)) and command else None
        self._available = True
        if executable is not None and shutil.which(str(executable)) is None:
            self._available = False

    def start(self) -> None:
        """Start the background sampler if the probe is available."""
        if not self._available or self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="gpu-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the sampler and wait for the background thread to exit."""
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join()
        self._thread = None

    def max_memory_mb(self) -> float | None:
        """Return the maximum memory usage recorded in MiB."""
        with self._lock:
            return self._max_mb

    def _run(self) -> None:
        while not self._stop_event.wait(self._interval):
            reading = self._probe.sample()
            if reading is None:
                continue
            with self._lock:
                if self._max_mb is None or reading > self._max_mb:
                    self._max_mb = reading
        # Capture one final reading at shutdown to avoid missing trailing usage spikes.
        reading = self._probe.sample()
        if reading is not None:
            with self._lock:
                if self._max_mb is None or reading > self._max_mb:
                    self._max_mb = reading


__all__ = ["GPUMemoryMonitor", "GPUMemoryProbe", "NvidiaSmiProbe"]
