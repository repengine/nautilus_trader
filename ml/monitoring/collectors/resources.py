"""
Resource utilization metrics collector for ML monitoring.

This module provides comprehensive tracking of system resource usage during ML
operations including memory, CPU, and GPU utilization.

"""

from __future__ import annotations

import os
import threading
from typing import Any

from ml._imports import HAS_PROMETHEUS
from ml.monitoring._config import MonitoringConfig
from ml.monitoring.collectors.base import BaseMetricsCollector


class ResourceUtilizationCollector(BaseMetricsCollector):
    """
    Collector for ML system resource utilization metrics.

    This collector tracks memory usage, CPU utilization, disk usage,
    and GPU metrics (if available) for ML operations and model inference.

    Key Metrics
    -----------
    - Memory usage by models and feature stores
    - CPU utilization during ML operations
    - GPU utilization and memory (if available)
    - Disk usage and I/O for data loading

    Parameters
    ----------
    config : MonitoringConfig
        Configuration for metrics collection.

    """

    def __init__(self, config: MonitoringConfig) -> None:
        """
        Initialize the resource utilization collector.

        Parameters
        ----------
        config : MonitoringConfig
            Configuration for metrics collection.

        """
        super().__init__(config)

        # Background monitoring thread
        self._monitoring_thread: threading.Thread | None = None
        self._monitoring_stop_event = threading.Event()
        self._monitoring_interval = 30.0  # seconds

    def _initialize_metrics(self) -> None:
        """
        Initialize Prometheus metrics for resource monitoring.
        """
        if not HAS_PROMETHEUS:
            return

        from ml.common.metrics_bootstrap import get_counter
        from ml.common.metrics_bootstrap import get_gauge

        prefix = self._config.metrics_prefix

        # Memory metrics
        self._model_memory_usage_bytes = get_gauge(
            f"{prefix}_model_memory_usage_bytes",
            "Memory usage by ML models",
            ["model", "memory_type"],
        )
        self._register_metric("model_memory_usage_bytes", self._model_memory_usage_bytes)

        self._feature_store_size_bytes = get_gauge(
            f"{prefix}_feature_store_size_bytes",
            "Size of feature store in bytes",
            ["storage_type"],
        )
        self._register_metric("feature_store_size_bytes", self._feature_store_size_bytes)

        self._python_memory_usage_bytes = get_gauge(
            f"{prefix}_python_memory_usage_bytes",
            "Python process memory usage",
            ["memory_type"],
        )
        self._register_metric("python_memory_usage_bytes", self._python_memory_usage_bytes)

        # CPU metrics
        self._cpu_usage_percent = get_gauge(
            f"{prefix}_cpu_usage_percent",
            "CPU usage percentage",
            ["core"],
        )
        self._register_metric("cpu_usage_percent", self._cpu_usage_percent)

        self._ml_cpu_time_seconds = get_counter(
            f"{prefix}_ml_cpu_time_seconds_total",
            "Total CPU time spent on ML operations",
            ["operation_type"],
        )
        self._register_metric("ml_cpu_time_seconds", self._ml_cpu_time_seconds)

        # GPU metrics (optional)
        self._gpu_utilization_percent = get_gauge(
            f"{prefix}_gpu_utilization_percent",
            "GPU utilization percentage",
            ["device", "metric"],
        )
        self._register_metric("gpu_utilization_percent", self._gpu_utilization_percent)

        self._gpu_memory_usage_bytes = get_gauge(
            f"{prefix}_gpu_memory_usage_bytes",
            "GPU memory usage in bytes",
            ["device", "memory_type"],
        )
        self._register_metric("gpu_memory_usage_bytes", self._gpu_memory_usage_bytes)

        # Disk and I/O metrics
        self._disk_usage_bytes = get_gauge(
            f"{prefix}_disk_usage_bytes",
            "Disk usage for ML data and models",
            ["path", "usage_type"],
        )
        self._register_metric("disk_usage_bytes", self._disk_usage_bytes)

        self._data_io_bytes_total = get_counter(
            f"{prefix}_data_io_bytes_total",
            "Total bytes read/written for ML data",
            ["operation", "data_type"],
        )
        self._register_metric("data_io_bytes_total", self._data_io_bytes_total)

        # Batch size and throughput metrics
        self._inference_batch_size = get_gauge(
            f"{prefix}_inference_batch_size",
            "Current inference batch size",
            ["model"],
        )
        self._register_metric("inference_batch_size", self._inference_batch_size)

        self._training_data_rows_processed_total = get_counter(
            f"{prefix}_training_data_rows_processed_total",
            "Total training data rows processed",
            ["dataset"],
        )
        self._register_metric(
            "training_data_rows_processed_total",
            self._training_data_rows_processed_total,
        )

    def record_model_memory_usage(
        self,
        model: str,
        memory_bytes: int,
        memory_type: str = "resident",
    ) -> None:
        """
        Record memory usage by a specific model.

        Parameters
        ----------
        model : str
            Model identifier.
        memory_bytes : int
            Memory usage in bytes.
        memory_type : str, default "resident"
            Type of memory (resident, virtual, gpu).

        """

        def _record() -> None:
            if self._model_memory_usage_bytes is not None:
                self._model_memory_usage_bytes.labels(
                    model=model,
                    memory_type=memory_type,
                ).set(max(0, memory_bytes))

        self._safe_record("model_memory", _record)

    def record_feature_store_size(
        self,
        size_bytes: int,
        storage_type: str = "memory",
    ) -> None:
        """
        Record feature store size.

        Parameters
        ----------
        size_bytes : int
            Size in bytes.
        storage_type : str, default "memory"
            Storage type (memory, disk, redis).

        """

        def _record() -> None:
            if self._feature_store_size_bytes is not None:
                self._feature_store_size_bytes.labels(
                    storage_type=storage_type,
                ).set(max(0, size_bytes))

        self._safe_record("feature_store_size", _record)

    def record_cpu_usage(
        self,
        usage_percent: float,
        core: str = "average",
    ) -> None:
        """
        Record CPU usage.

        Parameters
        ----------
        usage_percent : float
            CPU usage percentage (0-100).
        core : str, default "average"
            CPU core identifier or "average".

        """

        def _record() -> None:
            if self._cpu_usage_percent is not None:
                self._cpu_usage_percent.labels(
                    core=core,
                ).set(max(0.0, min(100.0, usage_percent)))

        self._safe_record("cpu_usage", _record)

    def record_ml_cpu_time(
        self,
        cpu_time_seconds: float,
        operation_type: str,
    ) -> None:
        """
        Record CPU time spent on ML operations.

        Parameters
        ----------
        cpu_time_seconds : float
            CPU time in seconds.
        operation_type : str
            Type of ML operation (training, inference, feature_computation).

        """

        def _record() -> None:
            if self._ml_cpu_time_seconds is not None:
                self._ml_cpu_time_seconds.labels(
                    operation_type=operation_type,
                ).inc(max(0.0, cpu_time_seconds))

        self._safe_record("ml_cpu_time", _record)

    def record_gpu_metrics(
        self,
        device: str,
        compute_utilization: float | None = None,
        memory_utilization: float | None = None,
        memory_used_bytes: int | None = None,
        memory_total_bytes: int | None = None,
    ) -> None:
        """
        Record GPU metrics (if available).

        Parameters
        ----------
        device : str
            GPU device identifier (e.g., "cuda:0").
        compute_utilization : float, optional
            Compute utilization percentage (0-100).
        memory_utilization : float, optional
            Memory utilization percentage (0-100).
        memory_used_bytes : int, optional
            Used GPU memory in bytes.
        memory_total_bytes : int, optional
            Total GPU memory in bytes.

        """

        def _record() -> None:
            if compute_utilization is not None and self._gpu_utilization_percent is not None:
                self._gpu_utilization_percent.labels(
                    device=device,
                    metric="compute",
                ).set(max(0.0, min(100.0, compute_utilization)))

            if memory_utilization is not None and self._gpu_utilization_percent is not None:
                self._gpu_utilization_percent.labels(
                    device=device,
                    metric="memory",
                ).set(max(0.0, min(100.0, memory_utilization)))

            if memory_used_bytes is not None and self._gpu_memory_usage_bytes is not None:
                self._gpu_memory_usage_bytes.labels(
                    device=device,
                    memory_type="used",
                ).set(max(0, memory_used_bytes))

            if memory_total_bytes is not None and self._gpu_memory_usage_bytes is not None:
                self._gpu_memory_usage_bytes.labels(
                    device=device,
                    memory_type="total",
                ).set(max(0, memory_total_bytes))

        self._safe_record("gpu_metrics", _record)

    def record_disk_usage(
        self,
        path: str,
        usage_bytes: int,
        usage_type: str = "data",
    ) -> None:
        """
        Record disk usage for ML operations.

        Parameters
        ----------
        path : str
            Disk path or identifier.
        usage_bytes : int
            Disk usage in bytes.
        usage_type : str, default "data"
            Type of usage (data, models, cache).

        """

        def _record() -> None:
            if self._disk_usage_bytes is not None:
                self._disk_usage_bytes.labels(
                    path=path,
                    usage_type=usage_type,
                ).set(max(0, usage_bytes))

        self._safe_record("disk_usage", _record)

    def record_data_io(
        self,
        bytes_transferred: int,
        operation: str,
        data_type: str = "bars",
    ) -> None:
        """
        Record data I/O operations.

        Parameters
        ----------
        bytes_transferred : int
            Number of bytes transferred.
        operation : str
            I/O operation (read, write).
        data_type : str, default "bars"
            Type of data transferred.

        """

        def _record() -> None:
            if self._data_io_bytes_total is not None:
                self._data_io_bytes_total.labels(
                    operation=operation,
                    data_type=data_type,
                ).inc(max(0, bytes_transferred))

        self._safe_record("data_io", _record)

    def record_inference_batch_size(
        self,
        model: str,
        batch_size: int,
    ) -> None:
        """
        Record inference batch size.

        Parameters
        ----------
        model : str
            Model identifier.
        batch_size : int
            Current batch size.

        """

        def _record() -> None:
            if self._inference_batch_size is not None:
                self._inference_batch_size.labels(
                    model=model,
                ).set(max(0, batch_size))

        self._safe_record("inference_batch_size", _record)

    def record_training_data_processed(
        self,
        rows: int,
        dataset: str = "train",
    ) -> None:
        """
        Record training data processing.

        Parameters
        ----------
        rows : int
            Number of rows processed.
        dataset : str, default "train"
            Dataset type (train, validation, test).

        """

        def _record() -> None:
            if self._training_data_rows_processed_total is not None:
                self._training_data_rows_processed_total.labels(
                    dataset=dataset,
                ).inc(max(0, rows))

        self._safe_record("training_data_processed", _record)

    def start_monitoring(self) -> None:
        """
        Start background monitoring of system resources.

        This starts a background thread that periodically collects system resource
        metrics.

        """
        if not self._enabled or self._monitoring_thread is not None:
            return

        self._monitoring_stop_event.clear()
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="ResourceMonitor",
        )
        self._monitoring_thread.start()

    def stop_monitoring(self) -> None:
        """
        Stop background monitoring of system resources.
        """
        if self._monitoring_thread is None:
            return

        self._monitoring_stop_event.set()
        self._monitoring_thread.join(timeout=5.0)
        self._monitoring_thread = None

    def _monitoring_loop(self) -> None:
        """
        Background monitoring loop for system resources.
        """
        while not self._monitoring_stop_event.wait(self._monitoring_interval):
            try:
                self._collect_system_metrics()
            except Exception:
                # Graceful degradation - don't fail if monitoring fails
                pass

    def _collect_system_metrics(self) -> None:
        """
        Collect system-level resource metrics.
        """
        try:
            # Python memory usage
            import psutil

            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()

            if self._python_memory_usage_bytes is not None:
                self._python_memory_usage_bytes.labels(
                    memory_type="rss",
                ).set(memory_info.rss)

                self._python_memory_usage_bytes.labels(
                    memory_type="vms",
                ).set(memory_info.vms)

            # CPU usage
            cpu_percent = process.cpu_percent()
            self.record_cpu_usage(cpu_percent, "process")

            # System CPU usage
            system_cpu = psutil.cpu_percent()
            self.record_cpu_usage(system_cpu, "system")

        except ImportError:
            # psutil not available - skip system monitoring
            pass
        except Exception:
            # Graceful degradation
            pass

        # Try to collect GPU metrics if available
        self._collect_gpu_metrics()

    def _collect_gpu_metrics(self) -> None:
        """
        Collect GPU metrics if CUDA/GPU libraries are available.
        """
        try:
            # Try to import GPU monitoring libraries
            import pynvml

            pynvml.nvmlInit()

            device_count = pynvml.nvmlDeviceGetCount()
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)

                # GPU utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)

                # Memory info
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

                device_name = f"cuda:{i}"
                self.record_gpu_metrics(
                    device=device_name,
                    compute_utilization=util.gpu,
                    memory_utilization=util.memory,
                    memory_used_bytes=mem_info.used,
                    memory_total_bytes=mem_info.total,
                )

        except ImportError:
            # GPU libraries not available
            pass
        except Exception:
            # Graceful degradation
            pass

    def get_resource_summary(self) -> dict[str, Any]:
        """
        Get comprehensive resource utilization summary.

        Returns
        -------
        Dict[str, Any]
            Resource utilization summary.

        """
        summary = {
            "python_memory_rss": self.get_metric_value(
                "python_memory_usage_bytes",
                {"memory_type": "rss"},
            ),
            "python_memory_vms": self.get_metric_value(
                "python_memory_usage_bytes",
                {"memory_type": "vms"},
            ),
            "cpu_usage_process": self.get_metric_value(
                "cpu_usage_percent",
                {"core": "process"},
            ),
            "cpu_usage_system": self.get_metric_value(
                "cpu_usage_percent",
                {"core": "system"},
            ),
            "feature_store_memory": self.get_metric_value(
                "feature_store_size_bytes",
                {"storage_type": "memory"},
            ),
        }

        return {k: v for k, v in summary.items() if v is not None}
