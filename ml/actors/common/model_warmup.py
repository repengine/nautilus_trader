"""
Model Warm-Up Component.

This module implements ONNX model loading, warm-up, feature parity checking,
and hot-reload scheduling for MLSignalActor decomposition.

The component provides:
- ONNX-optimized model loading with session options
- Model warm-up with dummy predictions (timing statistics)
- Feature parity smoke checks (online vs offline drift detection)
- Model hot-reload scheduling (file mtime monitoring)
- Deque-based buffer management for parity checks

Cold Path Component:
- All methods are cold path (no hot path performance constraints)
- Allocations allowed throughout
- Focus on correctness, error handling, observability

ONNX Runtime Integration:
- Uses ml._imports for dependency checking (HAS_ONNX, ort)
- Graceful degradation if onnxruntime not installed
- Clear error messages via check_ml_dependencies()

Metrics Emitted:
- ml_feature_parity_checks_total (counter)
- ml_feature_parity_drift (gauge)

Architecture Patterns (CLAUDE.md):
- Pattern 2: Protocol-First Interface Design (property accessors)
- Centralized ML Imports (ml._imports for ONNX)
- Centralized Metrics Bootstrap (get_counter, get_gauge)
- Config-Driven Development (all parameters configurable)

Critical Safeguards (CRITICAL_SAFEGUARDS.md):
- Category 2: No Stubs/TODOs (full implementations only)
- Category 3: Hot/Cold Path (all cold path, allocations OK)
- Category 6: Circular Imports (TYPE_CHECKING guard)
- Exception logging with exc_info=True

"""

from __future__ import annotations

import os
import time
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from ml._imports import HAS_ONNX
from ml._imports import check_ml_dependencies
from ml._imports import ort
from ml.common.metrics_bootstrap import get_counter
from ml.common.metrics_bootstrap import get_gauge
from ml.common.model_load_policy import apply_direct_model_load_policy


if TYPE_CHECKING:
    from collections.abc import Callable
    from logging import Logger

    from ml.config.runtime import OnnxRuntimeConfig


class ModelWarmUpComponent:
    """
    Component for ONNX model loading, warm-up, parity checking, and hot-reload.
    """

    def __init__(
        self,
        model_path: str,
        n_features: int,
        onnx_runtime_config: OnnxRuntimeConfig | None = None,
        warm_up_iterations: int = 10,
        enable_parity_check: bool = False,
        parity_tolerance: float = 1e-6,
        parity_window: int = 200,
        enable_hot_reload: bool = False,
        hot_reload_interval: float = 300.0,
        actor_id: str | None = None,
        log: Logger | None = None,
    ) -> None:
        """
        Initialize model warm-up component.

        Parameters
        ----------
        model_path : str
            Path to ONNX model file.
        n_features : int
            Number of features expected by model.
        onnx_runtime_config : OnnxRuntimeConfig | None, default=None
            ONNX runtime configuration for session options.
        warm_up_iterations : int, default=10
            Number of warm-up predictions to run.
        enable_parity_check : bool, default=False
            Whether to enable feature parity smoke checks.
        parity_tolerance : float, default=1e-6
            Maximum allowed drift between online and offline features.
        parity_window : int, default=200
            Number of recent bars/features to buffer for parity checks.
        enable_hot_reload : bool, default=False
            Whether to enable model hot-reloading.
        hot_reload_interval : float, default=300.0
            Minimum seconds between hot-reload checks.
        actor_id : str | None, default=None
            Actor identifier for logging and metrics (optional).
        log : Logger | None, default=None
            Logger instance (optional).

        Raises
        ------
        ValueError
            If n_features <= 0 or warm_up_iterations < 0.
        FileNotFoundError
            If model_path doesn't exist.
        ImportError
            If onnxruntime not installed.

        """
        # Validate inputs
        if n_features <= 0:
            raise ValueError(f"n_features ({n_features}) must be > 0")
        if warm_up_iterations < 0:
            raise ValueError(f"warm_up_iterations ({warm_up_iterations}) must be >= 0")
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Store config
        self._model_path = model_path
        self._n_features = n_features
        self._onnx_runtime_config = onnx_runtime_config
        self._warm_up_iterations = warm_up_iterations
        self._enable_parity_check = enable_parity_check
        self._parity_tolerance = parity_tolerance
        self._parity_window = parity_window
        self._enable_hot_reload = enable_hot_reload
        self._hot_reload_interval = hot_reload_interval
        self._actor_id = actor_id or "unknown"
        self._log = log

        # State tracking
        self._parity_checked = False
        self._last_check_time = 0.0
        self._model_mtime: float | None = None

        # Parity check buffers (only if enabled)
        self._recent_bars: deque[Any] = deque(maxlen=parity_window)
        self._recent_features: deque[npt.NDArray[np.float32]] = deque(maxlen=parity_window)

        # Initialize metrics
        self._parity_checks_counter = get_counter(
            "ml_feature_parity_checks_total",
            "Total feature parity checks performed",
            ["actor_id"],
        )
        self._parity_drift_gauge = get_gauge(
            "ml_feature_parity_drift",
            "Feature parity drift (max absolute difference)",
            ["actor_id"],
        )

    def load_model(self) -> tuple[Any, dict[str, Any]]:
        """
        Load ONNX model with optimizations (cold path).

        Returns
        -------
        tuple[Any, dict[str, Any]]
            (model, metadata) where:
            - model: ort.InferenceSession instance
            - metadata: Dict with input_names, output_names, model_path

        Raises
        ------
        ImportError
            If onnxruntime not available.
        FileNotFoundError
            If model file doesn't exist.

        """
        # Check ONNX availability
        if not HAS_ONNX:
            check_ml_dependencies(["onnxruntime"])

        # Verify model file still exists
        if not Path(self._model_path).exists():
            raise FileNotFoundError(f"Model file not found: {self._model_path}")

        # Get session options from config
        from ml.config.runtime import OnnxRuntimeConfig as _OnnxRuntimeConfig
        from ml.config.runtime import to_session_options

        rt = self._onnx_runtime_config or _OnnxRuntimeConfig()
        session_options, providers = to_session_options(rt)
        policy_result = apply_direct_model_load_policy(
            model_path=Path(self._model_path),
            context="model_warmup_component_load",
        )

        # Load model with ONNX Runtime
        if not HAS_ONNX:
            check_ml_dependencies(["onnxruntime"])
        if ort is None:
            check_ml_dependencies(["onnxruntime"])
        from ml.common.security import verify_artifact_integrity

        verify_artifact_integrity(
            Path(self._model_path),
            policy_result.expected_digest,
            strict=policy_result.strict_integrity,
        )
        assert ort is not None
        model = ort.InferenceSession(
            self._model_path,
            sess_options=session_options,
            providers=providers,
        )

        # Extract metadata
        input_names = [inp.name for inp in model.get_inputs()]
        output_names = [out.name for out in model.get_outputs()]

        metadata = dict(policy_result.metadata)
        metadata.update(
            {
                "artifact_sha256_digest": policy_result.expected_digest,
                "input_names": input_names,
                "output_names": output_names,
                "model_path": self._model_path,
            },
        )

        # Log success
        if self._log:
            self._log.info(f"Loaded optimized ONNX model: {self._model_path}")

        # Store initial mtime for hot-reload tracking
        self._model_mtime = os.path.getmtime(self._model_path)

        return model, metadata

    def warm_up_model(
        self,
        model: Any,
        predict_fn: Callable[[npt.NDArray[np.float32]], tuple[float, float]],
    ) -> None:
        """
        Warm up model with dummy predictions (cold path).

        Runs warm_up_iterations predictions with random features,
        logs average and P99 timing statistics.

        Parameters
        ----------
        model : Any
            ONNX InferenceSession to warm up.
        predict_fn : callable
            Function to call for predictions (signature: features -> (prediction, confidence)).

        """
        if self._warm_up_iterations <= 0:
            return

        rng = np.random.default_rng()
        dummy_features = rng.standard_normal(self._n_features).astype(np.float32)
        warm_up_times: list[float] = []

        for i in range(self._warm_up_iterations):
            start = time.perf_counter_ns()
            try:
                predict_fn(dummy_features)
            except Exception as e:
                if self._log:
                    self._log.debug(f"Warm-up iteration {i} failed: {e}", exc_info=True)
            warm_up_times.append((time.perf_counter_ns() - start) / 1_000_000)

        if warm_up_times and self._log:
            avg_ms = np.mean(warm_up_times)
            p99_ms = np.percentile(warm_up_times, 99)
            self._log.info(
                f"Model warm-up completed: avg={avg_ms:.3f}ms, P99={p99_ms:.3f}ms",
            )

    def check_parity(
        self,
        recent_bars: deque[Any],
        recent_features: deque[npt.NDArray[np.float32]],
        compute_features_fn: Callable[[Any], npt.NDArray[np.float32] | None],
    ) -> float:
        """
        Check feature parity between online and offline computation (cold path).

        Recomputes features offline from recent bars, compares with online features,
        calculates drift (max absolute difference), emits metrics, and warns if
        drift exceeds tolerance.

        Parameters
        ----------
        recent_bars : deque
            Deque of recent Bar objects.
        recent_features : deque
            Deque of recent feature vectors (numpy arrays).
        compute_features_fn : callable
            Function to compute features (signature: bar -> features).

        Returns
        -------
        float
            Maximum absolute drift between online and offline features.
            Returns 0.0 if insufficient data.

        """
        if not self._enable_parity_check:
            return 0.0

        try:
            # Recompute features offline
            offline_vectors: list[npt.NDArray[np.float32]] = []
            for bar in recent_bars:
                vec = compute_features_fn(bar)
                if vec is not None:
                    offline_vectors.append(vec.copy())

            # Check if we have enough data
            n_online = len(recent_features)
            n_offline = len(offline_vectors)
            n = min(n_online, n_offline)
            if n == 0:
                return 0.0

            # Stack arrays for comparison
            online = np.stack(list(recent_features)[-n:])
            offline = np.stack(offline_vectors[-n:])

            # Compute drift (max absolute difference)
            drift = float(np.max(np.abs(online - offline)))

            # Emit metrics
            self._parity_checks_counter.labels(actor_id=self._actor_id).inc()
            self._parity_drift_gauge.labels(actor_id=self._actor_id).set(drift)

            # Warn if drift exceeds tolerance
            if drift > self._parity_tolerance:
                if self._log:
                    self._log.warning(
                        f"Feature parity drift {drift:.6f} exceeds tolerance {self._parity_tolerance}",
                    )

            # Mark as checked
            self._parity_checked = True

            return drift

        except Exception as e:
            if self._log:
                self._log.error(f"Parity smoke check failed: {e}", exc_info=True)
            return 0.0

    def should_hot_reload(self) -> bool:
        """
        Check if model hot-reload should be performed (cold path).

        Returns True if:
        - enable_hot_reload is True
        - Sufficient time has passed since last check (>= hot_reload_interval)

        Updates _last_check_time when returning True.

        Returns
        -------
        bool
            Whether to proceed with hot-reload check.

        """
        if not self._enable_hot_reload:
            return False

        # Check if enough time has passed since last check
        current_time = time.time()
        if current_time - self._last_check_time < self._hot_reload_interval:
            return False

        self._last_check_time = current_time
        return True

    def execute_hot_reload(
        self,
        model_path: str,
        load_model_fn: Callable[[str], tuple[Any, dict[str, Any]]],
    ) -> tuple[Any, dict[str, Any]] | None:
        """
        Execute hot-reload if model file has been modified (cold path).

        Checks if model file exists and if its mtime has changed since
        last load. If modified, reloads model and updates mtime.

        Parameters
        ----------
        model_path : str
            Path to model file.
        load_model_fn : callable
            Function to load model (signature: path -> (model, metadata)).

        Returns
        -------
        tuple[Any, dict[str, Any]] | None
            (model, metadata) if reloaded, None if no reload needed or error.

        """
        try:
            # Check if file exists
            if not Path(model_path).exists():
                return None

            # Get current mtime
            current_mtime = os.path.getmtime(model_path)

            # Check if modified
            if self._model_mtime is not None and current_mtime <= self._model_mtime:
                return None  # Not modified

            # Log reload
            if self._log:
                self._log.info(f"Hot reloading model from {model_path}")

            # Reload model
            model, metadata = load_model_fn(model_path)
            self._model_mtime = current_mtime

            return model, metadata

        except Exception as e:
            if self._log:
                self._log.error(f"Failed to hot reload model: {e}", exc_info=True)
            return None

    def is_drift_policy_ready(
        self,
        *,
        baseline_samples: int,
        observed_samples: int,
        min_baseline_samples: int,
        min_observed_samples: int,
    ) -> bool:
        """
        Return whether runtime drift actions are eligible for enforcement.

        Parameters
        ----------
        baseline_samples : int
            Current number of baseline samples collected by drift monitoring.
        observed_samples : int
            Current number of observed inference samples.
        min_baseline_samples : int
            Required minimum baseline sample count.
        min_observed_samples : int
            Required minimum observed sample count.

        Returns
        -------
        bool
            ``True`` when both baseline and observed sample windows are satisfied.

        """
        baseline_target = max(1, int(min_baseline_samples))
        observed_target = max(baseline_target, int(min_observed_samples))
        return int(baseline_samples) >= baseline_target and int(observed_samples) >= observed_target

    # Property accessors
    @property
    def model_path(self) -> str:
        """
        Get model file path.
        """
        return self._model_path

    @property
    def n_features(self) -> int:
        """
        Get expected number of features.
        """
        return self._n_features

    @property
    def parity_checked(self) -> bool:
        """
        Get whether parity check has been performed.
        """
        return self._parity_checked

    @property
    def last_check_time(self) -> float:
        """
        Get last hot-reload check timestamp.
        """
        return self._last_check_time


__all__ = ["ModelWarmUpComponent"]
