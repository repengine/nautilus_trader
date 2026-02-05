"""
Multi-instrument ML signal actor with batched inference.

Scaffold implementation focusing on:
- Universe management (add/remove instruments)
- Pre-allocated batch feature tensor
- Batched inference scheduling off the hot path

Hot path guarantees:
- `on_bar` performs O(1) operations: compute features, copy into pre-allocated
  batch buffer, and returns. Flushing/inference is gated to size thresholds to
  avoid frequent calls; time-based flush can be added on a timer.

This scaffold reuses BaseMLInferenceActor for model/feature plumbing and defers
heavy work to the cold path. It emits basic batch metrics and integrates with
existing structured logging.

"""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import nullcontext
from typing import Any, Final

import numpy as np
import numpy.typing as npt
from nautilus_trader.model.data import Bar

from ml.actors.signal import MLSignalActor
from ml.common import normalize_prediction_batch
from ml.common import resolve_output_is_logits
from ml.common import resolve_positive_class_index
from ml.config.actors import MLSignalActorConfig as _BaseCfg


__all__ = [
    "MultiInstrumentSignalActor",
    "MultiInstrumentSignalActorConfig",
]


class MultiInstrumentSignalActorConfig(_BaseCfg, kw_only=True, frozen=True):
    """
    Configuration for multi-instrument batched inference.

    Attributes
    ----------
    max_batch_size:
        Maximum number of instruments per inference batch.
    feature_dim:
        Expected feature vector length. Used to pre-allocate the batch tensor.
    initial_universe:
        Optional list of instrument IDs to seed the universe.

    """

    max_batch_size: int = 128
    feature_dim: int = 64
    initial_universe: list[str] | None = None
    # Maximum latency (ms) to hold a non-empty batch before forcing a flush.
    # 0 disables time-based flushing (capacity-driven only).
    flush_max_latency_ms: int = 0


class _UniverseManager:
    """
    Tracks the active instrument universe for the actor.
    """

    def __init__(self, instruments: Iterable[str] | None = None) -> None:
        self._set: set[str] = set(instruments or [])

    def add(self, instrument_id: str) -> None:
        self._set.add(instrument_id)

    def remove(self, instrument_id: str) -> None:
        self._set.discard(instrument_id)

    def contains(self, instrument_id: str) -> bool:
        return instrument_id in self._set

    def size(self) -> int:
        return len(self._set)

    def snapshot(self) -> tuple[str, ...]:
        return tuple(self._set)

    def set_all(self, instruments: Iterable[str]) -> None:
        self._set = set(instruments)

    def clear(self) -> None:
        self._set.clear()


class MultiInstrumentSignalActor(MLSignalActor):
    """
    Batched ML inference actor for multiple instruments.

    Hot path collects feature vectors for instruments into a pre-allocated tensor, and
    performs batched inference when the batch reaches capacity.

    """

    def __init__(self, config: MultiInstrumentSignalActorConfig) -> None:
        super().__init__(config)
        self._cfg: Final[MultiInstrumentSignalActorConfig] = config

        # Universe
        self._universe = _UniverseManager(config.initial_universe)

        # Pre-allocated batch storage (hot path safe)
        self._max_batch: Final[int] = int(config.max_batch_size)
        self._feature_dim: Final[int] = int(config.feature_dim)
        self._batch_features: npt.NDArray[np.float32] = np.zeros(
            (self._max_batch, self._feature_dim),
            dtype=np.float32,
        )
        # Non-numpy structures for metadata (kept minimal and reused)
        self._batch_instruments: list[str] = []
        self._batch_bars: list[Bar] = []
        self._batch_size: int = 0
        self._batch_started_ns: int = 0
        # Prepared predictions queue (prediction, confidence) consumed by _predict
        self._prepared_preds: list[tuple[float, float]] = []

        # Metrics (module-local via MetricsManager)
        try:
            from ml.common.metrics_manager import MetricsManager as _MM
            from ml.common.metrics_manager import _CounterLike as _CounterLike
            from ml.common.metrics_manager import _GaugeLike as _GaugeLike
            from ml.common.metrics_manager import _HistogramLike as _HistogramLike

            mm = _MM.default()
            self._batch_total: _CounterLike = mm.counter(
                "ml_multi_infer_batch_total",
                "Total multi-instrument inference batches",
                ["actor"],
            )
            self._batch_seconds: _HistogramLike = mm.histogram(
                "ml_multi_infer_batch_seconds",
                "Multi-instrument batch inference duration (seconds)",
                ["actor"],
            )
            self._batch_size_hist: _HistogramLike = mm.histogram(
                "ml_multi_infer_batch_size",
                "Batch sizes for multi-instrument inference",
                ["actor"],
            )
            self._universe_size_gauge: _GaugeLike = mm.gauge(
                "ml_universe_size_gauge",
                "Current size of the actor instrument universe",
                ["actor"],
            )
        except Exception as exc:  # pragma: no cover - best-effort metrics
            self.log.exception(
                f"multi_signal.metrics_init_failed actor={self.id} error={exc!r}",
                exc,
            )

            class _NoMetric:
                def labels(self, *_: object, **__: object) -> _NoMetric:
                    return self

                def inc(self, *_: object, **__: object) -> None:
                    return None

                def observe(self, *_: object, **__: object) -> None:
                    return None

                def set(self, *_: object, **__: object) -> None:
                    return None

            self._batch_total = _NoMetric()
            self._batch_seconds = _NoMetric()
            self._batch_size_hist = _NoMetric()
            self._universe_size_gauge = _NoMetric()

    # ------------------------ Universe management (cold path) ------------------------
    def add_instrument(self, instrument_id: str) -> None:
        self._universe.add(instrument_id)
        self.log.info(
            "multi_signal.universe_add " f"instrument={instrument_id} size={self._universe.size()}",
        )
        self._set_universe_metric()

    def remove_instrument(self, instrument_id: str) -> None:
        self._universe.remove(instrument_id)
        self.log.info(
            "multi_signal.universe_remove "
            f"instrument={instrument_id} size={self._universe.size()}",
        )
        self._set_universe_metric()

    def set_universe(self, instruments: Iterable[str]) -> None:
        """
        Replace the active universe with the provided instruments (cold path).
        """
        self._universe.set_all(instruments)
        self.log.info(
            "multi_signal.universe_set "
            f"size={self._universe.size()} instruments={tuple(instruments)}",
        )
        self._set_universe_metric()

    def clear_universe(self) -> None:
        """
        Clear the active universe (cold path).
        """
        self._universe.clear()
        self.log.info("multi_signal.universe_clear")
        self._set_universe_metric()

    def _set_universe_metric(self) -> None:
        try:
            self._universe_size_gauge.labels(actor=self.id).set(self._universe.size())
        except Exception as exc:
            self.log.exception(
                f"multi_signal.universe_metric_set_failed actor={self.id} error={exc!r}",
                exc,
            )

    # --------------------------------- Hot path ---------------------------------
    def on_bar(self, bar: Bar) -> None:
        # Skip if circuit breaker open
        if self._circuit_breaker and not self._circuit_breaker.can_execute():
            return

        # Only process instruments in the universe if one is defined
        inst = str(bar.bar_type.instrument_id)
        if self._universe.size() > 0 and not self._universe.contains(inst):
            return

        # Compute features (zero-copy/low overhead expected)
        feat = self._compute_features(bar)
        if feat is None:
            return

        # Append to batch
        idx = self._batch_size
        if idx < self._max_batch:
            # Copy into pre-allocated tensor
            # Expect shape (feature_dim,), cast for safety
            self._batch_features[idx, : self._feature_dim] = feat[: self._feature_dim]
            self._batch_instruments.append(inst)
            self._batch_bars.append(bar)
            self._batch_size += 1
            if self._batch_size == 1:
                import time as _time

                self._batch_started_ns = _time.time_ns()

        # Flush when batch is full
        if self._batch_size >= self._max_batch:
            self._flush_batch()
        elif self._cfg.flush_max_latency_ms > 0 and self._batch_size > 0:
            # Best-effort time-based flushing when latency budget exceeded.
            try:
                import time as _time

                elapsed_ns = _time.time_ns() - self._batch_started_ns
                if elapsed_ns >= int(self._cfg.flush_max_latency_ms) * 1_000_000:
                    self._flush_batch()
            except Exception as exc:
                # Never impact hot path on timer errors
                self.log.exception(
                    f"multi_signal.latency_timer_failed actor={self.id} error={exc!r}",
                    exc,
                )

    # --------------------------------- Cold path ---------------------------------
    def _flush_batch(self) -> None:
        if self._batch_size == 0:
            return
        import time as _time

        t0 = _time.perf_counter()
        try:
            features_view = self._batch_features[: self._batch_size, : self._feature_dim]
            # Optional OpenTelemetry span (best-effort, does not affect hot path)
            span_ctx: Any
            try:
                from opentelemetry import trace as _trace

                tracer = _trace.get_tracer(__name__)
                span_ctx = tracer.start_as_current_span("ml.multi_infer.batch")
            except Exception as span_exc:
                self.log.exception(
                    f"multi_signal.span_init_failed actor={self.id} error={span_exc!r}",
                    span_exc,
                )
                span_ctx = nullcontext()

            with span_ctx:
                # Compute batch predictions once and stash for per-instrument pipeline
                preds, confs = self._infer_batch(features_view)
                self._prepared_preds = list(zip(preds.tolist(), confs.tolist()))
            # Dispatch per-instrument using existing protected helper for signal pipeline
            for i in range(self._batch_size):
                try:
                    self._generate_prediction_protected(self._batch_bars[i], features_view[i])
                except Exception as exc:
                    # Best-effort; do not break other instruments
                    self.log.warning(
                        "multi_signal.prediction_pipeline_failed "
                        f"instrument={self._batch_instruments[i]} index={i} "
                        f"batch_size={self._batch_size} error={exc!r}",
                    )
            # Observability (best-effort)
            try:
                self._batch_total.labels(actor=self.id).inc()
                self._batch_size_hist.labels(actor=self.id).observe(self._batch_size)
                self._batch_seconds.labels(actor=self.id).observe(_time.perf_counter() - t0)
            except Exception as metric_exc:
                self.log.exception(
                    "multi_signal.metrics_emit_failed "
                    f"actor={self.id} batch_size={self._batch_size} error={metric_exc!r}",
                    metric_exc,
                )
        finally:
            # Reset batch in O(1)
            self._batch_instruments.clear()
            self._batch_bars.clear()
            self._batch_size = 0
            self._batch_started_ns = 0
            self._prepared_preds.clear()

    # ----------------------------- Lifecycle overrides -----------------------------
    def on_start(self) -> None:
        """
        Extend start to align pre-allocated feature dimension with manifest/engineer.
        """
        super().on_start()
        try:
            # Prefer feature engineer dimension when available
            dim = getattr(self, "_feature_engineer", None)
            inferred = int(getattr(dim, "n_features", 0)) if dim is not None else 0
            if inferred <= 0:
                # Fall back to manifest feature count
                names = getattr(self, "_manifest_feature_names", [])
                inferred = len(names) if names else inferred
            if inferred > 0 and inferred != self._feature_dim:
                # Reallocate pre-allocated tensor to match real feature dimension
                new_dim = inferred
                self._batch_features = np.zeros((self._max_batch, new_dim), dtype=np.float32)
                # Record within a non-final shadow for debug (keep Final intact)
                self.log.info(
                    "multi_signal.feature_dim_adjusted "
                    f"previous_dim={self._feature_dim} inferred_dim={new_dim}",
                )
        except Exception as exc:
            # Never fail startup due to alignment; metrics/parity checks will surface problems
            self.log.exception(
                f"multi_signal.feature_dim_alignment_failed actor={self.id} error={exc!r}",
                exc,
            )

        # Advisory: auto-set universe from model registry metadata when not provided
        try:
            # Respect env/entrypoint overrides: only set when current universe is empty
            if self._universe.size() == 0:
                model_id = getattr(self, "_model_id", None)
                reg = getattr(self, "_model_registry", None)
                if model_id and reg is not None and hasattr(reg, "get_model"):
                    info = reg.get_model(model_id)
                    uids: list[str] | None = None
                    usyms: list[str] | None = None
                    if info is not None and isinstance(getattr(info, "metadata", {}), dict):
                        md = info.metadata
                        uids = (
                            md.get("universe_instrument_ids")
                            if isinstance(md.get("universe_instrument_ids"), list)
                            else None
                        )
                        usyms = (
                            md.get("universe_symbols")
                            if isinstance(md.get("universe_symbols"), list)
                            else None
                        )
                    if uids and len(uids) > 0:
                        self.set_universe(uids)
                        self.log.info(
                            "multi_signal.universe_metadata_load mode=instrument_ids "
                            f"instrument_count={len(uids)}",
                        )
                    elif usyms and len(usyms) > 0:
                        # Map bare symbols to instrument ids using configured bar_type venue as fallback
                        venue = None
                        try:
                            bt = getattr(self._config, "bar_type", None)
                            if bt is not None:
                                venue = str(getattr(bt.instrument_id, "venue", "")) or None
                        except Exception as venue_exc:
                            self.log.exception(
                                "multi_signal.universe_metadata_venue_lookup_failed "
                                f"actor={self.id} error={venue_exc!r}",
                                venue_exc,
                            )
                            venue = None

                        mapped = [
                            f"{s}.{venue}" if venue and "." not in str(s) else str(s) for s in usyms
                        ]
                        self.set_universe(mapped)
                        self.log.info(
                            "multi_signal.universe_metadata_load mode=symbols "
                            f"instrument_count={len(mapped)}",
                        )
        except Exception as exc:
            # Best-effort behavior; never fail actor startup due to metadata
            self.log.exception(
                f"multi_signal.universe_metadata_failed actor={self.id} error={exc!r}",
                exc,
            )

    # ----------------------------- Inference backend -----------------------------
    def _infer_batch(
        self,
        features: npt.NDArray[np.float32],
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """
        Run batched model inference.

        Returns (predictions, confidences) shaped (N,). Attempts ONNXRuntime vectorized
        inference when available; falls back to per-row predictions.

        """
        n = int(features.shape[0])
        preds = np.zeros((n,), dtype=np.float32)
        confs = np.zeros((n,), dtype=np.float32)

        # ONNXRuntime fast path (if model is an ORT session)
        try:
            model = getattr(self, "_model", None)
            meta = getattr(self, "_model_metadata", {})
            if model is not None and hasattr(model, "run") and isinstance(meta, dict):
                output_is_logits = resolve_output_is_logits(meta)
                positive_class_index = resolve_positive_class_index(meta)
                input_names = meta.get("input_names")
                input_name = input_names[0] if input_names else "input"
                outputs = model.run(None, {str(input_name): features})
                if len(outputs) >= 2:
                    preds = np.asarray(outputs[0])
                    confs = np.asarray(outputs[1])
                    return normalize_prediction_batch(
                        preds,
                        confs,
                        positive_class_index=positive_class_index,
                        output_is_logits=output_is_logits,
                    )
                if len(outputs) == 1:
                    preds = np.asarray(outputs[0])
                    return normalize_prediction_batch(
                        preds,
                        None,
                        positive_class_index=positive_class_index,
                        output_is_logits=output_is_logits,
                    )
        except Exception as exc:
            # Fall through to per-row inference on any failure
            self.log.exception(
                f"multi_signal.onnx_batch_run_failed actor={self.id} error={exc!r}",
                exc,
            )

        # Fallback: per-row predictions using inherited predictor
        for i in range(n):
            p, c = super()._predict(features[i])
            preds[i] = np.float32(p)
            confs[i] = np.float32(c)
        return preds, confs

    def _predict(self, features: npt.NDArray[np.float32]) -> tuple[float, float]:
        """
        Consume prepared batch predictions if available; otherwise fall back.
        """
        if self._prepared_preds:
            p, c = self._prepared_preds.pop(0)
            return float(p), float(c)
        return super()._predict(features)

    # MLSignalActor provides model loading, feature initialization, compute and predict.
