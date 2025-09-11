"""
Signal adapter protocol and example implementations.

This module defines a minimal protocol for model-driven signal adapters and
provides a simple example adapter. Adapters convert an actor context into a
configured `SignalGenerationStrategy` without modifying the actor itself.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from ml.actors.signal import SignalGenerationStrategy
from ml.actors.signal import ThresholdSignalStrategy


if TYPE_CHECKING:  # pragma: no cover - typing only
    from ml.actors.signal import MLSignalActor


class SignalAdapterProtocol(Protocol):
    """
    Protocol for model-driven decision policy adapters.

    Implementations return a concrete `SignalGenerationStrategy` when given
    an `MLSignalActor` context. This keeps the actor OCP-compliant by allowing
    new strategies to be supplied via manifests without code changes.

    """

    def make(self, actor: MLSignalActor) -> SignalGenerationStrategy: ...


class DynamicThresholdAdapter:
    """
    Example adapter that chooses a dynamic threshold from the actor.

    - If the actor maintains an adaptive threshold, use it.
    - Otherwise, fall back to the configured `prediction_threshold`.

    """

    def __init__(self) -> None:
        # No configuration required for the example adapter
        return

    def make(self, actor: MLSignalActor) -> SignalGenerationStrategy:
        # Prefer actor-provided adaptive threshold when available
        thr = float(
            getattr(
                actor,
                "_adaptive_threshold",
                getattr(actor._config, "prediction_threshold", 0.5),
            ),
        )
        return ThresholdSignalStrategy(threshold=thr)


def build_strategy_from_policy(
    *,
    policy_path: str,
    actor: MLSignalActor,
    config: dict[str, Any] | None = None,
) -> SignalGenerationStrategy:
    """
    Resolve and construct a strategy from a fully-qualified adapter path.

    Resolution order (strict validation):
    - Function adapter: `(actor) -> SignalGenerationStrategy`
    - Object with `make(actor)` method
    - Strategy class: try `cls(actor, **config)` then `cls(**config)`

    """
    import importlib

    cfg = config or {}
    module_name, _, cls_name = str(policy_path).rpartition(".")
    if not (module_name and cls_name):  # pragma: no cover - invalid inputs guarded by caller
        raise RuntimeError(f"Invalid policy path: {policy_path}")

    mod = importlib.import_module(module_name)
    target = getattr(mod, cls_name)

    # Function adapter: callable but not a class
    if callable(target) and not isinstance(target, type):
        strategy = target(actor)
        if not isinstance(strategy, SignalGenerationStrategy):
            raise RuntimeError(
                f"Adapter function did not return a SignalGenerationStrategy: {type(strategy)}",
            )
        strategy_typed: SignalGenerationStrategy = strategy
        return strategy_typed

    # Object with .make(actor)
    if hasattr(target, "make"):
        try:
            if isinstance(target, type):  # class defining an instance method `.make`
                instance = target(**cfg)
                strategy = instance.make(actor)
            else:
                # instance with `.make(actor)`
                maker = getattr(target, "make")
                strategy = maker(actor)
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"Adapter .make() failed: {exc}") from exc
        if not isinstance(strategy, SignalGenerationStrategy):
            raise RuntimeError(
                f"Adapter .make() did not return a SignalGenerationStrategy: {type(strategy)}",
            )
        strategy_typed2: SignalGenerationStrategy = strategy
        return strategy_typed2

    # Strategy class path: attempt construction
    if isinstance(target, type):
        try:
            strategy = target(actor, **cfg)
        except Exception:
            strategy = target(**cfg)
        if not isinstance(strategy, SignalGenerationStrategy):
            raise RuntimeError(
                f"Constructed object is not a SignalGenerationStrategy: {type(strategy)}",
            )
        strategy_typed3: SignalGenerationStrategy = strategy
        return strategy_typed3

    raise RuntimeError(f"Unsupported adapter target type for {policy_path}: {type(target)}")


__all__ = [
    "DynamicThresholdAdapter",
    "SignalAdapterProtocol",
    "build_strategy_from_policy",
]
