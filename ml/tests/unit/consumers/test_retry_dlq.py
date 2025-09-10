from __future__ import annotations

from ml.common.in_memory_bus import InMemoryPublisher
from ml.consumers.retry import RetriableConsumer, RetryPolicy
from ml.consumers.protocols import Envelope


def _mk(env_id: str = "e1", stage: str = "PREDICTION_EMITTED") -> Envelope:
    return {
        "id": env_id,
        "parent_id": None,
        "instrument_id": "EURUSD.SIM",
        "ts_event": 1,
        "stage": stage,
        "correlation_id": "c1",
        "payload": {"v": 1},
    }


def test_retriable_consumer_success_after_retries() -> None:
    attempts = {"count": 0}

    def flaky(_topic: str, _env: Envelope) -> None:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise RuntimeError("boom")
        return None

    dlq = InMemoryPublisher()
    # Subscribe to dlq to assert nothing lands there
    got_dlq: list[tuple[str, dict[str, object]]] = []
    dlq.subscribe("dlq.#", lambda t, p: got_dlq.append((t, p)))

    rc = RetriableConsumer(handler=flaky, dlq=dlq, policy=RetryPolicy(max_attempts=3))
    rc.handle("events.ml.PREDICTION_EMITTED", _mk())
    assert attempts["count"] == 3
    assert got_dlq == []


def test_retriable_consumer_to_dlq_on_exhausted_attempts() -> None:
    def always_fail(_topic: str, _env: Envelope) -> None:
        raise RuntimeError("boom")

    dlq = InMemoryPublisher()
    got_dlq: list[tuple[str, dict[str, object]]] = []
    dlq.subscribe("dlq.#", lambda t, p: got_dlq.append((t, p)))

    rc = RetriableConsumer(handler=always_fail, dlq=dlq, policy=RetryPolicy(max_attempts=2))
    rc.handle("events.ml.SIGNAL_EMITTED", _mk(stage="SIGNAL_EMITTED"))

    assert len(got_dlq) == 1
    topic, payload = got_dlq[0]
    assert topic == "dlq.SIGNAL_EMITTED"
    assert payload.get("attempts") == 2
