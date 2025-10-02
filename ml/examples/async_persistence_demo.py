"""
Demo: MLPersistenceWorker async persistence.

Shows the difference between synchronous and asynchronous persistence
in terms of latency and resilience to database failures.

Run with:
    python -m ml.examples.async_persistence_demo

"""

import asyncio
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from typing import Any, Final


@dataclass
class SlowStore:
    """Store that simulates slow database writes."""

    write_delay_ms: float = 5.0
    write_count: int = 0

    def write_features(
        self,
        feature_set_id: str,
        instrument_id: str,
        features: Mapping[str, float],
        ts_event: int,
        ts_init: int,
    ) -> None:
        """Simulate slow feature write."""
        time.sleep(self.write_delay_ms / 1000.0)
        self.write_count += 1

    def write_prediction(
        self,
        model_id: str,
        instrument_id: str,
        prediction: float,
        confidence: float,
        features: Mapping[str, float],
        inference_time_ms: float,
        ts_event: int,
        is_live: bool = False,
    ) -> None:
        """Simulate slow prediction write."""
        time.sleep(self.write_delay_ms / 1000.0)
        self.write_count += 1

    def write_batch(self, data: Sequence[Any], emit_events: bool = True) -> None:
        """No-op batch writer to satisfy protocol requirements."""
        self.write_count += len(data)

    def flush(self) -> None:
        """Flush method required by store protocols."""
        return None


async def demo_synchronous_writes() -> None:
    """Demonstrate synchronous write latency."""
    print("\n=== SYNCHRONOUS WRITES (Current Implementation) ===")

    store = SlowStore(write_delay_ms=5.0)
    bars_to_process: Final[int] = 10

    start = time.perf_counter()
    for i in range(bars_to_process):
        # Simulate bar processing
        bar_start = time.perf_counter()

        # Hot path - synchronous writes (BLOCKING)
        store.write_features(
            feature_set_id="test",
            instrument_id="EUR/USD.SIM",
            features={"rsi": 0.5},
            ts_event=i * 1_000_000_000,
            ts_init=i * 1_000_000_000,
        )
        store.write_prediction(
            model_id="test_model",
            instrument_id="EUR/USD.SIM",
            prediction=0.5,
            confidence=0.8,
            features={"rsi": 0.5},
            inference_time_ms=1.0,
            ts_event=i * 1_000_000_000,
        )

        bar_latency = (time.perf_counter() - bar_start) * 1000
        print(f"  Bar {i+1:2d}: {bar_latency:6.2f}ms")

    total_time = time.perf_counter() - start
    avg_latency = (total_time / bars_to_process) * 1000

    print(f"\nTotal time: {total_time:.3f}s")
    print(f"Avg latency per bar: {avg_latency:.2f}ms")
    print(f"Writes completed: {store.write_count}")


async def demo_async_writes() -> None:
    """Demonstrate async write latency with MLPersistenceWorker."""
    print("\n\n=== ASYNC WRITES (MLPersistenceWorker) ===")

    from ml.observability.ml_async_persistence import MLPersistenceWorker

    slow_store = SlowStore(write_delay_ms=5.0)
    worker = MLPersistenceWorker(
        feature_store=slow_store,
        model_store=slow_store,
        queue_maxsize=1000,
        flush_interval_seconds=0.5,  # Fast flush for demo
        batch_size=50,
    )

    # Start background worker
    worker.start()
    await asyncio.sleep(0.1)  # Let worker initialize

    bars_to_process: Final[int] = 10

    start = time.perf_counter()
    for i in range(bars_to_process):
        # Simulate bar processing
        bar_start = time.perf_counter()

        # Hot path - async writes (NON-BLOCKING)
        worker.enqueue_features(
            feature_set_id="test",
            instrument_id="EUR/USD.SIM",
            features={"rsi": 0.5},
            ts_event=i * 1_000_000_000,
            ts_init=i * 1_000_000_000,
        )
        worker.enqueue_prediction(
            model_id="test_model",
            instrument_id="EUR/USD.SIM",
            prediction=0.5,
            confidence=0.8,
            features={"rsi": 0.5},
            inference_time_ms=1.0,
            ts_event=i * 1_000_000_000,
        )

        bar_latency = (time.perf_counter() - bar_start) * 1000
        print(f"  Bar {i+1:2d}: {bar_latency:6.2f}ms (queue: {worker.queue_size()})")

    total_time = time.perf_counter() - start
    avg_latency = (total_time / bars_to_process) * 1000

    print(f"\nProcessing complete!")
    print(f"Total time: {total_time:.3f}s")
    print(f"Avg latency per bar: {avg_latency:.2f}ms")
    print(f"Queue size before drain: {worker.queue_size()}")

    # Drain queue
    print("\nDraining queue...")
    await worker.stop(drain=True, timeout=10.0)

    print(f"Writes completed: {slow_store.write_count}")
    print(f"Queue size after drain: {worker.queue_size()}")


async def demo_backpressure() -> None:
    """Demonstrate backpressure handling when queue fills."""
    print("\n\n=== BACKPRESSURE HANDLING (Queue Full) ===")

    from ml.observability.ml_async_persistence import MLPersistenceWorker

    # Very slow store + small queue to trigger backpressure
    slow_store = SlowStore(write_delay_ms=100.0)
    worker = MLPersistenceWorker(
        feature_store=slow_store,
        model_store=slow_store,
        queue_maxsize=5,  # Small queue
        flush_interval_seconds=1.0,
        batch_size=2,
    )

    worker.start()
    await asyncio.sleep(0.1)

    bars_to_process: Final[int] = 10
    dropped_count = 0

    print(f"Queue capacity: {worker.queue_maxsize}")
    print("Processing bars (will trigger backpressure)...\n")

    for i in range(bars_to_process):
        success = worker.enqueue_features(
            feature_set_id="test",
            instrument_id="EUR/USD.SIM",
            features={"rsi": 0.5},
            ts_event=i * 1_000_000_000,
            ts_init=i * 1_000_000_000,
        )

        if not success:
            dropped_count += 1
            print(f"  Bar {i+1:2d}: ❌ DROPPED (queue full: {worker.queue_size()})")
        else:
            print(f"  Bar {i+1:2d}: ✓ Enqueued (queue: {worker.queue_size()})")

    print(f"\nDropped writes: {dropped_count}/{bars_to_process}")
    print("Draining queue...")
    await worker.stop(drain=True, timeout=10.0)
    print(f"Writes completed: {slow_store.write_count}")


async def demo_db_failure_resilience() -> None:
    """Demonstrate resilience to database failures."""
    print("\n\n=== DATABASE FAILURE RESILIENCE ===")

    @dataclass
    class FailingStore:
        """Store that fails writes."""

        failure_mode: bool = True
        write_count: int = 0

        def write_features(
            self,
            feature_set_id: str,
            instrument_id: str,
            features: Mapping[str, float],
            ts_event: int,
            ts_init: int,
        ) -> None:
            if self.failure_mode:
                raise Exception("DB connection lost!")
            self.write_count += 1

        def write_prediction(
            self,
            model_id: str,
            instrument_id: str,
            prediction: float,
            confidence: float,
            features: Mapping[str, float],
            inference_time_ms: float,
            ts_event: int,
            is_live: bool = False,
        ) -> None:
            if self.failure_mode:
                raise Exception("DB connection lost!")
            self.write_count += 1

        def write_batch(self, data: Sequence[Any], emit_events: bool = True) -> None:
            if self.failure_mode:
                raise Exception("DB connection lost!")
            self.write_count += len(data)

        def flush(self) -> None:
            return None

    from ml.observability.ml_async_persistence import MLPersistenceWorker

    failing_store = FailingStore(failure_mode=True)
    worker = MLPersistenceWorker(
        feature_store=failing_store,
        model_store=failing_store,
        queue_maxsize=100,
        flush_interval_seconds=0.5,
    )

    worker.start()
    await asyncio.sleep(0.1)

    print("Scenario: Database is down, but inference continues...\n")

    # Process bars while DB is down
    for i in range(5):
        worker.enqueue_features(
            feature_set_id="test",
            instrument_id="EUR/USD.SIM",
            features={"rsi": 0.5},
            ts_event=i * 1_000_000_000,
            ts_init=i * 1_000_000_000,
        )
        print(f"  Bar {i+1}: ✓ Inference completed (queue: {worker.queue_size()})")

    await asyncio.sleep(1.0)  # Let background worker try to flush
    print(f"\nQueue size after failed flushes: {worker.queue_size()}")
    print("(Writes failed in background, but inference didn't stall)")

    # Simulate DB recovery
    print("\nDatabase recovered! Processing remaining queue...")
    failing_store.failure_mode = False

    await worker.stop(drain=True, timeout=5.0)
    print(f"Writes completed after recovery: {failing_store.write_count}")


async def main() -> None:
    """Run all demos."""
    print("=" * 70)
    print("MLPersistenceWorker Demo")
    print("=" * 70)

    # Demo 1: Synchronous vs Async latency
    await demo_synchronous_writes()
    await demo_async_writes()

    # Demo 2: Backpressure handling
    await demo_backpressure()

    # Demo 3: Database failure resilience
    await demo_db_failure_resilience()

    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  1. Async writes: ~10-20x faster than synchronous")
    print("  2. Backpressure: Graceful degradation when queue full")
    print("  3. Resilience: DB failures don't stall inference")
    print("  4. Observability: Full metrics for monitoring")


if __name__ == "__main__":
    asyncio.run(main())
