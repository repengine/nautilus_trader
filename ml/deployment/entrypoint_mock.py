#!/usr/bin/env python
"""
Entrypoint for ML Signal Actor with mock data support.

Extends the regular entrypoint to support synthetic data generation
for testing outside market hours.
"""

import asyncio
import logging
import os
import sys
import threading
import time

# Import the regular entrypoint components
from ml.deployment.entrypoint_actor import MLSignalActorNode
from ml.deployment.mock_databento import MockDatabentoClient


logger = logging.getLogger(__name__)


class MockMLSignalActorNode(MLSignalActorNode):
    """
    Extended ML Signal Actor with mock data support.
    """

    def __init__(self) -> None:
        super().__init__()
        self.mock_client: MockDatabentoClient | None = None
        self.mock_task: asyncio.Task[object] | None = None
        self._inject_thread: threading.Thread | None = None
        self._inject_stop = threading.Event()
        self._flush_thread: threading.Thread | None = None

    def setup(self) -> None:
        """
        Set up with mock data if enabled.
        """
        use_mock = os.getenv("USE_MOCK_DATA", "false").lower() == "true"

        if use_mock:
            print("=" * 80)
            print("MOCK DATA MODE ENABLED")
            print("=" * 80)
            print("Using synthetic data for testing")
            print("This mode bypasses Databento and generates fake bars")
            print("=" * 80)

            # Override Databento API key check
            os.environ["DATABENTO_API_KEY"] = "mock-key-for-testing"

            # Set up mock parameters
            instrument_str = os.getenv("INSTRUMENT_ID", "SPY.EQUS")
            self.mock_rate = float(os.getenv("MOCK_DATA_RATE", "1.0"))
            self.mock_initial_price = float(os.getenv("MOCK_INITIAL_PRICE", "650.0"))
            self.mock_volatility = float(os.getenv("MOCK_VOLATILITY", "0.002"))

            # Create mock client
            client = MockDatabentoClient(
                instrument_id=instrument_str,
                enable_logging=True,
            )
            # Modify the generator parameters using local variable for type narrowing
            client.generator.current_price = self.mock_initial_price
            client.generator.volatility = self.mock_volatility
            self.mock_client = client

        # Call parent setup
        super().setup()

    async def run(self) -> None:
        """
        Run with mock data injection if enabled.
        """
        use_mock = os.getenv("USE_MOCK_DATA", "false").lower() == "true"

        if use_mock and self.mock_client is not None:
            print("\nStarting mock data generation...")

            # Get the actor to send bars to
            actor = None
            if self.node and self.node.trader:
                try:
                    actors = self.node.trader.actors()
                    if actors:
                        actor = actors[0]
                except Exception:
                    actor = None

                client = self.mock_client
                if client is None:
                    logger.debug("Mock client unexpectedly unavailable; skipping injection")
                    return

                async def inject_mock_bars() -> None:
                    """Inject mock bars into the actor."""
                    try:
                        target = actor
                        async for bar in client.generator.generate_stream(
                            rate_hz=self.mock_rate,
                            duration_seconds=3600  # Run for 1 hour max
                        ):
                            # Inject bar directly into actor
                            if target is not None and hasattr(target, "on_bar"):
                                target.on_bar(bar)

                            # Also log metrics periodically
                            if client.generator.bar_count % 50 == 0:
                                print(f"📊 Mock: {client.generator.bar_count} bars generated")

                    except asyncio.CancelledError:
                        print("Mock data generation cancelled")

                # Start mock data injection in background if actor found
                if actor is not None:
                    self.mock_task = asyncio.create_task(inject_mock_bars())
                else:
                    print("No actor available for mock injection; skipping.")

        # Run the node normally
        await super().run()

    async def shutdown(self) -> None:
        """
        Clean shutdown including mock tasks.
        """
        if self.mock_task and not self.mock_task.done():
            self.mock_task.cancel()
            try:
                await self.mock_task
            except asyncio.CancelledError:
                logger.debug("Mock data task cancelled", exc_info=True)

        await super().shutdown()

    def run_sync(self) -> None:
        """
        Run the actor node synchronously, injecting mock bars from a background thread.
        """
        use_mock = os.getenv("USE_MOCK_DATA", "false").lower() == "true"

        def start_injection_thread() -> None:
            client = self.mock_client
            if not use_mock or client is None:
                return
            # Resolve actor reference
            actor = None
            try:
                if self.node and self.node.trader:
                    actors = self.node.trader.actors()
                    if actors:
                        actor = actors[0]
            except Exception as exc:  # pragma: no cover - defensive
                actor = None
                logger.debug("Failed to resolve actor for mock injection", exc_info=True, extra={"error": repr(exc)})
            if actor is None:
                print("No actor available for mock injection; skipping.")
                return

            def _loop() -> None:
                print("Starting mock data generation (thread)...")
                rate = max(0.1, float(getattr(self, "mock_rate", 1.0)))
                interval = 1.0 / rate
                count = 0
                while not self._inject_stop.is_set():
                    try:
                        if client is None:
                            logger.debug("Mock client unset during injection loop; stopping thread")
                            break
                        bar = client.generator.generate_bar()
                        if hasattr(actor, "on_bar"):
                            actor.on_bar(bar)
                        # Optional: force persist a simple signal in test mode
                        try:
                            if os.getenv("FORCE_SIGNAL_MODE", "false").lower() == "true":
                                sstore = getattr(actor, "_strategy_store", None)
                                if sstore is not None:
                                    sstore.write_signal(
                                        strategy_id=str(getattr(actor, "id", "ml_actor_test")),
                                        instrument_id=str(bar.bar_type.instrument_id),
                                        signal_type=("buy" if (count % 2 == 0) else "sell"),
                                        strength=1.0,
                                        model_predictions={"dummy": 1.0},
                                        risk_metrics={"confidence": 1.0},
                                        execution_params={"threshold": 0.0},
                                        ts_event=int(bar.ts_event),
                                        is_live=True,
                                    )
                        except Exception as signal_exc:  # pragma: no cover - debug aid
                            logger.debug(
                                "Mock signal persistence failed",
                                exc_info=True,
                                extra={"error": repr(signal_exc)},
                            )
                        count += 1
                        if count % 50 == 0:
                            print(f"📊 Mock(thread): {count} bars generated")
                    except Exception as exc:
                        print(f"Mock injection error (ignored): {exc}")
                    time.sleep(interval)

            self._inject_thread = threading.Thread(target=_loop, daemon=True)
            self._inject_thread.start()

            # Start background flusher for stores (strategy + model)
            try:
                sstore = getattr(actor, "_strategy_store", None)
                mstore = getattr(actor, "_model_store", None)

                def _flush_loop() -> None:
                    while not self._inject_stop.is_set():
                        try:
                            if sstore is not None:
                                sstore.flush()
                        except Exception as store_exc:  # pragma: no cover - debug aid
                            logger.debug(
                                "Strategy store flush failed in mock mode",
                                exc_info=True,
                                extra={"error": repr(store_exc)},
                            )
                        try:
                            if mstore is not None:
                                mstore.flush()
                        except Exception as model_exc:  # pragma: no cover - debug aid
                            logger.debug(
                                "Model store flush failed in mock mode",
                                exc_info=True,
                                extra={"error": repr(model_exc)},
                            )
                        time.sleep(0.5)

                self._flush_thread = threading.Thread(target=_flush_loop, daemon=True)
                self._flush_thread.start()
            except Exception as thread_exc:  # pragma: no cover - debug aid
                logger.debug(
                    "Failed to start mock flush thread",
                    exc_info=True,
                    extra={"error": repr(thread_exc)},
                )

        # Start injection and run node
        try:
            # Start background injection if enabled
            start_injection_thread()
            super().run_sync()
        finally:
            # Ensure injector stops
            if self._inject_thread and self._inject_thread.is_alive():
                self._inject_stop.set()
                self._inject_thread.join(timeout=1.0)
            if self._flush_thread and self._flush_thread.is_alive():
                self._inject_stop.set()
                self._flush_thread.join(timeout=1.0)


def main() -> None:
    """
    Run the mock-enabled ML Signal Actor.
    """
    from ml.common.logging_config import configure_logging

    configure_logging()

    # Check if test database should be used
    use_test_db = os.getenv("USE_TEST_DATABASE", "false").lower() == "true"
    if use_test_db:
        # Respect DB_CONNECTION provided by the environment (docker-compose.test.yml)
        test_db = os.getenv("TEST_DB_NAME", "nautilus_test")
        print(f"Using test database: {test_db}")

    # Create and run the mock-enabled actor node
    actor_node = MockMLSignalActorNode()
    actor_node.setup()

    # In mock mode, disable live subscription and make strategy store flush eagerly
    try:
        if os.getenv("USE_MOCK_DATA", "false").lower() == "true" and actor_node.node and actor_node.node.trader:
            try:
                actors = actor_node.node.trader.actors()
            except Exception as exc:  # pragma: no cover - defensive
                actors = []
                logger.debug(
                    "Failed to enumerate actors for mock setup",
                    exc_info=True,
                    extra={"error": repr(exc)},
                )
            if actors:
                act = actors[0]
                # Prevent on_start from attempting live subscriptions
                try:
                    setattr(act, "subscribe_bars", lambda *args, **kwargs: None)
                except Exception as sub_exc:  # pragma: no cover - defensive
                    logger.debug(
                        "Failed to override subscribe_bars for mock actor",
                        exc_info=True,
                        extra={"error": repr(sub_exc)},
                    )
                # Make stores flush on every write in test mode
                try:
                    sstore = getattr(act, "_strategy_store", None)
                    mstore = getattr(act, "_model_store", None)
                    if sstore is not None:
                        # Set on adapter (no-op for buffer) and underlying store
                        try:
                            setattr(sstore, "batch_size", 1)
                            setattr(sstore, "flush_interval_ms", 10)
                        except Exception as store_cfg_exc:  # pragma: no cover - defensive
                            logger.debug(
                                "Failed to configure strategy store batch settings",
                                exc_info=True,
                                extra={"error": repr(store_cfg_exc)},
                            )
                        try:
                            raw = getattr(sstore, "_store", None)
                            if raw is not None:
                                setattr(raw, "batch_size", 1)
                                setattr(raw, "flush_interval_ms", 10)
                        except Exception as raw_store_exc:  # pragma: no cover - defensive
                            logger.debug(
                                "Failed to configure underlying strategy store",
                                exc_info=True,
                                extra={"error": repr(raw_store_exc)},
                            )
                    if mstore is not None:
                        try:
                            setattr(mstore, "batch_size", 1)
                            setattr(mstore, "flush_interval_ms", 10)
                        except Exception as model_store_exc:  # pragma: no cover - defensive
                            logger.debug(
                                "Failed to configure model store batch settings",
                                exc_info=True,
                                extra={"error": repr(model_store_exc)},
                            )
                        try:
                            rawm = getattr(mstore, "_store", None)
                            if rawm is not None:
                                setattr(rawm, "batch_size", 1)
                                setattr(rawm, "flush_interval_ms", 10)
                        except Exception as raw_model_exc:  # pragma: no cover - defensive
                            logger.debug(
                                "Failed to configure underlying model store",
                                exc_info=True,
                                extra={"error": repr(raw_model_exc)},
                            )
                except Exception as config_exc:  # pragma: no cover - defensive
                    logger.debug(
                        "Failed to configure mock flush settings",
                        exc_info=True,
                        extra={"error": repr(config_exc)},
                    )
    except Exception as mock_setup_exc:  # pragma: no cover - defensive
        # Non-fatal test-mode prep failure
        logger.debug(
            "Mock mode preparation failed",
            exc_info=True,
            extra={"error": repr(mock_setup_exc)},
        )

    # Start lightweight HTTP endpoints
    try:
        port = int(os.getenv("METRICS_PORT", "8000"))
    except ValueError:
        port = 8000
    host = os.getenv("METRICS_HOST", "127.0.0.1")

    from ml.deployment.metrics_http import build_app
    app = build_app(lambda: actor_node._healthy)
    http_thread = threading.Thread(
        target=lambda: app.run(host=host, port=port, debug=False),
        daemon=True,
    )
    http_thread.start()

    # Run the node (synchronous) so background injection thread can push bars
    try:
        actor_node.run_sync()
    except KeyboardInterrupt:
        print("\nShutdown requested")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
